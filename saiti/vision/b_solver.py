#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vision/b_solver.py — B路几何拼接（DFS回溯算法）

算法来源：开源 puzzle_core/solver.py (_search_candidates)，针对本项目适配。

核心思路：
  1. 每片碎片建局部坐标（顶点 - 质心），piece 0 固定于原点
  2. DFS 回溯：对每个待放片，枚举已放片各条边，按切割边反平行对齐
  3. 重叠 > 300mm² 则剪枝（宽松阈值，容纳实测边长误差累积的假重叠）
  4. 全部放完后验证外缘是否构成矩形（凸包4顶点 + 直角 + 尺寸 + 面积），通过则收录
  5. 取最优方案，minAreaRect 旋转水平后平移到目标区
  6. SVD 求每片的旋转角 → 生成 PlacementResult
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from vision.detection import Piece, sort_vertices_cyclic
from vision.config import Config, get_config


# =============================================================================
# 几何工具
# =============================================================================

def _ccw(pts: np.ndarray) -> np.ndarray:
    """确保顶点逆时针（cv2 有向面积 < 0 则翻转）。"""
    p = np.asarray(pts, dtype=np.float64)
    if cv2.contourArea(p.astype(np.float32), oriented=True) < 0:
        p = p[::-1].copy()
    return p


def _edge_lengths(polygon: np.ndarray) -> np.ndarray:
    """返回各边长度数组（同顶点顺序）。"""
    pts = np.asarray(polygon, dtype=np.float64)
    return np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)


def _is_cut_edge(polygon: np.ndarray, tol_deg: float = 8.0) -> np.ndarray:
    """
    返回布尔数组 is_cut[i]：edge[i] (i→(i+1)%n) 是否为裁切边。

    判据（兼容凸/凹碎片）：
      - 反射顶点（内角 > 180°，CCW多边形 v1×v2 > 0）：必定是裁切交叉点
        → 两侧边均为裁切边
      - 直角顶点（内角 ≈ 90°，且为凸）：矩形边界角点
        → 两侧边均为边界边
      - 其他角度的凸顶点：裁切顶点
        → 两侧边均为裁切边

    叉积符号约定（_ccw 保证数学 y-UP CCW）：
      凸顶点 v1×v2 < 0；反射顶点 v1×v2 > 0
    """
    n = len(polygon)
    p = np.asarray(polygon, np.float64)
    # boundary[i]=True 表示顶点 i 是矩形边界角点（凸90°）
    boundary = np.zeros(n, dtype=bool)
    for i in range(n):
        v1 = p[(i - 1) % n] - p[i]
        v2 = p[(i + 1) % n] - p[i]
        l1 = float(np.linalg.norm(v1))
        l2 = float(np.linalg.norm(v2))
        if l1 < 1e-9 or l2 < 1e-9:
            continue
        # CCW(数学y-UP)：凸顶点 cross < 0，反射顶点 cross > 0
        cross = float(v1[0] * v2[1] - v1[1] * v2[0])
        if cross >= 0:
            # 反射顶点或共线顶点，一定是裁切交叉点，不是边界角
            continue
        cos_a = float(np.dot(v1, v2)) / (l1 * l2)
        ang = math.degrees(math.acos(float(np.clip(cos_a, -1.0, 1.0))))
        if abs(ang - 90.0) <= tol_deg:
            boundary[i] = True
    # edge i 是裁切边 ⟺ 两端顶点均非边界角点
    is_cut = np.ones(n, dtype=bool)
    for i in range(n):
        if boundary[i] or boundary[(i + 1) % n]:
            is_cut[i] = False
    return is_cut


def _edge_match_allowed(la: float, lb: float) -> bool:
    """切割边长容差：±2.5mm 或 ±5.5%（取较大）。"""
    return abs(la - lb) <= max(2.5, 0.055 * min(la, lb))


def _rigid_edge_alignment(
    q0: np.ndarray, q1: np.ndarray,
    p0: np.ndarray, p1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    求旋转矩阵 R 和平移向量 t，使得：
        R @ q0 + t = p0
        R @ q1 + t = p1
    调用时传入反向目标端点（p0=placed[e+1], p1=placed[e]），实现切割边反平行对齐。
    """
    qv = np.asarray(q1, np.float64) - np.asarray(q0, np.float64)
    pv = np.asarray(p1, np.float64) - np.asarray(p0, np.float64)
    angle = math.atan2(float(pv[1]), float(pv[0])) - math.atan2(float(qv[1]), float(qv[0]))
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    t = np.asarray(p0, np.float64) - R @ np.asarray(q0, np.float64)
    return R, t


@dataclass
class _Placement:
    rotation: np.ndarray     # 2×2 旋转矩阵
    translation: np.ndarray  # (2,) 平移向量

    def transform(self, pts: np.ndarray) -> np.ndarray:
        return np.asarray(pts, np.float64) @ self.rotation.T + self.translation


def _polygon_area(pts: np.ndarray) -> float:
    return abs(float(cv2.contourArea(np.asarray(pts, np.float32))))


def _intersection_area(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """计算两凸/非凸多边形的重叠面积（mm²）。"""
    a = np.asarray(poly_a, np.float32)
    b = np.asarray(poly_b, np.float32)
    if cv2.isContourConvex(a) and cv2.isContourConvex(b):
        area, _ = cv2.intersectConvexConvex(a, b)
        return float(area)
    # 非凸回退：光栅化
    all_pts = np.vstack([a, b]).astype(np.float64)
    min_xy = np.floor(all_pts.min(0) - 1.0)
    max_xy = np.ceil(all_pts.max(0) + 1.0)
    scale = 4.0
    size = np.maximum(np.ceil((max_xy - min_xy) * scale).astype(int) + 1, 2)
    masks = []
    for poly in (a, b):
        mask = np.zeros((int(size[1]), int(size[0])), dtype=np.uint8)
        pxs = np.round((poly.astype(np.float64) - min_xy) * scale).astype(np.int32)
        cv2.fillPoly(mask, [pxs], 255)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        masks.append(mask)
    return float(np.count_nonzero((masks[0] > 0) & (masks[1] > 0))) / (scale * scale)


def _layout_polygons(
    local_polys: List[np.ndarray],
    placements: Dict[int, _Placement],
) -> Dict[int, np.ndarray]:
    return {idx: pl.transform(local_polys[idx]) for idx, pl in placements.items()}


def _check_rectangle(
    polygons: Dict[int, np.ndarray],
    total_area: float,
    rect_fill: float = 0.83,
    direct_fill_min: float = 0.82,
    area_tol: float = 0.12,
    w_range: Tuple[float, float] = (60.0, 148.0),
    h_range: Tuple[float, float] = (40.0, 105.0),
) -> Tuple[bool, float, float, float, float, float]:
    """
    检验拼合外缘是否构成矩形。
    返回 (is_rect, width_mm, height_mm, score, rectangularity, direct_fill)，score 越低越好。

    判据（全部满足才通过）：
      1. 凸包面积 / 最小外接矩形面积 ≥ rect_fill
      2. 各片总面积 / 最小外接矩形面积 ≥ direct_fill_min（防间隙假通过）
      3. 最小外接矩形宽高在目标范围内
      4. |凸包面积 - total_area| / 凸包面积 < area_tol（排除重叠/大间隙）

    注意：rect_fill 只检查凸包是否贴近包围框（防外形不矩形），
          但凸包内可能存在片间间隙；direct_fill_min 直接用各片面积之和，
          能捕捉凸包内部间隙导致的假矩形。
    """
    all_pts = np.vstack([polygons[k] for k in sorted(polygons)]).astype(np.float32)
    hull = cv2.convexHull(all_pts)
    hull_area = max(float(cv2.contourArea(hull)), 1e-6)

    (_, _), (rw, rh), _ = cv2.minAreaRect(all_pts)
    width  = max(float(rw), float(rh))
    height = min(float(rw), float(rh))

    if not (w_range[0] <= width <= w_range[1] and h_range[0] <= height <= h_range[1]):
        return False, width, height, float('inf'), 0.0, 0.0

    rect_area = width * height
    rectangularity = hull_area / max(rect_area, 1e-6)
    if rectangularity < rect_fill:
        return False, width, height, float('inf'), rectangularity, total_area / max(rect_area, 1e-6)

    # 直接填充率：各片面积之和 / 包围框面积（凸包检查不到片间间隙）
    direct_fill = total_area / max(rect_area, 1e-6)
    if direct_fill < direct_fill_min:
        return False, width, height, float('inf'), rectangularity, direct_fill

    area_err = abs(hull_area - total_area) / hull_area
    if area_err > area_tol:
        return False, width, height, float('inf'), rectangularity, direct_fill

    score = 100.0 * area_err + (1.0 - rectangularity) * 200.0
    return True, width, height, score, rectangularity, direct_fill


# =============================================================================
# 边覆盖枚举器
# =============================================================================

def _iter_edge_coverings(
    lp: float,
    unplaced: List[int],
    edge_sizes: List[np.ndarray],
    used_edges: set,
    is_cut: Optional[List[np.ndarray]] = None,
    has_cut: Optional[List[bool]] = None,
    max_cover: int = 3,
    min_composite_edge: float = 0.0,
):
    """
    枚举已放边（长度 lp）的所有有效覆盖方案。
    每个方案是有序列表 [(piece_idx, edge_idx), ...]，顺序即沿边排列顺序。
    1片：1:1容差；多片：复合容差（7%）。

    is_cut[i][e]  : 第 i 片第 e 条边是否为裁切边
    has_cut[i]    : 第 i 片是否拥有至少一条裁切边
    有 is_cut 时，接合边优先只使用裁切边；若某片无裁切边则退回允许全部边。
    """
    import itertools
    ctol = max(3.5, 0.07 * lp)

    def _allowed_edges(idx: int) -> List[int]:
        """返回该片允许作为接合面的边索引列表。"""
        avail = [ne for ne in range(len(edge_sizes[idx]))
                 if (idx, ne) not in used_edges]
        if is_cut is None or not (has_cut is not None and has_cut[idx]):
            return avail
        # 只有该片 ≥2 条裁切边时才限制只用裁切边接合；
        # 仅 1 条可能是检测伪影，强制限制会漏掉正确拼法
        cut_count = int(np.sum(is_cut[idx]))
        if cut_count >= 2:
            cut_avail = [ne for ne in avail if is_cut[idx][ne]]
            if cut_avail:
                return cut_avail
        return avail

    # 1片：标准1:1容差（下限 4.0mm，百分比 7%，两取较大）
    for idx in unplaced:
        for ne in _allowed_edges(idx):
            lq = float(edge_sizes[idx][ne])
            if abs(lq - lp) <= max(4.0, 0.07 * min(lq, lp)):
                yield [(idx, ne)]

    # 多片：按排列枚举，不同排列顺序=不同几何方案
    for count in range(2, max_cover + 1):
        if len(unplaced) < count:
            continue
        for perm in itertools.permutations(range(len(unplaced)), count):
            idxs = [unplaced[p] for p in perm]
            edge_cands = [
                [(ne, float(edge_sizes[idx][ne])) for ne in _allowed_edges(idx)]
                for idx in idxs
            ]
            if any(len(ec) == 0 for ec in edge_cands):
                continue
            for combo in itertools.product(*edge_cands):
                lens = [lq for _, lq in combo]
                # 拒绝贡献边过短的复合方案（扑克外边缘不应作为切割接合面）
                if min_composite_edge > 0 and any(lq < min_composite_edge for lq in lens):
                    continue
                partial, ok = 0.0, True
                for lq_k in lens[:-1]:
                    partial += lq_k
                    if partial >= lp + ctol / 2:
                        ok = False
                        break
                if ok and abs(sum(lens) - lp) <= ctol:
                    yield [(idxs[k], combo[k][0]) for k in range(count)]


# =============================================================================
# DFS 搜索
# =============================================================================

@dataclass
class _Candidate:
    score: float
    width: float
    height: float
    placements: Dict[int, _Placement]
    edge_pairs: List[Tuple]   # [(placed_idx, pe, new_idx, ne, lp, lq), ...]


def _sample_edge_profile(
    gray: np.ndarray,
    verts_px: np.ndarray,
    edge_idx: int,
    inset_px: float = 5.0,
    n_samples: int = 24,
    margin_frac: float = 0.12,
) -> Optional[np.ndarray]:
    """
    沿边在内侧 inset_px 像素处采样灰度序列，用于拼接边纹理相关。
    内侧方向由边中点指向多边形质心确定，无需区分 CCW/CW。
    白纸片传 None 跳过，card 片才计算。
    """
    n = len(verts_px)
    p0 = verts_px[edge_idx].astype(np.float64)
    p1 = verts_px[(edge_idx + 1) % n].astype(np.float64)
    edge_vec = p1 - p0
    length = float(np.linalg.norm(edge_vec))
    if length < 8.0:   # < 2mm @4px/mm，太短无法采样
        return None
    centroid = verts_px.mean(0).astype(np.float64)
    to_c = centroid - (p0 + p1) * 0.5
    tc_len = float(np.linalg.norm(to_c))
    if tc_len < 1e-6:
        return None
    n_in = to_c / tc_len   # 单位内法线（指向质心）
    h, w = gray.shape[:2]
    vals: List[float] = []
    for t in np.linspace(margin_frac, 1.0 - margin_frac, n_samples):
        pt = p0 + t * edge_vec + n_in * inset_px
        xi, yi = int(round(float(pt[0]))), int(round(float(pt[1])))
        if 0 <= xi < w and 0 <= yi < h:
            vals.append(float(gray[yi, xi]))
    return np.array(vals, dtype=np.float32) if len(vals) >= n_samples // 2 else None


def _profile_corr(a: np.ndarray, b_rev: np.ndarray) -> Tuple[float, float]:
    """
    归一化互相关。b_rev 需由调用方传入 b[::-1]（两边方向相反）。
    Returns: (corr∈[-1,1], clarity=min(std_a, std_b))
    clarity 低（<阈值）时表示图案过均匀/杂乱，不可信，外部应跳过剪枝。
    """
    n = min(len(a), len(b_rev))
    if n < 6:
        return 0.0, 0.0
    fa = a[:n].astype(np.float64);  fa -= fa.mean()
    fb = b_rev[:n].astype(np.float64); fb -= fb.mean()
    sa, sb = float(np.std(fa)), float(np.std(fb))
    clarity = min(sa, sb)
    if sa < 1.0 or sb < 1.0:
        return 0.0, clarity
    corr = float(np.dot(fa, fb) / (sa * sb * n))
    return float(np.clip(corr, -1.0, 1.0)), clarity


def _search_candidates(
    local_polys: List[np.ndarray],
    total_area: float,
    max_states: int = 40000,
    min_composite_edge: float = 0.0,
    rect_fill: float = 0.77,
    direct_fill_min: float = 0.77,
    area_tol: float = 0.20,
    edge_profiles: Optional[List[List[Optional[np.ndarray]]]] = None,
    grad_corr_min: float = 0.30,
    grad_clarity_min: float = 8.0,
) -> List[_Candidate]:
    """
    DFS 回溯：piece 0 固定于原点，逐片尝试边对对齐。
    返回评分由低到高的候选方案（最多 80 个）。
    """
    n = len(local_polys)
    if n == 1:
        pl = _Placement(np.eye(2), np.zeros(2))
        return [_Candidate(0.0, 0.0, 0.0, {0: pl}, [])]

    edge_sizes = [_edge_lengths(poly) for poly in local_polys]
    # 无条件计算裁切边约束（凸/凹均适用；凸扑克牌碎片的边框边同样需要排除）
    is_cut  = [_is_cut_edge(poly) for poly in local_polys]
    has_cut = [bool(np.any(ic)) for ic in is_cut]
    # 若某片所有边均被判为边框边（全 False），则退回允许全部边，避免该片无法被拼接
    for i, ic in enumerate(is_cut):
        if not has_cut[i]:
            is_cut[i] = np.ones(len(ic), dtype=bool)
    candidates: List[_Candidate] = []
    state_counter = [0]
    identity = _Placement(np.eye(2), np.zeros(2))
    # 目标矩形尺寸上界（容差 30%），用于包围盒早剪枝
    _BB_LONG  = 152.0   # 长边最大 mm（+15% 标定容差）
    _BB_SHORT = 108.0   # 短边最大 mm

    def recurse(
        placements: Dict[int, _Placement],
        used_edges: set,
        edge_pairs: List[Tuple],
    ) -> None:
        if state_counter[0] >= max_states:
            return
        state_counter[0] += 1

        if len(placements) == n:
            polys = _layout_polygons(local_polys, placements)
            ok, w, h, score, rect_val, dfill = _check_rectangle(
                polys, total_area,
                rect_fill=rect_fill,
                direct_fill_min=direct_fill_min,
                area_tol=area_tol,
            )
            tree_str = "|".join(
                f"{pi_}e{pe_}->{ni_}e{ne_}({_lq:.1f})"
                for pi_, pe_, ni_, ne_, _lp, _lq in edge_pairs
            )
            if ok:
                # 分级惩罚：每个组合组超出的片数各+20分（2片组合→+20，3片→+40）
                host_count: Dict[tuple, int] = {}
                for pi_, pe_, ni_, ne_, _lp, _lq in edge_pairs:
                    key_ = (pi_, pe_)
                    host_count[key_] = host_count.get(key_, 0) + 1
                composite_penalty = sum(
                    (cnt - 1) * 100.0 for cnt in host_count.values() if cnt > 1
                )
                n_composite = sum(1 for cnt in host_count.values() if cnt > 1)
                max_cnt = max(host_count.values()) if host_count else 1
                score += composite_penalty
                candidates.append(_Candidate(score, w, h, dict(placements),
                                              list(edge_pairs)))
                if n_composite == 0:
                    tag = "[全1对1]"
                elif max_cnt >= 3:
                    tag = f"[含{max_cnt}片组合]"
                else:
                    tag = "[含2片组合]"
                print(f"    [通过] {w:.1f}x{h:.1f}mm score={score:.1f} rect={rect_val:.3f} fill={dfill:.3f} {tag}  {tree_str}")
            else:
                if rect_val > 0.60 or dfill > 0.60:
                    print(f"    [拒绝近] {w:.1f}x{h:.1f}mm rect={rect_val:.3f} fill={dfill:.3f}  {tree_str}")
            return

        placed_polys = _layout_polygons(local_polys, placements)
        unplaced = [i for i in range(n) if i not in placements]

        # 提前退出：已有足够高质量候选，且本层没有"相似宿主边"待探索。
        # 相似宿主边定义：已放片中有两条未用边长度相差 ≤ 5mm，
        # 若其中一条已触发过候选则另一条必须也被尝试，否则不允许早退。
        if len(candidates) >= 8 and candidates[0].score < 5.0:
            # 收集所有未用宿主边的长度
            avail_host_lens = []
            for _pi, _pp in placed_polys.items():
                _pn = len(_pp)
                for _pe in range(_pn):
                    if (_pi, _pe) not in used_edges:
                        avail_host_lens.append(float(edge_sizes[_pi][_pe]))
            # 若存在两条相差 ≤5mm 的可用宿主边，说明还有相似对等待验证，不早退
            _has_similar = False
            for _ai in range(len(avail_host_lens)):
                for _aj in range(_ai + 1, len(avail_host_lens)):
                    if abs(avail_host_lens[_ai] - avail_host_lens[_aj]) <= 5.0:
                        _has_similar = True
                        break
                if _has_similar:
                    break
            if not _has_similar:
                return

        for placed_idx, placed_poly in placed_polys.items():
            placed_n = len(placed_poly)
            # 长边优先：最长边辨识度最高，最先尝试可减少假配对并更快找到正确方案
            edge_order = sorted(range(placed_n),
                                key=lambda e: -float(edge_sizes[placed_idx][e]))
            for pe in edge_order:
                if (placed_idx, pe) in used_edges:
                    continue
                # 已放边不做几何 is_cut 过滤：K牌斜切会产生近 90° 假边框角，
                # 误判后会封掉合法配对。边的合法性由梯度相关剪枝负责。
                p0 = placed_poly[pe]
                p1 = placed_poly[(pe + 1) % placed_n]
                lp = float(np.linalg.norm(p1 - p0))

                for covering in _iter_edge_coverings(
                        lp, unplaced, edge_sizes, used_edges,
                        is_cut=is_cut, has_cut=has_cut,
                        min_composite_edge=min_composite_edge):
                    covered_idxs = [idx for idx, _ in covering]
                    lens = [float(edge_sizes[idx][ne]) for idx, ne in covering]
                    total_cov = sum(lens)

                    # 大边诊断（≥70mm）
                    _dbg_large = lp >= 70.0
                    # ── 梯度纹理剪枝（仅 card 片，白纸 profile=None 自动跳过）────
                    if edge_profiles is not None:
                        _grad_reject = False
                        for idx_k, ne_k in covering:
                            pa = edge_profiles[placed_idx][pe]
                            pb = edge_profiles[idx_k][ne_k]
                            if pa is not None and pb is not None:
                                corr, clarity = _profile_corr(pa, pb[::-1])
                                if clarity >= grad_clarity_min and corr < grad_corr_min:
                                    _grad_reject = True
                                    break
                        if _grad_reject:
                            continue

                    # 按各片边长比例沿已放边划分段，计算各片的刚性对齐
                    new_pls: Dict[int, _Placement] = {}
                    tfs = []
                    cum = 0.0
                    for k, (idx_k, ne_k) in enumerate(covering):
                        seg_p0 = p0 + (cum / total_cov) * (p1 - p0)
                        cum += lens[k]
                        seg_p1 = p0 + (cum / total_cov) * (p1 - p0)
                        poly_k = local_polys[idx_k]
                        q0_k = poly_k[ne_k]
                        q1_k = poly_k[(ne_k + 1) % len(poly_k)]
                        R_k, t_k = _rigid_edge_alignment(q0_k, q1_k, seg_p1, seg_p0)
                        pl_k = _Placement(R_k, t_k)
                        new_pls[idx_k] = pl_k
                        tfs.append(pl_k.transform(poly_k))

                    # 包围盒早剪枝：用 minAreaRect 取旋转最小矩形，避免 AABB 因旋转虚大
                    all_verts_f32 = np.vstack(list(placed_polys.values()) + tfs).astype(np.float32)
                    (_, _), (rw, rh), _ = cv2.minAreaRect(all_verts_f32)
                    bb_long  = max(float(rw), float(rh))
                    bb_short = min(float(rw), float(rh))
                    if bb_long > _BB_LONG or bb_short > _BB_SHORT:
                        continue

                    # 新片之间的重叠检查（始终执行）
                    invalid = False
                    for a in range(len(tfs)):
                        if invalid:
                            break
                        for b in range(a + 1, len(tfs)):
                            if _intersection_area(tfs[a], tfs[b]) > 600.0:
                                invalid = True
                                break
                    # 与已放片的重叠检查（始终执行，600mm²阈值可容纳测量误差引起的微小重叠）
                    if not invalid:
                        for tf_k in tfs:
                            for ex_poly in placed_polys.values():
                                if _intersection_area(tf_k, ex_poly) > 600.0:
                                    invalid = True
                                    break
                            if invalid:
                                break
                    if invalid:
                        continue

                    # 放入所有片 → 递归 → 回退
                    for idx_k, pl_k in new_pls.items():
                        placements[idx_k] = pl_k
                    used_edges.add((placed_idx, pe))
                    for idx_k, ne_k in covering:
                        used_edges.add((idx_k, ne_k))
                    for (idx_k, ne_k), lq_k in zip(covering, lens):
                        edge_pairs.append((placed_idx, pe, idx_k, ne_k, lp, lq_k))

                    recurse(placements, used_edges, edge_pairs)

                    for _ in covering:
                        edge_pairs.pop()
                    used_edges.discard((placed_idx, pe))
                    for idx_k, ne_k in covering:
                        used_edges.discard((idx_k, ne_k))
                    for idx_k in covered_idxs:
                        placements.pop(idx_k, None)

        # ── Multi-host adjacent seam: two adjacent placed edges host ONE new piece ──
        # 检查所有已放片边对 (ia,ea)+(ib,eb)，若端点贴合则视为连续缝，
        # 允许单个未放片的边 lq 覆盖整条复合缝（P4e1←P3e3+P2e0 即此情况）
        if len(unplaced) >= 1:
            for ia, poly_a in placed_polys.items():
                na_a = len(poly_a)
                for ea in range(na_a):
                    if (ia, ea) in used_edges:
                        continue
                    pa0 = poly_a[ea]
                    pa1 = poly_a[(ea + 1) % na_a]
                    la = float(np.linalg.norm(pa1 - pa0))

                    for ib, poly_b in placed_polys.items():
                        if ib == ia:
                            continue
                        nb_b = len(poly_b)
                        for eb in range(nb_b):
                            if (ib, eb) in used_edges:
                                continue
                            pb0 = poly_b[eb]
                            pb1 = poly_b[(eb + 1) % nb_b]
                            lb = float(np.linalg.norm(pb1 - pb0))

                            # 端点贴合：pa1 ≈ pb0 → 复合缝方向 pa0→pa1=pb0→pb1
                            adj_d = float(np.linalg.norm(pa1 - pb0))
                            adj_tol = max(5.0, 0.08 * max(la, lb))
                            if adj_d > adj_tol:
                                continue

                            seam_len = la + lb
                            ctol_ms = max(3.5, 0.07 * seam_len)

                            for idx in unplaced:
                                for ne in range(len(edge_sizes[idx])):
                                    if (idx, ne) in used_edges:
                                        continue
                                    lq = float(edge_sizes[idx][ne])
                                    if abs(lq - seam_len) > ctol_ms:
                                        continue

                                    poly_k = local_polys[idx]
                                    q0_k = poly_k[ne]
                                    q1_k = poly_k[(ne + 1) % len(poly_k)]
                                    # 反平行对齐：q0→pb1, q1→pa0
                                    R_k, t_k = _rigid_edge_alignment(q0_k, q1_k, pb1, pa0)
                                    pl_k = _Placement(R_k, t_k)
                                    tf_k = pl_k.transform(poly_k)

                                    # 包围盒早剪枝
                                    all_vf32 = np.vstack(
                                        list(placed_polys.values()) + [tf_k]
                                    ).astype(np.float32)
                                    (_, _), (rw2, rh2), _ = cv2.minAreaRect(all_vf32)
                                    if max(float(rw2), float(rh2)) > _BB_LONG or \
                                       min(float(rw2), float(rh2)) > _BB_SHORT:
                                        continue

                                    # 重叠检查
                                    invalid_ms = False
                                    for ex_poly in placed_polys.values():
                                        if _intersection_area(tf_k, ex_poly) > 600.0:
                                            invalid_ms = True
                                            break
                                    if invalid_ms:
                                        continue

                                    # 放入 → 递归 → 回退
                                    placements[idx] = pl_k
                                    used_edges.add((ia, ea))
                                    used_edges.add((ib, eb))
                                    used_edges.add((idx, ne))
                                    # 记录两条宿主边各对应新片的部分长度
                                    edge_pairs.append((ia, ea, idx, ne, la, lq * la / seam_len))
                                    edge_pairs.append((ib, eb, idx, ne, lb, lq * lb / seam_len))

                                    recurse(placements, used_edges, edge_pairs)

                                    edge_pairs.pop(); edge_pairs.pop()
                                    used_edges.discard((ia, ea))
                                    used_edges.discard((ib, eb))
                                    used_edges.discard((idx, ne))
                                    placements.pop(idx, None)

    recurse({0: identity}, set(), [])
    print(f"  [DFS] 共探索 {state_counter[0]} 个状态")
    candidates.sort(key=lambda c: c.score)
    return candidates[:80]


# =============================================================================
# 旋转 / 平移工具
# =============================================================================

def _rotation_matrix(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _svd_rotation_deg(src: np.ndarray, dst: np.ndarray) -> float:
    """SVD 求 src→dst 最优旋转角（度）。"""
    sc = src - src.mean(0)
    dc = dst - dst.mean(0)
    H = sc.T @ dc
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return math.degrees(math.atan2(float(R[1, 0]), float(R[0, 0])))


# =============================================================================
# 纹理评分（扑克牌候选消歧义）
# =============================================================================

_TEX_SCALE   = 4.0    # 渲染画布 px/mm
_TEX_BAND_MM = 4.0    # 缝两侧采样深度 mm
_TEX_N_ALONG = 20     # 沿缝采样点数
_TEX_N_PERP  = 5      # 垂直方向采样层数
_TEX_LAM     = 0.5    # J_total = J_shape + LAM * J_texture
_TEX_TOP_K   = 8      # 只对前 K 个几何候选做纹理评分


def _bilinear(gray: np.ndarray, x: float, y: float) -> Optional[float]:
    """灰度图双线性插值；越界返回 None。"""
    h, w = gray.shape
    xi, yi = int(x), int(y)
    if xi < 0 or yi < 0 or xi >= w - 1 or yi >= h - 1:
        return None
    fx, fy = x - xi, y - yi
    return float(
        gray[yi,     xi    ] * (1 - fx) * (1 - fy) +
        gray[yi,     xi + 1] *       fx * (1 - fy) +
        gray[yi + 1, xi    ] * (1 - fx) *       fy +
        gray[yi + 1, xi + 1] *       fx *       fy
    )


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """归一化互相关 [-1, 1]；均匀区域返回 0.0。"""
    a = a.ravel().astype(np.float32)
    b = b.ravel().astype(np.float32)
    if len(a) < 4:
        return 0.0
    a -= a.mean();  b -= b.mean()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _render_to_canvas(
    candidate: _Candidate,
    pieces: List["Piece"],
    local_polys: List[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    把每片碎片的 texture_bgr 按 placement 变换后渲染到公共画布。
    返回 (gray_canvas float32, min_xy_mm)；任何一片缺纹理数据则返回 (None, None)。

    顶点对应关系：
      local_polys[i] = _ccw(vertices_mm) - cen
      _ccw 可能翻转了 vertices_mm 的顺序；翻转标志 = contourArea(oriented=True) < 0
      翻转时 local_polys[i][j] 对应 vertices_px[N-1-j]，否则对应 vertices_px[j]
    """
    from vision.detection import Piece as _Piece  # 避免循环导入时重复定义
    placed = _layout_polygons(local_polys, candidate.placements)
    all_pts = np.vstack(list(placed.values()))
    min_xy = all_pts.min(0) - 5.0
    max_xy = all_pts.max(0) + 5.0
    cw = max(4, int((max_xy[0] - min_xy[0]) * _TEX_SCALE) + 2)
    ch = max(4, int((max_xy[1] - min_xy[1]) * _TEX_SCALE) + 2)
    canvas = np.zeros((ch, cw, 3), dtype=np.uint8)

    for idx, pl in candidate.placements.items():
        piece = pieces[idx]
        if (piece.texture_bgr is None or piece.texture_origin_px is None
                or piece.vertices_mm is None or piece.vertices_px is None):
            return None, None, None

        lp = local_polys[idx]       # local poly (CCW, zero-centered)
        N  = len(lp)

        # canvas dst points (assembly mm → canvas px)
        dst = ((pl.transform(lp) - min_xy) * _TEX_SCALE).astype(np.float32)

        # texture src points (same vertex ordering as lp)
        flip = (cv2.contourArea(piece.vertices_mm.astype(np.float32),
                                oriented=True) < 0)
        verts_px = piece.vertices_px[::-1].copy() if flip else piece.vertices_px.copy()
        ox, oy  = piece.texture_origin_px
        src = (verts_px.astype(np.float32)
               - np.array([ox, oy], dtype=np.float32))  # N×2, texture px

        # warp texture → canvas
        th, tw = piece.texture_bgr.shape[:2]
        try:
            if N == 3:
                M = cv2.getAffineTransform(src[:3], dst[:3])
                warped = cv2.warpAffine(piece.texture_bgr, M, (cw, ch),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT)
            elif N == 4:
                M = cv2.getPerspectiveTransform(src, dst)
                warped = cv2.warpPerspective(piece.texture_bgr, M, (cw, ch),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT)
            else:
                M, _ = cv2.findHomography(src, dst, 0)
                if M is None:
                    return None, None, None
                warped = cv2.warpPerspective(piece.texture_bgr, M, (cw, ch),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT)
        except cv2.error:
            return None, None, None

        poly_mask = np.zeros((ch, cw), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [dst.round().astype(np.int32)], 255)
        canvas[poly_mask > 0] = warped[poly_mask > 0]

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return gray, min_xy, canvas


def _score_one_seam(
    Gx: np.ndarray,
    Gy: np.ndarray,
    pt0: np.ndarray,
    pt1: np.ndarray,
) -> float:
    """
    梯度方向连续性得分（越低越好）。

    度量：
      方向差  = 1 - mean_w[ cos(θa - θb) ]  ∈ [0,1]
                cos(θa-θb) = dot(Ga,Gb)/(|Ga|·|Gb|)，无需 arctan2，不受角度绕环影响
      幅度不对称 = mean[ |Ma-Mb|/(Ma+Mb) ]     ∈ [0,1]
      均以 min(Ma,Mb) 加权 → 平坦区（近零梯度）贡献低，图案边缘贡献高

    得分 = 0.6*方向差 + 0.4*幅度不对称
    """
    d = pt1 - pt0
    length = float(np.linalg.norm(d))
    if length < 4.0:
        return 0.0

    d_unit = d / length
    n_unit  = np.array([-d_unit[1], d_unit[0]], dtype=np.float64)
    band_px = _TEX_BAND_MM * _TEX_SCALE

    cos_vals: List[float] = []
    wts:      List[float] = []
    asym_vals: List[float] = []

    for t in np.linspace(0.1, 0.9, _TEX_N_ALONG):
        p_s = pt0 + t * d
        for j in range(1, _TEX_N_PERP + 1):
            delta = j * band_px / _TEX_N_PERP
            pa = p_s + delta * n_unit
            pb = p_s - delta * n_unit

            gxa = _bilinear(Gx, pa[0], pa[1])
            gya = _bilinear(Gy, pa[0], pa[1])
            gxb = _bilinear(Gx, pb[0], pb[1])
            gyb = _bilinear(Gy, pb[0], pb[1])

            if gxa is None or gya is None or gxb is None or gyb is None:
                break

            ma = math.sqrt(gxa * gxa + gya * gya)
            mb = math.sqrt(gxb * gxb + gyb * gyb)
            w  = min(ma, mb)
            if w < 1.0:          # 平坦区梯度不可靠，跳过
                continue

            cos_d = (gxa * gxb + gya * gyb) / (ma * mb)
            cos_vals.append(cos_d)
            wts.append(w)
            asym_vals.append(abs(ma - mb) / (ma + mb))

    if len(cos_vals) < 4:
        return 0.0

    wts_arr   = np.array(wts, dtype=np.float32)
    mean_cos  = float(np.average(cos_vals,  weights=wts_arr))
    dir_score = 0.5 * (1.0 - mean_cos)              # [0, 1]
    asym_score = float(np.mean(asym_vals))           # [0, 1]

    return 0.6 * dir_score + 0.4 * asym_score


def _score_texture_candidate(
    candidate: _Candidate,
    pieces: List["Piece"],
    local_polys: List[np.ndarray],
) -> float:
    """
    计算一个候选的 J_texture：各缝得分按缝长加权平均。
    返回 0~1（越低表示纹理越连续）。
    """
    gray, min_xy, _ = _render_to_canvas(candidate, pieces, local_polys)
    if gray is None:
        return 0.0

    # 梯度图一次性计算，所有缝复用
    Gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    placed   = _layout_polygons(local_polys, candidate.placements)
    scored   = set()
    total    = 0.0
    weight   = 0.0

    for pi, pe, ni, ne, lp, lq in candidate.edge_pairs:
        key = tuple(sorted(((pi, pe), (ni, ne))))
        if key in scored:
            continue
        scored.add(key)

        poly_pi = placed[pi]
        nv = len(poly_pi)
        pt0_mm = poly_pi[pe]
        pt1_mm = poly_pi[(pe + 1) % nv]
        seam_len_mm = float(np.linalg.norm(pt1_mm - pt0_mm))
        if seam_len_mm < 8.0:
            continue

        pt0_px = (pt0_mm - min_xy) * _TEX_SCALE
        pt1_px = (pt1_mm - min_xy) * _TEX_SCALE

        s = _score_one_seam(Gx, Gy, pt0_px, pt1_px)
        total  += s * seam_len_mm
        weight += seam_len_mm

    return total / weight if weight > 0.0 else 0.0


def _rerank_by_texture(
    candidates: List[_Candidate],
    pieces: List["Piece"],
    local_polys: List[np.ndarray],
    lam: float = _TEX_LAM,
    top_k: int = _TEX_TOP_K,
) -> List[_Candidate]:
    """
    对前 top_k 个几何候选叠加纹理评分后重排序。
    J_total = J_shape + lam * J_texture（越低越好）。
    超出 top_k 的候选直接追加到末尾，不做评分。
    """
    to_score = candidates[:top_k]
    rest     = candidates[top_k:]

    ranked = []
    for c in to_score:
        jt      = _score_texture_candidate(c, pieces, local_polys)
        j_total = c.score + lam * jt
        print(f"    [纹理] shape={c.score:.2f}  tex={jt:.4f}  "
              f"total={j_total:.2f}  {c.width:.1f}x{c.height:.1f}mm")
        ranked.append((c, j_total))

    ranked.sort(key=lambda x: x[1])
    return [c for c, _ in ranked] + rest


# =============================================================================
# 角标对称验证
# =============================================================================

def _render_assembly_texture(
    cand: "_Candidate",
    cand_placed: Dict[int, np.ndarray],
    pieces: List["Piece"],
    piece_centers: List[np.ndarray],
    H_mm2px: np.ndarray,
    scale: float = 4.0,
) -> Optional[np.ndarray]:
    """
    用与角标图相同的坐标链渲染全图（BGR）。
    DFS mm → 碎片局部mm → 绝对mm → 相机像素 → 纹理局部像素。
    任意一片缺纹理则返回 None。
    """
    for pid in cand.placements:
        p = pieces[pid]
        if p.texture_bgr is None or p.texture_origin_px is None:
            return None

    all_v = np.vstack(list(cand_placed.values()))
    mn = all_v.min(0) - 2.0
    mx = all_v.max(0) + 2.0
    cw = max(4, int((mx[0] - mn[0]) * scale) + 2)
    ch = max(4, int((mx[1] - mn[1]) * scale) + 2)
    canvas = np.full((ch, cw, 3), 200, dtype=np.uint8)

    for pid, poly_dfs in cand_placed.items():
        p   = pieces[pid]
        pl  = cand.placements[pid]
        pts_canvas = ((poly_dfs - mn) * scale).astype(np.int32)
        mask = np.zeros((ch, cw), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_canvas], 255)
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            continue
        pts_dfs   = mn + np.column_stack([xs, ys]) / scale
        pts_local = (pts_dfs - pl.translation) @ pl.rotation
        pts_abs   = pts_local + piece_centers[pid]
        ones      = np.ones((len(pts_abs), 1))
        pts_h     = (H_mm2px @ np.hstack([pts_abs, ones]).T).T
        pts_px    = pts_h[:, :2] / pts_h[:, 2:3]
        ox, oy    = float(p.texture_origin_px[0]), float(p.texture_origin_px[1])
        tx = np.round(pts_px[:, 0] - ox).astype(int)
        ty = np.round(pts_px[:, 1] - oy).astype(int)
        th, tw = p.texture_bgr.shape[:2]
        valid  = (tx >= 0) & (tx < tw) & (ty >= 0) & (ty < th)
        canvas[ys[valid], xs[valid]] = p.texture_bgr[ty[valid], tx[valid]]

    return canvas


def _reconstruct_corner_patch_aligned(
    origin_pt: np.ndarray,
    d_u: np.ndarray,
    d_v: np.ndarray,
    size_mm: Tuple[float, float],
    best_placed: Dict[int, np.ndarray],
    placements: Dict[int, "_Placement"],
    pieces: List["Piece"],
    piece_centers: List[np.ndarray],
    H_mm2px: np.ndarray,
    scale: float = 4.0,
) -> np.ndarray:
    """
    沿牌面方向（d_u, d_v）重建角区灰度图像。
    origin_pt: 角区起点（DFS mm坐标，已含inset偏移）
    d_u, d_v: 牌面宽/高方向的单位向量（由minAreaRect计算）
    size_mm: (width_mm, height_mm)
    """
    w_px = max(1, int(size_mm[0] * scale))
    h_px = max(1, int(size_mm[1] * scale))
    patch = np.full((h_px, w_px), 200, dtype=np.uint8)

    for pid, poly_dfs in best_placed.items():
        p = pieces[pid]
        if p.texture_bgr is None or p.texture_origin_px is None:
            continue
        pl = placements[pid]

        # 将碎片多边形投影到角区坐标系（u=沿d_u，v=沿d_v）→ 生成掩码
        delta = (poly_dfs - origin_pt).astype(np.float64)
        u_coords = delta @ d_u
        v_coords = delta @ d_v
        poly_patch = np.column_stack([u_coords * scale,
                                      v_coords * scale]).astype(np.int32)
        mask = np.zeros((h_px, w_px), dtype=np.uint8)
        cv2.fillPoly(mask, [poly_patch], 255)
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            continue

        # 角区像素 → DFS mm → 碎片局部mm → 绝对mm → 相机像素 → 纹理局部像素
        pts_dfs = (origin_pt
                   + (xs / scale)[:, None] * d_u
                   + (ys / scale)[:, None] * d_v)
        pts_local = (pts_dfs - pl.translation) @ pl.rotation
        pts_abs   = pts_local + piece_centers[pid]
        ones      = np.ones((len(pts_abs), 1))
        pts_h     = (H_mm2px @ np.hstack([pts_abs, ones]).T).T
        pts_px    = pts_h[:, :2] / pts_h[:, 2:3]
        ox, oy    = float(p.texture_origin_px[0]), float(p.texture_origin_px[1])
        tx = np.round(pts_px[:, 0] - ox).astype(int)
        ty = np.round(pts_px[:, 1] - oy).astype(int)

        th, tw  = p.texture_bgr.shape[:2]
        valid   = (tx >= 0) & (tx < tw) & (ty >= 0) & (ty < th)
        gray_p  = cv2.cvtColor(p.texture_bgr, cv2.COLOR_BGR2GRAY)
        patch[ys[valid], xs[valid]] = gray_p[ty[valid], tx[valid]]

    return patch


def _check_corner_symmetry(
    best: "_Candidate",
    best_placed: Dict[int, np.ndarray],
    pieces: List["Piece"],
    piece_centers: List[np.ndarray],
    H_mm2px: np.ndarray,
    corner_w_mm: float = 20.0,
    corner_h_mm: float = 22.0,
    inset_mm: float = 3.0,
    ncc_thresh: float = 0.30,
) -> Tuple[bool, float]:
    """
    验证拼合扑克牌角标180°对称性（所有标准扑克均满足此性质）。
    用 minAreaRect 找真实角点，沿牌面方向（d_u, d_v）对齐采样，
    避免轴对齐 min/max 来自不同碎片导致角区落在 polygon 外的问题。
    """
    all_v = np.vstack(list(best_placed.values())).astype(np.float32)
    rect = cv2.minAreaRect(all_v.reshape(-1, 1, 2))
    box_pts_orig = cv2.boxPoints(rect).astype(np.float64)  # 原始几何角点，用于计算方向

    # 位置：将每个几何角点吸附到对角方向的极值多边形顶点（确保落在碎片上，不在缝隙中）
    all_v_f64 = all_v.astype(np.float64)
    _sums  = all_v_f64[:, 0] + all_v_f64[:, 1]
    _diffs = all_v_f64[:, 0] - all_v_f64[:, 1]
    _ext = np.array([
        all_v_f64[np.argmin(_sums)],
        all_v_f64[np.argmax(_sums)],
        all_v_f64[np.argmin(_diffs)],
        all_v_f64[np.argmax(_diffs)],
    ], dtype=np.float64)
    box_pts_snap = box_pts_orig.copy()
    for _bi in range(4):
        _dists = np.linalg.norm(_ext - box_pts_orig[_bi], axis=1)
        box_pts_snap[_bi] = _ext[np.argmin(_dists)]

    # 枚举策略：
    # 两条对角线（A=argmin/argmax x+y，B=另外两顶点）× 宽/高方向两种（w×h 或 h×w）
    # = 最多4个 NCC 值，取最大，消除：① 对角线选错 ② d_u/d_v方向与角标朝向不符
    sums = box_pts_orig[:, 0] + box_pts_orig[:, 1]
    i_a0 = int(np.argmin(sums))
    i_a1 = int(np.argmax(sums))
    diag_B = [i for i in range(4) if i != i_a0 and i != i_a1]
    i_b0, i_b1 = diag_B[0], diag_B[1]

    def _one_ncc(i_corner, i_opp, w_mm, h_mm):
        """返回(ncc, std0, std1, patch0, patch1, patch1r, ori0, ori1, du, dv)。"""
        v0 = box_pts_snap[i_corner]   # 位置：吸附到实际顶点
        v1 = box_pts_snap[i_opp]
        du = box_pts_orig[(i_corner + 1) % 4] - box_pts_orig[i_corner]  # 方向：用原始几何角点
        dv = box_pts_orig[(i_corner - 1) % 4] - box_pts_orig[i_corner]
        du = du / np.linalg.norm(du)
        dv = dv / np.linalg.norm(dv)
        ori0 = v0 + inset_mm * du + inset_mm * dv
        ori1 = v1 - inset_mm * du - inset_mm * dv - w_mm * du - h_mm * dv
        # 固定尺寸 patch（用于显示和 std 计算）
        p0 = _reconstruct_corner_patch_aligned(
            ori0, du, dv, (w_mm, h_mm),
            best_placed, best.placements, pieces, piece_centers, H_mm2px,
        )
        p1 = _reconstruct_corner_patch_aligned(
            ori1, du, dv, (w_mm, h_mm),
            best_placed, best.placements, pieces, piece_centers, H_mm2px,
        )
        p1r = cv2.rotate(p1, cv2.ROTATE_180)
        s0 = float(np.std(p0.astype(np.float64)))
        s1 = float(np.std(p1r.astype(np.float64)))
        if s0 < 1.0 or s1 < 1.0:
            return -1.0, s0, s1, p0, p1, p1r, ori0, ori1, du, dv
        # 带搜索窗口的 NCC：扩大 p1 范围，用 matchTemplate 容忍顶点定位误差 ±8mm
        _mg = 8.0
        ori1_wide = ori1 - _mg * du - _mg * dv
        p1_wide = _reconstruct_corner_patch_aligned(
            ori1_wide, du, dv, (w_mm + 2 * _mg, h_mm + 2 * _mg),
            best_placed, best.placements, pieces, piece_centers, H_mm2px,
        )
        p1r_wide = cv2.rotate(p1_wide, cv2.ROTATE_180)
        _res = cv2.matchTemplate(
            p1r_wide.astype(np.float32),
            p0.astype(np.float32),
            cv2.TM_CCOEFF_NORMED,
        )
        ncc = float(_res.max())
        return ncc, s0, s1, p0, p1, p1r, ori0, ori1, du, dv

    # 4种组合全部尝试
    variants = [
        (i_a0, i_a1, corner_w_mm, corner_h_mm, "A-wh"),
        (i_a0, i_a1, corner_h_mm, corner_w_mm, "A-hw"),
        (i_b0, i_b1, corner_w_mm, corner_h_mm, "B-wh"),
        (i_b0, i_b1, corner_h_mm, corner_w_mm, "B-hw"),
    ]
    best_score = -1e9   # 选变体用 max(0,ncc)×(s0+s1)，高对比度优先
    best_ncc, best_s0, best_s1 = -1.0, 0., 0.
    best_p0, best_p1, best_p1r, best_label = None, None, None, ""
    best_ori0, best_ori1, best_du, best_dv = None, None, None, None
    best_w, best_h = corner_w_mm, corner_h_mm
    for _ic, _io, _w, _h, _lbl in variants:
        _n, _s0, _s1, _p0, _p1, _p1r, _ori0, _ori1, _du, _dv = _one_ncc(_ic, _io, _w, _h)
        _sel = max(0.0, _n) * (_s0 + _s1)   # 角标区对比度高→std大→优先选
        print(f"  [变体] {_lbl}  NCC={_n:.3f}  s0={_s0:.1f}  s1={_s1:.1f}  score={_sel:.2f}")
        if _sel > best_score:
            best_score = _sel
            best_ncc, best_s0, best_s1 = _n, _s0, _s1
            best_p0, best_p1, best_p1r, best_label = _p0, _p1, _p1r, _lbl
            best_ori0, best_ori1, best_du, best_dv = _ori0, _ori1, _du, _dv
            best_w, best_h = _w, _h

    ncc, std_tl, std_br = best_ncc, best_s0, best_s1

    # 生成标注图（返回给调用方，由调用方决定保存时机）
    ann_img = None
    if best_ori0 is not None:
        _asm_img = _render_assembly_texture(best, best_placed, pieces, piece_centers, H_mm2px)
        if _asm_img is not None:
            _ann_scale = 4.0
            all_v_ann = np.vstack(list(best_placed.values()))
            _mn_ann = all_v_ann.min(0) - 2.0
            def _mm2canvas(pt_mm):
                v = (np.array(pt_mm, dtype=np.float64) - _mn_ann) * _ann_scale
                return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int32)
            for _ori, _col, _lbl2 in [
                (best_ori0, (0, 255, 0), "TL"),
                (best_ori1, (0, 0, 255), "BR"),
            ]:
                _c0 = _mm2canvas(_ori)
                _c1 = _mm2canvas(_ori + best_du * best_w + best_dv * best_h)
                _cr = _mm2canvas(_ori + best_du * best_w)
                _cd = _mm2canvas(_ori + best_dv * best_h)
                _box_ann = np.array([_c0, _cr, _c1, _cd], dtype=np.int32)
                cv2.polylines(_asm_img, [_box_ann], True, _col, 2)
                cv2.putText(_asm_img, _lbl2, (int(_c0[0]), int(_c0[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, _col, 2)
            print(f"  [角标调试] ori0(TL)={best_ori0}  ori1(BR)={best_ori1}")
            print(f"  [角标调试] du={best_du}  dv={best_dv}  w={best_w:.1f}  h={best_h:.1f}")
            ann_img = _asm_img

    # 对比度不足：无法判断（仍视为不通过）
    if std_tl < 5.0 or std_br < 5.0:
        print(f"  [角标对称] 对比度不足(tl={std_tl:.1f} br={std_br:.1f})，无法验证角标")
        return False, 0.0, ann_img, best_p0, best_p1, best_p1r

    ok = ncc > ncc_thresh
    print(f"  [角标对称] 变体={best_label}  tl_std={std_tl:.1f}  br_std={std_br:.1f}  "
          f"NCC={ncc:.3f}{'  [OK]通过' if ok else '  [WARN]不对称，拼合方向可能有误'}")
    return ok, ncc, ann_img, best_p0, best_p1, best_p1r


# =============================================================================
# 公开入口
# =============================================================================

def solve_geometric(
    pieces: List[Piece],
    cfg: Optional[Config] = None,
    puzzle_origin_mm: Optional[Tuple[float, float]] = None,
    rectified_bgr: Optional[np.ndarray] = None,
    H_px2mm: Optional[np.ndarray] = None,
) -> list:
    """
    B路几何拼接主函数（DFS回溯）。
    返回 List[PlacementResult]；PlacementResult 从 vision.solver 惰性导入以避免循环依赖。
    """
    # 惰性导入，避免与 vision.solver 的循环依赖
    from vision.solver import PlacementResult

    if cfg is None:
        cfg = get_config()

    if len(pieces) < 2:
        print("[B路] 需要至少 2 片碎片")
        return []
    if any(p.vertices_mm is None for p in pieces):
        print("[B路] 缺少 mm 顶点坐标，请先完成相机标定")
        return []

    # 过滤退化碎片：顶点<3（简化后2顶点多边形面积为0，无法参与拼接）
    _n_raw = len(pieces)
    pieces = [p for p in pieces if p.vertices_mm is not None and len(p.vertices_mm) >= 3]
    if len(pieces) < _n_raw:
        print(f"[B路] 已过滤 {_n_raw - len(pieces)} 片退化碎片（顶点<3）")
    if len(pieces) < 2:
        print("[B路] 过滤后剩余碎片不足 2 片")
        return []

    # ── 1. 构建局部多边形（CCW，吸嘴点归零）─────────────────────────────
    # 以吸嘴点为原点：DFS 结果直接是吸嘴坐标，无需额外偏移
    local_polys: List[np.ndarray] = []
    piece_centers: List[np.ndarray] = []
    for p in pieces:
        verts = _ccw(p.vertices_mm.astype(np.float64))
        suction = np.array(
            p.suction_point_mm if p.suction_point_mm is not None
            else p.center_mm   if p.center_mm        is not None
            else verts.mean(0), dtype=np.float64
        )
        local_polys.append(verts - suction)
        piece_centers.append(suction)

    total_area = sum(p.area_mm2 for p in pieces)

    # ── 1b. 伪顶点净化（mm 空间，绕开像素畸变）─────────────────────────
    # artifact_clean_enabled=False 时完全跳过净化（白纸调试或纯几何测试时使用）
    _min_e_white = getattr(cfg, 'rect_min_edge_mm',             15.0)
    _min_e_card  = getattr(cfg, 'card_artifact_min_edge_mm',  25.0)
    _min_e_force = getattr(cfg, 'card_artifact_force_edge_mm', 12.0)  # 小于此值强制删，不做角度检查
    _min_turn    = getattr(cfg, 'card_artifact_min_turn_deg',  20.0)
    _clean_enabled = getattr(cfg, 'artifact_clean_enabled',   True)

    # 净化前诊断（确认 vertices_mm 从检测过来时的初始状态）
    print("[B路] 净化前边长:")
    for _dp, _dlp in zip(pieces, local_polys):
        _ls = _edge_lengths(_dlp)
        _es = "  ".join(f"e{_j}:{_v:.1f}" for _j, _v in enumerate(_ls))
        print(f"  P{_dp.index}[{_dp.piece_type}] {len(_dlp)}边: {_es}")

    for i in range(len(local_polys)):
        if not _clean_enabled:
            pieces[i].num_sides   = len(local_polys[i])
            pieces[i].vertices_mm = local_polys[i] + piece_centers[i]
            continue
        lp = local_polys[i]
        pt = pieces[i].piece_type  # 'white' | 'card' | 'color' | 'unknown'
        ls_now = _edge_lengths(lp)
        # 若任意边 < 20mm，无论分类结果如何都走扑克逻辑
        # （白纸保证 >20mm，<20mm 的边必来自扑克印刷图案）
        # 白纸片：分类结果可信，直接走白纸路径，不做扑克净化
        # 扑克/彩色/未知：若分类有误仍可靠 <15mm 兜底（白纸保证 >20mm，留 5mm 容差）
        is_card = (pt in ('card', 'color', 'unknown') or
                   (pt != 'white' and len(ls_now) > 0 and float(np.min(ls_now)) < 15.0))

        if is_card:
            # Pass 1：短边去除——两端均为近共线（转角 < _min_turn）才认为是印刷伪影
            # 若一端是真实切割角（转角 >= _min_turn），保留该边（可能是真实短切边，如 K 牌 25mm 切口）
            for _ in range(len(lp)):
                if len(lp) <= 3:
                    break
                ls = _edge_lengths(lp)
                n = len(lp)
                removed = False
                for _jj in np.argsort(ls):
                    if ls[_jj] >= _min_e_card:
                        break
                    _jj1 = (_jj + 1) % n
                    # 端点 _jj 的转角
                    _pa = lp[(_jj - 1) % n] - lp[_jj]
                    _pb = lp[_jj1] - lp[_jj]
                    _la = float(np.linalg.norm(_pa)); _lb = float(np.linalg.norm(_pb))
                    _ta = (180.0 - math.degrees(math.acos(float(np.clip(
                        np.dot(_pa, _pb) / (_la * _lb), -1.0, 1.0))))) if _la > 1e-6 and _lb > 1e-6 else 0.0
                    # 端点 _jj1 的转角
                    _pc = lp[_jj] - lp[_jj1]
                    _pd = lp[(_jj1 + 1) % n] - lp[_jj1]
                    _lc = float(np.linalg.norm(_pc)); _ld = float(np.linalg.norm(_pd))
                    _tb = (180.0 - math.degrees(math.acos(float(np.clip(
                        np.dot(_pc, _pd) / (_lc * _ld), -1.0, 1.0))))) if _lc > 1e-6 and _ld > 1e-6 else 0.0
                    # ≥ _min_e_force 且至少一端为真实角 → 保留
                    if ls[_jj] >= _min_e_force and (_ta >= _min_turn or _tb >= _min_turn):
                        continue
                    # 两侧边延长线求交点，作为新真顶点替换短边两端（n→n-1）
                    _vp = lp[(_jj - 1) % n]
                    _vn = lp[(_jj1 + 1) % n]
                    _d1 = lp[_jj] - _vp        # 前向边方向 v_prev→v_a
                    _d2 = _vn - lp[_jj1]       # 后向边方向 v_b→v_next
                    _cr = float(_d1[0] * _d2[1] - _d1[1] * _d2[0])
                    _l1 = float(np.linalg.norm(_d1)); _l2 = float(np.linalg.norm(_d2))
                    # 归一化叉积 = |sin(θ)|；θ→0°或180°时交点在无穷远，属于虚假交点
                    _sin_ang = abs(_cr) / max(_l1 * _l2, 1e-9)
                    if _sin_ang > 0.26:  # sin(15°)≈0.26，两线夹角足够大才算有效
                        _dt = lp[_jj1] - _vp
                        _t  = float(_dt[0] * _d2[1] - _dt[1] * _d2[0]) / _cr
                        _corner = _vp + _t * _d1
                        print(f"  P{pieces[i].index}: 扑克伪短边→交点 "
                              f"e{_jj}={ls[_jj]:.1f}mm(转角{_ta:.0f}°/{_tb:.0f}°) 已删，{n}→{n - 1}边")
                        if _jj < _jj1:
                            lp = np.delete(lp, _jj1, axis=0)
                            lp[_jj] = _corner
                        else:  # 绕回：_jj=n-1, _jj1=0，先删高下标不影响低下标
                            lp = np.delete(lp, _jj, axis=0)
                            lp[_jj1] = _corner
                    else:
                        # 延长线近平行（夹角≈0°或180°），交点无效，退回双删
                        if n - 2 >= 3:
                            _to_del = sorted({_jj, _jj1}, reverse=True)
                            print(f"  P{pieces[i].index}: 扑克伪短边(近平行双删) "
                                  f"e{_jj}={ls[_jj]:.1f}mm(转角{_ta:.0f}°/{_tb:.0f}°) 已删，{n}→{n - 2}边")
                            lp = np.delete(lp, _to_del, axis=0)
                        else:
                            _del_one = _jj if _ta <= _tb else _jj1
                            print(f"  P{pieces[i].index}: 扑克伪短边(近平行单删) "
                                  f"e{_jj}={ls[_jj]:.1f}mm(转角{_ta:.0f}°/{_tb:.0f}°) 已删，{n}→{n - 1}边")
                            lp = np.delete(lp, _del_one, axis=0)
                    removed = True
                    break
                if not removed:
                    break

            # Pass 2：近共线顶点去除——转角 < min_turn 的顶点单独删
            for _ in range(len(lp)):
                if len(lp) <= 3:
                    break
                n = len(lp)
                _mt, _mi = 360.0, -1
                for k in range(n):
                    p0 = lp[(k - 1) % n]; p1 = lp[k]; p2 = lp[(k + 1) % n]
                    v1 = p0 - p1; l1 = float(np.linalg.norm(v1))
                    v2 = p2 - p1; l2 = float(np.linalg.norm(v2))
                    if l1 < 1e-6 or l2 < 1e-6:
                        continue
                    _cos  = float(np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0))
                    _turn = 180.0 - math.degrees(math.acos(_cos))
                    if _turn < _mt:
                        _mt, _mi = _turn, k
                if _mi >= 0 and _mt < _min_turn:
                    print(f"  P{pieces[i].index}: 扑克近共线顶点 "
                          f"转角={_mt:.1f}° 已删，{n}→{n - 1}边")
                    lp = np.delete(lp, _mi, axis=0)
                else:
                    break
        else:
            # 白纸片：原透视畸变逻辑（px_e 暂用 mm 值，比值判断已失效，
            # 仅靠 _min_e_white=15mm 绝对阈值兜底）
            while len(lp) > 3:
                ls = _edge_lengths(lp)
                j  = int(np.argmin(ls))
                if ls[j] >= _min_e_white:
                    break
                print(f"  P{pieces[i].index}: 透视畸变伪影 "
                      f"e{j}={ls[j]:.1f}mm 已删，{len(lp)}→{len(lp) - 1}边")
                lp = np.delete(lp, j, axis=0)

        local_polys[i]        = lp
        pieces[i].num_sides   = len(lp)
        pieces[i].vertices_mm = lp + piece_centers[i]

    print(f"[B路] {len(pieces)} 片碎片:")
    for k, (p, lp) in enumerate(zip(pieces, local_polys)):
        ls = _edge_lengths(lp)
        edge_str = "  ".join(f"e{i}:{v:.1f}" for i, v in enumerate(ls))
        print(f"  P{p.index} [{p.num_sides}边 {p.area_mm2:.0f}mm2]  {edge_str}")

    # ── 2. DFS 搜索 ─────────────────────────────────────────────────────
    # 边数最多的片优先作锚点（K牌5边中心件必须先放，才能做辐射形拼合）；
    # 边数相同时面积大的优先
    order = sorted(range(len(pieces)),
                   key=lambda i: (pieces[i].num_sides, pieces[i].area_mm2),
                   reverse=True)
    sorted_polys = [local_polys[i] for i in order]

    # ── 边缘灰度剖面（净化后mm顶点→H_mm2px精确映射→各片纹理局部坐标）────
    _H_mm2px: Optional[np.ndarray] = None
    if H_px2mm is not None:
        try:
            _H_mm2px = np.linalg.inv(H_px2mm)
        except np.linalg.LinAlgError:
            pass

    edge_profiles_card: List[List[Optional[np.ndarray]]] = []
    for i, p in enumerate(pieces):
        lp = local_polys[i]
        n_edges = len(lp)
        if (_H_mm2px is None or p.piece_type == 'white'
                or p.texture_bgr is None or p.texture_origin_px is None):
            edge_profiles_card.append([None] * n_edges)
            continue
        # 净化后局部mm → 绝对mm → 相机像素 → 纹理局部像素
        lp_global = lp + piece_centers[i]
        ox = float(p.texture_origin_px[0])
        oy = float(p.texture_origin_px[1])
        verts_tex = np.array([
            (_H_mm2px @ np.array([float(v[0]), float(v[1]), 1.0], dtype=np.float64))[:2]
            / (_H_mm2px @ np.array([float(v[0]), float(v[1]), 1.0], dtype=np.float64))[2]
            - np.array([ox, oy])
            for v in lp_global
        ], dtype=np.float64)
        gray_p = cv2.cvtColor(p.texture_bgr, cv2.COLOR_BGR2GRAY)
        _inset = getattr(cfg, 'grad_inset_px', 5.0)
        edge_profiles_card.append([
            _sample_edge_profile(gray_p, verts_tex, e, inset_px=_inset)
            for e in range(n_edges)
        ])

    sorted_card_profiles = [edge_profiles_card[i] for i in order]

    # 扑克场景：用专属宽松阈值（K牌切口不规则，矩形度/填充率天然偏低）
    _is_card_puzzle = any(p.piece_type == "card" for p in pieces)
    if _is_card_puzzle:
        _rect_fill    = getattr(cfg, 'card_rect_fill_min',   0.92)
        _direct_fill  = getattr(cfg, 'card_direct_fill_min', 0.90)
    else:
        _rect_fill    = getattr(cfg, 'rect_fill_min',   0.92)
        _direct_fill  = getattr(cfg, 'direct_fill_min', 0.77)

    # 扑克牌：DFS内边缘剖面梯度剪枝已禁用。
    # 原因：inset采样的是切口内侧纹理（P1切口上方 vs P2切口下方），
    # 两侧来自卡面不同位置，相关系数偶然接近0，会误拒绝正确配对。
    # 角标180°对称NCC（_check_corner_symmetry）承担真正的纹理验证职责。
    candidates = _search_candidates(
        sorted_polys, total_area,
        min_composite_edge=getattr(cfg, 'min_composite_edge_mm', 0.0),
        rect_fill=_rect_fill,
        direct_fill_min=_direct_fill,
        area_tol=getattr(cfg, 'rect_area_tol', 0.20),
        edge_profiles=None if _is_card_puzzle else sorted_card_profiles,
        grad_corr_min=getattr(cfg, 'grad_corr_min',    0.30),
        grad_clarity_min=getattr(cfg, 'grad_clarity_min', 8.0),
    )

    # 将 DFS 内部索引映射回原始 pieces 索引
    for c in candidates:
        c.placements = {order[k]: v for k, v in c.placements.items()}
        c.edge_pairs  = [(order[pi], pe, order[ni], ne, lp_, lq)
                         for pi, pe, ni, ne, lp_, lq in c.edge_pairs]

    if not candidates:
        print("[B路] 未找到合法拼接方案（外缘矩形验证失败）")
        return []

    # ── 候选边配对诊断（重排序前）────────────────────────────────────────
    print(f"[B路] 几何候选 {len(candidates)} 个（重排序前顺序）:")
    for _ci, _c in enumerate(candidates):
        _ep_str = " | ".join(
            f"P{pieces[pi].index}e{pe}<->P{pieces[ni].index}e{ne}"
            f"({lp_:.1f}/{lq:.1f})"
            for pi, pe, ni, ne, lp_, lq in _c.edge_pairs
        )
        print(f"  #{_ci} score={_c.score:.2f}  {_ep_str}")

    # ── 扑克牌碎片：用纹理评分对候选重排序 ───────────────────────────────
    if (len(candidates) > 1
            and any(p.piece_type == "card" for p in pieces)
            and any(p.texture_bgr is not None and p.texture_origin_px is not None
                    for p in pieces)):
        print("[B路] 扑克牌碎片：启用纹理评分重排序")
        candidates = _rerank_by_texture(candidates, pieces, local_polys)
        print("[B路] 重排序后顺序:")
        for _ci, _c in enumerate(candidates):
            _ep_str = " | ".join(
                f"P{pieces[pi].index}e{pe}<->P{pieces[ni].index}e{ne}"
                f"({lp_:.1f}/{lq:.1f})"
                for pi, pe, ni, ne, lp_, lq in _c.edge_pairs
            )
            print(f"  #{_ci} score={_c.score:.2f}  {_ep_str}")

    # ── 角标180°对称验证：轮询候选直到通过，或全部失败取几何最优 ──────
    _do_sym = (_H_mm2px is not None
               and any(p.texture_bgr is not None for p in pieces))
    best = candidates[0]
    best_placed = _layout_polygons(local_polys, best.placements)
    if _do_sym:
        _best_ncc_val    = -1.0
        _best_ncc_cand   = candidates[0]
        _best_ncc_placed = best_placed
        _best_ncc_idx    = 0
        _best_ann_img    = None
        _best_patch_tl   = None
        _best_patch_br   = None
        _best_patch_brr  = None
        _dbg_dir_ncc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "calib_data")
        _ncc_score_tol = 3.0   # 只对 score ≤ 最优+3 的候选跑 NCC，避免几何差的候选靠 NCC 翻盘
        for _ci, _cand in enumerate(candidates):
            if _cand.score > candidates[0].score + _ncc_score_tol:
                print(f"[B路] 候选#{_ci} score={_cand.score:.2f} 远差于最优，跳过NCC")
                continue
            _cand_placed = _layout_polygons(local_polys, _cand.placements)
            _ok_sym, _ncc, _ann_img, _p0, _p1, _p1r = _check_corner_symmetry(
                _cand, _cand_placed, pieces, piece_centers, _H_mm2px,
            )
            # 保存候选全图 debug（纹理图，fallback几何图）
            _dbg_img = _render_assembly_texture(
                _cand, _cand_placed, pieces, piece_centers, _H_mm2px)
            if _dbg_img is None:
                _COLORS_DBG = [(255,80,80),(80,255,80),(80,80,255),
                               (255,255,80),(80,255,255),(255,80,255)]
                _all_v = np.vstack(list(_cand_placed.values()))
                _mn = _all_v.min(0) - 5.0; _mx = _all_v.max(0) + 5.0
                _sc = 4.0
                _dw = max(4, int((_mx[0]-_mn[0])*_sc)+2)
                _dh = max(4, int((_mx[1]-_mn[1])*_sc)+2)
                _dbg_img = np.full((_dh, _dw, 3), 240, dtype=np.uint8)
                for _di, (_didx, _dpoly) in enumerate(_cand_placed.items()):
                    _pts = ((_dpoly - _mn) * _sc).astype(np.int32)
                    cv2.fillPoly(_dbg_img, [_pts], _COLORS_DBG[_di % len(_COLORS_DBG)])
                    cv2.polylines(_dbg_img, [_pts], True, (0,0,0), 2)
                    _cx, _cy = int(_pts[:,0].mean()), int(_pts[:,1].mean())
                    cv2.putText(_dbg_img, f"P{pieces[_didx].index}", (_cx-15, _cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(_dbg_img, f"#{_ci} NCC={_ncc:.3f}", (4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,180), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(_dbg_dir_ncc, f"debug_ncc_cand{_ci:02d}.png"),
                        _dbg_img)
            if _ncc > _best_ncc_val:
                _best_ncc_val    = _ncc
                _best_ncc_cand   = _cand
                _best_ncc_placed = _cand_placed
                _best_ncc_idx    = _ci
                _best_ann_img    = _ann_img
                _best_patch_tl   = _p0
                _best_patch_br   = _p1
                _best_patch_brr  = _p1r
            # 不用阈值做早退，遍历所有候选取NCC最高
            print(f"[B路] 候选#{_ci} NCC={_ncc:.3f}")
        # NCC超过阈值才用NCC排序，否则保持几何最优（几何分更可靠）
        if _best_ncc_val >= 0.40:
            best        = _best_ncc_cand
            best_placed = _best_ncc_placed
            print(f"[B路] NCC验证通过，使用NCC最高候选#{_best_ncc_idx} "
                  f"NCC={_best_ncc_val:.3f}  score={best.score:.2f}")
        else:
            print(f"[B路] NCC偏低({_best_ncc_val:.3f}<0.40)，保持几何最优候选#0  "
                  f"score={best.score:.2f}")
        # 保存NCC最高候选的调试图（与 debug_assembly_best 同一候选）
        _s = 4
        if _best_ann_img is not None:
            cv2.imwrite(os.path.join(_dbg_dir_ncc, "debug_corner_annotated.png"), _best_ann_img)
        if _best_patch_tl is not None:
            cv2.imwrite(os.path.join(_dbg_dir_ncc, "debug_corner_tl.png"),
                        cv2.resize(_best_patch_tl,
                                   (_best_patch_tl.shape[1]*_s, _best_patch_tl.shape[0]*_s),
                                   interpolation=cv2.INTER_NEAREST))
        if _best_patch_br is not None:
            cv2.imwrite(os.path.join(_dbg_dir_ncc, "debug_corner_br.png"),
                        cv2.resize(_best_patch_br,
                                   (_best_patch_br.shape[1]*_s, _best_patch_br.shape[0]*_s),
                                   interpolation=cv2.INTER_NEAREST))
        if _best_patch_brr is not None:
            cv2.imwrite(os.path.join(_dbg_dir_ncc, "debug_corner_br_rot.png"),
                        cv2.resize(_best_patch_brr,
                                   (_best_patch_brr.shape[1]*_s, _best_patch_brr.shape[0]*_s),
                                   interpolation=cv2.INTER_NEAREST))
    else:
        print(f"[B路] 最优: score={best.score:.2f}  {best.width:.1f}x{best.height:.1f}mm  "
              f"共 {len(candidates)} 个候选")

    # 无条件保存最终选用候选的全图
    _dbg_out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "calib_data", "debug_assembly_best.png")
    _bp = _layout_polygons(local_polys, best.placements)
    _bimg = _render_assembly_texture(best, _bp, pieces, piece_centers,
                                     _H_mm2px) if _H_mm2px is not None else None
    if _bimg is None:
        _COLORS_BEST = [(255,80,80),(80,255,80),(80,80,255),
                        (255,255,80),(80,255,255),(255,80,255)]
        _bv = np.vstack(list(_bp.values()))
        _bmn = _bv.min(0)-5.0; _bmx = _bv.max(0)+5.0; _bsc = 4.0
        _bw = max(4, int((_bmx[0]-_bmn[0])*_bsc)+2)
        _bh = max(4, int((_bmx[1]-_bmn[1])*_bsc)+2)
        _bimg = np.full((_bh, _bw, 3), 240, dtype=np.uint8)
        for _bi, (_bidx, _bpoly) in enumerate(_bp.items()):
            _bpts = ((_bpoly - _bmn) * _bsc).astype(np.int32)
            cv2.fillPoly(_bimg, [_bpts], _COLORS_BEST[_bi % len(_COLORS_BEST)])
            cv2.polylines(_bimg, [_bpts], True, (0,0,0), 2)
            _bcx, _bcy = int(_bpts[:,0].mean()), int(_bpts[:,1].mean())
            cv2.putText(_bimg, f"P{pieces[_bidx].index}", (_bcx-15, _bcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
    cv2.imwrite(_dbg_out, _bimg)
    print(f"[B路] 全图已保存 → {_dbg_out}")

    used_in_best: set = set()
    for pi, pe, ni, ne, _, _ in best.edge_pairs:
        used_in_best.add((pi, pe))
        used_in_best.add((ni, ne))

    # ── 闭合边扫描：检测已放片间剩余贴合边（环形拓扑第4条边）──────────
    closure_pairs: List[Tuple] = []
    placed_idxs = sorted(best_placed.keys())
    for ii, ia in enumerate(placed_idxs):
        poly_a = best_placed[ia]
        na = len(poly_a)
        for ib in placed_idxs[ii + 1:]:
            poly_b = best_placed[ib]
            nb = len(poly_b)
            for ea in range(na):
                if (ia, ea) in used_in_best:
                    continue
                pa0 = poly_a[ea]
                pa1 = poly_a[(ea + 1) % na]
                la = float(np.linalg.norm(pa1 - pa0))
                for eb in range(nb):
                    if (ib, eb) in used_in_best:
                        continue
                    pb0 = poly_b[eb]
                    pb1 = poly_b[(eb + 1) % nb]
                    lb = float(np.linalg.norm(pb1 - pb0))
                    if abs(la - lb) > max(2.5, 0.055 * min(la, lb)):
                        continue
                    # 反平行贴合：pa0≈pb1 且 pa1≈pb0
                    d = max(float(np.linalg.norm(pa0 - pb1)),
                            float(np.linalg.norm(pa1 - pb0)))
                    if d < max(5.0, 0.06 * la):
                        closure_pairs.append((ia, ea, ib, eb, la, lb))

    # 生成树边按宿主边分组（同一宿主边的多片组合归为1行）
    span_groups: Dict[tuple, list] = {}
    for pi, pe, ni, ne, lp_, lq in best.edge_pairs:
        key = (pi, pe)
        if key not in span_groups:
            span_groups[key] = []
        span_groups[key].append((ni, ne, lp_, lq))

    # 检测多宿主：同一新片边 (ni,ne) 被多条宿主边覆盖（多宿主复合缝）
    new_to_hosts: Dict[tuple, list] = {}
    for (pi, pe), covering in span_groups.items():
        if len(covering) == 1:
            ni, ne, lp_, lq = covering[0]
            new_to_hosts.setdefault((ni, ne), []).append((pi, pe, lp_))
    multi_host_keys = {k for k, v in new_to_hosts.items() if len(v) > 1}
    multi_host_host_edges = {
        (pi, pe) for k in multi_host_keys for pi, pe, _ in new_to_hosts[k]
    }

    n_comp_groups = sum(1 for v in span_groups.values() if len(v) > 1)
    n_mh_groups = len(multi_host_keys)
    tags = []
    if n_comp_groups:
        tags.append(f"{n_comp_groups}组覆盖")
    if n_mh_groups:
        tags.append(f"{n_mh_groups}组多宿主")
    tag_str = f"  其中{'、'.join(tags)}" if tags else ""
    # 逻辑条数 = span_groups 数 - 多宿主额外宿主条数
    logical_count = len(span_groups) - sum(len(v) - 1 for v in new_to_hosts.values() if len(v) > 1)
    print(f"[B路] 匹配的边对（生成树{logical_count}条{tag_str} + 闭合{len(closure_pairs)}条）:")

    printed_mh: set = set()
    for (pi, pe), covering in span_groups.items():
        lp_ = covering[0][2]
        if (pi, pe) in multi_host_host_edges:
            # 多宿主边：聚合为一条显示，跳过同一新片的后续宿主
            ni, ne = covering[0][0], covering[0][1]
            mk = (ni, ne)
            if mk in printed_mh:
                continue
            printed_mh.add(mk)
            hosts = new_to_hosts[mk]        # [(pi,pe,lp), ...]
            seam_total = sum(hlp for _, _, hlp in hosts)
            # 新片真实边长：多宿主各段 lq 之和即为真实边长（lq_k = lq_real * lp_k / seam）
            lq_real = sum(lq_k for _pi2, _pe2, ni2, ne2, _, lq_k in best.edge_pairs
                          if (ni2, ne2) == mk)
            host_parts = " + ".join(
                f"P{pieces[hp].index}e{he}({hlp:.1f})" for hp, he, hlp in hosts
            )
            print(f"  [{host_parts}] = {seam_total:.1f} <-> "
                  f"P{pieces[ni].index}e{ne}({lq_real:.1f})  "
                  f"diff={abs(seam_total - lq_real):.1f}mm  [多宿主]")
        elif len(covering) == 1:
            ni, ne, _, lq = covering[0]
            print(f"  P{pieces[pi].index}e{pe}({lp_:.1f}) <-> "
                  f"P{pieces[ni].index}e{ne}({lq:.1f})  diff={abs(lp_-lq):.1f}mm")
        else:
            total_lq = sum(lq for _, _, _, lq in covering)
            parts = " + ".join(
                f"P{pieces[ni].index}e{ne}({lq:.1f})" for ni, ne, _, lq in covering
            )
            print(f"  P{pieces[pi].index}e{pe}({lp_:.1f}) <-> [{parts}] = {total_lq:.1f}"
                  f"  diff={abs(lp_ - total_lq):.1f}mm  [组合]")
    for pi, pe, ni, ne, la, lb in closure_pairs:
        print(f"  P{pieces[pi].index}e{pe}({la:.1f}) <-> "
              f"P{pieces[ni].index}e{ne}({lb:.1f})  diff={abs(la-lb):.1f}mm  [闭合]")

    # ── 3. minAreaRect → 旋转对齐（长轴水平）──────────────────────────
    assembled = _layout_polygons(local_polys, best.placements)

    all_pts = np.vstack(list(assembled.values())).astype(np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(all_pts)).astype(np.float64)
    edges = np.roll(box, -1, axis=0) - box
    lengths = np.linalg.norm(edges, axis=1)
    longest = edges[int(np.argmax(lengths))]
    angle_deg = math.degrees(math.atan2(float(longest[1]), float(longest[0])))

    G = _rotation_matrix(-angle_deg)
    rotated: Dict[int, np.ndarray] = {idx: poly @ G.T for idx, poly in assembled.items()}
    all_rot = np.vstack(list(rotated.values()))
    minxy = all_rot.min(0)
    maxxy = all_rot.max(0)

    # 确保高 > 宽（长边竖向，与矩形长边平行）
    if (maxxy[0] - minxy[0]) > (maxxy[1] - minxy[1]):
        G90 = _rotation_matrix(90.0)
        G = G90 @ G
        rotated = {idx: poly @ G90.T for idx, poly in rotated.items()}
        all_rot = np.vstack(list(rotated.values()))
        minxy = all_rot.min(0)
        maxxy = all_rot.max(0)

    puzzle_w = float(maxxy[0] - minxy[0])
    puzzle_h = float(maxxy[1] - minxy[1])

    # ── 4. 平移到目标区 ────────────────────────────────────────────────
    if puzzle_origin_mm is None:
        sep_x_mm   = getattr(cfg, 'separator_x_mm', 105.0)
        paper_h    = 297.0
        sep_y_mm   = getattr(cfg, 'separator_y_mm', paper_h / 2.0)
        gap_mm     = getattr(cfg, 'placement_gap_mm', 15.0)       # 分界线下方安全间距
        edge_mm    = getattr(cfg, 'placement_edge_mm', 10.0)    # 纸张下边缘留白
        x_offset   = getattr(cfg, 'placement_x_offset_mm', 15.0) # 拼图区 X 正向额外偏移
        ox = max(5.0, sep_x_mm / 2.0 - puzzle_w / 2.0) + x_offset
        avail_top  = sep_y_mm + gap_mm                         # 可用区上沿
        avail_bot  = paper_h - edge_mm                         # 可用区下沿
        # 居中于可用区，同时保证不超出下沿
        oy = (avail_top + avail_bot) / 2.0 - puzzle_h / 2.0
        oy = max(avail_top, min(oy, avail_bot - puzzle_h))
        puzzle_origin_mm = (ox, oy)

    offset = np.array(puzzle_origin_mm, np.float64) - minxy
    final: Dict[int, np.ndarray] = {
        idx: poly + offset for idx, poly in rotated.items()
    }

    # ── 4b. 片间微撑开（各片向外推 piece_gap_mm，总间隙 = 2×）────────────
    gap_half = getattr(cfg, 'piece_gap_mm', 3.0)
    if gap_half > 0:
        all_cens = np.array([poly.mean(0) for poly in final.values()])
        centroid = all_cens.mean(0)
        for idx in final:
            d = final[idx].mean(0) - centroid
            d_norm = float(np.linalg.norm(d))
            if d_norm > 1e-6:
                final[idx] = final[idx] + (d / d_norm) * gap_half

    fill = total_area / max(puzzle_w * puzzle_h, 1.0)
    print(f"[B路] 拼合结果: {puzzle_w:.1f}x{puzzle_h:.1f}mm  fill={fill:.3f}")
    print("[B路] 各片目标位置:")
    for idx, fv in final.items():
        cx, cy = fv.mean(0)
        print(f"  P{pieces[idx].index}: target_center=({cx:.1f},{cy:.1f})")

    # ── 5. SVD 求旋转角 → 生成 PlacementResult ─────────────────────────
    results = []
    for idx, tgt_verts in final.items():
        p = pieces[idx]
        # 不重新排序 src_verts，避免与 dst_verts 顶点顺序错位破坏 SVD 旋转
        src_verts = p.vertices_mm.astype(np.float64)
        dst_verts = tgt_verts.astype(np.float64)

        try:
            delta_deg = _svd_rotation_deg(src_verts, dst_verts)
            delta_deg = (delta_deg + 180.0) % 360.0 - 180.0
        except Exception:
            delta_deg = 0.0

        # 吸嘴点通过 SVD 旋转映射到目标坐标系
        suction_src   = np.array(piece_centers[idx], dtype=np.float64)
        src_cen       = src_verts.mean(0)
        dst_cen       = dst_verts.mean(0)
        suction_delta = suction_src - src_cen
        R_svd         = _rotation_matrix(delta_deg)
        target_suction = tuple(float(v) for v in (dst_cen + R_svd @ suction_delta))
        suction        = tuple(float(v) for v in suction_src)

        results.append(PlacementResult(
            piece_index        = p.index,
            template_id        = -1,
            suction_point_mm   = suction,
            target_center_mm   = target_suction,
            delta_rotation_deg = delta_deg,
            match_rmse_mm      = 0.0,
            confidence         = round(fill, 3),
            method             = "geometric",
            target_vertices_mm = tgt_verts.copy(),
        ))
        print(f"  P{p.index}: suck({suction[0]:.1f},{suction[1]:.1f}) "
              f"-> place({target_suction[0]:.1f},{target_suction[1]:.1f}) "
              f"rot{delta_deg:+.1f}deg")

    return results
