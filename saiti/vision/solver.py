#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vision/solver.py — 拼接求解器

A路（自备碎片）：
  赛前把 4 片放到目标位姿，录制模板（边长序列 + 面积 + 方向 + 目标坐标）。
  比赛时排序边长 + 面积识别碎片编号，循环顶点对齐 + SVD 求旋转差，输出吸取/放置指令。

B路（现场白色碎片）：
  推断矩形尺寸 → 切割边配对 → 回溯搜索摆放（骨架，待实现）。
"""

from __future__ import annotations

import json
import math
import logging
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from vision.detection import Piece, sort_vertices_cyclic
from vision.config import Config, get_config

logger = logging.getLogger("solver")


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class PieceTemplate:
    """碎片模板（赛前录制）"""
    template_id: int
    label: str                            # 人类可读标签，如 "top_left"

    num_sides: int
    area_mm2: float
    edge_lengths_mm: List[float]          # 原始顺序（循环，用于循环对齐）
    edge_lengths_sorted: List[float]      # 排序后（用于快速识别，旋转无关）
    edge_directions_deg: List[float]      # 各边方向（建模时记录，用于求旋转差）

    vertices_canonical: np.ndarray        # 归一化顶点 (N,2)：重心为原点，scale = sqrt(area_mm2)
    target_center_mm: Tuple[float, float] # 目标放置质心（纸面 mm 坐标）
    target_angle_deg: float               # 目标旋转角（= 建模时 piece.angle_precise_deg）


@dataclass
class PlacementResult:
    """单片碎片的机器人摆放指令"""
    piece_index: int
    template_id: int                        # B路填 -1

    suction_point_mm: Tuple[float, float]   # 当前吸取点 → 下位机 pick 坐标
    target_center_mm: Tuple[float, float]   # 目标放置位置 → 下位机 place 坐标
    delta_rotation_deg: float               # 需要旋转的角度，顺时针为正

    match_rmse_mm: float = 0.0             # 边长匹配残差
    confidence: float = 0.0
    method: str = "template"               # "template" | "geometric"
    target_vertices_mm: Optional[np.ndarray] = None  # 精确目标顶点（B路填充）

    def __repr__(self) -> str:
        return (f"Placement(P{self.piece_index}→T{self.template_id}, "
                f"suck=({self.suction_point_mm[0]:.1f},{self.suction_point_mm[1]:.1f}), "
                f"place=({self.target_center_mm[0]:.1f},{self.target_center_mm[1]:.1f}), "
                f"rot={self.delta_rotation_deg:+.1f}°, conf={self.confidence:.2f})")


# =============================================================================
# 模板 I/O
# =============================================================================

def save_templates(templates: List[PieceTemplate], path: Path) -> None:
    data = []
    for t in templates:
        data.append({
            "template_id":         t.template_id,
            "label":               t.label,
            "num_sides":           t.num_sides,
            "area_mm2":            t.area_mm2,
            "edge_lengths_mm":     t.edge_lengths_mm,
            "edge_lengths_sorted": t.edge_lengths_sorted,
            "edge_directions_deg": t.edge_directions_deg,
            "vertices_canonical":  t.vertices_canonical.tolist(),
            "target_center_mm":    list(t.target_center_mm),
            "target_angle_deg":    t.target_angle_deg,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("已保存 %d 个模板 → %s", len(templates), path)


def load_templates(path: Path) -> List[PieceTemplate]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for d in data:
        out.append(PieceTemplate(
            template_id=d["template_id"],
            label=d.get("label", ""),
            num_sides=d["num_sides"],
            area_mm2=d["area_mm2"],
            edge_lengths_mm=d["edge_lengths_mm"],
            edge_lengths_sorted=d["edge_lengths_sorted"],
            edge_directions_deg=d["edge_directions_deg"],
            vertices_canonical=np.array(d["vertices_canonical"], dtype=np.float32),
            target_center_mm=tuple(d["target_center_mm"]),
            target_angle_deg=d["target_angle_deg"],
        ))
    logger.info("已加载 %d 个模板 ← %s", len(out), path)
    return out


def build_template_from_piece(
    piece: Piece,
    template_id: int,
    label: str = "",
) -> PieceTemplate:
    """
    从已检测的碎片（处于目标摆放位姿）构建模板。

    调用时机：赛前把碎片摆到拼图区正确位置并标定，让 detect_pieces 跑一帧。
    当前位置 = 目标位置，当前角度 = 目标角度。
    """
    # 优先用 mm 顶点，与 register_piece 的 SVD 坐标系一致
    if piece.vertices_mm is not None and len(piece.vertices_mm) >= 3:
        verts = sort_vertices_cyclic(piece.vertices_mm)
    else:
        verts = sort_vertices_cyclic(piece.vertices_px)
    center = verts.mean(axis=0)
    scale = math.sqrt(max(piece.area_mm2, 1.0))
    verts_norm = ((verts - center) / scale).astype(np.float32)

    target_center = piece.center_mm if piece.center_mm is not None else (0.0, 0.0)
    target_angle = (piece.angle_precise_deg if piece.angle_precise_set
                    else piece.angle_obb_deg)

    return PieceTemplate(
        template_id=template_id,
        label=label,
        num_sides=piece.num_sides,
        area_mm2=piece.area_mm2,
        edge_lengths_mm=list(piece.edge_lengths_mm),
        edge_lengths_sorted=sorted(piece.edge_lengths_mm),
        edge_directions_deg=list(piece.edge_directions_deg),
        vertices_canonical=verts_norm,
        target_center_mm=target_center,
        target_angle_deg=target_angle,
    )


# =============================================================================
# A路 — 工具函数
# =============================================================================

def _cyclic_rmse(piece_edges: List[float], tmpl_edges: List[float], offset: int) -> float:
    n = len(tmpl_edges)
    sq = sum((piece_edges[(offset + i) % len(piece_edges)] - tmpl_edges[i]) ** 2
             for i in range(n))
    return math.sqrt(sq / max(n, 1))


def _best_cyclic_offset(
    piece_edges: List[float],
    tmpl_edges: List[float],
) -> Tuple[int, float]:
    """找最优循环偏移 k，使 piece_edges[k:] ≈ tmpl_edges，返回 (k, rmse_mm)。"""
    if len(piece_edges) != len(tmpl_edges):
        return 0, 1e9
    best_k, best_rmse = 0, 1e9
    for k in range(len(piece_edges)):
        r = _cyclic_rmse(piece_edges, tmpl_edges, k)
        if r < best_rmse:
            best_rmse, best_k = r, k
    return best_k, best_rmse


def _rigid_transform_2d(src: np.ndarray, dst: np.ndarray) -> Tuple[float, float, float]:
    """SVD 求 2D 刚体变换：dst ≈ R(angle)*src + t，返回 (tx, ty, angle_deg)。"""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = cd - R @ cs
    return float(t[0]), float(t[1]), math.degrees(math.atan2(R[1, 0], R[0, 0]))


# =============================================================================
# A路 — 识别
# =============================================================================

def identify_piece(
    piece: Piece,
    templates: List[PieceTemplate],
    area_tol_ratio: float = 0.20,
    edge_tol_mm: float = 6.0,
) -> Tuple[Optional[PieceTemplate], float]:
    """
    按排序边长 + 面积匹配最近模板。

    主判据：排序边长序列 L2 距离（旋转无关）
    副判据：面积比差异，权重 15 mm/ratio
    硬过滤：边数必须相同

    Returns:
        (best_template, confidence)  confidence 0~1
    """
    if not templates:
        return None, 0.0

    piece_sorted = sorted(piece.edge_lengths_mm)
    best_tmpl, best_score = None, 1e9

    for tmpl in templates:
        if tmpl.num_sides != piece.num_sides:
            continue
        if len(tmpl.edge_lengths_sorted) != len(piece_sorted):
            continue

        area_diff = abs(piece.area_mm2 - tmpl.area_mm2) / max(tmpl.area_mm2, 1.0)
        if area_diff > area_tol_ratio:
            continue

        n = len(piece_sorted)
        edge_rmse = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(piece_sorted, tmpl.edge_lengths_sorted)) / max(n, 1)
        )
        score = edge_rmse + area_diff * 15.0

        if score < best_score:
            best_score, best_tmpl = score, tmpl

    if best_tmpl is None:
        return None, 0.0

    conf = max(0.0, min(1.0, 1.0 - best_score / edge_tol_mm))
    return best_tmpl, conf


# =============================================================================
# A路 — 配准
# =============================================================================

def register_piece(
    piece: Piece,
    template: PieceTemplate,
    px_per_mm: float = 4.0,
) -> Tuple[float, float]:
    """
    求当前碎片到目标位姿的旋转差。

    步骤：
      1. 循环对齐边长序列 → 顶点对应关系
      2. 从对应边方向差得到旋转初值
      3. 若有 mm 顶点坐标，用 SVD 精化

    Returns:
        (delta_rotation_deg, match_rmse_mm)
        delta_rotation_deg > 0 → 顺时针旋转
    """
    n = piece.num_sides
    if n == 0:
        return 0.0, 999.0

    offset, rmse = _best_cyclic_offset(
        list(piece.edge_lengths_mm),
        list(template.edge_lengths_mm),
    )

    # 方向差法：模板 edge[0] 方向 vs 碎片 edge[offset] 方向
    tmpl_dir0 = template.edge_directions_deg[0]
    piece_dirK = piece.edge_directions_deg[offset % n]
    rot_diff = (piece_dirK - tmpl_dir0 + 180.0) % 360.0 - 180.0
    delta = (-rot_diff + 180.0) % 360.0 - 180.0

    # SVD 精化（需要 mm 顶点坐标）
    # canonical 用 mm 顶点建立，SVD 直接给出从当前姿态到目标姿态的旋转角
    if piece.vertices_mm is not None and len(piece.vertices_mm) == n:
        try:
            verts_sorted = sort_vertices_cyclic(piece.vertices_mm)
            src_pts = np.array([verts_sorted[(offset + i) % n] for i in range(n)],
                               dtype=np.float64)
            scale = math.sqrt(max(template.area_mm2, 1.0))
            dst_pts = template.vertices_canonical.astype(np.float64) * scale
            _, _, svd_angle = _rigid_transform_2d(src_pts, dst_pts)
            delta = (svd_angle + 180.0) % 360.0 - 180.0
        except Exception:
            pass  # 保持方向差法结果

    return delta, rmse


# =============================================================================
# A路 — 主函数
# =============================================================================

def solve_template(
    pieces: List[Piece],
    templates: List[PieceTemplate],
    cfg: Optional[Config] = None,
) -> List[PlacementResult]:
    """A路主函数：识别 + 配准 → 输出摆放指令列表。"""
    if cfg is None:
        cfg = get_config()

    results: List[PlacementResult] = []
    used_tids: set = set()

    for piece in pieces:
        tmpl, conf = identify_piece(piece, templates)

        if tmpl is None:
            logger.warning("P%d 无法匹配模板 (sides=%d area=%.0fmm²)",
                           piece.index, piece.num_sides, piece.area_mm2)
            continue

        if tmpl.template_id in used_tids:
            logger.warning("P%d 匹配到已使用的 T%d，可能误识别",
                           piece.index, tmpl.template_id)
        used_tids.add(tmpl.template_id)

        delta, rmse = register_piece(piece, tmpl, cfg.px_per_mm)
        suction = piece.suction_point_mm or piece.center_mm or (0.0, 0.0)

        result = PlacementResult(
            piece_index=piece.index,
            template_id=tmpl.template_id,
            suction_point_mm=suction,
            target_center_mm=tmpl.target_center_mm,
            delta_rotation_deg=delta,
            match_rmse_mm=rmse,
            confidence=conf,
            method="template",
        )
        results.append(result)
        logger.info(
            "P%d[%ds] → T%d[%s] conf=%.2f rmse=%.1fmm | "
            "suck(%.1f,%.1f)→place(%.1f,%.1f) rot%+.1f°",
            piece.index, piece.num_sides,
            tmpl.template_id, tmpl.label, conf, rmse,
            suction[0], suction[1],
            tmpl.target_center_mm[0], tmpl.target_center_mm[1], delta,
        )

    return results


# =============================================================================
# B路 — 参考 puzzle-vision-simulator 重写
# =============================================================================

# ── 齐次变换工具 ──────────────────────────────────────────────────────────────

def _rigid_h(angle: float, tx: float, ty: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, tx], [s, c, ty], [0., 0., 1.]])


def _apply_h(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    p = np.hstack([np.asarray(pts, float), np.ones((len(pts), 1))])
    return (H @ p.T).T[:, :2]


def _align_edge(src_a, src_b, dst_a, dst_b) -> np.ndarray:
    """刚体变换 H 使 src_a→dst_a 且 src_b→dst_b（等价于开源 align_edge）。"""
    u = np.asarray(src_b, float) - np.asarray(src_a, float)
    v = np.asarray(dst_b, float) - np.asarray(dst_a, float)
    angle = math.atan2(float(v[1]), float(v[0])) - math.atan2(float(u[1]), float(u[0]))
    H = _rigid_h(angle, 0., 0.)
    mapped = _apply_h(np.array([src_a], float), H)[0]
    H[:2, 2] = np.asarray(dst_a, float) - mapped
    return H


def _ensure_cw(verts: np.ndarray) -> np.ndarray:
    """保证顶点顺时针顺序（图像/mm 坐标系 y 向下）。"""
    n = len(verts)
    signed = sum(
        float(verts[i][0]) * float(verts[(i+1)%n][1])
        - float(verts[(i+1)%n][0]) * float(verts[i][1])
        for i in range(n)
    )
    return verts if signed > 0 else verts[::-1].copy()


# ── 候选边对 ──────────────────────────────────────────────────────────────────

def _b_candidates(verts_list: List[np.ndarray]) -> List[Tuple]:
    """
    跨碎片边对：相对长度差 < 12%，按差值排序，最多保留 80 条。
    返回 (rel_diff, i, ei, j, ej)。
    """
    cands: List[Tuple] = []
    for i in range(len(verts_list)):
        ni = len(verts_list[i])
        for j in range(i + 1, len(verts_list)):
            nj = len(verts_list[j])
            for ei in range(ni):
                la = float(np.linalg.norm(
                    verts_list[i][(ei+1)%ni].astype(float) - verts_list[i][ei].astype(float)))
                for ej in range(nj):
                    lb = float(np.linalg.norm(
                        verts_list[j][(ej+1)%nj].astype(float) - verts_list[j][ej].astype(float)))
                    if max(la, lb) < 1e-6:
                        continue
                    rel = abs(la - lb) / max(la, lb)
                    if rel < 0.12:
                        cands.append((rel, i, ei, j, ej))
    cands.sort()
    return cands[:80]


# ── 合法匹配组合枚举 ──────────────────────────────────────────────────────────

def _b_matching_sets(n: int, cands: List[Tuple]):
    """枚举 n-1 条候选构成连通生成树（每条边最多用一次，每片至少一条）。"""
    from itertools import combinations as _comb
    r = n - 1
    K = len(cands)
    for sel_idx in _comb(range(K), r):
        sel = [cands[k] for k in sel_idx]
        used: set = set()
        ok = True
        degree = [0] * n
        graph: List[set] = [set() for _ in range(n)]
        for _, i, ei, j, ej in sel:
            if (i, ei) in used or (j, ej) in used:
                ok = False
                break
            used.add((i, ei)); used.add((j, ej))
            degree[i] += 1; degree[j] += 1
            graph[i].add(j); graph[j].add(i)
        if not ok or any(d == 0 for d in degree):
            continue
        seen, stack = {0}, [0]
        while stack:
            cur = stack.pop()
            for nb in graph[cur]:
                if nb not in seen:
                    seen.add(nb); stack.append(nb)
        if len(seen) == n:
            yield sel


# ── 组装 + 评分 ───────────────────────────────────────────────────────────────

def _b_assemble(
    verts_list: List[np.ndarray],
    matches: List[Tuple],
    total_area_mm2: float,
) -> Tuple[float, Dict[int, np.ndarray]]:
    """
    BFS 传播刚体变换组装碎片，返回 (score, {idx: placed_verts_mm})。
    score 越小越好（参照开源评分函数）。
    """
    n = len(verts_list)
    adj: Dict[int, List[Tuple[int,int,int]]] = {i: [] for i in range(n)}
    for _, i, ei, j, ej in matches:
        adj[i].append((j, ei, ej))
        adj[j].append((i, ej, ei))

    transforms: List[Optional[np.ndarray]] = [None] * n
    transforms[0] = np.eye(3)
    stack = [0]
    closure_error = 0.0

    while stack:
        cur = stack.pop()
        v_cur = verts_list[cur]
        nc = len(v_cur)
        for nb, ei_cur, ei_nb in adj[cur]:
            ia = v_cur[ei_cur].astype(float)
            ib = v_cur[(ei_cur + 1) % nc].astype(float)
            # 世界坐标下当前片的切割边端点
            wa, wb = _apply_h(np.array([ia, ib]), transforms[cur])
            # 邻接片的切割边 ja→jb 与 wa→wb 方向相反 → 对齐到 wb→wa
            v_nb = verts_list[nb]
            nn = len(v_nb)
            ja = v_nb[ei_nb].astype(float)
            jb = v_nb[(ei_nb + 1) % nn].astype(float)
            proposed = _align_edge(ja, jb, wb, wa)

            if transforms[nb] is None:
                transforms[nb] = proposed
                stack.append(nb)
            else:
                # 闭环误差（多路径到达同一片时的不一致量）
                pp = _apply_h(v_nb.astype(float), proposed)
                pe = _apply_h(v_nb.astype(float), transforms[nb])
                closure_error += float(np.linalg.norm(pp - pe, axis=1).mean())

    placed = {i: _apply_h(verts_list[i].astype(float), transforms[i]) for i in range(n)}

    # ── 栅格化评分（单位 mm，1mm ≈ 1px）──────────────────────────────
    all_pts = np.vstack(list(placed.values())).astype(np.float32)
    mn, mx = all_pts.min(0), all_pts.max(0)
    shift = (-mn + 5).astype(float)
    W = max(int(mx[0] - mn[0]) + 10, 1)
    H_px = max(int(mx[1] - mn[1]) + 10, 1)

    acc = np.zeros((H_px, W), np.int32)
    for v in placed.values():
        m = np.zeros((H_px, W), np.uint8)
        cv2.fillPoly(m, [(v + shift).astype(np.int32)], 1)
        acc += m.astype(np.int32)

    overlap = float(np.count_nonzero(acc > 1))
    union = (acc > 0).astype(np.uint8)

    cnts, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 1e9, placed
    cnt = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    rw, rh = rect[1]
    rect_area = rw * rh
    union_area = float(np.count_nonzero(union))
    fill_error = max(0.0, rect_area - union_area)

    # 长宽比软约束（9×5=1.8 ~ 12×9=1.33，放宽到 1.1~2.5）
    aspect = max(rw, rh) / max(1.0, min(rw, rh))
    aspect_penalty = max(0.0, 1.1 - aspect, aspect - 2.5) * rect_area

    disconnected = sum(cv2.contourArea(c) for c in cnts) - cv2.contourArea(cnt)
    perim_error = abs(cv2.arcLength(cnt, True) - 2.0 * (rw + rh))
    match_cost = sum(m[0] for m in matches)

    score = (
        closure_error * 8
        + overlap * 12
        + fill_error * 8
        + abs(union_area - total_area_mm2) * 4
        + abs(rect_area - total_area_mm2) * 3
        + aspect_penalty * 80
        + disconnected * 20
        + perim_error * 25
        + match_cost * 5000
    )
    return score, placed



def solve_geometric(
    pieces: List[Piece],
    cfg: Optional[Config] = None,
    puzzle_origin_mm: Optional[Tuple[float, float]] = None,
    rectified_bgr=None,
    H_px2mm=None,
) -> List[PlacementResult]:
    """B路几何拼接 — 委托给 vision.b_solver（DFS回溯算法）。"""
    from vision.b_solver import solve_geometric as _b_solve
    return _b_solve(pieces, cfg, puzzle_origin_mm, rectified_bgr=rectified_bgr,
                    H_px2mm=H_px2mm)


# =============================================================================
# 统一入口
# =============================================================================

def solve(
    pieces: List[Piece],
    templates: Optional[List[PieceTemplate]] = None,
    cfg: Optional[Config] = None,
    mode: str = "auto",
) -> List[PlacementResult]:
    """
    拼接求解统一入口。

    mode:
        "auto"      — 有模板走 A路，无模板走 B路
        "template"  — 强制 A路
        "geometric" — 强制 B路
    """
    if cfg is None:
        cfg = get_config()
    if mode == "auto":
        mode = "template" if templates else "geometric"
    if mode == "template":
        if not templates:
            logger.error("A路需要模板，但 templates=None")
            return []
        return solve_template(pieces, templates, cfg)
    return solve_geometric(pieces, cfg, puzzle_origin_mm=None)
