#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solver.py — A路模板匹配求解器（嵌入式精简版）

从 saiti/vision/solver.py 精简适配：
  - 保留 PieceTemplate 数据结构、模板 I/O、identify_piece、register_piece、solve_template
  - 移除 dataclass 依赖
  - 移除 B路 相关代码（B路由 b_solver.py 单独处理）
"""

from __future__ import annotations

import json
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from detection import Piece, sort_vertices_cyclic
from config import Config, get_config

logger = logging.getLogger("solver")


# =============================================================================
# 数据结构
# =============================================================================

class PieceTemplate:
    """碎片模板（赛前录制）"""

    def __init__(
        self,
        template_id: int,
        label: str = "",
        num_sides: int = 0,
        area_mm2: float = 0.0,
        edge_lengths_mm: Optional[List[float]] = None,
        edge_lengths_sorted: Optional[List[float]] = None,
        edge_directions_deg: Optional[List[float]] = None,
        vertices_canonical: Optional[np.ndarray] = None,
        target_center_mm: Tuple[float, float] = (0.0, 0.0),
        target_angle_deg: float = 0.0,
        target_suction_mm: Tuple[float, float] = (0.0, 0.0),
    ):
        self.template_id = template_id
        self.label = label
        self.num_sides = num_sides
        self.area_mm2 = area_mm2
        self.edge_lengths_mm = edge_lengths_mm or []
        self.edge_lengths_sorted = edge_lengths_sorted or []
        self.edge_directions_deg = edge_directions_deg or []
        self.vertices_canonical = (
            vertices_canonical if vertices_canonical is not None
            else np.array([], dtype=np.float32)
        )
        self.target_center_mm = target_center_mm
        self.target_angle_deg = target_angle_deg
        self.target_suction_mm = target_suction_mm

    def __repr__(self) -> str:
        return (f"PieceTemplate(T{self.template_id}[{self.label}], "
                f"{self.num_sides}sides, {self.area_mm2:.0f}mm²)")


class PlacementResult:
    """单片碎片的机器人摆放指令"""

    def __init__(
        self,
        piece_index: int,
        template_id: int = -1,
        suction_point_mm: Tuple[float, float] = (0.0, 0.0),
        target_center_mm: Tuple[float, float] = (0.0, 0.0),
        delta_rotation_deg: float = 0.0,
        match_rmse_mm: float = 0.0,
        confidence: float = 0.0,
        method: str = "template",
        target_vertices_mm: Optional[np.ndarray] = None,
    ):
        self.piece_index = piece_index
        self.template_id = template_id
        self.suction_point_mm = suction_point_mm
        self.target_center_mm = target_center_mm
        self.delta_rotation_deg = delta_rotation_deg
        self.match_rmse_mm = match_rmse_mm
        self.confidence = confidence
        self.method = method
        self.target_vertices_mm = target_vertices_mm

    def __repr__(self) -> str:
        return (f"Placement(P{self.piece_index}→T{self.template_id}, "
                f"suck=({self.suction_point_mm[0]:.1f},{self.suction_point_mm[1]:.1f}), "
                f"place=({self.target_center_mm[0]:.1f},{self.target_center_mm[1]:.1f}), "
                f"rot={self.delta_rotation_deg:+.1f}°, conf={self.confidence:.2f})")


# =============================================================================
# 模板 I/O
# =============================================================================

def save_templates(templates: List[PieceTemplate], path: Path) -> None:
    """保存模板到 JSON 文件。"""
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
            "target_suction_mm":   list(t.target_suction_mm),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("已保存 %d 个模板 → %s", len(templates), path)


def load_templates(path: Path) -> List[PieceTemplate]:
    """从 JSON 文件加载模板。"""
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
            target_suction_mm=tuple(d.get("target_suction_mm", d["target_center_mm"])),
        ))
    logger.info("已加载 %d 个模板 ← %s", len(out), path)
    return out


def build_template_from_piece(
    piece: Piece,
    template_id: int,
    label: str = "",
) -> PieceTemplate:
    """从已检测的碎片（处于目标摆放位姿）构建模板。"""
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
    target_suction = piece.suction_point_mm if piece.suction_point_mm is not None else target_center

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
        target_suction_mm=target_suction,
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
    """找最优循环偏移 k。"""
    if len(piece_edges) != len(tmpl_edges):
        return 0, 1e9
    best_k, best_rmse = 0, 1e9
    for k in range(len(piece_edges)):
        r = _cyclic_rmse(piece_edges, tmpl_edges, k)
        if r < best_rmse:
            best_rmse, best_k = r, k
    return best_k, best_rmse


def _rigid_transform_2d(src: np.ndarray, dst: np.ndarray) -> Tuple[float, float, float]:
    """SVD 求 2D 刚体变换。"""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    H_mat = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H_mat)
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
    """按排序边长 + 面积匹配最近模板。"""
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
    """求当前碎片到目标位姿的旋转差。"""
    n = piece.num_sides
    if n == 0:
        return 0.0, 999.0

    offset, rmse = _best_cyclic_offset(
        list(piece.edge_lengths_mm),
        list(template.edge_lengths_mm),
    )

    tmpl_dir0 = template.edge_directions_deg[0]
    piece_dirK = piece.edge_directions_deg[offset % n]
    rot_diff = (piece_dirK - tmpl_dir0 + 180.0) % 360.0 - 180.0
    delta = (-rot_diff + 180.0) % 360.0 - 180.0

    # SVD 精化
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
            pass

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

        # 目标吸取点：模板录制时已经记录绝对 mm 坐标，直接使用
        ts = tmpl.target_suction_mm
        target_pos = ts if (ts[0] != 0.0 or ts[1] != 0.0) else tmpl.target_center_mm

        # 从模板规范化顶点还原目标多边形（用于可视化预览）
        # 注意：vertices_canonical 录制时围绕质心归一化，必须加回 target_center_mm
        scale = math.sqrt(max(tmpl.area_mm2, 1.0))
        target_verts = (tmpl.vertices_canonical.astype(np.float64) * scale
                        + np.array(tmpl.target_center_mm, dtype=np.float64))

        result = PlacementResult(
            piece_index=piece.index,
            template_id=tmpl.template_id,
            suction_point_mm=suction,
            target_center_mm=target_pos,
            delta_rotation_deg=delta,
            match_rmse_mm=rmse,
            confidence=conf,
            method="template",
            target_vertices_mm=target_verts,
        )
        results.append(result)
        logger.info(
            "P%d[%ds] → T%d[%s] conf=%.2f rmse=%.1fmm | "
            "suck(%.1f,%.1f)→place(%.1f,%.1f) rot%+.1f°",
            piece.index, piece.num_sides,
            tmpl.template_id, tmpl.label, conf, rmse,
            suction[0], suction[1],
            target_pos[0], target_pos[1], delta,
        )

    return results
