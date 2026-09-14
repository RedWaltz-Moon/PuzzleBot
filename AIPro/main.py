#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — AIPro 主程序入口（A/B 路白纸 + C 路扑克）

功能：
  1. 加载标定 (calib_data/homography.json) — 像素→纸面 mm 坐标
  2. 打开摄像头 + UART 串口
  3. 等待 MCU READY 信号 (AA 00 mode BB)
     mode = 0x01 → A路（白纸几何拼接）
     mode = 0x02 → B路（白纸几何拼接）
     mode = 0x03 → C路（扑克牌几何拼接）
  4. 拍摄照片 → 检测碎片 → DFS 几何拼接 → 计算坐标
  5. 逐片发送坐标指令 (AA 01 ... BB)，每片等待 ACK (AA 02 BB)
  6. 循环直至全部发送完毕

用法：
    python main.py                          # 默认 A路
    python main.py -m B                     # B路白纸拼接
    python main.py --port /dev/ttyUSB0 --camera 1
    python main.py --simulate               # 模拟发送（A路）
    python main.py -m B --simulate          # 模拟发送（B路）
    python main.py --dry-run                # 仅视觉检测，不发串口
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# 将当前目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# saiti 检测模块路径（B路白纸 / C路扑克）
_SAITI_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'saiti')
_SAITI_SCRIPTS = os.path.join(_SAITI_ROOT, 'scripts')
for _p in (_SAITI_ROOT, _SAITI_SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import Config, get_config, set_config
from camera import Camera
from detection import (
    Piece, simplify_contour, _filter_short_edges, _compute_suction_point,
    analyze_piece_geometry, sort_vertices_cyclic,
    detect_pieces,
)
from solver import (
    load_templates, PieceTemplate, PlacementResult,
    solve_template,
)
from b_solver import solve_geometric, _ccw
try:
    from build_templates import detect_all_pieces as _detect_saiti
except ImportError:
    _detect_saiti = None
    logging.getLogger("main").warning("saiti 检测模块未找到，B/C路将回退到 AIPro 内置检测")

from protocol import (
    ProtocolUART, MODE_A, MODE_B, MODE_C,
    build_coord_frame, format_coord_frame,
)

# A4 尺寸 mm（可视化用）
_A4_W, _A4_H = 210.0, 297.0
_SCALE = 3.0  # 预览画布每 mm 像素数

# 各片颜色（BGR）
_COLORS = [
    (255, 100,  50),
    ( 50, 200, 255),
    ( 80, 255, 120),
    (200,  80, 255),
    (255, 220,  50),
    ( 50, 120, 255),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# =============================================================================
# 标定加载
# =============================================================================

def load_homography(path: Path) -> Optional[np.ndarray]:
    """从 calibrate_aruco.py 生成的 JSON 文件读取单应矩阵 H（像素→mm）。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        H = np.array(data["H"], dtype=np.float64)
        logger.info("已加载标定: %s", path)
        return H
    except Exception as e:
        logger.warning("读取标定文件失败: %s", e)
        return None


# =============================================================================
# 可视化（来自 test_b_live.py，保存到 debug_output/ 供无头环境调试）
# =============================================================================

def draw_detected(image: np.ndarray, pieces: list) -> np.ndarray:
    """在图像上叠加检测到的碎片轮廓、中心、吸取点。"""
    vis = image.copy()
    for p in pieces:
        cx, cy = int(p.center_px[0]), int(p.center_px[1])
        cv2.drawContours(vis, [p.contour.astype(np.int32)], -1, (0, 200, 255), 2)
        cv2.circle(vis, (cx, cy), 6, (0, 200, 255), -1)
        if p.suction_point_px:
            sx, sy = int(p.suction_point_px[0]), int(p.suction_point_px[1])
            cv2.drawMarker(vis, (sx, sy), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        label = f"P{p.index} {p.num_sides}s {p.area_mm2:.0f}mm2"
        cv2.putText(vis, label, (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return vis


def _draw_assembly_preview(pieces: list, results: list) -> np.ndarray:
    """
    单列 A4 预览（来自 test_b_live.py）：
    上半区 = 碎片当前位置（灰色轮廓）
    分界线
    下半区 = 目标拼合（彩色填充，顺时针旋转90°显示）
    """
    cw = int(_A4_W * _SCALE)
    ch = int(_A4_H * _SCALE)
    canvas = np.full((ch, cw, 3), 240, dtype=np.uint8)

    def mm2px(x, y):
        return (int(x * _SCALE), int(y * _SCALE))

    # A4 边框 + 分界线
    cv2.rectangle(canvas, (0, 0), (cw - 1, ch - 1), (180, 180, 180), 2)
    sep_y = int(_A4_H / 2 * _SCALE)
    cv2.line(canvas, (0, sep_y), (cw - 1, sep_y), (100, 100, 255), 2)
    cv2.putText(canvas, "Current", (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Target", (4, sep_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)

    piece_map = {p.index: p for p in pieces}

    # 预先计算所有目标顶点的质心，并整体顺时针旋转 90°
    raw_verts = {r.piece_index: r.target_vertices_mm
                 for r in results if r.target_vertices_mm is not None}
    rotated_target = {}
    if raw_verts:
        centroid = np.vstack(list(raw_verts.values())).mean(0)
        for pid, v in raw_verts.items():
            d = v - centroid
            rotated_target[pid] = centroid + np.column_stack([d[:, 1], -d[:, 0]])

    for ci, r in enumerate(results):
        color = _COLORS[ci % len(_COLORS)]
        p = piece_map.get(r.piece_index)
        if p is None or p.vertices_mm is None:
            continue

        # 上半区：当前位置
        pts_src = np.array([mm2px(v[0], v[1]) for v in p.vertices_mm], dtype=np.int32)
        cv2.polylines(canvas, [pts_src], True, (140, 140, 140), 1)
        cx = int(pts_src[:, 0].mean()); cy = int(pts_src[:, 1].mean())
        cv2.putText(canvas, f"P{r.piece_index}", (cx + 3, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1, cv2.LINE_AA)

        # 下半区：目标（旋转后）
        verts = rotated_target.get(r.piece_index)
        if verts is None:
            continue
        pts_dst = np.array([mm2px(v[0], v[1]) for v in verts], dtype=np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts_dst], color)
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
        cv2.polylines(canvas, [pts_dst], True, color, 2)
        cx2 = int(pts_dst[:, 0].mean()); cy2 = int(pts_dst[:, 1].mean())
        label = f"P{r.piece_index} {r.delta_rotation_deg:+.0f}d"
        cv2.putText(canvas, label, (cx2 + 3, cy2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (cx2 + 3, cy2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    # 拼合外框 + 尺寸
    if rotated_target:
        all_rot = np.vstack(list(rotated_target.values()))
        pts_all = np.array([[mm2px(v[0], v[1])] for v in all_rot], dtype=np.int32)
        hull = cv2.convexHull(pts_all)
        cv2.polylines(canvas, [hull], True, (0, 0, 200), 2)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts_all)
        w_mm = max(rw, rh) / _SCALE; h_mm = min(rw, rh) / _SCALE
        cx_h = int(hull[:, 0, 0].mean()); cy_h = int(hull[:, 0, 1].mean())
        cv2.putText(canvas, f"{w_mm:.0f}x{h_mm:.0f}mm", (cx_h - 30, cy_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 180), 2, cv2.LINE_AA)

    return canvas


def _draw_results(frame: np.ndarray, pieces: list, results: list,
                  H_px2mm: Optional[np.ndarray]) -> np.ndarray:
    """在图像上叠加检测轮廓 + 目标位置标注（来自 test_b_live.py）。"""
    vis = draw_detected(frame, pieces)
    H_inv = None
    if H_px2mm is not None:
        try:
            H_inv = np.linalg.inv(H_px2mm)
        except Exception:
            pass

    for r in results:
        tx, ty = r.target_center_mm
        label = f"P{r.piece_index} rot{r.delta_rotation_deg:+.1f}deg"
        if H_inv is not None:
            p = H_inv @ np.array([tx, ty, 1.0])
            p /= p[2]
            px, py = int(p[0]), int(p[1])
            cv2.circle(vis, (px, py), 10, (0, 60, 255), -1)
            cv2.putText(vis, label, (px + 12, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, label, (px + 12, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(vis, f"target({tx:.0f},{ty:.0f})",
                        (10, 60 + results.index(r) * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
    return vis


# =============================================================================
# 碎片检测（A路/B路共用 — 与 build_templates.py 完全一致）
# =============================================================================

def detect_all_pieces(
    frame: np.ndarray,
    cfg: Config,
    H_px2mm: Optional[np.ndarray],
    mode: str = "A",
) -> list:
    """
    直接在相机原始帧上检测碎片。
    策略：检测绿色纸面 → 纸面内白色HSV分割 → 轮廓 → Piece对象。

    Args:
        frame: BGR 图像
        cfg: 全局配置
        H_px2mm: 标定矩阵（像素→mm）
        mode: "A" 或 "B"，决定 piece_type 标记
    """
    fh, fw = frame.shape[:2]
    min_area_px = fh * fw * 0.001     # 0.1%
    max_area_px = fh * fw * 0.25      # 25%
    logger.info("面积阈值: %.0f ~ %.0f px²", min_area_px, max_area_px)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 步骤1: 找绿色纸面
    green_raw = cv2.inRange(
        hsv,
        np.array([cfg.paper_green_h_lo, cfg.paper_green_s_min, cfg.paper_green_v_min]),
        np.array([cfg.paper_green_h_hi, 255, 255]),
    )
    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    k_sml = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green_filled = cv2.morphologyEx(green_raw, cv2.MORPH_CLOSE, k_big, iterations=4)
    green_filled = cv2.morphologyEx(green_filled, cv2.MORPH_OPEN, k_sml, iterations=1)

    paper_contours, _ = cv2.findContours(green_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not paper_contours:
        logger.warning("未检测到绿色纸面！")
    paper_mask = np.zeros((fh, fw), dtype=np.uint8)
    if paper_contours:
        paper_cnt = max(paper_contours, key=cv2.contourArea)
        cv2.drawContours(paper_mask, [paper_cnt], -1, 255, cv2.FILLED)

    # 步骤2: 纸面内检测白色碎片 (HSV: 低饱和 + 高亮)
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 236]),
        np.array([179, 78, 255]),
    )
    white_in_paper = cv2.bitwise_and(white_mask, paper_mask)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    filled = cv2.morphologyEx(white_in_paper, cv2.MORPH_CLOSE, k3, iterations=2)

    # 调试输出
    debug_dir = Path("debug_output")
    debug_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(debug_dir / "01_frame.jpg"), frame)
    cv2.imwrite(str(debug_dir / "02_green_mask.jpg"), green_filled)
    cv2.imwrite(str(debug_dir / "03_white_mask.jpg"), white_in_paper)
    cv2.imwrite(str(debug_dir / "04_filled.jpg"), filled)

    # 步骤3: 提取轮廓
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.info("mask 中找到 %d 个原始轮廓", len(contours))

    pieces = []
    reject_area = 0
    reject_solidity = 0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not (min_area_px <= area <= max_area_px):
            reject_area += 1
            continue
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area / max(hull_area, 1.0)
        if solidity < cfg.piece_min_solidity:
            reject_solidity += 1
            continue
        m = cv2.moments(contour)
        if abs(m["m00"]) < 1e-9:
            continue

        center = (m["m10"] / m["m00"], m["m01"] / m["m00"])
        poly = simplify_contour(contour)

        # ── 合并极短边（simplify_contour 可能拆分真边，产生 < 4mm 假边）──
        _tiny_px = 4.0 * cfg.px_per_mm
        for _ in range(len(poly)):
            if len(poly) <= 3:
                break
            n = len(poly)
            _ls = np.array([float(np.linalg.norm(
                poly[(i + 1) % n].astype(float) - poly[i].astype(float)))
                for i in range(n)], dtype=np.float64)
            _j = int(np.argmin(_ls))
            if _ls[_j] >= _tiny_px:
                break
            # 合并进较短的邻居边
            _ja, _jb = _j, (_j + 1) % n
            _l_prev = _ls[(_j - 1) % n]
            _l_next = _ls[(_j + 1) % n]
            if _l_prev <= _l_next:
                poly = np.delete(poly, _ja, axis=0)
            else:
                poly = np.delete(poly, _jb, axis=0)

        suction_px = _compute_suction_point(contour)

        piece = Piece()
        piece.contour = contour
        piece.vertices_px = poly
        piece.center_px = center
        piece.area_px = area
        piece.solidity = solidity
        piece.suction_point_px = suction_px
        piece = analyze_piece_geometry(piece, cfg.px_per_mm)

        # mm 坐标 — 用标定 H 矩阵转换
        if H_px2mm is not None:
            def _p2m(x, y):
                p = np.array([x, y, 1.0], dtype=np.float64)
                q = H_px2mm @ p
                q /= q[2]
                return (float(q[0]), float(q[1]))
            piece.center_mm = _p2m(center[0], center[1])
            if poly is not None and len(poly) > 0:
                piece.vertices_mm = np.array(
                    [_p2m(float(v[0]), float(v[1])) for v in poly], dtype=np.float32)
            if suction_px is not None:
                piece.suction_point_mm = _p2m(*suction_px)
        else:
            piece.center_mm = (center[0] / cfg.px_per_mm, center[1] / cfg.px_per_mm)
            if poly is not None and len(poly) > 0:
                piece.vertices_mm = poly / cfg.px_per_mm
            if suction_px is not None:
                piece.suction_point_mm = (suction_px[0] / cfg.px_per_mm, suction_px[1] / cfg.px_per_mm)

        # 从 mm 顶点重算边长和面积
        if piece.vertices_mm is not None and len(piece.vertices_mm) >= 2:
            verts_mm = _ccw(piece.vertices_mm)  # 用 _ccw 替代 sort_vertices_cyclic，保证跨平台一致
            # A路：过滤mask噪声产生的假边（B路交给b_solver内部处理）
            if mode == "A":
                verts_mm = _filter_short_edges(verts_mm, cfg.min_piece_edge_mm)
            piece.vertices_mm = verts_mm
            n = len(verts_mm)
            piece.num_sides = n
            piece.edge_lengths_mm = []
            piece.edge_lengths_px = []
            for i in range(n):
                l_mm = float(np.linalg.norm(verts_mm[(i + 1) % n] - verts_mm[i]))
                piece.edge_lengths_mm.append(l_mm)
                piece.edge_lengths_px.append(l_mm * cfg.px_per_mm)
            x_mm = verts_mm[:, 0]
            y_mm = verts_mm[:, 1]
            piece.area_mm2 = abs(float(
                np.dot(x_mm, np.roll(y_mm, -1)) - np.dot(y_mm, np.roll(x_mm, -1))
            )) / 2.0

        # A/B路均为白纸碎片
        piece.piece_type = "white"
        pieces.append(piece)

    logger.info("过滤: 面积不符=%d, 凸度不足=%d", reject_area, reject_solidity)
    pieces.sort(key=lambda p: p.area_px, reverse=True)
    for i, p in enumerate(pieces):
        p.index = i + 1

    logger.info("检测到 %d 片碎片", len(pieces))
    for p in pieces:
        logger.info("  P%d: %d边, area=%.0fmm², center_mm=(%.1f,%.1f)",
                     p.index, p.num_sides, p.area_mm2,
                     p.center_mm[0] if p.center_mm else 0,
                     p.center_mm[1] if p.center_mm else 0)

    return pieces


# =============================================================================
# C路 扑克牌检测（相机帧直接检测，不走矫正管线）
# =============================================================================

def detect_card_pieces(
    frame: np.ndarray,
    cfg: Config,
    H_px2mm: Optional[np.ndarray],
    dbg: Optional[list] = None,
) -> list:
    """
    C路扑克牌检测：直接在相机帧上检测（不经过矫正），
    流程：绿纸mask → 纸面内四策略分割 → 轮廓提取 → 分类 → mm坐标。
    """
    from detection import (
        _segment_non_background, _segment_white_bright,
        _segment_color_hsv, _segment_cards,
        _compute_suction_point, analyze_piece_geometry, simplify_contour,
    )

    fh, fw = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ── 1. 绿色纸面 mask ──
    green_raw = cv2.inRange(
        hsv,
        np.array([cfg.paper_green_h_lo, cfg.paper_green_s_min, cfg.paper_green_v_min]),
        np.array([cfg.paper_green_h_hi, 255, 255]),
    )
    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    k_sml = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green_filled = cv2.morphologyEx(green_raw, cv2.MORPH_CLOSE, k_big, iterations=4)
    green_filled = cv2.morphologyEx(green_filled, cv2.MORPH_OPEN, k_sml, iterations=1)

    # 边界腐蚀：断开桌面/背景连通域
    green_filled[:65, :] = 0
    green_filled[-60:, :] = 0
    green_filled[:, :230] = 0
    green_filled[:, -205:] = 0

    paper_contours, _ = cv2.findContours(green_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paper_mask = np.zeros((fh, fw), dtype=np.uint8)
    if paper_contours:
        paper_cnt = max(paper_contours, key=cv2.contourArea)
        cv2.drawContours(paper_mask, [paper_cnt], -1, 255, cv2.FILLED)
    else:
        logger.warning("C路: 未检测到绿色纸面")
        return []

    # ── 2. 四通道融合（与 saiti PC 一致）──
    mask_bg = _segment_non_background(frame, cfg)   # 绿底差分（主通道）

    # 其余三通道在"去绿帧"上运行：将绿纸区域置黑，
    # 避免绿纸高饱和度被 _segment_cards 的饱和度通道误判为卡面
    frame_no_green = frame.copy()
    frame_no_green[mask_bg == 0] = 0      # 绿纸背景置黑
    frame_no_green[paper_mask == 0] = 0   # 纸面以外（桌面/背景）置黑，防止误检

    mask_color = _segment_color_hsv(frame_no_green, cfg)     # 高饱和彩色
    mask_card  = _segment_cards(frame_no_green, cfg)         # 扑克纹理/梯度

    mask_or = cv2.bitwise_or(mask_bg,    mask_color)
    mask_or = cv2.bitwise_or(mask_or,    mask_card)
    mask_merged = cv2.bitwise_and(mask_or, paper_mask)

    # 轮廓填充
    contours_fill, _ = cv2.findContours(mask_merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros((fh, fw), dtype=np.uint8)
    for c in contours_fill:
        area = float(cv2.contourArea(c))
        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / max(hull_area, 1.0)
        if area < 100 and solidity < 0.5:
            continue
        cv2.drawContours(mask_filled, [c], -1, 255, -1)

    # 形态学清理
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_filled = cv2.morphologyEx(mask_filled, cv2.MORPH_CLOSE, k_close)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_filled = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN, k_open)

    if dbg is not None:
        dbg.append(mask_filled.copy())

    # ── 3. 轮廓提取 + 筛选 ──
    contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = cfg.piece_min_area_mm2 * cfg.px_per_mm * cfg.px_per_mm
    max_area = cfg.piece_max_area_mm2 * cfg.px_per_mm * cfg.px_per_mm

    pieces = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not (min_area <= area <= max_area):
            continue
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area / max(hull_area, 1.0)
        if solidity < cfg.piece_min_solidity:
            continue
        m = cv2.moments(contour)
        if abs(m["m00"]) < 1e-9:
            continue

        center = (m["m10"] / m["m00"], m["m01"] / m["m00"])
        piece_type = "card"
        # 提取纹理
        x, y, bw, bh = cv2.boundingRect(contour)
        texture_bgr = frame[max(0, y):min(fh, y + bh),
                            max(0, x):min(fw, x + bw)].copy()
        piece_mask = np.zeros(texture_bgr.shape[:2], dtype=np.uint8)
        shifted = contour.copy()
        shifted[:, :, 0] -= x
        shifted[:, :, 1] -= y
        cv2.drawContours(piece_mask, [shifted], -1, 255, cv2.FILLED)
        texture_bgr[piece_mask == 0] = 0

        poly = simplify_contour(contour, max_vertices=6)

        # ── px 空间伪顶点净化（与 saiti PC detection.py 一致）──
        # 扑克牌印刷图案会产生短边和近共线假顶点，需在 px 空间清理
        # 否则多边形带入 DFS 后搜索空间暴增，无法完成合法拼接
        _art_px = getattr(cfg, 'card_artifact_min_mm', 13.0) * cfg.px_per_mm
        _turn_min = getattr(cfg, 'card_artifact_min_turn_deg', 20.0)
        for _ in range(len(poly)):
            if len(poly) <= 3:
                break
            n = len(poly)
            _ls = np.array([float(np.linalg.norm(
                poly[(i + 1) % n].astype(float) - poly[i].astype(float)))
                for i in range(n)], dtype=np.float64)
            _j = int(np.argmin(_ls))
            if _ls[_j] < _art_px:
                _ja, _jb = _j, (_j + 1) % n

                def _turn(idx):
                    _p0 = poly[(idx - 1) % n].astype(float)
                    _p1 = poly[idx].astype(float)
                    _p2 = poly[(idx + 1) % n].astype(float)
                    _v1 = _p0 - _p1
                    _v2 = _p2 - _p1
                    _l1 = float(np.linalg.norm(_v1))
                    _l2 = float(np.linalg.norm(_v2))
                    if _l1 < 1e-6 or _l2 < 1e-6:
                        return 0.0
                    return 180.0 - math.degrees(math.acos(
                        float(np.clip(np.dot(_v1, _v2) / (_l1 * _l2), -1.0, 1.0))))

                _del = _ja if _turn(_ja) <= _turn(_jb) else _jb
                poly = np.delete(poly, _del, axis=0)
                continue
            # 近共线检测
            _mt, _mi = 360.0, -1
            for i in range(n):
                _p0 = poly[(i - 1) % n].astype(float)
                _p1 = poly[i].astype(float)
                _p2 = poly[(i + 1) % n].astype(float)
                _v1 = _p0 - _p1
                _l1 = float(np.linalg.norm(_v1))
                _v2 = _p2 - _p1
                _l2 = float(np.linalg.norm(_v2))
                if _l1 < 1e-6 or _l2 < 1e-6:
                    continue
                _cos = float(np.clip(np.dot(_v1, _v2) / (_l1 * _l2), -1.0, 1.0))
                _turn = 180.0 - math.degrees(math.acos(_cos))
                if _turn < _mt:
                    _mt, _mi = _turn, i
            if _mi >= 0 and _mt < _turn_min:
                poly = np.delete(poly, _mi, axis=0)
            else:
                break

        suction_px = _compute_suction_point(contour)

        piece = Piece()
        piece.contour = contour
        piece.vertices_px = poly
        piece.center_px = center
        piece.area_px = area
        piece.solidity = solidity
        piece.suction_point_px = suction_px
        piece.piece_type = piece_type
        piece.texture_bgr = texture_bgr
        piece.texture_origin_px = (x, y)

        piece = analyze_piece_geometry(piece, cfg.px_per_mm)
        pieces.append(piece)

    pieces.sort(key=lambda p: p.area_px, reverse=True)
    for i, p in enumerate(pieces):
        p.index = i + 1

    # ── 4. mm 坐标（H_px2mm 直接映射，相机帧px → mm）──
    if H_px2mm is not None:
        for p in pieces:
            def _cam_to_mm(x, y):
                q = H_px2mm @ np.array([x, y, 1.0], dtype=np.float64)
                q /= q[2]
                return (float(q[0]), float(q[1]))

            p.center_mm = _cam_to_mm(p.center_px[0], p.center_px[1])
            if p.vertices_px is not None and len(p.vertices_px) > 0:
                p.vertices_mm = np.array(
                    [_cam_to_mm(float(v[0]), float(v[1])) for v in p.vertices_px],
                    dtype=np.float32,
                )
            if p.suction_point_px is not None:
                p.suction_point_mm = _cam_to_mm(*p.suction_point_px)

            # 从 mm 顶点重算边长和面积
            if p.vertices_mm is not None and len(p.vertices_mm) >= 2:
                verts_mm = _ccw(p.vertices_mm)
                p.vertices_mm = verts_mm
                n = len(verts_mm)
                p.num_sides = n
                p.edge_lengths_mm = []
                p.edge_lengths_px = []
                for i in range(n):
                    l_mm = float(np.linalg.norm(verts_mm[(i + 1) % n] - verts_mm[i]))
                    p.edge_lengths_mm.append(l_mm)
                    p.edge_lengths_px.append(l_mm * cfg.px_per_mm)
                x_mm_arr = verts_mm[:, 0]
                y_mm_arr = verts_mm[:, 1]
                p.area_mm2 = abs(float(
                    np.dot(x_mm_arr, np.roll(y_mm_arr, -1)) -
                    np.dot(y_mm_arr, np.roll(x_mm_arr, -1))
                )) / 2.0
    else:
        logger.warning("C路: 缺少标定矩阵，mm坐标使用 px_per_mm 近似")
        for p in pieces:
            p.vertices_mm = p.vertices_px / cfg.px_per_mm if p.vertices_px is not None else None
            p.suction_point_mm = (p.suction_point_px[0] / cfg.px_per_mm,
                                 p.suction_point_px[1] / cfg.px_per_mm) if p.suction_point_px else None
            p.center_mm = (p.center_px[0] / cfg.px_per_mm, p.center_px[1] / cfg.px_per_mm)

    logger.info("C路: 检测到 %d 片碎片", len(pieces))
    for p in pieces:
        logger.info("  P%d: type=%s %d边 area=%.0fmm²", p.index, p.piece_type, p.num_sides, p.area_mm2)

    return pieces


# =============================================================================
# saiti 碎片适配（物理坐标统一）
# =============================================================================

def _adapt_saiti_pieces(pieces: list, cfg: Config, piece_type: str = None) -> None:
    """
    将 saiti detect_all_pieces 返回的 Piece 对象适配为 b_solver 所需格式。

    saiti 用 sort_vertices_cyclic 排序顶点，方向不保证 CCW；
    AIPro b_solver 依赖 CCW 顺序做 DFS 边匹配。
    此处补一次 _ccw，并同步重算 edge_lengths_mm/px 和 num_sides。
    坐标系无需转换：saiti 同样使用 H_px2mm @ [x,y,1] 转换，与 AIPro 完全一致。
    """
    for p in pieces:
        if p.vertices_mm is not None and len(p.vertices_mm) >= 2:
            verts = _ccw(p.vertices_mm)
            p.vertices_mm = verts
            n = len(verts)
            p.num_sides = n
            p.edge_lengths_mm = [
                float(np.linalg.norm(verts[(i + 1) % n] - verts[i]))
                for i in range(n)
            ]
            p.edge_lengths_px = [l * cfg.px_per_mm for l in p.edge_lengths_mm]
        if piece_type is not None:
            p.piece_type = piece_type


# =============================================================================
# 模拟数据（测试用）
# =============================================================================

def simulate_results(mode: str = "A") -> list:
    """生成模拟的摆放指令，用于无硬件测试通信协议。"""
    if mode == "C":
        return [
            PlacementResult(
                piece_index=1, template_id=-1,
                suction_point_mm=(45.0, 80.0),
                target_center_mm=(55.0, 210.0),
                delta_rotation_deg=15.0,
                confidence=0.95, method="geometric",
            ),
            PlacementResult(
                piece_index=2, template_id=-1,
                suction_point_mm=(120.0, 75.0),
                target_center_mm=(105.0, 230.0),
                delta_rotation_deg=-30.0,
                confidence=0.92, method="geometric",
            ),
            PlacementResult(
                piece_index=3, template_id=-1,
                suction_point_mm=(60.0, 130.0),
                target_center_mm=(55.0, 270.0),
                delta_rotation_deg=45.0,
                confidence=0.88, method="geometric",
            ),
            PlacementResult(
                piece_index=4, template_id=-1,
                suction_point_mm=(150.0, 120.0),
                target_center_mm=(155.0, 260.0),
                delta_rotation_deg=-10.0,
                confidence=0.91, method="geometric",
            ),
        ]
    elif mode == "B":
        return [
            PlacementResult(
                piece_index=1, template_id=-1,
                suction_point_mm=(45.0, 80.0),
                target_center_mm=(55.0, 210.0),
                delta_rotation_deg=15.0,
                confidence=0.95, method="geometric",
            ),
            PlacementResult(
                piece_index=2, template_id=-1,
                suction_point_mm=(120.0, 75.0),
                target_center_mm=(105.0, 230.0),
                delta_rotation_deg=-30.0,
                confidence=0.92, method="geometric",
            ),
            PlacementResult(
                piece_index=3, template_id=-1,
                suction_point_mm=(60.0, 130.0),
                target_center_mm=(55.0, 270.0),
                delta_rotation_deg=45.0,
                confidence=0.88, method="geometric",
            ),
            PlacementResult(
                piece_index=4, template_id=-1,
                suction_point_mm=(150.0, 120.0),
                target_center_mm=(155.0, 260.0),
                delta_rotation_deg=-10.0,
                confidence=0.91, method="geometric",
            ),
        ]
    else:
        return [
            PlacementResult(
                piece_index=1, template_id=0,
                suction_point_mm=(45.0, 80.0),
                target_center_mm=(55.0, 210.0),
                delta_rotation_deg=15.0,
                confidence=0.95, method="template",
            ),
            PlacementResult(
                piece_index=2, template_id=1,
                suction_point_mm=(120.0, 75.0),
                target_center_mm=(105.0, 230.0),
                delta_rotation_deg=-30.0,
                confidence=0.92, method="template",
            ),
            PlacementResult(
                piece_index=3, template_id=2,
                suction_point_mm=(60.0, 130.0),
                target_center_mm=(55.0, 270.0),
                delta_rotation_deg=45.0,
                confidence=0.88, method="template",
            ),
            PlacementResult(
                piece_index=4, template_id=3,
                suction_point_mm=(150.0, 120.0),
                target_center_mm=(155.0, 260.0),
                delta_rotation_deg=-10.0,
                confidence=0.91, method="template",
            ),
        ]


# =============================================================================
# 发送摆放序列
# =============================================================================

def send_placement_sequence(
    uart: ProtocolUART,
    placements: list,
) -> None:
    """
    逐片发送坐标指令 → 等待 ACK → 下一片 → 循环直至全部发送完毕。

    流程:
        上位机 → 发 AA 01 ... BB（坐标指令）
        MCU 执行 → pick-and-place 动作序列
        MCU → 发 AA 02 BB（完成确认）
        上位机 → 等待下一条指令
    """
    total = len(placements)
    logger.info("开始发送 %d 条坐标指令...", total)

    for i, r in enumerate(placements):
        logger.info("── 第 %d/%d 片: P%d ──", i + 1, total, r.piece_index)

        x_grab  = r.suction_point_mm[0]
        y_grab  = r.suction_point_mm[1]
        x_place = r.target_center_mm[0]
        y_place = r.target_center_mm[1]
        yaw     = r.delta_rotation_deg  # +顺时针

        logger.info(
            "  吸取: (%.1f, %.1f)mm → 放置: (%.1f, %.1f)mm → 旋转: %+.1f°",
            x_grab, y_grab, x_place, y_place, yaw,
        )

        uart.send_coord_and_wait_ack(
            x_grab_mm=x_grab,
            y_grab_mm=y_grab,
            x_place_mm=x_place,
            y_place_mm=y_place,
            yaw_deg=yaw,
            ack_timeout=60.0,
        )

        logger.info("  P%d 完成 [OK]", r.piece_index)

    logger.info("全部 %d 片发送完毕！", total)


# =============================================================================
# 命令行入口
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIPro 拼图装置上位机")
    parser.add_argument("--port", "-p", type=str, default=None,
                        help="UART 串口设备 (默认从config读取)")
    parser.add_argument("--baudrate", "-b", type=int, default=115200,
                        help="波特率 (默认 115200)")
    parser.add_argument("--camera", "-c", type=int, default=None,
                        help="摄像头索引 (默认从config读取)")
    parser.add_argument("--templates", "-t", type=str, default="calib_data/templates.json",
                        help="模板文件路径 (仅A路)")
    parser.add_argument("--calib", type=str, default="calib_data/homography.json",
                        help="标定文件路径")
    parser.add_argument("--simulate", "-s", action="store_true",
                        help="使用模拟坐标测试 UART (不打开摄像头)")
    parser.add_argument("--mode", "-m", choices=["A", "B", "C"], default=None,
                        help="路线: A/B/C (默认从MCU READY信号自动识别)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅视觉检测，不发送串口指令")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = get_config()

    # ---- 更新配置 ----
    if args.port is not None:
        cfg.uart_port = args.port
    if args.baudrate is not None:
        cfg.uart_baudrate = args.baudrate
    if args.camera is not None:
        cfg.camera_index = args.camera

    # ---- 标定 ----
    calib_path = Path(args.calib)
    H_px2mm = None
    if calib_path.exists():
        H_px2mm = load_homography(calib_path)
    elif not args.simulate:
        logger.warning("标定文件不存在: %s", calib_path)

    # ---- 模拟模式 ----
    if args.simulate:
        sim_mode = args.mode or "A"
        logger.info("=== %s路 模拟模式 ===", sim_mode)
        placements = simulate_results(sim_mode)
        first = placements[:1]
        for r in first:
            print(f"  {r}")

        uart = ProtocolUART(
            port=cfg.uart_port,
            baudrate=cfg.uart_baudrate,
            timeout=cfg.uart_timeout,
        )
        try:
            uart.open()
            for r in first:
                frame = build_coord_frame(
                    r.suction_point_mm[0], r.suction_point_mm[1],
                    r.target_center_mm[0], r.target_center_mm[1],
                    r.delta_rotation_deg,
                )
                print(f"  {format_coord_frame(frame)}")
                print(f"  帧: {frame.hex(' ').upper()}")
                uart.send_frame(frame)
                uart.wait_ack(timeout=60.0)
                print("  发送完成")
        finally:
            uart.close()
        return 0

    # ---- 真实模式 ----
    cam = Camera(
        index=cfg.camera_index,
        width=cfg.camera_width,
        height=cfg.camera_height,
        fps=cfg.camera_fps,
    )

    uart = ProtocolUART(
        port=cfg.uart_port,
        baudrate=cfg.uart_baudrate,
        timeout=cfg.uart_timeout,
    )

    try:
        cam.open()
        uart.open()

        round_no = 0
        while True:
            round_no += 1
            print("\n" + "=" * 55)
            print(f"  第 {round_no} 轮 — 等待 MCU READY...")
            print("=" * 55)

            # ---- 等待 READY 信号，由 MCU 决定 A路/B路 ----
            mcumode = uart.wait_ready(timeout=120.0)

            if args.mode is not None:
                expected = MODE_A if args.mode == "A" else (MODE_B if args.mode == "B" else MODE_C)
                if mcumode != expected:
                    logger.warning("CLI指定%s路(mode=%d)，但MCU返回mode=%d，以CLI为准",
                                   args.mode, expected, mcumode)
                mcumode = expected

            if mcumode == MODE_A:
                route = "A"
            elif mcumode == MODE_B:
                route = "B"
            elif mcumode == MODE_C:
                route = "C"
            else:
                logger.warning("未知MCU模式 0x%02X，跳过本轮", mcumode)
                continue
            route_name = {"A": "A路(白纸几何)", "B": "B路(白纸几何)", "C": "C路(扑克牌)"}[route]
            logger.info("MCU模式: %s (mode=0x%02X)", route_name, mcumode)

            # ---- 拍摄 + 检测 + 求解（重试循环，最多10次）----
            debug_dir = Path("debug_output")
            debug_dir.mkdir(parents=True, exist_ok=True)
            placements = []
            for attempt in range(1, 11):
                logger.info("拍摄照片 (尝试 %d/10)...", attempt)
                time.sleep(0.5)
                frame = cam.capture_averaged(n_frames=3)
                logger.info("照片: %dx%d", frame.shape[1], frame.shape[0])

                if route == "A":
                    # A路：AIPro 白纸检测 + 几何求解（替代模板匹配）
                    pieces = detect_all_pieces(frame, cfg, H_px2mm, mode="A")
                elif route == "B":
                    # B路：saiti 白纸检测
                    if _detect_saiti is not None:
                        pieces = _detect_saiti(frame, cfg, H_px2mm, scene='white')
                        _adapt_saiti_pieces(pieces, cfg, piece_type="white")
                    else:
                        pieces = detect_all_pieces(frame, cfg, H_px2mm, mode="B")
                else:
                    # C路：saiti 扑克检测
                    if _detect_saiti is not None:
                        pieces = _detect_saiti(frame, cfg, H_px2mm, scene='card')
                        _adapt_saiti_pieces(pieces, cfg)
                    else:
                        pieces = detect_card_pieces(frame, cfg, H_px2mm)

                det_vis = draw_detected(frame, pieces)

                if not pieces:
                    print(f"\n[FAIL] 尝试{attempt}: 未检测到碎片")
                    det_vis_fail = frame.copy()
                    cv2.putText(det_vis_fail, f"FAIL attempt {attempt}/5 - no pieces",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imwrite(str(debug_dir / f"05_detected_r{round_no}_a{attempt}.jpg"), det_vis_fail)
                    if attempt < 5:
                        print("  等待 3s 后重拍...")
                        time.sleep(3)
                    continue

                print(f"\n[检测] 找到 {len(pieces)} 片碎片")
                for p in pieces:
                    ls = [f"{l:.1f}" for l in (p.edge_lengths_mm or [])]
                    print(f"  P{p.index}: {p.num_sides}边  area={p.area_mm2:.0f}mm2  edges={ls}")
                    if p.center_mm:
                        print(f"         center_mm=({p.center_mm[0]:.1f},{p.center_mm[1]:.1f})")

                cv2.imwrite(str(debug_dir / f"05_detected_r{round_no}_a{attempt}.jpg"), det_vis)

                # ---- 拼接求解（三路统一 DFS 几何拼接）----
                if len(pieces) < 2:
                    print(f"[SKIP] 尝试{attempt}: 碎片不足 2 片")
                    if attempt < 5:
                        print("  等待 3s 后重拍...")
                        time.sleep(3)
                    continue
                if any(p.vertices_mm is None for p in pieces):
                    print("[SKIP] 缺少 mm 坐标，请先完成相机标定")
                    return 1

                print("\n[求解] DFS 拼接中...")
                placements = solve_geometric(pieces, cfg, H_px2mm=H_px2mm)

                if not placements:
                    print(f"[FAIL] 尝试{attempt}: 未找到合法拼接方案")
                    print("  可能原因：")
                    print("  1. 碎片顶点识别有误（边数 / 边长不准确）")
                    print("  2. mm 坐标转换偏差过大（重新标定）")
                    print("  3. 碎片尺寸超出范围（宽 82~128mm，高 42~98mm）")
                    if attempt < 5:
                        print("  等待 3s 后重拍...")
                        time.sleep(3)
                    continue

                # 成功 — 保存预览
                preview = _draw_assembly_preview(pieces, placements)
                cv2.imwrite(str(debug_dir / f"06_assembly_preview_r{round_no}_a{attempt}.jpg"), preview)
                res_vis = _draw_results(frame, pieces, placements, H_px2mm)
                cv2.imwrite(str(debug_dir / f"07_results_overlay_r{round_no}_a{attempt}.jpg"), res_vis)

                print(f"\n[结果] 成功！共 {len(placements)} 片放置指令：")
                print("-" * 60)
                for r in placements:
                    print(f"  P{r.piece_index}:")
                    print(f"    吸取点  ({r.suction_point_mm[0]:.1f}, {r.suction_point_mm[1]:.1f}) mm")
                    print(f"    放置中心({r.target_center_mm[0]:.1f}, {r.target_center_mm[1]:.1f}) mm")
                    print(f"    旋转角  {r.delta_rotation_deg:+.1f} deg")
                    print(f"    置信度  {r.confidence:.3f}")
                print("-" * 60)
                break  # 成功，跳出重试

            # 10次全失败 → 回到等待下次 MCU 输入
            if not placements:
                print(f"\n[WARN] 10 次重试均无结果，等待下次 MCU 指令")
                continue

            # ---- 打印指令表 ----
            print("\n" + "=" * 65)
            print(f"  摆放指令列表 ({route_name}  第{round_no}轮)")
            print("=" * 65)
            header = (f"{'序号':>4s}  {'吸取X':>8s}  {'吸取Y':>8s}  "
                      f"{'放置X':>8s}  {'放置Y':>8s}  {'旋转':>8s}  {'置信度':>6s}")
            print(header)
            print("-" * 65)
            for i, r in enumerate(placements):
                print(f"{i+1:>4d}  {r.suction_point_mm[0]:8.1f}  {r.suction_point_mm[1]:8.1f}  "
                      f"{r.target_center_mm[0]:8.1f}  {r.target_center_mm[1]:8.1f}  "
                      f"{r.delta_rotation_deg:8.1f}  {r.confidence:6.3f}")
            print("=" * 65)

            # ---- 逐片发送 ----
            if args.dry_run:
                logger.info("--dry-run 模式，不发送串口指令")
            else:
                send_placement_sequence(uart, placements)

            print(f"\n=== 第 {round_no} 轮完成，回到 READY 等待 ===")

    except TimeoutError as e:
        logger.error("通信超时: %s", e)
        return 1
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.exception("运行错误: %s", e)
        return 1
    finally:
        uart.close()
        cam.close()

    logger.info("程序正常退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
