#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/test_solver.py — A路识别测试

把碎片摆到取料区，按 SPACE 冻结，测试能否正确识别并输出摆放指令。

用法：
    python scripts/test_solver.py
    python scripts/test_solver.py -c 1
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from vision.config import get_config
from vision.detection import sort_vertices_cyclic, Piece
from vision.solver import load_templates, solve_template, solve_geometric, PlacementResult
from scripts.build_templates import detect_all_pieces, draw_detected, _load_homography

# 每片碎片的可视化颜色 (BGR)
PIECE_COLORS = [
    ( 30, 160, 255),  # 橙
    ( 40, 200,  60),  # 绿
    (220,  80,  50),  # 蓝
    (180,  50, 200),  # 紫
]


def _fit_to_panel(
    verts_list: List[np.ndarray],
    px0: int, py0: int, pw: int, ph: int,
    margin: int = 45,
) -> tuple:
    """把一组 mm 多边形适配到像素面板，返回 (scale, ox, oy)。"""
    all_pts = np.vstack(verts_list)
    mn = all_pts.min(axis=0)
    mx = all_pts.max(axis=0)
    dw = max(float(mx[0] - mn[0]), 1.0)
    dh = max(float(mx[1] - mn[1]), 1.0)
    scale = min((pw - 2 * margin) / dw, (ph - 2 * margin) / dh)
    ox = px0 + margin + ((pw - 2 * margin) - dw * scale) / 2.0 - mn[0] * scale
    oy = py0 + margin + ((ph - 2 * margin) - dh * scale) / 2.0 - mn[1] * scale
    return scale, ox, oy


def _mm_to_px(verts_mm: np.ndarray, scale: float, ox: float, oy: float) -> np.ndarray:
    return (verts_mm * scale + np.array([ox, oy])).astype(np.int32)


def _label(canvas, text: str, cx: int, cy: int, color, font_scale=0.55):
    cv2.putText(canvas, text, (cx - 14, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, (cx - 14, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)


def draw_assembly_result(
    pieces: List[Piece],
    results: List[PlacementResult],
    canvas_w: int = 1100,
    canvas_h: int = 620,
) -> np.ndarray:
    """
    拼接方案示意图。
      左面板：目标位置（拼图区）
      右面板：当前位置（取料区）
      中线：分界线
      箭头：取料区 → 拼图区，标注旋转角
    两个面板各自自动缩放，不依赖纸面 mm 尺寸。
    """
    TOP   = 42       # 标题行高度
    MID_W = 4        # 分界线宽度
    mid_x = canvas_w // 2

    canvas = np.full((canvas_h, canvas_w, 3), 230, dtype=np.uint8)
    # 左面板背景
    cv2.rectangle(canvas, (0, TOP), (mid_x, canvas_h), (228, 238, 255), -1)
    # 右面板背景
    cv2.rectangle(canvas, (mid_x, TOP), (canvas_w, canvas_h), (228, 255, 235), -1)
    # 顶栏
    cv2.rectangle(canvas, (0, 0), (canvas_w, TOP - 2), (50, 50, 50), -1)
    cv2.putText(canvas, "目标位置（拼图区）",
                (mid_x // 2 - 90, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "当前位置（取料区）",
                (mid_x + mid_x // 2 - 90, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 210), 2, cv2.LINE_AA)
    # 分界线
    cv2.rectangle(canvas, (mid_x - MID_W // 2, 0),
                  (mid_x + MID_W // 2, canvas_h), (60, 60, 60), -1)

    matched: Dict[int, PlacementResult] = {r.piece_index: r for r in results}

    # 收集当前顶点 & 目标顶点
    cur_verts: Dict[int, np.ndarray] = {}
    tgt_verts: Dict[int, np.ndarray] = {}

    for p in pieces:
        if p.vertices_mm is None:
            continue
        vc = sort_vertices_cyclic(p.vertices_mm)
        cur_verts[p.index] = vc

        r = matched.get(p.index)
        if r is None:
            continue
        delta_rad = math.radians(r.delta_rotation_deg)
        c_, s_ = math.cos(delta_rad), math.sin(delta_rad)
        R_ = np.array([[c_, -s_], [s_, c_]])
        # 用稠密轮廓做目标形状，旋转中心用 shape 自身质心保证放置点准确
        shape_mm = (p.contour_mm
                    if p.contour_mm is not None and len(p.contour_mm) > 3
                    else vc)
        ctr = shape_mm.mean(axis=0)
        vt = (R_ @ (shape_mm - ctr).T).T + np.array(r.target_center_mm)
        tgt_verts[p.index] = vt

    if not cur_verts:
        return canvas

    # 适配变换
    panel_h = canvas_h - TOP
    if tgt_verts:
        ts, tox, toy = _fit_to_panel(list(tgt_verts.values()),
                                     0, TOP, mid_x, panel_h)
    cs, cox, coy = _fit_to_panel(list(cur_verts.values()),
                                 mid_x, TOP, mid_x, panel_h)

    centers_tgt: Dict[int, tuple] = {}
    centers_cur: Dict[int, tuple] = {}
    piece_by_idx = {p.index: p for p in pieces}

    for pi, vc in cur_verts.items():
        color = PIECE_COLORS[(pi - 1) % len(PIECE_COLORS)]
        cur_piece = piece_by_idx.get(pi)

        # ── 右面板：当前位置（稠密轮廓 > 简化多边形）──────────────
        cnt_mm = cur_piece.contour_mm if cur_piece is not None else None
        draw_c = cnt_mm if (cnt_mm is not None and len(cnt_mm) > 3) else vc
        pts_c = _mm_to_px(draw_c, cs, cox, coy)
        ov = canvas.copy()
        cv2.fillPoly(ov, [pts_c], color)
        cv2.addWeighted(ov, 0.28, canvas, 0.72, 0, canvas)
        cv2.polylines(canvas, [pts_c], True, color, 2, cv2.LINE_AA)
        ccx, ccy = int(pts_c[:, 0].mean()), int(pts_c[:, 1].mean())
        centers_cur[pi] = (ccx, ccy)
        cv2.circle(canvas, (ccx, ccy), 5, color, -1, cv2.LINE_AA)
        _label(canvas, f"P{pi}", ccx, ccy, color)

        # ── 左面板：目标位置 ──────────────────────────────────────
        if pi not in tgt_verts:
            continue
        vt = tgt_verts[pi]
        pts_t = _mm_to_px(vt, ts, tox, toy)
        ov2 = canvas.copy()
        cv2.fillPoly(ov2, [pts_t], color)
        cv2.addWeighted(ov2, 0.38, canvas, 0.62, 0, canvas)
        cv2.polylines(canvas, [pts_t], True, color, 2, cv2.LINE_AA)
        tcx, tcy = int(pts_t[:, 0].mean()), int(pts_t[:, 1].mean())
        centers_tgt[pi] = (tcx, tcy)
        cv2.circle(canvas, (tcx, tcy), 5, color, -1, cv2.LINE_AA)
        _label(canvas, f"P{pi}", tcx, tcy, color)

    # ── 箭头：右 → 左（穿越分界线） ──────────────────────────────
    for pi in centers_cur:
        if pi not in centers_tgt:
            continue
        src = centers_cur[pi]
        dst = centers_tgt[pi]
        color = PIECE_COLORS[(pi - 1) % len(PIECE_COLORS)]
        r = matched[pi]

        cv2.arrowedLine(canvas, src, dst, (40, 40, 40), 4,
                        cv2.LINE_AA, tipLength=0.035)
        cv2.arrowedLine(canvas, src, dst, color, 2,
                        cv2.LINE_AA, tipLength=0.035)

        mx_ = (src[0] + dst[0]) // 2
        my_ = (src[1] + dst[1]) // 2 - 12
        rot_str = f"{r.delta_rotation_deg:+.1f}°"
        cv2.putText(canvas, rot_str, (mx_, my_),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(canvas, rot_str, (mx_, my_),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return canvas

CALIB_FILE = Path(PROJECT_ROOT) / "calib_data" / "homography.json"
TMPL_FILE  = Path(PROJECT_ROOT) / "calib_data" / "templates.json"
WINDOW     = "A路识别测试  [SPACE=冻结  Q=退出]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", "-c", type=int, default=0)
    ap.add_argument("--mode", "-m", choices=["A", "B", "auto"], default="auto",
                    help="A=模板匹配  B=几何求解  auto=有模板用A否则B")
    args = ap.parse_args()

    cfg = get_config()

    if not TMPL_FILE.exists():
        print(f"ERROR: 模板文件不存在: {TMPL_FILE}")
        print("请先运行 python scripts/build_templates.py")
        return 1

    templates = load_templates(TMPL_FILE)
    print(f"[OK] 加载 {len(templates)} 个模板")
    for t in templates:
        print(f"     T{t.template_id}[{t.label}]  sides={t.num_sides}  "
              f"area={t.area_mm2:.0f}mm²  "
              f"edges={[f'{e:.1f}' for e in t.edge_lengths_mm]}")

    H_px2mm = None
    if CALIB_FILE.exists():
        H_px2mm = _load_homography(CALIB_FILE)
        print(f"[OK] 加载标定文件")
    else:
        print("[WARN] 未找到标定文件，mm 坐标不可用")

    cap = None
    for be in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(args.camera, be)
        if cap.isOpened():
            break
        cap.release(); cap = None
    if cap is None:
        print("ERROR: 摄像头无法打开")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\nSPACE=冻结检测  Q=退出")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.putText(frame, "SPACE=冻结  Q=退出",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        if key == ord(" "):
            cap.release()
            cv2.destroyAllWindows()

            print("\n[冻结] 检测中...")
            pieces = detect_all_pieces(frame, cfg, H_px2mm)
            print(f"检测到 {len(pieces)} 片碎片")

            if not pieces:
                print("未检测到碎片")
            else:
                for p in pieces:
                    print(f"  P{p.index}: sides={p.num_sides}  area={p.area_mm2:.0f}mm²  "
                          f"edges={[f'{e:.1f}' for e in p.edge_lengths_mm]}")

                use_b = (args.mode == "B") or (args.mode == "auto" and not templates)
                if use_b:
                    print("\n[求解] B路几何求解...")
                    results = solve_geometric(pieces, cfg)
                else:
                    print("\n[求解] A路匹配...")
                    results = solve_template(pieces, templates, cfg)

                if not results:
                    print("  ! 无匹配结果")
                else:
                    print(f"\n{'='*55}")
                    for r in results:
                        print(f"  P{r.piece_index} → T{r.template_id}  "
                              f"conf={r.confidence:.2f}  rmse={r.match_rmse_mm:.1f}mm")
                        print(f"    吸取点: ({r.suction_point_mm[0]:.1f}, {r.suction_point_mm[1]:.1f}) mm")
                        print(f"    放置点: ({r.target_center_mm[0]:.1f}, {r.target_center_mm[1]:.1f}) mm")
                        print(f"    旋转:   {r.delta_rotation_deg:+.1f}°")
                    print(f"{'='*55}")

            # 显示检测结果图
            det_vis = draw_detected(frame, pieces)

            # 叠加匹配结果文字
            if pieces and results:
                matched = {r.piece_index: r for r in results}
                for p in pieces:
                    r = matched.get(p.index)
                    cx, cy = int(p.center_px[0]), int(p.center_px[1])
                    if r:
                        label = f"T{r.template_id} {r.confidence:.2f}"
                        color = (0, 255, 0) if r.confidence >= 0.5 else (0, 165, 255)
                    else:
                        label = "?"
                        color = (0, 0, 255)
                    cv2.putText(det_vis, label, (cx - 30, cy + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                    cv2.putText(det_vis, label, (cx - 30, cy + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

            win2 = "检测结果  [任意键继续]"
            cv2.namedWindow(win2, cv2.WINDOW_NORMAL)
            cv2.imshow(win2, det_vis)

            # 拼接示意图
            if results:
                asm_vis = draw_assembly_result(pieces, results)
                win3 = "拼接方案  [任意键继续]"
                cv2.namedWindow(win3, cv2.WINDOW_NORMAL)
                cv2.imshow(win3, asm_vis)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # 重新打开摄像头
            cap = cv2.VideoCapture(args.camera)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    if cap and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
