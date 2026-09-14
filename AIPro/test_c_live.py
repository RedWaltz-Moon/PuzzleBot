#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_c_live.py — 三路算法调试工具（键盘模拟MCU）

流程：
  摄像头实时预览 → 按 1/2/3 选择路线并触发拍照 → 检测+求解 → 预览结果

  1 = A路（模板匹配）
  2 = B路（白纸几何拼接）
  3 = C路（扑克牌拼接）

用法：
    python test_c_live.py
    python test_c_live.py -c 1
    python test_c_live.py --image photo.jpg
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_config
from camera import Camera
from solver import (
    load_templates, PieceTemplate, PlacementResult, solve_template,
)
from b_solver import solve_geometric
from protocol import MODE_A, MODE_B, MODE_C
from main import detect_all_pieces, detect_card_pieces  # 直接复用生产代码，保证一致

# A4 尺寸 mm
_A4_W, _A4_H = 210.0, 297.0
_SCALE = 3.0

_COLORS = [
    (255, 100,  50),
    ( 50, 200, 255),
    ( 80, 255, 120),
    (200,  80, 255),
    (255, 220,  50),
    ( 50, 120, 255),
]

WINDOW = "调试工具  [1=A路  2=B路  3=C路  Q=退出]"
CALIB_FILE = Path(os.path.dirname(os.path.abspath(__file__))) / "calib_data" / "homography.json"
TMPL_FILE  = Path(os.path.dirname(os.path.abspath(__file__))) / "calib_data" / "templates.json"
DEBUG_DIR  = Path(os.path.dirname(os.path.abspath(__file__))) / "debug_output"

MODE_NAMES = {MODE_A: "A路(模板)", MODE_B: "B路(白纸)", MODE_C: "C路(扑克)"}


# =============================================================================
# 标定
# =============================================================================

def load_homography(path: Path) -> Optional[np.ndarray]:
    try:
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        return np.array(data["H"], dtype=np.float64)
    except Exception as e:
        print(f"[WARN] 读取标定失败: {e}")
        return None


# =============================================================================
# 可视化
# =============================================================================

def draw_detected(image, pieces):
    vis = image.copy()
    for p in pieces:
        cx, cy = int(p.center_px[0]), int(p.center_px[1])
        cv2.drawContours(vis, [p.contour.astype(np.int32)], -1, (0, 200, 255), 2)
        cv2.circle(vis, (cx, cy), 6, (0, 200, 255), -1)
        if p.suction_point_px:
            sx, sy = int(p.suction_point_px[0]), int(p.suction_point_px[1])
            cv2.drawMarker(vis, (sx, sy), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        type_str = f" [{p.piece_type}]" if p.piece_type != "unknown" else ""
        label = f"P{p.index} {p.num_sides}s {p.area_mm2:.0f}mm2{type_str}"
        cv2.putText(vis, label, (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return vis


def draw_assembly_preview(pieces, results):
    cw = int(_A4_W * _SCALE)
    ch = int(_A4_H * _SCALE)
    canvas = np.full((ch, cw, 3), 240, dtype=np.uint8)

    def mm2px(x, y):
        return (int(x * _SCALE), int(y * _SCALE))

    cv2.rectangle(canvas, (0, 0), (cw - 1, ch - 1), (180, 180, 180), 2)
    sep_y = int(_A4_H / 2 * _SCALE)
    cv2.line(canvas, (0, sep_y), (cw - 1, sep_y), (100, 100, 255), 2)
    cv2.putText(canvas, "Current", (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Target", (4, sep_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)

    piece_map = {p.index: p for p in pieces}

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

        pts_src = np.array([mm2px(v[0], v[1]) for v in p.vertices_mm], dtype=np.int32)
        cv2.polylines(canvas, [pts_src], True, (140, 140, 140), 1)
        cx = int(pts_src[:, 0].mean()); cy = int(pts_src[:, 1].mean())
        cv2.putText(canvas, f"P{r.piece_index}", (cx + 3, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1, cv2.LINE_AA)

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


def draw_results_overlay(frame, pieces, results, H_px2mm):
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
# 单次运行
# =============================================================================

def run_once(frame, mode, cfg, H_px2mm, templates):
    """对一帧图像执行 检测 → 求解 → 预览。"""
    mode_name = MODE_NAMES[mode]
    print(f"\n{'='*55}")
    print(f"  {mode_name} 检测中...")
    print(f"{'='*55}")

    DEBUG_DIR.mkdir(exist_ok=True)

    # ---- 检测（直接调用 main.py 的生产代码）----
    if mode == MODE_C:
        dbg = []
        pieces = detect_card_pieces(frame, cfg, H_px2mm, dbg=dbg)
        if dbg:
            win_mask = "分割 mask  [任意键继续]"
            cv2.namedWindow(win_mask, cv2.WINDOW_NORMAL)
            cv2.imshow(win_mask, dbg[0])
            cv2.waitKey(0)
            cv2.destroyWindow(win_mask)
    else:
        pieces = detect_all_pieces(frame, cfg, H_px2mm,
                                   mode="A" if mode == MODE_A else "B")

    print(f"\n[检测] 找到 {len(pieces)} 片碎片")
    if not pieces:
        print("  ! 未检测到碎片")
        return

    for p in pieces:
        ls = [f"{l:.1f}" for l in (p.edge_lengths_mm or [])]
        type_str = f" type={p.piece_type}" if p.piece_type != "unknown" else ""
        print(f"  P{p.index}: {p.num_sides}边  area={p.area_mm2:.0f}mm2  edges={ls}{type_str}")
        if p.center_mm:
            print(f"         center_mm=({p.center_mm[0]:.1f},{p.center_mm[1]:.1f})")
        if p.suction_point_mm:
            print(f"         suction_mm=({p.suction_point_mm[0]:.1f},{p.suction_point_mm[1]:.1f})")

    det_vis = draw_detected(frame, pieces)
    win_det = f"{mode_name} 检测结果  [任意键继续求解]"
    cv2.namedWindow(win_det, cv2.WINDOW_NORMAL)
    cv2.imshow(win_det, det_vis)
    cv2.waitKey(0)
    cv2.destroyWindow(win_det)

    # ---- 求解 ----
    if mode == MODE_A:
        if not templates:
            print("[SKIP] A路需要模板文件，请先运行 build_templates.py")
            return
        print("\n[求解] A路模板匹配...")
        results = solve_template(pieces, templates, cfg)
    else:
        if len(pieces) < 2:
            print(f"[SKIP] 碎片不足 2 片（{len(pieces)}片），跳过求解")
            return
        if any(p.vertices_mm is None for p in pieces):
            print("[SKIP] 缺少 mm 坐标")
            return
        print("\n[求解] DFS 拼接...")
        results = solve_geometric(pieces, cfg, H_px2mm=H_px2mm)

    if not results:
        print("[FAIL] 求解失败")
        if mode != MODE_A:
            print("  可能原因:")
            print("  1. 碎片顶点识别有误（边数/边长不准确）")
            print("  2. mm 坐标转换偏差过大")
            print("  3. 碎片尺寸超出范围")
        return

    # ---- 结果打印 ----
    print(f"\n[结果] 成功！共 {len(results)} 片：")
    print("-" * 60)
    for r in results:
        print(f"  P{r.piece_index}:")
        print(f"    吸取点  ({r.suction_point_mm[0]:.1f}, {r.suction_point_mm[1]:.1f}) mm")
        print(f"    放置中心({r.target_center_mm[0]:.1f}, {r.target_center_mm[1]:.1f}) mm")
        print(f"    旋转角  {r.delta_rotation_deg:+.1f} deg  置信度 {r.confidence:.3f}")
    print("-" * 60)

    # ---- 可视化 ----
    preview = draw_assembly_preview(pieces, results)
    win_prev = f"{mode_name} 拼接预览  [S=保存  任意键关闭]"
    cv2.namedWindow(win_prev, cv2.WINDOW_NORMAL)
    cv2.imshow(win_prev, preview)
    cv2.resizeWindow(win_prev, 420, 594)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("s"):
        out = DEBUG_DIR / "solve_preview.png"
        cv2.imwrite(str(out), preview)
        print(f"[保存] {out}")
    cv2.destroyWindow(win_prev)

    res_vis = draw_results_overlay(frame, pieces, results, H_px2mm)
    win_res = f"{mode_name} 求解结果  [任意键关闭]"
    cv2.namedWindow(win_res, cv2.WINDOW_NORMAL)
    cv2.imshow(win_res, res_vis)
    cv2.waitKey(0)
    cv2.destroyWindow(win_res)


# =============================================================================
# main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="三路算法调试工具")
    ap.add_argument("--camera", "-c", type=int, default=None)
    ap.add_argument("--image", "-img", type=str, default=None)
    args = ap.parse_args()

    cfg = get_config()
    if args.camera is not None:
        cfg.camera_index = args.camera

    H_px2mm = None
    if CALIB_FILE.exists():
        H_px2mm = load_homography(CALIB_FILE)
        print(f"[OK] 已加载标定: {CALIB_FILE}")
    else:
        print(f"[WARN] 未找到标定文件 {CALIB_FILE}")

    templates = []
    if TMPL_FILE.exists():
        templates = load_templates(TMPL_FILE)
        print(f"[OK] 已加载模板: {[f'T{t.template_id}[{t.label}]' for t in templates]}")
    else:
        print(f"[WARN] 未找到模板文件 {TMPL_FILE}（A路需要）")

    # 图片模式
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"ERROR: 无法读取 {args.image}")
            return 1

        print("\n选择路线:")
        print("  1 = A路（模板）")
        print("  2 = B路（白纸）")
        print("  3 = C路（扑克）")
        while True:
            raw = input("> ").strip()
            if raw == "1":
                mode = MODE_A; break
            elif raw == "2":
                mode = MODE_B; break
            elif raw == "3":
                mode = MODE_C; break
            print("  请输入 1/2/3")
        run_once(frame, mode, cfg, H_px2mm, templates)
        return 0

    # 摄像头模式
    cam = Camera(
        index=cfg.camera_index,
        width=cfg.camera_width,
        height=cfg.camera_height,
        fps=cfg.camera_fps,
    )
    cam.open()

    print()
    print("=" * 55)
    print("  三路算法调试工具")
    print("  1 = A路（模板）  2 = B路（白纸）  3 = C路（扑克）")
    print("  D = 查看mask    S = 截图    Q = 退出")
    print("=" * 55)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    current_mode = MODE_B  # 默认B路

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        vis = frame.copy()
        calib_str = "calib=OK" if H_px2mm is not None else "calib=NO"
        tmpl_str = f"templates={len(templates)}"
        mode_str = MODE_NAMES[current_mode]
        cv2.putText(vis, f"{calib_str}  {tmpl_str}  当前: {mode_str}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, "1=A路  2=B路  3=C路  D=mask  S=截图  Q=退出",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, vis)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        if key == ord("s"):
            out = DEBUG_DIR / "snapshot.jpg"
            cv2.imwrite(str(out), frame)
            print(f"[保存] {out}")

        if key == ord("d"):
            # 查看分割 mask（与 main.py detect_all_pieces 的白纸路径一致）
            fh, fw = frame.shape[:2]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            green_raw = cv2.inRange(hsv,
                np.array([cfg.paper_green_h_lo, cfg.paper_green_s_min, cfg.paper_green_v_min]),
                np.array([cfg.paper_green_h_hi, 255, 255]))
            k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            k_sml = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            green_filled = cv2.morphologyEx(green_raw, cv2.MORPH_CLOSE, k_big, iterations=4)
            green_filled = cv2.morphologyEx(green_filled, cv2.MORPH_OPEN, k_sml, iterations=1)
            paper_contours, _ = cv2.findContours(green_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            paper_mask = np.zeros((fh, fw), dtype=np.uint8)
            if paper_contours:
                paper_cnt = max(paper_contours, key=cv2.contourArea)
                cv2.drawContours(paper_mask, [paper_cnt], -1, 255, cv2.FILLED)
            white_mask = cv2.inRange(hsv,
                np.array([0, 0, 236]), np.array([179, 78, 255]))
            white_in_paper = cv2.bitwise_and(white_mask, paper_mask)
            k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            filled = cv2.morphologyEx(white_in_paper, cv2.MORPH_CLOSE, k3, iterations=2)
            win = "mask  [任意键关闭]"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.imshow(win, filled)
            cv2.waitKey(0)
            cv2.destroyWindow(win)

        if key in (ord("1"), ord("2"), ord("3")):
            if key == ord("1"):
                current_mode = MODE_A
            elif key == ord("2"):
                current_mode = MODE_B
            else:
                current_mode = MODE_C

            cam.close()
            cv2.destroyAllWindows()
            run_once(frame, current_mode, cfg, H_px2mm, templates)
            cam.open()
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            print(f"\n当前模式: {MODE_NAMES[current_mode]}  继续按 1/2/3 切换并拍照，Q 退出")

    cam.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
