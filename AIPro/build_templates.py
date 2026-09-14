#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_templates.py — A路模板录制工具（嵌入式精简版）

从 saiti/scripts/build_templates.py 精简适配：
  - 移除 Windows 特定后端
  - 简化交互流程
  - 适配香橙派摄像头

流程（赛前一次性操作）：
  1. 把 4 片碎片摆放到拼图区的最终正确位置
  2. 运行本脚本，摄像头实时预览
  3. 按 SPACE 冻结画面 → 自动检测碎片 → 终端逐片确认编号
  4. 按 S 保存到 calib_data/templates.json

用法：
    python build_templates.py
    python build_templates.py -c 1
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_config
from detection import (
    simplify_contour,
    _compute_suction_point,
    analyze_piece_geometry,
    sort_vertices_cyclic,
    Piece,
)
from solver import build_template_from_piece, save_templates, PieceTemplate
from camera import Camera

WINDOW     = "模板录制  [SPACE=冻结  S=保存  Q=退出]"
TMPL_FILE  = Path(os.path.dirname(os.path.abspath(__file__))) / "calib_data" / "templates.json"
CALIB_FILE = Path(os.path.dirname(os.path.abspath(__file__))) / "calib_data" / "homography.json"


def _load_homography(path: Path) -> Optional[np.ndarray]:
    """加载单应矩阵 H（像素→mm）。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return np.array(data["H"], dtype=np.float64)
    except Exception as e:
        print(f"[WARN] 读取标定文件失败: {e}")
        return None


def _px_to_mm(H: np.ndarray, x: float, y: float) -> tuple:
    """像素 → mm 坐标转换。"""
    p = np.array([x, y, 1.0], dtype=np.float64)
    q = H @ p
    q /= q[2]
    return (float(q[0]), float(q[1]))


# ---------------------------------------------------------------------------
# 全图碎片检测（用于模板录制：碎片在拼图区正确位置）
# ---------------------------------------------------------------------------

def detect_all_pieces(frame: np.ndarray, cfg: Config,
                      H_px2mm: Optional[np.ndarray]) -> List[Piece]:
    """
    直接在相机原始帧上检测碎片（不需要纸面矫正）。
    策略：检测绿色纸面 → 纸面内部非绿区域即为碎片。
    """
    fh, fw = frame.shape[:2]
    min_area_px = fh * fw * 0.001
    max_area_px = fh * fw * 0.25

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 找绿色纸面
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
        print("  [WARN] 未检测到绿色纸面，请调整 HSV 参数")
        return []
    paper_cnt = max(paper_contours, key=cv2.contourArea)
    paper_mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.drawContours(paper_mask, [paper_cnt], -1, 255, cv2.FILLED)

    # 纸面内检测白色碎片
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 236]),
        np.array([179, 78, 255]),
    )
    white_in_paper = cv2.bitwise_and(white_mask, paper_mask)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    filled = cv2.morphologyEx(white_in_paper, cv2.MORPH_CLOSE, k3, iterations=2)

    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pieces: List[Piece] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not (min_area_px <= area <= max_area_px):
            continue
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / max(hull_area, 1.0)
        if solidity < cfg.piece_min_solidity:
            continue
        m = cv2.moments(contour)
        if abs(m["m00"]) < 1e-9:
            continue

        center_px = (m["m10"] / m["m00"], m["m01"] / m["m00"])
        poly = simplify_contour(contour)
        suction_px = _compute_suction_point(contour)

        piece = Piece()
        piece.contour = contour
        piece.vertices_px = poly
        piece.center_px = center_px
        piece.area_px = area
        piece.solidity = solidity
        piece.suction_point_px = suction_px

        # mm 坐标
        if H_px2mm is not None:
            piece.center_mm = _px_to_mm(H_px2mm, center_px[0], center_px[1])
            if poly is not None and len(poly) > 0:
                piece.vertices_mm = np.array(
                    [_px_to_mm(H_px2mm, float(v[0]), float(v[1])) for v in poly],
                    dtype=np.float32,
                )
            if suction_px is not None:
                piece.suction_point_mm = _px_to_mm(H_px2mm, *suction_px)

        piece = analyze_piece_geometry(piece, cfg.px_per_mm)

        # 从 mm 顶点重算边长/面积
        if piece.vertices_mm is not None and len(piece.vertices_mm) >= 2:
            verts_mm = sort_vertices_cyclic(piece.vertices_mm)
            n = len(verts_mm)
            piece.edge_lengths_mm = []
            for i in range(n):
                l_mm = float(np.linalg.norm(verts_mm[(i + 1) % n] - verts_mm[i]))
                piece.edge_lengths_mm.append(l_mm)
                piece.edge_lengths_px.append(l_mm * cfg.px_per_mm)
            x_mm = verts_mm[:, 0]
            y_mm = verts_mm[:, 1]
            piece.area_mm2 = abs(float(
                np.dot(x_mm, np.roll(y_mm, -1)) - np.dot(y_mm, np.roll(x_mm, -1))
            )) / 2.0

        pieces.append(piece)

    pieces.sort(key=lambda p: p.area_px, reverse=True)
    for i, p in enumerate(pieces):
        p.index = i + 1
    return pieces


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def draw_detected(image: np.ndarray, pieces: List[Piece]) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# 终端交互
# ---------------------------------------------------------------------------

def assign_template_ids(pieces: List[Piece]) -> List[Optional[tuple]]:
    """终端逐片确认模板编号。"""
    print()
    print("=" * 55)
    print("  为每片碎片指定模板编号（输入数字 0-3，回车跳过）")
    print("  建议: 按左上=0, 右上=1, 右下=2, 左下=3 编号")
    print("=" * 55)

    ids = []
    for p in pieces:
        print(f"\n  P{p.index}  边数={p.num_sides}  面积={p.area_mm2:.0f}mm²  "
              f"中心=({p.center_px[0]:.0f},{p.center_px[1]:.0f})px")
        if p.center_mm:
            print(f"          mm坐标=({p.center_mm[0]:.1f},{p.center_mm[1]:.1f})")
        while True:
            raw = input("    模板编号 [0-3 / 回车跳过]: ").strip()
            if raw == "":
                ids.append(None)
                print("    → 跳过")
                break
            try:
                tid = int(raw)
                if 0 <= tid <= 9:
                    label = input(f"    标签 (如 top_left，回车留空): ").strip()
                    ids.append((tid, label))
                    print(f"    → T{tid}  '{label}'")
                    break
                else:
                    print("    请输入 0-9")
            except ValueError:
                print("    请输入数字")
    return ids


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="A路模板录制工具")
    ap.add_argument("--camera", "-c", type=int, default=None)
    ap.add_argument("--output", "-o", type=str, default=str(TMPL_FILE))
    args = ap.parse_args()

    cfg = get_config()

    # 加载标定
    H_px2mm: Optional[np.ndarray] = None
    if CALIB_FILE.exists():
        H_px2mm = _load_homography(CALIB_FILE)
        if H_px2mm is not None:
            print(f"[OK] 已加载标定文件: {CALIB_FILE}")
        else:
            print(f"[WARN] 标定文件解析失败，mm 坐标将不可用")
    else:
        print(f"[WARN] 未找到标定文件 {CALIB_FILE}，mm 坐标将不可用")

    # 打开摄像头（使用 Camera 类，带自动检测+预热）
    cam = Camera(index=args.camera if args.camera is not None else cfg.camera_index,
                 width=1280, height=720, fps=30)
    cam.open()

    print()
    print("=" * 55)
    print("  A路模板录制")
    print("  把碎片摆到 拼图区 最终正确位置，按 SPACE 冻结")
    print("  SPACE=冻结  S=保存  Q=退出")
    print("=" * 55)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    templates: List[PieceTemplate] = []

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        vis = frame.copy()
        cv2.putText(vis, f"T已录={len(templates)}  calib={'OK' if H_px2mm is not None else 'NO'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, "SPACE=冻结录制  S=保存  Q=退出",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW, vis)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):  # Q 或 ESC
            break

        if key == ord("s"):
            if not templates:
                print("  ! 尚未录制任何模板")
            else:
                out_path = Path(args.output)
                save_templates(templates, out_path)
                print(f"  [OK] 已保存 {len(templates)} 个模板到 {out_path}")

        if key == ord(" "):
            cam.close()
            cv2.destroyAllWindows()

            print("\n  [冻结] 正在检测碎片...")
            pieces = detect_all_pieces(frame, cfg, H_px2mm)
            print(f"  检测到 {len(pieces)} 片碎片")

            if not pieces:
                print("  ! 未检测到碎片")
                print("  提示: 检查光照、绿色纸面HSV参数")
                cam.open()
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                continue

            # 显示检测结果
            det_vis = draw_detected(frame, pieces)
            win_det = "检测结果  [任意键继续]"
            cv2.namedWindow(win_det, cv2.WINDOW_NORMAL)
            cv2.imshow(win_det, det_vis)
            cv2.waitKey(0)
            cv2.destroyWindow(win_det)

            # 终端指定模板 ID
            id_assignments = assign_template_ids(pieces)

            # 构建模板
            new_count = 0
            for piece, assignment in zip(pieces, id_assignments):
                if assignment is None:
                    continue
                tid, label = assignment
                existing_ids = {t.template_id for t in templates}
                if tid in existing_ids:
                    overwrite = input(f"    T{tid} 已存在，覆盖？[y/N]: ").strip().lower()
                    if overwrite != "y":
                        print(f"    → 跳过 T{tid}")
                        continue
                    templates = [t for t in templates if t.template_id != tid]

                tmpl = build_template_from_piece(piece, tid, label)
                templates.append(tmpl)
                new_count += 1
                print(f"  [OK] T{tid}[{label}]: sides={tmpl.num_sides} "
                      f"area={tmpl.area_mm2:.0f}mm² "
                      f"edges={[f'{e:.1f}' for e in tmpl.edge_lengths_mm]} "
                      f"target=({tmpl.target_center_mm[0]:.1f},{tmpl.target_center_mm[1]:.1f})mm "
                      f"angle={tmpl.target_angle_deg:.1f}°")

            print(f"\n  本次新增 {new_count} 个模板，共 {len(templates)} 个")
            print("  继续录制下一组，或按 S 保存，Q 退出\n")

            # 重新打开摄像头
            cam.open()
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    cam.close()
    cv2.destroyAllWindows()

    if templates:
        print(f"\n已录制 {len(templates)} 个模板，尚未保存，是否保存？")
        ans = input("保存 [Y/n]: ").strip().lower()
        if ans in ("", "y"):
            out_path = Path(args.output)
            save_templates(templates, out_path)
            print(f"[OK] 已保存到 {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
