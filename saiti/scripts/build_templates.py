#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_templates.py — A路模板录制工具

流程（赛前一次性操作）：
  1. 把 4 片碎片摆放到拼图区的 **最终正确位置**（即比赛时要还原的状态）
  2. 运行本脚本，摄像头拍摄整张纸面（含拼图区）
  3. 按 SPACE 冻结画面，脚本自动检测碎片，终端逐片确认编号和标签
  4. 按 S 保存到 calib_data/templates.json

关键约定：
  - 检测在矫正后的全纸面进行（separator_x=0），不限制取料区
  - 当前位置即目标位置（建模时碎片已在正确位姿）
  - 需要先有标定文件 calib_data/homography.json

用法：
    conda activate yolo
    python scripts/build_templates.py
    python scripts/build_templates.py -c 1
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from vision.config import get_config
from vision.detection import (
    _segment_white_bright,
    _segment_color_hsv,
    analyze_piece_geometry,
    simplify_contour,
    _compute_suction_point,
    sort_vertices_cyclic,
    Piece,
)
from vision.solver import build_template_from_piece, save_templates, PieceTemplate
import json
import math

WINDOW     = "模板录制  [SPACE=冻结  S=保存  Q=退出]"
CALIB_FILE = Path(PROJECT_ROOT) / "calib_data" / "homography.json"
TMPL_FILE  = Path(PROJECT_ROOT) / "calib_data" / "templates.json"


def _load_homography(path: Path) -> Optional[np.ndarray]:
    """从 calibrate_aruco.py 生成的 JSON 文件读取单应矩阵 H（像素→mm）。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return np.array(data["H"], dtype=np.float64)
    except Exception as e:
        print(f"[WARN] 读取标定文件失败: {e}")
        return None


def _px_to_mm(H: np.ndarray, x: float, y: float) -> tuple:
    """用单应矩阵把像素坐标转换为纸面 mm 坐标。"""
    p = np.array([x, y, 1.0], dtype=np.float64)
    q = H @ p
    q /= q[2]
    return (float(q[0]), float(q[1]))


# ---------------------------------------------------------------------------
# 全图碎片检测（不限制取料区）
# ---------------------------------------------------------------------------

def detect_all_pieces(frame: np.ndarray, cfg, H_px2mm: Optional[np.ndarray],
                      debug_mask_out: Optional[list] = None,
                      scene: str = 'white') -> List[Piece]:
    """
    直接在相机原始帧上检测碎片。

    scene='white' : 白纸路径——白色 HSV 阈值 + 3×3 闭运算
    scene='card'  : 扑克路径——绿底减除 + 15×15 闭运算
    """
    fh, fw = frame.shape[:2]
    min_area_px = fh * fw * 0.001    # 0.1 % — 过滤极小噪点
    max_area_px = fh * fw * 0.25     # 25 % — 过滤纸面本身

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ── 步骤 1：找绿色纸面轮廓（只用于定位纸面区域，不需要精确）──
    green_raw = cv2.inRange(
        hsv,
        np.array([cfg.paper_green_h_lo, cfg.paper_green_s_min, cfg.paper_green_v_min]),
        np.array([cfg.paper_green_h_hi, 255, 255]),
    )
    # 大核闭运算：把碎片占用的孔洞填掉，得到连续纸面区域
    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    k_sml = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green_filled = cv2.morphologyEx(green_raw, cv2.MORPH_CLOSE, k_big, iterations=4)
    green_filled = cv2.morphologyEx(green_filled, cv2.MORPH_OPEN,  k_sml, iterations=1)

    paper_contours, _ = cv2.findContours(green_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not paper_contours:
        print("  [WARN] 未检测到绿色纸面，请调整 HSV 参数")
        if debug_mask_out is not None:
            debug_mask_out.append(np.zeros((fh, fw), dtype=np.uint8))
        return []
    paper_cnt = max(paper_contours, key=cv2.contourArea)
    paper_mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.drawContours(paper_mask, [paper_cnt], -1, 255, cv2.FILLED)

    # ── 步骤 2：按场景走独立路径 ──────────────────────────────────────────
    # ── 白纸路径：原有白色 HSV 阈值 + 3×3 闭运算 ──────────────────────────
    if scene == 'white':
        white_mask = cv2.inRange(
            hsv,
            np.array([0,   0, 236]),
            np.array([179, 78, 255]),
        )
        white_in_paper = cv2.bitwise_and(white_mask, paper_mask)
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        filled = cv2.morphologyEx(white_in_paper, cv2.MORPH_CLOSE, k3, iterations=2)

    # ── 扑克路径：绿底减除 + 15×15 闭运算 ────────────────────────────────
    else:
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        green_dilated = cv2.dilate(green_raw, k_dil, iterations=1)
        # 用绿纸轮廓的凸包限定纸面区域：凸包紧贴绿纸外框，桌面在凸包外自然被截掉
        hull_paper = cv2.convexHull(paper_cnt)
        hull_mask = np.zeros((fh, fw), dtype=np.uint8)
        cv2.drawContours(hull_mask, [hull_paper], -1, 255, cv2.FILLED)
        # 小幅内缩去掉轮廓抖动边界
        k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        paper_mask_inner = cv2.erode(hull_mask, k_erode, iterations=1)
        card_fg = cv2.bitwise_and(cv2.bitwise_not(green_dilated), paper_mask_inner)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        filled = cv2.morphologyEx(card_fg, cv2.MORPH_CLOSE, k_close)
        # 不做 MORPH_OPEN：open 会腐蚀掉 close 在黑色图案与白底之间搭起的细桥，
        # 导致黑色区域变孤立小块被面积过滤器误杀。小噪点由面积和 solidity 过滤器处理。

    if debug_mask_out is not None:
        debug_mask_out.append(filled.copy())

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
        # 扑克场景：K 牌切口均为直线，碎片为凸多边形。
        # 先取凸包再简化：凸包自动压平圆角和边缘噪点，Douglas-Peucker 结果更干净。
        if scene == 'card':
            poly = simplify_contour(cv2.convexHull(contour))
        else:
            poly = simplify_contour(contour)
        if poly is None or len(poly) < 3:
            continue  # 简化后退化为线段，跳过
        suction_px = _compute_suction_point(contour)

        piece = Piece(
            index=0,
            contour=contour,
            vertices_px=poly,
            center_px=center_px,
            area_px=area,
            solidity=solidity,
            suction_point_px=suction_px,
        )

        # mm 坐标（用标定 H 直接转换相机像素 → 纸面 mm）
        if H_px2mm is not None:
            piece.center_mm = _px_to_mm(H_px2mm, center_px[0], center_px[1])
            if poly is not None and len(poly) > 0:
                piece.vertices_mm = np.array(
                    [_px_to_mm(H_px2mm, float(v[0]), float(v[1])) for v in poly],
                    dtype=np.float32,
                )
            if suction_px is not None:
                piece.suction_point_mm = _px_to_mm(H_px2mm, *suction_px)
            cnt_pts = contour[:, 0, :]
            step = max(1, len(cnt_pts) // 80)
            sampled = cnt_pts[::step]
            piece.contour_mm = np.array(
                [_px_to_mm(H_px2mm, float(pt[0]), float(pt[1])) for pt in sampled],
                dtype=np.float32,
            )

        # 几何分析：有 mm 顶点则从 mm 重算边长
        piece = analyze_piece_geometry(piece, cfg.px_per_mm)
        if piece.vertices_mm is not None and len(piece.vertices_mm) >= 2:
            verts_mm = sort_vertices_cyclic(piece.vertices_mm)
            n = len(verts_mm)
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

        # ── 扑克场景：过滤边长超出 K 牌范围的误判（极扁细条、桌面残留等）──
        if scene == 'card' and piece.edge_lengths_mm:
            _max_e_allowed = getattr(cfg, 'card_max_edge_mm', 120.0)
            if max(piece.edge_lengths_mm) > _max_e_allowed:
                continue

        # ── 场景决定碎片类型 ──
        # 白纸场景保留默认 "unknown"：b_solver 伪顶点净化对 unknown 走带转角保护的 card 路径，
        # 与原始行为一致；扑克场景标 "card" 以启用宽松阈值和纹理重排序。
        if scene == 'card':
            _xbb, _ybb, _wbb, _hbb = cv2.boundingRect(contour)
            _y1, _y2 = max(0, _ybb), min(fh, _ybb + _hbb)
            _x1, _x2 = max(0, _xbb), min(fw, _xbb + _wbb)
            _pmask = np.zeros((fh, fw), dtype=np.uint8)
            cv2.drawContours(_pmask, [contour], -1, 255, cv2.FILLED)
            _roi_mask = _pmask[_y1:_y2, _x1:_x2]
            piece.piece_type = "card"
            _tex = frame[_y1:_y2, _x1:_x2].copy()
            _tex[_roi_mask == 0] = 0
            piece.texture_bgr = _tex
            piece.texture_origin_px = (_xbb, _ybb)

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

        # 吸取点
        if p.suction_point_px:
            sx, sy = int(p.suction_point_px[0]), int(p.suction_point_px[1])
            cv2.drawMarker(vis, (sx, sy), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)

        # 编号 + 面积 + 边数
        label = f"P{p.index} {p.num_sides}s {p.area_mm2:.0f}mm2"
        cv2.putText(vis, label, (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------------------
# 终端交互：为每片碎片指定模板 ID
# ---------------------------------------------------------------------------

def assign_template_ids(pieces: List[Piece]) -> List[Optional[int]]:
    """
    终端逐片确认模板编号。
    返回与 pieces 等长的列表，None = 跳过该片。
    """
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
    ap.add_argument("--camera", "-c", type=int, default=0)
    ap.add_argument("--output", "-o", type=str, default=str(TMPL_FILE))
    args = ap.parse_args()

    cfg = get_config()

    # 尝试加载标定（H：矫正图像像素 → 纸面 mm）
    H_px2mm: Optional[np.ndarray] = None
    if CALIB_FILE.exists():
        H_px2mm = _load_homography(CALIB_FILE)
        if H_px2mm is not None:
            print(f"[OK] 已加载标定文件: {CALIB_FILE}")
        else:
            print(f"[WARN] 标定文件解析失败，mm 坐标将不可用")
    else:
        print(f"[WARN] 未找到标定文件 {CALIB_FILE}，mm 坐标将不可用")

    # 打开摄像头
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

    print()
    print("=" * 55)
    print("  A路模板录制")
    print("  把碎片摆到 拼图区 最终正确位置，按 SPACE 冻结")
    print("  SPACE=冻结  S=保存  Q=退出")
    print("=" * 55)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    templates: List[PieceTemplate] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy()
        cv2.putText(vis, f"T已录={len(templates)}  calib={'OK' if H_px2mm is not None else 'NO'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, "SPACE=冻结录制  D=调试mask  S=保存  Q=退出",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW, vis)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        if key == ord("s"):
            if not templates:
                print("  ! 尚未录制任何模板")
            else:
                out_path = Path(args.output)
                save_templates(templates, out_path)
                print(f"  [OK] 已保存 {len(templates)} 个模板到 {out_path}")

        # D 键：显示调试 mask（帮助调整分割参数）
        if key == ord("d"):
            ok2, dbg_frame = cap.read()
            dbg_mask_out: list = []
            detect_all_pieces(dbg_frame if ok2 else frame, cfg, H_px2mm, dbg_mask_out)
            if dbg_mask_out:
                win_dbg = "分割 mask  [任意键关闭]"
                cv2.namedWindow(win_dbg, cv2.WINDOW_NORMAL)
                cv2.imshow(win_dbg, dbg_mask_out[0])
                cv2.waitKey(0)
                cv2.destroyWindow(win_dbg)

        if key == ord(" "):
            cap.release()
            cv2.destroyAllWindows()

            print("\n  [冻结] 正在检测碎片...")
            dbg_mask_out2: list = []
            pieces = detect_all_pieces(frame, cfg, H_px2mm, dbg_mask_out2)
            print(f"  检测到 {len(pieces)} 片碎片")

            # 始终显示 mask，方便判断检测效果
            if dbg_mask_out2:
                win_mask = "分割 mask  [任意键继续]"
                cv2.namedWindow(win_mask, cv2.WINDOW_NORMAL)
                cv2.imshow(win_mask, dbg_mask_out2[0])
                cv2.waitKey(0)
                cv2.destroyWindow(win_mask)

            if not pieces:
                print("  ! 未检测到碎片")
                print("  提示: 按 D 键实时查看 mask；检查 tune_white.py 参数")
                cap = cv2.VideoCapture(args.camera)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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
                # 检查重复
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
            cap = cv2.VideoCapture(args.camera)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    if cap and cap.isOpened():
        cap.release()
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
