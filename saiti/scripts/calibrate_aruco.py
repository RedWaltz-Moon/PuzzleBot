#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/calibrate_aruco.py — ArUco 纸面标定

流程：
  1. 实时预览，摄像头对准 A4 纸面上的 ArUco 标记
  2. 按 SPACE 冻结画面
  3. 终端逐个输入每个标记在纸面上的 mm 坐标
  4. 自动计算单应矩阵 H（像素 → 纸面 mm）
  5. 按 S 保存到 calib_data/homography.json

坐标系：A4 纸面，原点左上角，右为 +X，下为 +Y（单位 mm）
推荐标记布局（4×4_50，打印在纸面固定位置）：
  ID 0 — 左上角附近，如 (20, 20)
  ID 1 — 右上角附近，如 (190, 20)
  ID 2 — 右下角附近，如 (190, 277)
  ID 3 — 左下角附近，如 (20, 277)
  （可自定义，只要与实际位置一致即可）

用法：
  python scripts/calibrate_aruco.py
  python scripts/calibrate_aruco.py -c 1
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.config import get_config

WINDOW = "ArUco 标定  [SPACE=冻结/重拍  S=保存  Q=退出]"


# ---------------------------------------------------------------------------
# ArUco 检测
# ---------------------------------------------------------------------------
def detect_aruco(frame: np.ndarray):
    """返回 {id: (cx_px, cy_px)}"""
    aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params       = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, params)
    gray         = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    result = {}
    if ids is not None:
        for i, mid in enumerate(ids.flatten()):
            pts    = corners[i].reshape(4, 2)
            cx, cy = pts.mean(axis=0)
            result[int(mid)] = (float(cx), float(cy))
    return result


def draw_markers(frame: np.ndarray, markers: dict) -> np.ndarray:
    vis = frame.copy()
    for mid, (cx, cy) in markers.items():
        ix, iy = int(cx), int(cy)
        cv2.drawMarker(vis, (ix, iy), (0, 255, 0), cv2.MARKER_CROSS, 28, 2)
        cv2.circle(vis, (ix, iy), 12, (0, 255, 0), 2)
        cv2.putText(vis, f"ID {mid}  ({cx:.0f},{cy:.0f})",
                    (ix + 16, iy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, f"ID {mid}  ({cx:.0f},{cy:.0f})",
                    (ix + 16, iy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    cnt_text = f"检测到 {len(markers)} 个标记"
    cv2.putText(vis, cnt_text, (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, cnt_text, (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if len(markers) >= 4 else (0, 120, 255),
                2, cv2.LINE_AA)
    cv2.putText(vis, "SPACE=冻结  Q=退出", (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------------------
# 输入 mm 坐标
# ---------------------------------------------------------------------------
def input_mm_coords(markers: dict) -> dict:
    """终端交互：为每个检测到的标记输入纸面 mm 坐标。
    返回 {id: (x_mm, y_mm)}"""
    print()
    print("=" * 50)
    print("  检测到以下标记，请输入各标记在纸面上的 mm 坐标")
    print("  坐标系：左上角为原点，右为 +X，下为 +Y")
    print("=" * 50)
    result = {}
    for mid in sorted(markers.keys()):
        cx, cy = markers[mid]
        print(f"\n  ID {mid}  像素位置: ({cx:.1f}, {cy:.1f})")
        while True:
            try:
                x = float(input(f"    纸面 X (mm): "))
                y = float(input(f"    纸面 Y (mm): "))
                result[mid] = (x, y)
                break
            except ValueError:
                print("    请输入数字")
    return result


# ---------------------------------------------------------------------------
# 计算单应矩阵 + 误差报告
# ---------------------------------------------------------------------------
def compute_homography(pixel_pts, mm_pts):
    """返回 (H, rmse, max_err, residuals)"""
    src = np.array(pixel_pts, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.array(mm_pts,    dtype=np.float32).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        raise RuntimeError("单应矩阵计算失败，请检查标定点是否共线或重复")

    residuals = []
    for i in range(len(src)):
        p = np.array([src[i, 0, 0], src[i, 0, 1], 1.0])
        q = H @ p
        q /= q[2]
        err = float(np.linalg.norm(q[:2] - dst[i, 0]))
        residuals.append(err)

    rmse    = float(np.sqrt(np.mean(np.array(residuals) ** 2)))
    max_err = float(np.max(residuals))
    return H, rmse, max_err, residuals


def print_report(markers, mm_map, residuals, rmse, max_err):
    print()
    print("=" * 50)
    print("  标定结果")
    print("=" * 50)
    print(f"  点数: {len(markers)}")
    print(f"  RMSE: {rmse:.3f} mm")
    print(f"  最大残差: {max_err:.3f} mm")
    print()
    print(f"  {'ID':>4}  {'像素X':>8}  {'像素Y':>8}  {'纸面X':>8}  {'纸面Y':>8}  {'残差':>8}")
    print("  " + "-" * 52)
    for i, mid in enumerate(sorted(markers.keys())):
        cx, cy   = markers[mid]
        mx, my   = mm_map[mid]
        res      = residuals[i] if i < len(residuals) else 0.0
        flag     = " !" if res > 2.0 else ""
        print(f"  {mid:>4}  {cx:>8.1f}  {cy:>8.1f}  {mx:>8.1f}  {my:>8.1f}  {res:>7.3f}{flag}")
    print()
    if rmse < 1.0:
        print("  [OK] 精度良好 (RMSE < 1 mm)")
    elif rmse < 2.0:
        print("  △ 精度一般 (RMSE < 2 mm)，可接受")
    else:
        print("  [WARN] 精度较差 (RMSE >= 2 mm)，建议重新标定")


# ---------------------------------------------------------------------------
# 保存 JSON（与 CalibrationManager.load() 兼容）
# ---------------------------------------------------------------------------
def save_homography(H, markers, mm_map, residuals, rmse, max_err, path: Path):
    pairs = []
    for i, mid in enumerate(sorted(markers.keys())):
        pairs.append({
            "pixel":        list(markers[mid]),
            "machine":      list(mm_map[mid]),
            "label":        f"ArUco_ID{mid}",
            "residual_mm":  residuals[i] if i < len(residuals) else 0.0,
        })
    data = {
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method":           "aruco_4x4_50",
        "H":                H.tolist(),
        "rmse_mm":          rmse,
        "max_residual_mm":  max_err,
        "num_pairs":        len(pairs),
        "pairs":            pairs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [OK] 已保存到: {path}")


# ---------------------------------------------------------------------------
# 验证：鼠标悬停显示 mm 坐标
# ---------------------------------------------------------------------------
def verify_mode(frame, H):
    print()
    print("  进入验证模式：移动鼠标查看纸面坐标，Q 退出")
    win = "验证  [Q=退出]"
    base = frame.copy()

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_MOUSEMOVE:
            return
        p = np.array([x, y, 1.0], dtype=np.float64)
        q = H @ p
        q /= q[2]
        mx, my = q[0], q[1]
        vis = base.copy()
        cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
        label = f"({mx:.1f}, {my:.1f}) mm"
        cv2.putText(vis, label, (x + 10, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (x + 10, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(win, vis)

    cv2.namedWindow(win)
    cv2.imshow(win, base)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        if cv2.waitKey(50) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyWindow(win)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", "-c", type=int, default=0)
    args = ap.parse_args()
    cfg  = get_config()

    cap = None
    for be in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(args.camera, be)
        if cap.isOpened():
            break
        cap.release(); cap = None
    if cap is None:
        print("ERROR: 摄像头无法打开"); return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=" * 50)
    print("  ArUco 纸面标定")
    print("  将摄像头对准 A4 纸面上的 ArUco 标记（至少 4 个）")
    print("  SPACE=冻结画面  Q=退出")
    print("=" * 50)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    frozen_frame  = None
    frozen_markers = None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        markers = detect_aruco(frame)
        vis     = draw_markers(frame, markers)
        cv2.imshow(WINDOW, vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            cap.release()
            cv2.destroyAllWindows()
            return 0

        if key == ord(" "):
            if len(markers) < 4:
                print(f"  ! 只检测到 {len(markers)} 个标记，需要至少 4 个，继续对准...")
                continue

            # 冻结画面
            frozen_frame   = frame.copy()
            frozen_markers = markers.copy()
            cap.release()
            cv2.destroyAllWindows()
            break

    if frozen_markers is None:
        print("未捕获标记，退出")
        return 1

    # 终端输入 mm 坐标
    mm_map = input_mm_coords(frozen_markers)

    # 按 ID 排序对齐
    ids        = sorted(frozen_markers.keys())
    pixel_pts  = [frozen_markers[i] for i in ids]
    mm_pts     = [mm_map[i] for i in ids]

    # 计算单应矩阵
    try:
        H, rmse, max_err, residuals = compute_homography(pixel_pts, mm_pts)
    except RuntimeError as e:
        print(f"  [ERROR] {e}")
        return 1

    frozen_vis = draw_markers(frozen_frame, frozen_markers)
    print_report(frozen_markers, mm_map, residuals, rmse, max_err)

    # 验证
    ans = input("进入鼠标验证模式？[Y/n]: ").strip().lower()
    if ans in ("", "y"):
        verify_mode(frozen_vis, H)

    # 保存
    ans = input("保存标定结果？[Y/n]: ").strip().lower()
    if ans in ("", "y"):
        save_homography(H, frozen_markers, mm_map, residuals, rmse, max_err,
                        cfg.homography_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
