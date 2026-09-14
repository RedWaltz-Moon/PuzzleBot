#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/test_b_live.py — B路真实环境测试

流程：
  摄像头 → 按 SPACE 拍照 → 检测碎片 → DFS 拼接求解 → 打印 PlacementResult

用法：
    "D:/Anaconda/envs/yolo/python.exe" -X utf8 scripts/test_b_live.py
    "D:/Anaconda/envs/yolo/python.exe" -X utf8 scripts/test_b_live.py -c 1
    "D:/Anaconda/envs/yolo/python.exe" -X utf8 scripts/test_b_live.py --image photo.jpg
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from vision.config import get_config
from vision.solver import solve_geometric

sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
from build_templates import detect_all_pieces, draw_detected, _load_homography

CALIB_FILE = Path(PROJECT_ROOT) / "calib_data" / "homography.json"
WINDOW = "B路实测  [SPACE=拍照  Q=退出]"

# A4 尺寸 mm
_A4_W, _A4_H = 210.0, 297.0

# 预览画布每 mm 对应的像素数
_SCALE = 3.0

# 各片颜色（BGR）
_COLORS = [
    (255, 100,  50),
    ( 50, 200, 255),
    ( 80, 255, 120),
    (200,  80, 255),
    (255, 220,  50),
    ( 50, 120, 255),
]


def _draw_assembly_preview(pieces, results):
    """
    单列 A4 预览：
    上半区 = 碎片当前位置（灰色轮廓）
    分界线
    下半区 = 目标拼合（彩色填充，顺时针旋转90°显示）
    """
    cw = int(_A4_W * _SCALE)
    ch = int(_A4_H * _SCALE)
    # 窗口宽度收窄到 A4 宽度，高度保持 A4 比例
    canvas = np.full((ch, cw, 3), 240, dtype=np.uint8)
    # 显示时外部调用 cv2.resizeWindow 控制窗口大小即可，此处保持 mm 比例不变

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
        # 90° CW: dx,dy → dy,-dx（相对质心）
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


def _draw_results(frame, pieces, results):
    """在图像上叠加检测轮廓 + 目标位置标注。"""
    vis = draw_detected(frame, pieces)
    try:
        H = _load_homography(CALIB_FILE)
        H_inv = np.linalg.inv(H) if H is not None else None
    except Exception:
        H_inv = None

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


def run_once(frame, cfg, H_px2mm, scene: str = 'white'):
    """对一帧图像执行完整的检测 + 求解流程。"""
    print(f"\n[检测] 正在分析图像...  场景={scene}")
    dbg: list = []
    pieces = detect_all_pieces(frame, cfg, H_px2mm, dbg, scene=scene)

    if dbg:
        win = "分割 mask  [任意键继续]"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, dbg[0])
        cv2.waitKey(0)
        cv2.destroyWindow(win)

    print(f"[检测] 找到 {len(pieces)} 片碎片")
    if not pieces:
        print("  提示: 检查绿色纸面 HSV 参数，或在摄像头主窗口按 D 查看 mask")
        return

    for p in pieces:
        ls = [f"{l:.1f}" for l in (p.edge_lengths_mm or [])]
        print(f"  P{p.index}: {p.num_sides}边  area={p.area_mm2:.0f}mm2  edges={ls}")
        if p.center_mm:
            print(f"         center_mm=({p.center_mm[0]:.1f},{p.center_mm[1]:.1f})")

    det_vis = draw_detected(frame, pieces)
    win2 = "检测结果  [任意键继续求解]"
    cv2.namedWindow(win2, cv2.WINDOW_NORMAL)
    cv2.imshow(win2, det_vis)
    cv2.waitKey(0)
    cv2.destroyWindow(win2)

    if len(pieces) < 2:
        print("[SKIP] 碎片不足 2 片，跳过求解")
        return
    if any(p.vertices_mm is None for p in pieces):
        print("[SKIP] 缺少 mm 坐标，请先完成相机标定 (scripts/calibrate_aruco.py)")
        return

    print("\n[求解] DFS 拼接中...")
    results = solve_geometric(pieces, cfg, H_px2mm=H_px2mm)

    if not results:
        print("[FAIL] 未找到合法拼接方案")
        print("  可能原因：")
        print("  1. 碎片顶点识别有误（边数 / 边长不准确）")
        print("  2. mm 坐标转换偏差过大（重新标定）")
        print("  3. 碎片尺寸超出范围（宽 82~128mm，高 42~98mm）")
        return

    print(f"\n[结果] 成功！共 {len(results)} 片放置指令：")
    print("-" * 60)
    for r in results:
        print(f"  P{r.piece_index}:")
        print(f"    吸取点  ({r.suction_point_mm[0]:.1f}, {r.suction_point_mm[1]:.1f}) mm")
        print(f"    放置中心({r.target_center_mm[0]:.1f}, {r.target_center_mm[1]:.1f}) mm")
        print(f"    旋转角  {r.delta_rotation_deg:+.1f} deg")
        print(f"    置信度  {r.confidence:.3f}")
    print("-" * 60)

    # 拼接预览（直观验证）
    preview = _draw_assembly_preview(pieces, results)
    win_prev = "拼接预览  [S=保存  任意键关闭]"
    cv2.namedWindow(win_prev, cv2.WINDOW_NORMAL)
    cv2.imshow(win_prev, preview)
    # 固定窗口宽度，高度按 A4 比例自动收窄（210:297 ≈ 1:1.414）
    cv2.resizeWindow(win_prev, 420, 594)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("s"):
        out = Path(PROJECT_ROOT) / "calib_data" / "solve_preview.png"
        cv2.imwrite(str(out), preview)
        print(f"[保存] {out}")
    cv2.destroyWindow(win_prev)

    # 相机图上叠加结果
    res_vis = _draw_results(frame, pieces, results)
    win3 = "B路求解结果  [任意键关闭]"
    cv2.namedWindow(win3, cv2.WINDOW_NORMAL)
    cv2.imshow(win3, res_vis)
    cv2.waitKey(0)
    cv2.destroyWindow(win3)


def main():
    ap = argparse.ArgumentParser(description="B路真实环境测试")
    ap.add_argument("--camera", "-c", type=int, default=0)
    ap.add_argument("--image", "-img", type=str, default=None)
    args = ap.parse_args()

    cfg = get_config()

    H_px2mm = None
    if CALIB_FILE.exists():
        H_px2mm = _load_homography(CALIB_FILE)
        print(f"[OK] 已加载标定: {CALIB_FILE}")
    else:
        print(f"[WARN] 未找到标定文件 {CALIB_FILE}")
        print("       mm 坐标不可用，求解将跳过（需先运行 calibrate_aruco.py）")

    # 场景选择
    print()
    while True:
        raw = input("场景类型  [1=白纸  2=扑克]: ").strip()
        if raw == '1':
            scene = 'white'
            break
        if raw == '2':
            scene = 'card'
            break
        print("  请输入 1 或 2")
    print(f"[场景] {scene}")

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"ERROR: 无法读取 {args.image}")
            return 1
        run_once(frame, cfg, H_px2mm, scene)
        return 0

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
    print("  B路实测  把白色碎片打散放到绿纸上")
    print("  SPACE=拍照并求解  D=查看 mask  Q=退出")
    print("=" * 55)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy()
        calib_str = "calib=OK" if H_px2mm is not None else "calib=NO"
        cv2.putText(vis, f"{calib_str}  SPACE=拍照  D=mask  Q=退出",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW, vis)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        if key == ord("d"):
            ok2, dbg_frame = cap.read()
            dbg: list = []
            detect_all_pieces(dbg_frame if ok2 else frame, cfg, H_px2mm, dbg, scene=scene)
            if dbg:
                win = "mask  [任意键关闭]"
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                cv2.imshow(win, dbg[0])
                cv2.waitKey(0)
                cv2.destroyWindow(win)

        if key == ord(" "):
            cap.release()
            cv2.destroyWindow(WINDOW)
            run_once(frame, cfg, H_px2mm, scene)
            cap = cv2.VideoCapture(args.camera)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            print("\n继续按 SPACE 重新拍照，Q 退出")

    if cap and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
