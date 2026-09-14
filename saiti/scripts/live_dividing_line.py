#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/live_dividing_line.py — 摄像头实时竖分界线检测

打开摄像头，实时检测 A4 纸中部黑色竖线：
  左侧 = 拼图区 (蓝)，右侧 = 取料区/纸片区 (绿)

使用：
    python scripts/live_dividing_line.py
    python scripts/live_dividing_line.py -c 1
    python scripts/live_dividing_line.py --save      # 按 S 保存当前帧
"""

import argparse
import os
import sys
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from vision.config import get_config
from vision.detection import find_paper_and_rectify, detect_dividing_line


def main():
    parser = argparse.ArgumentParser(description="摄像头实时竖分界线检测")
    parser.add_argument("--camera", "-c", type=int, default=0)
    parser.add_argument("--save", action="store_true", help="按 S 保存当前帧")
    args = parser.parse_args()

    cfg = get_config()

    # 尝试多个后端打开摄像头
    cap: Optional[cv2.VideoCapture] = None
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(args.camera, backend)
        if cap.isOpened():
            break
        cap.release()
        cap = None
    if cap is None:
        print(f"ERROR: 摄像头 {args.camera} 无法打开")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    ok, _ = cap.read()
    if not ok:
        print(f"ERROR: 摄像头 {args.camera} 无法打开")
        return 1

    print("=== 实时竖分界线检测 ===")
    print("  左侧 = 拼图区 | 右侧 = 取料区")
    print("  [Q] 退出")
    if args.save:
        print("  [S] 保存当前帧")
    print()

    # EMA 平滑
    separator_x: int = 640
    separator_smooth: float = 640.0
    confidence: float = 0.0
    alpha_base: float = 0.4

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        fh, fw = frame.shape[:2]

        # ---- 纸面定位 ----
        paper_conf = 0.0
        H = None
        try:
            rectified, H, paper_conf = find_paper_and_rectify(frame, cfg)
        except Exception:
            rectified = None

        # ---- 竖分界线检测 ----
        if rectified is not None:
            try:
                raw_x, confidence, debug_info = detect_dividing_line(rectified, cfg, debug=True)
                alpha = alpha_base * confidence
                separator_smooth = alpha * raw_x + (1 - alpha) * separator_smooth
                separator_x = int(round(separator_smooth))
            except Exception:
                pass  # 保持上一帧

        # ---- 可视化 ----
        vis = frame.copy()

        if rectified is not None and H is not None:
            rect_h, rect_w = rectified.shape[:2]

            try:
                H_inv = np.linalg.inv(H)

                paper_src = np.array([
                    [[0, 0]],
                    [[rect_w - 1, 0]],
                    [[rect_w - 1, rect_h - 1]],
                    [[0, rect_h - 1]],
                ], dtype=np.float32)
                raw_corners = cv2.perspectiveTransform(paper_src, H_inv).reshape(4, 2)
                paper_orig = raw_corners.astype(np.int32)

                sx = float(separator_x)
                top_pt = np.array([[[sx, 0.0]]], dtype=np.float32)
                bot_pt = np.array([[[sx, float(rect_h - 1)]]], dtype=np.float32)
                top_orig = cv2.perspectiveTransform(top_pt, H_inv)[0, 0]
                bot_orig = cv2.perspectiveTransform(bot_pt, H_inv)[0, 0]
                tx, ty = int(top_orig[0]), int(top_orig[1])
                bx, by = int(bot_orig[0]), int(bot_orig[1])

                cv2.line(vis, (tx, ty), (bx, by), (0, 255, 255), 2)

                # 左半区 = 拼图区 (蓝)
                overlay = vis.copy()
                left_pts = np.array([
                    paper_orig[0], (tx, ty), (bx, by), paper_orig[3],
                ], dtype=np.int32)
                cv2.fillPoly(overlay, [left_pts], (180, 80, 0))
                cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)

                # 右半区 = 取料区 (绿)
                overlay2 = vis.copy()
                right_pts = np.array([
                    (tx, ty), paper_orig[1], paper_orig[2], (bx, by),
                ], dtype=np.int32)
                cv2.fillPoly(overlay2, [right_pts], (0, 180, 0))
                cv2.addWeighted(overlay2, 0.15, vis, 0.85, 0, vis)

                cv2.polylines(vis, [paper_orig], True, (200, 200, 200), 1)

            except Exception:
                pass

        # 文字
        conf_color = (0, 255, 0) if confidence >= 0.5 else (0, 200, 255) if confidence >= 0.3 else (0, 0, 255)
        cv2.putText(vis, f"Line x={separator_x}  conf={confidence:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2, cv2.LINE_AA)
        cv2.putText(vis, "PUZZLE (left)", (10, fh - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 80, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, "MATERIAL (right)", (fw - 270, fh - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2, cv2.LINE_AA)

        if paper_conf < 0.5:
            cv2.putText(vis, "WARN: paper not found", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Dividing Line - Live [Q=Quit]", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s") and args.save:
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(cfg.calib_dir, f"live_{ts}.jpg")
            os.makedirs(cfg.calib_dir, exist_ok=True)
            cv2.imwrite(path, vis)
            print(f"  已保存: {path}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
