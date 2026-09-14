#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/tune_green_bg.py — 碎片检测调参

左：原图 + 检测到的轮廓（绿色=碎片，红=纸面轮廓）
右：碎片 mask（白=碎片，黑=背景）

滑块：
  H_lo / H_hi  — 绿色纸面色调范围
  S_min        — 白色碎片饱和度上限（调高=抓更多彩色）
  V_min        — 白色碎片亮度下限（调低=抓更暗的碎片）

按 P 暂停，S 打印当前参数，Q 退出。

用法：
    python scripts/tune_green_bg.py
    python scripts/tune_green_bg.py -c 1
"""

import argparse
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from vision.config import get_config

WINDOW = "tune  [P=暂停  S=打印参数  Q=退出]"


def process(frame, cfg, h_lo, h_hi, s_max, v_min):
    fh, fw = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 1. 绿色纸面定位（大核填孔，只求纸面轮廓）
    green = cv2.inRange(hsv,
        np.array([h_lo, cfg.paper_green_s_min, cfg.paper_green_v_min]),
        np.array([h_hi, 255, 255]))
    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    k_sml = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green_f = cv2.morphologyEx(green, cv2.MORPH_CLOSE, k_big, iterations=4)
    green_f = cv2.morphologyEx(green_f, cv2.MORPH_OPEN,  k_sml, iterations=1)

    paper_cnts, _ = cv2.findContours(green_f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paper_mask = np.zeros((fh, fw), dtype=np.uint8)
    paper_cnt = None
    if paper_cnts:
        paper_cnt = max(paper_cnts, key=cv2.contourArea)
        cv2.drawContours(paper_mask, [paper_cnt], -1, 255, cv2.FILLED)

    # 2. 纸面内白色碎片
    white = cv2.inRange(hsv,
        np.array([0,   0,     v_min]),
        np.array([179, s_max, 255]))
    piece_mask = cv2.bitwise_and(white, paper_mask)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    piece_mask = cv2.morphologyEx(piece_mask, cv2.MORPH_CLOSE, k3, iterations=2)

    # 3. 找碎片轮廓（面积过滤）
    min_area = fh * fw * 0.001
    max_area = fh * fw * 0.25
    raw_cnts, _ = cv2.findContours(piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    piece_cnts = [c for c in raw_cnts
                  if min_area <= cv2.contourArea(c) <= max_area
                  and cv2.contourArea(c) / max(cv2.contourArea(cv2.convexHull(c)), 1) >= 0.6]

    # 4. 可视化
    vis = frame.copy()
    if paper_cnt is not None:
        cv2.drawContours(vis, [paper_cnt], -1, (0, 0, 255), 2)  # 纸面边缘红色
    cv2.drawContours(vis, piece_cnts, -1, (0, 255, 0), 2)       # 碎片轮廓绿色
    for i, c in enumerate(piece_cnts):
        m = cv2.moments(c)
        if m["m00"] > 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            cv2.putText(vis, str(i + 1), (cx - 8, cy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(vis, f"碎片={len(piece_cnts)}  纸面={'OK' if paper_cnt is not None else 'NO'}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if paper_cnt is not None else (0, 0, 255), 2)

    return vis, piece_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", "-c", type=int, default=0)
    args = ap.parse_args()

    cfg = get_config()

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

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 400)

    cv2.createTrackbar("H_lo",  WINDOW, cfg.paper_green_h_lo,  179, lambda x: None)
    cv2.createTrackbar("H_hi",  WINDOW, cfg.paper_green_h_hi,  179, lambda x: None)
    cv2.createTrackbar("S_max", WINDOW, 70,                    255, lambda x: None)
    cv2.createTrackbar("V_min", WINDOW, 160,                   255, lambda x: None)

    frozen = False
    frozen_frame = None

    print("调整滑块，直到左图绿色轮廓准确框出所有碎片")
    print("P=暂停  S=打印参数  Q=退出")

    while True:
        if not frozen:
            ok, frame = cap.read()
            if not ok:
                break
        else:
            frame = frozen_frame.copy()

        h_lo  = cv2.getTrackbarPos("H_lo",  WINDOW)
        h_hi  = cv2.getTrackbarPos("H_hi",  WINDOW)
        s_max = cv2.getTrackbarPos("S_max", WINDOW)
        v_min = cv2.getTrackbarPos("V_min", WINDOW)

        vis, mask = process(frame, cfg, h_lo, h_hi, s_max, v_min)

        fh, fw = frame.shape[:2]
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(mask_bgr, "碎片 mask", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        combined = np.hstack([
            cv2.resize(vis,      (fw // 2, fh // 2)),
            cv2.resize(mask_bgr, (fw // 2, fh // 2)),
        ])
        if frozen:
            cv2.putText(combined, "PAUSED", (10, fh // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imshow(WINDOW, combined)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("p"):
            frozen = not frozen
            if frozen:
                frozen_frame = frame.copy()

        elif key == ord("s"):
            print(f"\n=== 当前参数 ===")
            print(f"paper_green_h_lo  = {h_lo}   # config.py")
            print(f"paper_green_h_hi  = {h_hi}   # config.py")
            print(f"white S_max       = {s_max}  # build_templates.py 第2步")
            print(f"white V_min       = {v_min}  # build_templates.py 第2步")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
