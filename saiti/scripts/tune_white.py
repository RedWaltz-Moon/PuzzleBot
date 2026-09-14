#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/tune_white.py — 白色碎片调参（边缘 + 凸度过滤）

4 个参数：
  canny_lo   Canny 低阈值（越小边越多）
  canny_hi   Canny 高阈值（越大只保留强边）
  close_kern 闭运算核（连接断边、填内孔）
  solidity   凸度下限（反光 < 0.5，白纸 > 0.75）

右图每个保留轮廓旁显示其 solidity 值，帮助判断阈值。

键位：  P=暂停/继续   S=打印参数   Q=退出
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.config import get_config

WINDOW = "tune_white  [P=暂停  S=保存  Q=退出]"


def _segment(frame, canny_lo, canny_hi, close_ks, sol_min, min_pct):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(blur)

    edges = cv2.Canny(eq, canny_lo, canny_hi)
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kd, iterations=2)
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kc)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = frame.shape[0] * frame.shape[1] * min_pct / 100.0
    kept, rejected = [], []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area:
            continue
        hull_area = float(cv2.contourArea(cv2.convexHull(c)))
        sol = area / hull_area if hull_area > 0 else 0.0
        if sol >= sol_min:
            kept.append((c, sol))
        else:
            rejected.append((c, sol))

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for c, _ in kept:
        cv2.drawContours(mask, [c], -1, 255, cv2.FILLED)

    return mask, kept, rejected


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
        print("ERROR: 摄像头无法打开"); return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("canny_lo",   WINDOW, cfg.white_canny_lo,              200, lambda _: None)
    cv2.createTrackbar("canny_hi",   WINDOW, cfg.white_canny_hi,              400, lambda _: None)
    cv2.createTrackbar("close_kern", WINDOW, cfg.white_close_kernel,           99, lambda _: None)
    cv2.createTrackbar("solid_x100", WINDOW, int(cfg.white_min_solidity * 100), 99, lambda _: None)

    paused = False
    frozen = None
    print("P=暂停/继续  S=打印参数  Q=退出")

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frozen = frame

        frame = frozen
        if frame is None:
            continue

        canny_lo = cv2.getTrackbarPos("canny_lo",   WINDOW)
        canny_hi = cv2.getTrackbarPos("canny_hi",   WINDOW)
        close_ks = cv2.getTrackbarPos("close_kern", WINDOW) | 1
        sol_min  = cv2.getTrackbarPos("solid_x100", WINDOW) / 100.0
        # min_area 固定 0.1%，用 solidity 区分反光
        min_pct  = 0.1

        mask, kept, rejected = _segment(frame, canny_lo, canny_hi,
                                         close_ks, sol_min, min_pct)

        right = frame.copy()

        # 被拒绝的轮廓（反光等）画红色
        for c, sol in rejected:
            cv2.drawContours(right, [c], -1, (0, 0, 180), 1)

        # 保留的轮廓画绿色 + solidity 标注
        for c, sol in kept:
            cv2.drawContours(right, [c], -1, (0, 220, 60), 2)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                label = f"{sol:.2f}"
                cv2.putText(right, label, (cx - 15, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(right, label, (cx - 15, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 1, cv2.LINE_AA)

        # mask 半透明叠加
        overlay = right.copy()
        overlay[mask > 0] = (0, 200, 60)
        right = cv2.addWeighted(overlay, 0.35, right, 0.65, 0)

        info = (f"kept={len(kept)}  rejected={len(rejected)}  "
                f"canny={canny_lo}/{canny_hi}  close={close_ks}  sol>={sol_min:.2f}")
        for img in (frame, right):
            cv2.putText(img, info, (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, info, (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (255, 255, 255), 1, cv2.LINE_AA)

        if paused:
            cv2.putText(right, "PAUSED", (8, 58), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 200, 255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW, np.hstack([frame, right]))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("p"):
            paused = not paused
        elif key == ord("s"):
            print(f"\n    white_canny_lo: int = {canny_lo}")
            print(f"    white_canny_hi: int = {canny_hi}")
            print(f"    white_close_kernel: int = {close_ks}")
            print(f"    white_min_solidity: float = {sol_min:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
