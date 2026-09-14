#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/show_card_mask.py — 显示绿底减除后的扑克牌闭运算 mask

用法：
    python scripts/show_card_mask.py --image photo.jpg
    python scripts/show_card_mask.py -c 0        # 摄像头，SPACE 拍照
"""
import argparse
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from vision.config import get_config
from vision.detection import find_paper_and_rectify, detect_dividing_line


def show_mask(frame):
    cfg = get_config()

    # 1. 矫正
    result = find_paper_and_rectify(frame, cfg)
    if result is None or result[0] is None:
        print("[ERR] 找不到绿色纸面")
        return

    rect_bgr, _, _ = result
    h, w = rect_bgr.shape[:2]

    # 2. 分界线 → 右半 ROI
    result_sep = detect_dividing_line(rect_bgr, cfg)
    sep_x = int(result_sep[0]) if result_sep is not None else w // 2
    roi = rect_bgr[:, sep_x:]

    # 3. 绿底减除
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(
        hsv,
        (cfg.paper_green_h_lo, cfg.paper_green_s_min, cfg.paper_green_v_min),
        (cfg.paper_green_h_hi, 255, 255),
    )
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green_mask = cv2.dilate(green_mask, k_dil, iterations=1)
    fg = cv2.bitwise_not(green_mask)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_open)

    k_close_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg_before = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close_small)

    # 4. 扑克牌额外闭运算（连接边缘反光小孔）
    k_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg_after = cv2.morphologyEx(fg_before, cv2.MORPH_CLOSE, k_fill)

    # 5. 轮廓叠加到原图
    overlay = roi.copy()
    cnts, _ = cv2.findContours(fg_after, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (0, 255, 0), 2)
    for i, c in enumerate(cnts):
        m = cv2.moments(c)
        if m["m00"] > 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            cv2.putText(overlay, str(i + 1), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # 6. 拼四格显示
    def to3(img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img

    top    = np.hstack([to3(green_mask), to3(fg_before)])
    bottom = np.hstack([to3(fg_after),   overlay])
    grid   = np.vstack([top, bottom])

    # 缩放到合理显示大小
    gh, gw = grid.shape[:2]
    scale = min(1.0, 1200 / gw, 900 / gh)
    if scale < 1.0:
        grid = cv2.resize(grid, (int(gw * scale), int(gh * scale)))

    # 标签
    labels = ["绿色mask", "开运算后fg", "闭运算后(扑克)", "轮廓叠加"]
    hh, ww = grid.shape[:2]
    qw, qh = ww // 2, hh // 2
    for idx, txt in enumerate(labels):
        ox = (idx % 2) * qw + 4
        oy = (idx // 2) * qh + 20
        cv2.putText(grid, txt, (ox, oy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Card Mask Debug  [S=保存  Q=退出]", grid)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("s"):
        out = os.path.join(PROJECT_ROOT, "calib_data", "card_mask_debug.png")
        cv2.imwrite(out, grid)
        print(f"[保存] {out}")
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", "-img", default=None)
    ap.add_argument("--camera", "-c",  type=int, default=0)
    args = ap.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"[ERR] 无法读取 {args.image}")
            return
        show_mask(frame)
        return

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    win = "预览  [SPACE=拍照  Q=退出]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("按 SPACE 拍照，Q 退出")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            cv2.destroyWindow(win)
            cap.release()
            show_mask(frame)
            return

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
