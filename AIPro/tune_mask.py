#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tune_mask.py — 实时调参预览工具：左边原图，右边 mask
用于调优绿色纸面 / 白色碎片 / 扑克卡片 的 HSV 和形态学参数

用法：
    python tune_mask.py                        # 自动检测摄像头
    python tune_mask.py -c 0                   # 指定摄像头
    python tune_mask.py -i test.jpg            # 调试单张图片（无头环境用）
    python tune_mask.py -i test.jpg -w white   # 白纸模式
    python tune_mask.py -i test.jpg -w card    # 扑克模式

键盘:
    W  → 白色碎片 mask 模式
    C  → 扑克卡片 mask 模式
    G  → 只看绿色纸面 mask
    A  → 全部叠加 (BGR)
    1-3 → 存快照到 debug_output/
    Q/ESC → 退出
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# 将当前目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_config
from camera import Camera

# ---------------------------------------------------------------------------
# 默认参数 —— 从 config.py 读取，trackbar 覆盖
# ---------------------------------------------------------------------------

def _make_params(cfg: Config, mode: str = "all") -> dict:
    """构建可调参数表，mode: "white" | "card" | "all"."""
    p = {}

    # ---- 绿色纸面 ----
    p["green_H_lo"]  = cfg.paper_green_h_lo
    p["green_H_hi"]  = cfg.paper_green_h_hi
    p["green_S_min"] = cfg.paper_green_s_min
    p["green_V_min"] = cfg.paper_green_v_min

    # ---- 白色碎片 HSV ----
    p["white_V_min"] = 236
    p["white_V_max"] = 255
    p["white_S_max"] = 78

    # ---- 扑克卡片 ----
    p["card_S_thresh"]  = getattr(cfg, "card_saturation_thresh",  25)
    p["card_G_thresh"]  = getattr(cfg, "card_gradient_thresh",    20)
    p["card_close_k"]   = getattr(cfg, "card_close_kernel",       15)
    p["card_area_pct"]  = 1.0   # 连通域最小面积 % (trackbar: 0.1~5.0)

    # ---- 形态学 ----
    p["morph_close_k"]  = 3     # 白色碎片闭运算核大小
    p["morph_close_n"]  = 2     # 闭运算迭代次数
    p["paper_close_k"]  = 25    # 绿纸闭运算核
    p["paper_close_n"]  = 4

    return p


# ---------------------------------------------------------------------------
# 各层 mask 计算
# ---------------------------------------------------------------------------

def _green_mask(hsv: np.ndarray, p: dict) -> Tuple[np.ndarray, np.ndarray]:
    """绿色纸面 mask."""
    raw = cv2.inRange(
        hsv,
        np.array([p["green_H_lo"], p["green_S_min"], p["green_V_min"]]),
        np.array([p["green_H_hi"], 255, 255]),
    )
    kb = int(p["paper_close_k"])
    ks = max(3, kb // 5)
    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kb, kb))
    k_sml = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    filled = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k_big, iterations=int(p["paper_close_n"]))
    filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, k_sml, iterations=1)
    return raw, filled


def _paper_mask_from_green(green_filled: np.ndarray, fh: int, fw: int) -> np.ndarray:
    """从绿色封闭区域提取 paper_mask（最大轮廓填充）。"""
    contours, _ = cv2.findContours(green_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((fh, fw), dtype=np.uint8)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)
    return mask


def _white_mask(hsv: np.ndarray, paper_mask: np.ndarray, p: dict) -> np.ndarray:
    """白色碎片 mask（纸面内 HSV 白色区域）。"""
    white = cv2.inRange(
        hsv,
        np.array([0, 0, int(p["white_V_min"])]),
        np.array([179, int(p["white_S_max"]), int(p["white_V_max"])]),
    )
    in_paper = cv2.bitwise_and(white, paper_mask)
    ck = max(1, int(p["morph_close_k"]))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (ck, ck))
    filled = cv2.morphologyEx(in_paper, cv2.MORPH_CLOSE, k, iterations=int(p["morph_close_n"]))
    return filled


def _card_mask(roi_bgr: np.ndarray, p: dict) -> np.ndarray:
    """扑克牌碎片 mask（S 通道 + Sobel 梯度）。"""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    s_ch = hsv[:, :, 1]

    _, s_mask = cv2.threshold(s_ch, int(p["card_S_thresh"]), 255, cv2.THRESH_BINARY)

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    grad_mag = np.clip(grad_mag, 0, 255).astype(np.uint8)
    _, grad_mask = cv2.threshold(grad_mag, int(p["card_G_thresh"]), 255, cv2.THRESH_BINARY)

    combined = cv2.bitwise_or(s_mask, grad_mask)

    ck = max(3, int(p["card_close_k"]))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (ck, ck))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)

    # 连通域面积过滤
    area_pct = max(0.1, min(5.0, p["card_area_pct"])) / 100.0
    min_area = int(area_pct * roi_bgr.shape[0] * roi_bgr.shape[1])
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    mask_out = np.zeros_like(combined)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask_out[labels == i] = 255

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_out = cv2.morphologyEx(mask_out, cv2.MORPH_OPEN, k_open)
    return mask_out


# ---------------------------------------------------------------------------
# 构建显示面板
# ---------------------------------------------------------------------------

def build_panel(frame_bgr: np.ndarray, p: dict, mode: str = "all") -> np.ndarray:
    """
    返回 (h, w*2, 3) 面板 —— 左半边原图，右半边 mask 叠加。
    mode: "white" | "card" | "green" | "all"
    """
    h, w = frame_bgr.shape[:2]

    # 1. 绿色纸面
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green_raw, green_filled = _green_mask(hsv, p)
    paper_mask = _paper_mask_from_green(green_filled, h, w)

    # 2. 白色碎片
    white_filled = _white_mask(hsv, paper_mask, p)

    # 3. 扑克卡片（全图 ROI = 纸面区域）
    card_filled = _card_mask(frame_bgr, p)

    # 选择当前模式的 mask（单通道 0/255）
    if mode == "green":
        mask_gray = green_filled
    elif mode == "white":
        mask_gray = white_filled
    elif mode == "card":
        mask_gray = card_filled
    else:  # "all"
        # 白+绿+卡片合并
        mask_gray = np.zeros((h, w), dtype=np.uint8)
        mask_gray[green_filled > 0] = 128
        mask_gray[white_filled > 0] = 255
        mask_gray[card_filled > 0] = 200

    # 右边 = 纯 mask 图（黑白）
    mask_vis = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)

    # 等比例缩放原图和 mask 到一样大小
    panel = np.hstack([frame_bgr, mask_vis])

    # 状态栏
    info_lines = [
        f"Mode: {mode}",
        f"Green: H=[{p['green_H_lo']}..{p['green_H_hi']}] S>={p['green_S_min']} V>={p['green_V_min']}",
        f"White: V=[{p['white_V_min']}..{p['white_V_max']}] S<={p['white_S_max']}  close={p['morph_close_k']}x{p['morph_close_n']}",
        f"Card:  S>={p['card_S_thresh']} G>={p['card_G_thresh']} close={p['card_close_k']} area>={p['card_area_pct']:.1f}%",
        f"Paper: close={p['paper_close_k']}x{p['paper_close_n']}",
        "[W]hite [C]ard [G]reen [A]ll  [1-3]snap  [Q]uit",
    ]
    y0 = panel.shape[0] - 20 * len(info_lines) - 10
    for i, line in enumerate(info_lines):
        cv2.putText(panel, line, (10, y0 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return panel


# ---------------------------------------------------------------------------
# Trackbar 回调（不做任何事，trackbar 的值在循环中手动读取）
# ---------------------------------------------------------------------------

def _noop(_=None):
    pass


def create_trackbars(window: str, p: dict, mode: str = "all"):
    """在窗口中创建调参滑条。"""
    cv2.createTrackbar("green_H_lo",  window, p["green_H_lo"],  180, _noop)
    cv2.createTrackbar("green_H_hi",  window, p["green_H_hi"],  180, _noop)
    cv2.createTrackbar("green_S_min", window, p["green_S_min"], 255, _noop)
    cv2.createTrackbar("green_V_min", window, p["green_V_min"], 255, _noop)

    if mode in ("white", "all"):
        cv2.createTrackbar("white_V_min", window, p["white_V_min"], 255, _noop)
        cv2.createTrackbar("white_V_max", window, p["white_V_max"], 255, _noop)
        cv2.createTrackbar("white_S_max", window, p["white_S_max"], 255, _noop)
        cv2.createTrackbar("morph_close_k", window, p["morph_close_k"], 31, _noop)
        cv2.createTrackbar("morph_close_n", window, p["morph_close_n"], 10, _noop)

    if mode in ("card", "all"):
        cv2.createTrackbar("card_S_thresh", window, p["card_S_thresh"], 255, _noop)
        cv2.createTrackbar("card_G_thresh", window, p["card_G_thresh"], 255, _noop)
        cv2.createTrackbar("card_close_k",  window, p["card_close_k"],  51,  _noop)
        cv2.createTrackbar("card_area_pct", window, int(p["card_area_pct"] * 10), 50, _noop)  # 0.1~5.0%

    cv2.createTrackbar("paper_close_k", window, p["paper_close_k"], 51, _noop)
    cv2.createTrackbar("paper_close_n", window, p["paper_close_n"], 10, _noop)


def read_trackbars(p: dict, mode: str = "all"):
    """从 trackbar 读取当前值。"""
    for key in ("green_H_lo", "green_H_hi", "green_S_min", "green_V_min",
                "paper_close_k", "paper_close_n"):
        if cv2.getTrackbarPos(key, WINDOW) >= 0:
            p[key] = cv2.getTrackbarPos(key, WINDOW)

    if mode in ("white", "all"):
        for key in ("white_V_min", "white_V_max", "white_S_max", "morph_close_k", "morph_close_n"):
            if cv2.getTrackbarPos(key, WINDOW) >= 0:
                p[key] = cv2.getTrackbarPos(key, WINDOW)

    if mode in ("card", "all"):
        for key in ("card_S_thresh", "card_G_thresh", "card_close_k"):
            if cv2.getTrackbarPos(key, WINDOW) >= 0:
                p[key] = cv2.getTrackbarPos(key, WINDOW)
        v = cv2.getTrackbarPos("card_area_pct", WINDOW)
        if v >= 0:
            p["card_area_pct"] = v / 10.0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

WINDOW = "Mask Tune  [L=原图 | R=mask]"


def main():
    ap = argparse.ArgumentParser(description="Mask 实时调参预览工具")
    ap.add_argument("--camera", "-c", type=int, default=-1, help="摄像头索引 (<0 = 自动)")
    ap.add_argument("--image", "-i", type=str, default=None, help="单张图片调试 (忽略摄像头)")
    ap.add_argument("--which", "-w", type=str, choices=["white","card","green","all"],
                    default="all", help="调参模式 (默认 all)")
    args = ap.parse_args()

    cfg = get_config()
    p = _make_params(cfg, args.which)
    mode = args.which

    # ---- 单图模式 ----
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"ERROR: 无法读取图片 {args.image}")
            return 1
        print(f"[IMG] {args.image}  {frame.shape[1]}x{frame.shape[0]}")
        panel = build_panel(frame, p, mode)
        cv2.imwrite("debug_output/tune_mask_snapshot.jpg", panel)
        print("  → 已保存 debug_output/tune_mask_snapshot.jpg")
        return 0

    # ---- 摄像头模式 ----
    cam = Camera(index=args.camera, width=1280, height=720, fps=30)
    cam.open()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    create_trackbars(WINDOW, p, mode)

    snap_counter = 0
    while True:
        ok, frame = cam.read()
        if not ok:
            continue

        read_trackbars(p, mode)
        panel = build_panel(frame, p, mode)
        cv2.imshow(WINDOW, panel)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("w"):
            mode = "white"
            cv2.destroyWindow(WINDOW)
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            create_trackbars(WINDOW, p, mode)
        elif key == ord("c"):
            mode = "card"
            cv2.destroyWindow(WINDOW)
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            create_trackbars(WINDOW, p, mode)
        elif key == ord("g"):
            mode = "green"
            cv2.destroyWindow(WINDOW)
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            create_trackbars(WINDOW, p, mode)
        elif key == ord("a"):
            mode = "all"
            cv2.destroyWindow(WINDOW)
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            create_trackbars(WINDOW, p, mode)
        elif key in (ord("1"), ord("2"), ord("3")):
            snap_counter += 1
            Path("debug_output").mkdir(exist_ok=True)
            fname = f"tune_{mode}_s{snap_counter:02d}.jpg"
            cv2.imwrite(f"debug_output/{fname}", panel)
            print(f"  [SAVE] debug_output/{fname}")

    cam.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
