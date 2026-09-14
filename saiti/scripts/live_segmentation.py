#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/live_segmentation.py — 摄像头实时碎片分割测试

三策略融合分割：
  颜色碎片 → HSV 饱和度
  白色碎片 → 背景差分 + Canny 边缘 + 连通域
  扑克牌碎片 → 饱和度 + 纹理梯度 + 保留彩色纹理图

使用：
    python scripts/live_segmentation.py
    python scripts/live_segmentation.py -c 1
    python scripts/live_segmentation.py --save      # 按 S 保存当前帧
"""

import argparse
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from vision.config import get_config
from vision.detection import (
    find_paper_and_rectify,
    detect_dividing_line,
    detect_pieces,
    draw_pieces,
)

# 类型缩写
TYPE_SHORT = {"color": "C", "white": "W", "card": "K", "unknown": "?"}
TYPE_COLOR = {
    "color": (0, 255, 0),      # 绿
    "white": (255, 255, 0),    # 青
    "card": (255, 0, 255),     # 品红
    "unknown": (0, 0, 255),    # 红
}


def main():
    parser = argparse.ArgumentParser(description="摄像头实时碎片分割测试")
    parser.add_argument("--camera", "-c", type=int, default=0)
    parser.add_argument("--save", action="store_true", help="按 S 保存当前帧")
    args = parser.parse_args()

    cfg = get_config()

    # 尝试多个后端打开摄像头
    cap: cv2.VideoCapture | None = None
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

    print("=== 实时碎片分割 (三策略融合) ===")
    print("  C=颜色碎片(HSV)  W=白色碎片(背景差分)  K=扑克牌(纹理)")
    print("  [Q] 退出")
    if args.save:
        print("  [S] 保存当前帧")
    print()

    # EMA 平滑（分界线）
    separator_smooth: float = 420.0
    separator_x: int = 420
    line_conf: float = 0.0
    paper_conf: float = 0.0
    alpha_base: float = 0.4

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        fh, fw = frame.shape[:2]

        # ---- 纸面定位 + 分界线 ----
        try:
            rectified, H, paper_conf = find_paper_and_rectify(frame, cfg)
        except Exception:
            rectified = None
            H = None
            paper_conf = 0.0

        if rectified is not None:
            try:
                raw_x, line_conf, _ = detect_dividing_line(rectified, cfg, debug=False)
                alpha = alpha_base * line_conf
                separator_smooth = alpha * raw_x + (1 - alpha) * separator_smooth
                separator_x = int(round(separator_smooth))
            except Exception:
                separator_x = int(round(separator_smooth))
        else:
            separator_x = int(round(separator_smooth))

        # ---- 碎片检测 ----
        pieces = []
        if rectified is not None:
            try:
                pieces = detect_pieces(frame, cfg)
            except Exception:
                pass

        # ---- 可视化 ----
        vis = frame.copy()

        if rectified is not None and H is not None:
            rect_h, rect_w = rectified.shape[:2]

            try:
                H_inv = np.linalg.inv(H)

                # 纸面四角
                paper_src = np.array([
                    [[0, 0]],
                    [[rect_w - 1, 0]],
                    [[rect_w - 1, rect_h - 1]],
                    [[0, rect_h - 1]],
                ], dtype=np.float32)
                paper_orig = cv2.perspectiveTransform(paper_src, H_inv).reshape(4, 2).astype(np.int32)

                # 分界线
                sx = float(separator_x)
                top_pt = np.array([[[sx, 0.0]]], dtype=np.float32)
                bot_pt = np.array([[[sx, float(rect_h - 1)]]], dtype=np.float32)
                top_orig = cv2.perspectiveTransform(top_pt, H_inv)[0, 0]
                bot_orig = cv2.perspectiveTransform(bot_pt, H_inv)[0, 0]
                tx, ty = int(top_orig[0]), int(top_orig[1])
                bx, by = int(bot_orig[0]), int(bot_orig[1])

                # 分界线
                cv2.line(vis, (tx, ty), (bx, by), (0, 255, 255), 2)

                # 左半区 = 拼图区 (蓝)
                overlay = vis.copy()
                left_pts = np.array([
                    paper_orig[0], (tx, ty), (bx, by), paper_orig[3],
                ], dtype=np.int32)
                cv2.fillPoly(overlay, [left_pts], (180, 80, 0))
                cv2.addWeighted(overlay, 0.12, vis, 0.88, 0, vis)

                # 右半区 = 取料区 (绿)
                overlay2 = vis.copy()
                right_pts = np.array([
                    (tx, ty), paper_orig[1], paper_orig[2], (bx, by),
                ], dtype=np.int32)
                cv2.fillPoly(overlay2, [right_pts], (0, 180, 0))
                cv2.addWeighted(overlay2, 0.12, vis, 0.88, 0, vis)

                cv2.polylines(vis, [paper_orig], True, (200, 200, 200), 1)

                # ---- 碎片轮廓映射到原始画面 ----
                if pieces:
                    for piece in pieces:
                        color = TYPE_COLOR.get(piece.piece_type, (0, 0, 255))

                        # 将轮廓点从矫正图映射回原始图
                        contour_pts = piece.contour.reshape(-1, 2).astype(np.float32)
                        contour_3d = np.hstack([
                            contour_pts,
                            np.ones((contour_pts.shape[0], 1), dtype=np.float32),
                        ])
                        mapped = (H_inv @ contour_3d.T).T
                        mapped[:, :2] /= mapped[:, 2:3]
                        mapped_pts = mapped[:, :2].reshape(-1, 1, 2).astype(np.int32)

                        cv2.polylines(vis, [mapped_pts], True, color, 2)

                        # 中心点映射
                        cx_px, cy_px = piece.center_px
                        pt_3d = np.array([[cx_px, cy_px, 1.0]], dtype=np.float32).T
                        mapped_center = (H_inv @ pt_3d).T
                        mapped_center[:, :2] /= mapped_center[:, 2:3]
                        mcx, mcy = int(mapped_center[0, 0]), int(mapped_center[0, 1])

                        cv2.circle(vis, (mcx, mcy), 5, color, -1)

                        tlabel = TYPE_SHORT.get(piece.piece_type, "?")
                        label = f"P{piece.index}[{tlabel}]"
                        cv2.putText(vis, label, (mcx + 8, mcy - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)

            except Exception:
                pass

        # ---- HUD ----
        # 分界线信息
        conf_color = (0, 255, 0) if line_conf >= 0.5 else (0, 200, 255) if line_conf >= 0.3 else (0, 0, 255)
        cv2.putText(vis, f"Line x={separator_x}  conf={line_conf:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2, cv2.LINE_AA)

        # 碎片统计
        type_counts = {"color": 0, "white": 0, "card": 0, "unknown": 0}
        for p in pieces:
            type_counts[p.piece_type] = type_counts.get(p.piece_type, 0) + 1

        stats_text = f"Pieces: {len(pieces)} "
        parts = []
        for t in ("color", "white", "card"):
            if type_counts[t] > 0:
                parts.append(f"{TYPE_SHORT[t]}x{type_counts[t]}")
        stats_text += "  ".join(parts)
        if type_counts["unknown"] > 0:
            stats_text += f"  ?x{type_counts['unknown']}"

        cv2.putText(vis, stats_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # 纸面状态
        if paper_conf < 0.5:
            cv2.putText(vis, "WARN: paper not found", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        # 图例
        legend_y = fh - 80
        for i, (t, name) in enumerate([("color", "Color(HSV)"), ("white", "White(BGSub)"), ("card", "Card(Tex)")]):
            color = TYPE_COLOR[t]
            lx = 10 + i * 185
            cv2.rectangle(vis, (lx, legend_y), (lx + 14, legend_y + 14), color, -1)
            cv2.putText(vis, f"{TYPE_SHORT[t]}={name}", (lx + 18, legend_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        cv2.putText(vis, "PUZZLE (left)", (10, fh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 80, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, "MATERIAL (right)", (fw - 270, fh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2, cv2.LINE_AA)

        cv2.imshow("Piece Segmentation - Live [Q=Quit, S=Save]", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s") and args.save:
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(cfg.calib_dir, f"seg_live_{ts}.jpg")
            os.makedirs(cfg.calib_dir, exist_ok=True)
            cv2.imwrite(path, vis)
            print(f"  已保存: {path}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
