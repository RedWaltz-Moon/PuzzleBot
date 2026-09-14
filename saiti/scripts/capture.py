#!/usr/bin/env python3
"""
scripts/capture.py — PC 拍照工具

打开摄像头，预览画面，按空格保存当前帧，方便离线调试标定。

使用：
    python scripts/capture.py                  # 默认摄像头
    python scripts/capture.py --camera 1       # 指定摄像头
    python scripts/capture.py --output test.jpg # 指定输出文件名
    python scripts/capture.py --no-average     # 单帧采集（不用多帧平均）

快捷键：
    SPACE  — 拍照保存
    ESC/Q  — 退出
"""

import argparse
import logging
import os
import sys

import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from vision.camera import Camera, list_cameras
from vision.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("capture")


def main():
    parser = argparse.ArgumentParser(description="PC 拍照工具 — 用于离线标定调试")
    parser.add_argument("--camera", "-c", type=int, default=0, help="摄像头索引")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出路径")
    parser.add_argument("--no-average", action="store_true", help="不使用多帧平均")
    args = parser.parse_args()

    # 检查摄像头
    cams = list_cameras()
    if not cams:
        print("未检测到摄像头！")
        return
    if args.camera not in cams:
        print(f"摄像头 index={args.camera} 不可用，可用: {cams}")
        args.camera = cams[0]
        print(f"改用 index={args.camera}")

    # 确定输出目录
    cfg = get_config()
    shot_dir = os.path.join(PROJECT_ROOT, cfg.calib_dir)
    os.makedirs(shot_dir, exist_ok=True)

    shot_count = 0

    print()
    print("=" * 50)
    print("  PC 拍照工具")
    print("=" * 50)
    print()
    print(f"  摄像头: index={args.camera}")
    print(f"  输出目录: {shot_dir}")
    print(f"  多帧平均: {'关闭' if args.no_average else '5帧平均'}")
    print()
    print("  快捷键:")
    print("    SPACE — 拍照保存")
    print("    ESC   — 退出")
    print()

    with Camera(index=args.camera) as cam:
        print(f"  实际分辨率: {cam.resolution}")
        print()

        while True:
            # 实时取一帧用于预览
            frame = cam.capture()
            display = frame.copy()

            # 显示提示文字
            cv2.putText(display, "SPACE: Capture | ESC: Quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Saved: {shot_count}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            cv2.imshow("Capture Tool", display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord("q"):  # ESC or Q
                break
            elif key == 32:  # SPACE
                # 拍照：用多帧平均降噪
                if args.no_average:
                    saved = frame
                else:
                    saved = cam.capture_averaged()

                # 生成文件名
                if args.output and shot_count == 0:
                    path = os.path.join(shot_dir, args.output)
                else:
                    import time
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(shot_dir, f"capture_{ts}_{shot_count:03d}.jpg")

                cv2.imwrite(path, saved)
                shot_count += 1
                print(f"  [{shot_count}] 已保存: {path}")

    cv2.destroyAllWindows()
    print(f"\n共拍摄 {shot_count} 张照片，保存在 {shot_dir}/")


if __name__ == "__main__":
    main()
