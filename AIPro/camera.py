#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera.py — USB 摄像头捕获（嵌入式适配版）

适配香橙派：
  - 移除 Windows 专用后端 (CAP_DSHOW, CAP_MSMF)
  - 使用 CAP_V4L2 (Linux V4L2) 或 CAP_ANY
  - 保留多帧平均降噪
  - index < 0 时自动检测第一个可用摄像头
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import get_config, Config


def _try_open_one(index: int, width: int = 1280, height: int = 720, fps: int = 30):
    """尝试打开并预热一个摄像头，成功返回 (cap, actual_w, actual_h)，失败返回 None。"""
    for backend in (cv2.CAP_V4L2, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        # 预热：读到有效帧才算成功
        for _ in range(30):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                return cap, actual_w, actual_h
            time.sleep(0.1)
        cap.release()
    return None


def auto_detect(width: int = 1280, height: int = 720, fps: int = 30):
    """自动检测第一个可用的摄像头。返回 (index, actual_w, actual_h)，失败抛异常。"""
    results = []
    for i in range(10):
        r = _try_open_one(i, width, height, fps)
        if r is not None:
            cap, aw, ah = r
            cap.release()
            results.append((i, aw, ah))
            print(f"[CAM] 自动检测: video{i} OK {aw}x{ah}")
    if not results:
        raise RuntimeError("自动检测失败：未找到可用的摄像头 (0-9)")
    return results[0]


def list_cameras(max_index: int = 5) -> List[int]:
    """枚举可用摄像头索引。"""
    available = []
    for i in range(max_index):
        for backend in (cv2.CAP_V4L2, cv2.CAP_ANY):
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    available.append(i)
                    cap.release()
                    break
            cap.release()
    return available


class Camera:
    """USB 摄像头封装。index < 0 时自动检测。"""

    def __init__(
        self,
        index: int = -1,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def resolution(self) -> Tuple[int, int]:
        """实际分辨率 (w, h)。"""
        if self._cap is None or not self._cap.isOpened():
            return (self.width, self.height)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            return (self.width, self.height)
        return (w, h)

    def open(self) -> None:
        """打开摄像头。index < 0 时自动检测。"""
        if self.index < 0:
            # 自动检测
            for i in range(10):
                r = _try_open_one(i, self.width, self.height, self.fps)
                if r is not None:
                    self._cap, aw, ah = r
                    self.index = i
                    break
            if self._cap is None:
                raise RuntimeError("自动检测失败：未找到可用的摄像头")
        else:
            r = _try_open_one(self.index, self.width, self.height, self.fps)
            if r is not None:
                self._cap, aw, ah = r
            else:
                raise RuntimeError(f"摄像头 {self.index} 打开/预热失败")

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[CAM] 摄像头 {self.index} 已打开: {actual_w}x{actual_h}")

    def close(self) -> None:
        """关闭摄像头。"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """读取一帧。返回 (ok, frame_bgr)。"""
        if self._cap is None:
            return False, None
        return self._cap.read()

    def capture(self) -> np.ndarray:
        """捕获一帧 BGR 图像。"""
        ok, frame = self.read()
        if not ok or frame is None:
            raise RuntimeError(f"摄像头 {self.index} 读取失败")
        return frame

    def capture_averaged(self, n_frames: int = 5) -> np.ndarray:
        """多帧平均降噪捕获。先清空缓冲区，再读；全失败则单帧兜底。"""
        # 清空 V4L2 缓冲区
        for _ in range(5):
            self._cap.grab()
        time.sleep(0.1)

        frames = []
        for _ in range(max(n_frames, 5)):
            ok, frame = self._cap.read()
            if ok and frame is not None and frame.size > 0:
                frames.append(frame.astype(np.float32))
            time.sleep(0.05)
            if len(frames) >= n_frames:
                break
        if frames:
            avg = np.mean(frames, axis=0)
            return np.clip(avg, 0, 255).astype(np.uint8)
        # 单帧兜底
        for _ in range(10):
            ok, frame = self._cap.read()
            if ok and frame is not None and frame.size > 0:
                print("[CAM] 多帧平均失败，使用单帧")
                return frame
            time.sleep(0.1)
        raise RuntimeError(f"摄像头 {self.index} 读取失败")
