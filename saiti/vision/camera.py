#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vision/camera.py — 摄像头捕获

镜头无畸变，不做畸变校正。
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np


def list_cameras(max_index: int = 5) -> List[int]:
    """枚举可用摄像头索引"""
    available = []
    for i in range(max_index):
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                # 尝试读一帧确认真的可用
                ok, _ = cap.read()
                if ok:
                    available.append(i)
                    cap.release()
                    break
            cap.release()
    return available


class Camera:
    """USB 摄像头封装。

    支持多帧平均以减少噪声。
    镜头无畸变，不处理畸变校正。
    """

    def __init__(
        self,
        index: int = 0,
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
        """实际分辨率 (w, h)"""
        if self._cap is None or not self._cap.isOpened():
            return (self.width, self.height)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            return (self.width, self.height)
        return (w, h)

    def open(self) -> None:
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            self._cap = cv2.VideoCapture(self.index, backend)
            if self._cap.isOpened():
                break
            self._cap.release()
            self._cap = None

        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 index={self.index}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        # 预热：丢弃前几帧
        for _ in range(5):
            self._cap.read()

    def close(self) -> None:
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
        """多帧平均降噪捕获。"""
        frames = []
        for _ in range(n_frames):
            ok, frame = self.read()
            if ok and frame is not None:
                frames.append(frame.astype(np.float32))
            time.sleep(0.02)
        if not frames:
            raise RuntimeError(f"摄像头 {self.index} 多帧捕获失败")
        avg = np.mean(frames, axis=0)
        return np.clip(avg, 0, 255).astype(np.uint8)
