#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py — 视觉子系统全局配置（嵌入式精简版）

从 saiti/vision/config.py 精简适配，移除 dataclass 依赖。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class Config:
    """视觉子系统配置"""

    # ---- 纸面检测（绿色纸面 HSV）----
    paper_green_h_lo: int = 40
    paper_green_h_hi: int = 88
    paper_green_s_min: int = 60
    paper_green_v_min: int = 40

    # ---- A4 纸面 ----
    a4_width: float = 210.0          # mm
    a4_height: float = 297.0         # mm
    px_per_mm: float = 4.0           # 矫正后图像分辨率 (px/mm)
    rectified_w: int = 840           # 210 * 4
    rectified_h: int = 1188          # 297 * 4

    # ---- 分界线检测 ----
    separator_search_ymin: float = 0.40
    separator_search_ymax: float = 0.60
    separator_search_xmin: float = 0.40
    separator_search_xmax: float = 0.60
    separator_dark_ratio: float = 0.15
    separator_contrast_min: float = 8.0
    separator_gradient_min: float = 3.0

    # ---- 白色碎片分割 ----
    white_canny_lo: int = 30
    white_canny_hi: int = 90
    white_close_kernel: int = 15
    white_min_solidity: float = 0.60
    white_min_area_ratio: float = 0.001

    # ---- 颜色碎片 HSV 分割 ----
    hsv_saturation_thresh: int = 35
    hsv_value_min: int = 30
    hsv_value_max: int = 255
    hsv_open_kernel: int = 3
    hsv_close_kernel: int = 9

    # ---- 扑克牌碎片分割 ----
    card_saturation_thresh: int = 25
    card_bright_thresh: int = 120
    card_gradient_thresh: int = 20
    card_close_kernel: int = 15
    card_texture_std_min: float = 12.0

    # ---- 碎片类型分类 ----
    white_value_min: int = 160             # 白纸亮度下限 (V通道)
    white_saturation_max: int = 40         # 白纸饱和度上限 (S通道)
    white_std_s_max: float = 20.0          # 白纸饱和度标准差上限（纹理均匀性）

    # ---- 扑克牌伪顶点净化 ----
    card_artifact_min_mm: float = 13.0           # detection.py px 空间粗筛阈值(mm)
    card_artifact_min_edge_mm: float = 35.0      # b_solver mm 空间精清阈值(mm)
    card_artifact_min_turn_deg: float = 40.0     # 近共线伪顶点转角上限(°)；真实切割角≈90°
    card_artifact_max_dev_mm: float = 8.0        # Pass2 偏移保护阈值(mm)；顶点偏移≥此值视为真实角
    artifact_clean_enabled: bool = True          # 启用伪顶点净化

    # ---- B路 DFS 矩形验证 ----
    rect_fill_min: float = 0.83            # 白纸：凸包/外接矩形面积比下限（与 saiti 一致）
    direct_fill_min: float = 0.87          # 白纸：各片总面积/外接矩形面积比下限
    card_rect_fill_min: float = 0.92       # 扑克牌：矩形度下限
    card_direct_fill_min: float = 0.84     # 扑克牌：直接填充率下限（与 saiti PC 一致）
    card_artifact_force_edge_mm: float = 12.0  # 扑克：小于此值强制删边，不做角度检查
    rect_area_tol: float = 0.20            # 凸包面积与总片面积差异容差（与 saiti 一致）
    min_composite_edge_mm: float = 5.0     # 复合覆盖各片贡献边长下限(mm)；<5mm视为噪声伪边

    # ---- 碎片筛选 ----
    piece_min_area_mm2: float = 50.0
    piece_max_area_mm2: float = 8000.0
    piece_min_solidity: float = 0.55

    # ---- 矩形匹配 ----
    rect_tolerance_mm: float = 3.0
    rect_min_edge_mm: float = 8.0

    # ---- 多边形短边过滤 ----
    min_piece_edge_mm: float = 7.0     # 小于此长度的边视为假边，合并相邻顶点

    # ---- B路 拼图放置区 ----
    separator_x_mm: float = 105.0         # 竖直分界线 X 坐标 mm（右半区左边界）
    separator_y_mm: float = 148.5         # 水平分界线 Y 坐标 mm（下半区上边界，默认 A4 半高）
    placement_gap_mm: float = 30.0        # 拼图与分界线的间距 mm
    placement_edge_mm: float = 15.0       # 拼图与纸边的间距 mm
    placement_x_offset_mm: float = 32.0   # 拼图 X 轴正向额外偏移 mm（微调用）
    piece_gap_mm: float = 4.0             # 相邻片间物理最小间隙 mm（吸收检测误差 + 定位公差；0=无缝拼接）

    # ---- 路径 ----
    calib_dir: str = "calib_data"
    templates_file: str = "templates.json"
    homography_file: str = "homography.json"

    @property
    def calib_path(self) -> Path:
        return Path(self.calib_dir)

    @property
    def templates_path(self) -> Path:
        return self.calib_path / self.templates_file

    @property
    def homography_path(self) -> Path:
        return self.calib_path / self.homography_file

    # ---- 通信 ----
    uart_port: str = '/dev/ttyACM0'
    uart_baudrate: int = 115200
    uart_timeout: float = 0.1

    # ---- 摄像头 ----
    camera_index: int = -1       # <0 = 自动检测第一个可用摄像头
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30


# 单例
_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置单例"""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(cfg: Config) -> None:
    """覆盖全局配置"""
    global _config
    _config = cfg
