#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vision/config.py — 视觉子系统全局配置

所有可调参数集中管理，方便调试和现场标定。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """视觉子系统配置"""

    # ---- 纸面检测（绿色纸面 HSV）----
    paper_green_h_lo: int = 40        # 绿色 Hue 下限（OpenCV 0-179）
    paper_green_h_hi: int = 88        # 绿色 Hue 上限
    paper_green_s_min: int = 60       # 绿色 S 下限（排除低饱和噪点）
    paper_green_v_min: int = 40       # 绿色 V 下限（排除过暗区域）

    # ---- A4 纸面 ----
    a4_width: float = 210.0          # mm
    a4_height: float = 297.0         # mm
    px_per_mm: float = 4.0           # 矫正后图像分辨率 (px/mm)
    rectified_w: int = 840           # 210 * 4
    rectified_h: int = 1188          # 297 * 4

    # ---- 分界线检测 ----
    # 搜索范围：纸面高度的百分比
    separator_search_ymin: float = 0.40   # 搜索上界（40%高度）
    separator_search_ymax: float = 0.60   # 搜索下界（60%高度）
    separator_search_xmin: float = 0.40   # 搜索左界（40%宽度）
    separator_search_xmax: float = 0.60   # 搜索右界（60%宽度）
    # 行得分：取每行最暗像素的比例
    separator_dark_ratio: float = 0.15    # 最暗 15% 参与评分
    # 判定阈值（基于原始像素值 0-255）
    separator_contrast_min: float = 8.0   # 行得分中位数 - 最小值 >= 此值才认为有线
    separator_gradient_min: float = 3.0   # 候选行附近梯度均值需 >= 此值

    # ---- 白色碎片分割（边缘 + 凸度过滤，对反光鲁棒）----
    white_canny_lo: int = 30              # Canny 低阈值
    white_canny_hi: int = 90             # Canny 高阈值
    white_close_kernel: int = 15          # 闭运算核（连接断边、填内孔，须为奇数）
    white_min_solidity: float = 0.60      # 凸度下限（凹形碎片 solidity 低，放宽至 0.60）
    white_min_area_ratio: float = 0.001   # 最小面积比（相对 ROI，过滤小噪点）

    # ---- 颜色碎片 HSV 分割 ----
    hsv_saturation_thresh: int = 35        # S通道阈值，高饱和度=彩色碎片
    hsv_value_min: int = 30                # V通道下限，排除阴影
    hsv_value_max: int = 255               # V通道上限
    hsv_open_kernel: int = 3
    hsv_close_kernel: int = 9

    # ---- 扑克牌碎片分割 ----
    card_saturation_thresh: int = 25       # 卡牌饱和度阈值（比纯色碎片略低）
    card_bright_thresh: int = 120          # 卡牌亮度阈值
    card_gradient_thresh: int = 20         # 纹理梯度幅值阈值
    card_close_kernel: int = 15            # 大核闭运算连接纹理区域
    card_texture_std_min: float = 12.0     # 纹理复杂度判定（S通道标准差）

    # ---- 碎片筛选 ----
    piece_min_area_mm2: float = 50.0      # 最小碎片面积 (mm²), 50mm²=800px@4px/mm
    piece_max_area_mm2: float = 8000.0    # 最大碎片面积 (mm²), 8000mm²=128000px
    piece_min_solidity: float = 0.55      # 最小凸度 (area / hull_area)；凹形碎片可低至 0.55
    card_artifact_min_mm: float = 13.0        # detection.py px 空间粗筛阈值(mm)
    card_artifact_min_edge_mm: float = 25.0  # b_solver mm 空间精清阈值；白纸>20mm 安全
    card_artifact_min_turn_deg: float = 20.0 # 近共线伪顶点转角上限(°)；真实切割角>30°
    artifact_clean_enabled: bool = True        # True=启用伪顶点净化；False=跳过（白纸/调试用）
    white_value_min: int = 160                 # 白纸亮度下限 (V通道)
    white_saturation_max: int = 40             # 白纸饱和度上限 (S通道)
    white_std_s_max: float = 20.0              # 白纸饱和度标准差上限（纹理均匀性）
    min_composite_edge_mm: float = 5.0        # 复合覆盖各片贡献边长下限(mm)；<5mm视为噪声伪边
    rect_fill_min: float = 0.92              # 白纸：凸包/外接矩形面积比下限（矩形度）
    direct_fill_min: float = 0.77            # 白纸：各片总面积/外接矩形面积比下限
    card_rect_fill_min: float = 0.92         # 扑克牌：矩形度下限
    card_direct_fill_min: float = 0.84       # 扑克牌：直接填充率下限
    rect_area_tol: float = 0.20              # 凸包面积与总片面积差异容差（检验重叠/间隙）

    # ---- 梯度纹理匹配 ----
    grad_inset_px: float = 5.0          # 采样点向内偏移量 (px)，避开背景像素
    grad_corr_min: float = 0.30         # 允许的最低归一化相关系数（低于此值拒绝配对）
    grad_clarity_min: float = 8.0       # 最低灰度标准差；低于此值图案过均匀/杂乱，跳过梯度剪枝

    # ---- B路拼接放置 ----
    separator_x_mm: float = 105.0          # 分界线 X 坐标 (mm)
    separator_y_mm: float = 148.5          # 分界线 Y 坐标 (mm，默认 A4 半高)
    placement_gap_mm: float = 15.0         # 拼图区距分界线间距 (mm)
    placement_edge_mm: float = 10.0        # 拼图区距纸边留白 (mm)
    placement_x_offset_mm: float = 15.0    # 拼图区 X 正向额外偏移 (mm)
    piece_gap_mm: float = 3.0              # 片间单侧撑开量 (mm)，总间隙 = 2×

    # ---- 矩形匹配 ----
    rect_tolerance_mm: float = 3.0        # 边长匹配容差 (mm)
    rect_min_edge_mm: float = 15.0        # 最小边界边长度 (mm)

    # ---- 路径 ----
    calib_dir: str = "calib_data"
    homography_file: str = "homography.json"

    @property
    def calib_path(self) -> Path:
        return Path(self.calib_dir)

    @property
    def homography_path(self) -> Path:
        return self.calib_path / self.homography_file


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
