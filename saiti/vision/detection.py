#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vision/detection.py — 碎片检测与分界线识别

管线：
  1. find_paper_and_rectify()    — A4 纸面定位 + 透视矫正
  2. detect_dividing_line()      — 分界线检测（黑色实线）
  3. segment_pieces()            — 碎片分割
  4. detect_pieces()             — 碎片检测入口
  5. analyze_piece_geometry()    — 碎片几何分析
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from vision.config import Config, get_config

logger = logging.getLogger("detection")


# =============================================================================
# Piece 数据结构
# =============================================================================

@dataclass
class Piece:
    """单枚碎片检测结果"""

    index: int                              # 碎片编号 (1-based)
    contour: np.ndarray                     # 原始轮廓点 (N, 1, 2)
    vertices_px: np.ndarray                 # 近似多边形顶点 (M, 2) px
    center_px: Tuple[float, float]          # 重心 px
    area_px: float                          # 面积 px²

    # 几何属性
    num_sides: int = 0                      # 顶点数
    solidity: float = 0.0                   # 凸度 = area / hull_area
    edge_lengths_px: List[float] = field(default_factory=list)
    edge_lengths_mm: List[float] = field(default_factory=list)
    edge_directions_deg: List[float] = field(default_factory=list)
    edge_straightness: List[float] = field(default_factory=list)
    angle_obb_deg: float = 0.0              # 最小外接矩形角度
    angle_precise_deg: float = 0.0          # 精确角度（最长边方向）
    angle_precise_set: bool = False

    # 边界边分类（矩形匹配）
    boundary_edges: Dict[int, str] = field(default_factory=dict)

    # 碎片类型
    piece_type: str = "unknown"             # "color" | "white" | "card" | "unknown"
    texture_bgr: Optional[np.ndarray] = None  # 卡牌纹理图 (BGR 裁剪区)
    texture_origin_px: Optional[Tuple[int, int]] = None  # 纹理图左上角在矫正图中的坐标 (x, y)
    mean_hsv: Optional[Tuple[float, float, float]] = None  # 平均 HSV

    area_mm2: float = 0.0                   # 面积 mm²
    suction_point_px: Optional[Tuple[float, float]] = None  # 最佳吸取点 px (最大内切圆圆心)
    suction_point_mm: Optional[Tuple[float, float]] = None  # 最佳吸取点 mm

    # 标定后坐标 (mm)
    center_mm: Optional[Tuple[float, float]] = None
    vertices_mm: Optional[np.ndarray] = None   # 简化多边形 (3-5 顶点)
    contour_mm: Optional[np.ndarray] = None    # 稠密轮廓 (用于可视化)

    def __repr__(self) -> str:
        return (f"Piece(P{self.index}, {self.num_sides}sides, "
                f"type={self.piece_type}, area={self.area_px:.0f}px², "
                f"center=({self.center_px[0]:.0f},{self.center_px[1]:.0f}))")


# =============================================================================
# 1. 纸面定位
# =============================================================================

def _find_green_paper_quad(
    image_bgr: np.ndarray,
    cfg: "Config",
) -> Tuple[Optional[np.ndarray], float]:
    """
    用 HSV 绿色阈值检测绿色 A4 纸面，返回 (四角点 float32 (4,2), 置信度)。
    比边缘法更稳定，不受纸面内碎片干扰。
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, w = image_bgr.shape[:2]

    green_mask = cv2.inRange(
        hsv,
        (cfg.paper_green_h_lo, cfg.paper_green_s_min, cfg.paper_green_v_min),
        (cfg.paper_green_h_hi, 255, 255),
    )

    # 形态学：先闭运算填碎片孔洞，再开运算去边缘噪点
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, k_close)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN,  k_open)

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 0.05 * w * h:
        return None, 0.0

    confidence = min(1.0, area / (0.7 * w * h))

    # 尝试直接近似为四边形
    peri = cv2.arcLength(largest, True)
    quad = None
    for eps in (0.02, 0.04, 0.06, 0.08, 0.10):
        approx = cv2.approxPolyDP(largest, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            break

    # 近似失败：从凸包取四个极端角点
    if quad is None:
        hull = cv2.convexHull(largest).reshape(-1, 2).astype(np.float32)
        # 左上=min(x+y)  右上=max(x-y)  右下=max(x+y)  左下=min(x-y)
        sums  = hull[:, 0] + hull[:, 1]
        diffs = hull[:, 0] - hull[:, 1]
        quad = np.array([
            hull[np.argmin(sums)],
            hull[np.argmax(diffs)],
            hull[np.argmax(sums)],
            hull[np.argmin(diffs)],
        ], dtype=np.float32)

    return quad, confidence


def order_quad_clockwise(points: np.ndarray) -> np.ndarray:
    """四点排序：左上 → 右上 → 右下 → 左下"""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    # 按 y 分上下两组
    sorted_y = pts[np.argsort(pts[:, 1])]
    top = sorted_y[:2]
    bottom = sorted_y[2:]
    # 各组按 x 排序
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _find_quad_from_lines(gray: np.ndarray, h: int, w: int) -> List[Tuple[float, np.ndarray]]:
    """
    用 HoughLinesP 检测线段，聚类角度后求交点，组合成四边形。
    对灰色纸张等低对比度场景更鲁棒。
    """
    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                             minLineLength=min(w, h) // 6, maxLineGap=min(w, h) // 10)
    if lines is None or len(lines) < 4:
        return []

    # 线段按角度分两组：接近水平的 (≈0° or ≈180°) 和接近垂直的 (≈90°)
    horiz_lines = []
    vert_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 20:
            continue
        if angle < 20 or angle > 160:
            horiz_lines.append((x1, y1, x2, y2, length))
        elif 70 < angle < 110:
            vert_lines.append((x1, y1, x2, y2, length))

    if len(horiz_lines) < 2 or len(vert_lines) < 2:
        return []

    # 取最长的 top/bottom 水平线和 left/right 垂直线
    horiz_lines.sort(key=lambda x: x[4], reverse=True)
    vert_lines.sort(key=lambda x: x[4], reverse=True)

    # 按 y 坐标分组：上组和下组
    h_ys = [(l[1] + l[3]) / 2 for l in horiz_lines[:6]]
    median_y = np.median(h_ys)
    top_lines = [l for l in horiz_lines[:6] if (l[1] + l[3]) / 2 < median_y]
    bot_lines = [l for l in horiz_lines[:6] if (l[1] + l[3]) / 2 >= median_y]

    # 按 x 坐标分组：左组和右组
    v_xs = [(l[0] + l[2]) / 2 for l in vert_lines[:6]]
    median_x = np.median(v_xs)
    left_lines = [l for l in vert_lines[:6] if (l[0] + l[2]) / 2 < median_x]
    right_lines = [l for l in vert_lines[:6] if (l[0] + l[2]) / 2 >= median_x]

    if not (top_lines and bot_lines and left_lines and right_lines):
        return []

    candidates: List[Tuple[float, np.ndarray]] = []

    for t_line in top_lines:
        for b_line in bot_lines:
            for l_line in left_lines:
                for r_line in right_lines:
                    # 计算四条线的交点作为四个角
                    def _line_intersection(l1, l2):
                        x1, y1, x2, y2 = l1[0], l1[1], l1[2], l1[3]
                        x3, y3, x4, y4 = l2[0], l2[1], l2[2], l2[3]
                        d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                        if abs(d) < 1e-9:
                            return None
                        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
                        ix = x1 + t * (x2 - x1)
                        iy = y1 + t * (y2 - y1)
                        return (ix, iy)

                    tl = _line_intersection(t_line, l_line)
                    tr = _line_intersection(t_line, r_line)
                    bl = _line_intersection(b_line, l_line)
                    br = _line_intersection(b_line, r_line)

                    if any(p is None for p in [tl, tr, bl, br]):
                        continue

                    pts = np.array([tl, tr, br, bl], dtype=np.float32)
                    # 检查是否在图像范围内
                    if np.any(pts < -50) or np.any(pts[:, 0] > w + 50) or np.any(pts[:, 1] > h + 50):
                        continue

                    # 检查面积
                    area = float(cv2.contourArea(pts.reshape(-1, 1, 2).astype(np.int32)))
                    if area < 0.03 * w * h:
                        continue

                    # 检查是凸四边形
                    if not cv2.isContourConvex(pts.reshape(4, 1, 2).astype(np.int32)):
                        continue

                    candidates.append((area, pts))

    return candidates


def find_paper_and_rectify(
    image_bgr: np.ndarray,
    cfg: Optional[Config] = None,
    debug_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    寻找绿色 A4 纸四边形并透视矫正。

    主策略：HSV 绿色阈值检测纸面 → 最大连通域 → 四角点
    回退：整幅图像作为纸面（confidence=0.3）

    Returns:
        rectified: 矫正后图像 (rectified_h, rectified_w, 3)
        H: 相机像素 → 矫正图像像素 单应矩阵 (3×3)
        confidence: 检测置信度 (0~1)
    """
    if cfg is None:
        cfg = get_config()

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("输入图像为空")

    h, w = image_bgr.shape[:2]
    rect_w, rect_h = cfg.rectified_w, cfg.rectified_h
    dst = np.array(
        [[0, 0], [rect_w - 1, 0], [rect_w - 1, rect_h - 1], [0, rect_h - 1]],
        dtype=np.float32,
    )

    quad, confidence = _find_green_paper_quad(image_bgr, cfg)

    if quad is None:
        logger.warning("绿色纸面检测失败，使用整幅图像作为纸面")
        quad = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        )
        confidence = 0.3
    else:
        logger.info("绿色纸面检测成功, conf=%.2f", confidence)

    src = order_quad_clockwise(quad)
    H = cv2.getPerspectiveTransform(src, dst)
    rectified = cv2.warpPerspective(image_bgr, H, (rect_w, rect_h))

    return rectified, H, confidence


# =============================================================================
# 2. 分界线识别 ← 核心新功能
# =============================================================================

def _score_vertical(
    gray_eq: np.ndarray,
    y0: int, y1: int, x0: int, x1: int,
    dark_ratio: float,
) -> dict:
    """
    搜索竖直线：取 ROI 转置后沿列投影，返回每列的暗度得分。
    """
    strip = gray_eq[y0:y1, x0:x1].T  # (cols, rows_per_col)
    n_cols = strip.shape[0]
    strip_w = strip.shape[1]

    # 黑帽变换突出暗色竖线（大核覆盖 2/3 ROI 高度，单行高度精确定位）
    kernel_size = max(15, strip_w * 2 // 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
    blackhat = cv2.morphologyEx(strip, cv2.MORPH_BLACKHAT, kernel)

    # 每列最暗像素均值
    sorted_rows = np.sort(strip, axis=1)
    dark_count = max(8, int(dark_ratio * strip_w))
    score_raw = sorted_rows[:, :dark_count].mean(axis=1).astype(np.float32)
    score_bh = blackhat.mean(axis=1).astype(np.float32)

    def _norm(s):
        d = s.max() - s.min()
        return (s - s.min()) / max(d, 1e-6)

    score = 0.6 * _norm(score_raw) + 0.4 * _norm(score_bh)

    idx = int(np.argmin(score))

    # 梯度验证
    grad = cv2.Sobel(strip, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.abs(grad).mean(axis=1)
    band = max(2, int(0.02 * n_cols))
    gs, ge = max(0, idx - band), min(n_cols, idx + band + 1)
    grad_ratio = float(grad_mag[gs:ge].max()) / max(float(np.median(grad_mag)), 1e-6)

    # 对比度验证
    raw_median = float(np.median(score_raw))
    raw_min = float(score_raw[idx])
    nb = max(3, int(0.05 * n_cols))
    a0, a1 = max(0, idx - nb * 2), max(0, idx - nb)
    b0, b1 = min(n_cols, idx + nb), min(n_cols, idx + nb * 2)
    above_m = float(np.median(score_raw[a0:a1])) if a1 > a0 else raw_median
    below_m = float(np.median(score_raw[b0:b1])) if b1 > b0 else raw_median
    neighbor_median = (above_m + below_m) / 2.0

    return {
        "score": score,
        "score_raw": score_raw,
        "strip": strip,
        "blackhat": blackhat,
        "grad_mag": grad_mag,
        "idx": idx,
        "candidate_pos": x0 + idx,
        "contrast": raw_median - raw_min,
        "local_contrast": neighbor_median - raw_min,
        "grad_ratio": grad_ratio,
        "x0": x0, "x1": x1, "y0": y0, "y1": y1,
    }


def _find_content_roi(gray: np.ndarray, dark_threshold: int = 35) -> Tuple[int, int, int, int]:
    """找出纸面内容区域 (排除透视矫正的黑色边框)。"""
    h, w = gray.shape[:2]
    _, content_mask = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_OPEN, kernel)
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)))
    contours, _ = cv2.findContours(content_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        margin = 6
        return (max(0, x + margin), max(0, y + margin),
                min(w, x + cw - margin), min(h, y + ch - margin))
    return (int(w * 0.03), int(h * 0.03), int(w * 0.97), int(h * 0.97))


def detect_dividing_line(
    rectified_bgr: np.ndarray,
    cfg: Optional[Config] = None,
    debug: bool = False,
) -> Tuple[int, float, dict]:
    """
    检测 A4 纸中部黑色竖直线。

    搜索纸面中部 40%-60% 区域内的竖直线，
    左半区 = 拼图区，右半区 = 取料区（纸片放置区）。

    Returns:
        separator_x: 分界线 X 坐标 (px)
        confidence: 检测置信度 (0.0 ~ 1.0)
        debug_info: 调试数据
    """
    if cfg is None:
        cfg = get_config()

    h, w = rectified_bgr.shape[:2]
    gray = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # 轻度去噪，保留线条边缘

    content_x0, content_y0, content_x1, content_y1 = _find_content_roi(gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    x0 = max(content_x0, int(cfg.separator_search_xmin * w))
    x1 = min(content_x1, int(cfg.separator_search_xmax * w))
    y0 = max(content_y0, int(cfg.separator_search_ymin * h))
    y1 = min(content_y1, int(cfg.separator_search_ymax * h))

    if x1 - x0 < 40 or y1 - y0 < 40:
        logger.warning("内容区域过小，使用完整图像搜索区域")
        x0 = int(cfg.separator_search_xmin * w)
        x1 = int(cfg.separator_search_xmax * w)
        y0 = int(cfg.separator_search_ymin * h)
        y1 = int(cfg.separator_search_ymax * h)

    result = _score_vertical(gray_eq, y0=y0, y1=y1, x0=x0, x1=x1,
                             dark_ratio=cfg.separator_dark_ratio)

    contrast = result["contrast"]
    local_c = result["local_contrast"]
    gr = result["grad_ratio"]
    cm = cfg.separator_contrast_min

    conf_c = min(1.0, contrast / cm)
    conf_l = min(1.0, local_c / (cm * 0.5))
    conf_g = min(1.0, gr / 1.5)
    confidence = float(0.4 * conf_c + 0.3 * conf_l + 0.3 * conf_g)

    is_valid = (contrast >= cm and local_c >= cm * 0.4)

    if is_valid:
        candidate_x = result["candidate_pos"]
        logger.info("分界线检测成功: x=%d, conf=%.2f, contrast=%.1f, grad=%.2f",
                     candidate_x, confidence, contrast, gr)
    else:
        candidate_x = int(round(w * 0.5))
        confidence = min(confidence, 0.3)
        logger.info("分界线检测失败 (contrast=%.1f,local=%.1f)，回退到中线 x=%d",
                     contrast, local_c, candidate_x)

    debug_info: dict = {}
    if debug:
        debug_info = {
            "gray_eq": gray_eq,
            "result": result,
            "candidate_pos": candidate_x,
            "confidence": confidence,
            "is_valid": is_valid,
            "x0": x0, "x1": x1, "y0": y0, "y1": y1,
            "content_x0": content_x0, "content_y0": content_y0,
            "content_x1": content_x1, "content_y1": content_y1,
        }

    return candidate_x, confidence, debug_info


def draw_dividing_line(
    rectified: np.ndarray,
    separator_x: int,
    confidence: float = 1.0,
    color: Tuple[int, int, int] = (40, 40, 40),
    thickness: int = 3,
) -> np.ndarray:
    """在矫正图上绘制竖直线分界线：左侧=拼图区，右侧=取料区。"""
    canvas = rectified.copy()
    h, w = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.line(canvas, (separator_x, 0), (separator_x, h - 1), color, thickness)

    puz_text = "PUZZLE"
    (tw, th), _ = cv2.getTextSize(puz_text, font, 0.7, 2)
    cv2.putText(canvas, puz_text, (separator_x // 2 - tw // 2, 30),
                font, 0.7, (200, 80, 0), 2, cv2.LINE_AA)

    mat_text = "MATERIAL"
    (tw, th), _ = cv2.getTextSize(mat_text, font, 0.7, 2)
    cx = separator_x + (w - separator_x) // 2
    cv2.putText(canvas, mat_text, (cx - tw // 2, 30),
                font, 0.7, (0, 100, 0), 2, cv2.LINE_AA)

    conf_text = f"conf: {confidence:.2f}"
    cv2.putText(canvas, conf_text, (separator_x + 6, 50),
                font, 0.45, (100, 100, 100), 1, cv2.LINE_AA)

    return canvas


def get_zones(separator_x: int, image_w: int) -> Dict[str, Tuple[int, int]]:
    """返回功能区 X 范围：puzzle_zone = 左侧, material_zone = 右侧。"""
    return {
        "puzzle_zone": (0, separator_x),
        "material_zone": (separator_x, image_w),
    }


# =============================================================================
# 3. 碎片分割 — 三策略融合
# =============================================================================

def estimate_paper_color(
    rectified_bgr: np.ndarray,
    separator_x: int,
    cfg: Optional[Config] = None,
) -> np.ndarray:
    """从右半区（取料区）边缘采样纸张底色 (LAB)。保留用于参考。"""
    if cfg is None:
        cfg = get_config()

    h, w = rectified_bgr.shape[:2]
    margin = int(12 * cfg.px_per_mm)
    lab = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    material_x0 = separator_x + margin
    samples = np.concatenate(
        [
            lab[margin:h - margin, material_x0:material_x0 + 12].reshape(-1, 3),
            lab[margin:h - margin, w - margin - 12:w - margin].reshape(-1, 3),
            lab[margin:margin + 12, material_x0:w - margin].reshape(-1, 3),
            lab[h - margin - 12:h - margin, material_x0:w - margin].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(samples, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# 3a. 白色碎片亮度分割（白色碎片在灰色底纸上）
# ---------------------------------------------------------------------------

def _segment_white_bright(roi_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """
    白色碎片分割（边缘 + 凸度过滤）。

    pipeline:
      CLAHE灰度 → Canny → 膨胀连边 → findContours
      → 面积过滤（去小噪点/反光）
      → solidity 过滤（反光边界软、不规则，solidity 低）
      → 填充保留轮廓 → mask

    反光点特征：边界模糊 → Canny 出不来封闭轮廓，或 solidity 极低 → 被过滤。
    白纸特征：物理硬边界 → 封闭轮廓，多边形，solidity 高。
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(blur)

    edges = cv2.Canny(eq, cfg.white_canny_lo, cfg.white_canny_hi)

    # 膨胀连接断边，再闭运算填内孔
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kd, iterations=2)
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.white_close_kernel, cfg.white_close_kernel))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kc)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
    min_area = roi_bgr.shape[0] * roi_bgr.shape[1] * cfg.white_min_area_ratio
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area:
            continue
        hull_area = float(cv2.contourArea(cv2.convexHull(c)))
        if hull_area < 1.0:
            continue
        if area / hull_area < cfg.white_min_solidity:
            continue
        cv2.drawContours(mask, [c], -1, 255, cv2.FILLED)

    return mask


# ---------------------------------------------------------------------------
# 3b. 颜色碎片 HSV 分割
# ---------------------------------------------------------------------------

def _segment_color_hsv(roi_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """
    颜色碎片分割：高饱和度区域 + 合理亮度范围。
    适用于彩色纸片（未来扩展）。
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    _, s_mask = cv2.threshold(s_channel, cfg.hsv_saturation_thresh, 255, cv2.THRESH_BINARY)
    v_mask = cv2.inRange(v_channel, cfg.hsv_value_min, cfg.hsv_value_max)
    mask = cv2.bitwise_and(s_mask, v_mask)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.hsv_open_kernel, cfg.hsv_open_kernel))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.hsv_close_kernel, cfg.hsv_close_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    return mask


# ---------------------------------------------------------------------------
# 3c. 扑克牌碎片分割
# ---------------------------------------------------------------------------

def _segment_cards(roi_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """
    扑克牌碎片分割：饱和度 + 纹理梯度 + 连通域面积过滤。

    卡牌有丰富的印刷图案：彩色区域(S高) + 纹理丰富(梯度大)。
    不使用V通道（会与白色碎片重叠）。
    大核闭运算连接纹理碎片 + 面积过滤噪声。
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]

    # 通道1: 彩色纹理区域
    _, s_mask = cv2.threshold(s_channel, cfg.card_saturation_thresh, 255, cv2.THRESH_BINARY)

    # 通道2: Sobel 梯度幅值（纹理边缘）
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    grad_mag = np.clip(grad_mag, 0, 255).astype(np.uint8)
    _, grad_mask = cv2.threshold(grad_mag, cfg.card_gradient_thresh, 255, cv2.THRESH_BINARY)

    # OR 合并
    combined = cv2.bitwise_or(s_mask, grad_mask)

    # 闭运算：先用小核连接，再根据连通域面积过滤噪声
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.card_close_kernel, cfg.card_close_kernel))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)

    # 连通域面积过滤：只保留大面积区域（卡牌纹理区域连通后面积大）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    mask = np.zeros_like(combined)
    min_card_area = int(0.01 * roi_bgr.shape[0] * roi_bgr.shape[1])  # 至少1%面积
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_card_area:
            mask[labels == i] = 255

    # 开运算最终去噪
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    return mask


# ---------------------------------------------------------------------------
# 3d. 绿色背景差分（凹形碎片最可靠的方法）
# ---------------------------------------------------------------------------

def _segment_non_background(roi_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """
    反绿背景差分：把绿色纸面 mask 取反，得到所有非绿色区域。

    对凹形（凹边）碎片最可靠：不依赖边缘闭合，直接以颜色区分前景/背景。
    白色碎片在绿色纸面上时，无论形状多凹，只要像素不是绿色就被标记。
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(
        hsv,
        (cfg.paper_green_h_lo, cfg.paper_green_s_min, cfg.paper_green_v_min),
        (cfg.paper_green_h_hi, 255, 255),
    )
    # 稍微扩展绿色区域，避免纸面边缘绿色像素残留
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green_mask = cv2.dilate(green_mask, k_dilate, iterations=1)

    # 非绿色 = 碎片前景
    fg = cv2.bitwise_not(green_mask)

    # 开运算去掉孤立小噪点（单像素反光）
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_open)

    # 闭运算填小孔（碎片内部轻微反光造成的绿色像素）
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close)

    return fg


# ---------------------------------------------------------------------------
# 3e. 融合分割 + 分水岭分离
# ---------------------------------------------------------------------------

def segment_pieces(
    rectified_bgr: np.ndarray,
    separator_x: int,
    cfg: Optional[Config] = None,
    debug_images: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """
    四策略融合分割：绿色背景差分（凹形碎片主力）+ HSV颜色 + 白色边缘 + 扑克牌纹理。

    1. 提取右半区 ROI（取料区）
    2. 分别运行四种分割方法
    3. 四通道 mask OR 融合
    4. 返回全尺寸 mask（左半区=0，右半区=融合结果）
    """
    if cfg is None:
        cfg = get_config()

    h, w = rectified_bgr.shape[:2]

    # 右半区 ROI（取料区）
    roi = rectified_bgr[:, separator_x:]
    roi_h, roi_w = roi.shape[:2]

    if roi_w < 20 or roi_h < 20:
        logger.warning("取料区过小，返回空 mask")
        return np.zeros((h, w), dtype=np.uint8)

    # ---- 四种分割策略 OR 融合（白纸/扑克通用）----
    mask_bg    = _segment_non_background(roi, cfg)  # 绿底差分（所有场景基础）
    mask_white = _segment_white_bright(roi, cfg)    # Canny白色边缘
    mask_color = _segment_color_hsv(roi, cfg)       # 高饱和彩色
    mask_card  = _segment_cards(roi, cfg)           # 扑克纹理/梯度

    # 仅当前景（非绿区域）内有足够饱和度时才启用扑克纹理mask，防止白纸场景误触发
    hsv_check = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    s_check   = hsv_check[:, :, 1]
    fg_check  = mask_bg  # 用绿底差分结果限定统计范围（排除绿色背景本身的高饱和）
    fg_pixels = int(cv2.countNonZero(fg_check))
    if fg_pixels > 0:
        sat_in_fg = float(cv2.countNonZero(
            cv2.bitwise_and(
                cv2.threshold(s_check, cfg.card_saturation_thresh, 255, cv2.THRESH_BINARY)[1],
                fg_check,
            )
        )) / fg_pixels
    else:
        sat_in_fg = 0.0

    if sat_in_fg < 0.005:   # 前景无饱和度信号 → 白纸场景，抑制扑克mask
        mask_card = np.zeros_like(mask_card)

    mask_merged = cv2.bitwise_or(mask_bg,    mask_color)
    mask_merged = cv2.bitwise_or(mask_merged, mask_white)
    mask_merged = cv2.bitwise_or(mask_merged, mask_card)

    # Debug 输出
    if debug_images is not None:
        debug_images["seg_non_bg"]       = mask_bg
        debug_images["seg_white_bright"] = mask_white
        debug_images["seg_hsv_color"]    = mask_color
        debug_images["seg_card_texture"] = mask_card
        debug_images["seg_merged"]       = mask_merged.copy()

    if debug_images is not None:
        debug_images["seg_merged"] = mask_merged.copy()

    # ---- 轮廓填充：找闭合区域填实心 ----
    # 边缘检测出来的mask可能是空心轮廓，通过 findContours + fillPoly 填实
    contours_fill, _ = cv2.findContours(mask_merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_material = np.zeros_like(mask_merged)
    for c in contours_fill:
        area = float(cv2.contourArea(c))
        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / max(hull_area, 1.0)
        # 过滤噪点轮廓
        if area < 100 and solidity < 0.5:
            continue
        cv2.drawContours(mask_material, [c], -1, 255, -1)

    if debug_images is not None:
        debug_images["seg_filled"] = mask_material.copy()

    # ---- 边缘保护：排除 ROI 边界 ----
    border = int(4 * cfg.px_per_mm)
    mask_material[:border, :] = 0
    mask_material[-border:, :] = 0
    mask_material[:, :border] = 0
    mask_material[:, roi_w - border:] = 0

    # ---- 形态学清理 ----
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_material = cv2.morphologyEx(mask_material, cv2.MORPH_CLOSE, k_close)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_material = cv2.morphologyEx(mask_material, cv2.MORPH_OPEN, k_open)

    # 构建全尺寸 mask
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:, separator_x:] = mask_material
    return mask


# ---------------------------------------------------------------------------
# 3e. 碎片类型分类
# ---------------------------------------------------------------------------

def _classify_piece_type(
    piece_contour: np.ndarray,
    roi_bgr: np.ndarray,
    masks: Dict[str, np.ndarray],
    cfg: Config,
) -> Tuple[str, Optional[np.ndarray], Tuple[float, float, float]]:
    """
    根据碎片在三种 mask 中的命中情况及 HSV 特征分类。

    Args:
        piece_contour: 碎片轮廓
        roi_bgr: 取料区 BGR 图像
        masks: {"color": ..., "white": ..., "card": ...} 三种分割 mask
        cfg: 配置

    Returns:
        piece_type: "color" | "white" | "card" | "unknown"
        texture_bgr: 卡牌纹理图 (仅 card 类型非 None)
        mean_hsv: 碎片区域平均 HSV
    """
    # 碎片局部 mask
    piece_mask = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(piece_mask, [piece_contour], -1, 255, -1)

    piece_area = float(cv2.countNonZero(piece_mask))
    if piece_area < 10:
        return ("unknown", None, (0.0, 0.0, 0.0))

    # 计算碎片在三种 mask 中的命中率
    hits: Dict[str, float] = {}
    for key, m in masks.items():
        overlap = cv2.bitwise_and(piece_mask, m)
        hits[key] = float(cv2.countNonZero(overlap)) / piece_area

    # 计算碎片区域 HSV 统计
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hsv_masked = hsv[piece_mask > 0]
    if len(hsv_masked) == 0:
        return ("unknown", None, (0.0, 0.0, 0.0))

    mean_h = float(np.mean(hsv_masked[:, 0]))
    mean_s = float(np.mean(hsv_masked[:, 1]))
    mean_v = float(np.mean(hsv_masked[:, 2]))
    std_s = float(np.std(hsv_masked[:, 1]))
    mean_hsv = (mean_h, mean_s, mean_v)

    std_v = float(np.std(hsv_masked[:, 2]))

    # 分类逻辑（优先级：white > card > color > unknown）
    # 白纸：高亮度 + 低饱和 + 均匀纹理，无论 card mask 是否命中都判白
    _white_v_min = getattr(cfg, 'white_value_min',      160)
    _white_s_max = getattr(cfg, 'white_saturation_max',  40)
    _white_std_s = getattr(cfg, 'white_std_s_max',       20.0)
    # 白纸优先：高亮度 + 低饱和 + 均匀（std_s低），即使有轻微纹理也优先判白
    if mean_v >= _white_v_min and mean_s <= _white_s_max and std_s <= _white_std_s:
        piece_type = "white"
        texture_bgr = None
    # 扑克牌：白纸检查未命中时，依赖 std_s（K牌白底+彩色图案 → std_s明显升高）
    elif std_s >= cfg.card_texture_std_min or hits.get("color", 0) >= 0.2:
        piece_type = "card"
        x, y, bw, bh = cv2.boundingRect(piece_contour)
        texture_bgr = roi_bgr[max(0, y):min(roi_bgr.shape[0], y + bh),
                               max(0, x):min(roi_bgr.shape[1], x + bw)].copy()
        texture_mask = piece_mask[max(0, y):min(roi_bgr.shape[0], y + bh),
                                   max(0, x):min(roi_bgr.shape[1], x + bw)]
        texture_bgr[texture_mask == 0] = 0
    elif hits.get("color", 0) >= 0.3 and mean_s >= cfg.hsv_saturation_thresh * 0.8:
        piece_type = "color"
        texture_bgr = None
    elif mean_s < cfg.hsv_saturation_thresh * 0.8:
        piece_type = "white"
        texture_bgr = None
    else:
        piece_type = "unknown"
        texture_bgr = None

    return (piece_type, texture_bgr, mean_hsv)


# =============================================================================
# 4. 碎片检测
# =============================================================================

def simplify_contour(
    contour: np.ndarray,
    max_vertices: int = 5,
) -> np.ndarray:
    """Douglas-Peucker 简化轮廓，保证顶点数 ≤ max_vertices。

    从细到粗扫描：连续两次 epsilon 得到相同有效顶点数，认为找到了自然多边形。
    这能正确保留"近平角"顶点（梯形钝角），不会被过粗的 epsilon 提前消掉。
    """
    peri = cv2.arcLength(contour, True)
    if peri < 1.0:
        return contour.reshape(-1, 2).astype(np.float32)

    prev_n = -1
    prev_approx = None
    for ratio in (0.007, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(contour, peri * ratio, True)
        n = len(approx)
        if 3 <= n <= max_vertices:
            if n == prev_n:
                # 连续两次相同 → 稳定，返回较细那次的结果
                return prev_approx.reshape(-1, 2).astype(np.float32)
            if prev_n > 0 and prev_n > n:
                # 顶点数下降：prev_n 是最细 epsilon 下的合法结果，直接返回
                # 避免粗 epsilon 的低精度结果覆盖真实多边形的顶点个数
                return prev_approx.reshape(-1, 2).astype(np.float32)
            prev_n = n
            prev_approx = approx
        else:
            # n 超出有效范围（过多顶点），重置，避免把噪声结果当前驱
            prev_n = -1
            prev_approx = None

    # 未触发稳定准则：返回最后一个有效结果（ratio=0.05 的粗略结果）
    if prev_approx is not None:
        return prev_approx.reshape(-1, 2).astype(np.float32)

    # 极端情况（轮廓复杂，仍 >max_vertices）：强制二分
    lo, hi = peri * 0.01, peri * 0.5
    best = cv2.approxPolyDP(contour, hi, True)
    for _ in range(20):
        mid = (lo + hi) / 2.0
        approx = cv2.approxPolyDP(contour, mid, True)
        if len(approx) <= max_vertices:
            best = approx
            hi = mid
        else:
            lo = mid
    return best.reshape(-1, 2).astype(np.float32)


def sort_vertices_cyclic(points: np.ndarray) -> np.ndarray:
    """顶点按重心角度排序（逆时针）"""
    pts = np.asarray(points, dtype=np.float64)
    c = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(angles)]


def _compute_suction_point(contour: np.ndarray) -> Tuple[float, float]:
    """计算最佳吸取点：轮廓内距边界最远点（最大内切圆圆心）。

    方法：距离变换 — 将轮廓填充为二值图，对内部做 L2 距离变换，
    取最大值点（该点到最近轮廓边界距离最大，是最稳定的吸取位置）。
    """
    x, y, bw, bh = cv2.boundingRect(contour)
    if bw < 2 or bh < 2:
        m = cv2.moments(contour)
        if abs(m["m00"]) > 1e-9:
            return (m["m10"] / m["m00"], m["m01"] / m["m00"])
        return (float(x + bw / 2), float(y + bh / 2))

    local = np.zeros((bh, bw), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, :, 0] -= x
    shifted[:, :, 1] -= y
    cv2.drawContours(local, [shifted], -1, 255, cv2.FILLED)

    dist = cv2.distanceTransform(local, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist)
    return (float(max_loc[0] + x), float(max_loc[1] + y))


def analyze_piece_geometry(piece: Piece, px_per_mm: float = 4.0) -> Piece:
    """
    分析一枚碎片的几何属性：边数、边长、方向、直线度。

    Args:
        piece: 碎片（已有 contour, vertices_px, center_px, area_px）
        px_per_mm: 像素/mm 比例

    Returns:
        更新后的 piece
    """
    piece.area_mm2 = piece.area_px / (px_per_mm ** 2)

    verts = sort_vertices_cyclic(piece.vertices_px)
    n = len(verts)
    piece.num_sides = n
    piece.edge_lengths_px = []
    piece.edge_lengths_mm = []
    piece.edge_directions_deg = []
    piece.edge_straightness = []

    for i in range(n):
        p0 = verts[i]
        p1 = verts[(i + 1) % n]
        vec = p1 - p0
        length = float(np.linalg.norm(vec))

        piece.edge_lengths_px.append(length)
        piece.edge_lengths_mm.append(length / px_per_mm)

        # 边方向（数学角度，逆时针，0=右）
        angle_rad = math.atan2(vec[1], vec[0])
        angle_deg = math.degrees(angle_rad)
        piece.edge_directions_deg.append(round(angle_deg, 1))

        # 直线度：边两端点形成的直线与轮廓上中间点的偏差
        straightness = 1.0
        if len(piece.contour) > 2:
            edge_pts = []
            for pt in piece.contour:
                px, py = pt[0]
                # 点到线段距离
                vx, vy = p1[0] - p0[0], p1[1] - p0[1]
                v_len_sq = vx * vx + vy * vy
                if v_len_sq < 1e-9:
                    continue
                t = max(0, min(1, ((px - p0[0]) * vx + (py - p0[1]) * vy) / v_len_sq))
                proj_x = p0[0] + t * vx
                proj_y = p0[1] + t * vy
                dist = math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
                edge_pts.append(dist)

            if edge_pts:
                straightness = 1.0 - min(1.0, np.median(edge_pts) / max(length * 0.1, 1.0))

        piece.edge_straightness.append(round(straightness, 3))

    # OBB 角度
    if n >= 3:
        rect = cv2.minAreaRect(piece.vertices_px.astype(np.float32))
        piece.angle_obb_deg = rect[2]

    # 精确角度：最长边的方向
    if piece.edge_lengths_px:
        longest_idx = int(np.argmax(piece.edge_lengths_px))
        piece.angle_precise_deg = piece.edge_directions_deg[longest_idx]
        piece.angle_precise_set = True

    return piece


def detect_pieces(
    image_bgr: np.ndarray,
    cfg: Optional[Config] = None,
    debug_images: Optional[Dict[str, np.ndarray]] = None,
) -> List[Piece]:
    """
    完整碎片检测管线。

    Steps:
      1. 纸面定位 → rectified
      2. 分界线检测 → separator_y  ← 关键步骤
      3. 碎片分割 → mask
      4. 轮廓提取 + 筛选 → pieces
      5. 几何分析

    Args:
        image_bgr: 输入 BGR 图像
        cfg: 配置
        debug_images: 调试图像字典 (output param)

    Returns:
        List[Piece]: 检测到的碎片列表 (按面积降序, ≤4)
    """
    if cfg is None:
        cfg = get_config()

    # Step 1 & 2: 纸面定位 + 分界线（在矫正图上进行）
    rectified, H, paper_conf = find_paper_and_rectify(image_bgr, cfg)
    separator_x, line_conf, debug_line = detect_dividing_line(rectified, cfg, debug=False)

    if debug_images is not None:
        debug_images["02_rectified"] = rectified
        debug_images["02_separator"] = draw_dividing_line(rectified, separator_x, line_conf)

    mask = segment_pieces(rectified, separator_x, cfg, debug_images=debug_images)
    if debug_images is not None:
        debug_images["03_mask"] = mask

    # ---- 构建用于分类的三种 mask（从 debug_images 读取） ----
    roi = rectified[:, separator_x:]
    masks_for_classify: Dict[str, np.ndarray] = {}
    if debug_images is not None:
        masks_for_classify["color"] = debug_images.get("seg_hsv_color",
                                          np.zeros(roi.shape[:2], dtype=np.uint8))
        masks_for_classify["white"] = debug_images.get("seg_white_bright",
                                          np.zeros(roi.shape[:2], dtype=np.uint8))
        masks_for_classify["card"] = debug_images.get("seg_card_texture",
                                         np.zeros(roi.shape[:2], dtype=np.uint8))
    else:
        # 非debug模式也构建分类用的masks
        masks_for_classify["color"] = _segment_color_hsv(roi, cfg)
        masks_for_classify["white"] = _segment_white_bright(roi, cfg)
        masks_for_classify["card"] = _segment_cards(roi, cfg)

    # Step 4: 轮廓提取
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = cfg.piece_min_area_mm2 * cfg.px_per_mm * cfg.px_per_mm
    max_area = cfg.piece_max_area_mm2 * cfg.px_per_mm * cfg.px_per_mm

    # 类型颜色映射
    type_colors = {
        "color": (0, 255, 0),    # 绿
        "white": (255, 255, 0),  # 青
        "card": (255, 0, 255),   # 品红
        "unknown": (0, 0, 255),  # 红
    }

    pieces: List[Piece] = []
    all_contours_vis = rectified.copy() if debug_images is not None else None

    for contour in contours:
        # 轮廓在原始坐标系（全图），需要映射到 ROI 坐标做分类
        contour_roi = contour.copy()
        contour_roi[:, :, 0] -= separator_x  # 平移 X 到 ROI 坐标

        area = float(cv2.contourArea(contour))
        if not (min_area <= area <= max_area):
            if debug_images is not None and all_contours_vis is not None:
                cv2.drawContours(all_contours_vis, [contour], -1, (0, 0, 255), 1)
            continue

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / max(hull_area, 1.0)
        if solidity < cfg.piece_min_solidity:
            if debug_images is not None and all_contours_vis is not None:
                cv2.drawContours(all_contours_vis, [contour], -1, (0, 0, 255), 1)
            continue

        m = cv2.moments(contour)
        if abs(m["m00"]) < 1e-9:
            continue

        center = (m["m10"] / m["m00"], m["m01"] / m["m00"])

        # ---- 碎片类型分类（在净化之前，用于隔离白纸逻辑）----
        piece_type, texture_bgr, mean_hsv = _classify_piece_type(
            contour_roi, roi, masks_for_classify, cfg,
        )

        poly = simplify_contour(contour, max_vertices=6)

        # ---- 伪顶点净化（仅对扑克/彩色片；白纸边长有 >20mm 保证，不需净化）----
        # 短边（< card_artifact_min_mm）：印刷图案凸起/凹陷的两端点距离 <1cm，
        #   两端点同时删去，邻居直接相连还原真实直边
        # 近共线顶点（转角 < card_artifact_min_turn_deg）：图案在直边上形成轻微拐点，
        #   单独删去该顶点
        if piece_type != 'white':
            _art_px   = getattr(cfg, 'card_artifact_min_mm',       13.0) * cfg.px_per_mm
            _turn_min = getattr(cfg, 'card_artifact_min_turn_deg', 20.0)
            for _ in range(len(poly)):
                if len(poly) <= 3:
                    break
                n = len(poly)
                _ls = np.array([float(np.linalg.norm(
                    poly[(i+1)%n].astype(float) - poly[i].astype(float)))
                    for i in range(n)], dtype=np.float64)
                _j = int(np.argmin(_ls))
                if _ls[_j] < _art_px:
                    # 短边：删转角更小的端点，将短边合并进相邻长边
                    # （不删两端点，避免4边形删完只剩2顶点）
                    _ja, _jb = _j, (_j + 1) % n
                    def _turn(idx):
                        _p0 = poly[(idx-1)%n].astype(float)
                        _p1 = poly[idx].astype(float)
                        _p2 = poly[(idx+1)%n].astype(float)
                        _v1 = _p0 - _p1; _v2 = _p2 - _p1
                        _l1 = float(np.linalg.norm(_v1)); _l2 = float(np.linalg.norm(_v2))
                        if _l1 < 1e-6 or _l2 < 1e-6:
                            return 0.0
                        return 180.0 - math.degrees(math.acos(
                            float(np.clip(np.dot(_v1, _v2) / (_l1 * _l2), -1.0, 1.0))))
                    _del = _ja if _turn(_ja) <= _turn(_jb) else _jb
                    poly = np.delete(poly, _del, axis=0)
                    continue
                # 近共线检测：找转角最小的顶点
                _mt, _mi = 360.0, -1
                for i in range(n):
                    _p0 = poly[(i-1)%n].astype(float)
                    _p1 = poly[i].astype(float)
                    _p2 = poly[(i+1)%n].astype(float)
                    _v1 = _p0 - _p1; _l1 = float(np.linalg.norm(_v1))
                    _v2 = _p2 - _p1; _l2 = float(np.linalg.norm(_v2))
                    if _l1 < 1e-6 or _l2 < 1e-6:
                        continue
                    _cos = float(np.clip(np.dot(_v1, _v2) / (_l1*_l2), -1.0, 1.0))
                    _turn = 180.0 - math.degrees(math.acos(_cos))
                    if _turn < _mt:
                        _mt, _mi = _turn, i
                if _mi >= 0 and _mt < _turn_min:
                    poly = np.delete(poly, _mi, axis=0)
                else:
                    break

        # ---- 所有类型提取纸面纹理图 ----
        if texture_bgr is None:
            rx, ry, rbw, rbh = cv2.boundingRect(contour_roi)
            crop = roi[max(0, ry):min(roi.shape[0], ry + rbh),
                       max(0, rx):min(roi.shape[1], rx + rbw)].copy()
            if crop.size > 0:
                lm = np.zeros(crop.shape[:2], dtype=np.uint8)
                tc = contour_roi.copy()
                tc[:, :, 0] -= rx
                tc[:, :, 1] -= ry
                cv2.drawContours(lm, [tc], -1, 255, cv2.FILLED)
                crop[lm == 0] = 0
                texture_bgr = crop

        # ---- 吸取点（最大内切圆圆心）----
        suction_px = _compute_suction_point(contour)

        crx, cry, _, _ = cv2.boundingRect(contour_roi)
        piece = Piece(
            index=0,  # 稍后编号
            contour=contour,
            vertices_px=poly,
            center_px=center,
            area_px=area,
            solidity=solidity,
            piece_type=piece_type,
            texture_bgr=texture_bgr,
            texture_origin_px=(max(0, int(crx)) + separator_x, max(0, int(cry))),
            mean_hsv=mean_hsv,
            suction_point_px=suction_px,
        )
        piece = analyze_piece_geometry(piece, cfg.px_per_mm)
        pieces.append(piece)

        if debug_images is not None and all_contours_vis is not None:
            color = type_colors.get(piece_type, (0, 255, 0))
            cv2.drawContours(all_contours_vis, [contour], -1, color, 2)

    # 按面积降序
    pieces.sort(key=lambda p: p.area_px, reverse=True)

    # 编号
    for i, p in enumerate(pieces):
        p.index = i + 1

    if debug_images is not None and all_contours_vis is not None:
        debug_images["09_all_contours"] = all_contours_vis

    logger.info("碎片检测完成: %d 片 (搜索到 %d 个轮廓) | %s",
                 len(pieces), len(contours),
                 ", ".join(f"P{p.index}={p.piece_type}" for p in pieces))
    return pieces


# =============================================================================
# 5. 可视化
# =============================================================================

def draw_pieces(
    image_bgr: np.ndarray,
    pieces: List[Piece],
    show_obb: bool = False,
    show_vertices: bool = True,
) -> np.ndarray:
    """在图像上绘制检测到的碎片，按类型着色。"""
    canvas = image_bgr.copy()

    type_colors = {
        "color": (0, 255, 0),      # 绿
        "white": (255, 255, 0),    # 青
        "card": (255, 0, 255),     # 品红
        "unknown": (0, 0, 255),    # 红
    }
    type_labels = {
        "color": "C",              # Color
        "white": "W",              # White
        "card": "K",               # Card (K for poker card)
        "unknown": "?",
    }

    for piece in pieces:
        color = type_colors.get(piece.piece_type, (0, 0, 255))
        tlabel = type_labels.get(piece.piece_type, "?")

        # 轮廓
        cv2.drawContours(canvas, [piece.contour.astype(np.int32)], -1, color, 2)

        # 中心
        cx, cy = int(round(piece.center_px[0])), int(round(piece.center_px[1]))
        cv2.circle(canvas, (cx, cy), 5, color, -1)

        # 吸取点（十字标记）
        if piece.suction_point_px is not None:
            sx = int(round(piece.suction_point_px[0]))
            sy = int(round(piece.suction_point_px[1]))
            cv2.drawMarker(canvas, (sx, sy), (0, 255, 255),
                           cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)

        # 标签：类型 + 编号
        label = f"P{piece.index}[{tlabel}]"
        if piece.vertices_mm is not None and piece.center_mm is not None:
            label += f" ({piece.center_mm[0]:.0f},{piece.center_mm[1]:.0f})mm"
        cv2.putText(canvas, label, (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        # 顶点
        if show_vertices and piece.vertices_px is not None:
            for v in piece.vertices_px:
                vx, vy = int(round(v[0])), int(round(v[1]))
                cv2.circle(canvas, (vx, vy), 3, (255, 255, 255), -1)

        # OBB
        if show_obb and len(piece.vertices_px) >= 3:
            rect = cv2.minAreaRect(piece.vertices_px.astype(np.float32))
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.polylines(canvas, [box], True, (200, 200, 200), 1)

    return canvas


# =============================================================================
# 6. 矩形边界边匹配
# =============================================================================

def classify_boundary_edges(
    pieces: List[Piece],
    rect_w_mm: float,
    rect_h_mm: float,
    tolerance_mm: float = 3.0,
    min_edge_mm: float = 15.0,
) -> None:
    """
    对每条边判断是否属于目标矩形的边界。

    修改 pieces[i].boundary_edges：
      "long_side"  → 匹配矩形长边 (top/bottom)
      "short_side" → 匹配矩形短边 (left/right)
      "internal"   → 内部拼接边

    Args:
        pieces: 已标定的碎片列表
        rect_w_mm: 目标矩形宽度 (mm)
        rect_h_mm: 目标矩形高度 (mm)
        tolerance_mm: 边长容差
        min_edge_mm: 最小边长阈值
    """
    # 确定短边/长边
    if rect_w_mm >= rect_h_mm:
        long_mm, short_mm = rect_w_mm, rect_h_mm
    else:
        long_mm, short_mm = rect_h_mm, rect_w_mm

    for piece in pieces:
        if not piece.edge_lengths_mm:
            continue

        piece.boundary_edges = {}
        for i, edge_mm in enumerate(piece.edge_lengths_mm):
            if edge_mm < min_edge_mm:
                piece.boundary_edges[i] = "internal"
                continue

            diff_long = abs(edge_mm - long_mm)
            diff_short = abs(edge_mm - short_mm)

            if diff_long <= tolerance_mm and diff_long <= diff_short:
                piece.boundary_edges[i] = "long_side"
            elif diff_short <= tolerance_mm and diff_short < diff_long:
                piece.boundary_edges[i] = "short_side"
            else:
                piece.boundary_edges[i] = "internal"


def validate_rectangle_fit(
    pieces: List[Piece],
    rect_w_mm: float,
    rect_h_mm: float,
    tolerance_mm: float = 3.0,
) -> dict:
    """
    验证边界边是否覆盖矩形四条边。

    Returns:
        {
            "valid": bool,
            "issues": [str],
            "side_coverage": {
                "long_side":  {"edges": [...], "total_mm", "expected_mm"},
                "short_side": {"edges": [...], "total_mm", "expected_mm"},
            },
            "pieces_without_boundary": [int],
        }
    """
    long_mm = max(rect_w_mm, rect_h_mm)
    short_mm = min(rect_w_mm, rect_h_mm)

    issues = []
    long_edges = []   # (piece_index, edge_index, length_mm)
    short_edges = []
    no_boundary = []

    for piece in pieces:
        has_boundary = False
        for i, label in piece.boundary_edges.items():
            if label == "long_side":
                long_edges.append((piece.index, i, piece.edge_lengths_mm[i]))
                has_boundary = True
            elif label == "short_side":
                short_edges.append((piece.index, i, piece.edge_lengths_mm[i]))
                has_boundary = True

        if not has_boundary:
            no_boundary.append(piece.index)

    total_long = sum(e[2] for e in long_edges)
    total_short = sum(e[2] for e in short_edges)

    # 需要覆盖 2 条长边 + 2 条短边
    if total_long < long_mm * 2 - tolerance_mm * 2:
        issues.append(f"长边覆盖不足: {total_long:.1f}/{long_mm*2:.0f}mm")
    if total_short < short_mm * 2 - tolerance_mm * 2:
        issues.append(f"短边覆盖不足: {total_short:.1f}/{short_mm*2:.0f}mm")

    if no_boundary:
        issues.append(f"P{no_boundary} 无边界边")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "side_coverage": {
            "long_side": {
                "edges": long_edges,
                "total_mm": total_long,
                "expected_mm": long_mm * 2,
            },
            "short_side": {
                "edges": short_edges,
                "total_mm": total_short,
                "expected_mm": short_mm * 2,
            },
        },
        "pieces_without_boundary": no_boundary,
    }


def draw_boundary_edges(
    image_bgr: np.ndarray,
    pieces: List[Piece],
    show_internal: bool = False,
    show_labels: bool = True,
) -> np.ndarray:
    """可视化边界边分类结果。"""
    canvas = image_bgr.copy()

    color_map = {
        "long_side": (0, 255, 0),       # 绿
        "short_side": (0, 255, 255),     # 黄
        "internal": (128, 128, 128),     # 灰
    }

    for piece in pieces:
        verts = sort_vertices_cyclic(piece.vertices_px)
        n = len(verts)

        for i, v0 in enumerate(verts):
            v1 = verts[(i + 1) % n]
            label = piece.boundary_edges.get(i, "")
            color = color_map.get(label, (0, 0, 255))

            if label == "internal" and not show_internal:
                continue
            if not label:
                continue

            pt0 = (int(round(v0[0])), int(round(v0[1])))
            pt1 = (int(round(v1[0])), int(round(v1[1])))
            cv2.line(canvas, pt0, pt1, color, 3)

            if show_labels and label:
                mid = ((pt0[0] + pt1[0]) // 2, (pt0[1] + pt1[1]) // 2)
                short = label.replace("_side", "")
                cv2.putText(canvas, short, mid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return canvas


# =============================================================================
# 7. 标定转换
# =============================================================================

def apply_calibration(
    pieces: List[Piece],
    pixel_to_machine: Callable[[float, float], Tuple[float, float]],
) -> List[Piece]:
    """
    将碎片的像素坐标转换为机械臂坐标 (mm)。

    Args:
        pieces: 碎片列表
        pixel_to_machine: 像素 → 机械臂坐标转换函数

    Returns:
        更新后的 pieces
    """
    for piece in pieces:
        px, py = piece.center_px
        piece.center_mm = pixel_to_machine(px, py)

        if piece.vertices_px is not None and len(piece.vertices_px) > 0:
            piece.vertices_mm = np.array(
                [pixel_to_machine(float(v[0]), float(v[1]))
                 for v in piece.vertices_px],
                dtype=np.float32,
            )

        if piece.suction_point_px is not None:
            piece.suction_point_mm = pixel_to_machine(*piece.suction_point_px)

    return pieces
