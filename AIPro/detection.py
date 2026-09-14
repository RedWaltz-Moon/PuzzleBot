#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detection.py — 碎片检测管线（嵌入式精简版）

从 saiti/vision/detection.py 精简适配：
  - 保留纸面定位 + 分界线检测 + 碎片分割 + 几何分析核心逻辑
  - 移除 dataclass，使用普通类
  - 移除 debug_images 字典传递，减少内存占用
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import Config, get_config

logger = logging.getLogger("detection")


# =============================================================================
# Piece 数据结构
# =============================================================================

class Piece:
    """单枚碎片检测结果"""

    def __init__(self):
        self.index: int = 0
        self.contour: np.ndarray = np.array([])
        self.vertices_px: np.ndarray = np.array([])
        self.center_px: Tuple[float, float] = (0.0, 0.0)
        self.area_px: float = 0.0

        # 几何属性
        self.num_sides: int = 0
        self.solidity: float = 0.0
        self.edge_lengths_px: List[float] = []
        self.edge_lengths_mm: List[float] = []
        self.edge_directions_deg: List[float] = []
        self.edge_straightness: List[float] = []
        self.angle_obb_deg: float = 0.0
        self.angle_precise_deg: float = 0.0
        self.angle_precise_set: bool = False

        # 边界边分类
        self.boundary_edges: Dict[int, str] = {}

        # 碎片类型
        self.piece_type: str = "unknown"
        self.texture_bgr: Optional[np.ndarray] = None
        self.texture_origin_px: Optional[Tuple[int, int]] = None  # 纹理图左上角在矫正图中的坐标
        self.mean_hsv: Optional[Tuple[float, float, float]] = None

        # 坐标
        self.area_mm2: float = 0.0
        self.suction_point_px: Optional[Tuple[float, float]] = None
        self.suction_point_mm: Optional[Tuple[float, float]] = None
        self.center_mm: Optional[Tuple[float, float]] = None
        self.vertices_mm: Optional[np.ndarray] = None
        self.contour_mm: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return (f"Piece(P{self.index}, {self.num_sides}sides, "
                f"type={self.piece_type}, area={self.area_px:.0f}px²)")


# =============================================================================
# 几何工具
# =============================================================================

def polygon_centroid(points: np.ndarray) -> np.ndarray:
    """多边形重心。"""
    pts = np.asarray(points, dtype=np.float64)
    m = cv2.moments(pts.astype(np.float32))
    if abs(m["m00"]) < 1e-9:
        return pts.mean(axis=0).astype(np.float32)
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]], dtype=np.float32)


def order_quad_clockwise(points: np.ndarray) -> np.ndarray:
    """四点排序：左上 → 右上 → 右下 → 左下。"""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sorted_y = pts[np.argsort(pts[:, 1])]
    top = sorted_y[:2]
    bottom = sorted_y[2:]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def sort_vertices_cyclic(points: np.ndarray) -> np.ndarray:
    """顶点按重心角度排序（逆时针）。"""
    pts = np.asarray(points, dtype=np.float64)
    c = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(angles)]


def simplify_contour(contour: np.ndarray, max_vertices: int = 5) -> np.ndarray:
    """Douglas-Peucker 简化轮廓。"""
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
                return prev_approx.reshape(-1, 2).astype(np.float32)
            if prev_n > 0 and prev_n > n:
                # 顶点数下降：prev_n 是最细 epsilon 下的合法结果，直接返回
                # 避免粗 epsilon 的低精度结果覆盖真实多边形的顶点个数
                return prev_approx.reshape(-1, 2).astype(np.float32)
            prev_n = n
            prev_approx = approx
        else:
            prev_n = -1
            prev_approx = None

    if prev_approx is not None:
        return prev_approx.reshape(-1, 2).astype(np.float32)

    # 二分回退
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


def _filter_short_edges(vertices: np.ndarray, min_length_px: float) -> np.ndarray:
    """过滤假边：迭代删除最短边（合并相邻顶点），直到所有边 >= min_length_px。"""
    if len(vertices) < 4 or min_length_px <= 0:
        return vertices

    verts = vertices.astype(np.float64)
    for _ in range(5):
        n = len(verts)
        if n <= 3:
            break

        # 找最短边
        best_len = float("inf")
        best_i = -1
        for i in range(n):
            L = float(np.linalg.norm(verts[(i + 1) % n] - verts[i]))
            if L < best_len:
                best_len = L
                best_i = i

        if best_len >= min_length_px:
            break

        # 合并最短边的两个顶点为中点
        j = (best_i + 1) % n
        mid = (verts[best_i] + verts[j]) / 2.0
        if n <= 3:
            break
        mask = np.ones(n, dtype=bool)
        mask[j] = False
        verts = verts[mask]
        verts[best_i] = mid

    return verts.astype(np.float32)


def _compute_suction_point(contour: np.ndarray) -> Tuple[float, float]:
    """最大内切圆圆心（最佳吸取点）。"""
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


# =============================================================================
# 1. 纸面定位
# =============================================================================

def _find_green_paper_quad(image_bgr: np.ndarray, cfg: Config) -> Tuple[Optional[np.ndarray], float]:
    """HSV 绿色阈值检测绿色纸面，返回 (四角点, 置信度)。"""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, w = image_bgr.shape[:2]

    green_mask = cv2.inRange(
        hsv,
        (cfg.paper_green_h_lo, cfg.paper_green_s_min, cfg.paper_green_v_min),
        (cfg.paper_green_h_hi, 255, 255),
    )

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, k_close)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, k_open)

    # 边界腐蚀：清除触碰图像边缘的绿色像素，断开桌面/背景连通域。
    # 绿底纸面完全在视野内不受影响；桌面延伸出画面则被切断。
    green_mask[:65, :] = 0
    green_mask[-60:, :] = 0
    green_mask[:, :220] = 0
    green_mask[:, -205:] = 0

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 0.05 * w * h:
        return None, 0.0

    confidence = min(1.0, area / (0.7 * w * h))

    # 近似四边形
    peri = cv2.arcLength(largest, True)
    quad = None
    for eps in (0.02, 0.04, 0.06, 0.08, 0.10):
        approx = cv2.approxPolyDP(largest, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            break

    if quad is None:
        hull = cv2.convexHull(largest).reshape(-1, 2).astype(np.float32)
        sums = hull[:, 0] + hull[:, 1]
        diffs = hull[:, 0] - hull[:, 1]
        quad = np.array([
            hull[np.argmin(sums)],
            hull[np.argmax(diffs)],
            hull[np.argmax(sums)],
            hull[np.argmin(diffs)],
        ], dtype=np.float32)

    return quad, confidence


def find_paper_and_rectify(
    image_bgr: np.ndarray,
    cfg: Optional[Config] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """寻找绿色 A4 纸四边形并透视矫正。"""
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
# 2. 分界线检测
# =============================================================================

def _score_vertical(gray_eq: np.ndarray, y0: int, y1: int, x0: int, x1: int,
                    dark_ratio: float) -> dict:
    """搜索竖直线：取 ROI 转置后沿列投影。"""
    strip = gray_eq[y0:y1, x0:x1].T
    n_cols = strip.shape[0]
    strip_w = strip.shape[1]

    kernel_size = max(15, strip_w * 2 // 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
    blackhat = cv2.morphologyEx(strip, cv2.MORPH_BLACKHAT, kernel)

    sorted_rows = np.sort(strip, axis=1)
    dark_count = max(8, int(dark_ratio * strip_w))
    score_raw = sorted_rows[:, :dark_count].mean(axis=1).astype(np.float32)
    score_bh = blackhat.mean(axis=1).astype(np.float32)

    def _norm(s):
        d = s.max() - s.min()
        return (s - s.min()) / max(d, 1e-6)

    score = 0.6 * _norm(score_raw) + 0.4 * _norm(score_bh)
    idx = int(np.argmin(score))

    grad = cv2.Sobel(strip, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.abs(grad).mean(axis=1)
    band = max(2, int(0.02 * n_cols))
    gs, ge = max(0, idx - band), min(n_cols, idx + band + 1)
    grad_ratio = float(grad_mag[gs:ge].max()) / max(float(np.median(grad_mag)), 1e-6)

    raw_median = float(np.median(score_raw))
    raw_min = float(score_raw[idx])
    nb = max(3, int(0.05 * n_cols))
    a0, a1 = max(0, idx - nb * 2), max(0, idx - nb)
    b0, b1 = min(n_cols, idx + nb), min(n_cols, idx + nb * 2)
    above_m = float(np.median(score_raw[a0:a1])) if a1 > a0 else raw_median
    below_m = float(np.median(score_raw[b0:b1])) if b1 > b0 else raw_median
    neighbor_median = (above_m + below_m) / 2.0

    return {
        "idx": idx,
        "candidate_pos": x0 + idx,
        "contrast": raw_median - raw_min,
        "local_contrast": neighbor_median - raw_min,
        "grad_ratio": grad_ratio,
    }


def _find_content_roi(gray: np.ndarray, dark_threshold: int = 35) -> Tuple[int, int, int, int]:
    """找出纸面内容区域（排除透视矫正的黑色边框）。"""
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
) -> Tuple[int, float]:
    """检测 A4 纸中部黑色竖直线。返回 (separator_x, confidence)。"""
    if cfg is None:
        cfg = get_config()

    h, w = rectified_bgr.shape[:2]
    gray = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    content_x0, content_y0, content_x1, content_y1 = _find_content_roi(gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    x0 = max(content_x0, int(cfg.separator_search_xmin * w))
    x1 = min(content_x1, int(cfg.separator_search_xmax * w))
    y0 = max(content_y0, int(cfg.separator_search_ymin * h))
    y1 = min(content_y1, int(cfg.separator_search_ymax * h))

    if x1 - x0 < 40 or y1 - y0 < 40:
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
        logger.info("分界线检测成功: x=%d, conf=%.2f", candidate_x, confidence)
    else:
        candidate_x = int(round(w * 0.5))
        confidence = min(confidence, 0.3)
        logger.info("分界线检测失败，回退到中线 x=%d", candidate_x)

    return candidate_x, confidence


# =============================================================================
# 3. 碎片分割
# =============================================================================

def _segment_white_bright(roi_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """白色碎片分割（Canny + 凸度过滤）。"""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(blur)
    edges = cv2.Canny(eq, cfg.white_canny_lo, cfg.white_canny_hi)
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


def _segment_color_hsv(roi_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """颜色碎片 HSV 分割。"""
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

    # 闭运算：先用大核连接纹理碎片
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.card_close_kernel, cfg.card_close_kernel))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)

    # 连通域面积过滤：只保留大面积区域（≥1% ROI面积）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    mask = np.zeros_like(combined)
    min_card_area = int(0.01 * roi_bgr.shape[0] * roi_bgr.shape[1])
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_card_area:
            mask[labels == i] = 255

    # 开运算最终去噪
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    return mask


def _segment_non_background(roi_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """
    反绿背景差分：把绿色纸面 mask 取反，得到所有非绿色区域。

    对凹形（凹边）碎片最可靠：不依赖边缘闭合，直接以颜色区分前景/背景。
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

    # 开运算去掉孤立小噪点
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_open)

    # 闭运算填小孔（碎片内部轻微反光造成的绿色像素）
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close)

    return fg


def segment_pieces(
    rectified_bgr: np.ndarray,
    separator_x: int,
    cfg: Optional[Config] = None,
    enable_card: bool = False,
) -> np.ndarray:
    """三策略融合分割：白色 + 颜色（+ 扑克纹理，若 enable_card）。"""
    if cfg is None:
        cfg = get_config()

    h, w = rectified_bgr.shape[:2]
    roi = rectified_bgr[:, separator_x:]
    roi_h, roi_w = roi.shape[:2]

    if roi_w < 20 or roi_h < 20:
        logger.warning("取料区过小，返回空 mask")
        return np.zeros((h, w), dtype=np.uint8)

    mask_white = _segment_white_bright(roi, cfg)
    mask_color = _segment_color_hsv(roi, cfg)

    # 扑克纹理mask — 仅 enable_card 时启用
    mask_card = np.zeros((roi_h, roi_w), dtype=np.uint8)
    mask_bg   = np.zeros((roi_h, roi_w), dtype=np.uint8)
    if enable_card:
        mask_bg   = _segment_non_background(roi, cfg)   # 绿底差分（凹形碎片主力）
        mask_card = _segment_cards(roi, cfg)

        # 绿底安全阀：若矫正后ROI中绿色占比过低，说明纸面定位/色偏导致
        # 绿底差分失效（全图被标记为前景），此时不合并 mask_bg
        hsv_check = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        green_in_roi = cv2.inRange(
            hsv_check,
            (cfg.paper_green_h_lo, cfg.paper_green_s_min, cfg.paper_green_v_min),
            (cfg.paper_green_h_hi, 255, 255),
        )
        green_ratio = float(cv2.countNonZero(green_in_roi)) / max(roi_h * roi_w, 1)
        bg_ratio = float(cv2.countNonZero(mask_bg)) / max(roi_h * roi_w, 1)
        if green_ratio < 0.05 or bg_ratio > 0.90:
            logger.info("C路: ROI绿色占比=%.3f bg占比=%.3f → 跳过绿底差分(纸面定位/色偏)",
                        green_ratio, bg_ratio)
            mask_bg = np.zeros((roi_h, roi_w), dtype=np.uint8)

        # 饱和度自抑制：前景中饱和度像素占比 < 0.5% → 清零card_mask（防白纸误触发）
        s_check = hsv_check[:, :, 1]
        fg_pixels = int(cv2.countNonZero(mask_bg) if cv2.countNonZero(mask_bg) > 0 else
                       (cv2.countNonZero(mask_card) + cv2.countNonZero(mask_color)))
        if fg_pixels > 0:
            sat_in_fg = float(cv2.countNonZero(
                cv2.bitwise_and(
                    cv2.threshold(s_check, cfg.card_saturation_thresh, 255, cv2.THRESH_BINARY)[1],
                    mask_bg if cv2.countNonZero(mask_bg) > 0 else mask_card,
                )
            )) / fg_pixels
        else:
            sat_in_fg = 0.0

        if sat_in_fg < 0.005:
            mask_card = np.zeros_like(mask_card)

    # 四通道 OR 融合（与 saiti PC 一致：bg差分是凹形碎片主力）
    mask_merged = cv2.bitwise_or(mask_bg,    mask_color)
    mask_merged = cv2.bitwise_or(mask_merged, mask_white)
    mask_merged = cv2.bitwise_or(mask_merged, mask_card)

    # 轮廓填充
    contours_fill, _ = cv2.findContours(mask_merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_material = np.zeros_like(mask_merged)
    for c in contours_fill:
        area = float(cv2.contourArea(c))
        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / max(hull_area, 1.0)
        if area < 100 and solidity < 0.5:
            continue
        cv2.drawContours(mask_material, [c], -1, 255, -1)

    # 边缘保护
    border = int(4 * cfg.px_per_mm)
    mask_material[:border, :] = 0
    mask_material[-border:, :] = 0
    mask_material[:, :border] = 0
    mask_material[:, roi_w - border:] = 0

    # 形态学清理
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_material = cv2.morphologyEx(mask_material, cv2.MORPH_CLOSE, k_close)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_material = cv2.morphologyEx(mask_material, cv2.MORPH_OPEN, k_open)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:, separator_x:] = mask_material
    return mask


# =============================================================================
# 3b. 碎片类型分类
# =============================================================================

def _classify_piece_type(
    piece_contour: np.ndarray,
    roi_bgr: np.ndarray,
    masks: Dict[str, np.ndarray],
    cfg: Config,
) -> Tuple[str, Optional[np.ndarray], Tuple[float, float, float]]:
    """
    根据碎片在三种 mask 中的命中情况及 HSV 特征分类。

    Args:
        piece_contour: 碎片轮廓 (ROI 坐标系)
        roi_bgr: 取料区 BGR 图像
        masks: {"color": ..., "white": ..., "card": ...} 三种分割 mask
        cfg: 配置

    Returns:
        piece_type: "color" | "white" | "card" | "unknown"
        texture_bgr: 卡牌纹理图 (仅 card 类型非 None)
        mean_hsv: 碎片区域平均 HSV
    """
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

    # 分类逻辑（优先级：white > card > color > unknown）
    _white_v_min = getattr(cfg, 'white_value_min', 160)
    _white_s_max = getattr(cfg, 'white_saturation_max', 40)
    _white_std_s = getattr(cfg, 'white_std_s_max', 20.0)

    if mean_v >= _white_v_min and mean_s <= _white_s_max and std_s <= _white_std_s:
        piece_type = "white"
        texture_bgr = None
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
# 4. 几何分析
# =============================================================================

def analyze_piece_geometry(piece: Piece, px_per_mm: float = 4.0) -> Piece:
    """分析碎片几何属性。"""
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

        angle_rad = math.atan2(vec[1], vec[0])
        angle_deg = math.degrees(angle_rad)
        piece.edge_directions_deg.append(round(angle_deg, 1))

        # 直线度
        straightness = 1.0
        if len(piece.contour) > 2:
            edge_pts = []
            for pt in piece.contour:
                px, py = pt[0]
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

    # 精确角度：最长边方向
    if piece.edge_lengths_px:
        longest_idx = int(np.argmax(piece.edge_lengths_px))
        piece.angle_precise_deg = piece.edge_directions_deg[longest_idx]
        piece.angle_precise_set = True

    return piece


# =============================================================================
# 5. 碎片检测
# =============================================================================

def detect_pieces(
    image_bgr: np.ndarray,
    cfg: Optional[Config] = None,
    enable_card: bool = False,
    return_homography: bool = False,
) -> List[Piece]:
    """
    完整碎片检测管线（矫正图坐标系）。

    Args:
        image_bgr: 输入 BGR 图像
        cfg: 配置
        enable_card: 启用卡片分类 + 纹理提取 + px级顶点清洗
        return_homography: 返回 (pieces, H) 而非 pieces
    """
    if cfg is None:
        cfg = get_config()

    rectified, H, paper_conf = find_paper_and_rectify(image_bgr, cfg)
    separator_x, line_conf = detect_dividing_line(rectified, cfg)
    mask = segment_pieces(rectified, separator_x, cfg, enable_card=enable_card)

    # 构建用于分类的三种 mask（enable_card 时使用）
    roi = rectified[:, separator_x:]
    masks_for_classify: Dict[str, np.ndarray] = {}
    if enable_card:
        masks_for_classify["color"] = _segment_color_hsv(roi, cfg)
        masks_for_classify["white"] = _segment_white_bright(roi, cfg)
        masks_for_classify["card"] = _segment_cards(roi, cfg)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = cfg.piece_min_area_mm2 * cfg.px_per_mm * cfg.px_per_mm
    max_area = cfg.piece_max_area_mm2 * cfg.px_per_mm * cfg.px_per_mm

    pieces: List[Piece] = []
    for contour in contours:
        # 轮廓在原始坐标系（全图），映射到 ROI 坐标做分类
        contour_roi = contour.copy()
        contour_roi[:, :, 0] -= separator_x

        area = float(cv2.contourArea(contour))
        if not (min_area <= area <= max_area):
            continue
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area / max(hull_area, 1.0)
        if solidity < cfg.piece_min_solidity:
            continue
        m = cv2.moments(contour)
        if abs(m["m00"]) < 1e-9:
            continue

        center = (m["m10"] / m["m00"], m["m01"] / m["m00"])

        # 碎片类型分类（enable_card 时）
        piece_type = "unknown"
        texture_bgr = None
        mean_hsv = None
        if enable_card:
            piece_type, texture_bgr, mean_hsv = _classify_piece_type(
                contour_roi, roi, masks_for_classify, cfg,
            )

        poly = simplify_contour(contour, max_vertices=6 if enable_card else 5)

        # ── 合并极短边（simplify_contour 可能拆分真边，产生 < 4mm 假边）──
        # 对所有类型生效：把极短边合并进较短的邻居边，恢复被拆分的真边
        _tiny_mm = 4.0  # 仅合并 < 4mm 的极短边，避免误伤真短边
        _tiny_px = _tiny_mm * cfg.px_per_mm
        for _ in range(len(poly)):
            if len(poly) <= 3:
                break
            n = len(poly)
            _ls = np.array([float(np.linalg.norm(
                poly[(i + 1) % n].astype(float) - poly[i].astype(float)))
                for i in range(n)], dtype=np.float64)
            _j = int(np.argmin(_ls))
            if _ls[_j] >= _tiny_px:
                break
            # 合并进较短的邻居边（如果两头转角都小，较短的邻居才是被拆分的另一半）
            _ja, _jb = _j, (_j + 1) % n
            _l_prev = _ls[(_j - 1) % n]
            _l_next = _ls[(_j + 1) % n]
            if _l_prev <= _l_next:
                poly = np.delete(poly, _ja, axis=0)   # 合并进前一条边
            else:
                poly = np.delete(poly, _jb, axis=0)   # 合并进后一条边

        # 伪顶点净化（仅对 card/color 片；白纸不进此路径）
        if enable_card and piece_type != 'white':
            _art_px = getattr(cfg, 'card_artifact_min_mm', 13.0) * cfg.px_per_mm
            _turn_min = getattr(cfg, 'card_artifact_min_turn_deg', 20.0)
            for _ in range(len(poly)):
                if len(poly) <= 3:
                    break
                n = len(poly)
                _ls = np.array([float(np.linalg.norm(
                    poly[(i + 1) % n].astype(float) - poly[i].astype(float)))
                    for i in range(n)], dtype=np.float64)
                _j = int(np.argmin(_ls))
                if _ls[_j] < _art_px:
                    _ja, _jb = _j, (_j + 1) % n

                    def _turn(idx):
                        _p0 = poly[(idx - 1) % n].astype(float)
                        _p1 = poly[idx].astype(float)
                        _p2 = poly[(idx + 1) % n].astype(float)
                        _v1 = _p0 - _p1
                        _v2 = _p2 - _p1
                        _l1 = float(np.linalg.norm(_v1))
                        _l2 = float(np.linalg.norm(_v2))
                        if _l1 < 1e-6 or _l2 < 1e-6:
                            return 0.0
                        return 180.0 - math.degrees(math.acos(
                            float(np.clip(np.dot(_v1, _v2) / (_l1 * _l2), -1.0, 1.0))))

                    _del = _ja if _turn(_ja) <= _turn(_jb) else _jb
                    poly = np.delete(poly, _del, axis=0)
                    continue
                # 近共线检测
                _mt, _mi = 360.0, -1
                for i in range(n):
                    _p0 = poly[(i - 1) % n].astype(float)
                    _p1 = poly[i].astype(float)
                    _p2 = poly[(i + 1) % n].astype(float)
                    _v1 = _p0 - _p1
                    _l1 = float(np.linalg.norm(_v1))
                    _v2 = _p2 - _p1
                    _l2 = float(np.linalg.norm(_v2))
                    if _l1 < 1e-6 or _l2 < 1e-6:
                        continue
                    _cos = float(np.clip(np.dot(_v1, _v2) / (_l1 * _l2), -1.0, 1.0))
                    _turn = 180.0 - math.degrees(math.acos(_cos))
                    if _turn < _mt:
                        _mt, _mi = _turn, i
                if _mi >= 0 and _mt < _turn_min:
                    poly = np.delete(poly, _mi, axis=0)
                else:
                    break

        suction_px = _compute_suction_point(contour)

        piece = Piece()
        piece.contour = contour
        piece.vertices_px = poly
        piece.center_px = center
        piece.area_px = area
        piece.solidity = solidity
        piece.suction_point_px = suction_px
        piece.piece_type = piece_type
        if texture_bgr is not None:
            piece.texture_bgr = texture_bgr
            # 纹理图原点在矫正图坐标系
            bx, by, _, _ = cv2.boundingRect(contour)
            piece.texture_origin_px = (bx, by)
        if mean_hsv is not None:
            piece.mean_hsv = mean_hsv

        piece = analyze_piece_geometry(piece, cfg.px_per_mm)
        pieces.append(piece)

    pieces.sort(key=lambda p: p.area_px, reverse=True)
    for i, p in enumerate(pieces):
        p.index = i + 1

    logger.info("碎片检测完成: %d 片", len(pieces))
    if return_homography:
        return pieces, H
    return pieces


# =============================================================================
# 6. 标定转换
# =============================================================================

def apply_calibration(
    pieces: List[Piece],
    pixel_to_machine: Callable[[float, float], Tuple[float, float]],
) -> List[Piece]:
    """将碎片像素坐标转换为机械臂 mm 坐标。"""
    for piece in pieces:
        px, py = piece.center_px
        piece.center_mm = pixel_to_machine(px, py)
        if piece.vertices_px is not None and len(piece.vertices_px) > 0:
            piece.vertices_mm = np.array(
                [pixel_to_machine(float(v[0]), float(v[1])) for v in piece.vertices_px],
                dtype=np.float32,
            )
        if piece.suction_point_px is not None:
            piece.suction_point_mm = pixel_to_machine(*piece.suction_point_px)
    return pieces
