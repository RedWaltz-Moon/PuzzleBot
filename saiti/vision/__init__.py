#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
saiti 视觉子系统 — 模块化视觉识别管线

管线：
  1. 纸面定位 (find_paper_and_rectify)
  2. 分界线检测 (detect_dividing_line)      ← 当前任务
  3. 碎片分割 (segment_pieces)
  4. 碎片分析 (analyze_pieces)
  5. 坐标转换 (apply_calibration)
"""
