# 视觉子系统 — 开发进度

---

## Scripts 目录

### 标定 / 工具
| 脚本 | 用途 | 运行方式 |
|---|---|---|
| `calibrate_aruco.py` | ArUco 标定 → 生成 `calib_data/homography.json` | `python scripts/calibrate_aruco.py` |
| `capture.py` | 实时预览 + SPACE 保存帧，用于离线抓标定图 | `python scripts/capture.py` |

### 实时诊断
| 脚本 | 用途 | 运行方式 |
|---|---|---|
| `live_dividing_line.py` | 实时显示纸面定位 + 分界线检测（蓝/绿区域遮罩） | `python scripts/live_dividing_line.py` |
| `live_segmentation.py` | 实时显示三策略碎片分割（白/彩/扑克）+ 分界线 | `python scripts/live_segmentation.py` |

### 参数调优
| 脚本 | 用途 | 调节范围 |
|---|---|---|
| `tune_white.py` | 白片 Canny 分割调参（4 滑块） | `canny_lo/hi`, `close_kernel`, `min_solidity` |
| `tune_green_bg.py` | `build_templates` 绿底检测调参 | 独立于 detection.py 的绿底 HSV 流水线 |

### 测试 / 联调
| 脚本 | 用途 | 说明 |
|---|---|---|
| `build_templates.py` | 构建 A 路模板，保存到 `calib_data/templates.json` | 赛前必跑；也被其他脚本 import |
| `test_b_route.py` | B 路 DFS 算法**离线**单测（合成多边形，无摄像头） | CI 级别快速验证 |
| `test_b_live.py` | B 路**实机**端到端测试：相机 → 检测 → DFS → 打印指令 | 主要实机联调入口 |
| `test_solver.py` | A/B 路**实机**综合测试：冻结帧 → 检测 → 求解 → 预览图 | A 路需要先跑 `build_templates.py` |

---

## 已完成模块

### 1. 纸面定位 (`vision/detection.py` → `find_paper_and_rectify`)
- HSV 绿色阈值检测纸面轮廓 → 最大连通域 → 四角点 → 透视矫正到 840×1188 px（4 px/mm）
- 四角点：优先 `approxPolyDP`，失败则取凸包极端角点
- 检测失败时降级为整幅图（confidence=0.3）

### 2. 分界线识别 (`vision/detection.py` → `detect_dividing_line`)
- 检测矫正后纸面中部黑色竖直线，搜索范围 40%–60%
- CLAHE → 黑帽变换 → 列投影评分 → 对比度验证
- **左侧 = 拼图区，右侧 = 取料区**

### 3. 碎片分割 (`vision/detection.py` → `segment_pieces`)
四策略 OR 融合：
| 策略 | 说明 |
|---|---|
| `_segment_non_background` | 反色绿底 HSV（处理凹形碎片） |
| `_segment_white_bright` | Canny + 闭运算 + solidity 过滤（白色碎片） |
| `_segment_color_hsv` | 高饱和度 HSV 阈值（彩色碎片） |
| `_segment_cards` | 饱和度 OR Sobel 梯度（扑克牌碎片） |

### 4. 碎片几何分析 (`vision/detection.py` → `analyze_piece_geometry`)
输出：`area_mm2`、`edge_lengths_mm`、`num_sides`、`angle_obb_deg`、`angle_precise_deg`、`suction_point_mm`（最大内切圆圆心，用于吸盘落点）

### 5. 轮廓简化 (`vision/detection.py` → `simplify_contour`)
Douglas-Peucker + 稳定性二分，保证顶点数 ≤ 5

### 6. ArUco 标定 (`scripts/calibrate_aruco.py`)
- 实时检测 DICT_4X4_50，SPACE 冻结，终端输入各标记 mm 坐标
- 计算像素→纸面 mm 单应矩阵 H，打印 RMSE/最大残差，保存 `calib_data/homography.json`

### 7. A 路求解 (`vision/solver.py` → `solve_template`)
- `identify_piece`：按排序边长 L2 + 面积比匹配模板
- `register_piece`：循环边对齐 + SVD 精化旋转
- `solve_template`：识别 → 注册 → 输出 `PlacementResult`

### 8. B 路 DFS 求解 (`vision/b_solver.py` → `solve_geometric`)
- 最大面积碎片固定原点，逐条放置边枚举覆盖方案
- 覆盖类型：**1:1 直接匹配** 和 **复合覆盖**（多片拼接一条边）
- 邻缝双宿主：单片横跨相邻两条已放边
- `_check_rectangle`：矩形度（凸包/OBB）、直填率、面积容差三重验证
- 分数 = 边长误差 + 复合惩罚（每多一片 +100）
- 返回最多 80 个候选，按分数排序；有纹理数据时用 SVD 纹理评分二次排名

**可配置阈值**（`config.py`）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `artifact_clean_enabled` | `False` | 伪顶点净化开关 |
| `rect_fill_min` | `0.92` | 矩形度下限（凸包/OBB） |
| `direct_fill_min` | `0.77` | 直填率下限（片总面积/OBB） |
| `rect_area_tol` | `0.20` | 面积误差容差 |
| `min_composite_edge_mm` | `5.0` | 复合覆盖最小贡献边长（mm） |
| `card_artifact_min_edge_mm` | `25.0` | 扑克牌伪短边删除阈值（mm） |
| `card_artifact_min_turn_deg` | `20.0` | 近共线伪顶点最大转角（°） |

---

## 已知问题 / 待修复

| # | 问题 | 位置 | 优先级 |
|---|---|---|---|
| 1 | `_BB_LONG=152` / `_BB_SHORT=108` 拼图外框硬编码，不读 config | `b_solver.py` | 中 |
| 2 | 白纸碎片被 `detection.py` 分类为 `card`/`unknown`，导致伪顶点净化误删边 | `detection.py` | 高 |
| 3 | `separator_gradient_min` 在 config 中定义但 `detect_dividing_line` 不读取（内部用硬编码 1.5） | `config.py` / `detection.py` | 低 |
| 4 | `card_bright_thresh` 在 config 中定义但无处使用 | `config.py` | 低 |
| 5 | `solver.py` 中 `_b_candidates`/`_b_matching_sets`/`_b_assemble` 为旧版死代码（已被 `b_solver.py` 取代） | `solver.py` | 低 |
| 6 | `estimate_paper_color` 无调用方，死代码 | `detection.py` | 低 |
| 7 | `artifact_clean_enabled` 在 config 默认 `False`，但 b_solver `getattr` 默认 `True`，两者不一致 | `config.py` / `b_solver.py` | 中 |

---

## 待完成

- [ ] **缺口梯度检测**：真实切割缺口 vs 光照/毛边，用 Sobel 梯度幅值区分
- [ ] **白纸碎片分类修复**：detection.py `_classify_piece_type` 对纯白碎片返回正确 `'white'`，以便恢复伪顶点净化
- [ ] **扑克牌碎片伪顶点净化联调**：`artifact_clean_enabled=True` + 真实扑克牌碎片测试
- [ ] **DFS 外框尺寸配置化**：`_BB_LONG`/`_BB_SHORT` 改从 config 读取
- [ ] **B 路 P3-P4 直接配对验证**：确认 DFS 能在正确惩罚下找到 4-3-3-3 构型
- [ ] **串口通信**：上位机向龙门架发送吸取/放置/旋转指令
- [ ] **solver.py 旧版死代码清理**：删除 `_b_candidates` 等三个函数
