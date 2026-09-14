# 视觉子系统算法说明

## 总体管线

```
相机帧
  │
  ├─ find_paper_and_rectify()   → 矫正图 (840×1188px, 4px/mm)
  ├─ detect_dividing_line()     → 分界线 x_mm
  ├─ segment_pieces()           → 碎片 mask
  ├─ detect_pieces()            → List[Piece]
  │
  ├─ A路 solve_template()       → List[PlacementResult]  （模板匹配）
  └─ B路 solve_geometric()      → List[PlacementResult]  （DFS几何拼接）
```

---

## 一、纸面定位与透视矫正

**文件：** `detection.py / find_paper_and_rectify()`

1. HSV 绿色阈值（H 40-88）提取纸面 mask，形态学闭开运算去噪。
2. `approxPolyDP` 逐渐放松 ε 拟合四边形；若失败则凸包取四极值角点。
3. 备用路径（灰色纸面）：`HoughLinesP` 检测线段 → 按水平/垂直聚类取 top/bottom/left/right → 四线交点。
4. `getPerspectiveTransform` → `warpPerspective` 输出 **210×297 mm，4 px/mm** 矫正图。

---

## 二、碎片分割

**文件：** `detection.py / segment_pieces()`

四路并行，OR 融合：

| 策略 | 方法 | 目标 |
|---|---|---|
| `_segment_non_background` | 绿色 HSV 差分 | 所有前景（通用基础）|
| `_segment_white_bright` | Canny 边缘 + 闭运算 | 白纸碎片 |
| `_segment_color_hsv` | 高饱和度阈值 | 彩色碎片 |
| `_segment_cards` | 纹理梯度 + 高饱和 | 扑克牌碎片 |

扑克纹理路径以前景饱和度 < 0.5% 为白纸场景判据，自动抑制（避免白纸误触发）。  
轮廓填充（`fillPoly`）将空心边缘 mask 填实；ROI 边界 4mm 清零防伪边；形态学闭开最终清理。

---

## 三、碎片几何分析

**文件：** `detection.py / analyze_piece_geometry()`

1. `simplify_contour(max_vertices=6)` — Douglas-Peucker 简化轮廓至 ≤6 顶点。
2. **伪顶点净化（像素空间，仅扑克/彩色片）**  
   - Pass A：最短边 < `card_artifact_min_mm`（13mm）→ 删转角更小的端点，防四边形退化。  
   - Pass B：转角 < `card_artifact_min_turn_deg`（20°）的近共线顶点 → 直接删除。
3. 提取边长（mm）、边方向、凸度、最大内切圆吸取点。
4. **碎片类型分类**：`_classify_piece_type()` 按亮度/饱和度/纹理标准差判 `white | card | color | unknown`。

---

## 四、A路：模板匹配

**文件：** `solver.py`

> 赛前把 4 片摆到目标位置录制模板，比赛时识别后用 SVD 求旋转差。

1. **识别**：排序边长 L2 距离 + 面积差加权评分 → 匹配最近模板（边数必须相同）。
2. **配准**：`_best_cyclic_offset` 找最优循环偏移 k → SVD 求刚体变换旋转角。
3. 输出 `PlacementResult`：吸取点 / 目标质心 / 旋转角。

---

## 五、B路：DFS 几何拼接

**文件：** `b_solver.py / solve_geometric()`

### 5.1 输入预处理

- 各片 `vertices_mm` 转局部多边形（以吸取点为原点，`_ccw` 保证逆时针）。
- **伪顶点净化（mm 空间）**，精度更高，弥补像素空间畸变：

  **Pass 1 — 短边处理**（`< card_artifact_min_edge_mm = 25mm`）  
  对每条最短边，计算两端点转角 `_ta`、`_tb`：

  ```
  if 边长 >= _min_e_force(12mm) AND (ta >= 20° OR tb >= 20°):
      保留（真实短切边，如 K 牌切口）
  elif |sin(夹角)| > 0.26（两侧线不平行）:
      两侧边延长线求交点 → 替换两端点，n→n-1  ← 圆角稳定化核心
  else（近平行）:
      双删或单删（保留 ≥3 边约束）
  ```

  **Pass 2 — 近共线顶点**：转角 < 20° → 单点删除。

  **白纸路径**：边长 > `rect_min_edge_mm`（15mm）保证，仅做近共线清理。

### 5.2 裁切边判定

`_is_cut_edge()`：凸顶点 90° ± 8° → 矩形边界角；其余凸顶点 + 反射顶点 → 裁切边。  
DFS 接合优先只使用裁切边（该片有 ≥2 条裁切边时约束，1 条时放开防误判）。

### 5.3 DFS 回溯

```
piece 0 固定原点
  for 已放边 (placed_idx, pe) in 长边优先顺序:
    for covering in _iter_edge_coverings(lp, unplaced, ...):
      ① 梯度纹理剪枝（仅扑克）
      ② 刚性对齐：_rigid_edge_alignment → Placement
      ③ 包围盒早剪枝（minAreaRect，长边 ≤152mm，短边 ≤108mm）
      ④ 重叠检查（> 300mm² 则剪枝）
      if 全部放完 → _check_rectangle() 验证
```

**状态上限**：`max_states = 40000`；候选上限 80 个。  
**早退条件**：≥8 个候选且最优分 < 5，且不存在两条相差 ≤5mm 的未探宿主边。

### 5.4 边覆盖枚举 `_iter_edge_coverings()`

| 类型 | 容差 |
|---|---|
| 1:1 匹配 | `max(4.0mm, 7% × min(lq, lp))` |
| 复合缝（n 片共同覆盖一条已放边） | `max(3.5mm, 7% × lp)` |

复合缝各片贡献 ≥ `min_composite_edge`（5mm），防噪声伪边。

### 5.5 矩形验证 `_check_rectangle()`

全部满足才通过：
1. 凸包 / minAreaRect ≥ `card_rect_fill_min`（0.92）
2. 各片总面积 / minAreaRect ≥ `card_direct_fill_min`（0.84）——防凸包内部间隙
3. 宽高在目标范围（60-148mm × 40-105mm）
4. `|凸包面积 - 总面积| / 凸包面积 < 0.20`——排除重叠/大间隙

综合分 = `100×面积误差 + 200×(1 - 矩形度)` + 复合缝惩罚（每个组合 +100）。

### 5.6 纹理重排序（扑克专用）

前 `top_k = 5` 几何候选叠加纹理评分：

```
J_total = J_shape + 0.5 × J_texture
```

`J_texture`：把各片 `texture_bgr` 按 placement 变换渲染到公共画布 → Sobel 梯度 → 沿每条缝两侧采样 → 梯度方向连续性 × 0.6 + 幅度不对称性 × 0.4，按缝长加权平均。

### 5.7 NCC 角标对称验证（扑克专用）

标准扑克牌角标满足 **180° 旋转对称**。取最优候选验证：

1. `minAreaRect` 得牌面轴方向 `(d_u, d_v)`；角点吸附到实际多边形极值顶点（避免落在缝隙）。
2. 两对角位置（左上 / 右下）各采 20×22mm 角区灰度图。
3. 右下 patch 旋转 180° 后用 `matchTemplate(TM_CCOEFF_NORMED)` 搜索（±8mm 窗口容忍顶点定位误差）。
4. 枚举 4 种 (对角线 × 宽高方向) 组合取最高 NCC；对比度优先（`score = NCC × (std0 + std1)`）。
5. `NCC ≥ 0.40` 且 `std > 5` → 通过。  
   失败时保持几何最优候选（不回退）并打印警告。

### 5.8 输出

`minAreaRect` 旋转水平后平移到拼图区目标位置，SVD 求每片旋转角，生成 `PlacementResult`（吸取点、目标质心、旋转角）。

---

## 六、关键参数速查

| 参数 | 文件 | 默认值 | 含义 |
|---|---|---|---|
| `card_artifact_min_mm` | config.py | 13.0mm | detection.py 像素空间粗筛 |
| `card_artifact_min_edge_mm` | config.py | 25.0mm | b_solver mm 空间短边阈值 |
| `card_artifact_force_edge_mm` | b_solver | 12.0mm | 强制删除（不检查转角） |
| `card_artifact_min_turn_deg` | config.py | 20.0° | 真实角判据 |
| `card_direct_fill_min` | config.py | 0.84 | 直接填充率下限 |
| `card_rect_fill_min` | config.py | 0.92 | 矩形度下限 |
| `min_composite_edge_mm` | config.py | 5.0mm | 复合缝单片贡献下限 |
| NCC 阈值 | b_solver | 0.40 | 角标对称验证通过线 |
| NCC 搜索窗口 | b_solver | ±8mm | matchTemplate 容差 |
| DFS 状态上限 | b_solver | 40000 | 防超时 |
