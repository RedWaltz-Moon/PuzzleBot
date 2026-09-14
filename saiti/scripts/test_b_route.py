#!/usr/bin/env python3
"""B路单元测试 — 合成顶点（无重叠分割）验证算法正确性。

矩形 100×59.1mm 分成 4 个无重叠碎片：
  P1: 三角形 (0,0)→(79.8,0)→(0,59.1)
  P2: 四边形 (79.8,0)→(63,37.1)→(63,59.1)→(0,59.1)   [中间连接件]
  P3: 四边形 (79.8,0)→(100,0)→(100,37.1)→(63,37.1)
  P4: 矩形   (63,37.1)→(100,37.1)→(100,59.1)→(63,59.1)

切割边（反平行对）：
  P1e1 ↔ P2e3  (斜边 99.3mm)
  P2e0 ↔ P3e3  (短斜边 40.7mm)
  P3e2 ↔ P4e0  (水平 37.0mm)
  P2e1 ↔ P4e3  (竖直 22.0mm)
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from vision.solver import _b_candidates, _b_matching_sets, _b_assemble, _ensure_cw

# ── 正确的无重叠顶点（CW，y向下坐标系）──────────────────────────────────────
P1_orig = np.array([[0.0, 0.0], [79.8, 0.0], [0.0, 59.1]])
P2_orig = np.array([[79.8, 0.0], [63.0, 37.1], [63.0, 59.1], [0.0, 59.1]])
P3_orig = np.array([[79.8, 0.0], [100.0, 0.0], [100.0, 37.1], [63.0, 37.1]])
P4_orig = np.array([[63.0, 37.1], [100.0, 37.1], [100.0, 59.1], [63.0, 59.1]])

originals = [P1_orig, P2_orig, P3_orig, P4_orig]
names = ['P1', 'P2', 'P3', 'P4']

# ── 验证面积（合计应 ≈ 5910 mm²）────────────────────────────────────────────
def shoelace(vv):
    n = len(vv)
    s = sum(float(vv[i][0])*float(vv[(i+1)%n][1]) - float(vv[(i+1)%n][0])*float(vv[i][1])
            for i in range(n))
    return s / 2.0

print("=== 面积验证 ===")
total_area = 0.0
for k, v in enumerate(originals):
    a = abs(shoelace(v))
    total_area += a
    print(f"  {names[k]}: {a:.1f} mm2")
print(f"  Total: {total_area:.1f} mm2  (expect ~5910)")
assert abs(total_area - 5910) < 30, f"area wrong: {total_area:.1f}"

# ── 随机打散（旋转 + 平移）───────────────────────────────────────────────────
np.random.seed(42)

def rand_scatter(vv):
    """随机旋转 + 随机平移，模拟相机检测到的碎片位置。"""
    angle = np.random.uniform(0, 2 * math.pi)
    tx, ty = np.random.uniform(-200, 200, 2)
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return (vv @ R.T) + np.array([tx, ty])

scattered = [rand_scatter(v.copy()) for v in originals]
verts_list = [_ensure_cw(v.copy()) for v in scattered]

print("\n=== 打散后边长 ===")
for k, vv in enumerate(verts_list):
    n = len(vv)
    lens = [float(np.linalg.norm(vv[(i+1)%n] - vv[i])) for i in range(n)]
    print(f"  {names[k]}: {[f'{l:.1f}' for l in lens]}")

# ── 候选边对 ────────────────────────────────────────────────────────────────
print("\n=== 候选边对 ===")
cands = _b_candidates(verts_list)
print(f"共 {len(cands)} 条:")
for rel, i, ei, j, ej in cands[:15]:
    ni, nj = len(verts_list[i]), len(verts_list[j])
    la = float(np.linalg.norm(verts_list[i][(ei+1)%ni] - verts_list[i][ei]))
    lb = float(np.linalg.norm(verts_list[j][(ej+1)%nj] - verts_list[j][ej]))
    print(f"  {names[i]}e{ei}({la:.1f}) <-> {names[j]}e{ej}({lb:.1f})  rel={rel:.3f}")

# ── 穷举组合 ────────────────────────────────────────────────────────────────
print("\n=== 穷举组合 ===")
best_score, best_matches, best_placed = 1e9, None, None
tried = 0
for matches in _b_matching_sets(len(verts_list), cands):
    tried += 1
    score, placed = _b_assemble(verts_list, matches, total_area)
    if score < best_score:
        best_score, best_matches, best_placed = score, list(matches), placed

print(f"尝试 {tried} 个组合")
if best_matches:
    pair_str = [f"{names[i]}e{ei}<->{names[j]}e{ej}"
                for _, i, ei, j, ej in best_matches]
    print(f"最优 score={best_score:.1f}: {pair_str}")

    # ── 每片拼合后的位置 ───────────────────────────────────────────────────
    print("\n每片中心（拼合后）:")
    for k, v in best_placed.items():
        cx, cy = v.mean(0)
        mn, mx = v.min(0), v.max(0)
        print(f"  {names[k]}: center=({cx:.1f},{cy:.1f})  bbox=[{mn[0]:.1f}~{mx[0]:.1f}, {mn[1]:.1f}~{mx[1]:.1f}]")

    # ── 整体拼合矩形 ───────────────────────────────────────────────────────
    import cv2
    all_pts = np.vstack(list(best_placed.values())).astype(np.float32)
    rect = cv2.minAreaRect(all_pts)
    rw, rh = sorted(rect[1], reverse=True)
    fill = total_area / max(rw * rh, 1)
    print(f"\n拼合矩形: {rw:.1f}×{rh:.1f} mm")
    print(f"fill = {fill:.3f}  (期望 ≥ 0.95)")

    if rw > 85 and rh > 50 and fill > 0.90:
        print("\n[OK] 算法正确")
    else:
        print(f"\n[FAIL] 算法错误：期望 ~100×59mm, fill≥0.95，实际 {rw:.1f}×{rh:.1f}, fill={fill:.3f}")
else:
    print("未找到有效组合！")
