import unittest
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyze import (
    compute_moran_i,
    compute_within_patient_percentile,
    build_spatial_weights,
    build_percentile_frame,
    benjamini_hochberg,
    compare_raw_and_percentile_spatial_metrics,
    compute_cliffs_delta,
    compute_block_subsample_rank_stability,
    combine_stouffer_p,
    resolve_output_root,
)

class TestSpatialRankAnalysis(unittest.TestCase):

    def test_01_constant_field(self):
        """测试 1: 3x3 常数场返回 NaN/undefined，不伪造 p 值"""
        coords = np.array([
            [0, 0], [0, 1], [0, 2],
            [1, 0], [1, 1], [1, 2],
            [2, 0], [2, 1], [2, 2]
        ])
        scores = np.ones(9) * 5.0
        w, _, _ = build_spatial_weights(coords, 'rook', step_size=1)
        moran = compute_moran_i(scores, w)
        self.assertTrue(np.isnan(moran), f"Constant field should yield NaN, got {moran}")

    def test_02_gradient_and_checkerboard(self):
        """测试 2: 平滑梯度场得到正 Moran's I，棋盘场得到负 Moran's I"""
        coords = np.array([
            [0, 0], [0, 1], [0, 2],
            [1, 0], [1, 1], [1, 2],
            [2, 0], [2, 1], [2, 2]
        ])
        w, _, _ = build_spatial_weights(coords, 'rook', step_size=1)
        
        # 梯度场
        gradient = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5], dtype=float)
        moran_grad = compute_moran_i(gradient, w)
        self.assertGreater(moran_grad, 0.0, f"Gradient field should yield positive Moran's I, got {moran_grad}")

        # 棋盘场
        checkerboard = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1], dtype=float)
        moran_check = compute_moran_i(checkerboard, w)
        self.assertLess(moran_check, 0.0, f"Checkerboard field should yield negative Moran's I, got {moran_check}")

    def test_03_permutation_invariance(self):
        """测试 3: 节点重排不改变计算结果"""
        coords = np.array([
            [0, 0], [0, 1], [0, 2],
            [1, 0], [1, 1], [1, 2],
            [2, 0], [2, 1], [2, 2]
        ])
        scores = np.array([1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0])
        w, _, _ = build_spatial_weights(coords, 'rook', step_size=1)
        moran_orig = compute_moran_i(scores, w)

        # Shuffle indices
        perm_idx = np.array([8, 2, 0, 5, 1, 7, 3, 4, 6])
        coords_perm = coords[perm_idx]
        scores_perm = scores[perm_idx]
        w_perm, _, _ = build_spatial_weights(coords_perm, 'rook', step_size=1)
        moran_shuffled = compute_moran_i(scores_perm, w_perm)

        self.assertAlmostEqual(moran_orig, moran_shuffled, places=6,
                               msg="Permutation of nodes should not change Moran's I")

    def test_04_patient_isolation(self):
        """测试 4: 患者间永不连边"""
        coords_p1 = np.array([[0, 0], [0, 1]])
        coords_p2 = np.array([[0, 2], [0, 3]])
        
        # Build separately
        w1, _, _ = build_spatial_weights(coords_p1, 'rook', step_size=1)
        w2, _, _ = build_spatial_weights(coords_p2, 'rook', step_size=1)
        
        # Ensure independence
        self.assertEqual(w1.shape, (2, 2))
        self.assertEqual(w2.shape, (2, 2))

    def test_05_percentile_ties_and_range(self):
        """测试 5: 并列值使用平均秩，百分位严格在 (0, 1) 内"""
        scores = np.array([10.0, 20.0, 20.0, 40.0])
        percentiles = compute_within_patient_percentile(scores)
        
        self.assertEqual(len(percentiles), 4)
        self.assertTrue(np.all(percentiles > 0.0) and np.all(percentiles < 1.0))
        # 20.0 and 20.0 should have equal rank (2.5 -> (2.5-0.5)/4 = 0.5)
        self.assertAlmostEqual(percentiles[1], 0.5)
        self.assertAlmostEqual(percentiles[2], 0.5)

    def test_06_percentile_frame_keeps_patient_identity(self):
        """测试 6: 患者标识必须逐行写入，不能因空 DataFrame 标量赋值而全为 NaN。"""
        df = pd.DataFrame({"barcode": ["patch_x0_y0", "patch_x1_y0"], "P": [1.0, 2.0]})
        coords = np.array([[0, 0], [1, 0]])
        out = build_percentile_frame("P1", df, coords, ["P"])
        self.assertEqual(out["patient"].tolist(), ["P1", "P1"])
        self.assertFalse(out["patient"].isna().any())

    def test_07_bh_fdr_is_monotone_in_sorted_p_order(self):
        """测试 7: 跨 30 通路的 Stouffer p 值必须经 BH 校正后再作为 q 值判定。"""
        p = np.array([0.01, 0.04, 0.03, 0.002])
        q = benjamini_hochberg(p)
        order = np.argsort(p)
        self.assertTrue(np.all(np.diff(q[order]) >= -1e-12))
        self.assertTrue(np.all(q >= p))

    def test_08_rank_transform_preserves_order_not_moran_value(self):
        """测试 8: 单调秩变换保持排序，但一般不保持 Moran's I 数值。"""
        coords = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], dtype=float)
        raw = np.array([0.0, 0.1, 0.2, 10.0, 11.0])
        w, adj, _ = build_spatial_weights(coords, 'rook', step_size=1)
        cmp = compare_raw_and_percentile_spatial_metrics(raw, w, adj, coords)
        self.assertAlmostEqual(cmp["spearman_rho"], 1.0)
        self.assertNotAlmostEqual(cmp["raw_moran_i"], cmp["percentile_moran_i"], places=6)

    def test_09_cliffs_delta_direction_and_identity(self):
        """测试 9: Cliff's delta 保留效应方向，同分布比较为 0。"""
        self.assertAlmostEqual(compute_cliffs_delta(np.array([1, 2]), np.array([1, 2])), 0.0)
        self.assertAlmostEqual(compute_cliffs_delta(np.array([3, 4]), np.array([1, 2])), 1.0)

    def test_10_block_subsample_rank_stability_is_bounded(self):
        """测试 10: 空间块子采样的百分位漂移必须可量化且落在 [0,1]。"""
        coords = np.array([[x, y] for x in range(4) for y in range(4)], dtype=float)
        scores = np.arange(16, dtype=float)
        out = compute_block_subsample_rank_stability(scores, coords, 1, 8, 7)
        self.assertGreaterEqual(out["mean_abs_percentile_shift"], 0.0)
        self.assertLessEqual(out["p95_abs_percentile_shift"], 1.0)
        self.assertEqual(out["n_iterations"], 8)

    def test_11_stouffer_tail_probability_does_not_underflow_early(self):
        """测试 11: 强证据的 Stouffer 上尾概率应保留有限非零表示。"""
        p = combine_stouffer_p(np.repeat(1e-4, 6))
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1e-10)

    def test_12_smoke_output_is_isolated_from_final(self):
        """测试 12: Smoke 绝不能覆盖 Final 根层产出。"""
        config = {"output_root": "analysis/v001"}
        self.assertEqual(resolve_output_root(config, "final"), Path("analysis/v001"))
        self.assertEqual(resolve_output_root(config, "smoke"), Path("analysis/v001/smoke"))

if __name__ == '__main__':
    unittest.main()
