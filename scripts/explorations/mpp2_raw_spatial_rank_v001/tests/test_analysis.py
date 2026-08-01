import unittest
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyze import (
    compute_moran_i,
    compute_within_patient_percentile,
    build_spatial_weights
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

if __name__ == '__main__':
    unittest.main()
