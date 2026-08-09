import numpy as np
import pytest

from experiments.workspaces.W004.code.phase3_pipeline import transforms


def test_percentile_rank_respects_slice_and_patient_groups():
    values = np.array([[1.0, 4.0], [3.0, 2.0], [9.0, 8.0], [5.0, 6.0]])
    np.testing.assert_allclose(
        transforms.within_group_percentile_rank(values, ["a", "a", "b", "b"]),
        [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
    )
    np.testing.assert_allclose(
        transforms.slice_percentile_rank(values, ["a", "a", "b", "b"]),
        transforms.within_group_percentile_rank(values, ["a", "a", "b", "b"]),
    )
    np.testing.assert_allclose(
        transforms.patient_percentile_rank(values, ["p1", "p1", "p2", "p2"]),
        transforms.within_group_percentile_rank(values, ["p1", "p1", "p2", "p2"]),
    )


def test_spatial_smoothing_identity_and_coordinate_permutation_hook():
    features = np.array([[0.0], [2.0], [10.0]])
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [20.0, 0.0]])
    np.testing.assert_array_equal(transforms.spatial_smooth(features, coords, lam=0.0), features)
    out = transforms.spatial_smooth(features, coords, k=1, lam=1.0)
    np.testing.assert_allclose(out[:, 0], [2.0, 0.0, 2.0])
    permuted = transforms.spatial_smooth(features, coords, k=1, lam=1.0, coordinate_permutation=[2, 1, 0])
    assert not np.array_equal(out, permuted)


def test_residual_calibrator_fit_is_training_only_and_transform_is_stable():
    gallery = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    target = gallery + np.array([10.0, -2.0])
    calibrator = transforms.GalleryResidualCalibrator().fit(gallery, target)
    np.testing.assert_allclose(calibrator.transform(np.array([[7.0, 8.0]])), [[17.0, 6.0]])
    with pytest.raises(RuntimeError):
        transforms.GalleryResidualCalibrator().transform(gallery)
    with pytest.raises(ValueError):
        calibrator.fit(gallery[:2], target)


def test_patient_balanced_prototypes_and_cosine_contrast():
    embeddings = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 3.0]])
    patients = ["a", "a", "b", "b"]
    high = np.array([True, False, True, False])
    high_proto, low_proto = transforms.patient_balanced_prototypes(embeddings, patients, high)
    np.testing.assert_allclose(high_proto, [0.5, 0.5])
    np.testing.assert_allclose(low_proto, [1.5, 1.5])
    contrast = transforms.prototype_cosine_contrast(np.array([[1.0, 0.0]]), np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    np.testing.assert_allclose(contrast, [1.0])


def test_uncertainty_and_control_generators_are_reproducible():
    np.testing.assert_allclose(
        transforms.combine_uncertainty(np.array([1.0]), np.array([2.0]), np.array([3.0]), weights=(1, 1, 2)),
        [2.25],
    )
    np.testing.assert_array_equal(transforms.identity_permutation(5), np.arange(5))
    data = np.arange(12, dtype=float).reshape(4, 3)
    a = transforms.patch_pathway_shuffle(data, seed=7)
    b = transforms.patch_pathway_shuffle(data, seed=7)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(np.sort(a, axis=0), np.sort(data, axis=0))
    controls_a = transforms.covariance_matched_generic_controls(data, n_controls=6, seed=7)
    controls_b = transforms.covariance_matched_generic_controls(data, n_controls=6, seed=7)
    assert controls_a.shape == (6, 30)
    np.testing.assert_array_equal(controls_a, controls_b)


def test_rejects_nonfinite_and_invalid_shapes():
    with pytest.raises(ValueError):
        transforms.within_group_percentile_rank(np.array([[np.nan]]), ["p"])
    with pytest.raises(ValueError):
        transforms.spatial_smooth(np.ones((2, 1)), np.ones((3, 2)))
    with pytest.raises(ValueError):
        transforms.covariance_matched_generic_controls(np.ones((4, 31)), n_controls=1, seed=0)


def test_knn_gallery_pathway_prototypes_and_counterfactuals():
    reference = np.array([[0.0], [1.0], [10.0], [11.0]])
    prediction = np.zeros((4, 2))
    target = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]])
    gallery = transforms.KNNGalleryResidualCalibrator(k=2).fit(reference, prediction, target)
    np.testing.assert_allclose(gallery.transform(np.array([[0.5]]), np.zeros((1, 2))), [[1.0, 2.0]])
    embeddings = np.array([[1.0, .1], [2.0, .1], [.1, 1.0], [.1, 2.0]])
    scores = np.tile(np.array([[0.0], [1.0], [0.0], [1.0]]), (1, 3))
    high, low = transforms.pathway_patient_balanced_prototypes(embeddings, scores, ["a", "a", "b", "b"], high_quantile=.75, low_quantile=.25)
    assert high.shape == low.shape == (3, 2)
    assert transforms.pathway_prototype_contrast(embeddings, high, low).shape == (4, 3)
    matrix = np.arange(12).reshape(4, 3)
    np.testing.assert_array_equal(transforms.permute_pathway_identity(matrix, [2, 0, 1]), matrix[:, [2, 0, 1]])
    shuffled = transforms.shuffle_patch_pathway_pairing(matrix, seed=4)
    assert {tuple(row) for row in shuffled} == {tuple(row) for row in matrix}
