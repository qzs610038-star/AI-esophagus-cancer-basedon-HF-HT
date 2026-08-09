from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.workspaces.W004.code.phase3_pipeline.baselines import fit_fold_baseline
from experiments.workspaces.W004.code.phase3_pipeline.contracts import (
    PatchBag,
    assert_train_only_fit,
    validate_fold_assignments,
    validate_manifest,
    validate_patch_bag,
    validate_pathway_names,
)
from experiments.workspaces.W004.code.phase3_pipeline.engine import (
    JointSlideExample,
    SlideExample,
    annotate_fold_predictions,
    annotate_oof,
    train_joint_single_fold,
    train_single_fold,
)
from experiments.workspaces.W004.code.phase3_pipeline.evaluation import (
    apply_calibration,
    conformal_risk_coverage,
    fit_calibration,
    leave_one_patient_out_auc,
    paired_fold_comparison,
    patient_cluster_bootstrap_auc,
    validate_repeated_holdout_predictions,
    validate_oof_table,
)
from experiments.workspaces.W004.code.phase3_pipeline.features import (
    ExternalEvaluationGuard,
    Phase3FeaturePipeline,
)
from experiments.workspaces.W004.code.phase3_pipeline.interpretability import (
    build_blinded_review_index,
    summarize_blinded_reviews,
)
from experiments.workspaces.W004.code.phase3_pipeline.io import (
    load_verified_patch_bag,
    sha256_file,
)
from experiments.workspaces.W004.code.phase3_pipeline.models import (
    JointEndpointMILModel,
    MILBagClassifier,
    PathologyPathwayResidualAdapter,
)
from experiments.workspaces.W004.code.phase3_pipeline.pathway_graph import (
    build_gene_overlap_graph,
    fixed_graph_diffusion,
)
from experiments.workspaces.W004.code.phase3_pipeline.smoke import run_smoke


def test_patch_manifest_and_nested_endpoint_contracts():
    bag = PatchBag("p1", "s1", np.ones((2, 4)), np.ones((2, 30)), np.ones((2, 2)), 1, 1)
    assert validate_patch_bag(bag) is bag
    with pytest.raises(ValueError, match="alignment"):
        validate_patch_bag(PatchBag("p", "s", np.ones((2, 4)), np.ones((3, 30)), np.ones((2, 2))))
    with pytest.raises(ValueError, match="nesting"):
        validate_patch_bag(PatchBag("p", "s", np.ones((2, 4)), np.ones((2, 30)), np.ones((2, 2)), 1, 0))
    table = pd.DataFrame({"case_id": ["p", "p"], "slide_id": ["s1", "s2"], "pcr": [1, 1], "mpr": [1, 1]})
    assert len(validate_manifest(table)) == 2
    assert len(validate_pathway_names([f"p{i}" for i in range(30)])) == 30


def test_slide_split_reports_patient_overlap_but_patient_split_rejects_it():
    assignments = pd.DataFrame(
        {"case_id": ["p1", "p1", "p2"], "slide_id": ["s1", "s2", "s3"], "split": ["train", "test", "val"]}
    )
    audit = validate_fold_assignments(assignments, split_unit="slide")
    assert audit["patient_overlap_status"] == "WARN_REVIEW_REQUIRED"
    with pytest.raises(ValueError, match="leakage"):
        validate_fold_assignments(assignments, split_unit="patient")
    with pytest.raises(ValueError, match="outside train"):
        assert_train_only_fit(["s1", "s2"], {"s1": "train", "s2": "test"})


def test_tabular_preprocessing_is_fitted_only_on_training_rows():
    x_train = np.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    result = fit_fold_baseline("pca_logistic", x_train, np.array([0, 0, 1, 1]), np.array([[0.0, 0.5]]), np.array([[3.0, 1.0]]), random_state=3)
    scaler = result.fitted_model.named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, x_train.mean(axis=0))
    assert result.validation_probability.shape == (1,)


def _examples(seed=3):
    generator = torch.Generator().manual_seed(seed)
    items = []
    for i in range(8):
        items.append(SlideExample(f"p{i // 2}", f"s{i}", torch.randn(4, 6, generator=generator), torch.randn(4, 30, generator=generator), i % 2))
    return items


def test_fold_engine_freezes_on_validation_and_emits_oof():
    items = _examples()
    model = MILBagClassifier(6, hidden_dim=8, fusion="concat", pooling="mean")
    result = train_single_fold(model, items[:4], items[4:6], items[6:], seed=5, epochs=2, patience=2)
    oof = annotate_oof(result.test_predictions, task="pCR", seed=5, fold=0, model_name="concat")
    assert len(result.history) == 2
    assert len(validate_oof_table(oof)) == 2


def test_fold_engine_requires_explicit_patient_overlap_policy():
    items = _examples()
    overlapping_test = [SlideExample(items[0].case_id, "new_slide", items[6].pathology, items[6].pathway, items[6].label), items[7]]
    with pytest.raises(ValueError, match="patient overlap"):
        train_single_fold(MILBagClassifier(6, hidden_dim=8), items[:4], items[4:6], overlapping_test, seed=1, epochs=1)
    result = train_single_fold(MILBagClassifier(6, hidden_dim=8), items[:4], items[4:6], overlapping_test, seed=1, epochs=1, patient_overlap_policy="allow_with_warning")
    assert result.evidence_scope == "slide_repeated_holdout_patient_nonindependent" and result.warnings


def test_joint_monotonic_training_and_unknown_label_mask():
    items = _examples()
    joint = [JointSlideExample(x.case_id, x.slide_id, x.pathology, x.pathway, float(x.label) if i != 5 else float("nan"), float(x.label)) for i, x in enumerate(items)]
    result = train_joint_single_fold(JointEndpointMILModel(6, hidden_dim=8), joint[:4], joint[4:6], joint[6:], seed=2, epochs=1)
    assert (result.test_predictions["pcr_pred"] <= result.test_predictions["mpr_pred"]).all()
    duplicate = [joint[0], joint[0]]
    with pytest.raises(ValueError, match="duplicate"):
        train_joint_single_fold(JointEndpointMILModel(6, hidden_dim=8), duplicate, joint[4:6], joint[6:], seed=2, epochs=1)
    bad = JointSlideExample("x", "bad", torch.full((4, 6), float("nan")), torch.ones(4, 30), 0.0, 0.0)
    with pytest.raises(ValueError, match="non-finite"):
        train_joint_single_fold(JointEndpointMILModel(6, hidden_dim=8), [bad], joint[4:6], joint[6:], seed=2, epochs=1)


def test_zero_init_pathology_residual_is_exact_identity():
    image = torch.randn(2, 3, 6)
    views = [torch.rand(2, 3, 30) for _ in range(4)]
    adapter = PathologyPathwayResidualAdapter(6)
    assert torch.equal(adapter(image, *views), image)


def test_patient_bootstrap_calibration_and_conformal():
    oof = pd.DataFrame(
        {"case_id": ["a", "a", "b", "b", "c", "c", "d", "d"], "slide_id": list("abcdefgh"), "y_true": [0, 0, 1, 1, 0, 0, 1, 1], "y_pred": [.1, .2, .8, .7, .3, .2, .9, .8]}
    )
    interval = patient_cluster_bootstrap_auc(oof, n_bootstrap=50, seed=2)
    assert interval["valid_draws"] > 0 and interval["ci95_low"] <= interval["ci95_high"]
    calibration = fit_calibration(np.array([0, 0, 1, 1]), np.array([.1, .3, .7, .9]))
    calibrated = apply_calibration(np.array([.2, .8]), calibration)
    assert np.all((calibrated > 0) & (calibrated < 1))
    conformal = conformal_risk_coverage(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9]), np.array([.4, .6]))
    assert conformal["set_size"].shape == (2,)


def test_repeated_holdout_is_not_mislabeled_partition_oof():
    frames = []
    for fold in (0, 1):
        prediction = pd.DataFrame({"case_id": ["a", "b"], "slide_id": ["s1", "s2"], "y_true": [0, 1], "y_pred": [.2, .8]})
        frames.append(annotate_fold_predictions(prediction, task="pCR", seed=1, fold=fold, model_name="m", prediction_kind="repeated_holdout_test", evidence_scope="slide_repeated_holdout_patient_nonindependent", warnings=["WARN_PATIENT_NONINDEPENDENCE:a"]))
    audit = validate_repeated_holdout_predictions(pd.concat(frames, ignore_index=True))
    assert audit["slides_repeated_across_folds"] == 2
    assert audit["evidence_scope"] == "repeated_holdout_not_partition_oof"


def test_verified_feature_io_and_hash_failure(tmp_path: Path):
    paths = {}
    for role, value in (("pathology", torch.ones(2, 4)), ("pathway", torch.ones(2, 30)), ("coordinates", torch.ones(2, 2))):
        path = tmp_path / f"{role}.pt"
        torch.save(value, path)
        paths[f"{role}_path"] = str(path)
        paths[f"{role}_sha256"] = sha256_file(path)
    record = {"case_id": "p", "slide_id": "s", **paths, "pcr": 0, "mpr": 0}
    assert load_verified_patch_bag(record).pathway.shape == (2, 30)
    record["pathway_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        load_verified_patch_bag(record)


def test_fixed_pathway_graph_is_ordered_and_shape_safe():
    names = ["a", "b"]
    graph = build_gene_overlap_graph(names, {"a": ["x", "y"], "b": ["y", "z"]})
    assert graph.shape == (2, 2) and graph[0, 1] == pytest.approx(1 / 3)
    assert fixed_graph_diffusion(np.ones((3, 2)), graph).shape == (3, 2)


def test_fitted_feature_pipeline_connects_phase2_to_phase3_without_external_fit():
    rng = np.random.default_rng(4)
    morphology = rng.normal(size=(12, 4))
    prediction = rng.normal(size=(12, 30))
    truth = prediction + rng.normal(scale=.1, size=(12, 30))
    patients = [f"p{i // 4}" for i in range(12)]
    pipeline = Phase3FeaturePipeline(gallery_k=2).fit_phase2_training_fold(morphology, prediction, truth, patients)
    lopo = pipeline.lopo_stability_report()
    assert lopo["pathway_count"] == 30 and lopo["finite"] is True
    bag = PatchBag("external", "slide", rng.normal(size=(5, 4)), rng.normal(size=(5, 30)), np.stack([np.arange(5), np.zeros(5)], axis=1))
    views = ExternalEvaluationGuard(pipeline).transform_only(bag, spatial_lambda=.2, spatial_k=2)
    assert views.absolute.shape == views.ordinal.shape == views.prototype.shape == views.uncertainty.shape == (5, 30)
    with pytest.raises(RuntimeError, match="forbidden"):
        ExternalEvaluationGuard(pipeline).fit(morphology)


def test_paired_oof_patient_deletion_and_blinded_review():
    base = pd.DataFrame({
        "case_id": ["a", "b", "c", "d"], "slide_id": ["s1", "s2", "s3", "s4"],
        "task": ["pCR"] * 4, "y_true": [0, 1, 0, 1], "seed": [1] * 4, "fold": [0] * 4,
    })
    left = base.assign(y_pred=[.2, .7, .4, .6], model="reference")
    right = base.assign(y_pred=[.1, .8, .3, .9], model="candidate")
    paired = paired_fold_comparison(pd.concat([left, right], ignore_index=True), "reference", "candidate")
    assert paired.loc[0, "delta_auc"] >= 0
    assert len(leave_one_patient_out_auc(right)) == 4
    manifest = pd.DataFrame({"case_id": ["a", "b"], "slide_id": ["s1", "s2"], "model": ["m1", "m2"], "heatmap_path": ["a.png", "b.png"]})
    reviewer, key = build_blinded_review_index(manifest, seed=2)
    reviews = reviewer[["review_id"]].assign(reviewer_id="r1", score=[1.0, 2.0])
    assert summarize_blinded_reviews(reviews, key)["n_reviews"].sum() == 2


def test_synthetic_smoke_cannot_claim_real_training_or_server_access():
    result = run_smoke(seed=9)
    assert result["mode"] == "synthetic_smoke_only"
    assert result["real_training"] is False and result["server_access"] is False
