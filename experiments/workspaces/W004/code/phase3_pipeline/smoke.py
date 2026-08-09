"""Synthetic end-to-end smoke runner; never consumes project or server data."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from .baselines import fit_fold_baseline
from .contracts import PatchBag, validate_fold_assignments, validate_patch_bag
from .engine import SlideExample, annotate_oof, train_single_fold
from .evaluation import binary_metrics, patient_cluster_bootstrap_auc, validate_oof_table
from .models import MILBagClassifier, PathologyPathwayResidualAdapter
from .pathway_graph import build_gene_overlap_graph, fixed_graph_diffusion
from .transforms import slice_percentile_rank, spatial_smooth


def run_smoke(seed: int = 7) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    rows = []
    examples = []
    pooled = []
    labels = []
    for index in range(12):
        case_id = f"case_{index // 2}"
        slide_id = f"slide_{index}"
        label = index % 2
        image = rng.normal(label * 0.25, 1, size=(6, 16)).astype("float32")
        pathway = rng.normal(label * 0.2, 1, size=(6, 30)).astype("float32")
        coords = np.stack((np.arange(6), np.zeros(6)), axis=1).astype("float32")
        validate_patch_bag(PatchBag(case_id, slide_id, image, pathway, coords, label, 1 if label else 0))
        ordinal = slice_percentile_rank(pathway, [slide_id] * len(pathway))
        smoothed = spatial_smooth(pathway, coords, k=2, lam=0.2)
        assert np.asarray(ordinal).shape == pathway.shape and np.asarray(smoothed).shape == pathway.shape
        examples.append(SlideExample(case_id, slide_id, torch.from_numpy(image), torch.from_numpy(pathway), label))
        pooled.append(np.concatenate((image.mean(0), pathway.mean(0))))
        labels.append(label)
        rows.append({"case_id": case_id, "slide_id": slide_id, "split": ("train" if index < 8 else "val" if index < 10 else "test")})
    split_audit = validate_fold_assignments(pd.DataFrame(rows), split_unit="slide")
    baseline = fit_fold_baseline("logistic", np.vstack(pooled[:8]), np.array(labels[:8]), np.vstack(pooled[8:10]), np.vstack(pooled[10:]), random_state=seed)
    model = MILBagClassifier(16, 30, hidden_dim=12, fusion="adaptive_gated", pooling="attention")
    trained = train_single_fold(model, examples[:8], examples[8:10], examples[10:], seed=seed, epochs=2, patience=2)
    oof = annotate_oof(trained.test_predictions, task="pCR", seed=seed, fold=0, model_name="adaptive_gated")
    validate_oof_table(oof)
    metrics = binary_metrics(oof["y_true"].to_numpy(), oof["y_pred"].to_numpy())
    bootstrap = patient_cluster_bootstrap_auc(oof, n_bootstrap=20, seed=seed)
    adapter = PathologyPathwayResidualAdapter(image_dim=16)
    tensor_image = torch.from_numpy(examples[0].pathology.numpy()).unsqueeze(0)
    tensor_pathway = torch.from_numpy(examples[0].pathway.numpy()).unsqueeze(0)
    adapted = adapter(tensor_image, tensor_pathway, tensor_pathway.sigmoid(), tensor_pathway.tanh(), tensor_pathway.sigmoid())
    if not torch.equal(adapted, tensor_image):
        raise AssertionError("zero-initialized pathology residual is not identity")
    names = [f"path_{i}" for i in range(30)]
    gene_sets = {name: [f"gene_{i}", "shared"] for i, name in enumerate(names)}
    graph = build_gene_overlap_graph(names, gene_sets)
    diffused = fixed_graph_diffusion(examples[0].pathway.numpy(), graph)
    return {
        "mode": "synthetic_smoke_only",
        "split_audit": split_audit,
        "baseline_validation_count": int(len(baseline.validation_probability)),
        "torch_epochs": len(trained.history),
        "test_metrics": metrics,
        "bootstrap": bootstrap,
        "pathway_graph_shape": list(graph.shape),
        "diffused_shape": list(diffused.shape),
        "real_training": False,
        "server_access": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", action="store_true", help="print machine-readable smoke summary")
    args = parser.parse_args()
    result = run_smoke(args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else "Phase3 synthetic smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
