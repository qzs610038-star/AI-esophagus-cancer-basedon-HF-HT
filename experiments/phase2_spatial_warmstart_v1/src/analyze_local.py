"""Local-only fixed beta=1 smoothing diagnostic for a returned raw run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.metrics import compute_regression_metrics
    from src.predict import fixed_beta_one_smoothing
    from src.spatial import SpatialGraph
else:
    from .metrics import compute_regression_metrics
    from .predict import fixed_beta_one_smoothing
    from .spatial import SpatialGraph


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _graph(path: Path, graph_parameters: dict) -> SpatialGraph:
    with np.load(path, allow_pickle=False) as arrays:
        patient = tuple(np.asarray(arrays["patient"], dtype=str).tolist())
        spots = tuple(np.asarray(arrays["patch_stem"], dtype=str).tolist())
        return SpatialGraph(
            neighbor_index=np.asarray(arrays["neighbor_index"]),
            neighbor_weight=np.asarray(arrays["neighbor_weight"]),
            neighbor_mask=np.asarray(arrays["neighbor_mask"]),
            self_weight=np.asarray(arrays["self_weight"]), degree=np.asarray(arrays["degree"]),
            point_ids=spots, patient_ids=patient,
            partitions=tuple("internal_val" for _ in spots),
            neighbor_distance_native_steps=np.asarray(arrays["neighbor_distance_native_steps"]),
            neighbor_cosine_similarity=np.asarray(arrays["neighbor_cosine_similarity"]),
            radius_in_native_steps=float(graph_parameters["radius_in_native_steps"]),
            max_neighbors=int(graph_parameters["max_neighbors"]),
            distance_sigma=float(graph_parameters["distance_sigma"]),
            image_temperature=float(graph_parameters["image_temperature"]),
            self_raw_weight=float(graph_parameters["self_raw_weight"]),
        )


def run_fixed_smoothing(run_dir: str | Path, *, arm: str = "spatial_residual_only") -> dict:
    run_dir = Path(run_dir).resolve()
    config = _read_json(run_dir / "config.json")
    raw = run_dir / "raw" / arm / "seed_42"
    prediction_path = raw / "step0_predictions.npz"
    graph_path = raw / "graph_internal_val.npz"
    if not prediction_path.is_file() or not graph_path.is_file():
        raise FileNotFoundError(
            f"fixed smoothing requires {prediction_path} and {graph_path} from a spatial arm"
        )
    graph = _graph(graph_path, config["parameters"]["graph"])
    with np.load(prediction_path, allow_pickle=False) as arrays:
        prediction = np.asarray(arrays["prediction_internal_val"], dtype=np.float64)
        target = np.asarray(arrays["target_internal_val"], dtype=np.float64)
        patient = np.asarray(arrays["patient_internal_val"], dtype=str)
        spot = np.asarray(arrays["patch_stem_internal_val"], dtype=str)
        pathways = np.asarray(arrays["pathway_names"], dtype=str)
    smoothed = fixed_beta_one_smoothing(prediction, graph)
    report = {
        "diagnostic": "fixed_prediction_smoothing", "beta": 1.0,
        "formula": "p + sum_j a_ij * (p_j - p_i)",
        "self_weight_included": True, "used_for_selection": False,
        "split": "internal_val", "baseline": compute_regression_metrics(prediction, target, patient),
        "smoothed": compute_regression_metrics(smoothed, target, patient),
        "external_status": "not_computed_without_matching_external_graph_and_predictions",
    }
    output = run_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "fixed_smoothing.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output / "fixed_smoothing_predictions.npz", prediction=smoothed, target=target,
        patient=patient, patch_stem=spot, pathway_names=pathways,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arm", default="spatial_residual_only")
    args = parser.parse_args()
    print(json.dumps(run_fixed_smoothing(args.run_dir, arm=args.arm), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
