"""CLI for preflight, training, explicit resume, inference, and external evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data import load_bundle, preflight
    from src.metrics import SelectionCandidate, compute_regression_metrics, select_best_candidate
    from src.predict import evaluate_external, fixed_beta_one_smoothing, load_model_bundle, predict_features
    from src.spatial import SpatialPoint, build_spatial_graph
    from src.training import ARM_ORDER, train_arm
else:
    from .data import load_bundle, preflight
    from .metrics import SelectionCandidate, compute_regression_metrics, select_best_candidate
    from .predict import evaluate_external, fixed_beta_one_smoothing, load_model_bundle, predict_features
    from .spatial import SpatialPoint, build_spatial_graph
    from .training import ARM_ORDER, train_arm


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_DIR / "config.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    plan = sub.add_parser("plan", help="show implementation and arm status")
    _common_config(plan)
    check = sub.add_parser("check", help="validate server sources without external labels")
    _common_config(check)
    train = sub.add_parser("train", help="train all arms or one arm from Phase-1 H/C")
    _common_config(train)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--weights-dir", type=Path, required=True)
    train.add_argument("--arm", choices=("all", *ARM_ORDER), default="all")
    train.add_argument("--device")
    resume = sub.add_parser("resume", help="explicitly resume one trajectory from last.pt")
    _common_config(resume)
    resume.add_argument("--run-dir", type=Path, required=True)
    resume.add_argument("--weights-dir", type=Path, required=True)
    resume.add_argument("--arm", choices=ARM_ORDER, required=True)
    resume.add_argument("--checkpoint", type=Path, required=True)
    resume.add_argument("--device")
    predict = sub.add_parser("predict", help="label-free inference from a model bundle")
    predict.add_argument("--bundle", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--device")
    external = sub.add_parser("external-eval", help="evaluate a fixed bundle, never select it")
    external.add_argument("--bundle", type=Path, required=True)
    external.add_argument("--input", type=Path, required=True)
    external.add_argument("--output", type=Path, required=True)
    external.add_argument("--device")
    prepare_xzy = sub.add_parser(
        "prepare-xzy-input",
        help="convert the existing fixed-E5 XZY cache to the external-eval contract",
    )
    prepare_xzy.add_argument("--input", type=Path, required=True)
    prepare_xzy.add_argument("--output", type=Path, required=True)
    prepare_xzy.add_argument("--native-step", type=float, required=True)
    return parser


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values.tolist()
    ], dtype=str)


def _inference_input(path: Path, *, require_targets: bool) -> dict[str, np.ndarray]:
    required = {"features", "graph_features", "patient", "patch_stem", "x", "y", "native_step"}
    with np.load(path, allow_pickle=False) as arrays:
        missing = sorted(required - set(arrays.files))
        if missing:
            raise KeyError(f"inference NPZ is missing fields: {missing}")
        result = {key: np.asarray(arrays[key]) for key in required}
        if "partition" in arrays:
            result["partition"] = np.asarray(arrays["partition"])
        if "target" in arrays:
            result["target"] = np.asarray(arrays["target"], dtype=np.float32)
        elif require_targets:
            raise KeyError("external evaluation input is missing standardized target")
    count = len(result["features"])
    for key, value in result.items():
        if len(value) != count:
            raise ValueError(f"inference field {key} length differs from features")
    return result


def _prepare_xzy_input(source: Path, output: Path, *, native_step: float) -> dict[str, Any]:
    """Map the already-standardized fixed-E5 XZY cache to the inference contract."""

    if not np.isfinite(native_step) or native_step <= 0:
        raise ValueError("native_step must be finite and positive")
    mapping = {
        "features": "cls",
        "graph_features": "prediction_z",
        "target": "target_z",
        "patient": "patient",
        "patch_stem": "patch_stem",
        "x": "x",
        "y": "y",
    }
    with np.load(source, allow_pickle=False) as arrays:
        missing = sorted(set(mapping.values()) - set(arrays.files))
        if missing:
            raise KeyError(f"fixed-E5 XZY cache is missing fields: {missing}")
        data = {target: np.asarray(arrays[source_key]) for target, source_key in mapping.items()}
    count = len(data["features"])
    for key, value in data.items():
        if len(value) != count:
            raise ValueError(f"XZY cache field {key} length differs from features")
    features = np.asarray(data["features"], dtype=np.float32)
    graph_features = np.asarray(data["graph_features"], dtype=np.float32)
    target = np.asarray(data["target"], dtype=np.float32)
    if features.ndim != 2 or graph_features.ndim != 2 or target.shape != graph_features.shape:
        raise ValueError("XZY cache must contain 2D cls and matching prediction_z/target_z")
    if not all(np.isfinite(values).all() for values in (features, graph_features, target)):
        raise ValueError("XZY cache contains non-finite feature, prediction, or target values")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=features,
        graph_features=graph_features,
        target=target,
        patient=_decode(data["patient"]),
        patch_stem=_decode(data["patch_stem"]),
        x=np.asarray(data["x"]),
        y=np.asarray(data["y"]),
        native_step=np.full(count, native_step, dtype=np.float32),
        partition=np.asarray(["external_test"] * count, dtype=str),
    )
    return {
        "status": "ok",
        "output": str(output.resolve()),
        "point_count": count,
        "native_step": float(native_step),
        "target_scale": "source_training_zscore_already_present_in_target_z",
        "external_used_for_selection": False,
    }


def _inference_graph(data: Mapping[str, np.ndarray], parameters: Mapping[str, Any]):
    patients = _decode(data["patient"])
    spots = _decode(data["patch_stem"])
    partitions = _decode(data.get("partition", np.asarray(["inference"] * len(patients))))
    points = [
        SpatialPoint(str(spot), str(patient), str(partition), float(x), float(y))
        for spot, patient, partition, x, y in zip(
            spots, patients, partitions, data["x"].tolist(), data["y"].tolist()
        )
    ]
    native: dict[str, float] = {}
    for patient, step in zip(patients.tolist(), data["native_step"].tolist()):
        previous = native.setdefault(patient, float(step))
        if not np.isclose(previous, float(step)):
            raise ValueError(f"native_step differs within patient={patient}")
    return build_spatial_graph(
        points, np.asarray(data["graph_features"], dtype=np.float32), native_steps=native,
        radius_in_native_steps=float(parameters["radius_in_native_steps"]),
        max_neighbors=int(parameters["max_neighbors"]),
        distance_sigma=float(parameters["distance_sigma"]),
        image_temperature=float(parameters["image_temperature"]),
        self_raw_weight=float(parameters["self_raw_weight"]),
    )


def _update_parent_record(run_dir: Path, checkpoint: Path) -> None:
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        return
    record = _read_json(run_path)
    record["parent_checkpoint"] = str(checkpoint.resolve())
    record["parent_run"] = str(checkpoint.resolve().parents[2]) if len(checkpoint.parents) >= 3 else None
    parent = torch.load(checkpoint, map_location="cpu", weights_only=False)
    record["resumed_arm"] = parent.get("arm")
    _write_json(run_path, record)


def _write_run_outputs(
    config: Mapping[str, Any], run_dir: Path, summaries: list[dict[str, Any]],
    *, parent_checkpoint: Path | None = None,
) -> None:
    candidates = [SelectionCandidate(**summary["best_metrics"]) for summary in summaries]
    selected = select_best_candidate(candidates, architecture_tiebreak=True)
    by_arm = {summary["arm"]: summary for summary in summaries}
    metric_names = ("patient_macro_pathway_pcc", "flattened_pooled_pcc", "zMSE")
    comparisons: dict[str, Any] = {}
    for arm in ("spatial_residual_only", "point_continue"):
        if arm in by_arm:
            comparisons[f"{arm}_minus_step0"] = {
                metric: float(by_arm[arm]["best_validation_metrics"][metric])
                - float(by_arm[arm]["step0_metrics"][metric])
                for metric in metric_names
            }
    if "spatial_joint" in by_arm and "point_continue" in by_arm:
        comparisons["spatial_joint_minus_point_continue"] = {
            metric: float(by_arm["spatial_joint"]["best_validation_metrics"][metric])
            - float(by_arm["point_continue"]["best_validation_metrics"][metric])
            for metric in metric_names
        }
    selection = {
        "selection_split": "internal_val", "external_used": False,
        "primary_metric": "patient_macro_pathway_pcc",
        "within_trajectory_order": ["max_patient_macro_pathway_pcc", "min_zMSE", "min_update"],
        "architecture_order": ["max_patient_macro_pathway_pcc", "min_zMSE", "complexity"],
        "seed": 42, "single_seed_only": True,
        "trajectories": summaries, "selected": asdict(selected),
        "paired_comparisons": comparisons,
    }
    _write_json(run_dir / "selection.json", selection)
    _write_json(run_dir / "metrics.json", {
        "status": "training_outputs_available", "selection": asdict(selected),
        "external_evaluation": "not_run", "seed_sd": "not_applicable_single_seed",
        "trajectories": [
            {"arm": row["arm"], "best": row["best_validation_metrics"],
             "step0": row["step0_metrics"]} for row in summaries
        ],
        "paired_comparisons": comparisons,
    })
    registry_path = run_dir / "model_weights.json"
    registry = _read_json(registry_path) if registry_path.is_file() else {
        "schema_version": "1.0", "experiment_id": config["experiment_id"],
        "batch_id": config["batch_id"], "files": [],
    }
    source = Path(config["inputs"]["stage1_checkpoint"])
    registry["source_stage1_checkpoint"] = {
        "path": str(source), "status": "present" if source.is_file() else "missing",
    }
    registry["trajectories"] = []
    for summary in summaries:
        row: dict[str, Any] = {"arm": summary["arm"], "seed": summary["seed"]}
        for key in ("warmup", "warmup_bundle", "formal", "last", "model_bundle"):
            row[key] = {
                "path": summary[key],
                "status": "present" if Path(summary[key]).is_file() else "missing",
            }
        registry["trajectories"].append(row)
    registry["return_policy"] = "server_weights_excluded_from_local_result_copy"
    if parent_checkpoint is not None:
        registry["parent_checkpoint"] = str(parent_checkpoint.resolve())
        registry["parent_run"] = str(parent_checkpoint.resolve().parents[2])
    _write_json(registry_path, registry)


def _run_training(args: argparse.Namespace, *, resume: bool) -> int:
    config = _read_json(args.config.resolve())
    run_dir, weights_dir = args.run_dir.resolve(), args.weights_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(config, PACKAGE_DIR, include_external=False)
    arms = [args.arm] if resume or args.arm != "all" else list(config["parameters"]["execution_order"])
    if set(arms) - set(ARM_ORDER):
        raise ValueError(f"invalid configured execution order: {arms}")
    checkpoint = args.checkpoint.resolve() if resume else None
    if checkpoint is not None:
        _update_parent_record(run_dir, checkpoint)
    summaries = [
        train_arm(
            config, bundle, arm, run_dir, weights_dir,
            resume_checkpoint=checkpoint, device=args.device,
        )
        for arm in arms
    ]
    _write_run_outputs(config, run_dir, summaries, parent_checkpoint=checkpoint)
    print(json.dumps({"status": "ok", "trajectories": summaries}, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--plan":
        argv[0] = "plan"
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return 2
    if args.command == "plan":
        config = _read_json(args.config.resolve())
        print(json.dumps({
            "status": config["implementation_status"], "experiment_id": config["experiment_id"],
            "arms": [row["id"] for row in config["parameters"]["arms"]],
            "training_executed": False,
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "check":
        config = _read_json(args.config.resolve())
        print(json.dumps(preflight(config, PACKAGE_DIR), ensure_ascii=False, indent=2))
        return 0
    if args.command == "train":
        return _run_training(args, resume=False)
    if args.command == "resume":
        return _run_training(args, resume=True)
    if args.command == "prepare-xzy-input":
        print(json.dumps(
            _prepare_xzy_input(args.input, args.output, native_step=args.native_step),
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    model, metadata = load_model_bundle(args.bundle, device=args.device)
    data = _inference_input(args.input, require_targets=args.command == "external-eval")
    graph = _inference_graph(data, metadata["graph_parameters"]) if metadata["spatial_enabled"] else None
    if args.command == "predict":
        prediction = predict_features(
            model, data["features"], spatial_enabled=metadata["spatial_enabled"],
            graph=graph, device=args.device,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output, prediction=prediction, patient=_decode(data["patient"]),
            patch_stem=_decode(data["patch_stem"]),
            pathway_names=np.asarray(metadata["pathway_names"], dtype=str),
        )
        return 0
    report = evaluate_external(
        model, data["features"], targets=data["target"], patient_ids=_decode(data["patient"]),
        spatial_enabled=metadata["spatial_enabled"], graph=graph, device=args.device,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "predictions.npz", prediction=report.pop("prediction"),
        target=report.pop("target"), patient=report.pop("patient_ids"),
        patch_stem=_decode(data["patch_stem"]),
        pathway_names=np.asarray(metadata["pathway_names"], dtype=str),
    )
    reconstruction = {
        "graph_features": data["graph_features"], "patient": _decode(data["patient"]),
        "patch_stem": _decode(data["patch_stem"]), "x": data["x"], "y": data["y"],
        "native_step": data["native_step"],
        "partition": _decode(data.get("partition", np.asarray(["external_test"] * len(data["features"])))),
    }
    if graph is not None:
        reconstruction.update({
            "neighbor_index": graph.neighbor_index, "neighbor_weight": graph.neighbor_weight,
            "neighbor_mask": graph.neighbor_mask, "self_weight": graph.self_weight,
            "degree": graph.degree,
            "neighbor_distance_native_steps": graph.neighbor_distance_native_steps,
            "neighbor_cosine_similarity": graph.neighbor_cosine_similarity,
        })
    np.savez_compressed(args.output / "graph_reconstruction.npz", **reconstruction)
    if graph is not None and metadata.get("provenance", {}).get("kind") == "warmup_step0":
        with np.load(args.output / "predictions.npz", allow_pickle=False) as saved:
            base_prediction = np.asarray(saved["prediction"], dtype=np.float64)
            target = np.asarray(saved["target"], dtype=np.float64)
            patients = np.asarray(saved["patient"], dtype=str)
        smoothed = fixed_beta_one_smoothing(base_prediction, graph)
        smoothing_report = {
            "beta": 1.0, "self_weight_included": True, "used_for_selection": False,
            "baseline": compute_regression_metrics(base_prediction, target, patients),
            "smoothed": compute_regression_metrics(smoothed, target, patients),
        }
        _write_json(args.output / "fixed_smoothing_metrics.json", smoothing_report)
        np.savez_compressed(
            args.output / "fixed_smoothing_predictions.npz", prediction=smoothed,
            target=target, patient=patients, patch_stem=_decode(data["patch_stem"]),
            pathway_names=np.asarray(metadata["pathway_names"], dtype=str),
        )
    _write_json(args.output / "metrics.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
