"""Independent package entry. Reads --config and --run-dir only."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.manifest import build_manifest_datasets
from models.registry import load_checkpoint, load_predictor
from predict import predict_and_export
from run_presets import resolve_run_kind
from sampling.registry import load_sampler
from targets.protocol import MissingTargetContract
from targets.registry import load_target
from train import train_and_export


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _require_dir(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} missing: {path}")
    return path


def _resolve_labels_root(configured: str) -> Path:
    path = Path(configured)
    if (path / "train").exists():
        return path
    nested = path / "group_2" / "labels"
    if nested.exists():
        return nested
    raise FileNotFoundError(
        "Repaired labels are missing. Expected train/ under "
        f"{path} or {nested}. This package will not fall back to unvalidated repo labels."
    )


def _resolve_training_label_root(target_id: str, inputs: dict) -> Path:
    """Task 2 must read original genes, not the 30 pathway z-score tables."""
    if target_id != "gene_expression":
        return _resolve_labels_root(str(inputs["labels_root"]))
    gene_root = str(inputs.get("gene_labels_root") or "").strip()
    if not gene_root:
        raise FileNotFoundError(
            "target_id=gene_expression requires inputs.gene_labels_root. "
            "This is the training input table: original expression of the 30 pathway member genes. "
            "Do not keep reading labels_root ssGSEA z-score files."
        )
    path = Path(gene_root)
    if not path.exists():
        raise FileNotFoundError(f"gene_labels_root missing: {path}")
    return path


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _preflight(split_info: dict, zscore_manifest: dict, train_patients: list[str], external_patient: str) -> None:
    if int(split_info.get("leakage_pairs", 0)) != 0:
        raise RuntimeError(f"split_info leakage_pairs={split_info['leakage_pairs']}; training is blocked")
    fit = set(zscore_manifest.get("fit_patients", []))
    if fit and fit != set(train_patients):
        raise RuntimeError(f"z-score fit_patients {sorted(fit)} != train_patients {train_patients}")
    if external_patient in fit or external_patient in train_patients:
        raise RuntimeError(f"external patient {external_patient} must stay out of training and z-score fit")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--fail", action="store_true")
    args = parser.parse_args()
    config = _read_json(args.config)
    run_dir = args.run_dir.resolve()
    print(f"Package: {PACKAGE_DIR}", flush=True)
    if args.fail:
        raise RuntimeError("主动报错：用于验证日志回传。")

    mode_file = PACKAGE_DIR / "run_mode.json"
    if not mode_file.exists():
        raise FileNotFoundError("run_mode.json is missing; this is the only file used to switch smoke/full/predict")
    run_kind = _read_json(mode_file).get("run_kind") or config.get("run_kind")
    preset = resolve_run_kind(str(run_kind))
    parameters = dict(config.get("parameters") or {})
    inputs = dict(config.get("inputs") or {})
    print(f"run_kind={preset['run_kind']} mode={preset['mode']} epochs={preset['num_epochs']}", flush=True)

    splits_dir = Path(inputs["splits_dir"]) if inputs.get("splits_dir") else PACKAGE_DIR / "assets" / "group_2"
    zscore_manifest_path = splits_dir / "zscore_manifest.json"
    split_info_path = splits_dir / "split_info.json"
    split_manifest_path = splits_dir / "split_manifest.csv"
    for path, label in (
        (zscore_manifest_path, "zscore_manifest"),
        (split_info_path, "split_info"),
        (split_manifest_path, "split_manifest"),
    ):
        _require_dir(path, label)

    target_id = str(parameters.get("target_id", "pathway_ssgsea"))
    target = load_target(target_id, zscore_manifest_path)
    spec = target.spec()
    sampler = load_sampler(str(parameters.get("sampling_id", "dense")), parameters)
    train_patients = [str(item) for item in parameters.get("train_patients", [])]
    external_patient = str(parameters.get("external_patient", "XZY"))
    _preflight(_read_json(split_info_path), _read_json(zscore_manifest_path), train_patients, external_patient)

    labels_root = _resolve_training_label_root(target_id, inputs)
    print(f"training_label_root={labels_root} target_id={target_id}", flush=True)
    cache_root = str(inputs["cache_root"])
    flat_cache_root = str(inputs["flat_cache_root"])
    allow_missing = bool(parameters.get("allow_missing", False))
    if not allow_missing:
        _require_dir(Path(cache_root), "cache_root")
        _require_dir(Path(flat_cache_root), "flat_cache_root")

    manifest = pd.read_csv(split_manifest_path)
    train_manifest = sampler.filter_manifest(manifest, "train")
    print(
        f"sampling={sampler.sampling_id}: train rows {int((manifest['split']=='train').sum())} -> "
        f"{int((train_manifest['split']=='train').sum())}; val/external stay dense",
        flush=True,
    )

    threads = int(parameters.get("num_threads", 8))
    torch.set_num_threads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    seed = int(parameters.get("seed", 42))
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} target={spec.target_id} n_outputs={spec.n_outputs} model={parameters.get('model_id')}", flush=True)

    train_ds, val_ds, ext_ds = build_manifest_datasets(
        manifest_df=train_manifest,
        cache_root=cache_root,
        flat_cache_root=flat_cache_root,
        labels_root=str(labels_root),
        target=target,
        train_mpp_id=int(parameters.get("train_mpp_id", 2)),
        external_mpp_id=int(parameters.get("external_mpp_id", 2)),
        external_patient=external_patient,
        train_patients=train_patients,
        allow_missing=allow_missing,
    )
    batch_size = int(parameters.get("batch_size", 32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    ext_loader = DataLoader(ext_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    in_dim = int(getattr(ext_ds, "feat_dim", 1536))
    predictor = load_predictor(
        model_id=str(parameters.get("model_id", "uni2h_mlp")),
        in_dim=in_dim,
        hidden=int(parameters.get("hidden_dim", 1024)),
        out_dim=spec.n_outputs,
        dropout=float(parameters.get("dropout", 0.3)),
        device=device,
    )
    shutil.copy2(splits_dir / "zscore_params_from_train.json", run_dir / "raw" / "zscore_params_from_train.json")
    shutil.copy2(mode_file, run_dir / "raw" / "run_mode.json")
    (run_dir / "raw" / "resolved.json").write_text(
        json.dumps(
            {
                "run_kind": preset["run_kind"],
                "preset": preset,
                "target": spec.to_dict(),
                "sampling": sampler.spec().to_dict(),
                "labels_root": str(labels_root),
                "data_manifest_id": inputs.get("data_manifest_id"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    package = _read_json(PACKAGE_DIR / "package.json")
    code_version = str(package.get("code_version", "unknown"))
    if preset["mode"] == "predict":
        checkpoint = Path(inputs.get("checkpoint") or inputs["accepted_checkpoint"])
        payload = load_checkpoint(checkpoint, map_location=device)
        predictor.model.load_state_dict(payload["model_state_dict"])
        predict_and_export(
            predictor=predictor,
            val_loader=val_loader,
            external_loader=ext_loader,
            val_ds=val_ds,
            external_ds=ext_ds,
            spec=spec,
            sampling_id=sampler.sampling_id,
            code_version=code_version,
            checkpoint_source=str(checkpoint),
            run_dir=run_dir,
            run_kind=preset["run_kind"],
        )
        return 0

    train_and_export(
        predictor=predictor,
        train_loader=train_loader,
        val_loader=val_loader,
        external_loader=ext_loader,
        train_ds=train_ds,
        val_ds=val_ds,
        external_ds=ext_ds,
        spec=spec,
        sampling_id=sampler.sampling_id,
        code_version=code_version,
        run_kind=preset["run_kind"],
        num_epochs=int(preset["num_epochs"]),
        patience=int(preset["patience"]),
        min_delta=float(preset["min_delta"]),
        lr=float(parameters.get("lr", 1e-4)),
        seed=seed,
        run_dir=run_dir,
        extra_meta={"hidden_dim": int(parameters.get("hidden_dim", 1024)), "dropout": float(parameters.get("dropout", 0.3))},
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MissingTargetContract as exc:
        print(str(exc), file=sys.stderr, flush=True)
        raise SystemExit(2)
