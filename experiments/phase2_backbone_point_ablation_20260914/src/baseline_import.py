"""Validate and import the accepted UNI2-h point result as a historical reference."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from data import IdentityRecord
from errors import ReferenceMismatchError


EXPECTED_CONTRACT = {
    "split_id": "MPP2/group_2",
    "label_version": "barcode-repair-v003",
    "input_dim": 1536,
    "hidden_dim": 256,
    "output_dim": 30,
    "dropout": 0.3,
    "optimizer": "AdamW",
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "betas": [0.9, 0.999],
    "optimizer_epsilon": 1e-8,
    "batch_size": 256,
    "keep_last_batch": True,
    "max_epochs": 60,
    "scheduler": "none",
    "precision": "float32",
    "amp": False,
    "tf32": False,
    "num_workers": 0,
    "cpu_threads": 8,
    "selection_metric": "patient_macro_pathway_pcc",
    "tie_metric": "patient_macro_z_mse_selection",
    "formal_start_epoch": 6,
    "checkpoint_tolerance": 1e-6,
    "early_stop_count_start_epoch": 16,
    "early_stop_min_delta": 1e-4,
    "early_stop_patience": 10,
    "constant_prediction_selection_penalty": -1.0,
    "model_seed_offset": 0,
    "center_seed_offset": 100000,
    "dropout_seed_offset": 200000,
}


def load_reference_entry(manifest_path: str | Path, seed: int) -> dict:
    try:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceMismatchError(f"无法读取历史参照清单: {exc}") from exc
    expected_top = {
        "schema_version": "1.0",
        "status": "historical_matched_reference_reused",
        "model": "uni2h",
        "arm": "point",
        "source_experiment": "phase2_softlink_local_v2",
        "source_batch": "20260908_005325_810_8d306c10",
        "source_date": "2026-09-08",
    }
    for key, expected in expected_top.items():
        if payload.get(key) != expected:
            raise ReferenceMismatchError(f"历史参照必须是匹配的 UNI2-h point；{key}={payload.get(key)!r}")
    if payload.get("contract") != EXPECTED_CONTRACT:
        raise ReferenceMismatchError("历史参照科学合同与本轮 point 合同不完全一致")
    seed_key = str(int(seed))
    if seed_key not in {"42", "43", "44"} or seed_key not in (payload.get("seeds") or {}):
        raise ReferenceMismatchError(f"历史参照没有登记 seed={seed}")
    entry = dict(payload["seeds"][seed_key])
    entry.update(
        seed=int(seed),
        status=payload["status"],
        model="uni2h",
        arm="point",
        source_experiment=payload["source_experiment"],
        source_batch=payload.get("source_batch"),
        source_date=payload.get("source_date"),
        contract=payload["contract"],
        limitations=payload.get("limitations") or {},
    )
    return entry


def _npz_scalar(archive: np.lib.npyio.NpzFile, key: str):
    if key not in archive.files:
        raise ReferenceMismatchError(f"历史预测缺字段 {key}")
    return np.asarray(archive[key]).item()


def validate_reference_files(
    entry: dict,
    historical_root: str | Path,
    *,
    train_count: int = 9472,
    internal_count: int = 1078,
    external_count: int = 1039,
    output_dim: int = 30,
) -> dict:
    root = Path(historical_root).resolve()
    names = (
        "internal_predictions",
        "external_predictions",
        "initial_weights",
        "center_orders",
        "formal_checkpoint_registered",
    )
    paths = {name: (root / str(entry[name])).resolve() for name in names}
    for name, path in paths.items():
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ReferenceMismatchError(f"历史参照路径越出根目录: {name}={path}") from exc
        if not path.is_file():
            raise ReferenceMismatchError(f"历史参照文件缺失: {name}={path}")

    try:
        with np.load(paths["center_orders"], allow_pickle=False) as archive:
            if int(np.asarray(archive["batch_size"]).item()) != 256:
                raise ReferenceMismatchError("历史点位顺序 batch_size 不是 256")
            if bool(np.asarray(archive["keep_last_batch"]).item()) is not True:
                raise ReferenceMismatchError("历史点位顺序没有保留末批")
            epoch_keys = sorted(
                (name for name in archive.files if name.startswith("epoch_")),
                key=lambda value: int(value.split("_", 1)[1]),
            )
            expected_keys = [f"epoch_{epoch}" for epoch in range(1, len(epoch_keys) + 1)]
            if epoch_keys != expected_keys or len(epoch_keys) < int(entry["formal_epoch"]):
                raise ReferenceMismatchError("历史点位顺序 epoch 不连续或未覆盖正式轮次")
            rng = np.random.Generator(np.random.PCG64(int(entry["seed"]) + 100_000))
            for key in epoch_keys:
                order = np.asarray(archive[key])
                expected_order = rng.permutation(int(train_count)).astype(np.int64, copy=False)
                if (
                    order.shape != (int(train_count),)
                    or not np.issubdtype(order.dtype, np.integer)
                    or not np.array_equal(order, expected_order)
                ):
                    raise ReferenceMismatchError(
                        f"历史点位顺序 {key} 不匹配 PCG64(seed+100000) 或训练身份数"
                    )
    except (KeyError, OSError, ValueError) as exc:
        if isinstance(exc, ReferenceMismatchError):
            raise
        raise ReferenceMismatchError(f"无法验证历史点位顺序: {exc}") from exc

    try:
        checkpoint = torch.load(
            paths["formal_checkpoint_registered"], map_location="cpu", weights_only=True
        )
        if not isinstance(checkpoint, dict):
            raise ReferenceMismatchError("历史 formal checkpoint 不是字典")
        endpoint = {
            "arm": "point",
            "seed": int(entry["seed"]),
            "epoch": int(entry["formal_epoch"]),
            "kind": "formal",
        }
        for key, expected in endpoint.items():
            if checkpoint.get(key) != expected:
                raise ReferenceMismatchError(
                    f"历史 formal checkpoint 的 {key} 不匹配正式端点"
                )
        state = checkpoint.get("model_state_dict")
        expected_state = {
            "shared.weight": (256, 1536),
            "shared.bias": (256,),
            "point_head.weight": (30, 256),
            "point_head.bias": (30,),
        }
        if not isinstance(state, dict) or set(state) != set(expected_state):
            raise ReferenceMismatchError("历史 formal checkpoint 不是纯 point 模型状态")
        for key, shape in expected_state.items():
            tensor = torch.as_tensor(state[key])
            if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
                raise ReferenceMismatchError(
                    f"历史 formal checkpoint 的 {key} 形状或数值非法"
                )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        if isinstance(exc, ReferenceMismatchError):
            raise
        raise ReferenceMismatchError(f"无法验证历史 formal checkpoint: {exc}") from exc

    try:
        # The accepted 2026-09-08 legacy archive stores identity strings as
        # NumPy object arrays. Pickle is enabled only for this explicitly
        # registered, local historical source; new outputs use plain Unicode.
        with np.load(paths["internal_predictions"], allow_pickle=True) as archive:
            if archive["pred_z"].shape != (int(internal_count), int(output_dim)):
                raise ReferenceMismatchError("历史内部预测形状不匹配")
            if archive["target_z"].shape != (int(internal_count), int(output_dim)):
                raise ReferenceMismatchError("历史内部真值形状不匹配")
            if _npz_scalar(archive, "arm") != "point" or int(_npz_scalar(archive, "seed")) != int(entry["seed"]):
                raise ReferenceMismatchError("历史内部预测不是同 seed point 正式结果")
            if int(_npz_scalar(archive, "epoch")) != int(entry["formal_epoch"]):
                raise ReferenceMismatchError("历史内部预测 epoch 与登记的 formal epoch 不一致")
            patient_field = "patient_id" if "patient_id" in archive.files else "patients"
            pathway_field = "pathway_names" if "pathway_names" in archive.files else "pathways"
            if len(archive[patient_field]) != int(internal_count) or len(archive[pathway_field]) != int(output_dim):
                raise ReferenceMismatchError("历史内部预测身份或通路字段长度不匹配")
            if not np.isfinite(archive["pred_z"]).all() or not np.isfinite(archive["target_z"]).all():
                raise ReferenceMismatchError("历史内部预测含 NaN/Inf")
        with np.load(paths["external_predictions"], allow_pickle=True) as archive:
            if "target_z" in archive.files:
                raise ReferenceMismatchError("服务器外部预测禁止包含 target_z")
            if archive["pred_z"].shape != (int(external_count), int(output_dim)):
                raise ReferenceMismatchError("历史外部预测形状不匹配")
            if "pred_raw" not in archive.files or archive["pred_raw"].shape != (
                int(external_count),
                int(output_dim),
            ):
                raise ReferenceMismatchError("历史外部原标度预测缺失或形状不匹配")
            if "checkpoint_metadata" in archive.files:
                endpoint_meta = dict(np.asarray(archive["checkpoint_metadata"]).item())
                endpoint_arm = endpoint_meta.get("arm")
                endpoint_seed = endpoint_meta.get("seed")
                endpoint_kind = endpoint_meta.get("checkpoint_kind")
            else:
                endpoint_arm = _npz_scalar(archive, "arm")
                endpoint_seed = _npz_scalar(archive, "seed")
                endpoint_kind = "formal"
            if endpoint_arm != "point" or int(endpoint_seed) != int(entry["seed"]):
                raise ReferenceMismatchError("历史外部预测不是同 seed point 正式端点")
            if endpoint_kind != "formal":
                raise ReferenceMismatchError("历史外部预测不是 formal 正式端点")
            if "checkpoint_metadata" in archive.files and int(endpoint_meta.get("epoch", -1)) != int(entry["formal_epoch"]):
                raise ReferenceMismatchError("历史外部预测 epoch 与 formal 正式端点不一致")
            if not np.isfinite(archive["pred_z"]).all() or not np.isfinite(archive["pred_raw"]).all():
                raise ReferenceMismatchError("历史外部预测含 NaN/Inf")
    except (KeyError, OSError, ValueError) as exc:
        if isinstance(exc, ReferenceMismatchError):
            raise
        raise ReferenceMismatchError(f"无法验证历史预测附件: {exc}") from exc

    return {
        **entry,
        "status": "historical_matched_reference_reused",
        "resolved_paths": {key: str(value) for key, value in paths.items()},
        "external_target_z_present": False,
    }


def extract_point_output_state(path: str | Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ReferenceMismatchError(f"无法安全读取历史 initial_weights.pt: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReferenceMismatchError("历史初值不是 state_dict")
    required = {"point_head.weight", "point_head.bias"}
    if not required.issubset(payload):
        raise ReferenceMismatchError("历史初值缺 point_head.weight/bias")
    weight = torch.as_tensor(payload["point_head.weight"]).detach().cpu().float()
    bias = torch.as_tensor(payload["point_head.bias"]).detach().cpu().float()
    if tuple(weight.shape) != (30, 256) or tuple(bias.shape) != (30,):
        raise ReferenceMismatchError(
            f"历史 point_head 初值必须为 weight=(30, 256)、bias=(30,)，实际={tuple(weight.shape)}/{tuple(bias.shape)}"
        )
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ReferenceMismatchError("历史 point_head 初值含 NaN/Inf")
    return {"weight": weight, "bias": bias}


def _legacy_field(archive: np.lib.npyio.NpzFile, *names: str) -> np.ndarray:
    for name in names:
        if name in archive.files:
            return np.asarray(archive[name]).astype(str)
    raise ReferenceMismatchError(f"历史预测身份缺字段，候选={list(names)}")


def validate_reference_identity_alignment(
    validated: dict,
    common_rows: list[IdentityRecord],
    pathway_names: list[str] | tuple[str, ...],
) -> None:
    """Require historical predictions to match the common rows in exact order."""

    expected_by_split = {
        "internal": [row.identity_key for row in common_rows if row.split == "internal_val"],
        "external": [row.identity_key for row in common_rows if row.split == "external_test"],
    }
    paths = validated["resolved_paths"]
    for kind, path_key in (
        ("internal", "internal_predictions"),
        ("external", "external_predictions"),
    ):
        with np.load(paths[path_key], allow_pickle=True) as archive:
            patients = _legacy_field(archive, "patient_id", "patients")
            sources = _legacy_field(archive, "slide_id", "source_groups")
            spots = _legacy_field(archive, "spot_id", "spots")
            pathways = _legacy_field(archive, "pathway_names", "pathways")
        actual = [f"{patient}|{source}|{spot}" for patient, source, spot in zip(patients, sources, spots)]
        if actual != expected_by_split[kind]:
            raise ReferenceMismatchError(
                f"历史 {kind} 预测身份或行顺序与共同身份清单不一致"
            )
        if pathways.tolist() != list(pathway_names):
            raise ReferenceMismatchError(f"历史 {kind} 预测通路顺序与本轮清单不一致")


def import_reference_artifacts(validated: dict, destination: str | Path) -> dict:
    """Copy small raw evidence only; keep the historical checkpoint at its source path."""

    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(f"参照导入目录已存在，禁止覆盖: {target}")
    target.mkdir(parents=True)
    resolved = {key: Path(value) for key, value in validated["resolved_paths"].items()}
    copied: dict[str, str] = {}
    for key in ("internal_predictions", "external_predictions", "initial_weights", "center_orders"):
        output = target / resolved[key].name
        shutil.copy2(resolved[key], output)
        copied[key] = str(output)
    record = {
        key: value
        for key, value in validated.items()
        if key != "resolved_paths"
    }
    record.update(
        copied_files=copied,
        historical_formal_checkpoint=str(resolved["formal_checkpoint_registered"]),
        resource_usage={"elapsed_seconds": None, "peak_gpu_memory_bytes": None, "status": "not_recorded"},
    )
    (target / "reference.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record
