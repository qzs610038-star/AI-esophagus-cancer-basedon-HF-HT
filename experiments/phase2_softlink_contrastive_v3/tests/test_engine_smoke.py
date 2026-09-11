"""Offline orchestration smoke test; all costly work is dependency-injected."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.engine import TorchRuntime, execute_experiment
import src.engine as engine_module


def test_warmup_reuses_identity_checked_frozen_cls_cache_after_failure(tmp_path, monkeypatch):
    class TinyBackbone(nn.Module):
        def forward_features(self, images):
            return torch.ones(images.shape[0], 1, 1536)

    class TinyRuntime(TorchRuntime):
        backbone_loads = 0

        def _device(self, context):
            return torch.device("cpu")

        def _fresh_backbone(self, context):
            self.backbone_loads += 1
            return TinyBackbone()

        def _batch(self, context, records, indices, **kwargs):
            chosen = [records[int(index)] for index in indices]
            return (
                torch.zeros(len(chosen), 3, 2, 2),
                torch.as_tensor(np.stack([record.raw_target for record in chosen])),
                [record.patient for record in chosen],
            )

    def stop_after_cache(*args, **kwargs):
        raise RuntimeError("synthetic failure after frozen CLS cache")

    monkeypatch.setattr(engine_module.training, "run_common_warmup", stop_after_cache)
    records = [
        SimpleNamespace(identity=(2, "P", f"patch_{index}"), patient="P", mpp_id=2,
                        patch_stem=f"patch_{index}", raw_target=np.zeros(30, dtype=np.float32))
        for index in range(2)
    ]
    context = {
        "config": {"training": {"precision": {"stage1": "float32"}}},
        "split": SimpleNamespace(protocol="original", fold="original"),
        "partitions": {"train": records, "val": records},
        "artifacts": {"zscore": {"mean": [0.0] * 30, "std": [1.0] * 30}},
        "resource_short_test": {"batch_size": 2},
    }
    runtime, destination = TinyRuntime(), tmp_path / "warmup"

    with pytest.raises(RuntimeError, match="synthetic failure"):
        runtime.warmup(context, 42, destination)
    assert runtime.backbone_loads == 1
    assert (destination / "frozen_cls_train.pt").is_file()
    assert (destination / "frozen_cls_internal_val.pt").is_file()

    with pytest.raises(RuntimeError, match="synthetic failure"):
        runtime.warmup(context, 42, destination)
    assert runtime.backbone_loads == 1


class FakeRuntime:
    def __init__(self):
        self.calls = []
        self.pilot_dirs = []

    def materialize(self, split, config):
        self.calls.append(("materialize", split.fold))
        return {"train": ["train"], "val": ["val"], "held": ["held"]}

    def fit_training_artifacts(self, partitions, split):
        self.calls.append(("fit", tuple(partitions["train"])))
        return {"z_ddof": 1, "patient_means": {"p": [0.0]}}

    def resource_short_test(self, context, candidates):
        self.calls.append(("short_test", tuple(candidates)))
        return {"batch_size": 64, "warmup_updates": 10, "timed_updates": 30}

    def warmup(self, context, seed, checkpoint_dir):
        self.calls.append(("warmup", seed))
        return {"checkpoint": str(checkpoint_dir / "warmup.pt")}

    def pilot(self, context, seed, lr, warmup, checkpoint_dir):
        self.calls.append(("pilot", lr))
        self.pilot_dirs.append(checkpoint_dir)
        return {"epoch": 5, "lr": lr, "patient_macro_pathway_pcc": lr, "zMSE": 1 / lr}

    def stage1(self, context, seed, cell, lr, warmup, checkpoint_dir, reused_pilot=None):
        self.calls.append(("stage1", cell, reused_pilot is not None))
        return {"history": [{"epoch": i, "patient_macro_pathway_pcc": 0.1 * i, "zMSE": 1 / i} for i in range(1, 6)], "checkpoint": str(checkpoint_dir / "formal.pt")}

    def cache_cls(self, context, seed, cell, endpoint, stage1_result, cache_dir):
        self.calls.append(("cache", cell, endpoint))
        return {"cache": str(cache_dir / "cls.npy")}

    def stage2(self, context, seed, task, stage1_result, cache, checkpoint_dir):
        self.calls.append(("stage2", task.cell, task.head, task.h_mode, task.endpoint))
        return {"epoch": task.endpoint, "patient_macro_pathway_pcc": 0.2, "zMSE": 0.5}

    def evaluate(self, context, seed, cell, stage1_result, destination):
        self.calls.append(("evaluate", cell, tuple(context["partitions"]["held"])))
        return {"selection_used": False, "pcc": 0.0}


def test_execute_backend_runs_chain_and_keeps_selection_and_outputs_separate(tmp_path):
    manifest = pd.DataFrame([
        {"patient": "p", "mpp_id": 1, "patch_stem": "a.png", "x": 0, "y": 0, "split": "train", "block_id": "t"},
        {"patient": "p", "mpp_id": 1, "patch_stem": "b.png", "x": 1, "y": 0, "split": "internal_val", "block_id": "v"},
    ])
    runtime = FakeRuntime()
    result = execute_experiment(
        config={"manifest": manifest, "training": {"batch_size_candidates": [64, 128], "learning_rates": {"lora_candidates": [1e-5, 3e-5, 1e-4]}}},
        run_dir=tmp_path / "run", weights_dir=tmp_path / "weights", protocol="original", seeds=(42,), folds=None,
        resume=False, dependencies=runtime,
    )
    assert result["exit_code"] == 0
    assert (tmp_path / "run" / "raw" / "original" / "original" / "seed_42" / "fixed_e5" / "stage2_results.json").is_file()
    assert (tmp_path / "run" / "raw" / "original" / "original" / "seed_42" / "best_epoch_sensitivity" / "stage2_results.json").is_file()
    assert (tmp_path / "run" / "model_weights.json").is_file()
    assert [call[0] for call in runtime.calls].count("pilot") == 3
    assert [call[0] for call in runtime.calls].count("stage1") == 6
    assert [call[0] for call in runtime.calls].count("stage2") == 22
    assert all(call[0] != "fit" or call[1] == ("train",) for call in runtime.calls)
    assert all(call[0] != "evaluate" or call[2] == ("held",) for call in runtime.calls)
    resumed = execute_experiment(
        config={"manifest": manifest, "training": {"batch_size_candidates": [64, 128], "learning_rates": {"lora_candidates": [1e-5, 3e-5, 1e-4]}}},
        run_dir=tmp_path / "run", weights_dir=tmp_path / "weights", protocol="original", seeds=(42,), folds=None,
        resume=True, dependencies=runtime,
    )
    assert resumed["exit_code"] == 0
    assert [call[0] for call in runtime.calls].count("stage2") == 22
    assert [call[0] for call in runtime.calls].count("materialize") == 2


def test_execute_returns_nonzero_when_runtime_fails(tmp_path):
    class FailingRuntime(FakeRuntime):
        def materialize(self, split, config):
            raise RuntimeError("missing PNG")

    manifest = pd.DataFrame([
        {"patient": "p", "mpp_id": 1, "patch_stem": "a.png", "x": 0, "y": 0, "split": "train", "block_id": "t"},
        {"patient": "p", "mpp_id": 1, "patch_stem": "b.png", "x": 1, "y": 0, "split": "internal_val", "block_id": "v"},
    ])
    result = execute_experiment({"manifest": manifest}, tmp_path / "run", tmp_path / "weights", "original", (42,), None, False, dependencies=FailingRuntime())
    assert result["exit_code"] == 1 and result["status"] == "failed"


def test_orchestrator_keyword_config_path_and_pilot_weight_paths_are_compatible(tmp_path):
    config_path = tmp_path / "config.json"
    manifest_path = tmp_path / "split.csv"
    pd.DataFrame([
        {"patient": "p", "mpp_id": 1, "patch_stem": "a.png", "x": 0, "y": 0, "split": "train", "block_id": "t"},
        {"patient": "p", "mpp_id": 1, "patch_stem": "b.png", "x": 1, "y": 0, "split": "internal_val", "block_id": "v"},
    ]).to_csv(manifest_path, index=False)
    config_path.write_text('{"data":{"split_manifest_file":"' + str(manifest_path).replace('\\', '\\\\') + '"},"training":{"batch_size_candidates":[64,128],"learning_rates":{"lora_candidates":[1e-5,3e-5,1e-4]}}}', encoding="utf-8")
    runtime = FakeRuntime()
    result = execute_experiment(config_path=config_path, run_dir=tmp_path / "run", weights_dir=tmp_path / "weights", protocol="original", seeds=(42,), dependencies=runtime)
    assert result["exit_code"] == 0
    assert len({str(path) for path in runtime.pilot_dirs}) == 3


def test_resource_short_test_completes_timed_updates_and_evaluation(monkeypatch):
    """Regression seam for the production resource-short-test public method."""
    class TinyProbe(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.teacher = nn.Linear(1, 1)

        def forward(self, images):
            return {"pred": images * self.scale}

    class TinyRuntime(TorchRuntime):
        def _base(self, context):
            return nn.Identity()

        def _device(self, context):
            return torch.device("cpu")

        def _batch(self, context, records, indices, **kwargs):
            size = len(indices)
            return torch.ones(size, 30), torch.zeros(size, 30), ["p"] * size

    monkeypatch.setattr(engine_module.models, "inject_independent_qv_lora", lambda backbone, **kwargs: backbone)
    monkeypatch.setattr(engine_module.models, "Stage1Model", TinyProbe)
    context = {
        "partitions": {"train": list(range(256))},
        "config": {"resource_short_test": {"minimum_memory_margin_bytes": 1}},
    }

    result = TinyRuntime().resource_short_test(context, [64, 128])

    assert {row["batch_size"] for row in result["measurements"]} == {64, 128}
    assert all(row["warmup_updates"] == 10 and row["timed_updates"] == 30 for row in result["measurements"])
    assert all(row["status"] == "passed" and len(row["prediction_sample"]) == 2 for row in result["measurements"])
    assert result["precision"] == "float32"
    assert result["selection_reason"] in {"batch128_selected", "batch64_higher_or_equal_image_throughput"}
    assert all("device_total_memory" in row and "free_memory_after_measurement" in row for row in result["measurements"])


def test_resource_short_test_recovers_from_batch128_oom_and_runs_batch64(monkeypatch):
    class OomAt128Probe(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.teacher = nn.Linear(1, 1)

        def forward(self, images):
            if images.shape[0] == 128:
                raise torch.cuda.OutOfMemoryError("synthetic batch128 OOM")
            return {"pred": images * self.scale}

    class TinyRuntime(TorchRuntime):
        def _base(self, context):
            return nn.Identity()

        def _device(self, context):
            return torch.device("cpu")

        def _batch(self, context, records, indices, **kwargs):
            size = len(indices)
            return torch.ones(size, 30), torch.zeros(size, 30), ["p"] * size

    monkeypatch.setattr(engine_module.models, "inject_independent_qv_lora", lambda backbone, **kwargs: backbone)
    monkeypatch.setattr(engine_module.models, "Stage1Model", OomAt128Probe)
    context = {
        "partitions": {"train": list(range(256))},
        "config": {"resource_short_test": {"minimum_memory_margin_bytes": 2 * 1024**3}},
    }

    result = TinyRuntime().resource_short_test(context, [64, 128])

    by_size = {row["batch_size"]: row for row in result["measurements"]}
    assert by_size[128]["status"] == "oom"
    assert by_size[64]["status"] == "passed"
    assert result["batch_size"] == 64
    assert result["selection_reason"] == "batch128_oom"


def test_resource_short_test_rejects_batch128_when_reserved_or_free_margin_is_tight(monkeypatch):
    class CpuOnlyModule(nn.Identity):
        def to(self, *args, **kwargs):
            return self

    class TinyProbe(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.teacher = nn.Linear(1, 1)

        def to(self, *args, **kwargs):
            return self

        def forward(self, images):
            return {"pred": images * self.scale}

    class SimulatedCudaRuntime(TorchRuntime):
        current_batch = 0

        def _base(self, context):
            return CpuOnlyModule()

        def _device(self, context):
            return torch.device("cuda")

        def _batch(self, context, records, indices, **kwargs):
            self.current_batch = len(indices)
            return torch.ones(len(indices), 30), torch.zeros(len(indices), 30), ["p"] * len(indices)

    runtime = SimulatedCudaRuntime()
    gib = 1024**3
    monkeypatch.setattr(engine_module.models, "inject_independent_qv_lora", lambda backbone, **kwargs: backbone)
    monkeypatch.setattr(engine_module.models, "Stage1Model", TinyProbe)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: 8 * gib)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *_: (15 if runtime.current_batch == 128 else 8) * gib)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: ((1 if runtime.current_batch == 128 else 8) * gib, 16 * gib))
    context = {
        "partitions": {"train": list(range(256))},
        "config": {
            "training": {"precision": {"stage1": "float32"}},
            "resource_short_test": {"minimum_memory_margin_bytes": 2 * gib},
        },
    }

    result = runtime.resource_short_test(context, [64, 128])

    assert result["batch_size"] == 64
    assert result["selection_reason"] == "batch128_insufficient_memory_margin"


def test_resource_short_test_keeps_completed_batch64_when_gpu_margin_is_tight(monkeypatch):
    """A completed fallback is usable, but its resource risk must be explicit."""
    class CpuOnlyModule(nn.Identity):
        def to(self, *args, **kwargs):
            return self

    class TinyProbe(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.teacher = nn.Linear(1, 1)

        def to(self, *args, **kwargs):
            return self

        def forward(self, images):
            return {"pred": images * self.scale}

    class SimulatedCudaRuntime(TorchRuntime):
        def _base(self, context):
            return CpuOnlyModule()

        def _device(self, context):
            return torch.device("cuda")

        def _batch(self, context, records, indices, **kwargs):
            return torch.ones(len(indices), 30), torch.zeros(len(indices), 30), ["p"] * len(indices)

    gib = 1024**3
    monkeypatch.setattr(engine_module.models, "inject_independent_qv_lora", lambda backbone, **kwargs: backbone)
    monkeypatch.setattr(engine_module.models, "Stage1Model", TinyProbe)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: 8 * gib)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *_: 15 * gib)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: (1 * gib, 16 * gib))
    context = {
        "partitions": {"train": list(range(256))},
        "config": {
            "training": {"precision": {"stage1": "float32"}},
            "resource_short_test": {"minimum_memory_margin_bytes": 2 * gib},
        },
    }

    result = SimulatedCudaRuntime().resource_short_test(context, [64, 128])

    by_size = {row["batch_size"]: row for row in result["measurements"]}
    assert by_size[64]["status"] == "passed"
    assert result["batch_size"] == 64
    assert result["fallback_used"] is True
    assert result["selection_reason"] == "batch64_insufficient_memory_margin"
    assert "batch64" in result["warning"]
    assert result["memory_risk"] is True
