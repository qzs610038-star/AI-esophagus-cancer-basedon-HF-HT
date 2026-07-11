#!/usr/bin/env python3
"""PFMval durable state, document, path and result management helpers.

The module intentionally uses only the Python standard library plus PyYAML,
which is already required by the project configuration.  All public helpers
accept an explicit project root so tests never mutate the live checkout.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - server preflight reports this cleanly
    yaml = None


SCHEMA_VERSION = "1.0"
MAX_RESULT_FILE_BYTES = 20 * 1024 * 1024
MAX_RESULT_TOTAL_BYTES = 50 * 1024 * 1024
SMOKE_MAX_EPOCHS = 3
MPP_TRAINING_PATH_IDS = {
    "mpp_data_root",
    "mpp_standard_splits",
    "server_mpp_flat_cache",
    "server_mpp_results",
}
MPP_TRAINING_PATH_PARAMETERS = {
    "mpp_root",
    "cache_root",
    "labels_root",
    "splits_root",
    "flat_cache_root",
    "output_root",
}
MPP_TRAINING_ALLOWED_PARAMETERS = {
    "train_mpp_id",
    "train_patients",
    "external_mpp_id",
    "external_patient",
    "val_strategy",
    "val_patient",
    "num_epochs",
    "batch_size",
    "lr",
    "seed",
    "num_threads",
    "dataset_name",
    "dropout",
    "hidden_dim",
    "allow_missing",
    "patience",
    "min_delta",
}
ALLOWED_RESULT_FILES = {
    "training_history.csv",
    "training_summary.txt",
    "per_pathway_pcc.csv",
    "best_epoch.txt",
    "metrics.json",
    "stderr_tail.txt",
    "stderr_tail.txt.gz",
    "stdout_tail.txt",
    "stdout_tail.txt.gz",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_rel(path: str | Path) -> str:
    value = str(path).replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_against_schema(instance: Any, schema_path: Path, label: str) -> bool:
    """Validate when jsonschema is installed; callers retain manual fallback checks."""
    try:
        import jsonschema
    except ImportError:
        return False
    try:
        jsonschema.validate(instance=instance, schema=read_json(schema_path))
    except jsonschema.ValidationError as exc:
        location = ".".join(str(item) for item in exc.absolute_path) or "<root>"
        raise ValueError(f"{label} schema violation at {location}: {exc.message}") from exc
    return True


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = text.replace("\r\n", "\n")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def write_json_atomic(path: Path, value: Any) -> None:
    write_text_atomic(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def git_head(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_commit_exists(root: Path, commit: str) -> bool:
    if not commit or commit == "unknown":
        return False
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def git_tracked_changes(root: Path) -> bool:
    try:
        unstaged = subprocess.run(["git", "diff", "--quiet"], cwd=root, check=False).returncode
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root, check=False).returncode
        return unstaged != 0 or staged != 0
    except OSError:
        return True


def git_path_is_tracked(root: Path, relative_path: str) -> bool:
    try:
        return subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative_path],
            cwd=root,
            check=False,
            capture_output=True,
        ).returncode == 0
    except OSError:
        return False


@contextmanager
def state_lock(root: Path) -> Iterator[None]:
    lock_path = root / "project_state" / ".state.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"state lock already exists: {lock_path}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()} created_at={utc_now()}\n".encode("utf-8"))
        os.close(descriptor)
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def read_directive_events(root: Path) -> List[Dict[str, Any]]:
    path = root / "project_state" / "directives.jsonl"
    events: List[Dict[str, Any]] = []
    seen_directives: set[str] = set()
    if not path.exists():
        raise FileNotFoundError(path)
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid directive JSON at line {line_number}: {exc}") from exc
        event_type = event.get("event_type")
        directive_id = event.get("directive_id")
        if not directive_id:
            raise ValueError(f"directive event missing directive_id at line {line_number}")
        if event_type == "directive":
            if directive_id in seen_directives:
                raise ValueError(f"duplicate directive id: {directive_id}")
            seen_directives.add(directive_id)
            required = {
                "issued_at", "summary", "scope", "topic", "status", "supersedes",
                "effective_from_revision", "affected_files", "source",
            }
            missing = sorted(required - set(event))
            if missing:
                raise ValueError(f"directive {directive_id} missing fields: {', '.join(missing)}")
        elif event_type == "status_update":
            if directive_id not in seen_directives:
                raise ValueError(f"status update references unknown directive: {directive_id}")
        else:
            raise ValueError(f"unknown directive event_type at line {line_number}: {event_type}")
        events.append(event)
    return events


def fold_directives(events: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    directives: Dict[str, Dict[str, Any]] = {}
    superseded_by: Dict[str, str] = {}
    for event in events:
        directive_id = str(event["directive_id"])
        if event["event_type"] == "directive":
            directives[directive_id] = dict(event)
            for old_id in event.get("supersedes", []):
                if old_id == directive_id:
                    raise ValueError(f"directive {directive_id} cannot supersede itself")
                superseded_by[str(old_id)] = directive_id
        else:
            directives[directive_id]["status"] = event["status"]
            if event.get("superseded_by"):
                superseded_by[directive_id] = str(event["superseded_by"])
    for old_id, new_id in superseded_by.items():
        if old_id not in directives:
            raise ValueError(f"directive {new_id} supersedes unknown directive {old_id}")
        directives[old_id]["status"] = "superseded"
        directives[old_id]["superseded_by"] = new_id

    # A cycle is always an authoring error, even when all involved records are old.
    for start in superseded_by:
        seen: set[str] = set()
        current = start
        while current in superseded_by:
            if current in seen:
                raise ValueError(f"directive supersession cycle includes {current}")
            seen.add(current)
            current = superseded_by[current]
    return directives


def active_directives(root: Path) -> Dict[str, Dict[str, Any]]:
    folded = fold_directives(read_directive_events(root))
    return {key: value for key, value in folded.items() if value.get("status") == "active"}


def append_directive(
    root: Path,
    *,
    summary: str,
    scope: str,
    topic: str,
    supersedes: Sequence[str],
    affected_files: Sequence[str],
) -> str:
    events = read_directive_events(root)
    folded = fold_directives(events)
    for old_id in supersedes:
        if old_id not in folded:
            raise ValueError(f"cannot supersede unknown directive: {old_id}")
    today = datetime.now().strftime("%Y%m%d")
    prefix = f"DIR-{today}-"
    sequence = max(
        [int(item["directive_id"].split("-")[-1]) for item in events if item.get("event_type") == "directive" and item["directive_id"].startswith(prefix)] or [0]
    ) + 1
    directive_id = f"{prefix}{sequence:03d}"
    state = read_json(root / "project_state" / "current_state.json")
    new_event = {
        "event_type": "directive",
        "directive_id": directive_id,
        "issued_at": utc_now(),
        "summary": summary.strip(),
        "scope": scope.strip(),
        "topic": topic.strip(),
        "status": "active",
        "supersedes": list(supersedes),
        "effective_from_revision": int(state["state_revision"]) + 1,
        "affected_files": [normalize_rel(item) for item in affected_files],
        "source": "explicit_user_instruction",
    }
    append_events: List[Dict[str, Any]] = []
    for old_id in supersedes:
        append_events.append({
            "event_type": "status_update",
            "directive_id": old_id,
            "status": "superseded",
            "changed_at": utc_now(),
            "superseded_by": directive_id,
        })
    append_events.append(new_event)
    path = root / "project_state" / "directives.jsonl"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for event in append_events:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return directive_id


def _stable_doc_id(path: str) -> str:
    known = {
        "project_state/plans/mpp_training.md": "plan-mpp-training",
        "project_state/plans/server_maintenance.md": "plan-server-maintenance",
        "CURRENT_STATE.md": "view-current-state",
        "AGENTS.md": "agent-entry",
    }
    return known.get(path, "doc-" + hashlib.sha1(path.encode("utf-8")).hexdigest()[:12])


def _document_category(path: str) -> str:
    if path.startswith("project_state/plans/"):
        return "状态方案"
    if "分析报告/" in path:
        return "分析报告"
    if "部署方案/" in path:
        return "部署方案"
    if "学习指南/" in path:
        return "学习指南"
    if path.startswith("02_组会汇报/"):
        return "组会汇报"
    if path.startswith(".claude/"):
        return "本地Agent视图"
    if path.startswith(".qoder/"):
        return "缺失Qoder视图"
    return "项目入口"


def _classify_document(path: str, state: Mapping[str, Any]) -> Tuple[str, str, str, List[str]]:
    lower = path.lower()
    active_plan_paths = {normalize_rel(item["path"]): scope for scope, item in state.get("active_plans", {}).items()}
    connectivity: List[str] = []
    if re.search(r"ssh|scp", lower):
        connectivity.append("ssh")
    if "tunnel" in lower:
        connectivity.append("remote_tunnel")
    if "cmd_server" in lower or "远程命令" in path:
        connectivity.append("http_remote_command")

    if path in active_plan_paths:
        return active_plan_paths[path], "normative", "active", connectivity
    if path == "AGENTS.md":
        return "agent_entry", "normative", "active", connectivity
    if path == "CLAUDE.md":
        return "agent_adapter", "reference", "active", connectivity
    if path == "automation/README.md":
        return "gitee_job_protocol", "normative", "active", connectivity
    if path in {"CURRENT_STATE.md", "README.md", "experiments/experiment_dashboard.md", ".claude/next-steps.md", ".claude/session-brief.md"}:
        scope = "current_state" if path == "CURRENT_STATE.md" else "project_summary"
        return scope, "derived", "active", connectivity
    if path == "experiments/decision_log.md":
        return "project_decisions", "reference", "active", connectivity
    if path.endswith("服务器路径索引_20260701.md"):
        return "server_paths", "normative", "active", connectivity
    if path.endswith("MPP2后续方案与LoRA新数据实验建议_20260709.md"):
        return "mpp_training_reference", "reference", "active", connectivity
    superseded_patterns = (
        "服务器手动部署速查手册", "服务器训练操作手册", "服务器部署指南",
        "服务器迁移指南", "执行计划_mpp五划分验证", "mpp五划分uni2h_mlp执行框架",
        "mpp1_mpp4训练执行方案", "mpp-v3bis内部验证集二次实验方案",
        "执行计划_mpp1-5统一标准重跑", "mamba与频域模块实验部署方案",
        "deploy/sync_guide.md",
    )
    if any(pattern.lower() in lower for pattern in superseded_patterns):
        return "historical_guidance", "reference", "superseded", connectivity
    if connectivity:
        return "historical_connectivity", "reference", "historical", connectivity
    if path.startswith(".qoder/"):
        return "missing_local_adapter", "reference", "missing", connectivity
    return "historical_reference", "reference", "historical", connectivity


def scan_documents(root: Path) -> Dict[str, Any]:
    state = read_json(root / "project_state" / "current_state.json")
    tracked_path = root / "project_state" / "document_registry.json"
    old_path = tracked_path if tracked_path.exists() else root / ".claude" / "doc-registry.json"
    old_entries: Dict[str, Dict[str, Any]] = {}
    if old_path.exists():
        old = read_json(old_path)
        old_entries = {normalize_rel(item["path"]): item for item in old.get("documents", [])}

    live_paths: set[str] = set()
    for directory in ("01_指南与解读", "02_组会汇报", "project_state/plans", "deploy", "automation"):
        base = root / directory
        if base.exists():
            live_paths.update(normalize_rel(path.relative_to(root)) for path in base.rglob("*.md"))
    for path in (
        "README.md", "AGENTS.md", "CURRENT_STATE.md", "CLAUDE.md",
        ".claude/next-steps.md", ".claude/session-brief.md", ".claude/maintenance-plan.md",
        "experiments/experiment_dashboard.md", "experiments/decision_log.md",
    ):
        if (root / path).exists():
            live_paths.add(path)

    all_paths = sorted(live_paths | set(old_entries))
    verified_at = utc_now()
    documents: List[Dict[str, Any]] = []
    for rel_path in all_paths:
        file_path = root / rel_path
        old = old_entries.get(rel_path, {})
        if file_path.exists():
            scope, authority, lifecycle, connectivity = _classify_document(rel_path, state)
            digest = sha256_file(file_path)
        else:
            scope, authority, lifecycle, connectivity = "missing_local_adapter", "reference", "missing", []
            digest = ""
        entry = {
            "doc_id": _stable_doc_id(rel_path),
            "path": rel_path,
            "category": old.get("category") or _document_category(rel_path),
            "scope": scope,
            "authority": authority,
            "lifecycle": lifecycle,
            "verified_at": verified_at,
            "state_revision": int(state["state_revision"]),
            "content_sha256": digest,
            "supersedes": [],
            "superseded_by": [],
            "truth_sources": ["project_state/current_state.json"],
            "purpose": old.get("purpose", ""),
            "created": old.get("created", ""),
            "tags": list(old.get("tags", [])),
            "connectivity_modes": connectivity,
            "availability": (
                "tracked"
                if not (
                    rel_path == "CLAUDE.md"
                    or rel_path.startswith(".claude/")
                    or rel_path.startswith(".qoder/")
                    or rel_path.startswith("02_组会汇报/")
                    or (
                        rel_path.startswith("01_指南与解读/")
                        and not rel_path.endswith("服务器路径索引_20260701.md")
                        and not rel_path.endswith("MPP2后续方案与LoRA新数据实验建议_20260709.md")
                    )
                )
                else "local_only"
            ),
        }
        if lifecycle == "superseded":
            if scope == "historical_guidance" and "mpp" in rel_path.lower():
                entry["superseded_by"] = ["plan-mpp-training"]
            elif scope == "historical_guidance":
                entry["superseded_by"] = ["plan-server-maintenance"]
        documents.append(entry)

    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": verified_at,
        "state_revision": int(state["state_revision"]),
        "documents": documents,
        "summary": {
            "total": len(documents),
            "live": sum(1 for item in documents if item["lifecycle"] != "missing"),
            "missing": sum(1 for item in documents if item["lifecycle"] == "missing"),
            "active": sum(1 for item in documents if item["lifecycle"] == "active"),
            "superseded": sum(1 for item in documents if item["lifecycle"] == "superseded"),
            "historical": sum(1 for item in documents if item["lifecycle"] == "historical"),
        },
    }


def load_server_paths(root: Path) -> Dict[str, Any]:
    path = root / "configs" / "server_paths.yaml"
    if yaml is None:
        raise RuntimeError("PyYAML is required to read configs/server_paths.yaml")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("server_paths.yaml must contain a mapping")
    return value


def _label_csv_stats(path: Path) -> Dict[str, Any]:
    row_count = 0
    first_values: Dict[str, Tuple[str, ...]] = {}
    duplicate_barcodes: set[str] = set()
    conflicting_barcodes: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return {"row_count": 0, "unique_barcode_count": 0, "duplicate_barcode_count": 0, "conflicting_duplicate_count": 0}
        barcode_field = next((name for name in reader.fieldnames if name.lower() in {"barcode", "patch", "patch_id", "spot_id"}), reader.fieldnames[0])
        value_fields = [name for name in reader.fieldnames if name != barcode_field]
        for row in reader:
            row_count += 1
            barcode = str(row.get(barcode_field, ""))
            values = tuple(str(row.get(field, "")) for field in value_fields)
            previous = first_values.get(barcode)
            if previous is None:
                first_values[barcode] = values
            else:
                duplicate_barcodes.add(barcode)
                if previous != values:
                    conflicting_barcodes.add(barcode)
    return {
        "row_count": row_count,
        "unique_barcode_count": len(first_values),
        "duplicate_barcode_count": len(duplicate_barcodes),
        "conflicting_duplicate_count": len(conflicting_barcodes),
    }


def _asset_role(rel_path: str) -> str:
    lower = rel_path.lower()
    if "overlap_embargo_audit" in lower:
        return "embargo_audit"
    if "zscore_params" in lower:
        return "zscore_params"
    if "zscore_manifest" in lower:
        return "zscore_manifest"
    if "split_manifest" in lower:
        return "split_manifest"
    if "/labels/" in lower and lower.endswith(".csv"):
        return "standardized_label"
    if "split_" in lower:
        return "split_metadata"
    return "asset"


def _embargo_audit_stats(path: Path) -> Dict[str, Any]:
    row_count = 0
    train_neighbor_count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row_count += 1
            if str(row.get("neighbor_final_split", "")).lower() == "train":
                train_neighbor_count += 1
    return {"row_count": row_count, "train_neighbor_count": train_neighbor_count}


def build_mpp_path_index(root: Path) -> Dict[str, Any]:
    split_root = root / "mpp_standard_splits"
    if not split_root.exists():
        raise FileNotFoundError(split_root)
    assets: List[Dict[str, Any]] = []
    group_summary: Dict[str, Dict[str, Any]] = {}
    labels_validated = True
    for path in sorted(item for item in split_root.rglob("*") if item.is_file() and item.name != "path_index.json"):
        rel = normalize_rel(path.relative_to(root))
        match = re.search(r"mpp_standard_splits/group_(\d+)", rel)
        group = int(match.group(1)) if match else None
        role = _asset_role(rel)
        entry: Dict[str, Any] = {
            "path": rel,
            "group": group,
            "role": role,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if role == "standardized_label":
            stats = _label_csv_stats(path)
            entry.update(stats)
            entry["barcode_unique"] = stats["row_count"] == stats["unique_barcode_count"]
            entry["conflict_free"] = stats["conflicting_duplicate_count"] == 0
            if not entry["barcode_unique"] or not entry["conflict_free"]:
                labels_validated = False
        elif role == "embargo_audit":
            entry.update(_embargo_audit_stats(path))
            if entry["train_neighbor_count"]:
                labels_validated = False
        elif path.name == "split_info.json":
            split_info = read_json(path)
            entry["overlap_policy"] = split_info.get("overlap_policy")
            entry["block_size"] = split_info.get("block_size")
            entry["leakage_pairs"] = split_info.get("leakage_pairs")
            if group in (3, 5) and split_info.get("leakage_pairs") != 0:
                labels_validated = False
        elif path.name == "zscore_manifest.json":
            zscore = read_json(path)
            entry["fit_split"] = zscore.get("fit_split")
            entry["ddof"] = zscore.get("ddof")
            entry["n_pathways"] = zscore.get("n_pathways")
            entry["n_train_samples"] = zscore.get("n_train_samples")
            entry["n_val_samples"] = zscore.get("n_val_samples")
            entry["n_external_samples"] = zscore.get("n_external_samples")
            if zscore.get("fit_split") != "train" or zscore.get("ddof") != 1 or zscore.get("n_pathways") != 30:
                labels_validated = False
        assets.append(entry)
        if group is not None:
            summary = group_summary.setdefault(str(group), {"file_count": 0, "total_bytes": 0, "embargo_audit_present": False})
            summary["file_count"] += 1
            summary["total_bytes"] += path.stat().st_size
            if role == "embargo_audit":
                summary["embargo_audit_present"] = True
    for group in (3, 5):
        if not group_summary.get(str(group), {}).get("embargo_audit_present"):
            labels_validated = False
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "source_root": "mpp_standard_splits",
        "raw_ssgsea_template": r"D:\AIPatho\Patch\visiumhd_patch\{group}\{patient}\{patient}_ssGSEA.csv",
        "external_xzy_raw_ssgsea": r"D:\AIPatho\Patch\visiumhd_patch\2\XZY\XZY_ssGSEA.csv",
        "fit_policy": "train_only_ddof_1",
        "external_policy": "MPP2_XZY_fixed_after_checkpoint_selection",
        "labels_validated": labels_validated,
        "validation_note": "False means at least one label CSV has duplicate/conflicting barcodes or a mandatory embargo audit is absent.",
        "groups": group_summary,
        "summary": {"file_count": len(assets), "total_bytes": sum(item["size_bytes"] for item in assets)},
        "assets": assets,
    }


def migrate_experiment_provenance(registry: MutableMapping[str, Any]) -> bool:
    changed = False
    standardized_ids = {f"mpp{group}_std10val{'_embargo' if group in (3, 5) else ''}_xzy_ext_uni2h_mlp_20260706" for group in range(1, 6)}
    for experiment in registry.get("experiments", []):
        experiment_id = experiment.get("id", "")
        done = str(experiment.get("status", "")).startswith("done")
        default_evidence = "accepted" if experiment_id in standardized_ids else ("historical" if done else "pending")
        defaults = {
            "source_commit": None,
            "job_id": None,
            "result_id": f"legacy-import-{experiment_id}" if done else None,
            "data_manifest_id": "mpp-standard-splits-v1" if experiment_id in standardized_ids else None,
            "path_index_version": SCHEMA_VERSION if experiment_id in standardized_ids else None,
            "result_manifest_sha256": None,
            "imported_at": experiment.get("completed_at") if done else None,
            "evidence_status": default_evidence,
            "provenance_complete": False if done else None,
            "supersedes_results": [],
        }
        for key, value in defaults.items():
            if key not in experiment:
                experiment[key] = value
                changed = True
    return changed


def _result_metrics_line(experiment: Mapping[str, Any]) -> str:
    fields = []
    if experiment.get("external_xzy_pcc") is not None:
        fields.append(f"PCC={experiment['external_xzy_pcc']}")
    if experiment.get("external_xzy_mae_raw") is not None:
        fields.append(f"raw_MAE={experiment['external_xzy_mae_raw']}")
    if experiment.get("external_xzy_r2_raw") is not None:
        fields.append(f"raw_R2={experiment['external_xzy_r2_raw']}")
    return ", ".join(fields) or "metrics recorded in Registry"


def render_current_state(root: Path, state: Mapping[str, Any], registry: Mapping[str, Any]) -> str:
    directives = active_directives(root)
    experiments = {item.get("id"): item for item in registry.get("experiments", [])}
    state_hash = sha256_bytes(canonical_json_bytes(state))
    lines = [
        "# PFMval Current State",
        "",
        "> AUTO-GENERATED by `python deploy/pfmval_ops.py state sync`. Do not edit manually.",
        f"> State revision: `{state['state_revision']}` | Updated: `{state['updated_at']}` | Source commit: `{state['source_commit']}`",
        f"> State SHA-256: `{state_hash}`",
        "",
        "## Current directives",
        "",
    ]
    for directive_id in state.get("active_directive_ids", []):
        directive = directives.get(directive_id)
        if directive:
            lines.append(f"- `{directive_id}` [{directive['scope']}/{directive['topic']}]: {directive['summary']}")
    lines.extend(["", "## Active plans", ""])
    for scope, plan in state.get("active_plans", {}).items():
        lines.append(f"- `{scope}`: [{plan['path']}]({plan['path']})")
    transport = state.get("server_transport", {})
    lines.extend([
        "",
        "## Server transport",
        "",
        f"- Mode: **{transport.get('mode', 'unknown')}** via remote `{transport.get('remote_name', 'unknown')}`.",
        f"- Forbidden direct channels: {', '.join(transport.get('forbidden_direct_connections', []))}.",
        "",
        "## Latest accepted results",
        "",
    ])
    for result_id in state.get("latest_accepted_result_ids", []):
        experiment = experiments.get(result_id)
        if experiment:
            lines.append(f"- `{result_id}` ({experiment.get('evidence_status', 'unknown')}): {_result_metrics_line(experiment)}")
        else:
            lines.append(f"- `{result_id}`: **missing from Registry**")
    if state.get("pending_result_ids"):
        lines.extend(["", "## Pending result imports", ""])
        lines.extend(f"- `{item}`" for item in state["pending_result_ids"])
    lines.extend(["", "## Hard blocks", ""])
    lines.extend(f"- `{item}`" for item in state.get("blocked_actions", []))
    lines.extend(["", "## Superseded conclusions", ""])
    lines.extend(f"- {item}" for item in state.get("superseded_conclusions", []))
    if state.get("notes"):
        lines.extend(["", "## Notes and integrity warnings", ""])
        lines.extend(f"- {item}" for item in state.get("notes", []))
    lines.extend([
        "",
        "## Required checks",
        "",
        "```powershell",
        "python deploy/pfmval_ops.py agent start-check --strict --task general",
        "python deploy/pfmval_ops.py paths validate",
        "```",
        "",
    ])
    return "\n".join(lines)


def _render_local_next_steps(state: Mapping[str, Any], registry: Mapping[str, Any]) -> str:
    policy = registry.get("current_mpp_policy", {})
    return "\n".join([
        "# 下一步方向（生成视图）",
        "",
        "> AUTO-GENERATED from `project_state/current_state.json`; do not edit manually.",
        f"> State revision: `{state['state_revision']}` | Updated: `{state['updated_at']}`",
        "",
        f"- 当前 MPP 主线：MPP{policy.get('selected_mpp', 'unknown')}。",
        f"- 下一步：{policy.get('next_recommended_experiment', 'review CURRENT_STATE.md')}。",
        "- 服务器通信：Gitee-only；SSH/SCP/HTTP/Tunnel 不是 active 通道。",
        "- 自动排障 watcher：后续独立任务，尚未启用。",
        "",
    ])


def _render_session_brief(state: Mapping[str, Any]) -> str:
    return "\n".join([
        "# 会话快照（生成视图）",
        "",
        "> AUTO-GENERATED from `project_state/current_state.json`; do not edit manually.",
        f"> State revision: `{state['state_revision']}` | Updated: `{state['updated_at']}`",
        "",
        "新会话必须先读取根目录 `CURRENT_STATE.md`。实验事实读取 Registry，服务器路径读取 `configs/server_paths.yaml`。",
        "",
        f"Pending result IDs: {', '.join(state.get('pending_result_ids', [])) or 'none'}",
        "",
    ])


def _readme_state_block(state: Mapping[str, Any], registry: Mapping[str, Any]) -> str:
    policy = registry.get("current_mpp_policy", {})
    return "\n".join([
        "<!-- project-state:start -->",
        "## 当前项目状态（自动生成）",
        "",
        f"- 状态版本：`{state['state_revision']}`；完整入口：[CURRENT_STATE.md](CURRENT_STATE.md)。",
        f"- 当前 MPP 主线：**MPP{policy.get('selected_mpp', 'unknown')}**；其它统一重跑结果保留为背景/方法参考。",
        "- 服务器通信：**Gitee-only**；SSH、SCP、HTTP 远程命令和 Tunnel 均非 active 通道。",
        "- 实验事实源：`experiments/experiment_registry.json`；Dashboard 为派生视图。",
        "",
        "<!-- project-state:end -->",
    ])


def replace_readme_state_block(readme: str, block: str) -> str:
    pattern = re.compile(r"<!-- project-state:start -->.*?<!-- project-state:end -->", re.DOTALL)
    if pattern.search(readme):
        return pattern.sub(block, readme)
    lines = readme.splitlines()
    if lines:
        return "\n".join([lines[0], "", block, ""] + lines[1:]) + ("\n" if readme.endswith("\n") else "")
    return block + "\n"


def compute_source_hashes(root: Path) -> Dict[str, str]:
    files = {
        "experiment_registry_sha256": root / "experiments" / "experiment_registry.json",
        "experiment_dashboard_sha256": root / "experiments" / "experiment_dashboard.md",
        "document_registry_sha256": root / "project_state" / "document_registry.json",
        "mpp_path_index_sha256": root / "mpp_standard_splits" / "path_index.json",
        "server_paths_sha256": root / "configs" / "server_paths.yaml",
    }
    hashes = {key: sha256_file(path) if path.exists() else "" for key, path in files.items()}
    document_registry = files["document_registry_sha256"]
    if document_registry.exists():
        # Derived views contain the revision and hashes produced from the state
        # package. Hashing those volatile fields creates an endless
        # state -> view -> document-registry -> state revision loop. The source
        # hash therefore covers lifecycle decisions and normative/reference
        # content, while excluding derived-view bookkeeping.
        semantic = read_json(document_registry)
        for field in ("updated_at", "state_revision", "summary"):
            semantic.pop(field, None)
        for document in semantic.get("documents", []):
            document.pop("verified_at", None)
            document.pop("state_revision", None)
            if document.get("authority") == "derived":
                document.pop("content_sha256", None)
        hashes["document_registry_sha256"] = sha256_bytes(canonical_json_bytes(semantic))
    return hashes


def sync_state(root: Path, *, force_revision: bool = False, write_views: bool = True) -> Dict[str, Any]:
    state_path = root / "project_state" / "current_state.json"
    state = read_json(state_path)
    registry = read_json(root / "experiments" / "experiment_registry.json")
    active = active_directives(root)
    desired_ids = list(active)
    new_hashes = compute_source_hashes(root)
    changed = (
        state.get("active_directive_ids") != desired_ids
        or state.get("source_hashes") != new_hashes
    )
    if changed or force_revision:
        state["state_revision"] = int(state.get("state_revision", 0)) + 1
        state["updated_at"] = utc_now()
    state["active_directive_ids"] = desired_ids
    if changed or force_revision:
        # This is intentionally the commit on which the state update is based.
        # Requiring it to equal the commit that later contains this file would
        # create an impossible self-referential commit loop.
        state["source_commit"] = git_head(root)
    state["source_hashes"] = new_hashes
    write_json_atomic(state_path, state)
    if write_views:
        write_text_atomic(root / "CURRENT_STATE.md", render_current_state(root, state, registry))
        claude_dir = root / ".claude"
        if claude_dir.exists():
            write_text_atomic(claude_dir / "next-steps.md", _render_local_next_steps(state, registry))
            write_text_atomic(claude_dir / "session-brief.md", _render_session_brief(state))
        readme_path = root / "README.md"
        if readme_path.exists():
            current = readme_path.read_text(encoding="utf-8")
            write_text_atomic(readme_path, replace_readme_state_block(current, _readme_state_block(state, registry)))
    return state


class ValidationReport:
    def __init__(self) -> None:
        self.pass_items: List[str] = []
        self.warn_items: List[str] = []
        self.fail_items: List[str] = []

    def passed(self, message: str) -> None:
        self.pass_items.append(message)

    def warn(self, message: str) -> None:
        self.warn_items.append(message)

    def fail(self, message: str) -> None:
        self.fail_items.append(message)

    @property
    def ok(self) -> bool:
        return not self.fail_items

    def emit(self) -> None:
        for message in self.pass_items:
            print(f"[PASS] {message}")
        for message in self.warn_items:
            print(f"[WARN] {message}")
        for message in self.fail_items:
            print(f"[FAIL] {message}")
        print(f"[SUMMARY] PASS={len(self.pass_items)} WARN={len(self.warn_items)} FAIL={len(self.fail_items)}")


def validate_server_paths(root: Path, report: ValidationReport, *, task: str = "general") -> None:
    try:
        registry = load_server_paths(root)
    except Exception as exc:
        report.fail(f"server path registry unreadable: {exc}")
        return
    if registry.get("schema_version") != SCHEMA_VERSION:
        report.fail("server path registry schema_version must be 1.0")
    paths = registry.get("paths")
    if not isinstance(paths, dict) or not paths:
        report.fail("server path registry contains no paths")
        return
    for path_id, entry in paths.items():
        if entry.get("status") not in {"active", "legacy", "deprecated"}:
            report.fail(f"path {path_id} has invalid status")
        if not entry.get("path"):
            report.fail(f"path {path_id} has no path value")
        # Server worktrees must not require local-only legacy assets. The
        # default/general task keeps local validation; explicit server checks
        # validate server/both paths without demanding local-only directories.
        check_local_path = task != "server" and entry.get("required_on") in {"local", "both"}
        if check_local_path and not re.match(r"^[A-Za-z]:[\\/]", str(entry["path"])):
            local_path = root / str(entry["path"])
            if not local_path.exists():
                report.fail(f"required local path missing: {path_id} -> {entry['path']}")
    active_path_files = [
        "extract_uni2h_mpp.py",
        "prepare_mpp_zscore.py",
        "train_mpp_uni2h_mlp.py",
        "scripts/generate_standard_splits.py",
        "scripts/audit_mpp_coordinates.py",
        "scripts/rebuild_zscore_from_manifest.py",
        "scripts/inspect_mpp_data.py",
    ]
    hardcoded_default = re.compile(r"(?:default\s*=|DEFAULT_[A-Z_]+\s*=|ROOT\s*=)\s*(?:Path\()?r?[\"']D:\\\\AIPatho", re.IGNORECASE)
    for relative in active_path_files:
        code_path = root / relative
        if not code_path.exists():
            report.fail(f"active path-aware script missing: {relative}")
            continue
        content = code_path.read_text(encoding="utf-8")
        if "get_registered_path" not in content:
            report.fail(f"active script does not resolve stable path ids: {relative}")
        if hardcoded_default.search(content):
            report.fail(f"active script still defines a hardcoded D:\\AIPatho default: {relative}")
    index_path = root / "mpp_standard_splits" / "path_index.json"
    if index_path.exists():
        index = read_json(index_path)
        if not index.get("labels_validated", False):
            message = "MPP standardized labels contain duplicate/conflicting barcodes; regeneration requires a separate approved task"
            if task == "training":
                report.fail(message)
            else:
                report.warn(message)
        else:
            report.passed("MPP label and embargo index is validated")
    else:
        report.warn("mpp_standard_splits/path_index.json is missing")


def validate_state(root: Path, *, strict: bool = False, task: str = "general") -> ValidationReport:
    report = ValidationReport()
    try:
        state = read_json(root / "project_state" / "current_state.json")
        registry = read_json(root / "experiments" / "experiment_registry.json")
        documents = read_json(root / "project_state" / "document_registry.json")
        directives = active_directives(root)
    except Exception as exc:
        report.fail(f"state package unreadable: {exc}")
        return report

    schema_root = root / "project_state" / "schemas"
    try:
        state_schema_checked = validate_against_schema(state, schema_root / "current_state.schema.json", "current_state")
        document_schema_checked = validate_against_schema(documents, schema_root / "document_registry.schema.json", "document_registry")
        directive_schema_checked = all(
            validate_against_schema(event, schema_root / "directives.schema.json", f"directive event {index}")
            for index, event in enumerate(read_directive_events(root), 1)
        )
        if state_schema_checked and document_schema_checked and directive_schema_checked:
            report.passed("JSON schemas validate current state, document registry and directives")
        else:
            report.warn("jsonschema package unavailable; manual state validation fallback used")
    except (FileNotFoundError, ValueError) as exc:
        report.fail(str(exc))

    required = {
        "schema_version", "state_revision", "updated_at", "source_commit", "active_directive_ids",
        "active_plans", "server_transport", "active_training_jobs", "pending_result_ids",
        "latest_accepted_result_ids", "blocked_actions", "superseded_conclusions", "source_hashes",
    }
    missing = sorted(required - set(state))
    if missing:
        report.fail(f"current_state missing fields: {', '.join(missing)}")
    elif state.get("schema_version") != SCHEMA_VERSION:
        report.fail("current_state schema_version must be 1.0")
    else:
        report.passed("current_state required fields")

    if state.get("active_directive_ids") != list(directives):
        report.fail("current_state active_directive_ids does not match folded directive log")
    topics: Dict[Tuple[str, str], List[str]] = {}
    for directive_id, directive in directives.items():
        topics.setdefault((directive["scope"], directive["topic"]), []).append(directive_id)
    conflicts = {key: ids for key, ids in topics.items() if len(ids) > 1}
    if conflicts:
        report.fail(f"unresolved active directive conflicts: {conflicts}")
    else:
        report.passed("active directives have no topic conflicts")

    document_list = documents.get("documents", [])
    doc_by_id = {item["doc_id"]: item for item in document_list}
    if len(doc_by_id) != len(document_list):
        report.fail("document registry contains duplicate doc_id values")
    normalized_paths = [normalize_rel(item.get("path", "")).lower() for item in document_list]
    if len(set(normalized_paths)) != len(normalized_paths):
        report.fail("document registry contains duplicate normalized paths")
    revision_delta = int(state.get("state_revision", 0)) - int(documents.get("state_revision", 0))
    if revision_delta > 1:
        report.fail("document registry is more than one state revision behind")
    for document in document_list:
        document_path = root / document.get("path", "")
        if document.get("lifecycle") == "active" and not document_path.exists() and document.get("availability") != "local_only":
            report.fail(f"active document is missing: {document.get('path')}")
        if document.get("lifecycle") == "active" and document.get("authority") == "normative" and document_path.exists():
            if document.get("content_sha256") != sha256_file(document_path):
                report.fail(f"active normative document hash is stale: {document.get('path')}")
        for successor in document.get("superseded_by", []):
            if successor not in doc_by_id:
                report.fail(f"superseded document points to unknown successor {successor}: {document.get('path')}")
    normative_scope: Dict[str, List[str]] = {}
    for item in document_list:
        if item.get("lifecycle") == "active" and item.get("authority") == "normative":
            normative_scope.setdefault(item.get("scope", ""), []).append(item["doc_id"])
    duplicates = {scope: ids for scope, ids in normative_scope.items() if len(ids) > 1}
    if duplicates:
        report.fail(f"multiple active normative documents in a scope: {duplicates}")
    for scope, plan in state.get("active_plans", {}).items():
        document = doc_by_id.get(plan.get("doc_id"))
        if not document:
            report.fail(f"active plan {scope} is absent from document registry")
        elif document.get("lifecycle") != "active" or document.get("path") != normalize_rel(plan.get("path", "")):
            report.fail(f"active plan {scope} points to non-active or mismatched document")
    forbidden = set(state.get("server_transport", {}).get("forbidden_direct_connections", []))
    if state.get("server_transport", {}).get("mode") != "gitee_only":
        report.fail("server transport is not gitee_only")
    for document in documents.get("documents", []):
        if document.get("lifecycle") == "active" and forbidden.intersection(document.get("connectivity_modes", [])):
            report.fail(f"active document conflicts with Gitee-only transport: {document['path']}")

    experiments = {item.get("id"): item for item in registry.get("experiments", [])}
    for result_id in state.get("latest_accepted_result_ids", []):
        experiment = experiments.get(result_id)
        if not experiment:
            report.fail(f"latest accepted result missing from Registry: {result_id}")
        elif experiment.get("evidence_status") != "accepted":
            report.fail(f"latest result is not accepted evidence: {result_id}")
        elif not experiment.get("provenance_complete", False):
            report.warn(f"accepted legacy result lacks a complete result envelope: {result_id}")
    for job in state.get("active_training_jobs", []):
        if job.get("experiment_id") not in experiments:
            report.fail(f"active job references unknown experiment: {job}")
        if not job.get("source_commit"):
            report.fail(f"active job has no source_commit: {job}")
        if job.get("phase") == "formal" and not job.get("formal_training_approved"):
            report.fail(f"formal job has no explicit user approval: {job.get('job_id')}")

    actual_hashes = compute_source_hashes(root)
    for key, expected in state.get("source_hashes", {}).items():
        actual = actual_hashes.get(key, "")
        if not expected:
            message = f"state source hash is empty: {key}"
            (report.fail if strict else report.warn)(message)
        elif actual != expected:
            report.fail(f"state source hash mismatch: {key}")
    current_view = root / "CURRENT_STATE.md"
    if current_view.exists():
        state_hash = sha256_bytes(canonical_json_bytes(state))
        if state_hash not in current_view.read_text(encoding="utf-8"):
            report.fail("CURRENT_STATE.md was not generated from the current state payload")
        else:
            report.passed("CURRENT_STATE.md matches current_state.json")
    else:
        report.fail("CURRENT_STATE.md missing")

    transaction_root = root / "project_state" / ".transactions"
    if transaction_root.exists() and any(transaction_root.iterdir()):
        report.fail("unfinished state transaction exists")
    if (root / "project_state" / ".state.lock").exists():
        report.fail("state lock exists; a writer may have been interrupted")

    imported_results = {item.get("result_id") for item in registry.get("experiments", []) if item.get("result_id")}
    inbox = root / "project_state" / "inbox"
    local_pending: List[str] = []
    if inbox.exists():
        for manifest_path in inbox.rglob("result.json"):
            try:
                result_id = read_json(manifest_path).get("result_id")
            except Exception as exc:
                report.fail(f"invalid inbox result envelope {manifest_path}: {exc}")
                continue
            if result_id and result_id not in imported_results:
                local_pending.append(result_id)
    if sorted(local_pending) != sorted(state.get("pending_result_ids", [])):
        report.fail(f"pending result state does not match inbox: inbox={local_pending} state={state.get('pending_result_ids', [])}")
    elif local_pending:
        report.warn(f"pending results must be imported before model conclusions: {local_pending}")

    validate_server_paths(root, report, task=task)
    if not report.fail_items:
        report.passed("state package validation completed")
    return report


def validate_result_envelope(bundle_dir: Path, manifest: Mapping[str, Any]) -> None:
    project_root = Path(__file__).resolve().parent.parent
    validate_against_schema(
        manifest,
        project_root / "project_state" / "schemas" / "result_envelope.schema.json",
        "result envelope",
    )
    required = {"schema_version", "result_id", "job_id", "experiment_id", "source_commit", "phase", "status", "created_at", "artifacts", "metrics"}
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"result envelope missing fields: {', '.join(missing)}")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported result envelope version: {manifest['schema_version']}")
    if manifest["phase"] not in {"preflight", "smoke", "formal"}:
        raise ValueError("invalid result phase")
    if manifest["status"] not in {"success", "failed", "incomplete"}:
        raise ValueError("invalid result status")
    for identifier in (manifest["result_id"], manifest["experiment_id"]):
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", str(identifier)):
            raise ValueError(f"unsafe result identifier: {identifier}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(manifest["job_id"])):
        raise ValueError(f"unsafe result job_id: {manifest['job_id']}")
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", str(manifest["source_commit"])):
        raise ValueError("result source_commit is not a Git commit hash")
    if manifest["phase"] == "formal" and manifest["status"] == "success" and not manifest.get("formal_training_approved"):
        raise ValueError("successful formal result has no explicit approval marker")
    if manifest["phase"] in {"smoke", "formal"} and manifest["status"] == "success":
        metrics = manifest.get("metrics", {})
        if not manifest.get("data_manifest_id") or not manifest.get("path_index_version"):
            raise ValueError("successful training result is missing data/path-index provenance")
        if "best_epoch" not in metrics:
            raise ValueError("successful training result is missing best_epoch")
        if not any(key in metrics for key in ("best_val_loss", "best_val_pcc", "external_xzy_pcc")):
            raise ValueError("successful training result has no selection/evaluation metric")
    total = 0
    for artifact in manifest.get("artifacts", []):
        raw_path = str(artifact.get("path", ""))
        if "\\" in raw_path:
            raise ValueError(f"artifact path must use bundle-relative POSIX separators: {raw_path}")
        rel = PurePosixPath(raw_path)
        if not raw_path or rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe artifact path: {rel}")
        path = bundle_dir / Path(*rel.parts)
        if path.name not in ALLOWED_RESULT_FILES:
            raise ValueError(f"artifact is not allowlisted: {rel}")
        if not path.exists() or not path.is_file():
            raise ValueError(f"artifact missing: {rel}")
        size = path.stat().st_size
        if size != int(artifact.get("size_bytes", -1)):
            raise ValueError(f"artifact size mismatch: {rel}")
        if size > MAX_RESULT_FILE_BYTES:
            raise ValueError(f"artifact exceeds {MAX_RESULT_FILE_BYTES} bytes: {rel}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(artifact.get("sha256", "")).lower()):
            raise ValueError(f"artifact has invalid SHA-256: {rel}")
        if sha256_file(path) != artifact.get("sha256"):
            raise ValueError(f"artifact hash mismatch: {rel}")
        total += size
    if total > MAX_RESULT_TOTAL_BYTES:
        raise ValueError(f"result bundle exceeds {MAX_RESULT_TOTAL_BYTES} bytes")
    expected_files = {"result.json", *(str(item.get("path", "")) for item in manifest.get("artifacts", []))}
    actual_files = {
        normalize_rel(path.relative_to(bundle_dir))
        for path in bundle_dir.rglob("*")
        if path.is_file()
    }
    unexpected_files = sorted(actual_files - expected_files)
    if unexpected_files:
        raise ValueError(f"result bundle contains unlisted files: {unexpected_files}")
    def check_finite(value: Any, location: str) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite metric at {location}")
        if isinstance(value, dict):
            for key, item in value.items():
                check_finite(item, f"{location}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                check_finite(item, f"{location}[{index}]")
    check_finite(manifest.get("metrics", {}), "metrics")
    for artifact in manifest.get("large_artifacts", []):
        server_path = str(artifact.get("server_path", ""))
        if not re.match(r"^[A-Za-z]:[\\/]", server_path):
            raise ValueError(f"large artifact is not an absolute Windows server path: {server_path}")
        if int(artifact.get("size_bytes", -1)) < 0:
            raise ValueError(f"large artifact has invalid size: {server_path}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(artifact.get("sha256", "")).lower()):
            raise ValueError(f"large artifact has invalid SHA-256: {server_path}")


def validate_result_job_binding(root: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    job_id = str(manifest.get("job_id", ""))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id):
        raise ValueError(f"unsafe result job_id: {job_id}")
    job_path = root / "automation" / "jobs" / job_id / "job.json"
    if not job_path.exists():
        raise ValueError(f"result has no dispatched job envelope: {normalize_rel(job_path.relative_to(root))}")
    job = read_json(job_path)
    validate_against_schema(
        job,
        root / "project_state" / "schemas" / "server_job.schema.json",
        "bound server job",
    )
    comparisons = {
        "job_id": job.get("job_id"),
        "experiment_id": job.get("experiment_id"),
        "source_commit": job.get("source_commit"),
        "phase": job.get("phase"),
        "data_manifest_id": job.get("data_manifest_id"),
        "path_index_version": job.get("path_index_version"),
    }
    for field, expected in comparisons.items():
        if manifest.get(field) != expected:
            raise ValueError(f"result/job binding mismatch for {field}: result={manifest.get(field)!r} job={expected!r}")
    registry = read_json(root / "experiments" / "experiment_registry.json")
    experiment = next((item for item in registry.get("experiments", []) if item.get("id") == job.get("experiment_id")), None)
    if experiment is None:
        raise ValueError("bound job references an unknown experiment")
    validate_job_semantics(
        str(job.get("phase")),
        str(job.get("command_id")),
        job.get("parameters", {}),
        experiment=experiment,
        path_ids=job.get("path_ids", []),
    )
    if job.get("phase") == "formal":
        approval = job.get("formal_training_approval") or {}
        if not approval.get("approved") or approval.get("source") != "explicit_user_instruction":
            raise ValueError("bound formal job has no explicit approval")
        if approval.get("job_id") != job_id or approval.get("source_commit") != job.get("source_commit"):
            raise ValueError("bound formal approval does not match job_id/source_commit")
    if git_head(root) != "unknown" and not git_commit_exists(root, str(job.get("source_commit", ""))):
        raise ValueError(f"bound job source_commit is unavailable locally: {job.get('source_commit')}")
    return job


def recover_incomplete_result_transactions(root: Path) -> List[str]:
    """Recover or finalize result-import transactions left by a killed process."""
    transaction_root = root / "project_state" / ".transactions"
    if not transaction_root.exists():
        return []
    targets = {
        "experiment_registry.json": root / "experiments" / "experiment_registry.json",
        "experiment_dashboard.md": root / "experiments" / "experiment_dashboard.md",
        "current_state.json": root / "project_state" / "current_state.json",
        "CURRENT_STATE.md": root / "CURRENT_STATE.md",
    }
    recovered: List[str] = []
    for transaction_dir in sorted(path for path in transaction_root.iterdir() if path.is_dir()):
        metadata_path = transaction_dir / "transaction.json"
        backup_dir = transaction_dir / "backup"
        if not metadata_path.exists() or not backup_dir.exists():
            raise ValueError(f"unrecoverable result transaction: {normalize_rel(transaction_dir.relative_to(root))}")
        metadata = read_json(metadata_path)
        status = metadata.get("status")
        staged_remaining = [name for name in targets if (transaction_dir / name).exists()]
        if status == "committing" and not staged_remaining:
            # Every os.replace completed; only cleanup/local-view refresh was
            # interrupted. The committed generation is authoritative.
            action = "finalized"
        else:
            # prepared, explicitly interrupted, or a partial committing set:
            # restore the complete previous generation from backups.
            required_backups = [name for name in targets if name != "CURRENT_STATE.md"]
            missing = [name for name in required_backups if not (backup_dir / name).exists()]
            if missing:
                raise ValueError(f"unrecoverable transaction backups are missing: {missing}")
            for name, target in targets.items():
                backup = backup_dir / name
                if backup.exists():
                    shutil.copy2(backup, target)
            action = "rolled_back"
        recovered.append(f"{transaction_dir.name}:{action}")
        shutil.rmtree(transaction_dir)
    if transaction_root.exists() and not any(transaction_root.iterdir()):
        transaction_root.rmdir()

    # Rebuild non-authoritative local views after either finalization or rollback.
    state = read_json(root / "project_state" / "current_state.json")
    registry = read_json(root / "experiments" / "experiment_registry.json")
    if (root / ".claude").exists():
        write_text_atomic(root / ".claude" / "next-steps.md", _render_local_next_steps(state, registry))
        write_text_atomic(root / ".claude" / "session-brief.md", _render_session_brief(state))
    readme_path = root / "README.md"
    if readme_path.exists():
        write_text_atomic(
            readme_path,
            replace_readme_state_block(readme_path.read_text(encoding="utf-8"), _readme_state_block(state, registry)),
        )
    return recovered


def import_result_bundle(root: Path, bundle_dir: Path) -> Dict[str, Any]:
    manifest_path = bundle_dir / "result.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    validate_result_envelope(bundle_dir, manifest)
    validate_result_job_binding(root, manifest)
    registry_path = root / "experiments" / "experiment_registry.json"
    dashboard_path = root / "experiments" / "experiment_dashboard.md"
    state_path = root / "project_state" / "current_state.json"
    transaction_root = root / "project_state" / ".transactions"
    transaction_id = f"txn-{uuid.uuid4().hex}"
    transaction_dir = transaction_root / transaction_id

    with state_lock(root):
        recover_incomplete_result_transactions(root)
        registry = read_json(registry_path)
        state = read_json(state_path)
        target = next((item for item in registry.get("experiments", []) if item.get("id") == manifest["experiment_id"]), None)
        if target is None:
            raise ValueError(f"unknown experiment id: {manifest['experiment_id']}")
        duplicate_owner = next(
            (
                item.get("id")
                for item in registry.get("experiments", [])
                if (
                    item.get("result_id") == manifest["result_id"]
                    or (item.get("last_preflight") or {}).get("result_id") == manifest["result_id"]
                )
                and item.get("id") != manifest["experiment_id"]
            ),
            None,
        )
        if duplicate_owner:
            raise ValueError(f"result_id is already bound to another experiment: {duplicate_owner}")
        if target.get("result_id") == manifest["result_id"] or (target.get("last_preflight") or {}).get("result_id") == manifest["result_id"]:
            return {"status": "already_imported", "result_id": manifest["result_id"]}
        if git_head(root) != "unknown" and not git_commit_exists(root, str(manifest["source_commit"])):
            raise ValueError(f"result source_commit is unavailable locally: {manifest['source_commit']}")
        existing_commit = target.get("source_commit")
        if manifest["phase"] != "preflight" and existing_commit and existing_commit != manifest["source_commit"]:
            raise ValueError(f"source_commit mismatch: registry={existing_commit} result={manifest['source_commit']}")
        imported_at = utc_now()
        manifest_sha256 = sha256_file(manifest_path)
        promotable_success = manifest["status"] == "success" and manifest["phase"] in {"smoke", "formal"}
        if manifest["phase"] == "preflight":
            target["last_preflight"] = {
                "result_id": manifest["result_id"],
                "job_id": manifest["job_id"],
                "source_commit": manifest["source_commit"],
                "status": manifest["status"],
                "imported_at": imported_at,
                "result_manifest_sha256": manifest_sha256,
            }
        else:
            previous_result_id = target.get("result_id")
            target["source_commit"] = manifest["source_commit"]
            target["job_id"] = manifest["job_id"]
            target["result_id"] = manifest["result_id"]
            target["data_manifest_id"] = manifest.get("data_manifest_id")
            target["path_index_version"] = manifest.get("path_index_version")
            target["result_manifest_sha256"] = manifest_sha256
            target["imported_at"] = imported_at
            target["evidence_status"] = "accepted" if promotable_success else ("rejected" if manifest["status"] == "failed" else "pending")
            target["provenance_complete"] = True
            target["result_phase"] = manifest["phase"]
            superseded = list(manifest.get("supersedes_results", []))
            if previous_result_id and previous_result_id != manifest["result_id"] and previous_result_id not in superseded:
                superseded.append(previous_result_id)
            target["supersedes_results"] = superseded
            if promotable_success:
                target["status"] = "done"
            elif manifest["status"] == "failed":
                target["status"] = "failed"
        metrics = manifest.get("metrics", {})
        allowed_metrics = {
            "best_epoch", "best_val_loss", "best_val_pcc", "train_val_gap", "external_xzy_pcc",
            "external_xzy_mae", "external_xzy_r2", "external_xzy_mae_raw", "external_xzy_r2_raw", "test_loss",
        }
        for key in allowed_metrics:
            if manifest["phase"] != "preflight" and key in metrics:
                target[key] = metrics[key]
        registry["updated_at"] = utc_now()

        # Import lazily to avoid a module cycle when finalize_experiment calls state sync.
        from scripts.finalize_experiment import build_dashboard

        dashboard_text = build_dashboard(registry)
        accepted_ids = list(state.get("latest_accepted_result_ids", []))
        if manifest["phase"] != "preflight":
            if target["evidence_status"] == "accepted" and target["id"] not in accepted_ids:
                if manifest["phase"] == "formal":
                    accepted_ids.append(target["id"])
            elif target["evidence_status"] != "accepted":
                accepted_ids = [item for item in accepted_ids if item != target["id"]]
        state["latest_accepted_result_ids"] = accepted_ids
        state["pending_result_ids"] = [item for item in state.get("pending_result_ids", []) if item != manifest["result_id"]]
        state["state_revision"] = int(state["state_revision"]) + 1
        state["updated_at"] = utc_now()
        state["source_commit"] = git_head(root)

        transaction_dir.mkdir(parents=True, exist_ok=False)
        transaction_manifest = {
            "transaction_id": transaction_id,
            "status": "prepared",
            "created_at": utc_now(),
            "before": {
                "registry_sha256": sha256_file(registry_path),
                "dashboard_sha256": sha256_file(dashboard_path),
                "state_sha256": sha256_file(state_path),
            },
        }
        write_json_atomic(transaction_dir / "transaction.json", transaction_manifest)
        backup_dir = transaction_dir / "backup"
        backup_dir.mkdir()
        shutil.copy2(registry_path, backup_dir / "experiment_registry.json")
        shutil.copy2(dashboard_path, backup_dir / "experiment_dashboard.md")
        shutil.copy2(state_path, backup_dir / "current_state.json")
        if (root / "CURRENT_STATE.md").exists():
            shutil.copy2(root / "CURRENT_STATE.md", backup_dir / "CURRENT_STATE.md")
        write_json_atomic(transaction_dir / "experiment_registry.json", registry)
        write_text_atomic(transaction_dir / "experiment_dashboard.md", dashboard_text)

        staged_hashes = compute_source_hashes(root)
        staged_hashes["experiment_registry_sha256"] = sha256_file(transaction_dir / "experiment_registry.json")
        staged_hashes["experiment_dashboard_sha256"] = sha256_file(transaction_dir / "experiment_dashboard.md")
        state["source_hashes"] = staged_hashes
        write_json_atomic(transaction_dir / "current_state.json", state)
        write_text_atomic(transaction_dir / "CURRENT_STATE.md", render_current_state(root, state, registry))

        # Validate staged payloads before replacing any visible file.
        read_json(transaction_dir / "experiment_registry.json")
        read_json(transaction_dir / "current_state.json")
        transaction_manifest["status"] = "committing"
        write_json_atomic(transaction_dir / "transaction.json", transaction_manifest)
        try:
            os.replace(transaction_dir / "experiment_registry.json", registry_path)
            os.replace(transaction_dir / "experiment_dashboard.md", dashboard_path)
            os.replace(transaction_dir / "current_state.json", state_path)
            os.replace(transaction_dir / "CURRENT_STATE.md", root / "CURRENT_STATE.md")
        except Exception:
            transaction_manifest["status"] = "interrupted"
            write_json_atomic(transaction_dir / "transaction.json", transaction_manifest)
            shutil.copy2(backup_dir / "experiment_registry.json", registry_path)
            shutil.copy2(backup_dir / "experiment_dashboard.md", dashboard_path)
            shutil.copy2(backup_dir / "current_state.json", state_path)
            if (backup_dir / "CURRENT_STATE.md").exists():
                shutil.copy2(backup_dir / "CURRENT_STATE.md", root / "CURRENT_STATE.md")
            raise
        # Refresh local-only views and README without incrementing the committed revision.
        final_state = read_json(state_path)
        final_registry = read_json(registry_path)
        if (root / ".claude").exists():
            write_text_atomic(root / ".claude" / "next-steps.md", _render_local_next_steps(final_state, final_registry))
            write_text_atomic(root / ".claude" / "session-brief.md", _render_session_brief(final_state))
        readme_path = root / "README.md"
        if readme_path.exists():
            write_text_atomic(readme_path, replace_readme_state_block(readme_path.read_text(encoding="utf-8"), _readme_state_block(final_state, final_registry)))
        shutil.rmtree(transaction_dir)
        if transaction_root.exists() and not any(transaction_root.iterdir()):
            transaction_root.rmdir()
    return {"status": "imported", "result_id": manifest["result_id"], "evidence_status": target["evidence_status"]}


def safe_job_parameters(parameters: Mapping[str, Any]) -> List[str]:
    argv: List[str] = []
    key_pattern = re.compile(r"^[a-z][a-z0-9_-]*$")
    value_pattern = re.compile(r"^[A-Za-z0-9_./:\\,+-]+$")
    for key in sorted(parameters):
        if not key_pattern.fullmatch(key):
            raise ValueError(f"unsafe job parameter name: {key}")
        value = parameters[key]
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        values = value if isinstance(value, list) else [value]
        argv.append(flag)
        for item in values:
            text = str(item)
            if not value_pattern.fullmatch(text):
                raise ValueError(f"unsafe job parameter value for {key}: {text}")
            argv.append(text)
    return argv


def validate_job_semantics(
    phase: str,
    command_id: str,
    parameters: Mapping[str, Any],
    *,
    experiment: Optional[Mapping[str, Any]] = None,
    path_ids: Sequence[str] = (),
) -> None:
    if command_id == "state_preflight":
        if phase != "preflight":
            raise ValueError("state_preflight command is only valid for phase=preflight")
        if parameters:
            raise ValueError("state_preflight does not accept training parameters")
        return
    if command_id != "standard_training":
        raise ValueError("command_id is not allowlisted")
    if phase not in {"smoke", "formal"}:
        raise ValueError("standard_training requires phase=smoke or phase=formal")
    if phase == "smoke":
        raw_epochs = parameters.get("num_epochs")
        if raw_epochs is None:
            raise ValueError("smoke training requires explicit num_epochs")
        try:
            epochs = int(raw_epochs)
        except (TypeError, ValueError) as exc:
            raise ValueError("smoke num_epochs must be an integer") from exc
        if epochs < 1 or epochs > SMOKE_MAX_EPOCHS:
            raise ValueError(f"smoke num_epochs must be between 1 and {SMOKE_MAX_EPOCHS}")
    elif phase == "formal":
        try:
            epochs = int(parameters.get("num_epochs"))
        except (TypeError, ValueError) as exc:
            raise ValueError("formal training requires a positive integer num_epochs") from exc
        if epochs < 1:
            raise ValueError("formal training requires a positive integer num_epochs")

    if experiment and str(experiment.get("script", "")).replace("\\", "/").endswith("train_mpp_uni2h_mlp.py"):
        provided_ids = set(path_ids)
        missing_ids = sorted(MPP_TRAINING_PATH_IDS - provided_ids)
        if missing_ids:
            raise ValueError(f"MPP training job is missing required path ids: {missing_ids}")
        path_overrides = sorted(MPP_TRAINING_PATH_PARAMETERS & set(parameters))
        if path_overrides:
            raise ValueError(f"MPP training paths are registry-bound and cannot be overridden: {path_overrides}")
        unknown_parameters = sorted(set(parameters) - MPP_TRAINING_ALLOWED_PARAMETERS)
        if unknown_parameters:
            raise ValueError(f"MPP training parameters are not allowlisted: {unknown_parameters}")
        if parameters.get("val_strategy") != "manifest":
            raise ValueError("MPP standard training requires val_strategy=manifest")
        try:
            mpp_id = int(parameters.get("train_mpp_id"))
        except (TypeError, ValueError) as exc:
            raise ValueError("MPP training requires train_mpp_id in 1..5") from exc
        if mpp_id not in range(1, 6):
            raise ValueError("MPP training requires train_mpp_id in 1..5")
        if phase == "formal" and parameters.get("allow_missing"):
            raise ValueError("formal MPP training cannot use allow_missing")


def create_job_manifest(
    root: Path,
    *,
    job_id: str,
    experiment_id: str,
    phase: str,
    command_id: str,
    path_ids: Sequence[str],
    parameters: Mapping[str, Any],
    approval_path: Optional[Path] = None,
    data_manifest_id: Optional[str] = None,
    path_index_version: Optional[str] = None,
) -> Dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id):
        raise ValueError("job_id contains unsafe characters")
    if phase not in {"preflight", "smoke", "formal"}:
        raise ValueError("invalid job phase")
    if command_id not in {"state_preflight", "standard_training"}:
        raise ValueError("command_id is not allowlisted")
    registry = read_json(root / "experiments" / "experiment_registry.json")
    experiment = next((item for item in registry.get("experiments", []) if item.get("id") == experiment_id), None)
    if experiment is None:
        raise ValueError(f"experiment is not registered: {experiment_id}")
    paths = load_server_paths(root).get("paths", {})
    unknown = sorted(set(path_ids) - set(paths))
    if unknown:
        raise ValueError(f"unknown path ids: {unknown}")
    safe_job_parameters(parameters)
    validate_job_semantics(phase, command_id, parameters, experiment=experiment, path_ids=path_ids)
    approval = None
    if phase == "formal":
        if approval_path is None or not approval_path.exists():
            raise ValueError("formal training requires an approval file")
        approval = read_json(approval_path)
        if not approval.get("approved") or approval.get("source") != "explicit_user_instruction":
            raise ValueError("formal training approval is not explicit")
        if approval.get("job_id") != job_id or approval.get("source_commit") != git_head(root):
            raise ValueError("formal approval does not bind the current job and source commit")
    if git_head(root) == "unknown":
        raise ValueError("job dispatch requires a Git checkout with a committed source")
    if git_tracked_changes(root):
        raise ValueError("job dispatch requires all tracked changes to be committed; untracked training artifacts may remain")
    required_tracked = [
        "deploy/pfmval_ops.py",
        "scripts/pfmval_state.py",
        "project_state/current_state.json",
        "experiments/experiment_registry.json",
        "configs/server_paths.yaml",
    ]
    missing_tracked = [path for path in required_tracked if not git_path_is_tracked(root, path)]
    if missing_tracked:
        raise ValueError(f"job dispatch source commit does not contain required files: {missing_tracked}")
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "experiment_id": experiment_id,
        "source_commit": git_head(root),
        "state_revision": read_json(root / "project_state" / "current_state.json")["state_revision"],
        "phase": phase,
        "command_id": command_id,
        "path_ids": list(path_ids),
        "parameters": dict(parameters),
        "data_manifest_id": data_manifest_id,
        "path_index_version": path_index_version,
        "created_at": utc_now(),
        "dispatch_branch": f"automation/local/{job_id}",
        "result_branch": f"automation/server/{job_id}",
        "formal_training_approval": approval,
        "artifact_policy": {
            "max_file_bytes": MAX_RESULT_FILE_BYTES,
            "max_total_bytes": MAX_RESULT_TOTAL_BYTES,
            "large_artifacts": "server_path_size_sha256_only",
        },
    }


def validate_job_manifest(root: Path, manifest: Mapping[str, Any], *, require_head: bool = True) -> None:
    validate_against_schema(
        manifest,
        root / "project_state" / "schemas" / "server_job.schema.json",
        "server job",
    )
    required = {"schema_version", "job_id", "experiment_id", "source_commit", "state_revision", "phase", "command_id", "path_ids", "parameters", "created_at", "artifact_policy"}
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"job manifest missing fields: {', '.join(missing)}")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported job manifest version")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(manifest.get("job_id", ""))):
        raise ValueError("job manifest contains an unsafe job_id")
    if not git_commit_exists(root, str(manifest["source_commit"])):
        raise ValueError(f"job source commit is unavailable locally: {manifest['source_commit']}")
    if require_head and manifest["source_commit"] != git_head(root):
        raise ValueError(f"job source commit {manifest['source_commit']} does not match HEAD {git_head(root)}")
    if manifest["phase"] == "formal":
        approval = manifest.get("formal_training_approval") or {}
        if not approval.get("approved") or approval.get("job_id") != manifest["job_id"] or approval.get("source_commit") != manifest["source_commit"]:
            raise ValueError("formal job has no valid bound approval")
    safe_job_parameters(manifest.get("parameters", {}))
    paths = load_server_paths(root).get("paths", {})
    unknown = sorted(set(manifest.get("path_ids", [])) - set(paths))
    if unknown:
        raise ValueError(f"job references unknown paths: {unknown}")
    registry = read_json(root / "experiments" / "experiment_registry.json")
    experiment = next((item for item in registry.get("experiments", []) if item.get("id") == manifest["experiment_id"]), None)
    if experiment is None:
        raise ValueError("job references unknown experiment")
    validate_job_semantics(
        str(manifest.get("phase")),
        str(manifest.get("command_id")),
        manifest.get("parameters", {}),
        experiment=experiment,
        path_ids=manifest.get("path_ids", []),
    )


def build_result_envelope(
    *,
    job: Mapping[str, Any],
    status: str,
    output_dir: Path,
    artifact_paths: Sequence[Path],
    metrics: Mapping[str, Any],
    large_artifact_paths: Sequence[Path] = (),
) -> Dict[str, Any]:
    if status not in {"success", "failed", "incomplete"}:
        raise ValueError("invalid result status")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"result output directory is not empty: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.building-", dir=output_dir.parent))
    try:
        artifacts: List[Dict[str, Any]] = []
        total = 0
        seen_names: set[str] = set()
        for source in artifact_paths:
            if source.name not in ALLOWED_RESULT_FILES:
                raise ValueError(f"artifact is not allowlisted: {source.name}")
            if source.name in seen_names:
                raise ValueError(f"duplicate artifact basename: {source.name}")
            seen_names.add(source.name)
            size = source.stat().st_size
            if size > MAX_RESULT_FILE_BYTES or total + size > MAX_RESULT_TOTAL_BYTES:
                raise ValueError(f"artifact budget exceeded by {source}")
            destination = temporary_dir / source.name
            shutil.copy2(source, destination)
            artifacts.append({"path": source.name, "kind": source.suffix.lstrip(".") or "text", "size_bytes": size, "sha256": sha256_file(destination)})
            total += size
        result_id = f"{job['job_id']}-result-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
        large_artifacts: List[Dict[str, Any]] = []
        for source in large_artifact_paths:
            if not source.exists() or not source.is_file():
                raise ValueError(f"large artifact is missing: {source}")
            large_artifacts.append({
                "server_path": str(source),
                "size_bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            })
        envelope = {
            "schema_version": SCHEMA_VERSION,
            "result_id": result_id,
            "job_id": job["job_id"],
            "experiment_id": job["experiment_id"],
            "source_commit": job["source_commit"],
            "phase": job["phase"],
            "status": status,
            "created_at": utc_now(),
            "formal_training_approved": bool((job.get("formal_training_approval") or {}).get("approved")),
            "data_manifest_id": job.get("data_manifest_id"),
            "path_index_version": job.get("path_index_version") or SCHEMA_VERSION,
            "artifacts": artifacts,
            "metrics": dict(metrics),
            "large_artifacts": large_artifacts,
            "supersedes_results": [],
        }
        write_json_atomic(temporary_dir / "result.json", envelope)
        if output_dir.exists():
            output_dir.rmdir()
        os.replace(temporary_dir, output_dir)
        return envelope
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
