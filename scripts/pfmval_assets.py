"""Read-only asset shadow, policy compilation, and close-preview helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


SENSITIVE_PARTS = {
    ".env",
    ".git",
    ".ssh",
    "token",
    "tokens",
    "credential",
    "credentials",
    "cache",
    "__pycache__",
}
PROTECTED_ROOTS = ("histogene", "egnv1", "egnv2")
FIXED_POLICY_RULES = (
    ("fixed-histogene", "histogene", "approval_required", "protected"),
    ("fixed-egnv1", "egnv1", "approval_required", "protected"),
    ("fixed-egnv2", "egnv2", "approval_required", "protected"),
    (
        "fixed-mpp-standard-splits",
        "mpp_standard_splits",
        "immutable",
        "active",
    ),
    (
        "fixed-mpp-repair-registry",
        "project_state/mpp_repair_registry.json",
        "append_only",
        "active",
    ),
)
TERMINAL_ATTEMPT_EVENTS = {
    "PREFLIGHT_FAILED",
    "ATTEMPT_FAILED",
    "IMPORTED",
    "REJECTED",
}


def _normalized(value: Path) -> str:
    return os.path.normcase(str(value.resolve(strict=False))).replace("\\", "/")


def _safe_for_inventory(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    lowered = {part.lower() for part in relative.parts}
    return not any(
        part in SENSITIVE_PARTS
        or "token" in part
        or "credential" in part
        for part in lowered
    )


def _metadata(path: Path, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    return {
        "path": relative,
        "kind": "directory" if path.is_dir() else "file",
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "hash_policy": "metadata_only",
    }


def build_shadow_inventory(root: Path) -> dict[str, Any]:
    """Inventory bounded metadata and lineage candidates without file hashes."""
    root = root.resolve(strict=False)
    active_mpp_trainers = sorted(root.glob("train_mpp_*.py"))
    shared_histogene_utils = False
    shared_neutral_metrics = False
    for trainer in active_mpp_trainers:
        if trainer.stat().st_size > 1024 * 1024:
            continue
        text = trainer.read_text(encoding="utf-8", errors="replace")
        if "from histogene.utils import compute_metrics" in text:
            shared_histogene_utils = True
        if "from pfmval_core.metrics import compute_metrics" in text:
            shared_neutral_metrics = True

    paths: set[Path] = set()
    for name in PROTECTED_ROOTS:
        protected = root / name
        if protected.exists():
            paths.add(protected)
    if shared_histogene_utils and (root / "histogene" / "utils.py").is_file():
        paths.add(root / "histogene" / "utils.py")
    if shared_neutral_metrics and (root / "pfmval_core" / "metrics.py").is_file():
        paths.add(root / "pfmval_core" / "metrics.py")
    if (root / "histogene" / "utils.py").is_file():
        paths.add(root / "histogene" / "utils.py")
    paths.update(active_mpp_trainers)

    # Metadata-only inventory of shallow ignored/local roots.  It never opens
    # their contents and remains bounded to two levels below the repository.
    for first in root.iterdir():
        if not _safe_for_inventory(first, root):
            continue
        if first.is_file():
            if first.suffix.lower() in {".ckpt", ".pt", ".pth", ".bin"}:
                paths.add(first)
            continue
        if first.name in {"project_state", "experiments", "tests", ".agents"}:
            continue
        try:
            for second in first.iterdir():
                if (
                    second.is_file()
                    and _safe_for_inventory(second, root)
                    and second.suffix.lower()
                    in {".ckpt", ".pt", ".pth", ".bin"}
                ):
                    paths.add(second)
        except (PermissionError, OSError):
            continue

    candidates = []
    for path in sorted(paths, key=lambda item: item.as_posix().lower()):
        if not _safe_for_inventory(path, root):
            continue
        item = _metadata(path, root)
        relative = item["path"]
        if relative == "histogene/utils.py" and shared_histogene_utils:
            item.update(
                {
                    "era": "shared_dependency",
                    "custody": "project",
                    "asset_class": "python_module",
                    "lifecycle": "protected",
                    "mutability": "approval_required",
                    "evidence_role": "runtime_dependency",
                    "lineage": [
                        trainer.relative_to(root).as_posix()
                        for trainer in active_mpp_trainers
                        if "from histogene.utils import compute_metrics"
                        in trainer.read_text(encoding="utf-8", errors="replace")
                    ],
                }
            )
        elif relative == "pfmval_core/metrics.py" and shared_neutral_metrics:
            item.update(
                {
                    "era": "shared_dependency",
                    "custody": "project",
                    "asset_class": "python_module",
                    "lifecycle": "active",
                    "mutability": "workflow_managed",
                    "evidence_role": "runtime_dependency",
                    "lineage": [
                        trainer.relative_to(root).as_posix()
                        for trainer in active_mpp_trainers
                        if "from pfmval_core.metrics import compute_metrics"
                        in trainer.read_text(encoding="utf-8", errors="replace")
                    ],
                }
            )
        elif relative == "histogene/utils.py":
            item.update(
                {
                    "era": "legacy_mixed",
                    "custody": "project",
                    "asset_class": "compatibility_module",
                    "lifecycle": "protected",
                    "mutability": "approval_required",
                    "evidence_role": "compatibility",
                    "replacement": "pfmval_core/metrics.py",
                }
            )
        elif relative in PROTECTED_ROOTS:
            item.update(
                {
                    "era": "legacy_mixed",
                    "custody": "project",
                    "asset_class": "code_root",
                    "lifecycle": "protected",
                    "mutability": "approval_required",
                    "evidence_role": "mixed",
                }
            )
        elif path in active_mpp_trainers:
            item.update(
                {
                    "era": "mpp_active",
                    "custody": "project",
                    "asset_class": "training_entrypoint",
                    "lifecycle": "active",
                    "mutability": "workflow_managed",
                    "evidence_role": "runtime_code",
                }
            )
        else:
            item.update(
                {
                    "era": "uncertain",
                    "custody": "project",
                    "asset_class": "large_local_artifact",
                    "lifecycle": "protected",
                    "mutability": "approval_required",
                    "evidence_role": "unknown",
                }
            )
        candidates.append(item)
    return {
        "schema_version": "1.0",
        "mode": "shadow",
        "content_hashes_computed": 0,
        "candidates": candidates,
        "conflicts": [],
        "side_effects": "none",
    }


def compile_asset_policy(root: Path) -> dict[str, Any]:
    registry_path = root / "project_state" / "asset_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    rules = []
    for asset in registry.get("assets", []):
        if asset.get("mutability") not in {
            "immutable",
            "append_only",
            "workflow_managed",
            "approval_required",
        }:
            continue
        rules.append(
            {
                "asset_id": asset.get("asset_id"),
                "path": str(asset.get("path", "")).replace("\\", "/").rstrip("/"),
                "mutability": asset.get("mutability"),
                "lifecycle": asset.get("lifecycle"),
            }
        )
    registered_paths = {str(rule["path"]).lower() for rule in rules}
    for asset_id, path, mutability, lifecycle in FIXED_POLICY_RULES:
        if path.lower() in registered_paths:
            continue
        rules.append(
            {
                "asset_id": asset_id,
                "path": path,
                "mutability": mutability,
                "lifecycle": lifecycle,
                "source": "fixed_boundary_shadow_fallback",
            }
        )
    return {
        "schema_version": "1.0",
        "mode": "shadow",
        "rules": sorted(rules, key=lambda item: item["path"].lower()),
    }


def evaluate_asset_writes(
    root: Path,
    policy: Mapping[str, Any],
    paths: Iterable[str],
    *,
    mode: str = "shadow",
) -> dict[str, Any]:
    if mode not in {"shadow", "enforce"}:
        raise ValueError("asset policy mode must be shadow or enforce")
    root_resolved = root.resolve(strict=False)
    root_normalized = _normalized(root_resolved)
    would_block = []
    allowed = []
    for raw in paths:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = root_resolved / candidate
        candidate_normalized = _normalized(candidate)
        if not (
            candidate_normalized == root_normalized
            or candidate_normalized.startswith(root_normalized + "/")
        ):
            would_block.append(
                {
                    "path": raw,
                    "reason": "path_escape",
                    "asset_id": None,
                }
            )
            continue
        relative = candidate_normalized[len(root_normalized) :].lstrip("/")
        matched = next(
            (
                rule
                for rule in policy.get("rules", [])
                if relative == str(rule["path"]).lower()
                or relative.startswith(str(rule["path"]).lower() + "/")
            ),
            None,
        )
        if matched:
            would_block.append(
                {
                    "path": raw,
                    "reason": str(matched["mutability"]),
                    "asset_id": matched.get("asset_id"),
                }
            )
        else:
            allowed.append(raw)
    return {
        "mode": mode,
        "blocked": mode == "enforce" and bool(would_block),
        "would_block": would_block,
        "allowed": allowed,
        "registry_writes": 0,
        "file_deletes": 0,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_close_preview(
    root: Path,
    *,
    workspace_id: str,
    observed_worktrees: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a side-effect-free close decision for one registered workspace."""
    registry = json.loads(
        (root / "project_state" / "workspace_registry.json").read_text(
            encoding="utf-8"
        )
    )
    workspace = next(
        (
            item
            for item in registry.get("workspaces", [])
            if item.get("workspace_id") == workspace_id
        ),
        None,
    )
    if workspace is None:
        raise ValueError(f"unknown workspace_id: {workspace_id}")
    local = workspace.get("hosts", {}).get("local")
    if not local:
        raise ValueError("workspace has no local host binding")
    physical_path = (root / str(local["relative_path"])).resolve(strict=False)
    if not physical_path.is_relative_to(root.resolve(strict=False)):
        raise ValueError("workspace close path escapes repository root")
    observed = next(
        (
            dict(item)
            for item in observed_worktrees
            if _normalized(Path(str(item["path"]))) == _normalized(physical_path)
        ),
        None,
    )
    blockers = []
    if workspace.get("lease"):
        blockers.append("active_lease")
    if observed is None:
        blockers.append("worktree_not_observed")
    else:
        if observed.get("dirty"):
            blockers.append("dirty_worktree")
        if int(observed.get("unpushed_commits", 0)) > 0:
            blockers.append("unpushed_commits")
        if local.get("branch") and observed.get("branch") != local.get("branch"):
            blockers.append("branch_mismatch")
        if (
            local.get("current_source_commit")
            and observed.get("head") != local.get("current_source_commit")
        ):
            blockers.append("head_mismatch")

    events = [
        event
        for event in _read_jsonl(
            root / "project_state" / "attempt_events.jsonl"
        )
        if event.get("workspace_id") == workspace_id
    ]
    attempts = sorted(
        {
            str(event.get("attempt_id"))
            for event in events
            if event.get("attempt_id")
        }
    )
    for attempt_id in attempts:
        attempt_events = {
            str(event.get("event_type"))
            for event in events
            if event.get("attempt_id") == attempt_id
        }
        if not (attempt_events & TERMINAL_ATTEMPT_EVENTS):
            blockers.append("nonterminal_attempt")
        if (
            "RESULT_RETURNED" in attempt_events
            and "IMPORTED" not in attempt_events
            and "REJECTED" not in attempt_events
        ):
            blockers.append("unimported_result")
        if (
            "EXPERIMENT_STARTED" in attempt_events
            and "RESULT_RETURNED" not in attempt_events
            and "ATTEMPT_FAILED" not in attempt_events
        ):
            blockers.append("unreturned_result")

    asset_registry = json.loads(
        (root / "project_state" / "asset_registry.json").read_text(
            encoding="utf-8"
        )
    )
    retained_assets = [
        item.get("asset_id")
        for item in asset_registry.get("assets", [])
        if item.get("owner_experiment_id") == workspace.get("experiment_id")
        and item.get("retention_policy")
    ]
    unique_blockers = sorted(set(blockers))
    return {
        "schema_version": "1.0",
        "operation": "close_preview",
        "workspace_id": workspace_id,
        "experiment_id": workspace.get("experiment_id"),
        "resolved_path": str(physical_path),
        "branch": local.get("branch"),
        "head": observed.get("head") if observed else None,
        "attempt_ids": attempts,
        "retained_asset_ids": sorted(
            str(item) for item in retained_assets if item
        ),
        "blockers": unique_blockers,
        "status": "close_ready" if not unique_blockers else "blocked",
        "side_effects": "none",
        "execute_authorized": False,
    }
