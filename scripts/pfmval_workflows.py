"""Offline, explicitly scoped workflow catalog for PFMval."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


_MARKER = re.compile(r"pfmval-workflow:\s*(\{.*\})\s*(?:-->)?\s*$")
_LIFECYCLES = {"candidate", "reviewed", "active", "deprecated"}
_HIGH_RISK = {"high", "critical"}


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _catalog_path(root: Path) -> Path:
    return root / "project_state" / "workflow_catalog.json"


def _empty_catalog() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "updated_at": None,
        "entries": [],
        "scans": [],
    }


def _read_catalog(root: Path) -> dict[str, Any]:
    path = _catalog_path(root)
    if not path.exists():
        return _empty_catalog()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("workflow catalog must be an object")
    value.setdefault("entries", [])
    value.setdefault("scans", [])
    return value


def _write_catalog(root: Path, catalog: Mapping[str, Any]) -> None:
    path = _catalog_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                dict(catalog),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _relative_selected_path(root: Path, raw: str) -> tuple[str, Path]:
    relative = Path(raw)
    if relative.is_absolute():
        raise ValueError(f"workflow scope path must be relative: {raw}")
    resolved_root = root.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"workflow scope escapes project root: {raw}")
    return relative.as_posix(), resolved


def _extract_descriptor(path: str, text: str) -> dict[str, Any] | None:
    for line in text.splitlines():
        match = _MARKER.search(line.strip())
        if not match:
            continue
        value = json.loads(match.group(1))
        if not isinstance(value, dict):
            raise ValueError(f"workflow marker is not an object: {path}")
        required = {"purpose", "trigger", "inputs", "outputs"}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(
                f"workflow marker missing {', '.join(missing)}: {path}"
            )
        return value
    return None


def _identity(descriptor: Mapping[str, Any]) -> str:
    return _canonical(
        {
            "purpose": descriptor["purpose"],
            "trigger": descriptor["trigger"],
            "inputs": descriptor["inputs"],
            "outputs": descriptor["outputs"],
        }
    )


def discover_workflows(
    root: Path,
    *,
    selected_paths: Sequence[str],
    tracked_paths: Iterable[str],
    reader: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    """Inspect only the explicitly selected, already tracked local paths."""

    if not selected_paths:
        return {
            "status": "not_requested",
            "reads": 0,
            "writes": 0,
            "candidates": [],
        }
    tracked = {Path(item).as_posix() for item in tracked_paths}
    read_text = reader or (lambda path: path.read_text(encoding="utf-8"))
    normalized: list[str] = []
    content_hashes: dict[str, str] = {}
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for raw in selected_paths:
        relative, path = _relative_selected_path(root, raw)
        if relative not in tracked:
            raise ValueError(f"workflow scope is not tracked: {relative}")
        text = read_text(path)
        normalized.append(relative)
        content_hashes[relative] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        descriptor = _extract_descriptor(relative, text)
        if descriptor is not None:
            grouped.setdefault(_identity(descriptor), []).append(
                (relative, descriptor)
            )

    candidates: list[dict[str, Any]] = []
    for identity, matches in sorted(grouped.items()):
        if len(matches) < 2:
            continue
        first = deepcopy(matches[0][1])
        workflow_id = "WF-CAND-" + hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:12].upper()
        candidates.append(
            {
                "workflow_id": workflow_id,
                "purpose": first["purpose"],
                "trigger": first["trigger"],
                "inputs": first["inputs"],
                "outputs": first["outputs"],
                "implementation_locations": sorted(item[0] for item in matches),
                "risk": first.get("risk", "unknown"),
                "approval_requirement": first.get("approval_requirement", ""),
                "version": first.get("version", "0.1.0"),
                "lifecycle": "candidate",
                "replacement_workflow_ids": [],
                "supersedes": [],
                "standardized": False,
                "stable_version": bool(first.get("stable_version", False)),
                "schema_refs": list(first.get("schema_refs", [])),
                "test_refs": list(first.get("test_refs", [])),
                "example_refs": list(first.get("example_refs", [])),
                "last_verified_at": None,
            }
        )
    scan_material = {
        "selected_paths": normalized,
        "content_hashes": content_hashes,
        "candidates": candidates,
    }
    scan_id = "SCAN-" + hashlib.sha256(
        _canonical(scan_material).encode("utf-8")
    ).hexdigest()[:16].upper()
    return {
        "status": (
            "candidate_review_required" if candidates else "no_duplicates_found"
        ),
        "scan_id": scan_id,
        "selected_paths": normalized,
        "content_hashes": content_hashes,
        "reads": len(normalized),
        "writes": 0,
        "candidates": candidates,
    }


def apply_workflow_discovery(
    root: Path,
    discovery: Mapping[str, Any],
) -> dict[str, Any]:
    if discovery.get("status") == "not_requested":
        return {"status": "not_requested", "writes": 0, "candidate_ids": []}
    scan_id = str(discovery.get("scan_id", ""))
    if not scan_id:
        raise ValueError("workflow discovery has no scan_id")
    catalog = _read_catalog(root)
    if any(item.get("scan_id") == scan_id for item in catalog["scans"]):
        return {
            "status": "noop",
            "writes": 0,
            "candidate_ids": [
                item["workflow_id"] for item in discovery.get("candidates", [])
            ],
        }
    known = {item["workflow_id"] for item in catalog["entries"]}
    for candidate in discovery.get("candidates", []):
        if candidate["workflow_id"] not in known:
            catalog["entries"].append(deepcopy(candidate))
            known.add(candidate["workflow_id"])
    catalog["scans"].append(
        {
            "scan_id": scan_id,
            "selected_paths": list(discovery.get("selected_paths", [])),
            "content_hashes": dict(discovery.get("content_hashes", {})),
        }
    )
    catalog["updated_at"] = _now()
    catalog["entries"].sort(key=lambda item: item["workflow_id"])
    _write_catalog(root, catalog)
    return {
        "status": "recorded",
        "writes": 1,
        "candidate_ids": [
            item["workflow_id"] for item in discovery.get("candidates", [])
        ],
    }


def register_workflow_manifest(
    root: Path,
    *,
    manifest_path: str,
    authorization_ref: str,
    tracked_paths: Iterable[str],
    active_authorization_refs: Iterable[str],
) -> dict[str, Any]:
    """Register explicitly approved manifest entries as candidates only."""

    relative, resolved = _relative_selected_path(root, manifest_path)
    tracked = {Path(item).as_posix() for item in tracked_paths}
    if relative not in tracked:
        raise ValueError(f"workflow manifest is not tracked: {relative}")
    active_refs = {str(item) for item in active_authorization_refs}
    if authorization_ref not in active_refs:
        raise ValueError(
            f"workflow authorization is not active: {authorization_ref}"
        )
    manifest_bytes = resolved.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("workflow manifest must be an object")
    if manifest.get("authorization_ref") != authorization_ref:
        raise ValueError("workflow manifest authorization_ref does not match")
    workflows = manifest.get("workflows")
    if not isinstance(workflows, list) or not workflows:
        raise ValueError("workflow manifest workflows must be a non-empty array")

    required = {
        "workflow_id",
        "purpose",
        "trigger",
        "inputs",
        "outputs",
        "implementation_locations",
        "risk",
        "approval_requirement",
        "version",
        "user_entry",
        "technical_routes",
    }
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    candidates: list[dict[str, Any]] = []
    manifest_ids: set[str] = set()
    for raw in workflows:
        if not isinstance(raw, dict):
            raise ValueError("workflow manifest entry must be an object")
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(
                f"workflow manifest entry missing {', '.join(missing)}"
            )
        workflow_id = str(raw["workflow_id"])
        if workflow_id in manifest_ids:
            raise ValueError(f"duplicate workflow_id in manifest: {workflow_id}")
        manifest_ids.add(workflow_id)
        candidate = deepcopy(raw)
        candidate.update(
            {
                "workflow_id": workflow_id,
                "lifecycle": "candidate",
                "replacement_workflow_ids": [],
                "supersedes": list(raw.get("supersedes", [])),
                "standardized": False,
                "stable_version": bool(raw.get("stable_version", False)),
                "schema_refs": list(raw.get("schema_refs", [])),
                "test_refs": list(raw.get("test_refs", [])),
                "example_refs": list(raw.get("example_refs", [])),
                "last_verified_at": None,
                "authorization_ref": authorization_ref,
                "registration_manifest": relative,
                "registration_manifest_sha256": manifest_sha256,
            }
        )
        registration_material = {
            key: value
            for key, value in candidate.items()
            if key
            not in {
                "lifecycle",
                "standardized",
                "last_verified_at",
                "registration_manifest",
                "registration_manifest_sha256",
                "registration_digest",
            }
        }
        candidate["registration_digest"] = hashlib.sha256(
            _canonical(registration_material).encode("utf-8")
        ).hexdigest()
        candidates.append(candidate)

    catalog = _read_catalog(root)
    existing = {
        str(item.get("workflow_id")): item for item in catalog["entries"]
    }
    new_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        current = existing.get(candidate["workflow_id"])
        if current is None:
            new_candidates.append(candidate)
            continue
        if current.get("registration_digest") != candidate["registration_digest"]:
            raise ValueError(
                "workflow_id is already bound to different content: "
                f"{candidate['workflow_id']}"
            )
    candidate_ids = [item["workflow_id"] for item in candidates]
    if not new_candidates:
        return {
            "status": "noop",
            "writes": 0,
            "candidate_ids": candidate_ids,
        }
    catalog["entries"].extend(new_candidates)
    _commit_catalog(root, catalog)
    return {
        "status": "registered",
        "writes": 1,
        "candidate_ids": candidate_ids,
    }


def list_workflows(root: Path) -> list[dict[str, Any]]:
    return deepcopy(_read_catalog(root)["entries"])


def _entry(catalog: dict[str, Any], workflow_id: str) -> dict[str, Any]:
    for item in catalog["entries"]:
        if item.get("workflow_id") == workflow_id:
            return item
    raise ValueError(f"unknown workflow_id: {workflow_id}")


def _commit_catalog(root: Path, catalog: dict[str, Any]) -> None:
    catalog["updated_at"] = _now()
    catalog["entries"].sort(key=lambda item: item["workflow_id"])
    _write_catalog(root, catalog)


def review_workflow(root: Path, workflow_id: str) -> dict[str, Any]:
    catalog = _read_catalog(root)
    entry = _entry(catalog, workflow_id)
    if entry["lifecycle"] != "candidate":
        raise ValueError("only candidate workflows can be reviewed")
    entry["lifecycle"] = "reviewed"
    _commit_catalog(root, catalog)
    return deepcopy(entry)


def _assert_activation_gate(
    entry: Mapping[str, Any],
    *,
    standardized: bool,
) -> None:
    if entry.get("lifecycle") != "reviewed":
        raise ValueError("workflow must be reviewed before activation")
    if entry.get("risk") in _HIGH_RISK and (
        not entry.get("approval_requirement")
        or not entry.get("schema_refs")
        or not entry.get("test_refs")
    ):
        raise ValueError(
            "high-risk workflow requires approval, schema, and tests"
        )
    if standardized and (
        not entry.get("stable_version")
        or not entry.get("schema_refs")
        or not entry.get("test_refs")
        or not entry.get("example_refs")
    ):
        raise ValueError(
            "standardized workflow requires stable version, schema, tests, and examples"
        )


def activate_workflow(
    root: Path,
    workflow_id: str,
    *,
    standardized: bool = False,
) -> dict[str, Any]:
    catalog = _read_catalog(root)
    entry = _entry(catalog, workflow_id)
    _assert_activation_gate(entry, standardized=standardized)
    entry["lifecycle"] = "active"
    entry["standardized"] = standardized
    entry["last_verified_at"] = _now()
    _commit_catalog(root, catalog)
    return deepcopy(entry)


def revise_workflow(
    root: Path,
    workflow_id: str,
    *,
    new_workflow_id: str,
    version: str,
) -> dict[str, Any]:
    catalog = _read_catalog(root)
    original = _entry(catalog, workflow_id)
    if any(item["workflow_id"] == new_workflow_id for item in catalog["entries"]):
        raise ValueError(f"workflow_id already exists: {new_workflow_id}")
    revised = deepcopy(original)
    revised.update(
        {
            "workflow_id": new_workflow_id,
            "version": version,
            "lifecycle": "active",
            "standardized": False,
            "supersedes": [workflow_id],
            "replacement_workflow_ids": [],
            "last_verified_at": None,
        }
    )
    original["lifecycle"] = "deprecated"
    original["replacement_workflow_ids"] = [new_workflow_id]
    catalog["entries"].append(revised)
    _commit_catalog(root, catalog)
    return deepcopy(revised)


def merge_workflows(
    root: Path,
    workflow_ids: Sequence[str],
    *,
    new_workflow_id: str,
    purpose: str,
    version: str,
) -> dict[str, Any]:
    if len(set(workflow_ids)) < 2:
        raise ValueError("workflow merge requires at least two distinct IDs")
    catalog = _read_catalog(root)
    if any(item["workflow_id"] == new_workflow_id for item in catalog["entries"]):
        raise ValueError(f"workflow_id already exists: {new_workflow_id}")
    sources = [_entry(catalog, item) for item in workflow_ids]
    merged = deepcopy(sources[0])
    merged.update(
        {
            "workflow_id": new_workflow_id,
            "purpose": purpose,
            "version": version,
            "lifecycle": "active",
            "standardized": False,
            "supersedes": list(workflow_ids),
            "replacement_workflow_ids": [],
            "implementation_locations": sorted(
                {
                    location
                    for source in sources
                    for location in source.get("implementation_locations", [])
                }
            ),
            "last_verified_at": None,
        }
    )
    for source in sources:
        source["lifecycle"] = "deprecated"
        source["replacement_workflow_ids"] = [new_workflow_id]
    catalog["entries"].append(merged)
    _commit_catalog(root, catalog)
    return deepcopy(merged)


def deprecate_workflow(
    root: Path,
    workflow_id: str,
    *,
    replacement_workflow_ids: Sequence[str],
) -> dict[str, Any]:
    catalog = _read_catalog(root)
    entry = _entry(catalog, workflow_id)
    for replacement in replacement_workflow_ids:
        _entry(catalog, replacement)
    entry["lifecycle"] = "deprecated"
    entry["replacement_workflow_ids"] = list(replacement_workflow_ids)
    entry["standardized"] = False
    _commit_catalog(root, catalog)
    return deepcopy(entry)


def validate_workflow_catalog(root: Path) -> dict[str, Any]:
    catalog = _read_catalog(root)
    errors: list[str] = []
    ids: set[str] = set()
    for entry in catalog["entries"]:
        workflow_id = str(entry.get("workflow_id", ""))
        if not workflow_id:
            errors.append("entry missing workflow_id")
        elif workflow_id in ids:
            errors.append(f"duplicate workflow_id: {workflow_id}")
        ids.add(workflow_id)
        if entry.get("lifecycle") not in _LIFECYCLES:
            errors.append(f"invalid lifecycle: {workflow_id}")
        if entry.get("standardized") and entry.get("lifecycle") != "active":
            errors.append(f"standardized workflow is not active: {workflow_id}")
    for entry in catalog["entries"]:
        for replacement in entry.get("replacement_workflow_ids", []):
            if replacement not in ids:
                errors.append(
                    f"unknown replacement {replacement}: {entry.get('workflow_id')}"
                )
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "entry_count": len(catalog["entries"]),
    }
