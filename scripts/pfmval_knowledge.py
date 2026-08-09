"""Local project facts and document-profile rendering."""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


DOCUMENT_PROFILES = {
    "project_fact",
    "meeting_brief",
    "research_path",
    "learning_guide",
    "paper_output",
    "lifecycle_review",
}
PROJECT_MATERIAL_PREFIXES = (
    "directive:",
    "fact:",
    "experiment:",
    "result:",
    "explore:",
    "file:",
)


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _facts_path(root: Path) -> Path:
    return root / "project_state" / "project_facts.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    values = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid project fact JSONL at line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"project fact at line {line_number} is not an object"
            )
        values.append(value)
    return values


def _write_jsonl_atomic(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for value in values:
                handle.write(_canonical(dict(value)) + "\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _validate_fact(fact: Mapping[str, Any]) -> None:
    required = {
        "fact_id",
        "kind",
        "statement",
        "source",
        "observed_at",
        "status",
        "supersedes",
    }
    missing = sorted(required - set(fact))
    if missing:
        raise ValueError(f"project fact missing: {', '.join(missing)}")
    if not str(fact["fact_id"]).strip() or not str(fact["statement"]).strip():
        raise ValueError("project fact id and statement must be non-empty")
    if fact["status"] not in {"candidate", "active", "superseded", "historical"}:
        raise ValueError("invalid project fact status")
    if not isinstance(fact["supersedes"], list):
        raise ValueError("project fact supersedes must be an array")


def list_project_facts(root: Path) -> list[dict[str, Any]]:
    return _read_jsonl(_facts_path(root))


def _current_facts(facts: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    values = list(facts)
    superseded_ids = {
        str(fact_id)
        for value in values
        for fact_id in value.get("supersedes", [])
    }
    return [
        value
        for value in values
        if value.get("status") == "active"
        and value.get("fact_id") not in superseded_ids
    ]


def preview_project_fact(root: Path, fact: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(_canonical(dict(fact)))
    _validate_fact(normalized)
    facts = list_project_facts(root)
    existing = next(
        (
            value
            for value in facts
            if value.get("fact_id") == normalized["fact_id"]
        ),
        None,
    )
    if existing is not None:
        comparable_existing = {
            key: value
            for key, value in existing.items()
            if key != "confirmation_source"
        }
        if _canonical(comparable_existing) == _canonical(normalized):
            return {"status": "already_recorded", "writes": 0}
        raise ValueError("fact_id is already bound to different content")
    current_same_kind = [
        value
        for value in _current_facts(facts)
        if value.get("kind") == normalized["kind"]
        and value.get("statement") != normalized["statement"]
    ]
    supersedes = {str(value) for value in normalized.get("supersedes", [])}
    unresolved = [
        value
        for value in current_same_kind
        if str(value.get("fact_id")) not in supersedes
    ]
    if unresolved:
        return {
            "status": "conflict_review_required",
            "writes": 0,
            "conflicting_fact_ids": sorted(
                str(value["fact_id"]) for value in unresolved
            ),
        }
    return {
        "status": "confirmation_required",
        "writes": 0,
        "superseded_fact_ids": sorted(supersedes),
    }


def confirm_project_fact(
    root: Path,
    fact: Mapping[str, Any],
    *,
    confirmation_source: str,
) -> dict[str, Any]:
    if confirmation_source != "explicit_user_confirmation":
        raise ValueError("project fact requires explicit user confirmation")
    preview = preview_project_fact(root, fact)
    if preview["status"] == "already_recorded":
        existing = next(
            value
            for value in list_project_facts(root)
            if value.get("fact_id") == fact.get("fact_id")
        )
        return {"status": "already_recorded", "fact": existing}
    if preview["status"] == "conflict_review_required":
        raise ValueError(
            "project fact conflict requires explicit supersedes or user review"
        )
    normalized = json.loads(_canonical(dict(fact)))
    normalized["confirmation_source"] = confirmation_source
    facts = list_project_facts(root)
    _write_jsonl_atomic(_facts_path(root), [*facts, normalized])
    return {"status": "recorded", "fact": normalized}


def _registry_experiment(
    root: Path,
    reference: str,
) -> Mapping[str, Any]:
    try:
        experiment_id, result_id = reference.split(":", 1)
    except ValueError as exc:
        raise ValueError(
            f"experiment result ref must be experiment:result: {reference}"
        ) from exc
    registry = json.loads(
        (root / "experiments" / "experiment_registry.json").read_text(
            encoding="utf-8"
        )
    )
    experiment = next(
        (
            item
            for item in registry.get("experiments", [])
            if item.get("id") == experiment_id
            and item.get("result_id") == result_id
        ),
        None,
    )
    if experiment is None:
        raise ValueError(f"unknown experiment result ref: {reference}")
    return experiment


def _meeting_brief(root: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    accepted_refs = list(payload.get("accepted_result_refs", []))
    failed_refs = list(payload.get("failed_refs", []))
    for reference in accepted_refs:
        experiment = _registry_experiment(root, str(reference))
        if experiment.get("evidence_status") != "accepted":
            raise ValueError(f"meeting accepted ref is not accepted: {reference}")
    for reference in failed_refs:
        experiment = _registry_experiment(root, str(reference))
        if (
            experiment.get("status") != "failed"
            and experiment.get("evidence_status") != "rejected"
        ):
            raise ValueError(f"meeting failed ref is not failed: {reference}")
    lines = [
        f"# {payload['title']}",
        "",
        "## 近期工作",
        *[f"- {value}" for value in payload.get("recent_work", [])],
        "",
        "## 已接纳证据",
        *[f"- `{value}`" for value in accepted_refs],
        "",
        "## 探索材料（非 accepted）",
        *[f"- `{value}`" for value in payload.get("explore_refs", [])],
        "",
        "## 失败/否定材料",
        *[f"- `{value}`" for value in failed_refs],
        "",
        "## 难点",
        *[f"- {value}" for value in payload.get("difficulties", [])],
        "",
        "## 下一步",
        *[f"- {value}" for value in payload.get("next_steps", [])],
        "",
    ]
    return {
        "document_id": payload["document_id"],
        "profile": "meeting_brief",
        "status": "draft",
        "lifecycle": "draft",
        "source_refs": [*accepted_refs, *failed_refs],
        "body": "\n".join(lines),
    }


def _research_path(payload: Mapping[str, Any]) -> dict[str, Any]:
    lines = [f"# {payload['title']}", "", "## 已确认结论"]
    lines.extend(f"- {value}" for value in payload["confirmed_conclusions"])
    lines.extend(["", "## 初步解释与不确定性"])
    lines.extend(
        f"- {value['statement']}（不确定性：{value['uncertainty']}）"
        for value in payload["interpretations"]
    )
    lines.extend(["", "## 被否定路线与停止原因"])
    lines.extend(
        f"- {value['route']}：{value['stop_reason']}"
        for value in payload["negative_routes"]
    )
    lines.extend(["", "## 开放假设"])
    lines.extend(f"- {value}" for value in payload["open_hypotheses"])
    lines.append("")
    return {
        "document_id": payload["document_id"],
        "profile": "research_path",
        "status": "draft",
        "lifecycle": "draft",
        "body": "\n".join(lines),
    }


def _learning_interface(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "document_id",
        "title",
        "topic",
        "audience",
        "prerequisites",
        "source_refs",
        "experiment_refs",
        "last_verified_commit",
        "supersedes",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(
            f"learning guide interface missing: {', '.join(missing)}"
        )
    return {
        "document_id": payload["document_id"],
        "profile": "learning_guide",
        "status": "interface_reserved",
        "lifecycle": "draft",
        "interface": dict(payload),
        "writes": 0,
    }


def _paper_interface(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "document_id",
        "title",
        "template_ref",
        "material_refs",
        "claim_refs",
        "section_mapping",
        "output_path",
        "review_status",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"paper interface missing: {', '.join(missing)}")
    for reference in payload["material_refs"]:
        if not str(reference).startswith(PROJECT_MATERIAL_PREFIXES):
            raise ValueError(
                f"external material requires user review: {reference}"
            )
    if payload["template_ref"] is not None:
        raise ValueError(
            "paper template handling requires a later approved design revision"
        )
    return {
        "document_id": payload["document_id"],
        "profile": "paper_output",
        "status": "template_pending",
        "lifecycle": "draft",
        "interface": dict(payload),
        "writes": 0,
    }


def build_document_profile(
    root: Path,
    *,
    profile: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if profile not in DOCUMENT_PROFILES:
        raise ValueError(f"unsupported document profile: {profile}")
    if profile == "meeting_brief":
        return _meeting_brief(root, payload)
    if profile == "research_path":
        return _research_path(payload)
    if profile == "learning_guide":
        registry_path = root / "project_state" / "document_registry.json"
        if registry_path.exists():
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registered = next(
                (
                    item
                    for item in registry.get("documents", [])
                    if item.get("doc_id") == str(payload.get("document_id", ""))
                    and item.get("lifecycle") == "active"
                    and item.get("doc_role") == "learning_guide"
                ),
                None,
            )
            if registered is not None:
                preview = review_document_freshness(
                    root,
                    document_id=str(registered["doc_id"]),
                )
                return {
                    "document_id": str(registered["doc_id"]),
                    "profile": "learning_guide",
                    "status": preview["freshness"],
                    "freshness": preview["freshness"],
                    "lifecycle": preview["lifecycle"],
                    "source_refs": list(registered.get("source_refs", [])),
                    "dependency_fingerprints": preview[
                        "current_fingerprints"
                    ],
                    "changed_dependencies": preview[
                        "changed_dependencies"
                    ],
                    "writes": 0,
                }
        return _learning_interface(payload)
    if profile == "paper_output":
        return _paper_interface(payload)
    if profile == "project_fact":
        return {
            "document_id": str(payload["fact_id"]),
            "profile": profile,
            "status": "confirmation_required",
            "lifecycle": "draft",
            "preview": preview_project_fact(root, payload),
            "writes": 0,
        }
    if profile == "lifecycle_review":
        preview = review_document_freshness(
            root,
            document_id=str(payload["document_id"]),
            current_fingerprints=payload["current_fingerprints"],
        )
        return {
            "document_id": str(payload["document_id"]),
            "profile": profile,
            "status": preview["freshness"],
            "lifecycle": preview["lifecycle"],
            "preview": preview,
            "writes": 0,
        }
    raise AssertionError(profile)


def review_document_freshness(
    root: Path,
    *,
    document_id: str,
    current_fingerprints: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    registry = json.loads(
        (root / "project_state" / "document_registry.json").read_text(
            encoding="utf-8"
        )
    )
    document = next(
        (
            item
            for item in registry.get("documents", [])
            if item.get("doc_id") == document_id
        ),
        None,
    )
    if document is None:
        raise ValueError(f"unknown document_id: {document_id}")
    if current_fingerprints is None:
        documents = {
            str(item.get("doc_id")): item
            for item in registry.get("documents", [])
            if item.get("doc_id")
        }
        resolved: dict[str, str] = {}
        for source_id in document.get("source_refs", []):
            source_key = str(source_id)
            source = documents.get(source_key)
            if source is None:
                resolved[source_key] = ""
                continue
            source_path = root / str(source.get("path", ""))
            if source_path.is_file():
                resolved[source_key] = hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest()
            else:
                resolved[source_key] = str(
                    source.get("content_sha256", "")
                )
        current_fingerprints = resolved
    current = dict(current_fingerprints)
    previous = document.get("dependency_fingerprints", {})
    changed = _canonical(previous) != _canonical(current)
    return {
        "document_id": document_id,
        "freshness": "review_due" if changed else document.get(
            "freshness",
            "fresh",
        ),
        "lifecycle": document.get("lifecycle"),
        "changed_dependencies": sorted(
            key
            for key in set(previous) | set(current)
            if previous.get(key) != current.get(key)
        ),
        "current_fingerprints": current,
        "writes": 0,
        "deletes": 0,
        "directive_closures": 0,
    }
