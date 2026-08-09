"""Fail-closed Phase 3 server preflight with one aggregated report.

This module is read-only.  It validates the user-approved 307-slide universe,
the selected slide-level split package and an explicit server asset manifest.
It never creates a job or starts training.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

HARD_GATE_NAMES = (
    "exclusion_set_integrity",
    "effective_307_coverage",
    "feature_mpp2_coverage",
    "pathway_30_order",
    "patch_coordinate_alignment",
    "train_only_fit_isolation",
)

RAW_PATHWAY_INPUT = "repaired_frozen_mpp2_raw_only"
LEGACY_ZSCORE_COMPATIBILITY = "legacy_zscore_compatibility"
PATHWAY_INPUT_MODES = frozenset({RAW_PATHWAY_INPUT, LEGACY_ZSCORE_COMPATIBILITY})


@dataclass(frozen=True)
class GateResult:
    name: str
    severity: str
    status: str
    detail: str


class GateReport:
    def __init__(self, *, pathway_input_mode: str = RAW_PATHWAY_INPUT) -> None:
        if pathway_input_mode not in PATHWAY_INPUT_MODES:
            raise ValueError(f"unsupported pathway input mode: {pathway_input_mode}")
        self.pathway_input_mode = pathway_input_mode
        self.results: list[GateResult] = []

    def add(self, name: str, severity: str, status: str, detail: str) -> None:
        if severity not in {"HARD_FAIL", "WARN"}:
            raise ValueError(f"invalid severity: {severity}")
        if status not in {"PASS", "WARN", "FAIL"}:
            raise ValueError(f"invalid status: {status}")
        self.results.append(GateResult(name, severity, status, detail))

    @property
    def failed(self) -> bool:
        return any(item.severity == "HARD_FAIL" and item.status == "FAIL" for item in self.results)

    def payload(self) -> dict[str, object]:
        counts = {name: sum(item.status == name for item in self.results) for name in ("PASS", "WARN", "FAIL")}
        compatibility_only = self.pathway_input_mode == LEGACY_ZSCORE_COMPATIBILITY
        return {
            "verdict": "NO-GO" if self.failed else ("COMPATIBILITY_ONLY" if compatibility_only else "GO"),
            "counts": counts,
            "hard_gate_names": list(HARD_GATE_NAMES),
            "pathway_input_mode": self.pathway_input_mode,
            "compatibility_only": compatibility_only,
            "training_authorized": False,
            "results": [asdict(item) for item in self.results],
            "training_started": False,
            "job_created": False,
        }


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _ids(rows: Iterable[Mapping[str, object]]) -> list[str]:
    return [str(row.get("slide_id", "")).strip() for row in rows]


def _validate_assignment_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_slide_ids: set[str],
) -> int:
    """Validate assignment CSVs using only the Python standard library."""

    required = {"case_id", "slide_id", "split"}
    if not rows:
        raise ValueError("assignment CSV is empty")
    if missing := required - set(rows[0]):
        raise ValueError(f"assignment missing columns: {sorted(missing)}")
    seen: set[str] = set()
    by_split: dict[str, set[str]] = {name: set() for name in ("train", "val", "test")}
    patient_by_split: dict[str, set[str]] = {name: set() for name in ("train", "val", "test")}
    for row in rows:
        slide_id = str(row.get("slide_id", "")).strip()
        case_id = str(row.get("case_id", "")).strip()
        split = str(row.get("split", "")).strip()
        if not slide_id or not case_id:
            raise ValueError("assignment contains empty case_id/slide_id")
        if slide_id in seen:
            raise ValueError(f"slide assigned more than once: {slide_id}")
        if split not in by_split:
            raise ValueError(f"unknown split label: {split}")
        seen.add(slide_id)
        by_split[split].add(slide_id)
        patient_by_split[split].add(case_id)
    if seen != expected_slide_ids:
        raise ValueError(
            f"fold coverage mismatch: missing={sorted(expected_slide_ids-seen)[:10]} "
            f"unexpected={sorted(seen-expected_slide_ids)[:10]}"
        )
    if any(not by_split[name] for name in by_split):
        raise ValueError("fold lacks train, val, or test rows")
    overlaps = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlaps.extend(patient_by_split[left] & patient_by_split[right])
    return len(set(overlaps))


def _validate_pathway_names_stdlib(names: object, expected_count: int = 30) -> tuple[str, ...]:
    if not isinstance(names, list):
        raise ValueError("pathway_names must be a list")
    cleaned = tuple(str(name).strip() for name in names)
    if len(cleaned) != expected_count or any(not name for name in cleaned):
        raise ValueError(f"expected {expected_count} non-empty pathway names, got {len(cleaned)}")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("pathway names must be unique and order-preserving")
    return cleaned


def _add_hard(report: GateReport, name: str, action) -> object | None:
    try:
        value = action()
    except Exception as exc:  # aggregate every hard-gate failure in one run
        report.add(name, "HARD_FAIL", "FAIL", str(exc))
        return None
    report.add(name, "HARD_FAIL", "PASS", str(value))
    return value


def audit_cohort_and_splits(
    experiment_root: Path,
    report: GateReport,
    *,
    source_count: int,
    excluded_count: int,
    effective_count: int,
    patient_count: int,
    seeds: Sequence[int],
    tasks: Sequence[str],
    folds_per_seed: int,
) -> set[str] | None:
    package = experiment_root / "slide_seed_package_v001"
    source_rows = _rows(package / "cohort_manifest.csv")
    effective_rows = _rows(package / "effective_cohort.csv")
    excluded_rows = _rows(package / "excluded_bad_slides.csv")
    source_ids, effective_ids, excluded_ids = map(_ids, (source_rows, effective_rows, excluded_rows))

    def exclusion_check() -> str:
        for label, values, expected in (
            ("source", source_ids, source_count),
            ("excluded", excluded_ids, excluded_count),
            ("effective", effective_ids, effective_count),
        ):
            if len(values) != expected or len(set(values)) != expected or any(not item for item in values):
                raise ValueError(f"{label} IDs must be {expected} unique non-empty slides, got rows={len(values)} unique={len(set(values))}")
        source, effective, excluded = set(source_ids), set(effective_ids), set(excluded_ids)
        if excluded - source:
            raise ValueError(f"excluded slides absent from source: {sorted(excluded - source)[:10]}")
        if effective & excluded:
            raise ValueError(f"excluded slides re-enter effective cohort: {sorted(effective & excluded)[:10]}")
        if source - excluded != effective:
            raise ValueError(
                f"source-minus-excluded mismatch: missing={sorted((source - excluded) - effective)[:10]} "
                f"unexpected={sorted(effective - (source - excluded))[:10]}"
            )
        return f"source={source_count}, excluded={excluded_count}, effective={effective_count}"

    exclusion_ok = _add_hard(report, "exclusion_set_integrity", exclusion_check)
    if exclusion_ok is None:
        return None

    effective = set(effective_ids)

    def coverage_check() -> str:
        patients = {str(row.get("case_id", "")).strip() for row in effective_rows}
        if len(patients) != patient_count or "" in patients:
            raise ValueError(f"effective patient count expected {patient_count}, got {len(patients)}")
        expected_files = len(seeds) * len(tasks) * folds_per_seed
        seen_files = 0
        overlap_counts: list[int] = []
        for seed in seeds:
            for task in tasks:
                for fold in range(folds_per_seed):
                    path = package / f"seed_{seed}" / task / f"slide_assignments_{fold}.csv"
                    if not path.is_file():
                        raise ValueError(f"missing assignment: {path.relative_to(experiment_root)}")
                    overlap_counts.append(
                        _validate_assignment_rows(_rows(path), expected_slide_ids=effective)
                    )
                    seen_files += 1
        if seen_files != expected_files:
            raise ValueError(f"expected {expected_files} assignment files, got {seen_files}")
        report.add(
            "slide_split_patient_overlap",
            "WARN",
            "WARN" if any(overlap_counts) else "PASS",
            f"approved slide-level design; overlap patients per fold range={min(overlap_counts)}..{max(overlap_counts)}",
        )
        return f"{seen_files} assignments each cover exactly {effective_count} effective slides"

    _add_hard(report, "effective_307_coverage", coverage_check)
    return effective


def _asset_records(asset_manifest: Mapping[str, object]) -> list[dict[str, object]]:
    records = asset_manifest.get("records")
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ValueError("asset manifest records must be a list of objects")
    return records


def _resolve_pathway_input_mode(config: Mapping[str, object]) -> str:
    mode = str(config.get("pathway_input_mode", config.get("accepted_pathway_input", RAW_PATHWAY_INPUT))).strip()
    if mode not in PATHWAY_INPUT_MODES:
        raise ValueError(f"pathway_input_mode must be one of {sorted(PATHWAY_INPUT_MODES)}, got {mode!r}")
    accepted = str(config.get("accepted_pathway_input", mode)).strip()
    if accepted != mode:
        raise ValueError(
            "accepted_pathway_input must equal pathway_input_mode for an explicit route binding: "
            f"{accepted!r} != {mode!r}"
        )
    if mode == LEGACY_ZSCORE_COMPATIBILITY:
        if config.get("compatibility_only") is not True:
            raise ValueError("legacy_zscore_compatibility requires compatibility_only=true")
        if config.get("raw_mpp2_claim") is not False:
            raise ValueError("legacy_zscore_compatibility requires raw_mpp2_claim=false")
        if config.get("pathway_evidence_status") not in {"diagnostic_only", "pending_review"}:
            raise ValueError(
                "legacy_zscore_compatibility pathway_evidence_status must be diagnostic_only or pending_review"
            )
    return mode


def _validate_manifest_pathway_route(
    asset_manifest: Mapping[str, object],
    pathway_input_mode: str,
) -> str:
    """Require an explicit compatibility annotation before accepting legacy z-scores."""

    manifest_mode = str(asset_manifest.get("pathway_input_mode", RAW_PATHWAY_INPUT)).strip()
    if manifest_mode != pathway_input_mode:
        raise ValueError(
            f"asset manifest pathway_input_mode mismatch: expected {pathway_input_mode!r}, got {manifest_mode!r}"
        )
    if pathway_input_mode == RAW_PATHWAY_INPUT:
        return "pathway input route=repaired_frozen_mpp2_raw_only"

    annotation = asset_manifest.get("compatibility_annotation")
    if not isinstance(annotation, Mapping):
        raise ValueError("legacy_zscore_compatibility requires compatibility_annotation")
    if annotation.get("source_kind") != "legacy_zscore":
        raise ValueError("legacy_zscore_compatibility requires source_kind=legacy_zscore")
    if annotation.get("raw_mpp2_claim") is not False:
        raise ValueError("legacy_zscore_compatibility must set raw_mpp2_claim=false")
    if annotation.get("evidence_status") not in {"diagnostic_only", "pending_review"}:
        raise ValueError(
            "legacy_zscore_compatibility evidence_status must be diagnostic_only or pending_review"
        )
    return "pathway input route=legacy_zscore_compatibility; raw MPP2 claim=false; compatibility-only"


def audit_assets(
    asset_manifest_path: Path,
    effective_ids: set[str],
    excluded_ids: set[str],
    canonical_pathway_names: Sequence[str],
    pathway_input_mode: str,
    report: GateReport,
) -> None:
    asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
    records = _asset_records(asset_manifest)

    def coverage_check() -> str:
        route_detail = _validate_manifest_pathway_route(asset_manifest, pathway_input_mode)
        slide_ids = [str(record.get("slide_id", "")).strip() for record in records]
        if len(slide_ids) != len(effective_ids) or len(set(slide_ids)) != len(effective_ids):
            raise ValueError(f"asset records must contain {len(effective_ids)} unique slide IDs, got rows={len(slide_ids)} unique={len(set(slide_ids))}")
        actual = set(slide_ids)
        if actual != effective_ids:
            raise ValueError(f"asset coverage mismatch: missing={sorted(effective_ids-actual)[:10]} unexpected={sorted(actual-effective_ids)[:10]}")
        if actual & excluded_ids:
            raise ValueError(f"excluded slides present in asset records: {sorted(actual & excluded_ids)[:10]}")
        for record in records:
            missing_roles = [role for role in ("pathology", "pathway", "coordinates") if not record.get(f"{role}_path") or not record.get(f"{role}_sha256")]
            if missing_roles:
                raise ValueError(f"{record.get('slide_id')} missing paths/hashes for {missing_roles}")
        return f"{route_detail}; all {len(effective_ids)} slides have explicit pathway/coordinate paths and hashes"

    _add_hard(report, "feature_mpp2_coverage", coverage_check)

    def pathway_check() -> str:
        actual = _validate_pathway_names_stdlib(asset_manifest.get("pathway_names", []), expected_count=30)
        expected = _validate_pathway_names_stdlib(list(canonical_pathway_names), expected_count=30)
        if actual != expected:
            mismatch = next((i for i, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]), None)
            raise ValueError(f"pathway order differs from accepted MPP2 order at index {mismatch}")
        return "30 pathway names exactly match accepted MPP2 order"

    _add_hard(report, "pathway_30_order", pathway_check)

    def alignment_check() -> str:
        # Heavy numerical readers are loaded only for the full asset gate.
        from .io import load_verified_patch_bag

        base = asset_manifest_path.parent
        for record in records:
            resolved = dict(record)
            for role in ("pathology", "pathway", "coordinates"):
                path = Path(str(record[f"{role}_path"]))
                resolved[f"{role}_path"] = str(path if path.is_absolute() else (base / path).resolve())
            load_verified_patch_bag(resolved, pathway_dim=30)
        return f"SHA, finite matrices and patch-row alignment passed for {len(records)} slides"

    _add_hard(report, "patch_coordinate_alignment", alignment_check)


def audit_train_only_policy(
    experiment_root: Path,
    config: Mapping[str, object],
    asset_manifest: Mapping[str, object],
    report: GateReport,
) -> None:
    def check() -> str:
        policy = config.get("training_fit_policy")
        if not isinstance(policy, Mapping):
            raise ValueError("training_fit_policy is missing")
        expected = {
            "fit_scope": "train_only",
            "validation_mode": "transform_only",
            "test_mode": "transform_only_after_checkpoint_freeze",
            "external_mode": "transform_only_no_fit",
        }
        bad = {key: (policy.get(key), value) for key, value in expected.items() if policy.get(key) != value}
        if bad:
            raise ValueError(f"training fit policy mismatch: {bad}")
        fit_records = asset_manifest.get("fit_scope_records")
        if not isinstance(fit_records, list) or not all(isinstance(item, Mapping) for item in fit_records):
            raise ValueError("asset manifest fit_scope_records must be a list of objects")
        expected: dict[tuple[int, str, int], set[str]] = {}
        package = experiment_root / "slide_seed_package_v001"
        for seed in [int(item) for item in config["seeds"]]:
            for task in [str(item) for item in config["tasks"]]:
                for fold in range(int(config["folds_per_seed"])):
                    rows = _rows(package / f"seed_{seed}" / task / f"slide_assignments_{fold}.csv")
                    expected[(seed, task, fold)] = {
                        str(row["slide_id"]).strip() for row in rows if str(row["split"]).strip() == "train"
                    }
        actual: dict[tuple[int, str, int], set[str]] = {}
        for record in fit_records:
            key = (int(record.get("seed", -1)), str(record.get("task", "")), int(record.get("fold", -1)))
            if key in actual:
                raise ValueError(f"duplicate fit scope record: {key}")
            fit_ids = {str(item).strip() for item in record.get("fit_slide_ids", [])}
            actual[key] = fit_ids
        if set(actual) != set(expected):
            raise ValueError(f"fit scope keys mismatch: missing={sorted(set(expected)-set(actual))[:5]} unexpected={sorted(set(actual)-set(expected))[:5]}")
        for key, train_ids in expected.items():
            split_rows = _rows(package / f"seed_{key[0]}" / key[1] / f"slide_assignments_{key[2]}.csv")
            split_by_id = {str(row["slide_id"]).strip(): str(row["split"]).strip() for row in split_rows}
            leaked = sorted(item for item in actual[key] if split_by_id.get(item) != "train")
            if leaked:
                raise ValueError(f"fit IDs outside train split for {key}: {leaked[:10]}")
            if actual[key] != train_ids:
                raise ValueError(
                    f"fit IDs must exactly equal train IDs for {key}: "
                    f"missing={sorted(train_ids-actual[key])[:10]} unexpected={sorted(actual[key]-train_ids)[:10]}"
                )
        return f"{len(expected)} fit scopes exactly equal their train split; validation/test/external are excluded"

    _add_hard(report, "train_only_fit_isolation", check)


def run_preflight(
    experiment_root: Path,
    config_path: Path,
    asset_manifest_path: Path | None,
    *,
    cohort_only: bool = False,
) -> GateReport:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        pathway_input_mode = _resolve_pathway_input_mode(config)
    except Exception as exc:
        report = GateReport()
        report.add("feature_mpp2_coverage", "HARD_FAIL", "FAIL", f"invalid pathway input route: {exc}")
        return report
    report = GateReport(pathway_input_mode=pathway_input_mode)
    if pathway_input_mode == LEGACY_ZSCORE_COMPATIBILITY:
        report.add(
            "pathway_input_route",
            "WARN",
            "WARN",
            "legacy z-score compatibility route; results are not raw MPP2 and are pending review",
        )
    else:
        report.add("pathway_input_route", "WARN", "PASS", "repaired frozen raw MPP2 route")
    effective = audit_cohort_and_splits(
        experiment_root,
        report,
        source_count=int(config["source_slide_count"]),
        excluded_count=int(config["excluded_bad_slide_count"]),
        effective_count=int(config["selected_effective_slide_count"]),
        patient_count=int(config["expected_patient_count"]),
        seeds=[int(item) for item in config["seeds"]],
        tasks=[str(item) for item in config["tasks"]],
        folds_per_seed=int(config["folds_per_seed"]),
    )
    if cohort_only:
        report.add("train_only_fit_isolation", "HARD_FAIL", "WARN", "cohort-only mode: per-fold fit IDs NOT RUN")
        report.add("server_assets", "WARN", "WARN", "cohort-only mode: server asset gates NOT RUN")
        return report
    if asset_manifest_path is None:
        for name in ("feature_mpp2_coverage", "pathway_30_order", "patch_coordinate_alignment", "train_only_fit_isolation"):
            report.add(name, "HARD_FAIL", "FAIL", "--asset-manifest is required for full server preflight")
        return report
    if effective is None:
        for name in ("feature_mpp2_coverage", "pathway_30_order", "patch_coordinate_alignment"):
            report.add(name, "HARD_FAIL", "FAIL", "effective cohort is invalid; asset validation cannot proceed")
        return report
    excluded = set(_ids(_rows(experiment_root / "slide_seed_package_v001" / "excluded_bad_slides.csv")))
    canonical = json.loads((experiment_root.parents[1] / "mpp_standard_splits/group_2/zscore_manifest.json").read_text(encoding="utf-8"))["pathway_names"]
    asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
    audit_assets(asset_manifest_path, effective, excluded, canonical, pathway_input_mode, report)
    audit_train_only_policy(experiment_root, config, asset_manifest, report)
    report.add("scientific_performance_gate", "WARN", "WARN", "performance/model promotion remains user-reviewed and is not a preflight blocker")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only Phase 3 W004 server preflight")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path)
    parser.add_argument("--cohort-only", action="store_true")
    args = parser.parse_args(argv)
    report = run_preflight(args.experiment_root.resolve(), args.config.resolve(), None if args.asset_manifest is None else args.asset_manifest.resolve(), cohort_only=args.cohort_only)
    payload = report.payload()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
