#!/usr/bin/env python3
"""
check_project_state.py — PFMval project state health checker.

Checks configuration consistency, experiment integrity, and documentation
completeness. Reports PASS/WARN/FAIL with exit codes.

Usage:
    python scripts/check_project_state.py --mode local
    python scripts/check_project_state.py --mode server
    python scripts/check_project_state.py --mode local --strict
    python scripts/check_project_state.py --check-registry-only --experiment-id <id>
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# Utility
# ============================================================

class Checker:
    """Collects check results and reports summary."""

    def __init__(self, mode="local", strict=False):
        self.mode = mode
        self.strict = strict
        self.results = []

    def add(self, level, check_name, message):
        self.results.append((level, check_name, message))
        icon = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}.get(level, level)
        print(f"{icon} {check_name}: {message}")

    def fail(self, name, msg):
        self.add("FAIL", name, msg)

    def warn(self, name, msg):
        if self.strict:
            self.add("FAIL", name, f"[STRICT] {msg}")
        else:
            self.add("WARN", name, msg)

    def ok(self, name, msg=""):
        self.add("PASS", name, msg)

    def skip(self, name, msg=""):
        self.add("SKIP", name, msg)

    def summary(self):
        fails = sum(1 for r in self.results if r[0] == "FAIL")
        warns = sum(1 for r in self.results if r[0] == "WARN")
        passes = sum(1 for r in self.results if r[0] == "PASS")

        print(f"\n{'='*60}")
        print(f"[SUMMARY] Mode={self.mode} Strict={self.strict}")
        print(f"          PASS={passes}  WARN={warns}  FAIL={fails}")
        if fails > 0:
            print(f"[EXIT] FAIL ({fails} failure(s)) — exit 1")
        else:
            print(f"[EXIT] OK (0 failures) — exit 0")
        return 1 if fails > 0 else 0


# ============================================================
# Check implementations
# ============================================================

def check_gitignore_whitelist(checker):
    """Check 1: .gitignore whitelists experiment management files."""
    gitignore_path = PROJECT_ROOT / ".gitignore"
    if not gitignore_path.exists():
        checker.fail("gitignore", ".gitignore file not found")
        return

    content = gitignore_path.read_text(encoding="utf-8")
    required = [
        "!experiments/experiment_registry.json",
        "!experiments/experiment_dashboard.md",
        "!experiments/decision_log.md",
    ]
    for line in required:
        if line in content:
            checker.ok(f"gitignore whitelist: {line.split('/')[-1]}")
        else:
            checker.fail(f"gitignore whitelist", f"Missing: {line}")


def check_doc_state_conflicts(checker):
    """Check 2: CLAUDE.md / next-steps.md / session-brief.md state conflicts."""
    files_to_check = {
        "CLAUDE.md": PROJECT_ROOT / "CLAUDE.md",
        ".claude/next-steps.md": PROJECT_ROOT / ".claude" / "next-steps.md",
        ".claude/session-brief.md": PROJECT_ROOT / ".claude" / "session-brief.md",
    }

    conflict_pairs = [
        # (pattern_a, pattern_b, description)
        (r"待.*启动|尚未启动|pending.*start|not.*started", r"已启动|已经.*启动|started|running", "startup status conflict"),
        (r"尚未.*HF_HUB_OFFLINE|缺.*HF_HUB_OFFLINE|HF.*未.*加", r"setdefault.*HF_HUB_OFFLINE|HF_HUB_OFFLINE.*离线|offline.*HF", "HF_HUB_OFFLINE status conflict"),
        (r"待.*9.*患者|等.*九.*患者|wait.*9.*patient", r"9.*患者.*到|九.*患者.*齐|9.*patient.*arrived", "9-patient data status conflict"),
    ]

    found_any = False
    for filename, filepath in files_to_check.items():
        if not filepath.exists():
            checker.skip(f"conflict:{filename}", "file not found (skipping)")
            continue
        content = filepath.read_text(encoding="utf-8")
        for pat_a, pat_b, desc in conflict_pairs:
            match_a = re.search(pat_a, content, re.IGNORECASE)
            match_b = re.search(pat_b, content, re.IGNORECASE)
            if match_a and match_b:
                found_any = True
                snippet_a = match_a.group(0)[:50]
                snippet_b = match_b.group(0)[:50]
                checker.warn(f"conflict:{filename}", f"'{desc}': [{snippet_a}] vs [{snippet_b}]")

    if not found_any:
        checker.ok("conflict:docs", "no state conflicts detected")


def check_training_script_safeguards(checker):
    """Check 3: train_online_cls.py and train_online_tokens.py contain required safeguards."""
    scripts = {
        "train_online_cls.py": PROJECT_ROOT / "train_online_cls.py",
        "train_online_tokens.py": PROJECT_ROOT / "train_online_tokens.py",
    }

    required_checks = [
        ("HF_HUB_OFFLINE", "HF_HUB_OFFLINE", r"HF_HUB_OFFLINE"),
        ("--num_threads", "--num_threads argument", r"num_threads"),
        ("torch.set_num_threads", "CPU thread limit enforcement", r"torch\.set_num_threads"),
        ("val_loss < best_val_loss", "val_loss-based best epoch selection", r"val_loss\s*<"),
    ]

    for script_name, script_path in scripts.items():
        if not script_path.exists():
            checker.skip(f"safeguard:{script_name}", "file not found")
            continue
        content = script_path.read_text(encoding="utf-8")
        for check_key, check_desc, pattern in required_checks:
            if re.search(pattern, content):
                checker.ok(f"safeguard:{script_name}:{check_key}", check_desc)
            else:
                checker.fail(f"safeguard:{script_name}:{check_key}", f"MISSING: {check_desc}")


def check_workflow_hook_files(checker):
    """Check 4: .claude/settings.json workflow hook files exist."""
    settings_path = PROJECT_ROOT / ".claude" / "settings.json"
    if not settings_path.exists():
        checker.skip("hook:settings.json", "not found")
        return

    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        checker.fail("hook:settings.json", "invalid JSON")
        return

    hooks = settings.get("hooks", {})
    hook_files_to_check = []

    def _extract_commands(entry_list, group_name, matcher_name=""):
        """Extract Python script paths from a list of hook command entries."""
        label = f"{group_name}:{matcher_name}" if matcher_name else group_name
        for entry in entry_list:
            cmd = entry.get("command", "")
            for match in re.finditer(r'"([^"]+\.py)"', cmd):
                py_path = match.group(1)
                hook_files_to_check.append((label, py_path))

    for hook_group_name, hook_group in hooks.items():
        if isinstance(hook_group, dict):
            # Legacy format: {"hooks": [{"command": "..."}]}
            _extract_commands(hook_group.get("hooks", []), hook_group_name)
        elif isinstance(hook_group, list):
            # Current format: [{"matcher": "...", "hooks": [{"command": "..."}]}]
            for matcher_entry in hook_group:
                if isinstance(matcher_entry, dict):
                    matcher = matcher_entry.get("matcher", "")
                    _extract_commands(matcher_entry.get("hooks", []), hook_group_name, matcher)

    if not hook_files_to_check:
        checker.skip("hook:files", "no hook scripts found in settings.json")
        return

    for group_name, path_str in hook_files_to_check:
        p = Path(path_str)
        if p.exists():
            checker.ok(f"hook:{group_name}:{p.name}", str(p))
        else:
            # Check if path relative to project root
            alt = PROJECT_ROOT / path_str if not Path(path_str).is_absolute() else Path(path_str)
            if alt.exists():
                checker.ok(f"hook:{group_name}:{alt.name}", str(alt))
            else:
                checker.warn(f"hook:{group_name}:{p.name}", f"file not found: {path_str}")


def check_doc_registry_coverage(checker):
    """Check 5: doc-registry.json covers all md files in 01_指南与解读/ and 02_组会汇报/."""
    registry_path = PROJECT_ROOT / ".claude" / "doc-registry.json"
    if not registry_path.exists():
        checker.skip("doc-registry:coverage", "doc-registry.json not found")
        return

    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        checker.fail("doc-registry:json", "invalid JSON")
        return

    doc_dirs = [
        PROJECT_ROOT / "01_指南与解读",
        PROJECT_ROOT / "02_组会汇报",
    ]

    registered_paths = set()
    for doc in registry.get("documents", []):
        registered_paths.add(doc.get("path", ""))

    missing = []
    for doc_dir in doc_dirs:
        if not doc_dir.exists():
            continue
        for md_file in doc_dir.rglob("*.md"):
            rel_path = str(md_file.relative_to(PROJECT_ROOT)).replace("\\", "/")
            if rel_path not in registered_paths:
                missing.append(rel_path)

    if missing:
        checker.warn("doc-registry:coverage", f"{len(missing)} unregistered file(s): {', '.join(missing[:5])}")
    else:
        checker.ok("doc-registry:coverage", "all md files registered")


def check_experiment_registry_schema(checker):
    """Check 6: experiment_registry.json schema completeness."""
    registry_path = PROJECT_ROOT / "experiments" / "experiment_registry.json"
    if not registry_path.exists():
        checker.fail("registry:schema", "experiment_registry.json not found")
        return

    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        checker.fail("registry:json", "invalid JSON")
        return

    required_top = ["version", "updated_at", "experiments"]
    for key in required_top:
        if key not in registry:
            checker.fail(f"registry:schema:top", f"missing top-level key: {key}")

    required_exp = [
        "id", "status", "script", "run_dir", "selection_metric",
        "created_at", "started_at", "completed_at"
    ]
    for i, exp in enumerate(registry.get("experiments", [])):
        eid = exp.get("id", f"index_{i}")
        for key in required_exp:
            if key not in exp:
                checker.fail(f"registry:schema:{eid}", f"missing field: {key}")

    valid_statuses = {"done", "done_incomplete_data", "running", "planned", "paused", "failed"}
    for exp in registry.get("experiments", []):
        eid = exp.get("id", "?")
        status = exp.get("status", "")
        if status not in valid_statuses:
            checker.warn(f"registry:schema:{eid}", f"unknown status: '{status}'")

    checker.ok("registry:schema", f"{len(registry.get('experiments', []))} experiments validated")


def check_run_dir_existence(checker):
    """Check 7: run_dir existence, context-dependent on mode."""
    registry_path = PROJECT_ROOT / "experiments" / "experiment_registry.json"
    if not registry_path.exists():
        return

    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return

    for exp in registry.get("experiments", []):
        eid = exp.get("id", "?")
        run_dir_str = exp.get("run_dir", "")
        if not run_dir_str:
            checker.skip(f"run_dir:{eid}", "no run_dir specified")
            continue

        run_dir = Path(run_dir_str)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir

        exists = run_dir.exists()
        has_csv = (run_dir / "training_history.csv").exists() if exists else False

        if exists:
            if has_csv:
                checker.ok(f"run_dir:{eid}", f"exists + CSV — {run_dir}")
            else:
                if checker.mode == "server":
                    checker.fail(f"run_dir:{eid}", f"exists but NO training_history.csv — {run_dir}")
                else:
                    checker.warn(f"run_dir:{eid}", f"exists but NO training_history.csv (likely server-trained)")
        else:
            status = exp.get("status", "")
            if status in ("running", "done", "done_incomplete_data"):
                if checker.mode == "server":
                    checker.fail(f"run_dir:{eid}", f"status={status} but run_dir missing: {run_dir}")
                else:
                    checker.warn(f"run_dir:{eid}", f"status={status} but run_dir missing locally (likely server-side)")
            else:
                checker.warn(f"run_dir:{eid}", f"status={status}, run_dir does not exist: {run_dir}")


def check_done_experiments_loss(checker):
    """Check 8: done experiments missing best_val_loss."""
    registry_path = PROJECT_ROOT / "experiments" / "experiment_registry.json"
    if not registry_path.exists():
        return

    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return

    for exp in registry.get("experiments", []):
        eid = exp.get("id", "?")
        status = exp.get("status", "")
        metric = exp.get("selection_metric", "")
        loss = exp.get("best_val_loss")

        if status == "done" and metric == "val_loss_min" and loss is None:
            checker.fail(f"done:loss:{eid}", "status=done + val_loss_min but best_val_loss is null")
        elif status == "done_incomplete_data":
            if loss is None:
                checker.warn(f"done:loss:{eid}", "status=done_incomplete_data, best_val_loss=null (expected)")
            else:
                checker.ok(f"done:loss:{eid}", "has best_val_loss despite done_incomplete_data — consider updating status to done")


def check_registry_entry_exists(experiment_id):
    """Check if a specific experiment ID exists in the registry."""
    registry_path = PROJECT_ROOT / "experiments" / "experiment_registry.json"
    if not registry_path.exists():
        print(f"[FAIL] Registry file not found: {registry_path}")
        return False

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    for exp in registry.get("experiments", []):
        if exp["id"] == experiment_id:
            status = exp.get("status", "unknown")
            print(f"[PASS] Experiment '{experiment_id}' found in registry (status={status})")
            return True

    print(f"[FAIL] Experiment '{experiment_id}' NOT found in registry")
    return False


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PFMval project state health checker")
    parser.add_argument("--mode", default="local", choices=["local", "server"],
                        help="local: missing run_dir is WARN; server: missing run_dir is FAIL")
    parser.add_argument("--strict", action="store_true",
                        help="Promote all WARN to FAIL")
    parser.add_argument("--check-registry-only", action="store_true",
                        help="Only check if experiment ID exists in registry")
    parser.add_argument("--experiment-id", default=None,
                        help="Experiment ID for --check-registry-only")
    args = parser.parse_args()

    if args.check_registry_only:
        if not args.experiment_id:
            print("[FAIL] --experiment-id required with --check-registry-only")
            sys.exit(1)
        exists = check_registry_entry_exists(args.experiment_id)
        sys.exit(0 if exists else 1)

    checker = Checker(mode=args.mode, strict=args.strict)
    print(f"[INFO] PFMval Project State Check — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Mode={args.mode}  Strict={args.strict}  Root={PROJECT_ROOT}")
    print()

    check_gitignore_whitelist(checker)
    check_doc_state_conflicts(checker)
    check_training_script_safeguards(checker)
    check_workflow_hook_files(checker)
    check_doc_registry_coverage(checker)
    check_experiment_registry_schema(checker)
    check_run_dir_existence(checker)
    check_done_experiments_loss(checker)

    exit_code = checker.summary()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
