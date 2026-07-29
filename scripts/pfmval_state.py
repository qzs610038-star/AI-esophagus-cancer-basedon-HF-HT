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
import platform
import re
import shutil
import subprocess
import sys
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


STATE_SCHEMA_VERSION = "1.0"
DOCUMENT_REGISTRY_SCHEMA_VERSION = "1.0"
SERVER_PATHS_SCHEMA_VERSION = "1.0"
REPAIR_SCHEMA_VERSION = "1.0"
DIAGNOSTIC_SCHEMA_VERSION = "1.0"
EXPLORATION_SCHEMA_VERSION = "1.0"
MPP_INDEX_SCHEMA_VERSION = "1.0"
JOB_SCHEMA_VERSION = "1.0"
RESULT_SCHEMA_VERSION = "1.0"
MAX_RESULT_FILE_BYTES = 20 * 1024 * 1024
MAX_RESULT_TOTAL_BYTES = 50 * 1024 * 1024
SMOKE_MAX_EPOCHS = 3
MPP_TRAINING_PATH_IDS = {
    "mpp_data_root",
    "mpp_standard_splits",
    "server_mpp_partner_cache",
    "server_mpp_flat_cache",
    "server_mpp_results",
}
MPP_TRAINING_PATH_PARAMETERS = {
    "mpp_root",
    "cache_root",
    "labels_root",
    "manifest_labels_root",
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

DIAGNOSTIC_COMMANDS = {
    "python_help": "Display allowlisted tool help only.",
    "environment_probe": "Collect interpreter, CUDA and disk environment metadata.",
    "path_probe": "Verify configured registered paths without changing them.",
    "cache_probe": "Inspect cache presence and readability without regeneration.",
    "dry_run": "Run an allowlisted command in dry-run mode only.",
    "single_batch_forward": "Run one non-training forward pass with no checkpoint selection.",
}
DIAGNOSTIC_RUNNER_COMMANDS = {"environment_probe"}
MPP_CACHE_PARITY_PATH_IDS = {
    "mpp_data_root",
    "mpp_standard_splits",
    "server_mpp_partner_cache",
    "server_mpp_flat_cache",
    "server_mpp_results",
}
MPP_CACHE_PARITY_PATH_PARAMETERS = {
    "mpp_root",
    "splits_root",
    "manifest_labels_root",
    "cache_root",
    "flat_cache_root",
    "output",
    "resource_release_ack",
    "data_manifest_id",
}
MPP_CACHE_PARITY_ALLOWED_PARAMETERS = {"samples_per_patient", "device"}
MPP_LORA_PATH_IDS = {
    "mpp_data_root",
    "mpp_standard_splits",
    "server_mpp_results",
    "server_mpp2_frozen_baseline_checkpoint",
}
MPP_LORA_PATH_PARAMETERS = {
    "mpp_root",
    "splits_root",
    "manifest_labels_root",
    "head_checkpoint",
    "output_root",
    "data_manifest_id",
}
MPP_LORA_ALLOWED_PARAMETERS = {
    "mode",
    "train_mpp_id",
    "external_mpp_id",
    "external_patient",
    "dataset_name",
    "num_epochs",
    "batch_size",
    "grad_accum_steps",
    "head_lr",
    "lora_lr",
    "weight_decay",
    "patience",
    "min_delta",
    "gradient_clip",
    "seed",
    "num_workers",
    "num_threads",
    "amp",
    "grad_checkpointing",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "head_checkpoint_sha256",
    "hidden_dim",
    "dropout",
}
MPP_PATHWAY_CALIBRATION_PATH_IDS = {
    "mpp_standard_splits",
    "server_mpp_partner_cache",
    "server_mpp_flat_cache",
    "server_mpp_results",
    "server_mpp2_frozen_baseline_checkpoint",
}
MPP_PATHWAY_CALIBRATION_PATH_PARAMETERS = {
    "splits_root",
    "manifest_labels_root",
    "cache_root",
    "flat_cache_root",
    "head_checkpoint",
    "output_dir",
}
MPP_PATHWAY_CALIBRATION_ALLOWED_PARAMETERS: set[str] = set()
ALLOWED_RESULT_FILES = {
    "training_history.csv",
    "training_summary.txt",
    "per_pathway_pcc.csv",
    "best_epoch.txt",
    "metrics.json",
    "cache_parity.csv",
    "cache_parity.json",
    "stderr_tail.txt",
    "stderr_tail.txt.gz",
    "stdout_tail.txt",
    "stdout_tail.txt.gz",
    "calibrator.json",
    "nested_lopo_metrics.json",
    "pathway_decisions.csv",
    "pathway_order.json",
    "zscore_params_from_train.json",
    "model_provenance.json",
    "sha256_manifest.json",
    "README_PHASE3_INPUT_CONTRACT.md",
    "predictions_internal_val_base.csv",
    "predictions_external_xzy.csv",
    "per_pathway_metrics.csv",
    "spatial_sensitivity.json",
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


def validate_against_schema_strict(instance: Any, schema_path: Path, label: str) -> str:
    """Validate a complete JSON Schema without requiring a Python package install."""
    force_powershell = os.environ.get("PFMVAL_FORCE_POWERSHELL_SCHEMA") == "1"
    if not force_powershell and validate_against_schema(instance, schema_path, label):
        return "python-jsonschema"

    powershell = shutil.which("pwsh")
    if powershell is None:
        raise RuntimeError(
            f"{label} requires jsonschema or PowerShell 7 Test-Json; neither is available"
        )

    validator_script = r"""
param(
    [Parameter(Mandatory = $true)][string]$InstancePath,
    [Parameter(Mandatory = $true)][string]$SchemaPath
)
$ErrorActionPreference = 'Stop'
$SchemaErrors = @()
$Payload = Get-Content -LiteralPath $InstancePath -Raw -Encoding UTF8
$IsValid = $Payload | Test-Json -SchemaFile $SchemaPath `
    -ErrorAction SilentlyContinue -ErrorVariable +SchemaErrors
if (-not $IsValid) {
    $Detail = ($SchemaErrors | Out-String).Trim()
    if (-not $Detail) { $Detail = 'Test-Json returned false' }
    [Console]::Error.WriteLine($Detail)
    exit 2
}
"""
    with tempfile.TemporaryDirectory(prefix="pfmval-schema-") as temp_dir:
        temp_root = Path(temp_dir)
        instance_path = temp_root / "instance.json"
        script_path = temp_root / "validate-schema.ps1"
        instance_path.write_text(
            json.dumps(instance, ensure_ascii=False),
            encoding="utf-8",
            newline="\n",
        )
        script_path.write_text(validator_script, encoding="utf-8", newline="\n")
        completed = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-File",
                str(script_path),
                "-InstancePath",
                str(instance_path),
                "-SchemaPath",
                str(schema_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    if completed.returncode != 0:
        detail = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or f"PowerShell exit {completed.returncode}"
        )
        raise ValueError(
            f"{label} schema violation via PowerShell Test-Json: {detail}"
        )
    return "powershell-test-json"


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


def _safe_branch_name(value: str, *, field: str) -> str:
    branch = value.strip()
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", branch)
        or ".." in branch
        or "//" in branch
        or "@{" in branch
        or branch.endswith((".", "/", ".lock"))
    ):
        raise ValueError(f"unsafe {field}: {value}")
    return branch


def _append_jsonl_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(event), ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def create_diagnostic_request(
    root: Path,
    *,
    diagnostic_id: str,
    source_commit: str,
    command_id: str,
    source_branch: str,
    return_branch: str,
) -> Path:
    """Create a non-executable, allowlisted diagnostic request and audit event.

    A server-side runner may interpret only ``command_id``.  The request has no
    arbitrary command text, experiment ID or training parameters by design.
    """
    if not re.fullmatch(r"diagnostic-[0-9]{8}-[A-Za-z0-9_.-]+", diagnostic_id):
        raise ValueError("diagnostic_id must match diagnostic-YYYYMMDD-name")
    if command_id not in DIAGNOSTIC_COMMANDS:
        raise ValueError(f"diagnostic command is not allowlisted: {command_id}")
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", source_commit) or not git_commit_exists(root, source_commit):
        raise ValueError("diagnostic source_commit must name an existing local Git commit")
    source_branch = _safe_branch_name(source_branch, field="source_branch")
    return_branch = _safe_branch_name(return_branch, field="return_branch")
    if not return_branch.startswith("automation/diagnostics"):
        raise ValueError("diagnostic return_branch must be rooted at automation/diagnostics")

    report = validate_diagnostic_state(root)
    if not report.ok:
        raise ValueError("diagnostic request blocked by safety-boundary validation: " + "; ".join(report.fail_items))
    request_dir = root / "automation" / "diagnostics" / diagnostic_id
    request_path = request_dir / "request.json"
    if request_path.exists():
        raise ValueError(f"diagnostic request already exists: {normalize_rel(request_path.relative_to(root))}")
    request = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_id": diagnostic_id,
        "created_at": utc_now(),
        "source_commit": source_commit,
        "source_branch": source_branch,
        "return_branch": return_branch,
        "command_id": command_id,
        "command_description": DIAGNOSTIC_COMMANDS[command_id],
        "execution_contract": {
            "arbitrary_shell": False,
            "training": False,
            "experiment_registry_write": False,
            "current_state_write": False,
            "protected_asset_write": False,
            "result_import": False,
            "output_root": normalize_rel(request_dir.relative_to(root)),
        },
    }
    operation_cards = build_diagnostic_operation_cards(root, request)
    write_json_atomic(request_path, request)
    operation_cards_path = request_dir / "operation_cards.md"
    write_text_atomic(operation_cards_path, operation_cards)
    _append_jsonl_event(root / "project_state" / "diagnostics.jsonl", {
        "event_type": "diagnostic_request",
        "diagnostic_id": diagnostic_id,
        "recorded_at": utc_now(),
        "source_commit": source_commit,
        "source_branch": source_branch,
        "return_branch": return_branch,
        "command_id": command_id,
        "request_path": normalize_rel(request_path.relative_to(root)),
        "operation_cards_path": normalize_rel(operation_cards_path.relative_to(root)),
    })
    return request_path


def _safe_project_relative_path(root: Path, value: str, *, field: str) -> Path:
    normalized = normalize_rel(value).strip("/")
    relative = PurePosixPath(normalized)
    if not normalized or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe {field}: {value}")
    candidate = (root / Path(*relative.parts)).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"{field} escapes project root: {value}")
    return candidate


def _read_jsonl_events(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    events: List[Dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(event, dict):
            raise ValueError(f"JSONL event must be an object at {path}:{line_number}")
        events.append(event)
    return events


def _powershell_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_diagnostic_operation_cards(root: Path, request: Mapping[str, Any]) -> str:
    """Build copy-ready Gitee diagnostic cards from the registered server paths."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to build diagnostic operation cards")
    registry_path = root / "configs" / "server_paths.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    paths = registry.get("paths", {}) if isinstance(registry, dict) else {}
    repo_entry = paths.get("server_repo_worktree", {})
    automation_entry = paths.get("server_automation_worktrees", {})
    server_repo = str(repo_entry.get("path", "")).strip()
    automation_root = str(automation_entry.get("path", "")).strip()
    if not server_repo or not automation_root:
        raise ValueError("diagnostic cards require server_repo_worktree and server_automation_worktrees")

    diagnostic_id = str(request["diagnostic_id"])
    source_branch = str(request["source_branch"])
    source_commit = str(request["source_commit"])
    return_branch = str(request["return_branch"])
    current_branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    if not current_branch:
        current_branch = "<governance-branch>"

    repo_ps = _powershell_literal(server_repo)
    automation_ps = _powershell_literal(automation_root)
    diagnostic_ps = _powershell_literal(diagnostic_id)
    governance_ps = _powershell_literal(current_branch)
    source_branch_ps = _powershell_literal(source_branch)
    source_commit_ps = _powershell_literal(source_commit)
    return_branch_ps = _powershell_literal(return_branch)
    output_rel = f"automation/diagnostics/{diagnostic_id}/environment_probe.json"
    if request["command_id"] not in DIAGNOSTIC_RUNNER_COMMANDS:
        return f"""# {diagnostic_id} 操作卡

> `{request["command_id"]}` 已在请求 allowlist 中，但当前没有固定 runner。
> 固定传输通道：Gitee；不得把请求解释为任意 shell 授权。

## 已解析参数

- 治理请求分支：`{current_branch}`
- 源码分支：`{source_branch}`
- 源码提交：`{source_commit}`
- 回传分支：`{return_branch}`
- 服务器仓库：`{server_repo}`
- 自动化工作树根：`{automation_root}`

## 卡 1：本地发布请求

```powershell
git push gitee HEAD:{current_branch}
git push gitee {source_branch}:{source_branch}
```

## 卡 2：服务器受限执行与回传

`BLOCKED`：当前版本没有 `{request["command_id"]}` 的固定收集器；请另开治理任务实现并测试，不得手写服务器命令替代。

## 卡 3：本地取回与验证

`NOT APPLICABLE`：没有固定 runner 输出时，不得登记完成事件。
"""

    return f"""# {diagnostic_id} 操作卡

> 仅适用于 allowlist 中的 `{request["command_id"]}` 诊断；不是训练、结果导入或实验结论。
> 固定传输通道：Gitee。输出必须为 UTF-8（无 BOM）且使用 LF 换行。

## 已解析参数

- 治理请求分支：`{current_branch}`
- 源码分支：`{source_branch}`
- 源码提交：`{source_commit}`
- 回传分支：`{return_branch}`
- 服务器仓库：`{server_repo}`
- 自动化工作树根：`{automation_root}`

## 卡 1：本地发布请求

```powershell
git push gitee HEAD:{current_branch}
git push gitee {source_branch}:{source_branch}
```

停止条件：任一 push 失败，或远端源码分支未包含 `{source_commit}`。

## 卡 2：服务器受限执行与回传

```powershell
$repo = {repo_ps}
$automationRoot = {automation_ps}
$diagnosticId = {diagnostic_ps}
$governanceBranch = {governance_ps}
$sourceBranch = {source_branch_ps}
$sourceCommit = {source_commit_ps}
$returnBranch = {return_branch_ps}
$requestWorktree = Join-Path $automationRoot ($diagnosticId + '-request')

git -C $repo fetch gitee `
  ('+refs/heads/' + $governanceBranch + ':refs/remotes/gitee/' + $governanceBranch) `
  ('+refs/heads/' + $sourceBranch + ':refs/remotes/gitee/' + $sourceBranch)
if ((git -C $repo rev-parse ('gitee/' + $sourceBranch)) -ne $sourceCommit) {{ throw 'source SHA mismatch' }}
git -C $repo worktree add --detach $requestWorktree ('gitee/' + $governanceBranch)
python (Join-Path $requestWorktree 'deploy/pfmval_ops.py') agent start-check --task diagnostic --host-scope server
python (Join-Path $requestWorktree 'deploy/pfmval_ops.py') diagnostic run-allowlisted --diagnostic-id $diagnosticId
git -C $requestWorktree switch -C $returnBranch
git -C $requestWorktree add -- {output_rel}
git -C $requestWorktree commit -m ('diagnostic: return ' + $diagnosticId)
git -C $requestWorktree push gitee ('HEAD:' + $returnBranch)
```

停止条件：诊断门禁失败、源码 SHA 不一致、工作树非干净状态、输出字节契约失败，或 push 失败。

## 卡 3：本地取回与验证

```powershell
git fetch gitee +refs/heads/{return_branch}:refs/remotes/gitee/{return_branch}
git restore --source gitee/{return_branch} -- {output_rel}
python deploy/pfmval_ops.py diagnostic record --diagnostic-id {diagnostic_id} --output {output_rel}
python deploy/pfmval_ops.py agent start-check --strict
```

验收条件：`diagnostic record` 返回哈希且严格门禁 `FAIL=0`；不得创建 result envelope，不得消耗 run unit。
"""


def _assert_utf8_lf_json(path: Path) -> Dict[str, Any]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"diagnostic JSON must be UTF-8 without BOM: {path.name}")
    if b"\r" in raw:
        raise ValueError(f"diagnostic JSON must use LF line endings: {path.name}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"diagnostic JSON is not valid UTF-8: {path.name}") from exc
    if not text.endswith("\n"):
        raise ValueError(f"diagnostic JSON must end with LF: {path.name}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"diagnostic JSON is invalid: {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"diagnostic JSON root must be an object: {path.name}")
    return payload


def _diagnostic_git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _registered_server_path(root: Path, path_id: str) -> Path:
    if yaml is None:
        raise RuntimeError("PyYAML is required to resolve diagnostic worktrees")
    registry = yaml.safe_load((root / "configs" / "server_paths.yaml").read_text(encoding="utf-8"))
    entry = registry.get("paths", {}).get(path_id, {}) if isinstance(registry, dict) else {}
    raw = str(entry.get("path", "")).strip()
    if not raw:
        raise ValueError(f"registered server path is missing: {path_id}")
    path = Path(raw)
    return path if path.is_absolute() else root / path


def _prepare_diagnostic_source_worktree(root: Path, request: Mapping[str, Any]) -> Path:
    automation_root = _registered_server_path(root, "server_automation_worktrees").resolve()
    source_worktree = (automation_root / f'{request["diagnostic_id"]}-source').resolve()
    if not source_worktree.is_relative_to(automation_root):
        raise ValueError("diagnostic source worktree escapes the registered automation root")
    source_commit = str(request["source_commit"])
    if source_worktree.exists():
        if _diagnostic_git_output(source_worktree, "rev-parse", "HEAD") != source_commit:
            raise ValueError("existing diagnostic source worktree has the wrong commit")
    else:
        source_worktree.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(source_worktree), source_commit],
            cwd=root,
            check=True,
        )
    if _diagnostic_git_output(source_worktree, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("diagnostic source worktree must be clean")
    if _diagnostic_git_output(source_worktree, "branch", "--show-current"):
        raise ValueError("diagnostic source worktree must use detached HEAD")
    return source_worktree


def _collect_environment_probe(source_worktree: Path, request: Mapping[str, Any]) -> Dict[str, Any]:
    torch_info: Dict[str, Any]
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        torch_info = {
            "torch_version": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "cuda_available": cuda_available,
            "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
            "device_name": str(torch.cuda.get_device_name(0)) if cuda_available else None,
        }
    except Exception as exc:  # pragma: no cover - depends on server environment
        torch_info = {
            "available": False,
            "error_type": type(exc).__name__,
        }
    usage = shutil.disk_usage(source_worktree)
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_id": request["diagnostic_id"],
        "command_id": request["command_id"],
        "source_commit": request["source_commit"],
        "source_branch": request["source_branch"],
        "detached_head": True,
        "executed_at": utc_now(),
        "exit_code": 0,
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "platform": platform.platform(),
        },
        "pytorch_cuda": torch_info,
        "environment": {
            "source_worktree": str(source_worktree),
            "disk_total_bytes": usage.total,
            "disk_free_bytes": usage.free,
        },
    }


def run_allowlisted_diagnostic(root: Path, *, diagnostic_id: str) -> Dict[str, Any]:
    """Execute one fixed diagnostic collector; arbitrary commands are impossible."""
    if not re.fullmatch(r"diagnostic-[0-9]{8}-[A-Za-z0-9_.-]+", diagnostic_id):
        raise ValueError("diagnostic_id must match diagnostic-YYYYMMDD-name")
    request_dir = root / "automation" / "diagnostics" / diagnostic_id
    request_path = request_dir / "request.json"
    if not request_path.is_file():
        raise ValueError(f"diagnostic request is missing: {request_path}")
    request = read_json(request_path)
    command_id = str(request.get("command_id", ""))
    if command_id not in DIAGNOSTIC_RUNNER_COMMANDS:
        raise ValueError(f"diagnostic command has no fixed runner: {command_id}")
    contract = request.get("execution_contract", {})
    forbidden = (
        "arbitrary_shell",
        "training",
        "experiment_registry_write",
        "current_state_write",
        "protected_asset_write",
        "result_import",
    )
    if any(contract.get(key) is not False for key in forbidden):
        raise ValueError("diagnostic execution contract permits a forbidden action")
    report = validate_diagnostic_state(root)
    if not report.ok:
        raise ValueError("diagnostic runner blocked by safety-boundary validation: " + "; ".join(report.fail_items))

    source_worktree = _prepare_diagnostic_source_worktree(root, request)
    if _diagnostic_git_output(source_worktree, "rev-parse", "HEAD") != request["source_commit"]:
        raise ValueError("diagnostic source SHA mismatch")
    payload = _collect_environment_probe(source_worktree, request)
    output_path = request_dir / "environment_probe.json"
    write_json_atomic(output_path, payload)
    checked = _assert_utf8_lf_json(output_path)
    if checked.get("source_commit") != request["source_commit"] or checked.get("exit_code") != 0:
        raise ValueError("environment probe output does not match the request")
    return {
        "diagnostic_id": diagnostic_id,
        "command_id": command_id,
        "source_worktree": str(source_worktree),
        "output": {
            "path": normalize_rel(output_path.relative_to(root)),
            "size_bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "encoding": "utf-8",
            "bom": False,
            "eol": "lf",
        },
    }


def create_exploration_session(root: Path, *, session_id: str, purpose: str) -> Path:
    if not re.fullmatch(r"explore-[0-9]{8}-[A-Za-z0-9_.-]+", session_id):
        raise ValueError("session_id must match explore-YYYYMMDD-name")
    if not purpose.strip():
        raise ValueError("exploration purpose must not be empty")
    output_dir = root / "experiments" / "explorations" / session_id
    manifest_path = output_dir / "manifest.json"
    if output_dir.exists():
        raise ValueError(f"exploration session already exists: {normalize_rel(output_dir.relative_to(root))}")
    manifest = {
        "schema_version": EXPLORATION_SCHEMA_VERSION,
        "session_id": session_id,
        "created_at": utc_now(),
        "purpose": purpose.strip(),
        "scope": "local_explore_only",
        "restrictions": {
            "server_execution": False,
            "training_data": False,
            "comparable_metrics": False,
            "experiment_registry_write": False,
            "protected_asset_write": False,
        },
    }
    write_json_atomic(manifest_path, manifest)
    _append_jsonl_event(root / "project_state" / "exploration_log.jsonl", {
        "event_type": "exploration_created",
        "session_id": session_id,
        "recorded_at": utc_now(),
        "purpose": purpose.strip(),
        "manifest_path": normalize_rel(manifest_path.relative_to(root)),
    })
    return manifest_path


def record_diagnostic_outputs(
    root: Path,
    *,
    diagnostic_id: str,
    outputs: Sequence[str],
) -> Dict[str, Any]:
    if not re.fullmatch(r"diagnostic-[0-9]{8}-[A-Za-z0-9_.-]+", diagnostic_id):
        raise ValueError("diagnostic_id must match diagnostic-YYYYMMDD-name")
    request_dir = (root / "automation" / "diagnostics" / diagnostic_id).resolve()
    request_path = request_dir / "request.json"
    if not request_path.is_file():
        raise ValueError(f"diagnostic request is missing: {normalize_rel(request_path.relative_to(root))}")
    request = read_json(request_path)
    if request.get("diagnostic_id") != diagnostic_id or request.get("command_id") not in DIAGNOSTIC_COMMANDS:
        raise ValueError("diagnostic request is malformed or not allowlisted")
    if not outputs:
        raise ValueError("diagnostic completion requires at least one output file")
    if request.get("command_id") == "environment_probe":
        expected = f"automation/diagnostics/{diagnostic_id}/environment_probe.json"
        normalized_outputs = [normalize_rel(item) for item in outputs]
        if normalized_outputs != [expected]:
            raise ValueError("environment_probe requires exactly its canonical environment_probe.json output")
    artifacts: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw_output in outputs:
        output_path = _safe_project_relative_path(root, raw_output, field="diagnostic output")
        if not output_path.is_relative_to(request_dir) or output_path == request_path:
            raise ValueError("diagnostic output must be a returned file beneath its diagnostic output root")
        if not output_path.is_file():
            raise ValueError(f"diagnostic output is missing: {raw_output}")
        relative = normalize_rel(output_path.relative_to(root))
        if relative in seen:
            raise ValueError(f"duplicate diagnostic output: {relative}")
        seen.add(relative)
        size_bytes = output_path.stat().st_size
        if size_bytes > MAX_RESULT_FILE_BYTES:
            raise ValueError(f"diagnostic output exceeds {MAX_RESULT_FILE_BYTES} bytes: {relative}")
        artifact = {"path": relative, "size_bytes": size_bytes, "sha256": sha256_file(output_path)}
        if request.get("command_id") == "environment_probe":
            payload = _assert_utf8_lf_json(output_path)
            if (
                payload.get("diagnostic_id") != diagnostic_id
                or payload.get("source_commit") != request.get("source_commit")
                or payload.get("exit_code") != 0
            ):
                raise ValueError("environment_probe output identity or exit_code does not match the request")
            artifact["byte_contract"] = {"encoding": "utf-8", "bom": False, "eol": "lf"}
        artifacts.append(artifact)
    event = {
        "event_type": "diagnostic_completed",
        "diagnostic_id": diagnostic_id,
        "recorded_at": utc_now(),
        "source_commit": request["source_commit"],
        "command_id": request["command_id"],
        "server_write": False,
        "outputs": artifacts,
    }
    _append_jsonl_event(root / "project_state" / "diagnostics.jsonl", event)
    return event


def list_exploration_sessions(root: Path) -> List[Dict[str, Any]]:
    sessions: Dict[str, Dict[str, Any]] = {}
    for event in _read_jsonl_events(root / "project_state" / "exploration_log.jsonl"):
        session_id = event.get("session_id")
        if not session_id:
            continue
        current = sessions.setdefault(str(session_id), {"session_id": str(session_id), "status": "unknown"})
        current.update(event)
        if event.get("event_type") == "exploration_created":
            current["status"] = "active"
        elif event.get("event_type") == "exploration_promoted":
            current["status"] = "promoted"
    return [sessions[key] for key in sorted(sessions)]


def exploration_cleanup_candidates(root: Path, *, older_than_days: int) -> List[Dict[str, Any]]:
    if older_than_days < 1:
        raise ValueError("older_than_days must be at least 1")
    now = datetime.now(timezone.utc)
    candidates: List[Dict[str, Any]] = []
    for session in list_exploration_sessions(root):
        if session.get("status") != "active":
            continue
        try:
            created_at = datetime.fromisoformat(str(session["recorded_at"]).replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        age_days = (now - created_at.astimezone(timezone.utc)).days
        if age_days >= older_than_days:
            candidates.append({
                "session_id": session["session_id"],
                "age_days": age_days,
                "manifest_path": session.get("manifest_path"),
                "action": "review_only_no_deletion",
            })
    return candidates


def promote_exploration_script(
    root: Path,
    *,
    source: str,
    target: str,
    directive_id: str,
) -> Path:
    directives = active_directives(root)
    if directive_id not in directives:
        raise ValueError("exploration promotion requires an active explicit directive")
    source_path = _safe_project_relative_path(root, source, field="exploration source")
    target_path = _safe_project_relative_path(root, target, field="promotion target")
    exploration_root = (root / "scripts" / "explorations").resolve()
    scripts_root = (root / "scripts").resolve()
    if not source_path.is_relative_to(exploration_root):
        raise ValueError("exploration source must be under scripts/explorations")
    if not target_path.is_relative_to(scripts_root) or target_path.is_relative_to(exploration_root):
        raise ValueError("promotion target must be under scripts/ but outside scripts/explorations")
    if not source_path.is_file():
        raise ValueError(f"exploration source is missing: {source}")
    if target_path.exists():
        raise ValueError(f"promotion target already exists: {target}")
    source_text = source_path.read_text(encoding="utf-8")
    if "PFMVAL_EXPLORE" not in source_text:
        raise ValueError("exploration source lacks required PFMVAL_EXPLORE marker")
    target_text = "\n".join(line for line in source_text.splitlines() if "PFMVAL_EXPLORE" not in line) + "\n"
    write_text_atomic(target_path, target_text)
    _append_jsonl_event(root / "project_state" / "exploration_log.jsonl", {
        "event_type": "exploration_promoted",
        "session_id": None,
        "recorded_at": utc_now(),
        "directive_id": directive_id,
        "source": normalize_rel(source_path.relative_to(root)),
        "target": normalize_rel(target_path.relative_to(root)),
        "note": "candidate_only_requires_commit_registry_and_smoke_or_formal_dispatch",
    })
    return target_path


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


def _git_output(root: Path, args: Sequence[str], *, text: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True,
        text=text, encoding="utf-8" if text else None,
    )
    return completed.stdout


def _safe_evidence_path(value: str) -> str:
    normalized = normalize_rel(value).strip("/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe repair evidence path: {value}")
    return normalized


def _validate_repair_evidence_payload(
    root: Path,
    *,
    evidence_commit: str,
    evidence_path: str,
    audit_raw: bytes,
    summary_raw: bytes,
    expected_audit_sha256: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    try:
        audit = json.loads(audit_raw.decode("utf-8-sig"))
        summary = json.loads(summary_raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"repair evidence JSON is invalid: {exc}") from exc
    actual_audit_sha = sha256_bytes(audit_raw)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_audit_sha256.lower()):
        raise ValueError("expected audit SHA-256 is invalid")
    if actual_audit_sha != expected_audit_sha256.lower() or summary.get("audit_sha256") != actual_audit_sha:
        raise ValueError(
            "repair evidence audit SHA-256 mismatch: "
            f"expected={expected_audit_sha256} summary={summary.get('audit_sha256')} actual={actual_audit_sha}"
        )
    required_summary = {
        "schema_version", "status", "source_commit", "staging_version", "server_stage_path",
        "audit_sha256", "generated_asset_count", "source_asset_count",
        "patient_barcode_one_to_one", "dataset_labels_unique", "training_gate",
    }
    missing_summary = sorted(required_summary - set(summary))
    if missing_summary:
        raise ValueError(f"repair evidence summary missing fields: {missing_summary}")
    if summary.get("schema_version") != REPAIR_SCHEMA_VERSION or audit.get("schema_version") != REPAIR_SCHEMA_VERSION:
        raise ValueError("repair evidence schema_version must be 1.0")
    if summary.get("status") != "repair_staging_verified":
        raise ValueError("repair evidence summary is not verified")
    if (
        summary.get("patient_barcode_one_to_one") is not True
        or summary.get("dataset_labels_unique") is not True
        or summary.get("training_gate") != "blocked_pending_evidence_import_and_explicit_gate_release"
    ):
        raise ValueError("repair evidence summary validation flags are not satisfied")
    if audit.get("server_transport") != "gitee_only":
        raise ValueError("repair evidence transport is not gitee_only")
    if audit.get("mpp_ids") != [1, 2, 3, 4, 5]:
        raise ValueError("repair evidence must cover MPP1-5")
    for field in ("source_commit", "staging_version"):
        if audit.get(field) != summary.get(field):
            raise ValueError(f"repair evidence audit/summary mismatch for {field}")
    source_commit = str(audit.get("source_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit) or not git_commit_exists(root, source_commit):
        raise ValueError("repair evidence source_commit is unavailable")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, evidence_commit],
        cwd=root, check=False, capture_output=True,
    ).returncode != 0:
        raise ValueError("repair evidence commit is not descended from source_commit")
    changed_paths = set(str(_git_output(
        root,
        ["diff", "--name-only", source_commit, evidence_commit],
        text=True,
    )).splitlines())
    allowed_evidence_paths = {
        f"{evidence_path}/server_asset_audit_manifest.json",
        f"{evidence_path}/server_verification.json",
    }
    if changed_paths != allowed_evidence_paths:
        raise ValueError(f"repair evidence branch contains non-evidence changes: {sorted(changed_paths)}")
    server_stage_path = str(summary.get("server_stage_path", ""))
    if not re.match(r"^[A-Za-z]:[\\/]", server_stage_path):
        raise ValueError("repair evidence server_stage_path is not absolute")
    generated = audit.get("generated_assets")
    sources = audit.get("source_assets")
    if not isinstance(generated, list) or not isinstance(sources, list):
        raise ValueError("repair evidence assets must be lists")
    if len(generated) != int(summary["generated_asset_count"]) or len(sources) != int(summary["source_asset_count"]):
        raise ValueError("repair evidence asset counts do not match summary")
    generated_paths = [str(item.get("path", "")) for item in generated]
    source_paths = [str(item.get("path", "")) for item in sources]
    if len(set(generated_paths)) != len(generated_paths) or len(set(source_paths)) != len(source_paths):
        raise ValueError("repair evidence contains duplicate asset paths")
    for asset in [*generated, *sources]:
        if not re.fullmatch(r"[0-9a-f]{64}", str(asset.get("sha256", "")).lower()):
            raise ValueError(f"repair evidence asset has invalid SHA-256: {asset.get('path')}")
        if int(asset.get("size_bytes", -1)) < 0:
            raise ValueError(f"repair evidence asset has invalid size: {asset.get('path')}")
    labels = [item for item in generated if item.get("role") == "standardized_label"]
    if not labels:
        raise ValueError("repair evidence contains no standardized labels")
    for label in labels:
        if (
            int(label.get("duplicate_barcode_count", -1)) != 0
            or int(label.get("row_count", -1)) != int(label.get("unique_barcode_count", -2))
        ):
            raise ValueError(f"repair evidence label uniqueness failed: {label.get('path')}")
    label_groups = {
        int(match.group(1))
        for item in labels
        if (match := re.match(r"group_([1-5])/", str(item.get("path", ""))))
    }
    if label_groups != {1, 2, 3, 4, 5}:
        raise ValueError(f"repair evidence labels do not cover MPP1-5: {sorted(label_groups)}")
    validation = audit.get("validation", {})
    required_validation = {
        "patient_barcode_one_to_one": True,
        "dataset_labels_unique": True,
        "existing_mpp_assets_modified": False,
        "published_from_versioned_staging": True,
    }
    if any(validation.get(key) is not expected for key, expected in required_validation.items()):
        raise ValueError("repair evidence validation flags are not all satisfied")
    if audit.get("training_gate", {}).get("status") != "blocked_pending_evidence_import_and_explicit_gate_release":
        raise ValueError("repair evidence was produced without the required training gate")
    split_groups = set()
    for asset in sources:
        normalized = str(asset.get("path", "")).replace("\\", "/")
        marker = "mpp_standard_splits/"
        if marker not in normalized:
            continue
        relative = marker + normalized.split(marker, 1)[1]
        match = re.fullmatch(r"mpp_standard_splits/group_([1-5])/split_manifest\.csv", relative)
        if not match:
            continue
        blob = _git_output(root, ["show", f"{source_commit}:{relative}"])
        if not isinstance(blob, bytes):
            raise AssertionError("git blob read returned text unexpectedly")
        if len(blob) != int(asset["size_bytes"]) or sha256_bytes(blob) != asset["sha256"]:
            raise ValueError(f"repair evidence split manifest mismatch: {relative}")
        split_groups.add(int(match.group(1)))
    if split_groups != {1, 2, 3, 4, 5}:
        raise ValueError(f"repair evidence split manifests are incomplete: {sorted(split_groups)}")
    data_manifest_id = f"{audit['staging_version']}:{actual_audit_sha[:16]}"
    record = {
        "evidence_id": audit["staging_version"],
        "data_manifest_id": data_manifest_id,
        "status": "verified_pending_gate_release",
        "evidence_commit": evidence_commit,
        "evidence_path": evidence_path,
        "source_commit": source_commit,
        "audit_sha256": actual_audit_sha,
        "server_stage_path": server_stage_path,
        "generated_asset_count": len(generated),
        "source_asset_count": len(sources),
        "label_asset_count": len(labels),
        "imported_at": utc_now(),
    }
    return audit, summary, record


def import_mpp_repair_evidence_from_git(
    root: Path,
    *,
    git_ref: str,
    evidence_path: str,
    expected_audit_sha256: str,
) -> Dict[str, Any]:
    safe_path = _safe_evidence_path(evidence_path)
    evidence_commit = str(_git_output(root, ["rev-parse", f"{git_ref}^{{commit}}"], text=True)).strip()
    tree_paths = str(_git_output(root, ["ls-tree", "-r", "--name-only", evidence_commit, "--", safe_path], text=True)).splitlines()
    expected_paths = {
        f"{safe_path}/server_asset_audit_manifest.json",
        f"{safe_path}/server_verification.json",
    }
    if set(tree_paths) != expected_paths:
        raise ValueError(f"repair evidence directory must contain exactly audit and summary: {tree_paths}")
    audit_raw = _git_output(root, ["show", f"{evidence_commit}:{safe_path}/server_asset_audit_manifest.json"])
    summary_raw = _git_output(root, ["show", f"{evidence_commit}:{safe_path}/server_verification.json"])
    if not isinstance(audit_raw, bytes) or not isinstance(summary_raw, bytes):
        raise AssertionError("git evidence blob read returned text unexpectedly")
    audit, summary, record = _validate_repair_evidence_payload(
        root,
        evidence_commit=evidence_commit,
        evidence_path=safe_path,
        audit_raw=audit_raw,
        summary_raw=summary_raw,
        expected_audit_sha256=expected_audit_sha256,
    )
    registry_path = root / "project_state" / "mpp_repair_registry.json"
    with state_lock(root):
        registry = read_json(registry_path) if registry_path.exists() else {
            "schema_version": REPAIR_SCHEMA_VERSION,
            "updated_at": utc_now(),
            "active_data_manifest_id": None,
            "repairs": [],
        }
        existing = next((item for item in registry.get("repairs", []) if item.get("evidence_id") == record["evidence_id"]), None)
        if existing:
            if existing.get("audit_sha256") != record["audit_sha256"]:
                raise ValueError("repair evidence id is already bound to a different audit")
            return {**existing, "evidence_status": existing["status"], "status": "already_imported"}
        destination = root / "project_state" / "evidence" / "mpp" / record["evidence_id"]
        if destination.exists() and any(destination.iterdir()):
            raise ValueError(f"repair evidence destination is not empty: {destination}")
        write_text_atomic(destination / "server_asset_audit_manifest.json", audit_raw.decode("utf-8-sig"))
        write_text_atomic(destination / "server_verification.json", summary_raw.decode("utf-8-sig"))
        if sha256_file(destination / "server_asset_audit_manifest.json") != record["audit_sha256"]:
            raise ValueError("canonical repair audit hash changed during local import")
        registry["repairs"].append(record)
        registry["updated_at"] = utc_now()
        write_json_atomic(registry_path, registry)
        state_path = root / "project_state" / "current_state.json"
        state = read_json(state_path)
        state["mpp_repair"] = {
            "verified_evidence_id": record["evidence_id"],
            "verified_data_manifest_id": record["data_manifest_id"],
            "active_data_manifest_id": None,
            "status": "verified_pending_explicit_gate_release",
        }
        write_json_atomic(state_path, state)
        sync_state(root, force_revision=True)
        write_json_atomic(root / "project_state" / "document_registry.json", scan_documents(root))
        sync_state(root)
    return {**record, "evidence_status": record["status"], "status": "imported"}


def active_mpp_repair(root: Path) -> Optional[Dict[str, Any]]:
    registry_path = root / "project_state" / "mpp_repair_registry.json"
    state_path = root / "project_state" / "current_state.json"
    if not registry_path.exists() or not state_path.exists():
        return None
    registry = read_json(registry_path)
    state = read_json(state_path)
    active_id = (state.get("mpp_repair") or {}).get("active_data_manifest_id")
    if not active_id or registry.get("active_data_manifest_id") != active_id:
        return None
    record = next(
        (item for item in registry.get("repairs", []) if item.get("data_manifest_id") == active_id and item.get("status") == "active"),
        None,
    )
    if record is None:
        return None
    canonical = (
        root / "project_state" / "evidence" / "mpp" / str(record.get("evidence_id"))
        / "server_asset_audit_manifest.json"
    )
    if not canonical.is_file() or sha256_file(canonical) != record.get("audit_sha256"):
        return None
    return record


def verify_mpp_repair_server_assets(root: Path, record: Mapping[str, Any]) -> Dict[str, Any]:
    evidence_id = str(record.get("evidence_id", ""))
    canonical_audit = (
        root / "project_state" / "evidence" / "mpp" / evidence_id
        / "server_asset_audit_manifest.json"
    )
    if not canonical_audit.is_file():
        raise ValueError(f"canonical MPP repair audit is missing: {canonical_audit}")
    expected_audit_sha = str(record.get("audit_sha256", "")).lower()
    if sha256_file(canonical_audit) != expected_audit_sha:
        raise ValueError("canonical MPP repair audit hash mismatch")
    stage = Path(str(record.get("server_stage_path", "")))
    if not stage.is_dir():
        raise ValueError(f"MPP repaired-label staging directory is missing: {stage}")
    stage_audit = stage / "server_asset_audit_manifest.json"
    audit = read_json(canonical_audit)
    if not stage_audit.is_file() or read_json(stage_audit) != audit:
        raise ValueError("server staging audit content does not match canonical evidence")
    stage_resolved = stage.resolve()
    verified = 0
    for asset in audit.get("generated_assets", []):
        raw_path = str(asset.get("path", ""))
        relative = PurePosixPath(raw_path)
        if not raw_path or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe repaired-label asset path: {raw_path}")
        path = stage.joinpath(*relative.parts).resolve()
        if not path.is_relative_to(stage_resolved) or not path.is_file():
            raise ValueError(f"repaired-label asset is missing: {raw_path}")
        if path.stat().st_size != int(asset.get("size_bytes", -1)):
            raise ValueError(f"repaired-label asset size mismatch: {raw_path}")
        if sha256_file(path) != str(asset.get("sha256", "")).lower():
            raise ValueError(f"repaired-label asset hash mismatch: {raw_path}")
        verified += 1
    if verified == 0:
        raise ValueError("MPP repair audit contains no generated assets")
    return {"verified_generated_assets": verified, "audit_sha256": expected_audit_sha}


def activate_mpp_repair_evidence(
    root: Path,
    *,
    evidence_id: str,
    directive_id: str,
) -> Dict[str, Any]:
    directives = active_directives(root)
    directive = directives.get(directive_id)
    if not directive or (
        directive.get("scope") != "mpp_data"
        or directive.get("topic") != "barcode_repair_gate_release"
        or directive.get("source") != "explicit_user_instruction"
    ):
        raise ValueError("activation requires an active explicit gate-release directive")
    registry_path = root / "project_state" / "mpp_repair_registry.json"
    if not registry_path.exists():
        raise ValueError("MPP repair evidence registry is missing")
    with state_lock(root):
        registry = read_json(registry_path)
        record = next((item for item in registry.get("repairs", []) if item.get("evidence_id") == evidence_id), None)
        if not record:
            raise ValueError(f"MPP repair evidence is not imported: {evidence_id}")
        if record.get("status") not in {"verified_pending_gate_release", "active"}:
            raise ValueError(f"MPP repair evidence is not activatable: {record.get('status')}")
        state_path = root / "project_state" / "current_state.json"
        state = read_json(state_path)
        repair_state = state.get("mpp_repair") or {}
        if repair_state.get("verified_evidence_id") != evidence_id:
            raise ValueError("current state does not bind the requested repair evidence")
        record["status"] = "active"
        record["activated_at"] = utc_now()
        record["gate_release_directive_id"] = directive_id
        registry["active_data_manifest_id"] = record["data_manifest_id"]
        registry["updated_at"] = utc_now()
        write_json_atomic(registry_path, registry)
        repair_state.update({
            "active_data_manifest_id": record["data_manifest_id"],
            "status": "active",
            "gate_release_directive_id": directive_id,
        })
        state["mpp_repair"] = repair_state
        state["blocked_actions"] = [
            item for item in state.get("blocked_actions", [])
            if item != "new_mpp_training_until_conflicting_duplicate_barcodes_are_resolved"
        ]
        notes = list(state.get("notes", []))
        note = f"Active MPP repaired-label data manifest: {record['data_manifest_id']}"
        if note not in notes:
            notes.append(note)
        state["notes"] = notes
        write_json_atomic(state_path, state)
        sync_state(root, force_revision=True)
        write_json_atomic(root / "project_state" / "document_registry.json", scan_documents(root))
        sync_state(root)
    return {**record}


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
            for field in ("changed_at", "reason", "completion_evidence"):
                if field in event:
                    directives[directive_id][field] = event[field]
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
    related_experiment_ids: Sequence[str] = (),
    review_after: Optional[str] = None,
    completion_evidence: Sequence[str] = (),
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
    if review_after:
        try:
            datetime.fromisoformat(review_after.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("review_after must be an ISO-8601 date or datetime") from exc
        new_event["review_after"] = review_after
    if related_experiment_ids:
        new_event["related_experiment_ids"] = [str(item).strip() for item in related_experiment_ids if str(item).strip()]
    if completion_evidence:
        new_event["completion_evidence"] = [str(item).strip() for item in completion_evidence if str(item).strip()]
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


def transition_directive(
    root: Path,
    *,
    directive_id: str,
    status: str,
    reason: str,
    completion_evidence: Sequence[str] = (),
    superseded_by: Optional[str] = None,
) -> Dict[str, Any]:
    if status not in {"completed", "superseded", "cancelled"}:
        raise ValueError("directive transition status must be completed, superseded or cancelled")
    if not reason.strip():
        raise ValueError("directive transition requires a reason")
    events = read_directive_events(root)
    directives = fold_directives(events)
    directive = directives.get(directive_id)
    if directive is None:
        raise ValueError(f"unknown directive: {directive_id}")
    if directive.get("status") != "active":
        raise ValueError(f"only active directives can transition: {directive_id}")
    event: Dict[str, Any] = {
        "event_type": "status_update",
        "directive_id": directive_id,
        "status": status,
        "changed_at": utc_now(),
        "reason": reason.strip(),
        "source": "explicit_user_instruction",
    }
    if completion_evidence:
        event["completion_evidence"] = [str(item).strip() for item in completion_evidence if str(item).strip()]
    if status == "superseded":
        if not superseded_by or superseded_by == directive_id or superseded_by not in directives:
            raise ValueError("superseded transition requires another existing directive via superseded_by")
        event["superseded_by"] = superseded_by
    elif superseded_by:
        raise ValueError("superseded_by is only valid for a superseded transition")
    _append_jsonl_event(root / "project_state" / "directives.jsonl", event)
    return event


def directive_lifecycle_candidates(root: Path, *, older_than_days: int) -> List[Dict[str, Any]]:
    """Return review candidates only; this function never changes directive status."""
    if older_than_days < 1:
        raise ValueError("older_than_days must be at least 1")
    now = datetime.now(timezone.utc)
    experiments = {
        str(item.get("id")): item
        for item in read_json(root / "experiments" / "experiment_registry.json").get("experiments", [])
        if item.get("id")
    }
    terminal_statuses = {"completed", "closed", "done", "failed", "cancelled", "rejected"}
    candidates: List[Dict[str, Any]] = []
    for directive_id, directive in active_directives(root).items():
        reasons: List[str] = []
        try:
            issued_at = datetime.fromisoformat(str(directive["issued_at"]).replace("Z", "+00:00"))
            if (now - issued_at.astimezone(timezone.utc)).days >= older_than_days:
                reasons.append("age_threshold")
        except (KeyError, ValueError):
            reasons.append("issued_at_unparseable")
        review_after = directive.get("review_after")
        if review_after:
            try:
                review_at = datetime.fromisoformat(str(review_after).replace("Z", "+00:00"))
                if review_at.astimezone(timezone.utc) <= now:
                    reasons.append("review_after_due")
            except ValueError:
                reasons.append("review_after_unparseable")
        related_ids = [str(item) for item in directive.get("related_experiment_ids", [])]
        if related_ids and all(
            experiment_id in experiments and str(experiments[experiment_id].get("status", "")).lower() in terminal_statuses
            for experiment_id in related_ids
        ):
            reasons.append("related_experiments_terminal")
        if reasons:
            candidates.append({
                "directive_id": directive_id,
                "reasons": reasons,
                "review_after": review_after,
                "related_experiment_ids": related_ids,
                "action": "review_only_no_automatic_transition",
            })
    return candidates


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
    if path.startswith(".agents/skills/"):
        return "Agent Skill"
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
    canonical_skill_paths: Dict[str, str] = {}
    adapter_skill_paths: Dict[str, str] = {}
    for skill_name, skill in state.get("active_skills", {}).items():
        canonical_path = normalize_rel(str(skill.get("canonical_path", "")))
        if canonical_path:
            canonical_skill_paths[canonical_path] = str(skill_name)
        for adapter_path in skill.get("adapter_paths", []):
            normalized_adapter = normalize_rel(str(adapter_path))
            if normalized_adapter:
                adapter_skill_paths[normalized_adapter] = str(skill_name)
    pending_plan_paths = {
        normalize_rel(str(item.get("path", ""))): (str(review_id), str(item.get("status", "pending_review")))
        for review_id, item in state.get("pending_plan_reviews", {}).items()
        if item.get("path")
    }
    connectivity: List[str] = []
    if re.search(r"ssh|scp", lower):
        connectivity.append("ssh")
    if "tunnel" in lower:
        connectivity.append("remote_tunnel")
    if "cmd_server" in lower or "远程命令" in path:
        connectivity.append("http_remote_command")

    if path in active_plan_paths:
        return active_plan_paths[path], "normative", "active", connectivity
    if path in canonical_skill_paths:
        return f"skill:{canonical_skill_paths[path]}", "normative", "active", connectivity
    if path in adapter_skill_paths:
        return f"skill_adapter:{adapter_skill_paths[path]}", "reference", "active", connectivity
    if path in pending_plan_paths:
        review_id, status = pending_plan_paths[path]
        lifecycle = "approved_design" if status == "approved_design" else "pending_review"
        return f"plan_review:{review_id}", "reference", lifecycle, connectivity
    if path == "AGENTS.md":
        return "agent_entry", "normative", "active", connectivity
    if path == "CLAUDE.md":
        return "agent_adapter", "reference", "active", connectivity
    if path == "automation/README.md":
        return "gitee_job_protocol", "normative", "active", connectivity
    if path in {
        "CURRENT_STATE.md",
        "README.md",
        "PROJECT_GUIDE.md",
        "experiments/experiment_dashboard.md",
        "experiments/experiment_progress.md",
        ".claude/next-steps.md",
        ".claude/session-brief.md",
    }:
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
    old_registry: Dict[str, Any] = {}
    old_entries: Dict[str, Dict[str, Any]] = {}
    if old_path.exists():
        old_registry = read_json(old_path)
        old_entries = {
            normalize_rel(item["path"]): item
            for item in old_registry.get("documents", [])
        }

    live_paths: set[str] = set()
    for directory in (
        "01_指南与解读", "02_组会汇报", "project_state/plans", "deploy",
        "automation", ".agents/skills", ".claude/skills",
    ):
        base = root / directory
        if base.exists():
            live_paths.update(normalize_rel(path.relative_to(root)) for path in base.rglob("*.md"))
    for path in (
        "README.md", "PROJECT_GUIDE.md", "AGENTS.md", "CURRENT_STATE.md", "CLAUDE.md",
        ".claude/next-steps.md", ".claude/session-brief.md", ".claude/maintenance-plan.md",
        "experiments/experiment_dashboard.md", "experiments/experiment_progress.md",
        "experiments/decision_log.md",
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
            "supersedes": list(old.get("supersedes", [])),
            "superseded_by": list(old.get("superseded_by", [])),
            "truth_sources": list(
                old.get(
                    "truth_sources",
                    (
                        [
                            "project_state/document_registry.json",
                            "project_state/asset_registry.json",
                        ]
                        if rel_path == "PROJECT_GUIDE.md"
                        else (
                            ["experiments/experiment_registry.json"]
                            if rel_path
                            in {
                                "experiments/experiment_dashboard.md",
                                "experiments/experiment_progress.md",
                            }
                            else ["project_state/current_state.json"]
                        )
                    ),
                )
            ),
            "purpose": old.get("purpose", ""),
            "created": old.get("created", ""),
            "tags": list(old.get("tags", [])),
            "connectivity_modes": connectivity,
            "availability": (
                "tracked"
                if not (
                    rel_path == "CLAUDE.md"
                    or (
                        rel_path.startswith(".claude/")
                        and not (
                            rel_path.startswith(".claude/skills/")
                            and rel_path.endswith("/SKILL.md")
                        )
                    )
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
        if old:
            stable_fields = {
                key: value
                for key, value in entry.items()
                if key not in {"verified_at", "state_revision"}
            }
            old_stable_fields = {
                key: value
                for key, value in old.items()
                if key not in {"verified_at", "state_revision"}
            }
            if stable_fields == old_stable_fields:
                entry["verified_at"] = old.get("verified_at", verified_at)
                entry["state_revision"] = int(
                    old.get("state_revision", state["state_revision"])
                )
        documents.append(entry)

    registry = {
        "schema_version": DOCUMENT_REGISTRY_SCHEMA_VERSION,
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
            "pending_review": sum(1 for item in documents if item["lifecycle"] == "pending_review"),
            "approved_design": sum(1 for item in documents if item["lifecycle"] == "approved_design"),
        },
    }
    if old_registry:
        stable_registry = {
            key: value
            for key, value in registry.items()
            if key not in {"updated_at", "state_revision"}
        }
        old_stable_registry = {
            key: value
            for key, value in old_registry.items()
            if key not in {"updated_at", "state_revision"}
        }
        if stable_registry == old_stable_registry:
            registry["updated_at"] = old_registry.get("updated_at", verified_at)
            registry["state_revision"] = int(
                old_registry.get("state_revision", state["state_revision"])
            )
    return registry


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
        "schema_version": MPP_INDEX_SCHEMA_VERSION,
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
            "path_index_version": MPP_INDEX_SCHEMA_VERSION if experiment_id in standardized_ids else None,
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
    repair = state.get("mpp_repair")
    if repair:
        lines.extend([
            "",
            "## MPP repair evidence",
            "",
            f"- Verified evidence: `{repair.get('verified_evidence_id')}`.",
            f"- Verified data manifest: `{repair.get('verified_data_manifest_id')}`.",
            f"- Active data manifest: `{repair.get('active_data_manifest_id') or 'none'}`.",
            f"- Gate status: **{repair.get('status', 'unknown')}**.",
        ])
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
        "- 用户导航：[PROJECT_GUIDE.md](PROJECT_GUIDE.md)；简洁实验进度：[experiments/experiment_progress.md](experiments/experiment_progress.md)。",
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
        "experiment_progress_sha256": root / "experiments" / "experiment_progress.md",
        "document_registry_sha256": root / "project_state" / "document_registry.json",
        "asset_registry_sha256": root / "project_state" / "asset_registry.json",
        "workspace_registry_sha256": root / "project_state" / "workspace_registry.json",
        "workflow_catalog_sha256": root / "project_state" / "workflow_catalog.json",
        "mpp_path_index_sha256": root / "mpp_standard_splits" / "path_index.json",
        "server_paths_sha256": root / "configs" / "server_paths.yaml",
        "mpp_repair_registry_sha256": root / "project_state" / "mpp_repair_registry.json",
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


def validate_server_paths(
    root: Path,
    report: ValidationReport,
    *,
    task: str = "general",
    host_scope: Optional[str] = None,
) -> None:
    try:
        registry = load_server_paths(root)
    except Exception as exc:
        report.fail(f"server path registry unreadable: {exc}")
        return
    if registry.get("schema_version") != SERVER_PATHS_SCHEMA_VERSION:
        report.fail("server path registry schema_version must be 1.0")
    paths = registry.get("paths")
    if not isinstance(paths, dict) or not paths:
        report.fail("server path registry contains no paths")
        return
    if host_scope is None:
        # Backward-compatible alias: task=server historically meant a server
        # host check. New callers should pass host_scope explicitly.
        host_scope = "server" if task == "server" else "local"
    if host_scope not in {"local", "server"}:
        report.fail(f"invalid host scope: {host_scope}")
        return
    required_scopes = {"local", "both"} if host_scope == "local" else {"server", "both"}

    for path_id, entry in paths.items():
        if entry.get("status") not in {"active", "legacy", "deprecated"}:
            report.fail(f"path {path_id} has invalid status")
        if not entry.get("path"):
            report.fail(f"path {path_id} has no path value")
        if entry.get("required_on") in required_scopes and not re.match(r"^[A-Za-z]:[\\/]", str(entry["path"])):
            scoped_path = root / str(entry["path"])
            if not scoped_path.exists():
                report.fail(f"required {host_scope} path missing: {path_id} -> {entry['path']}")
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
            repair = active_mpp_repair(root) if task == "training" else None
            if task == "training" and repair is None:
                report.fail(message)
            elif task == "training":
                report.warn(
                    "legacy standardized labels remain conflicted; training is bound to active repaired staging "
                    f"{repair['data_manifest_id']}"
                )
            else:
                report.warn(message)
        else:
            report.passed("MPP label and embargo index is validated")
    else:
        report.warn("mpp_standard_splits/path_index.json is missing")


def validate_governance_v3_state(
    root: Path,
    report: ValidationReport,
    *,
    host_scope: Optional[str] = None,
) -> None:
    schema_root = root / "project_state" / "schemas"
    workspace_path = root / "project_state" / "workspace_registry.json"
    if not workspace_path.exists():
        report.fail("workflow governance v3 workspace registry is missing")
        return
    try:
        workspace_registry = read_json(workspace_path)
        validate_against_schema(
            workspace_registry,
            schema_root / "workspace_registry.schema.json",
            "workspace registry",
        )
        workspace_ids: set[str] = set()
        experiment_ids: set[str] = set()
        workspace_numbers: list[int] = []
        for workspace in workspace_registry.get("workspaces", []):
            workspace_id = str(workspace.get("workspace_id", ""))
            match = re.fullmatch(r"W([0-9]{3,})", workspace_id)
            if not match:
                raise ValueError(
                    f"invalid workspace registry id: {workspace_id}"
                )
            if workspace_id in workspace_ids:
                raise ValueError(
                    f"duplicate workspace registry id: {workspace_id}"
                )
            experiment_id = str(workspace.get("experiment_id", ""))
            if experiment_id in experiment_ids:
                raise ValueError(
                    "one experiment is bound to multiple workspaces: "
                    f"{experiment_id}"
                )
            workspace_ids.add(workspace_id)
            experiment_ids.add(experiment_id)
            workspace_numbers.append(int(match.group(1)))
            for host in workspace.get("hosts", {}).values():
                relative_path = Path(str(host.get("relative_path", "")))
                if relative_path.is_absolute():
                    if host.get("host_scope") != "local":
                        raise ValueError("only local workspace bindings may use absolute paths")
                    # A server-side Gitee archive is intentionally not a local
                    # Git worktree.  Its integrity is verified by job run-v2;
                    # this structural validator must not query a local-host
                    # workspace path while validating the server package.
                    if host_scope == "server":
                        continue
                    expected_branch = str(host.get("branch", ""))
                    expected_commit = str(host.get("current_source_commit", ""))
                    completed = subprocess.run(
                        ["git", "-C", str(root), "worktree", "list", "--porcelain"],
                        check=True,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                    )
                    records: list[Dict[str, str]] = []
                    current: Dict[str, str] = {}
                    for line in completed.stdout.splitlines():
                        if not line:
                            if current:
                                records.append(current)
                                current = {}
                            continue
                        key, _, value = line.partition(" ")
                        if key == "worktree":
                            current["path"] = value
                        elif key == "HEAD":
                            current["head"] = value
                        elif key == "branch":
                            current["branch"] = value.removeprefix("refs/heads/")
                    if current:
                        records.append(current)
                    registered = relative_path.resolve(strict=True)
                    matched = next(
                        (
                            item
                            for item in records
                            if Path(item.get("path", "")).resolve(strict=True) == registered
                        ),
                        None,
                    )
                    if matched is None:
                        raise ValueError(f"absolute workspace is not a local Git worktree: {registered}")
                    if matched.get("branch") != expected_branch:
                        raise ValueError("absolute workspace branch does not match registry")
                    if matched.get("head") != expected_commit:
                        raise ValueError("absolute workspace HEAD does not match registry")
                elif ".." in relative_path.parts:
                    raise ValueError(
                        "workspace path escapes registered root: "
                        f"{relative_path}"
                    )
        next_number = int(workspace_registry.get("next_workspace_number", 0))
        if next_number <= max(workspace_numbers, default=0):
            raise ValueError(
                "workspace next number would reuse an allocated identity"
            )

        asset_registry = read_json(
            root / "project_state" / "asset_registry.json"
        )
        validate_against_schema(
            asset_registry,
            schema_root / "asset_registry.schema.json",
            "asset registry",
        )
        asset_ids: set[str] = set()
        asset_paths: set[str] = set()
        for asset in asset_registry.get("assets", []):
            asset_id = str(asset.get("asset_id", ""))
            asset_path = normalize_rel(str(asset.get("path", ""))).lower()
            if asset_id in asset_ids:
                raise ValueError(f"duplicate asset id: {asset_id}")
            if asset_path in asset_paths:
                raise ValueError(f"duplicate asset path: {asset_path}")
            if Path(asset_path).is_absolute() or ".." in Path(asset_path).parts:
                raise ValueError(f"asset path escapes repository: {asset_path}")
            asset_ids.add(asset_id)
            asset_paths.add(asset_path)

        for index, approval in enumerate(
            _read_jsonl_events(
                root / "project_state" / "experiment_approvals.jsonl"
            ),
            1,
        ):
            validate_against_schema(
                approval,
                schema_root / "experiment_approval_v2.schema.json",
                f"experiment approval {index}",
            )
        for index, event in enumerate(
            _read_jsonl_events(
                root / "project_state" / "attempt_events.jsonl"
            ),
            1,
        ):
            validate_against_schema(
                event,
                schema_root / "attempt_event_v2.schema.json",
                f"attempt event {index}",
            )
        for index, record in enumerate(
            _read_jsonl_events(
                root / "project_state" / "scientific_records.jsonl"
            ),
            1,
        ):
            validate_against_schema(
                record,
                schema_root / "scientific_record.schema.json",
                f"scientific record {index}",
            )
        for index, fact in enumerate(
            _read_jsonl_events(
                root / "project_state" / "project_facts.jsonl"
            ),
            1,
        ):
            validate_against_schema(
                fact,
                schema_root / "project_fact.schema.json",
                f"project fact {index}",
            )
        workflow_catalog_path = (
            root / "project_state" / "workflow_catalog.json"
        )
        workflow_catalog_schema = (
            schema_root / "workflow_catalog.schema.json"
        )
        if workflow_catalog_path.exists() != workflow_catalog_schema.exists():
            raise ValueError(
                "workflow catalog and schema must either both exist or both be absent"
            )
        if workflow_catalog_path.exists():
            workflow_catalog = read_json(workflow_catalog_path)
            validate_against_schema(
                workflow_catalog,
                workflow_catalog_schema,
                "workflow catalog",
            )
            workflow_ids = [
                str(item.get("workflow_id", ""))
                for item in workflow_catalog.get("entries", [])
            ]
            if len(workflow_ids) != len(set(workflow_ids)):
                raise ValueError("duplicate workflow catalog id")
            known_workflow_ids = set(workflow_ids)
            for item in workflow_catalog.get("entries", []):
                workflow_id = str(item.get("workflow_id", ""))
                if item.get("standardized") and item.get("lifecycle") != "active":
                    raise ValueError(
                        "standardized workflow is not active: "
                        f"{workflow_id}"
                    )
                missing_replacements = sorted(
                    set(item.get("replacement_workflow_ids", []))
                    - known_workflow_ids
                )
                if missing_replacements:
                    raise ValueError(
                        "workflow replacement is not registered: "
                        f"{workflow_id} -> {missing_replacements}"
                    )
    except (FileNotFoundError, ValueError, TypeError) as exc:
        report.fail(f"workflow governance v3 state invalid: {exc}")
        return
    report.passed(
        "workflow governance v3 workspace/asset/approval/attempt/science/fact/workflow state validates"
    )


def validate_state(
    root: Path,
    *,
    strict: bool = False,
    task: str = "general",
    host_scope: Optional[str] = None,
) -> ValidationReport:
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
    elif state.get("schema_version") != STATE_SCHEMA_VERSION:
        report.fail("current_state schema_version must be 1.0")
    else:
        report.passed("current_state required fields")

    repair_state = state.get("mpp_repair")
    repair_registry_path = root / "project_state" / "mpp_repair_registry.json"
    if repair_state:
        if not repair_registry_path.exists():
            report.fail("current_state references MPP repair evidence but registry is missing")
        else:
            repair_registry = read_json(repair_registry_path)
            try:
                validate_against_schema(
                    repair_registry,
                    schema_root / "mpp_repair_registry.schema.json",
                    "MPP repair registry",
                )
            except (FileNotFoundError, ValueError) as exc:
                report.fail(str(exc))
            verified_id = repair_state.get("verified_evidence_id")
            verified = next(
                (item for item in repair_registry.get("repairs", []) if item.get("evidence_id") == verified_id),
                None,
            )
            if not verified:
                report.fail("current_state verified MPP repair evidence is absent from registry")
            elif repair_state.get("verified_data_manifest_id") != verified.get("data_manifest_id"):
                report.fail("current_state MPP repair data manifest does not match registry")
            else:
                canonical_audit = (
                    root / "project_state" / "evidence" / "mpp" / str(verified.get("evidence_id"))
                    / "server_asset_audit_manifest.json"
                )
                if not canonical_audit.is_file() or sha256_file(canonical_audit) != verified.get("audit_sha256"):
                    report.fail("canonical MPP repair audit hash does not match registry")
                else:
                    report.passed("MPP repair evidence registry matches current state")

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
    doc_by_path = {
        normalize_rel(str(item.get("path", ""))).lower(): item
        for item in document_list
    }
    for skill_name, skill in state.get("active_skills", {}).items():
        canonical_path = normalize_rel(str(skill.get("canonical_path", "")))
        canonical = doc_by_path.get(canonical_path.lower())
        if not canonical:
            report.fail(f"active skill {skill_name} canonical path is absent from document registry")
        elif (
            canonical.get("lifecycle") != "active"
            or canonical.get("authority") != "normative"
            or canonical.get("scope") != f"skill:{skill_name}"
        ):
            report.fail(f"active skill {skill_name} canonical document classification is invalid")
        for adapter_path in skill.get("adapter_paths", []):
            normalized_adapter = normalize_rel(str(adapter_path))
            adapter = doc_by_path.get(normalized_adapter.lower())
            if not adapter:
                report.fail(f"active skill {skill_name} adapter path is absent from document registry: {normalized_adapter}")
            elif (
                adapter.get("lifecycle") != "active"
                or adapter.get("authority") != "reference"
                or adapter.get("scope") != f"skill_adapter:{skill_name}"
            ):
                report.fail(f"active skill {skill_name} adapter classification is invalid: {normalized_adapter}")
    for review_id, review in state.get("pending_plan_reviews", {}).items():
        review_path = normalize_rel(str(review.get("path", "")))
        document = doc_by_path.get(review_path.lower())
        expected_lifecycle = "approved_design" if review.get("status") == "approved_design" else "pending_review"
        if not document:
            report.fail(f"plan review {review_id} is absent from document registry")
        elif document.get("lifecycle") != expected_lifecycle:
            report.fail(f"plan review {review_id} lifecycle does not match current state")
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

    validate_server_paths(root, report, task=task, host_scope=host_scope)
    validate_governance_v3_state(root, report, host_scope=host_scope)
    if not report.fail_items:
        report.passed("state package validation completed")
    return report


def validate_diagnostic_state(root: Path) -> ValidationReport:
    """Validate only the safety boundary required for a non-evidence diagnostic.

    This deliberately does not call :func:`validate_state`: document freshness,
    experiment provenance and MPP training-path checks are not prerequisites for
    a read-only, allowlisted server diagnostic.  It still treats unreadable state,
    invalid schemas, unfinished transactions and non-Gitee transport as blockers.
    """
    report = ValidationReport()
    try:
        state = read_json(root / "project_state" / "current_state.json")
        documents = read_json(root / "project_state" / "document_registry.json")
        events = read_directive_events(root)
        directives = fold_directives(events)
    except Exception as exc:
        report.fail(f"diagnostic state package unreadable: {exc}")
        return report

    schema_root = root / "project_state" / "schemas"
    try:
        checked = (
            validate_against_schema(state, schema_root / "current_state.schema.json", "current_state")
            and validate_against_schema(documents, schema_root / "document_registry.schema.json", "document_registry")
            and all(
                validate_against_schema(event, schema_root / "directives.schema.json", f"directive event {index}")
                for index, event in enumerate(events, 1)
            )
        )
        if checked:
            report.passed("diagnostic state schemas validate")
        else:
            report.warn("jsonschema package unavailable; diagnostic state uses manual validation")
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
    elif state.get("schema_version") != STATE_SCHEMA_VERSION:
        report.fail("current_state schema_version must be 1.0")
    else:
        report.passed("diagnostic current_state required fields")

    if state.get("active_directive_ids") != [directive_id for directive_id, value in directives.items() if value.get("status") == "active"]:
        report.fail("current_state active_directive_ids does not match folded directive log")
    else:
        report.passed("diagnostic directive index matches append-only log")

    topics: Dict[Tuple[str, str], List[str]] = {}
    for directive_id, directive in directives.items():
        if directive.get("status") == "active":
            topics.setdefault((str(directive.get("scope", "")), str(directive.get("topic", ""))), []).append(directive_id)
    conflicts = {key: ids for key, ids in topics.items() if len(ids) > 1}
    if conflicts:
        report.fail(f"unresolved active directive conflicts: {conflicts}")

    transport = state.get("server_transport", {})
    if transport.get("mode") != "gitee_only":
        report.fail("server transport is not gitee_only")
    elif not transport.get("remote_name"):
        report.fail("server transport has no configured remote name")
    else:
        report.passed("diagnostic transport remains Gitee-only")
    forbidden = set(transport.get("forbidden_direct_connections", []))
    for document in documents.get("documents", []):
        if document.get("lifecycle") == "active" and forbidden.intersection(document.get("connectivity_modes", [])):
            report.fail(f"active document conflicts with Gitee-only transport: {document.get('path')}")

    transaction_root = root / "project_state" / ".transactions"
    if transaction_root.exists() and any(transaction_root.iterdir()):
        report.fail("unfinished state transaction exists")
    if (root / "project_state" / ".state.lock").exists():
        report.fail("state lock exists; a writer may have been interrupted")

    # These checks remain visible to the operator but do not block diagnostic-only work.
    if int(state.get("state_revision", 0)) - int(documents.get("state_revision", 0)) > 1:
        report.warn("document registry is more than one state revision behind; diagnostic only")
    current_view = root / "CURRENT_STATE.md"
    if not current_view.exists():
        report.warn("CURRENT_STATE.md missing; diagnostic only")
    elif sha256_bytes(canonical_json_bytes(state)) not in current_view.read_text(encoding="utf-8"):
        report.warn("CURRENT_STATE.md is stale; diagnostic only")
    if not report.fail_items:
        report.passed("diagnostic safety boundary validation completed")
    return report


def validate_result_envelope(bundle_dir: Path, manifest: Mapping[str, Any]) -> None:
    project_root = Path(__file__).resolve().parent.parent
    if manifest.get("schema_version") == "2.0":
        from scripts.pfmval_result_bundle import validate_result_bundle_v1

        validate_result_bundle_v1(project_root, bundle_dir)
        return

    validate_against_schema(
        manifest,
        project_root / "project_state" / "schemas" / "result_envelope.schema.json",
        "result envelope",
    )
    required = {"schema_version", "result_id", "job_id", "experiment_id", "source_commit", "phase", "status", "created_at", "artifacts", "metrics"}
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"result envelope missing fields: {', '.join(missing)}")
    if manifest["schema_version"] != RESULT_SCHEMA_VERSION:
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


def _validate_job_v2_governance_binding(
    root: Path,
    job: Mapping[str, Any],
) -> None:
    workspace_registry_path = root / "project_state" / "workspace_registry.json"
    if not workspace_registry_path.exists():
        raise ValueError("job v2 has no workspace registry")
    workspace_registry = read_json(workspace_registry_path)
    workspace = next(
        (
            item
            for item in workspace_registry.get("workspaces", [])
            if item.get("workspace_id") == job.get("workspace_id")
        ),
        None,
    )
    if workspace is None:
        raise ValueError("job v2 references an unknown workspace")
    if workspace.get("experiment_id") != job.get("experiment_id"):
        raise ValueError("job v2 workspace/experiment binding mismatch")

    registry = read_json(root / "experiments" / "experiment_registry.json")
    experiment = next(
        (
            item
            for item in registry.get("experiments", [])
            if item.get("id") == job.get("experiment_id")
        ),
        None,
    )
    if experiment is None:
        raise ValueError("job v2 references an unknown experiment")
    experiment_comparisons = {
        "workspace_id": job.get("workspace_id"),
        "protocol_revision": job.get("protocol_revision"),
        "critical_contract_sha256": job.get("critical_contract_sha256"),
    }
    for field, expected in experiment_comparisons.items():
        if experiment.get(field) != expected:
            raise ValueError(f"job v2 experiment binding mismatch for {field}")

    approvals = _read_jsonl_events(
        root / "project_state" / "experiment_approvals.jsonl"
    )
    approval = next(
        (
            item
            for item in approvals
            if item.get("approval_id") == job.get("approval_id")
            and item.get("status") == "active"
        ),
        None,
    )
    if approval is None:
        raise ValueError("job v2 references a missing or inactive approval")
    approval_comparisons = {
        "experiment_id": job.get("experiment_id"),
        "protocol_revision": job.get("protocol_revision"),
        "phase": job.get("phase"),
        "critical_contract_sha256": job.get("critical_contract_sha256"),
    }
    for field, expected in approval_comparisons.items():
        if approval.get(field) != expected:
            raise ValueError(f"job v2 approval binding mismatch for {field}")

    attempts = _read_jsonl_events(
        root / "project_state" / "attempt_events.jsonl"
    )
    prepared = next(
        (
            item
            for item in attempts
            if item.get("event_type") == "ATTEMPT_PREPARED"
            and item.get("workspace_id") == job.get("workspace_id")
            and item.get("attempt_id") == job.get("attempt_id")
        ),
        None,
    )
    if prepared is None:
        raise ValueError("job v2 has no prepared attempt")
    attempt_comparisons = {
        "job_id": job.get("job_id"),
        "experiment_id": job.get("experiment_id"),
        "approval_id": job.get("approval_id"),
        "protocol_revision": job.get("protocol_revision"),
        "phase": job.get("phase"),
        "run_units": job.get("run_units"),
        "critical_contract_sha256": job.get("critical_contract_sha256"),
        "workspace_branch": job.get("workspace_branch"),
    }
    for field, expected in attempt_comparisons.items():
        if prepared.get(field) != expected:
            raise ValueError(f"job v2 attempt binding mismatch for {field}")
    if not job.get("adaptation_record_id"):
        if prepared.get("source_commit") != job.get("source_commit"):
            raise ValueError("job v2 attempt binding mismatch for source_commit")


def validate_result_job_binding(root: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    job_id = str(manifest.get("job_id", ""))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id):
        raise ValueError(f"unsafe result job_id: {job_id}")
    job_path = root / "automation" / "jobs" / job_id / "job.json"
    if not job_path.exists():
        raise ValueError(f"result has no dispatched job envelope: {normalize_rel(job_path.relative_to(root))}")
    job = read_json(job_path)
    if job.get("schema_version") == "2.0":
        from scripts.pfmval_governance import validate_job_v2

        validate_against_schema(
            job,
            root / "project_state" / "schemas" / "server_job_v2.schema.json",
            "bound server job v2",
        )
        validate_job_v2(job)
        comparisons = {
            "job_id": job.get("job_id"),
            "experiment_id": job.get("experiment_id"),
            "workspace_id": job.get("workspace_id"),
            "attempt_id": job.get("attempt_id"),
            "protocol_revision": job.get("protocol_revision"),
            "approval_id": job.get("approval_id"),
            "critical_contract_sha256": job.get(
                "critical_contract_sha256"
            ),
            "source_commit": job.get("source_commit"),
            "phase": job.get("phase"),
            "run_units": job.get("run_units"),
        }
        for field, expected in comparisons.items():
            if manifest.get(field) != expected:
                raise ValueError(
                    f"result/job binding mismatch for {field}: "
                    f"result={manifest.get(field)!r} job={expected!r}"
                )
        _validate_job_v2_governance_binding(root, job)
        if git_head(root) != "unknown" and not git_commit_exists(
            root,
            str(job.get("source_commit", "")),
        ):
            raise ValueError(
                "bound job source_commit is unavailable locally: "
                f"{job.get('source_commit')}"
            )
        return job

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
        "experiment_progress.md": root / "experiments" / "experiment_progress.md",
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
            required_backups = [
                name
                for name in (
                    "experiment_registry.json",
                    "experiment_dashboard.md",
                    "current_state.json",
                )
            ]
            if metadata.get("before", {}).get("progress_sha256"):
                required_backups.append("experiment_progress.md")
            missing = [name for name in required_backups if not (backup_dir / name).exists()]
            if missing:
                raise ValueError(f"unrecoverable transaction backups are missing: {missing}")
            for name, target in targets.items():
                backup = backup_dir / name
                if backup.exists():
                    shutil.copy2(backup, target)
                elif (
                    name == "experiment_progress.md"
                    and target.exists()
                    and not metadata.get("before", {}).get("progress_sha256")
                ):
                    target.unlink()
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
    bound_job = validate_result_job_binding(root, manifest)
    manifest_sha256 = sha256_file(manifest_path)
    registry_path = root / "experiments" / "experiment_registry.json"
    dashboard_path = root / "experiments" / "experiment_dashboard.md"
    progress_path = root / "experiments" / "experiment_progress.md"
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
            existing_record = (
                target
                if target.get("result_id") == manifest["result_id"]
                else target.get("last_preflight") or {}
            )
            if existing_record.get("result_manifest_sha256") != manifest_sha256:
                raise ValueError(
                    "result_id is already bound to a different bundle SHA"
                )
            return {"status": "already_imported", "result_id": manifest["result_id"]}
        if git_head(root) != "unknown" and not git_commit_exists(root, str(manifest["source_commit"])):
            raise ValueError(f"result source_commit is unavailable locally: {manifest['source_commit']}")
        existing_commit = target.get("source_commit")
        if manifest["phase"] != "preflight" and existing_commit and existing_commit != manifest["source_commit"]:
            raise ValueError(f"source_commit mismatch: registry={existing_commit} result={manifest['source_commit']}")
        imported_at = utc_now()
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
            input_binding = (
                bound_job.get("input_binding", {})
                if bound_job.get("schema_version") == "2.0"
                else {}
            )
            target["data_manifest_id"] = (
                manifest.get("data_manifest_id")
                or input_binding.get("data_manifest_id")
            )
            target["path_index_version"] = (
                manifest.get("path_index_version")
                or input_binding.get("path_index_version")
            )
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
        from scripts.pfmval_views import build_experiment_progress

        dashboard_text = build_dashboard(registry)
        progress_text = build_experiment_progress(registry)
        accepted_ids = list(state.get("latest_accepted_result_ids", []))
        if manifest["phase"] != "preflight":
            if target["evidence_status"] == "accepted" and target["id"] not in accepted_ids:
                if manifest["phase"] == "formal":
                    accepted_ids.append(target["id"])
            elif target["evidence_status"] != "accepted":
                accepted_ids = [item for item in accepted_ids if item != target["id"]]
        state["latest_accepted_result_ids"] = accepted_ids
        state["pending_result_ids"] = [item for item in state.get("pending_result_ids", []) if item != manifest["result_id"]]
        update_mpp2_baseline_guard_from_result(state, target, manifest)
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
                "progress_sha256": (
                    sha256_file(progress_path)
                    if progress_path.exists()
                    else None
                ),
                "state_sha256": sha256_file(state_path),
            },
        }
        write_json_atomic(transaction_dir / "transaction.json", transaction_manifest)
        backup_dir = transaction_dir / "backup"
        backup_dir.mkdir()
        shutil.copy2(registry_path, backup_dir / "experiment_registry.json")
        shutil.copy2(dashboard_path, backup_dir / "experiment_dashboard.md")
        if progress_path.exists():
            shutil.copy2(progress_path, backup_dir / "experiment_progress.md")
        shutil.copy2(state_path, backup_dir / "current_state.json")
        if (root / "CURRENT_STATE.md").exists():
            shutil.copy2(root / "CURRENT_STATE.md", backup_dir / "CURRENT_STATE.md")
        write_json_atomic(transaction_dir / "experiment_registry.json", registry)
        write_text_atomic(transaction_dir / "experiment_dashboard.md", dashboard_text)
        write_text_atomic(transaction_dir / "experiment_progress.md", progress_text)

        staged_hashes = compute_source_hashes(root)
        staged_hashes["experiment_registry_sha256"] = sha256_file(transaction_dir / "experiment_registry.json")
        staged_hashes["experiment_dashboard_sha256"] = sha256_file(transaction_dir / "experiment_dashboard.md")
        staged_hashes["experiment_progress_sha256"] = sha256_file(
            transaction_dir / "experiment_progress.md"
        )
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
            os.replace(transaction_dir / "experiment_progress.md", progress_path)
            os.replace(transaction_dir / "current_state.json", state_path)
            os.replace(transaction_dir / "CURRENT_STATE.md", root / "CURRENT_STATE.md")
        except Exception:
            transaction_manifest["status"] = "interrupted"
            write_json_atomic(transaction_dir / "transaction.json", transaction_manifest)
            shutil.copy2(backup_dir / "experiment_registry.json", registry_path)
            shutil.copy2(backup_dir / "experiment_dashboard.md", dashboard_path)
            if (backup_dir / "experiment_progress.md").exists():
                shutil.copy2(
                    backup_dir / "experiment_progress.md",
                    progress_path,
                )
            elif progress_path.exists():
                progress_path.unlink()
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
        # Job keys are the training script's argparse option names. Preserve
        # underscores exactly; silent style conversion previously produced
        # syntactically safe but invalid argv (for example --batch-size when
        # the script declares --batch_size).
        flag = "--" + key
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


def evaluate_mpp2_baseline_guard(
    guard: Mapping[str, Any], metrics: Mapping[str, Any]
) -> Dict[str, Any]:
    """Compare a repaired MPP2 baseline with the frozen historical reference."""
    required = ("external_xzy_pcc", "external_xzy_mae_raw", "external_xzy_r2_raw")
    missing = [key for key in required if key not in metrics]
    if missing:
        return {
            "status": "incomplete_block_lora",
            "missing_metrics": missing,
            "triggered_metrics": [],
        }

    reference = guard["reference_metrics"]
    thresholds = guard["thresholds"]
    observed = {key: float(metrics[key]) for key in required}
    triggered: List[str] = []
    if float(reference["external_xzy_pcc"]) - observed["external_xzy_pcc"] >= float(thresholds["pcc_abs_drop"]):
        triggered.append("external_xzy_pcc")
    mae_relative_increase = (
        observed["external_xzy_mae_raw"] / float(reference["external_xzy_mae_raw"])
    ) - 1.0
    if mae_relative_increase >= float(thresholds["raw_mae_relative_increase"]):
        triggered.append("external_xzy_mae_raw")
    if float(reference["external_xzy_r2_raw"]) - observed["external_xzy_r2_raw"] >= float(thresholds["raw_r2_abs_drop"]):
        triggered.append("external_xzy_r2_raw")
    return {
        "status": "triggered_rerun_mpp1_5" if triggered else "passed_allow_lora",
        "missing_metrics": [],
        "triggered_metrics": triggered,
        "observed_metrics": observed,
        "deltas": {
            "pcc_drop": float(reference["external_xzy_pcc"]) - observed["external_xzy_pcc"],
            "raw_mae_relative_increase": mae_relative_increase,
            "raw_r2_drop": float(reference["external_xzy_r2_raw"]) - observed["external_xzy_r2_raw"],
        },
    }


def update_mpp2_baseline_guard_from_result(
    state: MutableMapping[str, Any],
    experiment: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    if experiment.get("repair_guard_role") != "mpp2_repaired_frozen_baseline":
        return
    repair = state.get("mpp_repair") or {}
    guard = repair.get("baseline_guard")
    if not isinstance(guard, MutableMapping):
        raise ValueError("repaired MPP2 baseline result has no configured baseline guard")
    guard["observed_result_id"] = result.get("result_id")
    guard["evaluated_at"] = utc_now()
    if result.get("status") != "success" or result.get("phase") not in {"smoke", "formal"}:
        evaluation = {
            "status": "incomplete_block_lora",
            "missing_metrics": [],
            "triggered_metrics": [],
            "reason": "baseline result was not a successful training result",
        }
    else:
        evaluation = evaluate_mpp2_baseline_guard(guard, result.get("metrics", {}))
    guard.update(evaluation)
    blocked = list(state.get("blocked_actions", []))
    lora_block = "mpp2_lora_until_repaired_baseline_passes_guard"
    rerun_block = "rerun_repaired_mpp1_5_before_lora"
    if guard["status"] == "passed_allow_lora":
        blocked = [item for item in blocked if item not in {lora_block, rerun_block}]
    else:
        if lora_block not in blocked:
            blocked.append(lora_block)
        if guard["status"] == "triggered_rerun_mpp1_5" and rerun_block not in blocked:
            blocked.append(rerun_block)
    state["blocked_actions"] = blocked
    repair["baseline_guard"] = guard
    state["mpp_repair"] = repair


def validate_mpp_sequence_gate(root: Path, experiment: Mapping[str, Any]) -> None:
    if experiment.get("repair_guard_role") != "mpp2_lora_r8_smoke":
        return
    state = read_json(root / "project_state" / "current_state.json")
    status = (((state.get("mpp_repair") or {}).get("baseline_guard") or {}).get("status"))
    if status != "passed_allow_lora":
        raise ValueError(
            f"MPP2 LoRA is blocked until the repaired frozen baseline passes the drift guard; status={status or 'missing'}"
        )


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
    if command_id == "cache_parity":
        if phase != "preflight":
            raise ValueError("cache_parity command is only valid for phase=preflight")
        script = str((experiment or {}).get("script", "")).replace("\\", "/")
        if not script.endswith("scripts/check_mpp_online_cache_parity.py"):
            raise ValueError("cache_parity requires the registered MPP parity script")
        provided_ids = set(path_ids)
        missing_ids = sorted(MPP_CACHE_PARITY_PATH_IDS - provided_ids)
        if missing_ids:
            raise ValueError(f"cache parity job is missing required path ids: {missing_ids}")
        path_overrides = sorted(MPP_CACHE_PARITY_PATH_PARAMETERS & set(parameters))
        if path_overrides:
            raise ValueError(
                f"cache parity paths and gates are registry-bound and cannot be overridden: {path_overrides}"
            )
        unknown_parameters = sorted(set(parameters) - MPP_CACHE_PARITY_ALLOWED_PARAMETERS)
        if unknown_parameters:
            raise ValueError(f"cache parity parameters are not allowlisted: {unknown_parameters}")
        try:
            sample_count = int(parameters.get("samples_per_patient", 8))
        except (TypeError, ValueError) as exc:
            raise ValueError("samples_per_patient must be an integer") from exc
        if sample_count < 1 or sample_count > 64:
            raise ValueError("samples_per_patient must be between 1 and 64")
        if parameters.get("device", "cuda") not in {"cuda", "cpu"}:
            raise ValueError("cache parity device must be cuda or cpu")
        return
    if command_id == "mpp_pathway_ridge_calibration":
        if phase != "formal":
            raise ValueError("MPP2 pathway Ridge calibration requires phase=formal")
        script = str((experiment or {}).get("script", "")).replace("\\", "/")
        if not script.endswith("scripts/fit_mpp2_pathway_ridge_calibration.py"):
            raise ValueError("MPP2 pathway Ridge calibration requires its registered script")
        if not bool((experiment or {}).get("execution_approved")):
            raise ValueError("MPP2 pathway Ridge calibration is not execution-approved")
        if not (experiment or {}).get("execution_directive_id"):
            raise ValueError("MPP2 pathway Ridge calibration lacks an execution directive binding")
        missing_ids = sorted(MPP_PATHWAY_CALIBRATION_PATH_IDS - set(path_ids))
        if missing_ids:
            raise ValueError(f"MPP2 pathway Ridge calibration is missing required path ids: {missing_ids}")
        overrides = sorted(MPP_PATHWAY_CALIBRATION_PATH_PARAMETERS & set(parameters))
        if overrides:
            raise ValueError(f"MPP2 pathway Ridge calibration paths are registry-bound: {overrides}")
        unknown = sorted(set(parameters) - MPP_PATHWAY_CALIBRATION_ALLOWED_PARAMETERS)
        if unknown:
            raise ValueError(f"MPP2 pathway Ridge calibration parameters are not allowlisted: {unknown}")
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

    script_name = str((experiment or {}).get("script", "")).replace("\\", "/")
    if script_name.endswith("train_mpp_uni2h_lora.py"):
        if phase != "smoke":
            raise ValueError("MPP2 LoRA trainer is currently restricted to phase=smoke")
        provided_ids = set(path_ids)
        missing_ids = sorted(MPP_LORA_PATH_IDS - provided_ids)
        if missing_ids:
            raise ValueError(f"MPP2 LoRA job is missing required path ids: {missing_ids}")
        path_overrides = sorted(MPP_LORA_PATH_PARAMETERS & set(parameters))
        if path_overrides:
            raise ValueError(f"MPP2 LoRA paths are registry-bound and cannot be overridden: {path_overrides}")
        unknown_parameters = sorted(set(parameters) - MPP_LORA_ALLOWED_PARAMETERS)
        if unknown_parameters:
            raise ValueError(f"MPP2 LoRA parameters are not allowlisted: {unknown_parameters}")
        required = {"mode", "dataset_name", "seed", "head_checkpoint_sha256"}
        missing = sorted(required - set(parameters))
        if missing:
            raise ValueError(f"MPP2 LoRA job is missing required parameters: {missing}")
        if parameters.get("mode") not in {"frozen", "lora"}:
            raise ValueError("MPP2 LoRA mode must be frozen or lora")
        if int(parameters.get("seed")) != 42:
            raise ValueError("MPP2 paired smoke is fixed to seed=42")
        if int(parameters.get("train_mpp_id", 2)) != 2 or int(parameters.get("external_mpp_id", 2)) != 2:
            raise ValueError("MPP2 paired smoke is fixed to train_mpp_id=2 and external_mpp_id=2")
        if str(parameters.get("external_patient", "XZY")) != "XZY":
            raise ValueError("MPP2 paired smoke is fixed to external_patient=XZY")
        if int(parameters.get("lora_rank", 8)) != 8 or float(parameters.get("lora_alpha", 16.0)) != 16.0:
            raise ValueError("MPP2 paired smoke is fixed to LoRA rank=8 and alpha=16")
        return

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
    if command_id not in {"state_preflight", "cache_parity", "standard_training", "mpp_pathway_ridge_calibration"}:
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
    validate_mpp_sequence_gate(root, experiment)
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
        "schema_version": JOB_SCHEMA_VERSION,
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


def validate_job_manifest(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    require_head: bool = True,
    source_git_root: Path | None = None,
) -> None:
    """Validate a job envelope and its governance binding.

    ``root`` is always the governance-state root.  A server-side Gitee archive
    is intentionally not a Git worktree, so schema and approval binding remain
    rooted there while source-object availability can be checked against the
    separately pinned source worktree.
    """
    if manifest.get("schema_version") == "2.0":
        from scripts.pfmval_governance import validate_job_v2

        validate_against_schema(
            manifest,
            root / "project_state" / "schemas" / "server_job_v2.schema.json",
            "server job v2",
        )
        validate_job_v2(manifest)
        git_root = source_git_root or root
        if not git_commit_exists(git_root, str(manifest["source_commit"])):
            raise ValueError(
                "job source commit is unavailable locally: "
                f"{manifest['source_commit']}"
            )
        if require_head and manifest["source_commit"] != git_head(git_root):
            raise ValueError(
                f"job source commit {manifest['source_commit']} "
                f"does not match HEAD {git_head(git_root)}"
            )
        _validate_job_v2_governance_binding(root, manifest)
        return

    validate_against_schema(
        manifest,
        root / "project_state" / "schemas" / "server_job.schema.json",
        "server job",
    )
    required = {"schema_version", "job_id", "experiment_id", "source_commit", "state_revision", "phase", "command_id", "path_ids", "parameters", "created_at", "artifact_policy"}
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"job manifest missing fields: {', '.join(missing)}")
    if manifest["schema_version"] != JOB_SCHEMA_VERSION:
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
    validate_mpp_sequence_gate(root, experiment)
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
            "schema_version": RESULT_SCHEMA_VERSION,
            "result_id": result_id,
            "job_id": job["job_id"],
            "experiment_id": job["experiment_id"],
            "source_commit": job["source_commit"],
            "phase": job["phase"],
            "status": status,
            "created_at": utc_now(),
            "formal_training_approved": bool((job.get("formal_training_approval") or {}).get("approved")),
            "data_manifest_id": job.get("data_manifest_id"),
            "path_index_version": job.get("path_index_version") or MPP_INDEX_SCHEMA_VERSION,
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
