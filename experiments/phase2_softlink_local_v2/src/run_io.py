"""Batch directories, per-task run folders, snapshots, logs, and events.

Directory ids are time plus a random identifier. Source/config versions are
human version strings and file copies; this module does not compute hashes.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_batch_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    token = f"{random.randrange(16**8):08x}"
    return f"{stamp}_{token}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def jsonable_config(config: dict) -> dict:
    skip = {"_config_path"}
    return {key: value for key, value in config.items() if key not in skip}


def append_event(path: Path, *, stage: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": utc_now(), "stage": stage, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_batch_dir(runs_root: str | Path, batch_id: str | None = None) -> Path:
    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    batch_dir = root / (batch_id or new_batch_id())
    if batch_dir.exists():
        raise FileExistsError(f"批次目录已存在，拒绝覆盖: {batch_dir}")
    batch_dir.mkdir(parents=True, exist_ok=False)
    return batch_dir


def write_config_snapshot(batch_dir: Path, config: dict) -> Path:
    path = Path(batch_dir) / "config_snapshot.json"
    payload = jsonable_config(config)
    payload["_snapshot_written_at"] = utc_now()
    write_json(path, payload)
    return path


def copy_package_snapshot(dest_dir: Path, package_dir: Path | None = None) -> Path:
    src = (package_dir or PACKAGE_ROOT) / "package.json"
    dest = Path(dest_dir) / "package.json"
    shutil.copy2(src, dest)
    return dest


def create_source_snapshot(batch_dir: Path, package_dir: Path | None = None) -> Path:
    """Freeze executable sources once per batch; traceability uses copies, not hashes."""
    package = package_dir or PACKAGE_ROOT
    dest = Path(batch_dir) / "source_snapshot"
    if dest.exists():
        raise FileExistsError(f"源码快照目录已存在，拒绝覆盖: {dest}")
    dest.mkdir(parents=True)
    (dest / "src").mkdir()
    for path in sorted((package / "src").glob("*.py")):
        shutil.copy2(path, dest / "src" / path.name)
    if (package / "docs").is_dir():
        (dest / "docs").mkdir()
        for path in sorted((package / "docs").glob("*")):
            if path.is_file():
                shutil.copy2(path, dest / "docs" / path.name)
    shutil.copy2(package / "package.json", dest / "package.json")
    return dest


def copy_provenance(run_dir: Path, package_dir: Path | None = None) -> Path:
    package = package_dir or PACKAGE_ROOT
    dest = Path(run_dir) / "provenance"
    dest.mkdir(parents=True, exist_ok=True)
    src_dest = dest / "src"
    src_dest.mkdir(exist_ok=True)
    for path in sorted((package / "src").glob("*.py")):
        shutil.copy2(path, src_dest / path.name)
    docs = package / "docs"
    if docs.is_dir():
        doc_dest = dest / "docs"
        doc_dest.mkdir(exist_ok=True)
        for path in sorted(docs.glob("*.md")):
            shutil.copy2(path, doc_dest / path.name)
        cfg_copy = docs / "Phase2软连接升级_集中配置_v2_1_20260907.json"
        if cfg_copy.is_file():
            shutil.copy2(cfg_copy, doc_dest / cfg_copy.name)
    shutil.copy2(package / "package.json", dest / "package.json")
    write_json(
        dest / "code_sources_note.json",
        {
            "note_zh": "源码与配置用人工 code_version / config_version 和文件副本追溯，不计算哈希。",
            "package_dir": str(package),
            "copied_at": utc_now(),
        },
    )
    return dest


def create_run_dir(batch_dir: Path, run_id: str) -> Path:
    run_dir = Path(batch_dir) / run_id
    if run_dir.exists():
        raise FileExistsError(f"运行目录已存在，拒绝覆盖: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "raw", "provenance"):
        (run_dir / name).mkdir()
    return run_dir


def initial_run_record(
    *,
    package: dict,
    run_id: str,
    run_dir: Path,
    arm: str | None,
    seed: int | None,
    artifact_kind: str,
    entrypoint: str,
    extra: dict | None = None,
) -> dict:
    record = {
        "experiment_id": package.get("experiment_id"),
        "code_version": package.get("code_version"),
        "plan_version": package.get("plan_version"),
        "run_id": run_id,
        "arm": arm,
        "seed": seed,
        "demo_only": bool(package.get("demo_only")),
        "artifact_kind": artifact_kind,
        "status": "starting",
        "started_at": utc_now(),
        "ended_at": None,
        "exit_code": None,
        "run_directory": str(run_dir),
        "entrypoint": entrypoint,
        "resume_supported": False,
        "resume_used": False,
    }
    if extra:
        record.update(extra)
    return record


class _Tee:
    def __init__(self, *files, prefix: str = ""):
        self.files = files
        self.prefix = prefix
        self._lock = threading.Lock()
        self.encoding = "utf-8"
        self.errors = "replace"

    def write(self, data: str) -> int:
        if not data:
            return 0
        text = f"{self.prefix}{data}" if self.prefix else data
        with self._lock:
            for handle in self.files:
                handle.write(text)
                handle.flush()
        return len(data)

    def flush(self) -> None:
        with self._lock:
            for handle in self.files:
                handle.flush()

    def isatty(self) -> bool:
        return False


@contextmanager
def capture_stdio(run_dir: Path):
    logs = Path(run_dir) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    startup = logs / "startup.log"
    startup.write_text(f"Run directory: {Path(run_dir).resolve()}\n", encoding="utf-8")
    stdout_path = logs / "stdout.log"
    stderr_path = logs / "errors.log"
    console_path = logs / "console.log"
    stdout_f = stdout_path.open("w", encoding="utf-8")
    stderr_f = stderr_path.open("w", encoding="utf-8")
    console_f = console_path.open("w", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    # console prefixes mark the source stream; cross-stream order is not a clock.
    sys.stdout = _Tee(stdout_f, console_f, old_out, prefix="")
    # Write raw stderr to errors.log, prefixed copy to console.log, and original stderr.
    sys.stderr = _SplitStderr(stderr_f, console_f, old_err)
    try:
        print(f"Run directory: {Path(run_dir).resolve()}", flush=True)
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        for handle in (stdout_f, stderr_f, console_f):
            handle.close()


class _SplitStderr:
    def __init__(self, errors_file, console_file, original):
        self.errors_file = errors_file
        self.console_file = console_file
        self.original = original
        self._lock = threading.Lock()
        self.encoding = "utf-8"
        self.errors = "replace"

    def write(self, data: str) -> int:
        if not data:
            return 0
        with self._lock:
            self.errors_file.write(data)
            self.errors_file.flush()
            prefixed = data if data.startswith("[stderr]") else "".join(
                f"[stderr] {line}" if line else line for line in data.splitlines(keepends=True)
            )
            if data and not data.endswith("\n") and "[stderr]" not in prefixed:
                prefixed = f"[stderr] {data}"
            self.console_file.write(prefixed)
            self.console_file.flush()
            self.original.write(data)
            self.original.flush()
        return len(data)

    def flush(self) -> None:
        with self._lock:
            self.errors_file.flush()
            self.console_file.flush()
            self.original.flush()

    def isatty(self) -> bool:
        return False


def package_payload(package_dir: Path | None = None) -> dict:
    return read_json((package_dir or PACKAGE_ROOT) / "package.json")


def default_runs_root(config: dict) -> Path:
    runtime = config.get("runtime") or {}
    if runtime.get("runs_root"):
        return Path(runtime["runs_root"])
    raise KeyError("config.runtime.runs_root 未设置")


def default_weights_root(config: dict) -> Path:
    runtime = config.get("runtime") or {}
    if runtime.get("weights_root"):
        return Path(runtime["weights_root"])
    raise KeyError("config.runtime.weights_root 未设置")


def interpreter_from_config(config: dict) -> str | None:
    runtime = config.get("runtime") or {}
    return runtime.get("python_interpreter")


def relative_to_batch(batch_dir: Path, path: Path) -> str:
    return Path(path).resolve().relative_to(Path(batch_dir).resolve()).as_posix()


def copy_tree_if_present(src: Path, dest: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    elif src.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
