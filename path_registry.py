"""Resolve stable path IDs from configs/server_paths.yaml.

Training code should depend on path IDs, not repeat server absolute paths.  The
registry keeps legacy paths for auditability and never deletes them implicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = PROJECT_ROOT / "configs" / "server_paths.yaml"


def load_path_registry(registry_path: Path = REGISTRY_PATH) -> Dict[str, Any]:
    value = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "1.0":
        raise ValueError(f"invalid PFMval path registry: {registry_path}")
    return value


def get_registered_path(
    path_id: str,
    *,
    require_exists: bool = False,
    registry_path: Path = REGISTRY_PATH,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    registry = load_path_registry(registry_path)
    entry = registry.get("paths", {}).get(path_id)
    if not entry:
        raise KeyError(f"unknown PFMval path id: {path_id}")
    raw = str(entry.get("path", ""))
    if not raw:
        raise ValueError(f"PFMval path id has no value: {path_id}")
    path = Path(raw)
    if not path.is_absolute():
        root_resolved = project_root.resolve()
        path = (root_resolved / path).resolve()
        if not path.is_relative_to(root_resolved):
            raise ValueError(f"registered relative path escapes project root: {path_id} -> {raw}")
    if require_exists and not path.exists():
        raise FileNotFoundError(f"registered path does not exist: {path_id} -> {path}")
    return path


def format_registered_template(template_id: str, **values: str | int) -> Path:
    registry = load_path_registry()
    entry = registry.get("templates", {}).get(template_id)
    if not entry:
        raise KeyError(f"unknown PFMval path template id: {template_id}")
    try:
        return Path(str(entry["path"]).format(**values))
    except KeyError as exc:
        raise ValueError(f"missing template value for {template_id}: {exc}") from exc
