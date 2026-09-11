"""Run/cache paths and the weight-path registry; no hashes and no weight copies."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

def task_directory(root: str|Path, *, protocol: str, fold: str, seed: int, cell: str, endpoint: int) -> Path:
    return Path(root) / str(protocol) / str(fold) / f"seed_{int(seed)}" / str(cell) / f"epoch_{int(endpoint)}"

def cache_directory(cache_root: str|Path, **key: Any) -> Path:
    path=task_directory(cache_root,**key); path.mkdir(parents=True,exist_ok=True); return path

def weight_registry_entry(weights_root: str|Path, **key: Any) -> dict:
    directory=task_directory(weights_root,**key).resolve()
    warmup, formal, last = str(directory/"warmup.pt"), str(directory/"formal.pt"), str(directory/"last.pt")
    return {"protocol":key["protocol"],"fold":key["fold"],"seed":int(key["seed"]),"cell":key["cell"],"endpoint":int(key["endpoint"]),"weight_directory":str(directory),"warmup":warmup,"formal":formal,"last":last,"warmup_checkpoint":warmup,"formal_checkpoint":formal,"last_checkpoint":last}

def write_model_weights(run_dir: str|Path, weights_root: str|Path, **key: Any) -> Path:
    """Register absolute server paths even in dry-run; this never creates a weight."""
    path=Path(run_dir)/"model_weights.json"; path.parent.mkdir(parents=True,exist_ok=True)
    payload={"schema_version":"v4","weights_separate_from_runs":True,"entries":[weight_registry_entry(weights_root,**key)]}
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8"); return path
