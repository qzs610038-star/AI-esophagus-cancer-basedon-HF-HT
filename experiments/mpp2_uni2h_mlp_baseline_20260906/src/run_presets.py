"""Map run_mode.json to training or prediction presets."""

PRESETS = {
    "smoke": {
        "mode": "train",
        "num_epochs": 2,
        "patience": 10,
        "min_delta": 1e-4,
    },
    "full": {
        "mode": "train",
        "num_epochs": 50,
        "patience": 10,
        "min_delta": 1e-4,
    },
    "predict": {
        "mode": "predict",
        "num_epochs": 0,
        "patience": 10,
        "min_delta": 1e-4,
    },
}


def resolve_run_kind(run_kind: str) -> dict:
    if run_kind not in PRESETS:
        raise ValueError(
            f"run_kind must be one of {sorted(PRESETS)}: got {run_kind!r}. "
            "Edit only run_mode.json."
        )
    resolved = dict(PRESETS[run_kind])
    resolved["run_kind"] = run_kind
    return resolved
