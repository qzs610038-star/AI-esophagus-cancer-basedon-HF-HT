# Incident: MPP2 baseline cache-root mismatch (2026-07-11)

## Symptom

The repaired MPP2 frozen baseline passed state, staging, GPU, and registry
preflight, then stopped while building the manifest dataset:

```text
HYZ15040/train: 1309/1309 未匹配
patch_x10248_y10248: no .pt
ValueError: HYZ15040/train: 1309 未匹配 (allow_missing=False)
```

The failed run is not a valid training result. `--allow_missing` must not be
used for a formal baseline.

## Evidence

The server inventory found no MPP2 training features under:

```text
D:\AIPatho\qzs\pfmval_deploy_git\mpp_uni2h_cache
```

The same server contains 1454 feature files under:

```text
D:\AIPatho\qzs\pfmval_deploy_git\uni2h_cache\MPP2_UNI\HYZ15040
```

The external MPP2/XZY cache remains a flat-cache responsibility.

## Root cause

`deploy/pfmval_ops.py` injected both `cache_root` and `flat_cache_root` from
`server_mpp_flat_cache`. The manifest loader therefore searched the wrong root
for MPP2 training features. `configs/server_paths.yaml` already distinguishes
the partner cache (`server_mpp_partner_cache`) from the flat cache
(`server_mpp_flat_cache`).

## Corrective action

- MPP training `cache_root` now resolves from `server_mpp_partner_cache`.
- `flat_cache_root` remains bound to `server_mpp_flat_cache` for external XZY.
- MPP formal job semantics require both cache path IDs.
- Regression coverage asserts both injected paths and their roles.
- Do not rebuild repaired labels or alter protected MPP assets for this issue.

## Gate status

The baseline must be rerun after the corrected job is dispatched. LoRA remains
blocked until the successful repaired MPP2 baseline is imported and evaluated
by the baseline guard.
