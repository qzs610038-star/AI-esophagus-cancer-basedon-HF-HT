# Gene→pathway smoke-prediction evaluation

Standalone, read-only evaluation for the specified 319-gene smoke run. It reads the prediction run's train-only per-gene scaler and prediction CSVs, but reads pathway truth **only** from the shared static files in `../outputs_v002/`. It never scores the prediction table's `true_<gene>` values and never refits a scaler. Results are explicitly `exploratory_smoke` / `provenance_pending`, not a formal performance claim.

## Run

From this directory:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lzd/miniconda3/envs/aipatho/bin/python -B -m unittest -v
PYTHONDONTWRITEBYTECODE=1 /home/lzd/miniconda3/envs/aipatho/bin/python -B evaluate_gene_predictions.py --output-dir results_smoke_eval_v003
```

Input paths default to the provided `20260910_154315_123_2d81878b` run, package frozen split manifest, current JFX/other patient pre-ssGSEA CSVs, and sibling `outputs_v002`. Every input may be overridden with CLI options (`--run-dir`, `--prediction-internal`, `--prediction-external`, `--run-json`, `--target-spec`, `--source-scaler`, `--gene-dir`, `--split-manifest`, `--v002-dir`). `--output-dir` must name a new direct child of this directory; existing, symlinked, or outside paths are rejected.

## Guardrails

- The two prediction tables must match the exact `patient_id,patch_id,x,y,split` + interleaved `true_gene,pred_gene` target-spec order + run metadata header. Duplicate/missing/extra identities, coordinates, split labels, non-finite values, or scaler/target order mismatches stop the run.
- Internal validation must exactly match the 1,078 `split_manifest.csv` internal-val identities; XZY must exactly match the 1,039 frozen candidate identities. External input may label its split `external` or `external_test`; output stores the original source value as `prediction_source_split` and canonicalizes the evaluation group to `XZY`.
- True gene z values are inverse-transformed only to audit them against the source pre-ssGSEA gene CSVs (`max abs error <= 2e-6`). Predicted z values are inverse-transformed with the prediction run's source scaler, then reordered by gene name to v002's frozen 319-gene order before ssGSEA.
- Only predicted genes are scored, using the sibling scorer, v002's fixed member mapping, and its recorded GSEApy 1.1.11 settings. Pathway predictions are transformed with v002 train-only raw-ES parameters and the frozen post-transform clip. Static candidate raw/z labels are joined, never regenerated.
- XZY is kept in a separate exploratory result and must not be used for model/configuration/parameter selection. No historical direct-pathway arm is loaded or compared. No pooled sample PCC is reported: the outputs contain patient×pathway metrics and split/pathway patient-macro and median PCC.
- Constant vectors yield null PCC/R² with explicit reasons. z-space MAE/RMSE are primary; raw-ES MAE/RMSE are secondary.

## Outputs

Final results are written only to `results_smoke_eval_v002/`:

- `predictions_internal_val.csv`, `predictions_XZY.csv`: shared true raw/z and gene-derived predicted raw/z pathway values by exact identity.
- `metrics_by_patient_pathway.csv`: internal patient or XZY × pathway metrics, including null-PCC reasons.
- `metrics_by_split_pathway.csv`: separate `internal_val` and `XZY` summaries, macro/median across per-patient PCCs (not pooled PCC).
- `gene_scaler_comparison.csv`: source prediction scaler versus v002 gene scaler by gene name.
- `evaluation_summary.json`, `sha256_audit.json`: run kind/status, row/column counts, scaler-inverse verification, scoring settings, input/truth/tool/output SHA-256 and limitations.

Run metadata records `run_kind=smoke`. These files are exploratory only and cannot support a formal performance claim while the shared truth remains provenance-pending.

`results_smoke_eval_v001/` is the initial successful run retained unchanged; v002 adds explicit input/output row-column counts, preserves the source external split label, and avoids wide-frame insertion warnings.
