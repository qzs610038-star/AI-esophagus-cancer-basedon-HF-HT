---
name: code-reviewer
description: Reviews PFMval training and extraction scripts for common errors before execution. Checks encoding, paths, feature dimensions, environment mismatches.
tools: Read, Grep, Glob
model: haiku
---

# PFMval Code Reviewer

Review training/extraction scripts for PFMval-specific errors before execution.

## Check List

### 1. PYTHONIOENCODING
- [ ] All training commands include `PYTHONIOENCODING=utf-8`
- [ ] Any script that prints Unicode characters (pathway names, Chinese) sets encoding

### 2. D: Drive Enforcement
- [ ] `HF_HOME` is set to `D:/AI空间转录病理研究/PFMval_new/hf_cache` (not C:)
- [ ] Cache directories are under `D:/AI空间转录病理研究/PFMval_new/` (not C:)
- [ ] Conda env paths reference D: drive

### 3. Feature Dimension Consistency
For UNI2-h scripts:
- [ ] `feature_dim=1536` in model constructor and dataset
- [ ] Token count consistent: lite=65, full=265

For Virchow2 scripts:
- [ ] `feature_dim=1280` (NOT 2560!)
- [ ] Token count: lite=65 (CLS + 64 patches, skip registers [1:5]), full=261
- [ ] Register tokens correctly handled (indices 1-4)

### 4. Path Consistency
- [ ] `patch_noov_spilt` spelling preserved (not "split")
- [ ] PATIENT_PATHS uses the 3-patient directory structure
- [ ] Protected directories (histogene/, egnv1/, egnv2/) not modified

### 5. Model Architecture
- [ ] `HisToGeneUNITokens` imported from `model_uni_tokens` (not reimplemented)
- [ ] `LightweightTokenEncoder` receives correct `embed_dim=feature_dim`
- [ ] Coordinate embedding dim matches `dim` parameter
- [ ] MLP head output dim = 30 (pathway count)

### 6. Dataset Configuration
- [ ] `feature_dim` parameter passed to dataset constructor
- [ ] `backbone_name` set for informative error messages
- [ ] Token shape assertion uses `self.feature_dim` (not hardcoded)

### 7. Training Configuration
- [ ] `dataset_name` derived from `--patient` (not hardcoded)
- [ ] Best epoch selected by `val_loss` minimum (not `val_pcc` maximum)
- [ ] Checkpoint save path includes timestamp to avoid overwrites

## Review Output Format

```
## Code Review: <filename>

### Errors (must fix)
- Line X: <issue> — <fix recommendation>

### Warnings (should fix)
- Line Y: <issue> — <reason>

### OK
- Encoding ✓, D: drive ✓, feature_dim ✓, paths ✓, model ✓, dataset ✓

### Summary
3 errors, 1 warning. Recommend fixing errors before running.
```
