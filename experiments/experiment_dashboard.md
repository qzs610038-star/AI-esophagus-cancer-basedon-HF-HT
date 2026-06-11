# Experiment Dashboard

> Source of truth: `experiments/experiment_registry.json`
> Last updated: 2026-06-11

## Status Overview

| ID | Family | Status | Fold | Encoder | Tokens | Mode | Token Select | Best Val PCC | Best Val Loss | Train-Val Gap | Best Ep |
|----|--------|--------|:----:|---------|:------:|:----:|:---:|:------------:|:-------------:|:-------------:|:-----:|
| C1_gfnet_lora_65t_fold1 | online_tokens | ✅ done | 1 | gfnet | 65 | lora | cls_patch64 | **0.4169** | 0.3321 | 0.1937 | 3 |
| A0b_gfnet_clean_token_65t_fold1 | online_tokens | ✅ done | 1 | gfnet | 65 | frozen | cls_patch64 | 0.3967 | 0.3320 | 0.1600 | 2 |
| B1b_cls_freq_mean_fold1 | online_cls | ✅ done | 1 | — | — | frozen | cls_patch64 | 0.3955 | 0.3313 | 0.1985 | 2 |
| smoke_gfnet_65t | online_tokens | ✅ done | 1 | gfnet | 65 | frozen | legacy_firstN | 0.3933 | 0.3314 | 0.0935 | 1 |
| online_tokens_gfnet_fold1_65t_legacy | online_tokens | ✅ done | 1 | gfnet | 65 | frozen | legacy_firstN | 0.3914 | 0.3337 | 0.1683 | 2 |
| B1a_cls_freq_gfnet_fold1 | online_cls | ✅ done | 1 | — | — | frozen | cls_patch64 | 0.3881 | 0.3297 | 0.1959 | 2 |
| online_tokens_transformer_fold1_65t | online_tokens | ⚠️ incomplete | 1 | transformer | 65 | frozen | legacy_firstN | 0.3821 | — | — | 4 |
| A1_gfnet_modrelu_65t_fold1 | online_tokens | ❌ failed | 1 | gfnet | 65 | frozen | cls_patch64 | 0.3724 | 0.3394 | 0.2412 | 4 |

## Reference Baselines (not in current registry)

| ID | Family | Mode | Val PCC | Notes |
|----|--------|:----:|:---:|------|
| CLS LoRA r=8 legacy | online_cls | lora | **0.4322** | 🏆 Current strongest overall |
| CLS Frozen legacy | online_cls | frozen | 0.4113 | CLS frozen baseline |

## Decision Gates Summary

| Gate | Condition | Verdict | Action |
|------|-----------|:---:|------|
| A1 vs A0b | modReLU must improve over clean-token baseline | 🔴 FAIL | Close A2/A3/AFNO. Frequency direction terminated. |
| B1a vs B1b | GFNet freq branch must outperform mean pool | 🔴 FAIL (tie) | Close CLS+freq side-branch. No frequency mechanism advantage. |
| C1 ≥ 0.41 | Token+LoRA must surpass CLS Frozen | 🟢 PASS | Establish Token+LoRA as official track. Expand to Fold2/3. |

## Direction Map

```
🟢 Token + LoRA (C1)           → PCC 0.4169, best Token result. Expand Fold2/3.
🟢 CLS LoRA r=8                → PCC 0.4322, main baseline. Maintain.
🟡 cls_patch64 token selection  → +1.4% vs legacy. Adopt as new default.
🔴 GFNet frequency token mixing → Closed. A1 modReLU fails.
🔴 CLS + frequency side-branch  → Closed. B1a ≈ B1b.
🔴 AFNO / A2 / A3              → Cancelled. Gate conditions not met.
🔴 265-token GFNet             → Cancelled. Frequency direction closed.
```
