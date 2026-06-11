# Decision Log

> Key architectural and strategic decisions for the PFMval project.
> See `experiment_registry.json` for per-experiment decision gates.

| Date | Decision | Reason | Consequence |
|------|----------|--------|-------------|
| 2026-06-11 | **Close frequency direction (A2/A3/AFNO)** | A1 modReLU degraded val_pcc by -6.1% and widened Train-Val Gap by +50% vs A0b clean-token baseline. B1a gfnet branch tied B1b mean pool; both below CLS Frozen. Two independent evidence lines disprove frequency mechanism for this task. | A2 (frequency augmentation), A3 (AFNO), CLS+freq side-branch all cancelled. Frequency token mixing archived. |
| 2026-06-11 | **Token+LoRA established as official track (C1)** | GFNet LoRA r=8 + cls_patch64 achieved PCC=0.4169, surpassing CLS Frozen (0.4113) and becoming the best Token-mode result ever. LoRA gain +5.1% vs frozen, consistent with CLS LoRA gain. | Expand to Fold2/3 cross-validation. Token+LoRA is the primary Token-mode configuration going forward. |
| 2026-06-11 | **cls_patch64 adopted as new default token selection** | A0b cls_patch64 (0.3967) outperformed legacy_firstN (0.3914) by +1.4% with narrower Train-Val Gap. Excluding register tokens from FFT sequence improves both performance and semantic clarity. | All future Token-mode experiments use cls_patch64. Legacy results preserved for reference. |
| 2026-06-11 | **CLS LoRA r=8 remains primary baseline** | C1 (0.4169) still trails CLS LoRA (0.4322) by 0.0153. CLS LoRA is the strongest validated configuration and should anchor all comparisons. | CLS LoRA stays as main track; Token+LoRA is the challenger/alternative. |
| 2026-06-10 | GFNet 65-token Fold1 gates the frequency branch | Smoke test outperformed Transformer by +5.6% (PCC 0.3933 vs 0.3724) | If formal PCC >= 0.3821, test 265-token GFNet; otherwise pause frequency branch |
| 2026-06-04 | Pause 3-patient wrap-up, wait for 9-patient data | 9-patient data arriving within days; 3-patient Fold3 has limited impact on conclusions | All 3-patient trailing experiments suspended; full matrix rerun on 9 patients |
| 2026-06-04 | LoRA r=8 is optimal for 3-patient cross-patient scenario | Stage2/Stage3 unfreezing harmed generalization; LoRA's low-rank constraint is best regularizer | Stick with LoRA r=8 for 9-patient experiments |
