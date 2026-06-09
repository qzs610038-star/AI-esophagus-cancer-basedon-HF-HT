"""
compare_token_encoder_runs.py — 对比 Token Encoder 实验结果

用法:
    python compare_token_encoder_runs.py
"""
from pathlib import Path
import pandas as pd

runs = {
    "transformer_65t": Path("checkpoints/online_tokens/frozen_r8_online_tokens_cross_fold1_65t_transformer"),
    "gfnet_65t": Path("checkpoints/online_tokens/frozen_r8_online_tokens_cross_fold1_65t_gfnet"),
    "transformer_smoke": Path("checkpoints/online_tokens/frozen_r8_online_tokens_cross_fold1_65t_transformer_smoke"),
    "gfnet_smoke": Path("checkpoints/online_tokens/frozen_r8_online_tokens_cross_fold1_65t_gfnet_smoke"),
}

print("=" * 72)
print("Token Encoder 实验对比")
print("=" * 72)

for name, run_dir in runs.items():
    hist_path = run_dir / "training_history.csv"
    if not hist_path.exists():
        print(f"\n{name}: MISSING (no {hist_path})")
        continue
    hist = pd.read_csv(hist_path)
    best = hist.loc[hist["val_loss"].idxmin()]
    gap = float(best["train_pcc"] - best["val_pcc"])
    print(
        f"{name:.<40s} "
        f"ep={int(best['epoch']):3d}  "
        f"val_loss={best['val_loss']:.4f}  "
        f"val_pcc={best['val_pcc']:.4f}  "
        f"train_pcc={best['train_pcc']:.4f}  "
        f"gap={gap:.4f}"
    )

# -- comparison --
print()
print("=" * 72)
print("GFNet vs Transformer (smoke test, 1-epoch comparison)")
print("=" * 72)

ts = runs["transformer_smoke"]
gs = runs["gfnet_smoke"]
if ts.exists() and gs.exists():
    t_hist = pd.read_csv(ts / "training_history.csv")
    g_hist = pd.read_csv(gs / "training_history.csv")
    t_best = t_hist.loc[t_hist["val_loss"].idxmin()]
    g_best = g_hist.loc[g_hist["val_loss"].idxmin()]
    delta = float(g_best["val_pcc"] - t_best["val_pcc"])
    pct = delta / t_best["val_pcc"] * 100
    print(f"  GFNet val_pcc:     {g_best['val_pcc']:.4f}")
    print(f"  Transformer val_pcc: {t_best['val_pcc']:.4f}")
    print(f"  Delta:              {delta:+.4f} ({pct:+.1f}%)")
    t_gap = float(t_best["train_pcc"] - t_best["val_pcc"])
    g_gap = float(g_best["train_pcc"] - g_best["val_pcc"])
    print(f"  GFNet Train-Val Gap:     {g_gap:.4f}")
    print(f"  Transformer Train-Val Gap: {t_gap:.4f}")
    gap_delta = g_gap - t_gap
    print(f"  Gap Delta:               {gap_delta:+.4f}")

# -- CLS historical reference --
print()
print("=" * 72)
print("Historical CLS Baselines (for reference)")
print("=" * 72)
print(f"  CLS frozen Fold1:    0.4113")
print(f"  CLS LoRA Fold1:      0.4322")
print(f"  Token Transformer:   {t_best['val_pcc']:.4f} (1ep smoke, 65-token)")
print(f"  Token GFNet:         {g_best['val_pcc']:.4f} (1ep smoke, 65-token)")
