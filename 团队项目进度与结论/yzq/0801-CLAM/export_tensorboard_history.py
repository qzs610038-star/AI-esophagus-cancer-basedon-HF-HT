import argparse
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_event_scalars(event_file):
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    rows = []
    for tag in accumulator.Tags().get("scalars", []):
        for event in accumulator.Scalars(tag):
            rows.append({
                "tag": tag,
                "epoch": int(event.step),
                "value": float(event.value),
                "wall_time": float(event.wall_time),
                "event_file": str(event_file),
            })
    return rows


def event_sort_key(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return 0


def load_run(logdir):
    logdir = Path(logdir)
    if not logdir.exists():
        raise FileNotFoundError(f"Logdir does not exist: {logdir}")

    all_rows = []
    fold_dirs = sorted([p for p in logdir.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))
    if not fold_dirs:
        fold_dirs = [logdir]

    for fold_dir in fold_dirs:
        fold = int(fold_dir.name) if fold_dir.name.isdigit() else 0
        event_files = sorted(fold_dir.glob("events.out.tfevents*"), key=event_sort_key)
        for event_file in event_files:
            rows = read_event_scalars(event_file)
            for row in rows:
                row["fold"] = fold
            all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError(f"No scalar TensorBoard events found under {logdir}")

    history = pd.DataFrame(all_rows)
    history = history.sort_values(["fold", "tag", "epoch", "wall_time"])
    # If multiple event files contain the same fold/tag/epoch, keep the last written value.
    history = history.drop_duplicates(["fold", "tag", "epoch"], keep="last")
    return history


def make_wide(history):
    wide = history.pivot_table(index=["fold", "epoch"], columns="tag", values="value", aggfunc="last")
    wide = wide.reset_index()
    wide.columns.name = None
    return wide


def make_best_summary(wide):
    rows = []
    for fold, group in wide.groupby("fold"):
        row = {"fold": int(fold)}
        if "val/loss" in group:
            valid = group.dropna(subset=["val/loss"])
            if not valid.empty:
                best = valid.loc[valid["val/loss"].idxmin()]
                row["best_by_val_loss_epoch"] = int(best["epoch"])
                row["best_val_loss"] = float(best["val/loss"])
                for col in ["val/auc", "val/error", "train/loss", "train/error", "train/clustering_loss", "val/inst_loss"]:
                    if col in best and pd.notna(best[col]):
                        row[f"at_best_{col.replace('/', '_')}"] = float(best[col])
        if "val/auc" in group:
            valid = group.dropna(subset=["val/auc"])
            if not valid.empty:
                best_auc = valid.loc[valid["val/auc"].idxmax()]
                row["best_by_val_auc_epoch"] = int(best_auc["epoch"])
                row["best_val_auc"] = float(best_auc["val/auc"])
        for col in ["final/test_auc", "final/test_error", "final/val_auc", "final/val_error"]:
            if col in group:
                valid = group.dropna(subset=[col])
                if not valid.empty:
                    row[col.replace("/", "_")] = float(valid.iloc[-1][col])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fold")


def main(args):
    history = load_run(args.logdir)
    wide = make_wide(history)
    best = make_best_summary(wide)

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.logdir) / "tb_export"
    output_dir.mkdir(parents=True, exist_ok=True)

    long_path = output_dir / "history_long.csv"
    wide_path = output_dir / "history_wide.csv"
    best_path = output_dir / "best_epochs.csv"

    history.to_csv(long_path, index=False)
    wide.to_csv(wide_path, index=False)
    best.to_csv(best_path, index=False)

    fold_dir = output_dir / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    for fold, group in wide.groupby("fold"):
        fold_path = fold_dir / f"fold_{int(fold)}_history.csv"
        group.drop(columns=["fold"]).to_csv(fold_path, index=False)

    print(f"Read TensorBoard scalars from: {args.logdir}")
    print(f"Scalar tags: {', '.join(sorted(history['tag'].unique()))}")
    print(f"Saved long history: {long_path}")
    print(f"Saved wide history: {wide_path}")
    print(f"Saved best epoch summary: {best_path}")
    print(f"Saved per-fold histories: {fold_dir}")
    print()
    print(best.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CLAM TensorBoard scalar history to CSV files.")
    parser.add_argument("--logdir", required=True, help="Training result directory, e.g. results/ESCC_pCR_UNI2H_CLAM_s1")
    parser.add_argument("--output_dir", default=None, help="Output directory. Defaults to <logdir>/tb_export")
    main(parser.parse_args())
