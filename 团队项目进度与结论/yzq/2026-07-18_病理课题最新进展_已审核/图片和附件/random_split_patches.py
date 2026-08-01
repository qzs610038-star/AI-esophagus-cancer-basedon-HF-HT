import argparse
import random
import shutil
from pathlib import Path


def split_patches(src_dir: Path, dst_root: Path, val_ratio: float, seed: int) -> None:
    if not src_dir.exists():
        raise FileNotFoundError(f"Source folder does not exist: {src_dir}")

    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1, e.g. 0.1")

    png_files = sorted([p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
    total = len(png_files)
    if total == 0:
        raise RuntimeError(f"No PNG files found in: {src_dir}")

    rng = random.Random(seed)
    rng.shuffle(png_files)

    val_count = int(total * val_ratio)
    val_files = png_files[:val_count]
    train_files = png_files[val_count:]

    train_dir = dst_root / "train_patches"
    val_dir = dst_root / "val_patches"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for p in train_files:
        shutil.copy2(p, train_dir / p.name)
    for p in val_files:
        shutil.copy2(p, val_dir / p.name)

    print("=== Split Summary ===")
    print(f"Source: {src_dir}")
    print(f"Destination root: {dst_root}")
    print(f"Total PNG: {total}")
    print(f"Train count: {len(train_files)}")
    print(f"Val count: {len(val_files)}")
    print(f"Val ratio (actual): {len(val_files) / total:.4f}")
    print(f"Random seed: {seed}")
    print(f"Train folder: {train_dir}")
    print(f"Val folder: {val_dir}")


def main():
    parser = argparse.ArgumentParser(description="Randomly split PNG patches into train/val folders.")
    parser.add_argument(
        "--src",
        default=r"D:\PycharmProjects\AIPath-data\patch\LMZ12939_noov",
        help="Source folder containing PNG patches.",
    )
    parser.add_argument(
        "--dst",
        default=r"D:\PycharmProjects\AIPath-data\patch\LMZ12939_noov_split",
        help="Destination root folder. train_patches and val_patches will be created inside.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio, default is 0.1 (10%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    split_patches(
        src_dir=Path(args.src),
        dst_root=Path(args.dst),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
