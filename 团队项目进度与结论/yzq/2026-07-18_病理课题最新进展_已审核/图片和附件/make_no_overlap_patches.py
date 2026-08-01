import argparse
import math
import re
import shutil
from pathlib import Path


XY_PATTERN = re.compile(r"x(-?\d+)_y(-?\d+)", re.IGNORECASE)


def parse_xy_from_filename(filename: str):
    stem = Path(filename).stem
    match = XY_PATTERN.search(stem)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def infer_step(values):
    unique_sorted = sorted(set(values))
    diffs = [b - a for a, b in zip(unique_sorted[:-1], unique_sorted[1:]) if b > a]
    if not diffs:
        return None

    step = 0
    for diff in diffs:
        step = math.gcd(step, diff)

    if step <= 0:
        step = min(diffs)
    return step


def select_non_overlapping(records, step_x, step_y):
    min_x = min(r["x"] for r in records)
    min_y = min(r["y"] for r in records)

    phase_groups = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
    invalid_grid_records = []

    for rec in records:
        dx = rec["x"] - min_x
        dy = rec["y"] - min_y

        if dx % step_x != 0 or dy % step_y != 0:
            invalid_grid_records.append(rec)
            continue

        gx = dx // step_x
        gy = dy // step_y
        phase = (gx % 2, gy % 2)
        phase_groups[phase].append(rec)

    best_phase = max(phase_groups.keys(), key=lambda p: len(phase_groups[p]))
    return phase_groups[best_phase], best_phase, invalid_grid_records


def main():
    parser = argparse.ArgumentParser(
        description="Extract a non-overlapping subset from 50%-overlapped patch PNGs."
    )
    parser.add_argument(
        "--src",
        default=r"D:\PycharmProjects\AIPath-data\patch\HYZ15040_org",
        help="Source folder that contains overlapped patch PNGs.",
    )
    parser.add_argument(
        "--dst",
        default=r"D:\PycharmProjects\AIPath-data\patch\HYZ15040_noov",
        help="Destination folder for non-overlapping patch PNGs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview selection only, do not copy files.",
    )
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.exists():
        raise FileNotFoundError(f"Source folder does not exist: {src_dir}")

    png_files = sorted([p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
    print(f"Found PNG files: {len(png_files)}")
    if not png_files:
        return

    records = []
    unparsed = []
    for p in png_files:
        xy = parse_xy_from_filename(p.name)
        if xy is None:
            unparsed.append(p)
            continue
        x, y = xy
        records.append({"path": p, "x": x, "y": y})

    print(f"Parsed coordinates: {len(records)}")
    print(f"Unparsed filenames: {len(unparsed)}")
    if unparsed:
        print("First 5 unparsed files:")
        for p in unparsed[:5]:
            print(f"  - {p.name}")

    if not records:
        raise RuntimeError("No valid coordinates found in filenames. Nothing to split.")

    step_x = infer_step([r["x"] for r in records])
    step_y = infer_step([r["y"] for r in records])
    if step_x is None or step_y is None:
        raise RuntimeError("Unable to infer coordinate step from filenames.")

    print(f"Inferred stride: step_x={step_x}, step_y={step_y}")
    print("Expected non-overlap sampling interval: 2 * stride in each axis")

    selected, phase, invalid_grid_records = select_non_overlapping(records, step_x, step_y)
    print(f"Selected phase: {phase}")
    print(f"Selected non-overlapping patches: {len(selected)}")
    print(f"Skipped (off-grid relative to inferred stride): {len(invalid_grid_records)}")

    if args.dry_run:
        print("Dry-run mode enabled. No files copied.")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for rec in selected:
        dst_path = dst_dir / rec["path"].name
        shutil.copy2(rec["path"], dst_path)
        copied += 1

    print(f"Copied {copied} files to: {dst_dir}")


if __name__ == "__main__":
    main()
