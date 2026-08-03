import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch


COORD_RE = re.compile(r"_x(-?\d+)_y(-?\d+)(?:\.[^.]+)?$", re.IGNORECASE)


def parse_coord(value):
    name = Path(str(value)).stem
    match = COORD_RE.search(name)
    if match is None:
        raise ValueError(f"Cannot parse x/y coordinates from patch name: {value}")
    return int(match.group(1)), int(match.group(2))


def expected_slide_ids(csv_path):
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path)
    if "slide_id" not in df.columns:
        raise ValueError(f"{csv_path} does not contain a slide_id column")
    return set(df["slide_id"].astype(str))


def find_score_csv(input_root, slide_id, suffix):
    candidates = [
        input_root / f"{slide_id}{suffix}",
        input_root / f"{slide_id}.csv",
    ]
    matches = [path for path in candidates if path.exists()]
    if not matches:
        globbed = sorted(input_root.glob(f"{slide_id}*.csv"))
        matches.extend(globbed)
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one score CSV for {slide_id}, found {len(matches)}")
    return matches[0]


def infer_score_columns(df, patch_name_col, requested_columns):
    if requested_columns:
        missing = [column for column in requested_columns if column not in df.columns]
        if missing:
            raise ValueError(f"Requested score columns not found: {missing}")
        return requested_columns

    score_columns = [column for column in df.columns if column != patch_name_col]
    numeric_columns = []
    for column in score_columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().all():
            numeric_columns.append(column)

    if not numeric_columns:
        raise ValueError("No numeric score columns found")
    return numeric_columns


def load_reference_coords(reference_h5_dir, slide_id):
    h5_path = reference_h5_dir / f"{slide_id}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing reference h5 file: {h5_path}")
    with h5py.File(h5_path, "r") as h5_file:
        if "coords" not in h5_file:
            raise KeyError(f"{h5_path} does not contain a coords dataset")
        coords = np.asarray(h5_file["coords"][:], dtype=np.int32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coords shape (n, 2) in {h5_path}, got {coords.shape}")
    return coords


def convert_slide(
    csv_path,
    slide_id,
    reference_h5_dir,
    output_pt_dir,
    output_h5_dir,
    patch_name_col,
    requested_score_columns,
    overwrite=False,
    write_h5=True,
):
    pt_output = output_pt_dir / f"{slide_id}.pt"
    h5_output = output_h5_dir / f"{slide_id}.h5"

    if pt_output.exists() and (not write_h5 or h5_output.exists()) and not overwrite:
        return {"slide_id": slide_id, "status": "skipped_existing", "patches": None, "feature_dim": None}

    reference_coords = load_reference_coords(reference_h5_dir, slide_id)
    df = pd.read_csv(csv_path)
    if patch_name_col not in df.columns:
        raise ValueError(f"{csv_path} does not contain patch name column: {patch_name_col}")

    score_columns = infer_score_columns(df, patch_name_col, requested_score_columns)
    df = df[[patch_name_col, *score_columns]].copy()
    df["_coord"] = df[patch_name_col].map(parse_coord)

    duplicated = df[df["_coord"].duplicated(keep=False)]
    if not duplicated.empty:
        examples = duplicated[patch_name_col].head(5).tolist()
        raise ValueError(f"Duplicated patch coordinates in {csv_path}: {examples}")

    coord_to_scores = {}
    for _, row in df.iterrows():
        coord_to_scores[row["_coord"]] = row[score_columns].to_numpy(dtype=np.float32)

    ordered_rows = []
    missing_coords = []
    for x, y in reference_coords.tolist():
        coord = (int(x), int(y))
        if coord not in coord_to_scores:
            missing_coords.append(coord)
        else:
            ordered_rows.append(coord_to_scores[coord])

    if missing_coords:
        raise ValueError(
            f"{csv_path} is missing {len(missing_coords)} coordinates present in reference h5; "
            f"first missing: {missing_coords[:5]}"
        )

    reference_coord_set = {tuple(coord) for coord in reference_coords.tolist()}
    extra_coords = sorted(set(coord_to_scores) - reference_coord_set)
    if extra_coords:
        raise ValueError(
            f"{csv_path} has {len(extra_coords)} coordinates not present in reference h5; "
            f"first extra: {extra_coords[:5]}"
        )

    score_array = np.asarray(ordered_rows, dtype=np.float32)
    score_tensor = torch.from_numpy(score_array)
    torch.save(score_tensor, pt_output)

    if write_h5:
        with h5py.File(h5_output, "w") as h5_file:
            h5_file.create_dataset(
                "features",
                data=score_array,
                chunks=(min(1024, len(score_array)), score_array.shape[1]),
            )
            h5_file.create_dataset(
                "coords",
                data=reference_coords,
                chunks=(min(1024, len(reference_coords)), 2),
            )

    return {
        "slide_id": slide_id,
        "status": "converted",
        "patches": int(score_array.shape[0]),
        "feature_dim": int(score_array.shape[1]),
        "source_csv": str(csv_path),
    }


def main(args):
    input_root = Path(args.input_root)
    reference_feature_dir = Path(args.reference_feature_dir)
    reference_h5_dir = reference_feature_dir / "h5_files"
    output_dir = Path(args.output_dir)
    output_pt_dir = output_dir / "pt_files"
    output_h5_dir = output_dir / "h5_files"
    output_pt_dir.mkdir(parents=True, exist_ok=True)
    if not args.pt_only:
        output_h5_dir.mkdir(parents=True, exist_ok=True)

    expected_ids = expected_slide_ids(args.expected_csv)
    reference_h5_files = sorted(reference_h5_dir.glob("*.h5"), key=lambda path: path.stem)
    slide_ids = [path.stem for path in reference_h5_files]
    if expected_ids is not None:
        slide_ids = [slide_id for slide_id in slide_ids if slide_id in expected_ids]

    requested_score_columns = None
    if args.score_columns:
        requested_score_columns = [column.strip() for column in args.score_columns.split(",") if column.strip()]

    print(f"Input score root: {input_root}")
    print(f"Reference feature dir: {reference_feature_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Slides to convert: {len(slide_ids)}")

    records = []
    score_columns_written = None
    for idx, slide_id in enumerate(slide_ids, start=1):
        print(f"[{idx}/{len(slide_ids)}] {slide_id}")
        csv_path = find_score_csv(input_root, slide_id, args.score_csv_suffix)
        record = convert_slide(
            csv_path=csv_path,
            slide_id=slide_id,
            reference_h5_dir=reference_h5_dir,
            output_pt_dir=output_pt_dir,
            output_h5_dir=output_h5_dir,
            patch_name_col=args.patch_name_col,
            requested_score_columns=requested_score_columns,
            overwrite=args.overwrite,
            write_h5=not args.pt_only,
        )
        records.append(record)
        print(f"  {record['status']}: patches={record['patches']} dim={record['feature_dim']}")

        if record["status"] == "converted":
            df = pd.read_csv(csv_path, nrows=1)
            score_columns_written = infer_score_columns(df, args.patch_name_col, requested_score_columns)

    summary = pd.DataFrame(records)
    summary.to_csv(output_dir / "conversion_summary.csv", index=False)

    if score_columns_written is None and slide_ids:
        first_csv = find_score_csv(input_root, slide_ids[0], args.score_csv_suffix)
        first_df = pd.read_csv(first_csv, nrows=1)
        score_columns_written = infer_score_columns(first_df, args.patch_name_col, requested_score_columns)
    if score_columns_written is not None:
        pd.DataFrame({"feature_index": range(len(score_columns_written)), "score_column": score_columns_written}).to_csv(
            output_dir / "score_columns.csv",
            index=False,
        )

    if expected_ids is not None:
        reference_ids = {path.stem for path in reference_h5_files}
        missing_reference = sorted(expected_ids - reference_ids)
        extra_reference = sorted(reference_ids - expected_ids)
        pd.DataFrame({"missing_reference_slide_id": missing_reference}).to_csv(
            output_dir / "missing_from_reference.csv",
            index=False,
        )
        pd.DataFrame({"extra_reference_slide_id": extra_reference}).to_csv(
            output_dir / "extra_reference_not_in_expected_csv.csv",
            index=False,
        )
        print(f"Expected slides: {len(expected_ids)}")
        print(f"Missing expected reference h5 files: {len(missing_reference)}")
        print(f"Extra reference h5 files not in expected csv: {len(extra_reference)}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert per-patch gene-set score CSVs into CLAM feature format, ordered exactly like a "
            "reference CLAM h5_files/coords directory."
        )
    )
    parser.add_argument("--input_root", type=str, required=True, help="Folder containing one score CSV per slide.")
    parser.add_argument(
        "--reference_feature_dir",
        type=str,
        required=True,
        help="Existing CLAM feature dir whose h5_files/coords define the patch order.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output feature dir with pt_files/ and h5_files/.")
    parser.add_argument("--expected_csv", type=str, default=None, help="Optional CSV with slide_id column to filter/check slides.")
    parser.add_argument("--patch_name_col", type=str, default="patch_name", help="CSV column containing patch names.")
    parser.add_argument("--score_csv_suffix", type=str, default="_gene_scores.csv", help="Score CSV suffix after slide_id.")
    parser.add_argument(
        "--score_columns",
        type=str,
        default=None,
        help="Optional comma-separated score columns. Defaults to all numeric columns except patch_name_col.",
    )
    parser.add_argument("--pt_only", action="store_true", help="Only write CLAM pt_files, skip h5_files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing slide .pt/.h5 outputs.")
    main(parser.parse_args())
