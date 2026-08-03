import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch


COORD_RE = re.compile(r"_x(-?\d+)_y(-?\d+)\.pt$", re.IGNORECASE)


def parse_coord(path):
    match = COORD_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot parse x/y coordinates from file name: {path.name}")
    return int(match.group(1)), int(match.group(2))


def load_feature(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("features", "feature", "feat", "embedding", "emb"):
            if key in obj:
                obj = obj[key]
                break
        else:
            tensor_values = [value for value in obj.values() if torch.is_tensor(value)]
            if len(tensor_values) != 1:
                raise ValueError(f"Cannot identify feature tensor in dict file: {path}")
            obj = tensor_values[0]

    if not torch.is_tensor(obj):
        raise TypeError(f"Expected a tensor or tensor dict in {path}, got {type(obj)}")

    feat = obj.detach().cpu().float().squeeze()
    if feat.ndim != 1:
        raise ValueError(f"Expected one 1D feature per patch in {path}, got shape {tuple(feat.shape)}")
    return feat


def expected_slide_ids(csv_path):
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path)
    if "slide_id" not in df.columns:
        raise ValueError(f"{csv_path} does not contain a slide_id column")
    return set(df["slide_id"].astype(str))


def convert_slide(slide_dir, output_pt_dir, output_h5_dir, overwrite=False, write_h5=True):
    slide_id = slide_dir.name
    pt_output = output_pt_dir / f"{slide_id}.pt"
    h5_output = output_h5_dir / f"{slide_id}.h5"

    if pt_output.exists() and (not write_h5 or h5_output.exists()) and not overwrite:
        return {"slide_id": slide_id, "status": "skipped_existing", "patches": None, "feature_dim": None}

    patch_files = sorted(slide_dir.glob("*.pt"), key=lambda path: (*parse_coord(path)[::-1], path.name))
    if not patch_files:
        return {"slide_id": slide_id, "status": "no_patch_pt_files", "patches": 0, "feature_dim": None}

    features = []
    coords = []
    feature_dim = None
    for patch_file in patch_files:
        x, y = parse_coord(patch_file)
        feat = load_feature(patch_file)
        if feature_dim is None:
            feature_dim = int(feat.numel())
        elif int(feat.numel()) != feature_dim:
            raise ValueError(
                f"Inconsistent feature dimension in {slide_id}: expected {feature_dim}, "
                f"got {int(feat.numel())} from {patch_file.name}"
            )
        features.append(feat)
        coords.append((x, y))

    feature_tensor = torch.stack(features, dim=0)
    torch.save(feature_tensor, pt_output)

    if write_h5:
        # Use tolist() instead of tensor.numpy() to avoid PyTorch/NumPy ABI issues in some Windows envs.
        feature_array = np.asarray(feature_tensor.tolist(), dtype=np.float32)
        coord_array = np.asarray(coords, dtype=np.int32)
        with h5py.File(h5_output, "w") as h5_file:
            h5_file.create_dataset("features", data=feature_array, chunks=(min(1024, len(feature_array)), feature_dim))
            h5_file.create_dataset("coords", data=coord_array, chunks=(min(1024, len(coord_array)), 2))

    return {"slide_id": slide_id, "status": "converted", "patches": len(patch_files), "feature_dim": feature_dim}


def main(args):
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_pt_dir = output_dir / "pt_files"
    output_h5_dir = output_dir / "h5_files"
    output_pt_dir.mkdir(parents=True, exist_ok=True)
    if not args.pt_only:
        output_h5_dir.mkdir(parents=True, exist_ok=True)

    expected_ids = expected_slide_ids(args.expected_csv)
    slide_dirs = sorted([path for path in input_root.iterdir() if path.is_dir()], key=lambda path: path.name)
    if expected_ids is not None:
        slide_dirs = [path for path in slide_dirs if path.name in expected_ids]

    print(f"Input root: {input_root}")
    print(f"Output dir: {output_dir}")
    print(f"Slides to convert: {len(slide_dirs)}")

    records = []
    for idx, slide_dir in enumerate(slide_dirs, start=1):
        print(f"[{idx}/{len(slide_dirs)}] {slide_dir.name}")
        record = convert_slide(
            slide_dir=slide_dir,
            output_pt_dir=output_pt_dir,
            output_h5_dir=output_h5_dir,
            overwrite=args.overwrite,
            write_h5=not args.pt_only,
        )
        records.append(record)
        print(f"  {record['status']}: patches={record['patches']} dim={record['feature_dim']}")

    summary = pd.DataFrame(records)
    summary.to_csv(output_dir / "conversion_summary.csv", index=False)

    if expected_ids is not None:
        converted_or_seen = {path.name for path in slide_dirs}
        missing = sorted(expected_ids - converted_or_seen)
        extra = sorted({path.name for path in input_root.iterdir() if path.is_dir()} - expected_ids)
        pd.DataFrame({"missing_slide_id": missing}).to_csv(output_dir / "missing_from_input.csv", index=False)
        pd.DataFrame({"extra_slide_id": extra}).to_csv(output_dir / "extra_not_in_expected_csv.csv", index=False)
        print(f"Expected slides: {len(expected_ids)}")
        print(f"Missing expected slide folders: {len(missing)}")
        print(f"Extra slide folders not in expected csv: {len(extra)}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert per-patch .pt feature files into CLAM feature directory format."
    )
    parser.add_argument("--input_root", type=str, required=True, help="Folder containing one subfolder per slide.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output feature dir with pt_files/ and h5_files/.")
    parser.add_argument("--expected_csv", type=str, default=None, help="Optional CSV with slide_id column to filter/check slides.")
    parser.add_argument("--pt_only", action="store_true", help="Only write CLAM pt_files, skip h5_files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing slide .pt/.h5 outputs.")
    main(parser.parse_args())
