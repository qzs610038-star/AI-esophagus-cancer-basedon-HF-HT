import argparse
from pathlib import Path

import h5py
import numpy as np
import torch


def load_pt(path):
    features = torch.load(path, map_location="cpu").detach().cpu().float()
    if features.ndim != 2:
        raise ValueError(f"Expected 2D tensor in {path}, got shape {tuple(features.shape)}")
    return features


def load_h5(path):
    with h5py.File(path, "r") as h5_file:
        if "features" not in h5_file or "coords" not in h5_file:
            raise KeyError(f"{path} must contain features and coords datasets")
        features = np.asarray(h5_file["features"][:], dtype=np.float32)
        coords = np.asarray(h5_file["coords"][:], dtype=np.int32)
    return features, coords


def concat_slide(slide_id, feature_dir_a, feature_dir_b, output_dir, overwrite=False, write_h5=True):
    pt_a = feature_dir_a / "pt_files" / f"{slide_id}.pt"
    pt_b = feature_dir_b / "pt_files" / f"{slide_id}.pt"
    h5_a = feature_dir_a / "h5_files" / f"{slide_id}.h5"
    h5_b = feature_dir_b / "h5_files" / f"{slide_id}.h5"

    out_pt = output_dir / "pt_files" / f"{slide_id}.pt"
    out_h5 = output_dir / "h5_files" / f"{slide_id}.h5"
    if out_pt.exists() and (not write_h5 or out_h5.exists()) and not overwrite:
        return {"slide_id": slide_id, "status": "skipped_existing", "patches": None, "feature_dim": None}

    features_a = load_pt(pt_a)
    features_b = load_pt(pt_b)
    if features_a.shape[0] != features_b.shape[0]:
        raise ValueError(f"{slide_id}: patch counts differ: {features_a.shape[0]} vs {features_b.shape[0]}")

    features = torch.cat([features_a, features_b], dim=1)
    torch.save(features, out_pt)

    if write_h5:
        h5_features_a, coords_a = load_h5(h5_a)
        h5_features_b, coords_b = load_h5(h5_b)
        if not np.array_equal(coords_a, coords_b):
            raise ValueError(f"{slide_id}: h5 coords differ between input feature dirs")
        if h5_features_a.shape[0] != h5_features_b.shape[0]:
            raise ValueError(f"{slide_id}: h5 patch counts differ")

        h5_features = np.concatenate([h5_features_a, h5_features_b], axis=1).astype(np.float32)
        with h5py.File(out_h5, "w") as h5_file:
            h5_file.create_dataset(
                "features",
                data=h5_features,
                chunks=(min(1024, len(h5_features)), h5_features.shape[1]),
            )
            h5_file.create_dataset(
                "coords",
                data=coords_a,
                chunks=(min(1024, len(coords_a)), 2),
            )

    return {
        "slide_id": slide_id,
        "status": "converted",
        "patches": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
    }


def main(args):
    feature_dir_a = Path(args.feature_dir_a)
    feature_dir_b = Path(args.feature_dir_b)
    output_dir = Path(args.output_dir)
    (output_dir / "pt_files").mkdir(parents=True, exist_ok=True)
    if not args.pt_only:
        (output_dir / "h5_files").mkdir(parents=True, exist_ok=True)

    slide_ids_a = {path.stem for path in (feature_dir_a / "pt_files").glob("*.pt")}
    slide_ids_b = {path.stem for path in (feature_dir_b / "pt_files").glob("*.pt")}
    common_slide_ids = sorted(slide_ids_a & slide_ids_b)
    missing_in_b = sorted(slide_ids_a - slide_ids_b)
    missing_in_a = sorted(slide_ids_b - slide_ids_a)

    print(f"Feature dir A: {feature_dir_a}")
    print(f"Feature dir B: {feature_dir_b}")
    print(f"Output dir: {output_dir}")
    print(f"Slides to concatenate: {len(common_slide_ids)}")
    print(f"Missing in B: {len(missing_in_b)}")
    print(f"Missing in A: {len(missing_in_a)}")

    records = []
    for idx, slide_id in enumerate(common_slide_ids, start=1):
        print(f"[{idx}/{len(common_slide_ids)}] {slide_id}")
        record = concat_slide(
            slide_id=slide_id,
            feature_dir_a=feature_dir_a,
            feature_dir_b=feature_dir_b,
            output_dir=output_dir,
            overwrite=args.overwrite,
            write_h5=not args.pt_only,
        )
        records.append(record)
        print(f"  {record['status']}: patches={record['patches']} dim={record['feature_dim']}")

    import pandas as pd

    pd.DataFrame(records).to_csv(output_dir / "concat_summary.csv", index=False)
    pd.DataFrame({"missing_in_b": missing_in_b}).to_csv(output_dir / "missing_in_b.csv", index=False)
    pd.DataFrame({"missing_in_a": missing_in_a}).to_csv(output_dir / "missing_in_a.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate two CLAM feature directories along feature dimension.")
    parser.add_argument("--feature_dir_a", required=True, help="First CLAM feature directory.")
    parser.add_argument("--feature_dir_b", required=True, help="Second CLAM feature directory.")
    parser.add_argument("--output_dir", required=True, help="Output CLAM feature directory.")
    parser.add_argument("--pt_only", action="store_true", help="Only write pt_files, skip h5_files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    main(parser.parse_args())
