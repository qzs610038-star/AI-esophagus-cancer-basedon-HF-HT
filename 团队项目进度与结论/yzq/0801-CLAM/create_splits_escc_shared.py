import argparse
import os

import numpy as np
import pandas as pd


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def save_split_csv(split_ids, filename):
    max_len = max(len(ids) for ids in split_ids.values())
    data = {}
    for split_name, ids in split_ids.items():
        padded = list(ids) + [np.nan] * (max_len - len(ids))
        data[split_name] = padded
    pd.DataFrame(data).to_csv(filename, index=False)


def save_bool_split_csv(slide_ids, split_ids, filename):
    split_lookup = {}
    for split_name, ids in split_ids.items():
        for slide_id in ids:
            split_lookup[slide_id] = split_name

    rows = []
    for slide_id in slide_ids:
        split_name = split_lookup[slide_id]
        rows.append({
            "slide_id": slide_id,
            "train": split_name == "train",
            "val": split_name == "val",
            "test": split_name == "test",
        })
    pd.DataFrame(rows).to_csv(filename, index=False)


def save_descriptor(patient_df, split_patient_ids, filename):
    rows = []
    for split_name, patient_ids in split_patient_ids.items():
        split_df = patient_df[patient_df["case_id"].isin(patient_ids)]
        row = {"split": split_name, "patients": len(split_df), "slides": int(split_df["slide_count"].sum())}
        for col in ["pCR", "MPR", "strat_label"]:
            counts = split_df[col].value_counts().sort_index()
            for key, value in counts.items():
                row[f"{col}_{key}"] = int(value)
        rows.append(row)
    pd.DataFrame(rows).fillna(0).to_csv(filename, index=False)


def main(args):
    slide_df = pd.read_csv(args.csv_path)
    required_cols = {"case_id", "slide_id", "pCR", "MPR"}
    missing_cols = required_cols - set(slide_df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in {args.csv_path}: {sorted(missing_cols)}")

    slide_df["pCR"] = slide_df["pCR"].map(parse_bool)
    slide_df["MPR"] = slide_df["MPR"].map(parse_bool)

    patient_rows = []
    for case_id, group in slide_df.groupby("case_id", sort=False):
        pcr_values = group["pCR"].unique()
        mpr_values = group["MPR"].unique()
        if len(pcr_values) != 1 or len(mpr_values) != 1:
            raise ValueError(f"Patient {case_id} has inconsistent pCR/MPR labels across slides")
        pcr = bool(pcr_values[0])
        mpr = bool(mpr_values[0])
        patient_rows.append({
            "case_id": case_id,
            "pCR": int(pcr),
            "MPR": int(mpr),
            "strat_label": f"pCR{int(pcr)}_MPR{int(mpr)}",
            "slide_count": len(group),
        })
    patient_df = pd.DataFrame(patient_rows)

    rng = np.random.default_rng(args.seed)
    output_dir = os.path.join(args.output_root, f"{args.output_name}_{int(args.label_frac * 100)}")
    os.makedirs(output_dir, exist_ok=True)

    label_counts = patient_df["strat_label"].value_counts().sort_index()
    print("Patient-level stratification labels:")
    print(label_counts.to_string())

    for fold in range(args.k):
        train_patients = []
        val_patients = []
        test_patients = []

        for strat_label, label_df in patient_df.groupby("strat_label", sort=True):
            patient_ids = label_df["case_id"].to_numpy()
            shuffled = rng.permutation(patient_ids)
            n = len(shuffled)
            val_n = int(np.round(n * args.val_frac))
            test_n = int(np.round(n * args.test_frac))
            if n > 0 and args.val_frac > 0:
                val_n = max(1, val_n)
            if n - val_n > 0 and args.test_frac > 0:
                test_n = max(1, test_n)
            if val_n + test_n >= n and n > 1:
                overflow = val_n + test_n - (n - 1)
                test_n = max(0, test_n - overflow)

            val_ids = shuffled[:val_n]
            test_ids = shuffled[val_n:val_n + test_n]
            train_ids = shuffled[val_n + test_n:]

            val_patients.extend(val_ids.tolist())
            test_patients.extend(test_ids.tolist())
            train_patients.extend(train_ids.tolist())

        split_patient_ids = {
            "train": train_patients,
            "val": val_patients,
            "test": test_patients,
        }

        split_slide_ids = {}
        for split_name, patient_ids in split_patient_ids.items():
            split_slide_ids[split_name] = slide_df[slide_df["case_id"].isin(patient_ids)]["slide_id"].tolist()

        save_split_csv(split_slide_ids, os.path.join(output_dir, f"splits_{fold}.csv"))
        save_bool_split_csv(slide_df["slide_id"].tolist(), split_slide_ids, os.path.join(output_dir, f"splits_{fold}_bool.csv"))
        save_descriptor(patient_df, split_patient_ids, os.path.join(output_dir, f"splits_{fold}_descriptor.csv"))

    print(f"Saved {args.k} shared splits to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create shared ESCC pCR/MPR patient-level stratified splits")
    parser.add_argument("--csv_path", type=str, default="dataset_csv/ESCC_clam_metadata.csv")
    parser.add_argument("--output_root", type=str, default="splits")
    parser.add_argument("--output_name", type=str, default="ESCC_shared_pCR_MPR")
    parser.add_argument("--label_frac", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    main(parser.parse_args())
