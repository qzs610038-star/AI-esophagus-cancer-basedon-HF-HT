import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def load_slide_feature(pt_path, pooling="mean"):
    features = torch.load(pt_path, map_location="cpu").detach().cpu().float()
    if features.ndim != 2:
        raise ValueError(f"Expected 2D tensor in {pt_path}, got shape {tuple(features.shape)}")
    if pooling == "mean":
        pooled = features.mean(dim=0)
    elif pooling == "max":
        pooled = features.max(dim=0).values
    elif pooling == "mean_std":
        pooled = torch.cat([features.mean(dim=0), features.std(dim=0, unbiased=False)], dim=0)
    elif pooling == "mean_max":
        pooled = torch.cat([features.mean(dim=0), features.max(dim=0).values], dim=0)
    else:
        raise ValueError(f"Unsupported pooling: {pooling}")
    return pooled.numpy().astype(np.float32)


def pool_array(features, pooling="mean"):
    if features.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {features.shape}")
    if pooling == "mean":
        return features.mean(axis=0)
    if pooling == "max":
        return features.max(axis=0)
    if pooling == "mean_std":
        return np.concatenate([features.mean(axis=0), features.std(axis=0)])
    if pooling == "mean_max":
        return np.concatenate([features.mean(axis=0), features.max(axis=0)])
    raise ValueError(f"Unsupported pooling: {pooling}")


def load_patch_array(pt_path):
    features = torch.load(pt_path, map_location="cpu").detach().cpu().float()
    if features.ndim != 2:
        raise ValueError(f"Expected 2D tensor in {pt_path}, got shape {tuple(features.shape)}")
    return features.numpy().astype(np.float32)


def build_slide_table(csv_path, pt_dir, pooling):
    df = pd.read_csv(csv_path)
    if not {"case_id", "slide_id", "label"}.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain case_id, slide_id, label columns")

    label_values = sorted(df["label"].unique().tolist())
    if len(label_values) != 2:
        raise ValueError(f"Expected binary labels, got {label_values}")
    # Keep the positive class as the second label from the task-specific mapping below.
    label_map = {"non_pCR": 0, "pCR": 1, "non_MPR": 0, "MPR": 1}
    if not set(label_values).issubset(label_map):
        raise ValueError(f"Unsupported labels {label_values}; update label_map if needed")

    rows = []
    features = []
    for _, row in df.iterrows():
        slide_id = str(row["slide_id"])
        pt_path = pt_dir / f"{slide_id}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Missing feature file: {pt_path}")
        rows.append({
            "case_id": str(row["case_id"]),
            "slide_id": slide_id,
            "label": row["label"],
            "Y": label_map[row["label"]],
        })
        features.append(load_slide_feature(pt_path, pooling=pooling))

    table = pd.DataFrame(rows)
    x_slide = np.vstack(features)
    return table, x_slide


def build_multimodal_slide_table(csv_path, path_pt_dir, gene_pt_dir, path_pooling, gene_pooling):
    df = pd.read_csv(csv_path)
    if not {"case_id", "slide_id", "label"}.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain case_id, slide_id, label columns")

    label_values = sorted(df["label"].unique().tolist())
    if len(label_values) != 2:
        raise ValueError(f"Expected binary labels, got {label_values}")
    label_map = {"non_pCR": 0, "pCR": 1, "non_MPR": 0, "MPR": 1}
    if not set(label_values).issubset(label_map):
        raise ValueError(f"Unsupported labels {label_values}; update label_map if needed")

    rows = []
    path_features = []
    gene_features = []
    for _, row in df.iterrows():
        slide_id = str(row["slide_id"])
        path_pt_path = path_pt_dir / f"{slide_id}.pt"
        gene_pt_path = gene_pt_dir / f"{slide_id}.pt"
        if not path_pt_path.exists():
            raise FileNotFoundError(f"Missing pathology feature file: {path_pt_path}")
        if not gene_pt_path.exists():
            raise FileNotFoundError(f"Missing gene feature file: {gene_pt_path}")
        rows.append({
            "case_id": str(row["case_id"]),
            "slide_id": slide_id,
            "label": row["label"],
            "Y": label_map[row["label"]],
        })
        path_features.append(load_slide_feature(path_pt_path, pooling=path_pooling))
        gene_features.append(load_slide_feature(gene_pt_path, pooling=gene_pooling))

    table = pd.DataFrame(rows)
    x_path_slide = np.vstack(path_features)
    x_gene_slide = np.vstack(gene_features)
    return table, x_path_slide, x_gene_slide


def aggregate_patients(table, x_slide):
    patient_rows = []
    patient_features = []
    for case_id, idx in table.groupby("case_id", sort=False).groups.items():
        indices = np.array(list(idx))
        labels = table.loc[indices, "Y"].unique()
        if len(labels) != 1:
            raise ValueError(f"Patient {case_id} has inconsistent labels: {labels}")
        patient_rows.append({"case_id": case_id, "Y": int(labels[0]), "n_slides": len(indices)})
        patient_features.append(x_slide[indices].mean(axis=0))
    return pd.DataFrame(patient_rows), np.vstack(patient_features)


def aggregate_patient_features(table, x_slide):
    patient_features = []
    for _, idx in table.groupby("case_id", sort=False).groups.items():
        indices = np.array(list(idx))
        patient_features.append(x_slide[indices].mean(axis=0))
    return np.vstack(patient_features)


def fit_pre_pool_path_pca(slide_ids, path_pt_dir, n_components, max_patches, seed):
    rng = np.random.default_rng(seed)
    sampled = []
    counts = []
    slide_ids = list(slide_ids)
    for slide_id in slide_ids:
        arr = load_patch_array(path_pt_dir / f"{slide_id}.pt")
        counts.append(arr.shape[0])

    total_patches = int(np.sum(counts))
    if max_patches <= 0 or max_patches >= total_patches:
        sample_counts = counts
    else:
        sample_counts = []
        remaining = max_patches
        remaining_total = total_patches
        for count in counts:
            if remaining <= 0:
                sample_counts.append(0)
                remaining_total -= count
                continue
            take = int(round(remaining * count / remaining_total)) if remaining_total > 0 else 0
            take = min(count, max(0, take))
            sample_counts.append(take)
            remaining -= take
            remaining_total -= count

    for slide_id, take in zip(slide_ids, sample_counts):
        if take <= 0:
            continue
        arr = load_patch_array(path_pt_dir / f"{slide_id}.pt")
        if take < arr.shape[0]:
            idx = rng.choice(arr.shape[0], size=take, replace=False)
            arr = arr[idx]
        sampled.append(arr)

    if not sampled:
        raise ValueError("No patches sampled for pre-pooling PCA")

    x_sample = np.vstack(sampled).astype(np.float32)
    n_components = min(int(n_components), x_sample.shape[0], x_sample.shape[1])
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=seed)
    x_scaled = scaler.fit_transform(x_sample)
    pca.fit(x_scaled)
    return scaler, pca


def transform_path_slide_pre_pool_pca(slide_ids, path_pt_dir, pooling, scaler, pca, batch_size):
    features = []
    for slide_id in slide_ids:
        arr = load_patch_array(path_pt_dir / f"{slide_id}.pt")
        chunks = []
        for start in range(0, arr.shape[0], batch_size):
            chunk = arr[start:start + batch_size]
            chunks.append(pca.transform(scaler.transform(chunk)).astype(np.float32))
        reduced = np.vstack(chunks)
        features.append(pool_array(reduced, pooling=pooling).astype(np.float32))
    return np.vstack(features)


def split_slide_ids(split_csv, split_name):
    split_df = pd.read_csv(split_csv)
    if split_name not in split_df.columns:
        raise ValueError(f"{split_csv} does not contain column {split_name}")
    return set(split_df[split_name].dropna().astype(str))


def evaluate(y_true, score, pred=None):
    if pred is None:
        pred = (score >= 0.5).astype(int)
    out = {
        "n": int(len(y_true)),
        "auc": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) > 1 else np.nan,
        "acc": float(accuracy_score(y_true, pred)),
    }
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    out.update({"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])})
    return out, pred


def predict_scores(model, x):
    pred = model.predict(x)

    prob = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError(f"Model classes do not contain positive class 1: {classes}")
        prob = proba[:, classes.index(1)]

    if hasattr(model, "decision_function"):
        score = model.decision_function(x)
        if np.ndim(score) == 2:
            classes = list(model.classes_)
            if 1 not in classes:
                raise ValueError(f"Model classes do not contain positive class 1: {classes}")
            score = score[:, classes.index(1)]
    elif prob is not None:
        score = prob
    else:
        score = pred

    return np.asarray(score), np.asarray(pred), None if prob is None else np.asarray(prob)


def parse_optional_int(value):
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return None
    return int(value)


def parse_gamma(value):
    if isinstance(value, str) and value.lower() in {"scale", "auto"}:
        return value.lower()
    return float(value)


def make_model(args, params):
    class_weight = "balanced" if args.class_weight_balanced else None

    if args.classifier == "logreg":
        model = LogisticRegression(
            class_weight=class_weight,
            C=float(params.get("C", args.C)),
            max_iter=args.max_iter,
            solver="liblinear",
            random_state=args.seed,
        )
        return make_pipeline(StandardScaler(), model)

    if args.classifier == "linear_svm":
        model = SVC(
            kernel="linear",
            C=float(params.get("C", args.C)),
            class_weight=class_weight,
            probability=False,
            random_state=args.seed,
        )
        return make_pipeline(StandardScaler(), model)

    if args.classifier == "rbf_svm":
        model = SVC(
            kernel="rbf",
            C=float(params.get("C", args.C)),
            gamma=params.get("gamma", args.gamma),
            class_weight=class_weight,
            probability=False,
            random_state=args.seed,
        )
        return make_pipeline(StandardScaler(), model)

    if args.classifier == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", args.rf_n_estimators)),
            max_depth=params.get("max_depth", args.rf_max_depth),
            min_samples_leaf=int(params.get("min_samples_leaf", args.rf_min_samples_leaf)),
            class_weight=class_weight,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )

    if args.classifier == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is not installed. Install it or choose another --classifier.") from exc

        return XGBClassifier(
            n_estimators=int(params.get("n_estimators", args.xgb_n_estimators)),
            max_depth=int(params.get("max_depth", args.xgb_max_depth)),
            learning_rate=float(params.get("learning_rate", args.xgb_learning_rate)),
            subsample=float(params.get("subsample", args.xgb_subsample)),
            colsample_bytree=float(params.get("colsample_bytree", args.xgb_colsample_bytree)),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )

    raise ValueError(f"Unsupported classifier: {args.classifier}")


def default_params(args):
    if args.classifier in {"logreg", "linear_svm"}:
        return {"C": args.C}
    if args.classifier == "rbf_svm":
        return {"C": args.C, "gamma": args.gamma}
    if args.classifier == "random_forest":
        return {
            "n_estimators": args.rf_n_estimators,
            "max_depth": args.rf_max_depth,
            "min_samples_leaf": args.rf_min_samples_leaf,
        }
    if args.classifier == "xgboost":
        return {
            "n_estimators": args.xgb_n_estimators,
            "max_depth": args.xgb_max_depth,
            "learning_rate": args.xgb_learning_rate,
            "subsample": args.xgb_subsample,
            "colsample_bytree": args.xgb_colsample_bytree,
        }
    raise ValueError(f"Unsupported classifier: {args.classifier}")


def tuning_grid(args):
    if args.classifier in {"logreg", "linear_svm"}:
        return [{"C": float(C)} for C in args.C_grid]

    if args.classifier == "rbf_svm":
        return [
            {"C": float(C), "gamma": parse_gamma(gamma)}
            for C, gamma in itertools.product(args.C_grid, args.gamma_grid)
        ]

    if args.classifier == "random_forest":
        return [
            {"n_estimators": int(n), "max_depth": parse_optional_int(depth), "min_samples_leaf": int(leaf)}
            for n, depth, leaf in itertools.product(
                args.rf_n_estimators_grid, args.rf_max_depth_grid, args.rf_min_samples_leaf_grid
            )
        ]

    if args.classifier == "xgboost":
        return [
            {
                "n_estimators": int(n),
                "max_depth": int(depth),
                "learning_rate": float(lr),
                "subsample": float(subsample),
                "colsample_bytree": float(colsample),
            }
            for n, depth, lr, subsample, colsample in itertools.product(
                args.xgb_n_estimators_grid,
                args.xgb_max_depth_grid,
                args.xgb_learning_rate_grid,
                args.xgb_subsample_grid,
                args.xgb_colsample_bytree_grid,
            )
        ]

    raise ValueError(f"Unsupported classifier: {args.classifier}")


def get_train_data(args, split_name, split_slide_sets, split_patient_sets, table, x_slide,
                   patient_table, x_patient, slide_index, patient_index):
    if args.train_level == "patient":
        patient_ids = sorted(split_patient_sets[split_name])
        idx = [patient_index[c] for c in patient_ids]
        x = x_patient[idx]
        y = patient_table.loc[idx, "Y"].to_numpy()
    else:
        slide_ids = sorted(split_slide_sets[split_name])
        idx = [slide_index[s] for s in slide_ids]
        x = x_slide[idx]
        y = table.loc[idx, "Y"].to_numpy()
    return x, y


def is_multimodal(args):
    return args.path_feature_dir is not None or args.gene_feature_dir is not None


def validate_multimodal_args(args):
    if not is_multimodal(args):
        return
    if args.path_feature_dir is None or args.gene_feature_dir is None:
        raise ValueError("Use both --path_feature_dir and --gene_feature_dir for multimodal baselines.")
    if args.path_pca_dim < 0:
        raise ValueError("--path_pca_dim must be >= 0")
    if args.path_pca_stage == "pre_pool" and args.path_pca_dim == 0:
        raise ValueError("--path_pca_stage pre_pool requires --path_pca_dim > 0")


def validate_gene_transform_args(args):
    if args.gene_feature_transform == "none":
        return
    if args.concat_path_dim <= 0 or args.concat_gene_dim <= 0:
        raise ValueError("--concat_path_dim and --concat_gene_dim must be positive.")
    if args.gene_kernel_n_components <= 0:
        raise ValueError("--gene_kernel_n_components must be positive.")
    if args.gene_kernel_gamma is not None and args.gene_kernel_gamma <= 0:
        raise ValueError("--gene_kernel_gamma must be positive when set.")


def split_indices(args, split_name, split_slide_sets, split_patient_sets, patient_index, slide_index):
    if args.train_level == "patient":
        patient_ids = sorted(split_patient_sets[split_name])
        return [patient_index[c] for c in patient_ids]
    slide_ids = sorted(split_slide_sets[split_name])
    return [slide_index[s] for s in slide_ids]


def split_concatenated_path_gene(x, path_dim, gene_dim, pooling, name):
    expected_dim = int(path_dim) + int(gene_dim)
    if x.shape[1] == expected_dim:
        return x[:, :path_dim], x[:, path_dim:path_dim + gene_dim]

    if pooling in {"mean_std", "mean_max"} and x.shape[1] == expected_dim * 2:
        first = x[:, :expected_dim]
        second = x[:, expected_dim:]
        x_path = np.hstack([first[:, :path_dim], second[:, :path_dim]])
        x_gene = np.hstack([
            first[:, path_dim:path_dim + gene_dim],
            second[:, path_dim:path_dim + gene_dim],
        ])
        return x_path, x_gene

    raise ValueError(
        f"{name} has feature dimension {x.shape[1]}, but expected {expected_dim} for mean/max pooling "
        f"or {expected_dim * 2} for mean_std/mean_max pooling based on --concat_path_dim "
        f"{path_dim} and --concat_gene_dim {gene_dim}."
    )


def fit_gene_feature_transform(args, x_gene_fit, seed):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_gene_fit)
    mode = args.gene_feature_transform

    if mode == "poly2":
        transformer = PolynomialFeatures(
            degree=2,
            interaction_only=args.gene_poly_interaction_only,
            include_bias=False,
        )
        transformer.fit(x_scaled)
        output_dim = int(transformer.transform(x_scaled[:1]).shape[1])
        return {
            "mode": mode,
            "scaler": scaler,
            "transformer": transformer,
            "kernel_gamma": np.nan,
            "kernel_components": 0,
            "output_dim": output_dim,
        }

    gamma = args.gene_kernel_gamma
    if gamma is None:
        gamma = 1.0 / max(1, x_gene_fit.shape[1])

    if mode == "rbf_nystroem":
        n_components = min(int(args.gene_kernel_n_components), x_scaled.shape[0])
        transformer = Nystroem(
            kernel="rbf",
            gamma=gamma,
            n_components=n_components,
            random_state=seed,
        )
    elif mode == "rbf_sampler":
        n_components = int(args.gene_kernel_n_components)
        transformer = RBFSampler(
            gamma=gamma,
            n_components=n_components,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported gene feature transform: {mode}")

    transformer.fit(x_scaled)
    output_dim = int(transformer.transform(x_scaled[:1]).shape[1])
    if not args.gene_kernel_only:
        output_dim += int(x_scaled.shape[1])
    return {
        "mode": mode,
        "scaler": scaler,
        "transformer": transformer,
        "kernel_gamma": float(gamma),
        "kernel_components": int(n_components),
        "output_dim": output_dim,
    }


def apply_gene_feature_transform(bundle, x_gene):
    if bundle is None:
        return x_gene
    x_scaled = bundle["scaler"].transform(x_gene)
    x_transformed = bundle["transformer"].transform(x_scaled)
    if bundle["mode"] in {"rbf_nystroem", "rbf_sampler"} and not bundle["kernel_only"]:
        x_transformed = np.hstack([x_scaled, x_transformed])
    return x_transformed.astype(np.float32)


def transform_gene_block(args, x_gene_slide, x_gene_patient, fit_idx, seed):
    if args.gene_feature_transform == "none":
        return x_gene_slide, x_gene_patient, {}

    x_gene_fit = x_gene_patient[fit_idx] if args.train_level == "patient" else x_gene_slide[fit_idx]
    bundle = fit_gene_feature_transform(args, x_gene_fit, seed)
    bundle["kernel_only"] = bool(args.gene_kernel_only)

    x_gene_slide_out = apply_gene_feature_transform(bundle, x_gene_slide)
    x_gene_patient_out = apply_gene_feature_transform(bundle, x_gene_patient)
    info = {
        "gene_feature_transform": args.gene_feature_transform,
        "gene_input_dim": int(x_gene_slide.shape[1]),
        "gene_output_dim": int(x_gene_slide_out.shape[1]),
    }
    if args.gene_feature_transform == "poly2":
        info["gene_poly_interaction_only"] = bool(args.gene_poly_interaction_only)
    else:
        info["gene_kernel_gamma"] = bundle["kernel_gamma"]
        info["gene_kernel_components"] = bundle["kernel_components"]
        info["gene_kernel_only"] = bool(args.gene_kernel_only)
    return x_gene_slide_out, x_gene_patient_out, info


def prepare_multimodal_fold_features(args, split_slide_sets, split_patient_sets, patient_index, slide_index,
                                     x_path_slide, x_gene_slide, x_path_patient, x_gene_patient):
    pca_info = {"path_pca_dim": int(args.path_pca_dim), "path_pca_components": 0}
    fit_idx = split_indices(args, "train", split_slide_sets, split_patient_sets, patient_index, slide_index)
    if args.path_pca_dim > 0:
        fit_source = x_path_patient[fit_idx] if args.train_level == "patient" else x_path_slide[fit_idx]
        n_components = min(int(args.path_pca_dim), fit_source.shape[0], fit_source.shape[1])
        if n_components < 1:
            raise ValueError(f"Cannot fit PCA with shape {fit_source.shape}")

        scaler = StandardScaler()
        pca = PCA(n_components=n_components, random_state=args.seed)
        pca.fit(scaler.fit_transform(fit_source))
        x_path_slide_out = pca.transform(scaler.transform(x_path_slide)).astype(np.float32)
        x_path_patient_out = pca.transform(scaler.transform(x_path_patient)).astype(np.float32)
        pca_info["path_pca_components"] = int(n_components)
        pca_info["path_pca_explained_variance_ratio_sum"] = float(pca.explained_variance_ratio_.sum())
    else:
        x_path_slide_out = x_path_slide
        x_path_patient_out = x_path_patient

    x_gene_slide_out, x_gene_patient_out, gene_info = transform_gene_block(
        args,
        x_gene_slide=x_gene_slide,
        x_gene_patient=x_gene_patient,
        fit_idx=fit_idx,
        seed=args.seed,
    )

    x_slide = np.hstack([x_path_slide_out, x_gene_slide_out]).astype(np.float32)
    x_patient = np.hstack([x_path_patient_out, x_gene_patient_out]).astype(np.float32)
    pca_info["slide_feature_dim"] = int(x_slide.shape[1])
    pca_info["patient_feature_dim"] = int(x_patient.shape[1])
    pca_info.update(gene_info)
    return x_slide, x_patient, pca_info


def main(args):
    validate_multimodal_args(args)
    validate_gene_transform_args(args)
    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_multimodal(args):
        path_pooling = args.path_pooling or args.pooling
        gene_pooling = args.gene_pooling or args.pooling
        path_pt_dir = Path(args.data_root_dir) / args.path_feature_dir / "pt_files"
        gene_pt_dir = Path(args.data_root_dir) / args.gene_feature_dir / "pt_files"
        table, x_path_slide, x_gene_slide = build_multimodal_slide_table(
            args.csv_path,
            path_pt_dir=path_pt_dir,
            gene_pt_dir=gene_pt_dir,
            path_pooling=path_pooling,
            gene_pooling=gene_pooling,
        )
        patient_table, x_path_patient = aggregate_patients(table, x_path_slide)
        x_gene_patient = aggregate_patient_features(table, x_gene_slide)
        x_slide = np.hstack([x_path_slide, x_gene_slide]).astype(np.float32)
        x_patient = np.hstack([x_path_patient, x_gene_patient]).astype(np.float32)
    else:
        pt_dir = Path(args.data_root_dir) / args.feature_dir / "pt_files"
        table, x_slide = build_slide_table(args.csv_path, pt_dir, pooling=args.pooling)
        patient_table, x_patient = aggregate_patients(table, x_slide)
        if args.gene_feature_transform != "none":
            x_path_slide_concat, x_gene_slide_concat = split_concatenated_path_gene(
                x_slide,
                path_dim=args.concat_path_dim,
                gene_dim=args.concat_gene_dim,
                pooling=args.pooling,
                name="slide features",
            )
            x_path_patient_concat, x_gene_patient_concat = split_concatenated_path_gene(
                x_patient,
                path_dim=args.concat_path_dim,
                gene_dim=args.concat_gene_dim,
                pooling=args.pooling,
                name="patient features",
            )

    slide_index = {slide_id: i for i, slide_id in enumerate(table["slide_id"])}
    patient_index = {case_id: i for i, case_id in enumerate(patient_table["case_id"])}

    summary_rows = []
    for fold in range(args.k):
        split_csv = split_dir / f"splits_{fold}.csv"
        split_slide_sets = {name: split_slide_ids(split_csv, name) for name in ["train", "val", "test"]}
        split_patient_sets = {}
        for split_name, slide_ids in split_slide_sets.items():
            cases = set(table.loc[table["slide_id"].isin(slide_ids), "case_id"].astype(str))
            split_patient_sets[split_name] = cases

        if is_multimodal(args):
            if args.path_pca_stage == "pre_pool":
                train_slide_ids = sorted(split_slide_sets["train"])
                scaler, pca = fit_pre_pool_path_pca(
                    slide_ids=train_slide_ids,
                    path_pt_dir=path_pt_dir,
                    n_components=args.path_pca_dim,
                    max_patches=args.path_pca_max_patches,
                    seed=args.seed + fold,
                )
                ordered_slide_ids = table["slide_id"].astype(str).tolist()
                x_path_slide_pre = transform_path_slide_pre_pool_pca(
                    slide_ids=ordered_slide_ids,
                    path_pt_dir=path_pt_dir,
                    pooling=path_pooling,
                    scaler=scaler,
                    pca=pca,
                    batch_size=args.path_pca_batch_size,
                )
                x_path_patient_pre = aggregate_patient_features(table, x_path_slide_pre)
                fit_idx = split_indices(
                    args,
                    "train",
                    split_slide_sets,
                    split_patient_sets,
                    patient_index,
                    slide_index,
                )
                x_gene_slide_out, x_gene_patient_out, gene_info = transform_gene_block(
                    args,
                    x_gene_slide=x_gene_slide,
                    x_gene_patient=x_gene_patient,
                    fit_idx=fit_idx,
                    seed=args.seed + fold,
                )
                x_slide_fold = np.hstack([x_path_slide_pre, x_gene_slide_out]).astype(np.float32)
                x_patient_fold = np.hstack([x_path_patient_pre, x_gene_patient_out]).astype(np.float32)
                pca_info = {
                    "path_pca_dim": int(args.path_pca_dim),
                    "path_pca_components": int(pca.n_components_),
                    "path_pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
                    "slide_feature_dim": int(x_slide_fold.shape[1]),
                    "patient_feature_dim": int(x_patient_fold.shape[1]),
                }
                pca_info.update(gene_info)
            else:
                x_slide_fold, x_patient_fold, pca_info = prepare_multimodal_fold_features(
                    args,
                    split_slide_sets=split_slide_sets,
                    split_patient_sets=split_patient_sets,
                    patient_index=patient_index,
                    slide_index=slide_index,
                    x_path_slide=x_path_slide,
                    x_gene_slide=x_gene_slide,
                    x_path_patient=x_path_patient,
                    x_gene_patient=x_gene_patient,
                )
        else:
            if args.gene_feature_transform != "none":
                fit_idx = split_indices(
                    args,
                    "train",
                    split_slide_sets,
                    split_patient_sets,
                    patient_index,
                    slide_index,
                )
                x_gene_slide_out, x_gene_patient_out, gene_info = transform_gene_block(
                    args,
                    x_gene_slide=x_gene_slide_concat,
                    x_gene_patient=x_gene_patient_concat,
                    fit_idx=fit_idx,
                    seed=args.seed + fold,
                )
                x_slide_fold = np.hstack([x_path_slide_concat, x_gene_slide_out]).astype(np.float32)
                x_patient_fold = np.hstack([x_path_patient_concat, x_gene_patient_out]).astype(np.float32)
                pca_info = {
                    "path_pca_dim": 0,
                    "path_pca_components": 0,
                    "slide_feature_dim": int(x_slide_fold.shape[1]),
                    "patient_feature_dim": int(x_patient_fold.shape[1]),
                }
                pca_info.update(gene_info)
            else:
                x_slide_fold = x_slide
                x_patient_fold = x_patient
                pca_info = {
                    "path_pca_dim": 0,
                    "path_pca_components": 0,
                    "slide_feature_dim": int(x_slide.shape[1]),
                    "patient_feature_dim": int(x_patient.shape[1]),
                }

        x_train, y_train = get_train_data(args, "train", split_slide_sets, split_patient_sets,
                                          table, x_slide_fold, patient_table, x_patient_fold,
                                          slide_index, patient_index)

        selected_params = default_params(args)
        tuning_records = []
        use_val_tuning = args.tune_on_val or args.tune_C_on_val
        if use_val_tuning:
            x_val_for_tuning, y_val_for_tuning = get_train_data(args, "val", split_slide_sets, split_patient_sets,
                                                                table, x_slide_fold, patient_table, x_patient_fold,
                                                                slide_index, patient_index)
            best_auc = -np.inf
            best_acc = -np.inf
            for params in tuning_grid(args):
                candidate_model = make_model(args, params)
                candidate_model.fit(x_train, y_train)
                val_score, val_pred, _ = predict_scores(candidate_model, x_val_for_tuning)
                val_metrics, _ = evaluate(y_val_for_tuning, val_score, val_pred)
                tuning_records.append({
                    "fold": fold,
                    "params": json.dumps(params, sort_keys=True),
                    "val_auc": val_metrics["auc"],
                    "val_acc": val_metrics["acc"],
                })
                auc_for_selection = val_metrics["auc"]
                if np.isnan(auc_for_selection):
                    auc_for_selection = -np.inf
                if auc_for_selection > best_auc or (auc_for_selection == best_auc and val_metrics["acc"] > best_acc):
                    best_auc = auc_for_selection
                    best_acc = val_metrics["acc"]
                    selected_params = params

        if args.refit_train_val_after_tuning and use_val_tuning:
            x_val_for_refit, y_val_for_refit = get_train_data(args, "val", split_slide_sets, split_patient_sets,
                                                              table, x_slide_fold, patient_table, x_patient_fold,
                                                              slide_index, patient_index)
            x_fit = np.vstack([x_train, x_val_for_refit])
            y_fit = np.concatenate([y_train, y_val_for_refit])
        else:
            x_fit = x_train
            y_fit = y_train

        model = make_model(args, selected_params)
        model.fit(x_fit, y_fit)

        fold_patient_records = []
        fold_slide_records = []
        row = {
            "fold": fold,
            "pooling": args.pooling,
            "classifier": args.classifier,
            "train_level": args.train_level,
            "selected_params": json.dumps(selected_params, sort_keys=True),
            **pca_info,
        }
        if is_multimodal(args):
            row.update({
                "feature_mode": "multimodal",
                "path_feature_dir": args.path_feature_dir,
                "gene_feature_dir": args.gene_feature_dir,
                "path_pooling": args.path_pooling or args.pooling,
                "gene_pooling": args.gene_pooling or args.pooling,
            })
        else:
            row.update({
                "feature_mode": "single",
                "path_feature_dir": "",
                "gene_feature_dir": "",
                "path_pooling": "",
                "gene_pooling": "",
            })

        for split_name in ["train", "val", "test"]:
            patient_ids = sorted(split_patient_sets[split_name])
            if args.train_level == "patient":
                p_idx = [patient_index[c] for c in patient_ids]
                p_score, p_pred, p_prob = predict_scores(model, x_patient_fold[p_idx])
                if p_prob is None:
                    p_prob = np.full(len(p_score), np.nan, dtype=np.float32)
                p_y = patient_table.loc[p_idx, "Y"].to_numpy()
            else:
                patient_scores = []
                patient_probs = []
                patient_labels = []
                patient_preds = []
                for case_id in patient_ids:
                    case_slide_ids = table.loc[table["case_id"].astype(str) == case_id, "slide_id"].astype(str).tolist()
                    case_slide_idx = [slide_index[s] for s in case_slide_ids]
                    case_slide_score, case_slide_pred, case_slide_prob = predict_scores(model, x_slide_fold[case_slide_idx])
                    patient_scores.append(float(case_slide_score.mean()))
                    if case_slide_prob is not None:
                        patient_probs.append(float(case_slide_prob.mean()))
                    else:
                        patient_probs.append(np.nan)
                    patient_preds.append(int(np.mean(case_slide_pred) >= 0.5))
                    patient_labels.append(int(patient_table.loc[patient_index[case_id], "Y"]))
                p_score = np.asarray(patient_scores, dtype=np.float32)
                p_prob = np.asarray(patient_probs, dtype=np.float32)
                p_pred = np.asarray(patient_preds, dtype=np.int64)
                p_y = np.asarray(patient_labels, dtype=np.int64)
            p_metrics, p_pred = evaluate(p_y, p_score, p_pred)
            for key, value in p_metrics.items():
                row[f"patient_{split_name}_{key}"] = value
            for case_id, y, score, prob, pred in zip(patient_ids, p_y, p_score, p_prob, p_pred):
                fold_patient_records.append({
                    "fold": fold, "split": split_name, "case_id": case_id,
                    "Y": int(y), "Y_hat": int(pred), "score_1": float(score), "p_1": float(prob),
                    "n_slides": int(patient_table.loc[patient_index[case_id], "n_slides"]),
                })

            slide_ids = sorted(split_slide_sets[split_name])
            s_idx = [slide_index[s] for s in slide_ids]
            s_score, s_pred, s_prob = predict_scores(model, x_slide_fold[s_idx])
            if s_prob is None:
                s_prob = np.full(len(s_score), np.nan, dtype=np.float32)
            s_y = table.loc[s_idx, "Y"].to_numpy()
            s_metrics, s_pred = evaluate(s_y, s_score, s_pred)
            for key, value in s_metrics.items():
                row[f"slide_{split_name}_{key}"] = value
            for slide_id, y, score, prob, pred in zip(slide_ids, s_y, s_score, s_prob, s_pred):
                fold_slide_records.append({
                    "fold": fold, "split": split_name, "case_id": str(table.loc[slide_index[slide_id], "case_id"]),
                    "slide_id": slide_id, "Y": int(y), "Y_hat": int(pred), "score_1": float(score), "p_1": float(prob),
                })

        summary_rows.append(row)
        pd.DataFrame(fold_patient_records).to_csv(output_dir / f"fold_{fold}_patient_predictions.csv", index=False)
        pd.DataFrame(fold_slide_records).to_csv(output_dir / f"fold_{fold}_slide_predictions.csv", index=False)
        if tuning_records:
            pd.DataFrame(tuning_records).to_csv(output_dir / f"fold_{fold}_C_tuning.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    metric_cols = [
        "patient_val_auc", "patient_test_auc", "patient_val_acc", "patient_test_acc",
        "slide_val_auc", "slide_test_auc", "slide_val_acc", "slide_test_acc",
    ]
    aggregate = summary[metric_cols].agg(["mean", "std"])
    aggregate.to_csv(output_dir / "summary_mean_std.csv")

    print(f"Saved baseline outputs to: {output_dir}")
    print(summary[["fold", "pooling", "classifier", "train_level", "selected_params"] + metric_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(aggregate.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple ESCC baseline: pooled UNI2-H features + logistic regression.")
    parser.add_argument("--csv_path", default="dataset_csv/ESCC_pCR_clam.csv")
    parser.add_argument("--data_root_dir", default=r"D:\PycharmProjects\AIPath-data")
    parser.add_argument("--feature_dir", default="ESCC_uni2h_features")
    parser.add_argument("--path_feature_dir", default=None,
                        help="Optional pathology feature directory for multimodal baselines.")
    parser.add_argument("--gene_feature_dir", default=None,
                        help="Optional gene-score feature directory for multimodal baselines.")
    parser.add_argument("--split_dir", default="splits/ESCC_pCR_100")
    parser.add_argument("--output_dir", default="baseline_results/ESCC_pCR_meanpool_logreg")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--pooling", choices=["mean", "max", "mean_std", "mean_max"], default="mean")
    parser.add_argument("--path_pooling", choices=["mean", "max", "mean_std", "mean_max"], default=None,
                        help="Pooling for --path_feature_dir. Defaults to --pooling.")
    parser.add_argument("--gene_pooling", choices=["mean", "max", "mean_std", "mean_max"], default=None,
                        help="Pooling for --gene_feature_dir. Defaults to --pooling.")
    parser.add_argument("--path_pca_dim", type=int, default=0,
                        help="Fold-wise PCA dimension for pathology features before concatenating gene features. 0 disables PCA.")
    parser.add_argument("--path_pca_stage", choices=["post_pool", "pre_pool"], default="post_pool",
                        help="Apply pathology PCA after pooling (post_pool) or before pooling patch features (pre_pool).")
    parser.add_argument("--path_pca_max_patches", type=int, default=50000,
                        help="Maximum train patches sampled per fold to fit pre_pool pathology PCA. <=0 uses all train patches.")
    parser.add_argument("--path_pca_batch_size", type=int, default=4096,
                        help="Batch size for transforming patch features with pre_pool pathology PCA.")
    parser.add_argument("--gene_feature_transform",
                        choices=["none", "poly2", "rbf_nystroem", "rbf_sampler"],
                        default="none",
                        help="Optional fold-wise expansion for gene features before concatenating with pathology features.")
    parser.add_argument("--concat_path_dim", type=int, default=1536,
                        help="Pathology dimension in --feature_dir when using gene_feature_transform on concatenated features.")
    parser.add_argument("--concat_gene_dim", type=int, default=30,
                        help="Gene dimension in --feature_dir when using gene_feature_transform on concatenated features.")
    parser.add_argument("--gene_poly_interaction_only", action="store_true",
                        help="For poly2, use only pairwise interactions and original features, excluding squared terms.")
    parser.add_argument("--gene_kernel_n_components", type=int, default=300,
                        help="Number of RBF kernel approximation features for rbf_nystroem/rbf_sampler.")
    parser.add_argument("--gene_kernel_gamma", type=float, default=None,
                        help="RBF gamma for gene kernel features. Defaults to 1 / gene_input_dim after scaling.")
    parser.add_argument("--gene_kernel_only", action="store_true",
                        help="For RBF kernel transforms, use only kernel features instead of [scaled original gene, kernel features].")
    parser.add_argument("--classifier", choices=["logreg", "linear_svm", "rbf_svm", "random_forest", "xgboost"],
                        default="logreg")
    parser.add_argument("--train_level", choices=["patient", "slide"], default="patient",
                        help="Train the classifier on patient-level aggregated features or slide-level features.")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--C_grid", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0, 10.0],
                        help="Candidate C values for validation tuning of Logistic Regression and SVM.")
    parser.add_argument("--gamma", type=parse_gamma, default="scale")
    parser.add_argument("--gamma_grid", nargs="+", default=["scale", "auto", "0.001", "0.01", "0.1"],
                        help="Candidate gamma values for RBF-SVM validation tuning.")
    parser.add_argument("--tune_on_val", action="store_true",
                        help="Tune classifier hyperparameters on validation AUC within each fold.")
    parser.add_argument("--tune_C_on_val", action="store_true",
                        help="Backward-compatible alias for --tune_on_val.")
    parser.add_argument("--refit_train_val_after_tuning", action="store_true",
                        help="After selecting hyperparameters on validation, refit the final model on train+val before test evaluation.")
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--class_weight_balanced", action="store_true", default=True)
    parser.add_argument("--rf_n_estimators", type=int, default=300)
    parser.add_argument("--rf_max_depth", type=parse_optional_int, default=None)
    parser.add_argument("--rf_min_samples_leaf", type=int, default=1)
    parser.add_argument("--rf_n_estimators_grid", type=int, nargs="+", default=[100, 300])
    parser.add_argument("--rf_max_depth_grid", nargs="+", default=["none", "3", "5"])
    parser.add_argument("--rf_min_samples_leaf_grid", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--xgb_n_estimators", type=int, default=100)
    parser.add_argument("--xgb_max_depth", type=int, default=2)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.05)
    parser.add_argument("--xgb_subsample", type=float, default=0.8)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.8)
    parser.add_argument("--xgb_n_estimators_grid", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--xgb_max_depth_grid", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--xgb_learning_rate_grid", type=float, nargs="+", default=[0.03, 0.1])
    parser.add_argument("--xgb_subsample_grid", type=float, nargs="+", default=[0.8, 1.0])
    parser.add_argument("--xgb_colsample_bytree_grid", type=float, nargs="+", default=[0.8, 1.0])
    main(parser.parse_args())
