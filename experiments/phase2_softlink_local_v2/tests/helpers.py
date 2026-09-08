from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config
from data import make_point_table

PACKAGE = Path(__file__).resolve().parents[1]


def package_config():
    return load_config(PACKAGE / "config.json", package_dir=PACKAGE)


def tiny_model_config(hidden=17, relation=11, input_dim=8, output_dim=5, dropout=0.0):
    return {
        "data": {"input_dim": input_dim, "output_dim": output_dim},
        "model": {
            "hidden_dim": hidden,
            "relation_dim": relation,
            "dropout": dropout,
            "activation": "gelu_exact",
            "shared_bias": True,
            "readout_bias": True,
            "relation_bias": False,
            "spatial_bias": False,
            "spatial_initialization": "zeros",
            "normalization_epsilon": 1e-12,
        },
        "graph": {
            "hops": 1,
            "radius_in_native_steps": 1.5,
            "max_neighbors": 8,
            "distance_sigma": 1.0,
            "use_image_similarity": True,
            "image_temperature": 0.2,
            "self_raw_weight": 1.0,
            "directed": True,
        },
        "relation": {
            "max_near": 8,
            "max_far": 32,
            "tau_y": 1.0,
            "tau_z": 0.1,
            "far_patient_scope": "all",
        },
        "precheck": {
            "relation_seed": 20260908,
            "relation_complete_shuffles": 3,
            "graph_splits": ["train", "internal_val"],
            "automatic_parameter_changes": False,
            "low_valid_rate_warn": 0.2,
        },
        "training": {"batch_size": 4, "keep_last_batch": True},
    }


def geometry_frame(rows):
    return pd.DataFrame(rows)


def verified_points(
    *,
    patients,
    slides,
    spots,
    splits,
    x,
    y,
    features,
    labels=None,
    s_by_slide=None,
):
    table = make_point_table(
        patient_ids=patients,
        spot_ids=spots,
        splits=splits,
        x=x,
        y=y,
        slide_ids=slides,
        features=np.asarray(features, dtype=np.float32),
        labels_z=None if labels is None else np.asarray(labels, dtype=np.float32),
        slide_status="verified",
        sort_identities=True,
    )
    unique_slides = sorted(set(slides))
    geo_rows = []
    for slide in unique_slides:
        step = 1.0 if s_by_slide is None else float(s_by_slide[slide])
        geo_rows.append(
            {
                "patient_id": "",
                "slide_id": slide,
                "s": step,
                "coordinate_unit": "test",
                "patch_coverage_size": step,
                "source": "test",
                "status": "verified",
            }
        )
    return table, geometry_frame(geo_rows)
