"""Explicit, hash-verifying feature I/O for a later governed server wrapper."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .contracts import PatchBag, validate_patch_bag


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual.lower() != str(expected).lower():
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")


def load_numeric_matrix(path: str | Path) -> np.ndarray:
    """Load an explicit .pt or .npy matrix without discovering sibling files."""

    source = Path(path)
    if source.suffix.lower() == ".pt":
        try:
            value = torch.load(source, map_location="cpu", weights_only=True)
        except TypeError:  # compatibility with older torch releases
            value = torch.load(source, map_location="cpu")
        if isinstance(value, Mapping):
            for key in ("features", "values", "coordinates", "coords"):
                if key in value:
                    value = value[key]
                    break
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    elif source.suffix.lower() == ".npy":
        value = np.load(source, allow_pickle=False)
    else:
        raise ValueError(f"unsupported feature suffix: {source.suffix}")
    array = np.asarray(value)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{source} must contain a finite two-dimensional numeric matrix")
    return array.astype(np.float32, copy=False)


def load_verified_patch_bag(record: Mapping[str, object], pathway_dim: int = 30) -> PatchBag:
    """Load exactly the three paths and hashes named by a manifest record."""

    required = {
        "case_id", "slide_id", "pathology_path", "pathway_path", "coordinates_path",
        "pathology_sha256", "pathway_sha256", "coordinates_sha256",
    }
    if missing := required - set(record):
        raise ValueError(f"feature record missing fields: {sorted(missing)}")
    for role in ("pathology", "pathway", "coordinates"):
        verify_sha256(str(record[f"{role}_path"]), str(record[f"{role}_sha256"]))
    bag = PatchBag(
        case_id=str(record["case_id"]),
        slide_id=str(record["slide_id"]),
        pathology=load_numeric_matrix(str(record["pathology_path"])),
        pathway=load_numeric_matrix(str(record["pathway_path"])),
        coordinates=load_numeric_matrix(str(record["coordinates_path"])),
        pcr=None if record.get("pcr") is None else int(record["pcr"]),
        mpr=None if record.get("mpr") is None else int(record["mpr"]),
    )
    return validate_patch_bag(bag, pathway_dim=pathway_dim)
