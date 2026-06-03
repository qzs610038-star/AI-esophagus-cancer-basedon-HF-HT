"""Extract UNI2-h features for a single patient and cache to disk."""
import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import torch
from config_utils import load_config, get_device

from uni2h_des import load_uni2h_backbone, extract_and_cache_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patient", required=True, choices=["HYZ15040", "JFX0729", "LMZ12939"])
    p.add_argument("--cache_root", required=True)
    p.add_argument("--hf_local_only", action="store_true", default=True)
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()

    # Determine patches directories from config
    config = load_config()
    device = get_device(config)
    print(f"Device: {device}")

    # Patches follow standard structure: data_new_3ST/patch_noov_spilt/{patient}_noov_split/
    patches_base = _PROJECT_ROOT / "data_new_3ST" / "patch_noov_spilt" / f"{args.patient}_noov_split"
    train_patches = str(patches_base / "train_patches")
    val_patches = str(patches_base / "val_patches")

    print(f"Train patches: {train_patches}")
    print(f"Val patches: {val_patches}")

    # Load backbone
    backbone, transform, feature_dim = load_uni2h_backbone(
        device=device, local_only=args.hf_local_only
    )
    print(f"UNI2-h loaded. Feature dim={feature_dim}")

    # Extract features
    cache_root = Path(args.cache_root)
    train_cache = cache_root / "train"
    val_cache = cache_root / "val"
    train_cache.mkdir(parents=True, exist_ok=True)
    val_cache.mkdir(parents=True, exist_ok=True)

    n_train = extract_and_cache_features(
        backbone, transform, train_patches, str(train_cache), device, rebuild=args.rebuild
    )
    n_val = extract_and_cache_features(
        backbone, transform, val_patches, str(val_cache), device, rebuild=args.rebuild
    )
    print(f"Done: {n_train} train + {n_val} val features cached for {args.patient}")


if __name__ == "__main__":
    main()
