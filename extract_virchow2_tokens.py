"""
extract_virchow2_tokens.py
==========================
从 Virchow2 提取完整 token 序列，保存为 per-patch .pt 缓存文件。

Virchow2 token 结构: [261, 1280] = CLS(1) + Register(4) + Patch(256)
  - lite: CLS + 前64个 patch token（跳过 register）-> [65, 1280]
  - full: 全部 261 tokens -> [261, 1280]

用法:
  PYTHONIOENCODING=utf-8 "C:\Program Files\Python313\python.exe" extract_virchow2_tokens.py --patient HYZ15040 --mode lite
  PYTHONIOENCODING=utf-8 "C:\Program Files\Python313\python.exe" extract_virchow2_tokens.py --patient HYZ15040 JFX0729 LMZ12939 --mode lite
"""

import argparse
import os
import time
from pathlib import Path

import torch
from PIL import Image

from virchow2.virchow2_utils import load_virchow2_backbone
from config_utils import get_patient_paths

LITE_TOKENS = 65   # CLS(1) + 64 patch tokens (skip 4 registers)
FULL_TOKENS = 261  # all tokens


def extract_tokens_for_split(
    backbone,
    transform,
    patches_dir,
    cache_dir,
    device,
    mode="lite",
    rebuild=False,
):
    """提取一个 split 的 Virchow2 token 特征并缓存。"""
    cache_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in patches_dir.iterdir() if p.suffix.lower() == ".png"])
    total = len(image_files)
    if total == 0:
        print(f"  [WARN] 目录为空: {patches_dir}")
        return 0

    num_written = 0
    num_skipped = 0
    t0 = time.time()

    backbone.eval()
    with torch.inference_mode():
        for i, img_path in enumerate(image_files):
            cache_path = cache_dir / f"{img_path.stem}.pt"
            if cache_path.exists() and not rebuild:
                num_skipped += 1
                if (i + 1) % 200 == 0 or (i + 1) == total:
                    elapsed = time.time() - t0
                    print(f"  进度: {i+1}/{total}  (写入 {num_written}, 跳过 {num_skipped})  "
                          f"耗时 {elapsed:.1f}s")
                continue

            image = Image.open(img_path).convert("RGB")
            x = transform(image).unsqueeze(0).to(device, non_blocking=True)

            all_tokens = backbone(x)  # [1, 261, 1280]

            if mode == "lite":
                # CLS(0) + patches(5:69) — 跳过 register tokens(1:5)
                cls_token = all_tokens[:, 0:1, :]       # [1, 1, 1280]
                patch_tokens = all_tokens[:, 5:69, :]   # [1, 64, 1280]
                tokens = torch.cat([cls_token, patch_tokens], dim=1)  # [1, 65, 1280]
            else:
                tokens = all_tokens  # [1, 261, 1280]

            torch.save(tokens.squeeze(0).cpu().float(), cache_path)
            num_written += 1

            if (i + 1) % 200 == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  进度: {i+1}/{total}  (写入 {num_written}, 跳过 {num_skipped})  "
                      f"耗时 {elapsed:.1f}s  速度 {speed:.1f} img/s")

    return num_written


def main():
    parser = argparse.ArgumentParser(description="提取 Virchow2 token 序列特征缓存")
    parser.add_argument("--patient", nargs="+", required=True,
                        choices=["HYZ15040", "JFX0729", "LMZ12939"],
                        help="患者名称，可多个")
    parser.add_argument("--mode", default="lite", choices=["full", "lite"],
                        help="full: 261 tokens; lite: 65 tokens (默认 lite)")
    parser.add_argument("--output_dir", default="virchow2_cache_tokens",
                        help="输出根目录 (默认 virchow2_cache_tokens)")
    parser.add_argument("--rebuild", action="store_true",
                        help="强制重新提取已有缓存")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("加载 Virchow2 backbone ...")
    backbone, transform, feat_dim = load_virchow2_backbone(device=device)
    print(f"特征维度: {feat_dim}")

    mode_desc = f"lite ({LITE_TOKENS} tokens)" if args.mode == "lite" else f"full ({FULL_TOKENS} tokens)"
    print(f"提取模式: {mode_desc}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)

    for patient in args.patient:
        print(f"\n{'='*60}")
        print(f"患者: {patient}")
        print(f"{'='*60}")

        pc = get_patient_paths(patient, backbone='virchow2')
        splits = [
            ('train', Path(pc['train_patches']), Path(pc['token_cache_train'])),
            ('val',   Path(pc['val_patches']),   Path(pc['token_cache_val'])),
        ]

        for split_name, patches_dir, cache_dir in splits:
            print(f"\n  [{split_name}]")
            print(f"  输入: {patches_dir}")
            print(f"  输出: {cache_dir}")

            if not patches_dir.exists():
                print(f"  [ERROR] 目录不存在: {patches_dir}")
                continue

            num_written = extract_tokens_for_split(
                backbone=backbone,
                transform=transform,
                patches_dir=patches_dir,
                cache_dir=cache_dir,
                device=device,
                mode=args.mode,
                rebuild=args.rebuild,
            )

            pt_files = list(cache_dir.glob("*.pt"))
            print(f"  完成! 本次写入 {num_written}, 缓存总计 {len(pt_files)} 个 .pt 文件")

            if pt_files:
                sample = torch.load(pt_files[0], map_location="cpu", weights_only=True)
                sizes = [f.stat().st_size for f in pt_files]
                avg_kb = sum(sizes) / len(sizes) / 1024
                print(f"  样本 shape: {list(sample.shape)}")
                print(f"  平均大小: {avg_kb:.1f} KB")

    print(f"\n{'='*60}")
    print("Virchow2 token 提取完成!")


if __name__ == "__main__":
    main()
