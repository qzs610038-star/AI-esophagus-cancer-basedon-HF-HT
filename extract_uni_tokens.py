"""
extract_uni_tokens.py
=====================
从 UNI2-h 模型提取完整 token 序列（而非仅 CLS 全局池化），
保存为 per-patch .pt 缓存文件。

两种模式:
  - full : 保留全部 265 个 token  -> [265, 1536]
  - lite : CLS + 前 64 个 patch token -> [65, 1536]

用法示例:
  python extract_uni_tokens.py --patient HYZ15040 --mode lite
  python extract_uni_tokens.py --patient HYZ15040 JFX0729 LMZ12939 --mode full
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# ── 复用项目已有的 backbone 加载 ──────────────────────────────
from uni2h.uni2h_utils import load_uni2h_backbone
from config_utils import get_patient_paths

LITE_TOKENS = 65   # CLS(1) + 前64个patch token
FULL_TOKENS = 265  # CLS(1) + reg(8) + patch(256)


def extract_tokens_for_split(
    backbone: torch.nn.Module,
    transform,
    patches_dir: Path,
    cache_dir: Path,
    device: torch.device,
    mode: str = "lite",
    batch_size: int = 1,
    rebuild: bool = False,
) -> int:
    """提取一个 split（train 或 val）的 token 特征并缓存。"""
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
                if (i + 1) % 100 == 0 or (i + 1) == total:
                    elapsed = time.time() - t0
                    print(f"  进度: {i+1}/{total}  (已写入 {num_written}, 跳过 {num_skipped})  耗时 {elapsed:.1f}s")
                continue

            image = Image.open(img_path).convert("RGB")
            x = transform(image).unsqueeze(0).to(device, non_blocking=True)

            # forward_features 返回完整 token 序列 [1, N, 1536]
            all_tokens = backbone.forward_features(x)  # [1, 265, 1536]

            if mode == "lite":
                tokens = all_tokens[:, :LITE_TOKENS, :]  # [1, 65, 1536]
            else:
                tokens = all_tokens  # [1, 265, 1536]

            torch.save(tokens.squeeze(0).cpu().float(), cache_path)
            num_written += 1

            if (i + 1) % 100 == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  进度: {i+1}/{total}  (已写入 {num_written}, 跳过 {num_skipped})  "
                      f"耗时 {elapsed:.1f}s  速度 {speed:.1f} img/s")

    return num_written


def main():
    parser = argparse.ArgumentParser(description="提取 UNI2-h token 序列特征缓存")
    parser.add_argument("--patient", nargs="+", required=True,
                        choices=["HYZ15040", "JFX0729", "LMZ12939"],
                        help="患者名称，可多个")
    parser.add_argument("--mode", default="lite", choices=["full", "lite"],
                        help="full: 全部265 token; lite: CLS+前64 patch token (默认 lite)")
    parser.add_argument("--output_dir", default="uni2h_cache_tokens",
                        help="输出根目录 (默认 uni2h_cache_tokens)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="处理批大小 (默认 1)")
    parser.add_argument("--rebuild", action="store_true",
                        help="强制重新提取已有缓存")
    args = parser.parse_args()

    # ── 设备 ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── 加载 backbone ──
    print("加载 UNI2-h backbone ...")
    backbone, transform, feat_dim = load_uni2h_backbone(device=device)
    print(f"特征维度: {feat_dim}")

    token_desc = f"lite ({LITE_TOKENS} tokens)" if args.mode == "lite" else f"full ({FULL_TOKENS} tokens)"
    print(f"提取模式: {token_desc}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)

    for patient in args.patient:
        print(f"\n{'='*60}")
        print(f"患者: {patient}")
        print(f"{'='*60}")

        pc = get_patient_paths(patient, backbone='uni_tokens')
        splits = [
            ('train', Path(pc['train_patches']), Path(pc['token_cache_train'])),
            ('val',   Path(pc['val_patches']),   Path(pc['token_cache_val'])),
        ]

        for split_name, patches_dir, cache_dir in splits:
            print(f"\n  [{split_name}]")
            print(f"  输入: {patches_dir}")
            print(f"  输出: {cache_dir}")

            if not patches_dir.exists():
                print(f"  [ERROR] 输入目录不存在: {patches_dir}")
                continue

            num_written = extract_tokens_for_split(
                backbone=backbone,
                transform=transform,
                patches_dir=patches_dir,
                cache_dir=cache_dir,
                device=device,
                mode=args.mode,
                batch_size=args.batch_size,
                rebuild=args.rebuild,
            )

            # ── 统计 ──
            pt_files = list(cache_dir.glob("*.pt"))
            print(f"  完成! 本次写入 {num_written} 个文件, 缓存总计 {len(pt_files)} 个 .pt 文件")

            if pt_files:
                sample = torch.load(pt_files[0], map_location="cpu")
                sizes = [f.stat().st_size for f in pt_files]
                avg_kb = sum(sizes) / len(sizes) / 1024
                print(f"  样本 shape: {list(sample.shape)}")
                print(f"  平均文件大小: {avg_kb:.1f} KB")

    print(f"\n{'='*60}")
    print("全部提取完成!")


if __name__ == "__main__":
    main()
