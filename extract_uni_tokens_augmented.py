"""
extract_uni_tokens_augmented.py
================================
基于 extract_uni_tokens.py 扩展，增加 H&E 颜色空间增强。
对每张原始 patch 图像生成 N+1 份 token 特征：
  - 1 份原始特征 (_aug0.pt)
  - N 份 HED 颜色空间随机扰动后的增强特征 (_aug1.pt ~ _augN.pt)

增强方式：在 HED 颜色空间对 H/E/D 三通道分别加全局高斯偏移，
保持组织结构不变，仅改变染色强度分布。

用法示例:
  python extract_uni_tokens_augmented.py --n_augments 3
  python extract_uni_tokens_augmented.py --patients HYZ15040 JFX0729 --n_augments 5 --h_sigma 0.08
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2hed, hed2rgb
from tqdm import tqdm

# ── 复用项目已有的 backbone 加载 ──────────────────────────────
from uni2h.uni2h_utils import load_uni2h_backbone
from config_utils import get_patient_paths

LITE_TOKENS = 65   # CLS(1) + 前64个patch token
FULL_TOKENS = 265  # CLS(1) + reg(8) + patch(256)


# ══════════════════════════════════════════════════════════════
# H&E 颜色增强
# ══════════════════════════════════════════════════════════════

def augment_he_image(image_np, h_sigma=0.05, e_sigma=0.05, d_sigma=0.02):
    """
    对H&E图像进行HED颜色空间随机扰动。

    Args:
        image_np: numpy array, shape [H, W, 3], dtype float64, range [0, 1]
        h_sigma: Hematoxylin通道扰动强度
        e_sigma: Eosin通道扰动强度
        d_sigma: DAB通道扰动强度

    Returns:
        augmented: numpy array, same shape, range [0, 1]
    """
    hed = rgb2hed(image_np)  # [H, W, 3] in HED space

    # 对每个通道添加全局偏移（而非逐像素噪声，保持结构）
    hed[:, :, 0] += np.random.normal(0, h_sigma)  # Hematoxylin
    hed[:, :, 1] += np.random.normal(0, e_sigma)  # Eosin
    hed[:, :, 2] += np.random.normal(0, d_sigma)  # DAB

    # 转回RGB并裁剪
    augmented = hed2rgb(hed)
    augmented = np.clip(augmented, 0, 1)
    return augmented


# ══════════════════════════════════════════════════════════════
# Token 提取核心逻辑
# ══════════════════════════════════════════════════════════════

def extract_single_image_tokens(backbone, transform, image_pil, device, mode="lite"):
    """
    对单张 PIL 图像提取 UNI2-h token 特征。

    Args:
        backbone: UNI2-h 模型
        transform: 预处理 transform（resize 224x224 + normalize）
        image_pil: PIL.Image (RGB)
        device: torch device
        mode: "lite" 或 "full"

    Returns:
        tokens: Tensor [num_tokens, 1536]
    """
    x = transform(image_pil).unsqueeze(0).to(device, non_blocking=True)

    with torch.amp.autocast('cuda'):
        all_tokens = backbone.forward_features(x)  # [1, 265, 1536]

    if mode == "lite":
        tokens = all_tokens[:, :LITE_TOKENS, :]  # [1, 65, 1536]
    else:
        tokens = all_tokens  # [1, 265, 1536]

    return tokens.squeeze(0).cpu().float()


def extract_augmented_tokens_for_split(
    backbone: torch.nn.Module,
    transform,
    patches_dir: Path,
    cache_dir: Path,
    device: torch.device,
    mode: str = "lite",
    n_augments: int = 3,
    h_sigma: float = 0.05,
    e_sigma: float = 0.05,
    d_sigma: float = 0.02,
    skip_existing: bool = True,
) -> dict:
    """
    对一个 split（train 或 val）提取原始 + 增强 token 特征并缓存。

    Returns:
        stats: dict with keys 'total', 'written', 'skipped'
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in patches_dir.iterdir() if p.suffix.lower() == ".png"])
    total = len(image_files)
    if total == 0:
        print(f"  [WARN] 目录为空: {patches_dir}")
        return {"total": 0, "written": 0, "skipped": 0}

    num_written = 0
    num_skipped = 0

    backbone.eval()
    with torch.inference_mode():
        for img_path in tqdm(image_files, desc="  提取中", unit="img"):
            stem = img_path.stem

            # ── 处理原始图像 (aug0) ──
            cache_path_orig = cache_dir / f"{stem}_aug0.pt"
            need_original = not (cache_path_orig.exists() and skip_existing)

            # ── 检查增强版本 ──
            aug_paths = []
            need_augments = []
            for ai in range(1, n_augments + 1):
                ap = cache_dir / f"{stem}_aug{ai}.pt"
                aug_paths.append(ap)
                need_augments.append(not (ap.exists() and skip_existing))

            # 如果全部已存在则跳过
            if not need_original and not any(need_augments):
                num_skipped += 1
                continue

            # 加载原始图像
            image_pil = Image.open(img_path).convert("RGB")

            # ── 提取原始特征 ──
            if need_original:
                tokens_orig = extract_single_image_tokens(
                    backbone, transform, image_pil, device, mode
                )
                torch.save(tokens_orig, cache_path_orig)
                num_written += 1
            else:
                num_skipped += 1

            # ── 提取增强特征 ──
            image_np = np.array(image_pil).astype(np.float64) / 255.0  # [H, W, 3], [0, 1]

            for ai, (ap, need) in enumerate(zip(aug_paths, need_augments), start=1):
                if not need:
                    num_skipped += 1
                    continue

                # HED颜色增强
                aug_np = augment_he_image(image_np, h_sigma, e_sigma, d_sigma)

                # 转回 PIL（uint8）
                aug_uint8 = (aug_np * 255).astype(np.uint8)
                aug_pil = Image.fromarray(aug_uint8, mode="RGB")

                # 提取 token
                tokens_aug = extract_single_image_tokens(
                    backbone, transform, aug_pil, device, mode
                )
                torch.save(tokens_aug, ap)
                num_written += 1

    return {"total": total, "written": num_written, "skipped": num_skipped}


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="提取 UNI2-h token 特征（含 H&E 颜色增强版本）"
    )
    parser.add_argument("--patients", nargs="+",
                        default=["HYZ15040", "JFX0729", "LMZ12939"],
                        choices=["HYZ15040", "JFX0729", "LMZ12939"],
                        help="患者列表 (默认全部三位)")
    parser.add_argument("--input_dir", default="data_new_3ST/patch_noov_spilt",
                        help="原始patch图像根目录 (默认 data_new_3ST/patch_noov_spilt)")
    parser.add_argument("--output_dir", default="uni2h_cache_tokens_aug",
                        help="增强特征输出根目录 (默认 uni2h_cache_tokens_aug)")
    parser.add_argument("--mode", default="lite", choices=["full", "lite"],
                        help="full: 全部265 token; lite: CLS+前64 patch token (默认 lite)")
    parser.add_argument("--n_augments", type=int, default=3,
                        help="增强份数（不含原始），默认 3")
    parser.add_argument("--h_sigma", type=float, default=0.05,
                        help="Hematoxylin 通道扰动标准差 (默认 0.05)")
    parser.add_argument("--e_sigma", type=float, default=0.05,
                        help="Eosin 通道扰动标准差 (默认 0.05)")
    parser.add_argument("--d_sigma", type=float, default=0.02,
                        help="DAB 通道扰动标准差 (默认 0.02)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="推理batch大小 (默认 1，8GB显存安全值)")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="跳过已存在的.pt文件 (默认 True)")
    parser.add_argument("--no_skip_existing", action="store_true",
                        help="强制重新提取所有文件")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="推理设备 (默认 cuda)")
    args = parser.parse_args()

    # 处理 skip_existing 逻辑
    skip_existing = not args.no_skip_existing

    # ── 设备 ──
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if args.device == "cuda":
            print("[WARN] CUDA不可用，回退到CPU")

    print(f"{'='*60}")
    print(f"H&E 增强版 UNI2-h Token 特征提取")
    print(f"{'='*60}")
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"患者: {args.patients}")
    print(f"模式: {args.mode} ({LITE_TOKENS if args.mode == 'lite' else FULL_TOKENS} tokens)")
    print(f"增强份数: {args.n_augments} (总计 {args.n_augments + 1} 份/图像)")
    print(f"HED扰动: H={args.h_sigma}, E={args.e_sigma}, D={args.d_sigma}")
    print(f"输出目录: {args.output_dir}")
    print(f"跳过已有: {skip_existing}")
    print(f"{'='*60}")

    # ── 加载 backbone ──
    print("\n加载 UNI2-h backbone ...")
    backbone, transform, feat_dim = load_uni2h_backbone(device=device)
    print(f"特征维度: {feat_dim}")
    print(f"模型加载完成!\n")

    total_written_all = 0
    total_files_all = 0
    t_global = time.time()

    for patient in args.patients:
        print(f"\n{'='*60}")
        print(f"患者: {patient}")
        print(f"{'='*60}")

        pc = get_patient_paths(patient, backbone='uni_tokens_aug')
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

            stats = extract_augmented_tokens_for_split(
                backbone=backbone,
                transform=transform,
                patches_dir=patches_dir,
                cache_dir=cache_dir,
                device=device,
                mode=args.mode,
                n_augments=args.n_augments,
                h_sigma=args.h_sigma,
                e_sigma=args.e_sigma,
                d_sigma=args.d_sigma,
                skip_existing=skip_existing,
            )

            total_written_all += stats["written"]
            total_files_all += stats["total"]

            # ── 统计 ──
            pt_files = list(cache_dir.glob("*.pt"))
            print(f"  完成! 本次写入 {stats['written']} 个文件, "
                  f"跳过 {stats['skipped']} 个, "
                  f"缓存总计 {len(pt_files)} 个 .pt 文件")

            if pt_files:
                sample = torch.load(pt_files[0], map_location="cpu")
                sizes = [f.stat().st_size for f in pt_files[:100]]  # 取前100个估算
                avg_kb = sum(sizes) / len(sizes) / 1024
                print(f"  样本 shape: {list(sample.shape)}")
                print(f"  平均文件大小: {avg_kb:.1f} KB")

    elapsed_total = time.time() - t_global
    print(f"\n{'='*60}")
    print(f"全部提取完成!")
    print(f"总计: {total_files_all} 张原图 × {args.n_augments + 1} 份 = "
          f"{total_files_all * (args.n_augments + 1)} 份特征")
    print(f"本次写入: {total_written_all} 个 .pt 文件")
    print(f"总耗时: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
