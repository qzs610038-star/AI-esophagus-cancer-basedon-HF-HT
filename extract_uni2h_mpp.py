"""
extract_uni2h_mpp.py — MPP 五划分 UNI2-h 特征提取（CLS token only）

从 MPP 数据根目录的 patch_images/*.png 提取 UNI2-h CLS token 特征
并缓存为 mpp_uni2h_cache/{mpp_id}/{patient}/*.pt，每个 .pt shape [1536]。

MPP 专用脚本：路径体系独立于 config_utils（该模块为旧 Phase2 三患者设计）。
迁移到新机器时需改 --mpp_root 默认值或显式传参。

用法:
    python extract_uni2h_mpp.py --mpp_id 3 --patient HYZ15040 JFX LMZ12939
    python extract_uni2h_mpp.py --mpp_id 2 --patient XZY
"""

import argparse
import os
import time
from pathlib import Path

import torch
from PIL import Image

from uni2h.uni2h_utils import load_uni2h_backbone

# ── 显式离线 + HF 缓存路径（服务器共享缓存） ──
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "D:\\AIPatho\\shared\\.cache\\huggingface")


def extract_cls_for_split(
    backbone: torch.nn.Module,
    transform,
    patches_dir: Path,
    cache_dir: Path,
    device: torch.device,
    batch_size: int = 16,
    rebuild: bool = False,
) -> int:
    """提取一个 patient split 的 CLS token 特征并缓存。"""
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
        for i in range(0, total, batch_size):
            batch_files = image_files[i : i + batch_size]
            batch_paths = []

            for img_path in batch_files:
                cache_path = cache_dir / f"{img_path.stem}.pt"
                if cache_path.exists() and not rebuild:
                    num_skipped += 1
                    continue
                batch_paths.append(img_path)

            if not batch_paths:
                continue

            batch_images = []
            for img_path in batch_paths:
                image = Image.open(img_path).convert("RGB")
                x = transform(image).unsqueeze(0)
                batch_images.append(x)

            x = torch.cat(batch_images, dim=0).to(device, non_blocking=True)
            all_tokens = backbone.forward_features(x)  # [B, 265, 1536]
            cls_tokens = all_tokens[:, 0, :]           # [B, 1536]

            for j, img_path in enumerate(batch_paths):
                cache_path = cache_dir / f"{img_path.stem}.pt"
                torch.save(cls_tokens[j].cpu().float(), cache_path)
                num_written += 1

            if (i + batch_size) >= total or (i // batch_size) % 5 == 0:
                elapsed = time.time() - t0
                speed = (i + len(batch_paths)) / elapsed if elapsed > 0 else 0
                print(f"  进度: {min(i+batch_size, total)}/{total}  "
                      f"(已写入 {num_written}, 跳过 {num_skipped})  "
                      f"耗时 {elapsed:.1f}s  速度 {speed:.1f} img/s")

    return num_written


def main():
    parser = argparse.ArgumentParser(description="MPP UNI2-h CLS token 特征提取")
    parser.add_argument("--mpp_root", default=r"D:\AIPatho\Patch\visiumhd_patch",
                        help="MPP 数据根目录（默认服务器路径；迁移时改此值或传参）")
    parser.add_argument("--mpp_id", type=int, required=True,
                        help="MPP 划分编号 (1-5)")
    parser.add_argument("--patient", nargs="+", required=True,
                        help="患者名称列表，如 HYZ15040 JFX LMZ12939")
    parser.add_argument("--output_root", default="mpp_uni2h_cache",
                        help="输出根目录 (默认 mpp_uni2h_cache)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批处理大小 (默认 16)")
    parser.add_argument("--rebuild", action="store_true",
                        help="强制重新提取已有缓存")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("加载 UNI2-h backbone ...")
    backbone, transform, feat_dim = load_uni2h_backbone(device=device)
    print(f"特征维度: {feat_dim}  (CLS-only: [{feat_dim}])")
    print(f"MPP root: {args.mpp_root}")
    print(f"MPP id:   {args.mpp_id}")
    print(f"输出:     {args.output_root}/{args.mpp_id}/{{patient}}/*.pt")
    print("=" * 60)

    for patient in args.patient:
        print(f"\n{'='*60}")
        print(f"患者: {patient}")
        print(f"{'='*60}")

        patches_dir = Path(args.mpp_root) / str(args.mpp_id) / patient / "patch_images"
        cache_dir = Path(args.output_root) / str(args.mpp_id) / patient

        if not patches_dir.exists():
            print(f"  [ERROR] patch_images 目录不存在: {patches_dir}")
            continue

        num_written = extract_cls_for_split(
            backbone=backbone,
            transform=transform,
            patches_dir=patches_dir,
            cache_dir=cache_dir,
            device=device,
            batch_size=args.batch_size,
            rebuild=args.rebuild,
        )

        pt_files = list(cache_dir.glob("*.pt"))
        print(f"  完成! 本次写入 {num_written} 个文件, 缓存总计 {len(pt_files)} 个 .pt 文件")
        if pt_files:
            sample = torch.load(pt_files[0], map_location="cpu")
            avg_kb = sum(f.stat().st_size for f in pt_files) / len(pt_files) / 1024
            print(f"  样本 shape: {list(sample.shape)}")
            print(f"  平均文件大小: {avg_kb:.1f} KB")

    print(f"\n{'='*60}")
    print("全部提取完成!")


if __name__ == "__main__":
    main()
