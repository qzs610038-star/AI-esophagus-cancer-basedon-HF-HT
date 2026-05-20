"""
extract_omiclip_features.py
============================
从 OmiCLIP 模型（coca_ViT-L-14）提取 patch-level token 序列特征，
保存为 per-patch .pt 缓存文件，用于下游空间转录组预测任务。

模型信息:
  - 架构: coca_ViT-L-14 (open_clip 格式)
  - 输入: 224×224, CLIP标准归一化
  - 输出: model.visual(x)[1] → [B, 255, 768] token序列

两种模式:
  - full : 保留全部 255 个 token  -> [255, 768]
  - lite : 前 64 个 token          -> [64, 768]

运行环境:
  D:\\conda_envs\\loki_env\\python.exe (Python 3.9, open_clip 2.26.1)

用法示例:
  D:\\conda_envs\\loki_env\\python.exe extract_omiclip_features.py --patient HYZ15040 --mode full
  D:\\conda_envs\\loki_env\\python.exe extract_omiclip_features.py --patient HYZ15040 JFX0729 LMZ12939 --mode full
  D:\\conda_envs\\loki_env\\python.exe extract_omiclip_features.py --patient HYZ15040 --mode lite --batch_size 32
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

try:
    import open_clip
except ImportError:
    print("[ERROR] open_clip 未安装，请在 loki_env 环境中运行此脚本。")
    print("  D:\\conda_envs\\loki_env\\python.exe extract_omiclip_features.py ...")
    sys.exit(1)

from config_utils import get_patient_paths, get_omiclip_checkpoint_path

# ── OmiCLIP 模型配置 ─────────────────────────────────────────
OMICLIP_CHECKPOINT = get_omiclip_checkpoint_path()
OMICLIP_ARCH = "coca_ViT-L-14"
FEAT_DIM = 768
FULL_TOKENS = 255   # model.visual(x)[1] 输出的 token 数
LITE_TOKENS = 64    # lite 模式保留前 64 个 token
INPUT_SIZE = 224

# CLIP 标准归一化参数
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class PatchDataset(Dataset):
    """批量读取 patch 图像的 Dataset，支持跳过已有缓存。"""

    def __init__(self, image_files: list, cache_dir: Path, transform, rebuild: bool = False):
        self.transform = transform
        self.items = []  # (image_path, cache_path) 对

        for img_path in image_files:
            cache_path = cache_dir / f"{img_path.stem}.pt"
            if cache_path.exists() and not rebuild:
                continue  # 跳过已有缓存
            self.items.append((img_path, cache_path))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, cache_path = self.items[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            x = self.transform(image)
        except Exception as e:
            print(f"  [WARN] 读取图像失败: {img_path} -> {e}")
            # 返回零张量作为 placeholder
            x = torch.zeros(3, INPUT_SIZE, INPUT_SIZE)
        return x, str(cache_path), str(img_path)


def build_transform():
    """构建 OmiCLIP 输入预处理 pipeline：Resize + CenterCrop + Normalize。"""
    return transforms.Compose([
        transforms.Resize(INPUT_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def load_omiclip_model(device: torch.device):
    """加载 OmiCLIP 模型 (coca_ViT-L-14 + 预训练权重)。"""
    print(f"加载 OmiCLIP 模型: {OMICLIP_ARCH}")
    print(f"权重路径: {OMICLIP_CHECKPOINT}")

    if not os.path.exists(OMICLIP_CHECKPOINT):
        print(f"[ERROR] 权重文件不存在: {OMICLIP_CHECKPOINT}")
        sys.exit(1)

    # 创建模型结构（不加载预训练权重）
    model = open_clip.create_model(OMICLIP_ARCH, pretrained=None)

    # 加载 OmiCLIP checkpoint
    print("加载 checkpoint ...")
    ckpt = torch.load(OMICLIP_CHECKPOINT, map_location="cpu")

    # 提取 state_dict（适配不同格式）
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # 去除可能的 "module." 前缀（DDP训练产物）
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        cleaned_state_dict[new_key] = v

    # 加载权重（允许部分不匹配）
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        print(f"  [INFO] 缺失的 keys ({len(missing)} 个): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [INFO] 多余的 keys ({len(unexpected)} 个): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model = model.to(device)
    model.eval()
    print(f"OmiCLIP 模型加载完成，已移至 {device}")
    return model


def extract_tokens_for_split(
    model: torch.nn.Module,
    transform,
    patches_dir: Path,
    cache_dir: Path,
    device: torch.device,
    mode: str = "full",
    batch_size: int = 16,
    rebuild: bool = False,
) -> int:
    """提取一个 split（train 或 val）的 OmiCLIP token 特征并缓存。"""
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有 PNG 图像
    image_files = sorted([p for p in patches_dir.iterdir() if p.suffix.lower() == ".png"])
    total = len(image_files)
    if total == 0:
        print(f"  [WARN] 目录为空: {patches_dir}")
        return 0

    # 统计已有缓存
    existing_count = sum(1 for f in image_files if (cache_dir / f"{f.stem}.pt").exists())
    if not rebuild:
        print(f"  总计 {total} 个 patch, 已有缓存 {existing_count} 个, 需处理 {total - existing_count} 个")
    else:
        print(f"  总计 {total} 个 patch (rebuild 模式，全部重新提取)")

    # 构建 Dataset + DataLoader
    dataset = PatchDataset(image_files, cache_dir, transform, rebuild=rebuild)
    if len(dataset) == 0:
        print(f"  ✓ 所有 {existing_count} 个 patch 已有缓存，无需重新提取。")
        return 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    num_written = 0
    num_errors = 0

    # 批量提取
    with torch.inference_mode():
        pbar = tqdm(dataloader, desc="  提取中", unit="batch", leave=True)
        for batch_imgs, batch_cache_paths, batch_img_paths in pbar:
            batch_imgs = batch_imgs.to(device, non_blocking=True)

            try:
                # model.visual(x) 返回 (pooled, tokens)
                # tokens: [B, 255, 768]
                _, tokens = model.visual(batch_imgs)
            except Exception as e:
                print(f"\n  [ERROR] 前向传播失败: {e}")
                num_errors += len(batch_cache_paths)
                continue

            # 根据模式截取 token
            if mode == "lite":
                tokens = tokens[:, :LITE_TOKENS, :]  # [B, 64, 768]
            # full 模式保留全部 [B, 255, 768]

            # 逐样本保存
            for j in range(tokens.shape[0]):
                cache_path = Path(batch_cache_paths[j])
                try:
                    # 确保父目录存在（兑容含中文路径的 Windows）
                    os.makedirs(str(cache_path.parent), exist_ok=True)
                    # 先写入 BytesIO，再用 Python 原生 open() 写文件
                    # （torch._C.PyTorchFileWriter 在 C++ 层无法处理中文路径）
                    buf = io.BytesIO()
                    torch.save(tokens[j].cpu().float(), buf)
                    buf.seek(0)
                    with open(str(cache_path), 'wb') as fp:
                        fp.write(buf.read())
                    num_written += 1
                except Exception as e:
                    print(f"\n  [ERROR] 保存失败: {cache_path} -> {e}")
                    num_errors += 1

            # 更新进度条描述
            pbar.set_postfix(written=num_written, errors=num_errors)

    if num_errors > 0:
        print(f"  [WARN] 共 {num_errors} 个错误")

    return num_written


def main():
    parser = argparse.ArgumentParser(
        description="提取 OmiCLIP (coca_ViT-L-14) token 序列特征缓存",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python extract_omiclip_features.py --patient HYZ15040 --mode full
  python extract_omiclip_features.py --patient HYZ15040 JFX0729 LMZ12939 --mode full --batch_size 32
        """,
    )
    parser.add_argument("--patient", nargs="+", required=True,
                        choices=["HYZ15040", "JFX0729", "LMZ12939"],
                        help="患者名称，可多个")
    parser.add_argument("--mode", default="full", choices=["full", "lite"],
                        help="full: 全部 255 token [255,768]; lite: 前 64 token [64,768] (默认 full)")
    parser.add_argument("--output_dir", default="omiclip_cache",
                        help="输出根目录 (默认 omiclip_cache)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批处理大小 (默认 16)")
    parser.add_argument("--rebuild", action="store_true",
                        help="强制重新提取，忽略已有缓存")
    args = parser.parse_args()

    # ── 设备 ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ── 加载模型 ──
    model = load_omiclip_model(device)

    # ── 构建预处理 ──
    transform = build_transform()

    # ── 信息打印 ──
    token_desc = f"full ({FULL_TOKENS} tokens, dim={FEAT_DIM})" if args.mode == "full" \
        else f"lite ({LITE_TOKENS} tokens, dim={FEAT_DIM})"
    print(f"\n提取模式: {token_desc}")
    print(f"输出目录: {args.output_dir}")
    print(f"批大小: {args.batch_size}")
    print("=" * 60)

    total_written = 0

    for patient in args.patient:
        print(f"\n{'='*60}")
        print(f"患者: {patient}")
        print(f"{'='*60}")

        pc = get_patient_paths(patient, backbone='omiclip')
        splits = [
            ('train', Path(pc['train_patches']), Path(pc['cache_train'])),
            ('val',   Path(pc['val_patches']),   Path(pc['cache_val'])),
        ]

        for split_name, patches_dir, cache_dir in splits:
            print(f"\n  [{split_name}]")
            print(f"  输入: {patches_dir}")
            print(f"  输出: {cache_dir}")

            if not patches_dir.exists():
                print(f"  [ERROR] 输入目录不存在: {patches_dir}")
                continue

            num_written = extract_tokens_for_split(
                model=model,
                transform=transform,
                patches_dir=patches_dir,
                cache_dir=cache_dir,
                device=device,
                mode=args.mode,
                batch_size=args.batch_size,
                rebuild=args.rebuild,
            )

            total_written += num_written

            # ── 统计缓存 ──
            pt_files = list(cache_dir.glob("*.pt"))
            print(f"  完成! 本次写入 {num_written} 个文件, 缓存总计 {len(pt_files)} 个 .pt 文件")

            if pt_files:
                sample = torch.load(pt_files[0], map_location="cpu")
                sizes = [f.stat().st_size for f in pt_files[:100]]  # 取前100个估算
                avg_kb = sum(sizes) / len(sizes) / 1024
                print(f"  样本 shape: {list(sample.shape)}")
                print(f"  平均文件大小: {avg_kb:.1f} KB")

    print(f"\n{'='*60}")
    print(f"全部提取完成! 共写入 {total_written} 个缓存文件。")
    print(f"缓存目录: {project_root / args.output_dir}")


if __name__ == "__main__":
    main()
