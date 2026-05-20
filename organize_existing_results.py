"""整理已有训练结果：为每次训练生成 model_params.txt 并复制到可视化文件夹"""
import pandas as pd
import shutil
import os
from datetime import datetime

from config_utils import get_project_root

# ─── 项目路径 ────────────────────────────────────────────────────────────────
PROJECT_ROOT = get_project_root()
HISTOGENE_DIR = os.path.join(PROJECT_ROOT, "histogene")
RESULTS_VIS_DIR = os.path.join(HISTOGENE_DIR, "checkpoints", "results_vis")

# ─── 训练配置（所有训练共用相同模型参数，来自 train.py argparse 默认值）────────
MODEL_PARAMS = {
    'img_size':    (224,   '输入图像尺寸，与 ImageNet 标准一致'),
    'patch_size':  (16,    'ViT patch 分割粒度，14×14=196 个 token'),
    'model_dim':   (1024,  '嵌入维度，ViT-Large 标准配置'),
    'depth':       (8,     'Transformer 层数，略低于标准 12 层以控制参数量'),
    'heads':       (16,    '多头注意力，每头 64 维子空间'),
    'mlp_dim':     (2048,  'FFN 隐藏层，嵌入维度的 2 倍'),
    'n_pos':       (128,   '坐标嵌入表大小'),
    'n_targets':   (30,    '预测通路数'),
    'dropout':     (0.3,   'Dropout 比率，高于标准 0.1 以适应小数据'),
}

TRAIN_PARAMS = {
    'epochs':        (150,           '最大训练轮数（配合早停）'),
    'batch_size':    (64,            '批大小，兼顾显存和梯度稳定性'),
    'learning_rate': ('1e-4',        'AdamW 初始学习率'),
    'optimizer':     ('AdamW',       'weight_decay=1e-4，解耦正则化'),
    'loss':          ('HuberLoss',   'δ=1.0，对异常值鲁棒'),
    'scheduler':     ('ReduceLROnPlateau', 'factor=0.5, patience=5'),
    'early_stop':    ('patience 15', '基于 val_loss'),
    'AMP':           ('启用',        '混合精度训练'),
}

# 总参数量 ≈ 70.6M（来自模型实例化计数）
N_PARAMS = 70_600_000

# 各数据集样本数
DATASETS = {
    'HYZ15040':        {'train_samples': 2390,  'val_samples': 265},
    'JFX0729':         {'train_samples': 7055,  'val_samples': 783},
    'LMZ12939':        {'train_samples': 6762,  'val_samples': 751},
    'MultiPatient_3ST': {'train_samples': 16207, 'val_samples': 1799},
}

# 可视化时间戳文件夹名
VIS_FOLDERS = {
    'HYZ15040':        'HYZ15040_20260416_213453',
    'JFX0729':         'JFX0729_20260416_224437',
    'LMZ12939':        'LMZ12939_20260417_114425',
    'MultiPatient_3ST': 'MultiPatient_3ST_20260417_140522',
}


def generate_model_params_txt(dataset_name, history_csv, vis_dir):
    """从 training_history CSV 提取指标，生成 model_params.txt"""
    df = pd.read_csv(history_csv)

    # 最佳 val_loss 对应的 epoch
    best_row = df.loc[df['val_loss'].idxmin()]
    best_epoch = int(best_row['epoch'])
    best_val_loss = best_row['val_loss']
    best_val_pcc = best_row['val_pcc']
    best_val_r2 = best_row['val_r2']

    # 最终 epoch 训练指标
    last_row = df.iloc[-1]
    final_train_pcc = last_row['train_pcc']
    total_epochs = int(last_row['epoch'])

    # 过拟合 Gap
    overfit_gap = final_train_pcc - best_val_pcc

    # 样本数
    ds_info = DATASETS.get(dataset_name, {})
    train_samples = ds_info.get('train_samples')
    val_samples = ds_info.get('val_samples')

    # 从文件夹名提取时间戳作为训练完成时间
    folder_name = VIS_FOLDERS.get(dataset_name, '')
    if folder_name:
        # 格式: DatasetName_YYYYMMDD_HHMMSS
        parts = folder_name.rsplit('_', 2)
        try:
            ts_str = f"{parts[-2]}_{parts[-1]}"
            train_time = datetime.strptime(ts_str, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, IndexError):
            train_time = "未知"
    else:
        train_time = "未知"

    lines = []
    lines.append("=" * 50)
    lines.append("HisToGene 模型训练参数")
    lines.append("=" * 50)
    lines.append(f"训练时间: {train_time}")
    lines.append(f"数据集: {dataset_name}")
    if train_samples is not None:
        lines.append(f"训练样本: {train_samples}")
    if val_samples is not None:
        lines.append(f"验证样本: {val_samples}")
    lines.append("")

    lines.append("--- 模型架构参数 ---")
    for name, (val, desc) in MODEL_PARAMS.items():
        lines.append(f"{name:<14} = {str(val):<12} # {desc}")
    if N_PARAMS >= 1e6:
        params_str = f"≈ {N_PARAMS / 1e6:.1f}M"
    elif N_PARAMS >= 1e3:
        params_str = f"≈ {N_PARAMS / 1e3:.1f}K"
    else:
        params_str = str(N_PARAMS)
    lines.append(f"总参数量        {params_str}")
    lines.append("")

    lines.append("--- 训练超参数 ---")
    for name, (val, desc) in TRAIN_PARAMS.items():
        lines.append(f"{name:<14} = {str(val):<12} # {desc}")
    lines.append("")

    lines.append("--- 训练结果 ---")
    lines.append(f"总 Epoch: {total_epochs}")
    lines.append(f"最佳 Epoch: {best_epoch}")
    lines.append(f"Best Val PCC: {best_val_pcc:.4f}")
    lines.append(f"Best Val R²: {best_val_r2:.4f}")
    lines.append(f"Best Val Loss: {best_val_loss:.4f}")
    lines.append(f"最终 Train PCC: {final_train_pcc:.4f}")
    lines.append(f"过拟合 Gap (PCC): {overfit_gap:.4f}")

    output_path = os.path.join(vis_dir, "model_params.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [OK] model_params.txt -> {output_path}")


def main():
    print("=" * 60)
    print("整理已有训练结果")
    print("=" * 60)

    for dataset_name, folder_name in VIS_FOLDERS.items():
        print(f"\n--- {dataset_name} ---")

        # 可视化文件夹路径
        vis_dir = os.path.join(RESULTS_VIS_DIR, folder_name)
        if not os.path.isdir(vis_dir):
            print(f"  [SKIP] 可视化文件夹不存在: {vis_dir}")
            continue

        # training_history CSV 路径
        history_csv = os.path.join(HISTOGENE_DIR, f"training_history_{dataset_name}.csv")
        if not os.path.isfile(history_csv):
            print(f"  [SKIP] 训练历史文件不存在: {history_csv}")
            continue

        # 1. 生成 model_params.txt
        try:
            generate_model_params_txt(dataset_name, history_csv, vis_dir)
        except Exception as e:
            print(f"  [ERROR] 生成 model_params.txt 失败: {e}")

        # 2. 复制 training_history CSV
        dst_csv = os.path.join(vis_dir, os.path.basename(history_csv))
        if os.path.isfile(dst_csv):
            print(f"  [SKIP] training_history CSV 已存在: {dst_csv}")
        else:
            try:
                shutil.copy2(history_csv, dst_csv)
                print(f"  [OK] training_history CSV -> {dst_csv}")
            except Exception as e:
                print(f"  [ERROR] 复制训练历史 CSV 失败: {e}")

    # ── 汇总 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("整理完成，各文件夹内容：")
    print("=" * 60)
    for dataset_name, folder_name in VIS_FOLDERS.items():
        vis_dir = os.path.join(RESULTS_VIS_DIR, folder_name)
        if os.path.isdir(vis_dir):
            files = sorted(os.listdir(vis_dir))
            print(f"\n{folder_name}/")
            for f in files:
                print(f"  - {f}")


if __name__ == "__main__":
    main()
