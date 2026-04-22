import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import os
import json
import matplotlib.pyplot as plt

# --- 1. 配置路径 (同上) ---
# 预提取的特征和标签文件路径 (请修改为你保存的路径)
train_features_path = r"D:\AIPatho\hhy\ESCC\openmidnight\data\train_features.npy"
train_labels_path = r"D:\AIPatho\hhy\ESCC\openmidnight\data\train_labels.npy"
val_features_path = r"D:\AIPatho\hhy\ESCC\openmidnight\data\val_features.npy"
val_labels_path = r"D:\AIPatho\hhy\ESCC\openmidnight\data\val_labels.npy"

for path in [train_features_path, train_labels_path, val_features_path, val_labels_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")


# --- 2. 自定义 Dataset (同上) ---
class PrecomputedFeaturesDataset(Dataset):
    def __init__(self, features_path, labels_path):
        self.features = np.load(features_path).astype(np.float32)
        self.labels = np.load(labels_path).astype(np.float32)
        print(f"Dataset loaded from {features_path} and {labels_path}. Shape: features {self.features.shape}, labels {self.labels.shape}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# --- 3. MLP 回归头定义 (可考虑稍微增加复杂度) ---
class MLPRegressor(nn.Module):
    def __init__(self, input_dim=1536, hidden_dims=[256], output_dim=8, dropout_rate=0.1):
        super(MLPRegressor, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate) # Add dropout for potential regularization
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# --- 4. 计算详细指标的函数 (同上，已包含 float 转换) ---
def calculate_detailed_metrics(y_true, y_pred, num_outputs=8):
    metrics = {}

    overall_mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    overall_mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    overall_r2 = r2_score(y_true.flatten(), y_pred.flatten())
    overall_pcc, _ = pearsonr(y_true.flatten(), y_pred.flatten())

    metrics['overall'] = {
        'mse': float(overall_mse),
        'mae': float(overall_mae),
        'r2': float(overall_r2),
        'pcc': float(overall_pcc)
    }

    per_output_metrics = {}
    for i in range(num_outputs):
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]

        mse_i = mean_squared_error(true_i, pred_i)
        mae_i = mean_absolute_error(true_i, pred_i)
        r2_i = r2_score(true_i, pred_i)
        pcc_i, _ = pearsonr(true_i, pred_i)

        per_output_metrics[f'gene_set_{i}'] = {
            'mse': float(mse_i),
            'mae': float(mae_i),
            'r2': float(r2_i),
            'pcc': float(pcc_i)
        }

    metrics['per_output'] = per_output_metrics

    return metrics


# --- 5. 训练和验证函数 (同上) ---
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    num_batches = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        with torch.no_grad():
            mae = torch.mean(torch.abs(outputs - targets)).item()
        running_mae += mae
        num_batches += 1

    epoch_loss = running_loss / num_batches
    epoch_mae = running_mae / num_batches
    return epoch_loss, epoch_mae


def validate_epoch(model, dataloader, device):
    model.eval()
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    all_targets_np = np.concatenate(all_targets, axis=0)
    all_outputs_np = np.concatenate(all_outputs, axis=0)

    metrics = calculate_detailed_metrics(all_targets_np, all_outputs_np, num_outputs=8)

    return metrics, all_targets_np, all_outputs_np


def main():
    # --- 6. 加载数据集 (同上) ---
    train_dataset = PrecomputedFeaturesDataset(train_features_path, train_labels_path)
    val_dataset = PrecomputedFeaturesDataset(val_features_path, val_labels_path)

    # --- 7. 创建 DataLoader (同上) ---
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- 8. 检查设备 (同上) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 9. 创建模型、损失函数、优化器 (调整) ---
    # 尝试稍微复杂一点的 MLP: 1536 -> 512 -> 256 -> 8
    # 也可以先用回 1536 -> 256 -> 8 看看效果
    model = MLPRegressor(input_dim=1536, hidden_dims=[512, 256], output_dim=8, dropout_rate=0.1).to(device)
    criterion = nn.MSELoss()
    # 使用 AdamW 优化器，它对权重衰减的处理可能更好
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) # Lower initial LR and add WD

    # --- 10. 添加学习率调度器 ---
    # ReduceLROnPlateau: 当验证损失停止改善时，学习率衰减
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # --- 11. 训练循环 (调整) ---
    num_epochs = 200 # 可能需要更多 epoch
    best_val_overall_loss = float('inf')
    best_model_weights = None

    train_losses = []
    val_overall_metrics_history = []
    val_per_output_metrics_history = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        print(f'Train Loss (MSE): {train_loss:.6f}, Train MAE: {train_mae:.6f}')

        val_metrics, _, _ = validate_epoch(model, val_loader, device)
        val_overall_metrics_history.append(val_metrics['overall'])
        val_per_output_metrics_history.append(val_metrics['per_output'])

        print(f'Val Overall - MSE: {val_metrics["overall"]["mse"]:.6f}, '
              f'MAE: {val_metrics["overall"]["mae"]:.6f}, '
              f'R²: {val_metrics["overall"]["r2"]:.6f}, '
              f'PCC: {val_metrics["overall"]["pcc"]:.6f}')

        # 保存最佳模型 (基于整体验证 MSE)
        if val_metrics['overall']['mse'] < best_val_overall_loss:
            best_val_overall_loss = val_metrics['overall']['mse']
            best_model_weights = model.state_dict().copy()
            print(f"*** New Best Val Overall MSE: {best_val_overall_loss:.6f} ***")

        # 更新学习率 (基于验证 MSE)
        scheduler.step(val_metrics['overall']['mse']) # Pass the metric to monitor

    # --- 12. 加载并保存最佳模型 (同上) ---
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        model_save_path = os.path.join(os.path.dirname(__file__), 'best_mlp_regressor_with_scheduler.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f"\nTraining completed! Best model saved to: {model_save_path}")
        print(f"Best validation overall MSE: {best_val_overall_loss:.6f}")

    # --- 13. 最终验证与指标报告 (同上) ---
    print("\n--- Final Validation on Best Model ---")
    final_val_metrics, final_targets, final_predictions = validate_epoch(model, val_loader, device)

    print("\n--- Final Overall Metrics ---")
    print(f"MSE: {final_val_metrics['overall']['mse']:.6f}")
    print(f"MAE: {final_val_metrics['overall']['mae']:.6f}")
    print(f"R²: {final_val_metrics['overall']['r2']:.6f}")
    print(f"PCC: {final_val_metrics['overall']['pcc']:.6f}")

    print("\n--- Final Per-Output Metrics (Gene Sets) ---")
    for i in range(8):
        gs_key = f'gene_set_{i}'
        print(f"Gene Set {i}: MSE={final_val_metrics['per_output'][gs_key]['mse']:.6f}, "
              f"MAE={final_val_metrics['per_output'][gs_key]['mae']:.6f}, "
              f"R²={final_val_metrics['per_output'][gs_key]['r2']:.6f}, "
              f"PCC={final_val_metrics['per_output'][gs_key]['pcc']:.6f}")

    # --- 14. 保存指标历史 (同上) ---
    metrics_log_path = os.path.join(os.path.dirname(__file__), 'training_metrics_log_with_scheduler.json')
    metrics_data = {
        'train_losses': train_losses,
        'val_overall_metrics': val_overall_metrics_history,
        'val_per_output_metrics': val_per_output_metrics_history,
        'final_best_val_overall_mse': best_val_overall_loss,
        'final_val_metrics': final_val_metrics
    }
    with open(metrics_log_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print(f"\nMetrics history saved to: {metrics_log_path}")

    # --- 15. 绘制指标趋势图 (同上) ---
    plot_metrics(train_losses, val_overall_metrics_history, val_per_output_metrics_history)


def plot_metrics(train_losses, val_overall_metrics, val_per_output_metrics):
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training and Validation Metrics Over Epochs')

    # Plot Overall MSE
    axes[0, 0].plot(epochs, train_losses, label='Train MSE')
    val_mse = [m['mse'] for m in val_overall_metrics]
    axes[0, 0].plot(epochs, val_mse, label='Val MSE')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_title('Overall MSE')
    axes[0, 0].legend()

    # Plot Overall R²
    val_r2 = [m['r2'] for m in val_overall_metrics]
    axes[0, 1].plot(epochs, val_r2, label='Val R²')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].set_title('Overall R²')
    axes[0, 1].legend()

    # Plot Overall PCC
    val_pcc = [m['pcc'] for m in val_overall_metrics]
    axes[1, 0].plot(epochs, val_pcc, label='Val PCC')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('PCC')
    axes[1, 0].set_title('Overall PCC')
    axes[1, 0].legend()

    # Plot Per-Output R² (example for a few gene sets)
    for i in range(min(4, 8)):
        gs_key = f'gene_set_{i}'
        val_r2_gs = [m[gs_key]['r2'] for m in val_per_output_metrics]
        axes[1, 1].plot(epochs, val_r2_gs, label=f'GS {i} R²')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].set_title('Per-Output R² (First 4 Gene Sets)')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'metrics_plot_with_scheduler.png'))
    print("Metrics plot saved to: metrics_plot_with_scheduler.png")


# --- 程序入口 ---
if __name__ == "__main__":
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    main()
