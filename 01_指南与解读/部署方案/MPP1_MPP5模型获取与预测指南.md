# 📋 空间转录组通路预测模型（MPP-1 至 MPP-5）共享与预测加载指南

本指南用于指导队友如何在服务器上获取 **MPP-1 至 MPP-5** 的最新统一标准重跑模型，并在后续预测实验中正确加载模型与归一化参数，最终还原出物理尺度预测结果。

---

## 一、 服务器文件获取路径

每个 MPP 训练好的 MLP 模型权重（`.pth`）与训练集的 z-score 归一化参数（`.json`）为一一绑定关系，必须配套下载：

### 1. MPP-1 模型文件（维持原最高分辨率 / 100% 步长）
*   **服务器目录**：
    `D:\AIPatho\qzs\pfmval_deploy_git\checkpoints\mpp_uni2h_mlp\mpp1_std10val_xzy_ext_uni2h_mlp_20260706\`
*   **核心下载文件**：
    *   `best_checkpoint.pth` (MLP 模型权重)
    *   `zscore_params_from_train.json` (对应归一化均值/标准差)

### 2. MPP-2 模型文件（与队列分辨率一致 0.27 MPP / 100% 步长）
*   **服务器目录**：
    `D:\AIPatho\qzs\pfmval_deploy_git\checkpoints\mpp_uni2h_mlp\mpp2_std10val_xzy_ext_uni2h_mlp_20260706\`
*   **核心下载文件**：
    *   `best_checkpoint.pth` (MLP 模型权重)
    *   `zscore_params_from_train.json` (对应归一化均值/标准差)

### 3. MPP-3 模型文件（与队列分辨率一致 0.27 MPP / 50% 重叠步长）
*   **说明**：已施加严密的空间 Embargo 邻近隔离，保证 train/val 无物理重叠泄露。
*   **服务器目录**：
    `D:\AIPatho\qzs\pfmval_deploy_git\checkpoints\mpp_uni2h_mlp\mpp3_std10val_embargo_xzy_ext_uni2h_mlp_20260706\`
*   **核心下载文件**：
    *   `best_checkpoint.pth` (MLP 模型权重)
    *   `zscore_params_from_train.json` (对应归一化均值/标准差)

### 4. MPP-4 模型文件（预训练模型最佳性能分辨率 0.54 MPP / 100% 步长）
*   **服务器目录**：
    `D:\AIPatho\qzs\pfmval_deploy_git\checkpoints\mpp_uni2h_mlp\mpp4_std10val_xzy_ext_uni2h_mlp_20260706\`
*   **核心下载文件**：
    *   `best_checkpoint.pth` (MLP 模型权重)
    *   `zscore_params_from_train.json` (对应归一化均值/标准差)

### 5. MPP-5 模型文件（预训练模型最佳性能分辨率 0.54 MPP / 50% 重叠步长）
*   **说明**：已施加严密的空间 Embargo 邻近隔离，保证 train/val 无物理重叠泄露。
*   **服务器目录**：
    `D:\AIPatho\qzs\pfmval_deploy_git\checkpoints\mpp_uni2h_mlp\mpp5_std10val_embargo_xzy_ext_uni2h_mlp_20260706\`
*   **核心下载文件**：
    *   `best_checkpoint.pth` (MLP 模型权重)
    *   `zscore_params_from_train.json` (对应归一化均值/标准差)

---

## 二、 核心预测逻辑与 Python 加载示例

> [!IMPORTANT]
> **关键机制说明**：
> 训练好的 MLP 模型输出的是位于 **z-score 归一化空间**中的值（`y_pred_z`）。队友在对切片样本输入特征执行预测后，**必须使用配套的 `zscore_params_from_train.json` 将其反归一化（Inverse Transform）还原至原始的物理尺度（Raw ssGSEA scale）**，这样输出的数值在跨 MPP 实验与后续分析中才具备统计学上的可比性与实际的生物学物理误差度量。

以下是可直接复制并嵌入后续预测 pipeline 的 Python 代码片段：

```python
import json
import torch
import torch.nn as nn
import numpy as np

# 1. 定义 MLP 模型结构（必须与训练结构完全一致）
class MppUni2hMLP(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=1024, output_dim=30, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# 2. 初始化模型并加载下载好的模型权重 (以 GPU/CPU 自适应为例)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MppUni2hMLP().to(device)
model.load_state_dict(torch.load("best_checkpoint.pth", map_location=device))
model.eval()

# 3. 读取配套的反归一化配置文件 zscore_params_from_train.json
with open("zscore_params_from_train.json", "r", encoding="utf-8") as f:
    zscore_config = json.load(f)

# 按标准通路列表顺序提取对应的 mean 和 std
pathways = zscore_config["pathways"]
means = np.array([zscore_config["means"][p] for p in pathways], dtype=np.float32)
stds = np.array([zscore_config["stds"][p] for p in pathways], dtype=np.float32)

# 4. 执行预测
# 特征输入: 维度为 [B, 1536] 的 UNI2-h 特征矩阵 (B 为 patch 数量)
# 注意：传入的特征必须与您选用的 MPP 模型分辨率对齐（例如：若使用 MPP-3，请输入 0.27 MPP 下 50% 步长提取的特征）
features_np = np.random.randn(10, 1536).astype(np.float32) # 此处替换为您的真实特征
x_tensor = torch.tensor(features_np, dtype=torch.float32).to(device)

with torch.no_grad():
    y_pred_z = model(x_tensor).cpu().numpy()  # 得到归一化空间的预测通路活性 [B, 30]

# 5. 反归一化还原：将预测值缩放并平移至原始 ssGSEA 物理尺度
y_pred_raw = y_pred_z * stds + means  # [B, 30]

print("物理尺度 Raw-scale 预测结果已输出！矩阵维度:", y_pred_raw.shape)
```
