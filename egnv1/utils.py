"""EGN-v1 工具函数：指标计算等（与 EGNv2/utils.py 一致）"""

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def pearson_corrcoef(y_true, y_pred):
    """计算皮尔逊相关系数"""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_metrics(y_true, y_pred):
    """计算 MSE、MAE、R²、PCC"""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    pcc = pearson_corrcoef(y_true, y_pred)
    return {'mse': mse, 'mae': mae, 'r2': r2, 'pcc': pcc}
