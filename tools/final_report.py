import pandas as pd
import os
from config_utils import get_project_root

BASE_DIR = get_project_root()

# 从visualize_all_models.py脚本收集的数据
script_data_single = {
    ('HisToGene', 'HYZ15040'): 0.5164,
    ('HisToGene', 'JFX0729'): 0.6050,
    ('HisToGene', 'LMZ12939'): 0.5287,
    ('HisToGene-UNI', 'HYZ15040'): 0.5773,
    ('HisToGene-UNI', 'JFX0729'): 0.6121,
    ('HisToGene-UNI', 'LMZ12939'): 0.5385,
    ('EGN-v1', 'HYZ15040'): 0.2328,
    ('EGN-v1', 'JFX0729'): 0.3141,
    ('EGN-v1', 'LMZ12939'): 0.2165,
    ('EGN-v2', 'HYZ15040'): 0.4052,
    ('EGN-v2', 'JFX0729'): 0.4448,
    ('EGN-v2', 'LMZ12939'): 0.3854,
    ('EGN-v2+UNI', 'HYZ15040'): 0.6075,
    ('EGN-v2+UNI', 'JFX0729'): 0.5627,
    ('EGN-v2+UNI', 'LMZ12939'): 0.5086,
}

script_data_cross = {
    'HisToGene': 0.1815,
    'HisToGene-UNI': 0.3946,
    'EGN-v2': 0.2337,
    'EGN-v2+UNI': 0.3812,
}

# model_params.txt中的值
model_params_single = {
    ('HisToGene', 'HYZ15040'): 0.5164,
    ('HisToGene-UNI', 'HYZ15040'): 0.5773,
    ('EGN-v2+UNI', 'HYZ15040'): 0.6075,
}

model_params_cross = {
    'HisToGene': 0.1178,  # 最佳epoch 2
    'HisToGene-UNI': 0.3946,
    'EGN-v2': 0.1950,  # 最佳epoch 16
    'EGN-v2+UNI': 0.3537,  # 最佳epoch 16
}

print('\n' + '='*80)
print('完整的数据一致性对比报告')
print('='*80)

print('\n【单患者训练数据】\n')
print('模型                  数据集      脚本值     model_params   一致性')
print('-' * 80)

for (model, dataset), script_val in script_data_single.items():
    model_val = model_params_single.get((model, dataset))
    if model_val is not None:
        match = '✓' if abs(script_val - model_val) < 0.0001 else 'X'
        print(f'{model:18} {dataset:10} {script_val:.4f}    {model_val:.4f}        {match}')

print('\n【跨患者泛化数据】\n')
print('模型                  脚本值     model_params   一致性  备注')
print('-' * 80)

for model in ['HisToGene', 'HisToGene-UNI', 'EGN-v2', 'EGN-v2+UNI']:
    script_val = script_data_cross[model]
    model_val = model_params_cross[model]
    match = '✓' if abs(script_val - model_val) < 0.0001 else 'X'
    diff = abs(script_val - model_val)
    if diff > 0.0001:
        print(f'{model:18} {script_val:.4f}    {model_val:.4f}        {match}  ↑ 差异={diff:.4f}')
    else:
        print(f'{model:18} {script_val:.4f}    {model_val:.4f}        {match}')
