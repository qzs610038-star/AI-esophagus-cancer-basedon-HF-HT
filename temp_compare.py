import pandas as pd
import os

BASE_DIR = r'd:\AI空间转录病理研究\PFMval_new'

# 脚本输出的数据（从visualize_all_models.py运行结果）
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

# CSV中的最佳epoch值
csv_best = {
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

csv_best_cross = {
    'HisToGene': 0.1815,
    'HisToGene-UNI': 0.3946,
    'EGN-v2': 0.2337,
    'EGN-v2+UNI': 0.3812,
}

print('\n====== 单患者 - 脚本 vs CSV最佳epoch ======\n')
for key in script_data_single:
    s = script_data_single[key]
    c = csv_best[key]
    match = '✓' if abs(s - c) < 0.0001 else 'X'
    print(f'{match} {key[0]:15} {key[1]:10}: 脚本={s:.4f}, CSV最佳={c:.4f}')

print('\n====== 跨患者 - 脚本 vs CSV最佳epoch ======\n')
for model in ['HisToGene', 'HisToGene-UNI', 'EGN-v2', 'EGN-v2+UNI']:
    s = script_data_cross[model]
    c = csv_best_cross[model]
    match = '✓' if abs(s - c) < 0.0001 else 'X'
    print(f'{match} {model:15}: 脚本={s:.4f}, CSV最佳={c:.4f}')
