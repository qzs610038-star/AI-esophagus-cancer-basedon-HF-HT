import os

# 读取model_params.txt文件，提取Best Val PCC值

def read_model_params(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            for line in content.split('\n'):
                if 'Best Val PCC' in line or 'Best Test PCC' in line:
                    # 从 'Best Val PCC: 0.5164' 中提取 0.5164
                    parts = line.split(':')
                    if len(parts) >= 2:
                        val_str = parts[1].strip()
                        try:
                            return float(val_str)
                        except:
                            pass
    except:
        pass
    return None

BASE_DIR = r'd:\AI空间转录病理研究\PFMval_new'

# 单患者
model_params_data = {}

# HisToGene
model_params_data[('HisToGene', 'HYZ15040')] = read_model_params(
    os.path.join(BASE_DIR, 'histogene/checkpoints/results_vis/HYZ15040_20260416_213453/model_params.txt'))

# HisToGene-UNI  
model_params_data[('HisToGene-UNI', 'HYZ15040')] = read_model_params(
    os.path.join(BASE_DIR, 'histogene/checkpoints/results_vis/HYZ15040_UNI_20260422_232743/model_params.txt'))

# EGN-v2+UNI
model_params_data[('EGN-v2+UNI', 'HYZ15040')] = read_model_params(
    os.path.join(BASE_DIR, 'egnv2/checkpoints/results_vis/HYZ15040_UNI_20260425_000841/model_params.txt'))

# 跨患者
model_params_data[('HisToGene-UNI', 'CrossPatient')] = read_model_params(
    os.path.join(BASE_DIR, 'histogene/checkpoints/results_vis/CrossPatient_JFX_LMZ_to_HYZ_20260424_221349/model_params.txt'))

model_params_data[('EGN-v2+UNI', 'CrossPatient')] = read_model_params(
    os.path.join(BASE_DIR, 'egnv2/checkpoints/results_vis/CrossPatient_JFX_LMZ_to_HYZ_UNI_20260424_232239/model_params.txt'))

print('\n====== model_params.txt中的值 ======\n')
for key, val in model_params_data.items():
    print(f'{key}: {val}')
