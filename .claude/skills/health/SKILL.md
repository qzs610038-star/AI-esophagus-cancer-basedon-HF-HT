---
name: health
description: Environment health check — verify Python, CUDA, key directories, and project config are ready for training.
argument-hint: [--quick] [--full]
disable-model-invocation: true
allowed-tools: Bash, Read, Glob
---

# Environment Health Check — 环境健康检查

验证 PFMval 项目训练环境是否正常，在每次开始训练前或怀疑环境异常时使用。

## 调用格式

```
/health           # 标准检查（Python + CUDA + 关键目录）
/health --quick   # 快速检查（仅 Python + CUDA，跳过目录扫描）
/health --full    # 完整检查（含磁盘空间 + 依赖版本）
```

## 检查项目

### Level 1: Python 解释器（必查）

| 解释器 | 路径 | 用途 |
|--------|------|------|
| Python 3.13 | `C:\Program Files\Python313\python.exe` | HisToGene/UNI/Virchow2/OmiCLIP |
| Conda py310 | `D:\conda_envs\pfmval_py310\python.exe` | EGN-v2/GAT (需要 PyG) |

检查逻辑：
1. 文件是否存在
2. `python --version` 输出版本号
3. 能否 `import torch`

### Level 2: CUDA 可用性（必查）

```bash
"C:\Program Files\Python313\python.exe" -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}'); print(f'CUDA version: {torch.version.cuda}')"
```

检查项：
- `torch.cuda.is_available()` == True
- GPU 名称正确（RTX 4060 Laptop / RTX 4080）
- CUDA 版本与 PyTorch 编译版本匹配（cu118）

### Level 3: 关键目录存在性（标准检查）

| 目录/文件 | 路径模式 | 说明 |
|-----------|---------|------|
| HE 切片 | `data_new_3ST/patch_noov_spilt/{HYZ15040,JFX0729,LMZ12939}_noov_split/` | 3 患者 patch 数据 |
| UNI 缓存 | `uni2h_cache/{HYZ15040,JFX0729,LMZ12939}/{train,val}/` | UNI 特征缓存 |
| ssGSEA 标签 | `data_new_3ST/ssGSEA_zscore/{HYZ15040,JFX0729,LMZ12939}_ssGSEA_zscore.csv` | 训练标签 |
| config.yaml | `config.yaml` | 路径配置文件 |
| config_utils.py | `config_utils.py` | 路径工具模块 |

### Level 4: 完整检查（--full）

- 磁盘剩余空间（D 盘，训练输出 + checkpoint 需 > 20GB）
- 关键 Python 包版本：torch, numpy, pandas, scikit-learn, pyyaml, plyer
- 可选 backbone 权重存在性：Virchow2 (model.safetensors), OmiCLIP (checkpoint.pt)
- `config_utils.py` 路径解析测试：`python config_utils.py` 不报错

## 执行模板

```bash
# 1. Python 解释器检查
ls -la "C:\Program Files\Python313\python.exe" 2>/dev/null && echo "✅ Python 3.13" || echo "❌ Python 3.13 NOT FOUND"
ls -la "D:\conda_envs\pfmval_py310\python.exe" 2>/dev/null && echo "✅ Conda py310" || echo "❌ Conda py310 NOT FOUND"

# 2. CUDA 检查
PYTHONIOENCODING=utf-8 "C:\Program Files\Python313\python.exe" -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA: {torch.version.cuda}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# 3. 关键目录
for d in "data_new_3ST/patch_noov_spilt" "uni2h_cache" "config.yaml" "config_utils.py"; do
  ls -la "$d" 2>/dev/null > /dev/null && echo "✅ $d" || echo "❌ $d MISSING"
done

# 4. 路径配置验证
PYTHONIOENCODING=utf-8 "C:\Program Files\Python313\python.exe" config_utils.py 2>&1 && echo "✅ config_utils" || echo "❌ config_utils FAILED"
```

## 输出格式

```
╔══════════════════════════════════════════╗
║     PFMval Environment Health Check     ║
╠══════════════════════════════════════════╣
║ Python 3.13 (HisToGene)  ......... ✅    ║
║ Conda py310 (EGN-v2/GAT) ...... ✅    ║
║ CUDA ....................... ✅ RTX 4060║
║ HE patches ................. ✅ 3/3     ║
║ UNI cache .................. ✅ 3/3     ║
║ ssGSEA labels .............. ✅ 3/3     ║
║ config.yaml ................ ✅         ║
║ config_utils.py ............ ✅         ║
╚══════════════════════════════════════════╝
  Status: ALL CHECKS PASSED ✅
```

如某项失败，显示 ❌ 并给出修复建议。

## 服务器端检查

在服务器上运行时，路径自动适配：
- Python: `C:\ProgramData\miniconda3\python.exe`
- 项目目录: `D:\AIPatho\qzs`
- venv: `C:\Users\AIPatho1\pfmval_env\Scripts\Activate.ps1`

如果检测到当前环境是服务器（通过检查项目路径或 hostname），自动切换检查路径。

## 使用场景

```bash
# 会话开始时
/health --quick

# 环境变更后
/health

# 训练前全面检查
/health --full
```
