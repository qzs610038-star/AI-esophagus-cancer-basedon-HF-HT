# PFMval 项目基本规则

> 本文件定义项目硬性规则，必须无条件遵守。

---

## 一、项目背景

- **项目名称**：PFMval（Pathology Foundation Model Validation）
- **研究目标**：基于食管癌H&E病理切片+空间转录组数据，预测基因通路（ssGSEA）活性评分
- **数据集**：3个食管癌患者（HYZ15040、JFX0729、LMZ12939），**不是肺癌**
- **预测目标**：30条基因通路的ssGSEA z-score值
- **工作目录**：`d:\AI空间转录病理研究\PFMval_new`

---

## 二、模型体系

| 模型 | 状态 | 特征维度 | 参数量 | 说明 |
|------|------|---------|--------|------|
| HisToGene 原版 | 活跃 | 图像直接输入 | ~70.6M | ViT架构，基线模型 |
| HisToGene-UNI | 活跃 | 1536维(UNI2-h) | ~4.0M | UNI预提取特征替代ViT |
| EGN-v1 | **已淘汰** | 2048维(ResNet50) | ~6.8M | 性能最差，不再纳入对比分析 |
| EGN-v2 原版 | 活跃 | 2048维(ResNet50) | ~3.0M | GraphSAGE+Exemplar |
| EGN-v2+UNI | 活跃 | 1536维(UNI2-h) | ~2.8M | 主力模型，跨患者泛化最强 |

**重要**：EGN-v1已淘汰，所有对比分析严格排除EGN-v1（除非用户明确要求包含）。

---

## 三、文件管理铁律

1. **受保护目录**：`histogene/`、`egnv1/`、`egnv2/` 目录下的文件**严禁修改**
2. **只新增不修改**：所有适配工作（数据加载、模型定义、训练脚本等）通过在项目根目录新建独立文件实现
3. **受保护目录的bug修复**：不修改源码，通过运行时参数（如 `--dataset_name`）显式指定正确值规避

---

## 四、运行环境

1. **HisToGene系列训练**：`C:\Program Files\Python313\python.exe`（已装torch 2.6.0+cu118）
2. **EGN-v2系列训练（需PyG）**：`D:\conda_envs\pfmval_py310\python.exe`（含torch_geometric）
3. **Conda base环境**：已装torch 2.6.0+cu118，可直接运行无需激活特定env
4. **Conda激活命令**：`(& "D:\miniconda\Scripts\conda.exe" "shell.powershell" "hook") | Out-String | Invoke-Expression`
5. **PowerShell**：不支持 `&&`，使用 `;` 分隔语句

---

## 五、训练输出标准化

1. **可视化结果**：保存至 `{model_dir}/checkpoints/results_vis/{dataset}_{timestamp}/` 时间戳子目录，**绝不覆盖历史结果**
2. **必输出文件**：
   - `model_params.txt` — 模型参数设置+关键指标
   - `training_history_{dataset}.csv` — 逐epoch训练记录
   - `predictions.csv` — 逐样本预测结果（列名：`true_通路名`/`pred_通路名`）
   - `training_curves.png` — 训练曲线图
   - `pcc_barplot.png` — 逐通路PCC柱状图
   - `per_pathway_pcc.csv` — 逐通路PCC可编辑表格（pathway/pcc/r²/mae/rank，按PCC降序）
3. **最佳epoch标准**：以 `val_loss` 最小为准（不是 val_pcc 最大）
4. **generate_full_report 函数**：全局只调用一次，避免重复创建时间戳目录

---

## 六、训练控制机制

1. **暂停训练**：创建 `PAUSE_TRAINING` 空文件触发（PowerShell：`New-Item -Path 'PAUSE_TRAINING' -ItemType File -Force`）
2. **断点续训**：`--resume <checkpoint_path>` 参数
3. **通知方式**：plyer桌面Toast弹窗，禁用Ctrl+C中断

---

## 七、数据路径

```
# 三患者数据
data_new_3ST/patch_noov_spilt/HYZ15040_noov_split/
data_new_3ST/patch_noov_spilt/JFX0729_noov_split/
data_new_3ST/patch_noov_spilt/LMZ12939_noov_split/

# 旧格式（HYZ15040独有）
HYZ15040_old/train_patches/
HYZ15040_old/val_patches/

# UNI特征缓存
uni2h_cache/{patient}/train/  和 val/

# ssGSEA标签
{patient}_ssGSEA_scores_zscore.csv

# 训练历史与检查点
{model_dir}/checkpoints/{dataset}/
{model_dir}/training_history_{dataset}.csv
```

---

## 八、Git提交规则

忽略：训练数据目录(`data_new_3ST/`)、模型权重(`*.pth`)、虚拟环境(`.venv/`)、缓存(`__pycache__/`、`uni2h_cache/`)
