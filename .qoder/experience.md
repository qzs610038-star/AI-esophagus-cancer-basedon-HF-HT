# PFMval 经验教训与最佳实践

> 本文件为软性指南，帮助避免历史踩坑，提升开发效率。

---

## 一、数据处理经验

### 1.1 三层交集 vs 二层交集

- **三层交集**（传 `--patches_dir`）：patches PNG ∩ CSV标签 ∩ .pt缓存 → 样本更少
- **二层交集**（不传 `--patches_dir`）：cache ∩ labels → 样本更多，效果更好
- **建议**：EGN-v2+UNI训练默认不传 `--patches_dir`

### 1.2 predictions.csv 列名格式

- `visualize_results.py` 要求列名为 `true_通路名`/`pred_通路名`
- 部分旧脚本输出 `通路名_true` 格式，导致逐通路指标解析失败
- **建议**：新脚本统一使用 `true_xxx`/`pred_xxx` 格式

### 1.3 数据集特殊路径

- `data_new_3ST/JFX0729` 不存在，实际路径为 `data_new_3ST/patch_noov_spilt/JFX0729_noov_split`
- LMZ12939同理
- 注意：`patch_noov_spilt`（原始拼写如此，含typo `spilt`而非`split`，不要修正路径名）

---

## 二、训练经验

### 2.1 dataset_name 默认值陷阱

- **历史教训**：`train_egnv2_uni.py` 默认 `dataset_name="HYZ15040_UNI"` 导致 LMZ12939 训练输出目录误标
- HYZ15040_UNI 检查点被覆盖，需重新训练
- **修复方案**：默认值设为 None + 从 `--patient` 自动推导
- `histogene/train_uni.py:167` 同样有此问题，因受保护不修改，手动指定 `--dataset_name` 规避

### 2.2 UNI特征的差异化价值

| 维度 | HisToGene | EGN-v2 |
|------|-----------|--------|
| 单患者提升 | +2.3% ~ +11.8% | +28% ~ +50% |
| 跨患者提升 | +235% | +81% |

- 对EGN-v2效果显著（替换冻结ResNet-50），对HisToGene提升有限（ViT本身有学习能力）

### 2.3 跨患者泛化训练

- 训练集：JFX0729 + LMZ12939，测试集：HYZ15040
- 所有模型跨患者泛化性能显著衰减（-31.7% ~ -77.2%）
- HisToGene-UNI衰减最小(-31.7%)，HisToGene原版衰减最大(-77.2%)

---

## 三、可视化经验

### 3.1 时间戳隔离

- 每次生成的可视化结果必须保存到独立时间戳子目录
- `generate_full_report` 只能全局调用一次
- 新模型可视化需主动检索历史规范，不能凭猜测

### 3.2 数据一致性

- 最佳epoch基于 val_loss 最小（非 val_pcc 最大），两者可能对应不同epoch
- `model_params.txt` 可能记录中间checkpoint值，完整训练历史以CSV为准
- 汇总对比时务必使用同一标准（val_loss最小epoch）

### 3.3 逐通路PCC双输出

- 图片：`pcc_barplot.png`（直观展示）
- 表格：`per_pathway_pcc.csv`（可编辑，便于筛选高预测效果通路）

---

## 四、文档编写经验

### 4.1 组会汇报文档

- 使用 `>` 块引用标注初学者注释
- 关键数据变更时同步更新所有相关表格和分析段落
- 新增模型结果需完整覆盖：指标表、衰减表、分析要点、结论

### 4.2 技术方案文档

- 面向初学者，使用ASCII数据流图
- 包含耦合度分析、代码框架、风险评估
- 实际代码路径和命令可直接复制运行

---

## 五、环境踩坑

### 5.1 Python路径

- `C:\Program Files\Python313\python.exe`：通用训练（torch可用）
- `D:\conda_envs\pfmval_py310\python.exe`：PyG相关任务（EGN-v2系列）
- 系统默认python可能无torch，不可直接使用

### 5.2 Conda环境

- pfmval环境可能不存在，base环境已有torch
- C盘空间不足，统一使用D盘部署
