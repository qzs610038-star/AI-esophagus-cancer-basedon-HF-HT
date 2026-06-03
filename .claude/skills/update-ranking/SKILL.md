---
name: update-ranking
description: Update model performance ranking document after training completes. Auto-extracts metrics, sorts tables, and refreshes key conclusions.
argument-hint: [<checkpoint-dir> | --scan | --from-log]
disable-model-invocation: true
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# Update Ranking — 模型性能排名更新

训练完成后，自动更新 `模型性能排名_Model_Performance_Ranking.md`，确保排名文档始终反映最新实验状态。

## 调用格式

```
/update-ranking                          # 扫描最近 checkpoint，交互式选择要录入的模型
/update-ranking <checkpoint-dir>         # 指定 checkpoint 目录，自动提取并更新
/update-ranking --scan                   # 扫描全部 checkpoint，列出未录入的实验
/update-ranking --from-log               # 从 experiments_log.csv 重建排名表
```

## 更新流程

### Step 1: 提取训练结果

从指定 checkpoint 目录读取：

```bash
# 1.1 定位 training_history.csv
# 1.2 找到 val_loss 最小的 epoch（最佳 epoch）
# 1.3 提取该 epoch 的 val_pcc, val_mae, val_r², train_loss, train_pcc
# 1.4 如果有 per_pathway_pcc.csv，提取 Top 5 通路
# 1.5 读取 args.json（如果存在）获取超参数
```

### Step 2: 推断模型身份

从目录名 + args 推断：

| 目录名模式 | 模型 | Backbone | 配置 |
|-----------|------|----------|------|
| `*AugMix*tv_*` | HisToGene-UNI-Tokens + AugMix + TV | UNI2-h 1536d | 含 TV Loss |
| `*AugMix*` | HisToGene-UNI-Tokens + AugMix | UNI2-h 1536d | AugMix 增强 |
| `*UNI_tokens*` | HisToGene-UNI-Tokens | UNI2-h 1536d | 方案B base |
| `*UNI*GAT*` | HisToGene-UNI + GAT | UNI2-h + Graph | GAT 变体 |
| `*Virchow2*` | HisToGene-Virchow2 Tokens | Virchow2 1280d | — |
| `*OmiCLIP*` / `*omiclip*` | HisToGene-OmiCLIP | OmiCLIP 768d | — |
| `*egnv2*UNI*` | EGN-v2+UNI | UNI2-h + GNN | — |
| `*egnv2*` | EGN-v2 (ResNet-50) | ResNet-50 + GNN | — |
| `*HisToGene*orig*` | HisToGene 原版 (ViT) | ViT-Large | — |

### Step 3: 判断训练类型

| 条件 | 类型 | 更新表 |
|------|------|--------|
| 目录名含 `CrossPatient` + `Fold` | 跨患者单折 | 快速排名总表 + 三折CV表 |
| 目录名含 `CrossPatient` 无 Fold | 跨患者（旧格式）| 快速排名总表 |
| 目录名含患者名（HYZ/JFX/LMZ）无 CrossPatient | 单患者 | 单患者表 |
| 三折均完成 | 跨患者 3 折汇总 | 三折CV对比表 |

### Step 4: 更新排名文档

对每个需要更新的表：

1. **快速排名总表**（跨患者 Fold1）：
   - 如果同模型已有旧记录：比较 PCC，保留更优的，在备注中标记 `🔄 更新于 YYYY-MM-DD`
   - 如果新模型：插入正确排名位置
   - 重新排序（按 PCC 降序）

2. **三折交叉验证对比表**：
   - 如果是三折的新 fold：填入对应列
   - 如果三折齐了：计算均值，填入 3 折平均列

3. **单患者训练性能表**：
   - 按 backbone 分组更新
   - 填入对应患者列的 PCC

4. **架构超参数速查**：
   - 如果是新模型：追加 `### N. ModelName` 节
   - 如果是已有模型的更新：更新超参 + 最佳结果行

5. **关键结论**（文档末尾编号列表）：
   - 如果新模型进入 Top 3：添加结论
   - 如果刷新某 backbone 最佳：更新相关结论
   - 如果结论总数 > 12：合并次要结论

### Step 5: 更新元数据

- 文档顶部 `> 更新时间：YYYY-MM-DD` 改为当前日期
- 在更新说明注释中添加一行：`> **YYYY-MM-DD 更新**：<变更摘要>`

## 关键规则

1. **不覆盖历史**：同名模型的旧记录保留在备注中（如 `旧: PCC=0.3835, 2026-05-01`）
2. **过拟合 Gap 自动计算**：`Gap = train_pcc - val_pcc`（取最佳 epoch）
3. **权重路径**：记录相对于项目根目录的路径，不硬编码盘符
4. **单患者不更新跨患者表**：单患者结果只更新单患者表
5. **EGN-v1 不录入**：如果检测到 EGN-v1 结果，忽略并提示
6. **OmiCLIP 跨患者标注警告**：跨患者 OmiCLIP 结果在备注中加 `⚠️ 跨患者不建议`

## 与 /experiment-log 联动

如果 `experiments_log.csv` 存在，优先从日志读取结构化数据（避免从目录名推断的不确定性）。

```bash
# 流程：
# 1. 检查 experiments_log.csv 是否存在
# 2. 如果有日志：搜索对应 dataset_name 的行
# 3. 如果无日志：从 training_history.csv + 目录名提取
# 4. 更新排名文档后，提示是否也要 /experiment-log --update
```

## 示例

```bash
# 训练完成后一键更新
/update-ranking "checkpoints/CrossPatient_JFX_LMZ_to_HYZ_UNI_tokens_AugMix_20260530"

# 扫描全部 checkpoint，找出未录入的实验
/update-ranking --scan

# 从实验日志完全重建排名表（谨慎使用，会覆盖手工编辑的注释）
/update-ranking --from-log
```

## 输出示例

```
╔══════════════════════════════════════════════╗
║  模型性能排名更新                              ║
╠══════════════════════════════════════════════╣
║ 模型: HisToGene-UNI-Tokens + AugMix + TV     ║
║ Backbone: UNI2-h (1536d)                     ║
║ 配置: Fold1 (JFX+LMZ→HYZ), epoch=14          ║
║ Val PCC: 0.4170 → 排名 #2 (不变)              ║
║ Val MAE: 0.0399 | R²: 0.1512                 ║
╠══════════════════════════════════════════════╣
║ 更新表: 快速排名总表 ✅                        ║
║ 更新表: 三折CV对比表 ✅                        ║
║ 更新表: 架构超参数速查 ✅                       ║
║ 更新表: 关键结论 ✅                            ║
╠══════════════════════════════════════════════╣
║ 文档已更新: 模型性能排名_Model_Performance_Ranking.md ║
╚══════════════════════════════════════════════╝
```
