---
name: experiment-log
description: Auto-log training experiment results to a centralized CSV after each training run. Enables quick comparison and prevents lost experiment records.
argument-hint: <checkpoint-dir> [--update]
disable-model-invocation: true
allowed-tools: Bash, Read, Write, Glob
---

# Experiment Log — 实验记录器

自动将训练结果追加到中心化实验日志 CSV，与 `/compare` 联动。

## 调用格式

```
/experiment-log <checkpoint-dir>
/experiment-log --update  # 扫描全部 checkpoint 目录，补录遗漏
/experiment-log --list    # 列出所有已记录实验
```

## 日志文件

`experiments_log.csv` 位于项目根目录，列结构：

```csv
timestamp,dataset_name,backbone,patient_config,train_patients,eval_patient,
best_epoch,val_loss,val_pcc,val_mae,val_r2,
lr,dropout,mixup_alpha,batch_size,feature_dim,encoder_layers,
checkpoint_path,notes
```

## 数据来源

从 `training_history.csv` 自动提取：
1. 找到 `val_loss` 最小的 epoch
2. 提取该 epoch 的所有指标
3. 从 `args.json` 或目录名推断超参数

## 执行逻辑

```python
# 1. 定位 checkpoint 目录下的 training_history.csv
history = pd.read_csv(f"{checkpoint_dir}/training_history.csv")
best_idx = history['val_loss'].idxmin()
best = history.iloc[best_idx]

# 2. 尝试读取 args.json 获取超参数
try: args = json.load(open(f"{checkpoint_dir}/args.json"))
except: args = infer_from_dirname(checkpoint_dir)

# 3. 追加到 experiments_log.csv
log_entry = {
    'timestamp': datetime.now().isoformat(),
    'dataset_name': ...,
    'best_epoch': int(best['epoch']),
    'val_pcc': float(best['val_pcc']),
    ...
}
append_to_csv('experiments_log.csv', log_entry)
```

## 使用场景

```bash
# 训练完成后记录
/train virchow2 HYZ15040 --epochs 50
/experiment-log checkpoints/HisToGene_Virchow2_HYZ15040_20260519_150000

# 补录历史实验
/experiment-log --update

# 查看实验总览
/experiment-log --list
```

## 与 /compare 联动

`/compare` 命令自动读取 `experiments_log.csv`，无需手动指定路径：

```bash
/compare --backbone uni --patient HYZ15040   # 自动从日志找到对应实验
/compare --latest 3                           # 对比最近 3 个实验
```

## CSV 示例

```csv
timestamp,dataset_name,backbone,patient_config,best_epoch,val_pcc,val_mae,lr,dropout
2026-05-19T15:30,HYZ15040_Virchow2_Tokens,Virchow2,single,42,0.4183,0.0399,5e-05,0.5
2026-05-19T16:45,CrossPatient_UNI_Tokens,UNI2-h,cross_fold1,38,0.4095,0.0417,5e-05,0.5
```
