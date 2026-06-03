---
name: post-train
description: One-stop post-training pipeline — locate latest checkpoint, extract metrics, generate report, log experiment, update ranking, check doc sync.
argument-hint: [<checkpoint-dir> | --latest | --all]
disable-model-invocation: true
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# Post-Train — 训练后一站式流水线

训练完成后，串联全部后续步骤，避免遗漏。

## 调用格式

```
/post-train                        # 自动定位最新 checkpoint，交互式确认
/post-train <checkpoint-dir>       # 指定 checkpoint 目录
/post-train --latest               # 无交互，直接处理最新 checkpoint
/post-train --scan                 # 扫描全部 checkpoint，列出待处理
```

## 执行流程

### Phase 1: 定位 & 提取

```
1. 定位最新 checkpoint 目录（按 modification time）
2. 读取 training_history.csv → 找到 val_loss 最小 epoch
3. 提取最佳 epoch 指标: val_pcc, val_mae, val_r², train_pcc
4. 读取 args.json（如有）获取超参数
5. 读取 per_pathway_pcc.csv（如有）获取 Top 5 / Bottom 5 通路
```

### Phase 2: 可视化验证

```
6. 检查 results_vis 目录是否已有完整报告
7. 如缺失: 调用 generate_full_report
8. 验证 6 个标准输出文件齐全
```

### Phase 3: 记录 & 排名

```
9. 运行 /experiment-log <checkpoint-dir>
10. 运行 /update-ranking <checkpoint-dir>
```

### Phase 4: 文档同步检查

```
11. 检查 CLAUDE.md 模型体系表是否需要更新（新模型 / PCC 变化）
12. 检查 basic_rule.md 模型体系是否需要更新
13. 检查 experience.md 最佳性能基线是否需要更新
14. 检查 README.md 是否需要更新
15. 生成更新建议（diff 预览），逐项确认
```

### Phase 5: 记忆更新

```
16. 如有新的踩坑经验，提示保存到 memory
17. 如有新的 backbone/方法，提示更新 maintenance-plan.md
```

## 跳过策略

| 步骤 | 跳过条件 |
|------|---------|
| generate_full_report | 已有完整报告（6 文件齐全） |
| experiment-log | experiments_log.csv 中已有同名记录 |
| update-ranking | PCC 未进入 Top 10 或无明显变化 |
| CLAUDE.md 更新 | 非新模型且 PCC 变化 < 0.005 |
| basic_rule.md | 非新模型 |
| experience.md | PCC 未刷新 baseline |

## 输出示例

```
╔══════════════════════════════════════════════╗
║        Post-Train Pipeline                   ║
╠══════════════════════════════════════════════╣
║ Checkpoint: CrossPatient_JFX_LMZ_to_HYZ_... ║
║ Model: HisToGene-UNI-Tokens + AugMix + TV   ║
║ Best Epoch: 14 | Val PCC: 0.4170            ║
╠══════════════════════════════════════════════╣
║ ✅ Phase 1: Metrics extracted                ║
║ ✅ Phase 2: Report verified (6/6 files)      ║
║ ✅ Phase 3: Experiment logged                ║
║ ✅ Phase 3: Ranking updated (rank #2)        ║
║ ⚠️ Phase 4: CLAUDE.md PCC changed 0.4142→0.4170 — update? ║
║ ⬜ Phase 4: basic_rule.md — skip (not new model) ║
║ ⬜ Phase 4: experience.md — skip (no baseline change) ║
║ ⬜ Phase 5: Memory — no new pitfalls detected ║
╚══════════════════════════════════════════════╝
```

## 与其他 Skill 的关系

```
/post-train
  ├── generate_full_report (if missing)
  ├── /experiment-log
  ├── /update-ranking
  └── manual review: CLAUDE.md / basic_rule.md / experience.md / README.md
```

## 示例

```bash
# 训练完成后
/post-train

# 指定目录
/post-train "checkpoints/CrossPatient_Fold1_UNI_AugMix_TV_20260530"

# 批量处理（多个新 checkpoint）
/post-train --scan
```