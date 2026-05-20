---
name: parallel-extract
description: Parallelize feature extraction across multiple patients and backbones using subagents. Each subagent handles one (backbone, patient) combination independently.
argument-hint: <backbone> <patients> [--mode lite|full]
disable-model-invocation: true
allowed-tools: Bash, Agent
---

# Parallel Extract — 并行特征提取器

利用 Subagents 并行提取多患者 × 多 backbone 的特征，每个组合独立运行。

## 调用格式

```
/parallel-extract <backbone> <patients> [--mode lite]
```

**backbone**: `uni` | `virchow2` | `omiclip` | `all`
**patients**: `HYZ15040` | `JFX0729` | `LMZ12939` | `all`

## 并行策略

对每个 (backbone, patient) 组合，启动一个独立的 Subagent 执行提取：

```
用户: /parallel-extract virchow2 all

主 Agent 创建 3 个并行 Subagents:
┌─────────────────────────────────────────────────────┐
│ Subagent-1: extract virchow2 HYZ15040               │
│ Subagent-2: extract virchow2 JFX0729                │
│ Subagent-3: extract virchow2 LMZ12939               │
└─────────────────────────────────────────────────────┘
         ↓ 三者同时运行，无相互依赖
主 Agent 等待全部完成 → 汇总统计 → 报告
```

## 执行命令（每个 Subagent）

```bash
# UNI / Virchow2
PYTHONIOENCODING=utf-8 "C:\Program Files\Python313\python.exe" \
    "d:\AI空间转录病理研究\PFMval_new\extract_<backbone>_tokens.py" \
    --patient <patient> --mode lite

# OmiCLIP
"D:\conda_envs\loki_env\python.exe" \
    "d:\AI空间转录病理研究\PFMval_new\extract_omiclip_features.py" \
    --patient <patient> --mode lite
```

## Subagent 配置

每个 Subagent 使用以下配置：
- `subagent_type`: `general-purpose`
- `run_in_background`: `true`
- 每个 Subagent 独立工作，仅返回统计结果（不返回完整日志）

## 汇总输出

所有 Subagent 完成后，主 Agent 输出汇总：

```
并行特征提取完成 — virchow2 × 3 patients
┌──────────┬──────────┬──────────┬─────────────┐
│ Patient  │  Train   │   Val    │  磁盘占用    │
├──────────┼──────────┼──────────┼─────────────┤
│ HYZ15040 │ 3526 .pt │ 1510 .pt │   2.3 GB    │
│ JFX0729 │ 3291 .pt │ 1410 .pt │   2.1 GB    │
│ LMZ12939 │ 3761 .pt │ 1612 .pt │   2.4 GB    │
├──────────┼──────────┼──────────┼─────────────┤
│ Total    │  10578   │  4532    │   6.8 GB    │
└──────────┴──────────┴──────────┴─────────────┘
样本 shape: [65, 1280] (lite mode)
总耗时: ~35 min (并行) vs ~105 min (串行)
```

## 注意事项

- **需要足够显存**：3 个 Subagent 同时加载 backbone 可能 OOM（每个约 2.5 GB）。如果 OOM，降级为串行
- **OmiCLIP 不支持并行**：loki_env 限制，OmiCLIP 提取只能串行
- **Subagent 超时**：每个 Subagent 超时设为 2 小时（full mode 可能较慢）
- **如果某个 Subagent 失败**：不影响其他 Subagent，主 Agent 在汇总中标注失败项
