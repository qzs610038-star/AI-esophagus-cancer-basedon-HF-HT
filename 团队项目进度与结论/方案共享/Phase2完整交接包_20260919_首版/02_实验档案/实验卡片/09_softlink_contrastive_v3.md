# Softlink contrastive v3：LoRA 与对比学习的无明确增量结果

**结论。** 配方升级后，冻结纯回归单点基座在单种子第 5 轮的内部双 PCC/zMSE 为 0.6649/0.8034/0.3646，XZY 为 0.5425/0.6633/0.6639；LoRA 与对比学习臂没有显示超出这一基座的明确增量。该负结果已接纳。

## 目的、对照与方法

- 目的：在统一升级配方下，分别测试 LoRA rank=2/8、中心／全局对比和空间继承是否提供超出纯回归的增益。
- 数据／方法：MPP2、既定划分、30 通路、种子 42；stage1 固定第5轮，stage2 最多60 epoch且从第6 epoch开始选择；学习率选择先看 r8 regression 第5轮内部等权 PCC，再以 zMSE 与较小学习率打平。XZY 为只读历史参考。该批共登记 40 个指标格，非三种子实验。
- 对照：`frozen_regression` 是解释基座；不同 stage 与 head 的格子按预设矩阵统一报告。

## 已登记结果与状态

| 代表性格子 | 内部等权 PCC | 内部整体展平 PCC | XZY等权 PCC | XZY整体展平 PCC |
|---|---:|---:|---:|---:|
| frozen_regression，第5轮 | 0.6649 | 0.8034 | 0.5425 | 0.6633 |
| stage1 外部等权最高：frozen centered contrastive | 0.664886 | 未在摘要中列出 | 0.543011 | 未在摘要中列出 |
| stage2 外部等权最高：r8 global contrastive/reset/spatial | 0.622299 | 0.776481 | 0.539291 | 0.659742 |

所有数值为单种子，SD 不适用。摘要格的“最高”仅定位对应分析输出；模型选择仍按预设内部合同进行。

## 停止／转向、论文用途与评审

- **停止原因：** accepted 范围将 LoRA 与对比学习归入无增量探索，主结果继续沿用冻结基线与空间消融证据。
- **论文用途：** 方法探索的负向证据；不与 Softlink local v2 的三种子四臂结果混排。
- **评审问题：** (1) 冻结回归基座是否在每个 stage 都清楚对应？(2) 多格试验怎样报告才不形成事后挑选？(3) 单种子负结果能支持到什么程度？
- **证据路径：** 当前状态见 [`../../06_证据摘录/现行实验事实_20260920.json`](../../06_证据摘录/现行实验事实_20260920.json)；历史证据见包内 [`../../06_证据摘录/实验登记摘录.json`](../../06_证据摘录/实验登记摘录.json)；原仓库结果。

## 登记定位

实验 ID `phase2_softlink_contrastive_v3`；result ID `phase2-softlink-contrastive-v3-20260909-101623-result`；result root `experiments/results/phase2_softlink_contrastive_v3/20260909_101623_341_f0de1302`。可核对指标矩阵 `.../analysis/pcc_side_by_side.csv`、登记摘要 `.../analysis/registration_summary.json` 和外部汇总 `.../external_xzy/summary.json`。Registry 未登记单一脚本或 experiment package；勿将 raw 目录中的阶段文件当作统一入口。

## 关键流程与源码

```text
共同预热后，比较冻结编码器与 rank=2/8 的 LoRA（低秩适配）候选。
各候选分别采用纯回归，或加入由标签教师分布约束的软对比损失。
用 rank=8 回归候选第 5 轮的内部 PCC 选择学习率，按既定规则打破并列。
第一阶段保留固定第 5 轮主分支与内部最佳轮次的敏感性分支。
第二阶段组合共享映射的继承/重置方式与单点/空间残差头。
第二阶段最多训练 60 轮，从第 6 轮开始按内部验证选择检查点。
固定各端点后评价 XZY；两阶段、两种选择规则分别汇总。
```

[关键实现（GitHub 固定版本）](https://github.com/qzs610038-star/AI-esophagus-cancer-basedon-HF-HT/blob/8a9c0d08991deee08f6510e1344195568bf9a70a/experiments/phase2_softlink_contrastive_v3/runner.py)。完整源码入口、历史版本与运行留档见[实验代码索引](../../06_证据摘录/实验代码索引.md)。

## 原始训练记录

本卡对应的原始训练记录、配置、训练历史与预测入口：[查看 09 号原始记录](../../07_原始训练记录/09/README.md)。
