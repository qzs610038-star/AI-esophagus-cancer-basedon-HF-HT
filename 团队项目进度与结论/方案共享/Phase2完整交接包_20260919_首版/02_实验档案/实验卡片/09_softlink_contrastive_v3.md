# Softlink contrastive v3：LoRA 与对比学习的无明确增量结果

**结论。** 配方升级后，冻结纯回归单点基座在单种子第 5 轮的内部双 PCC/zMSE 为 0.6649/0.8034/0.3646，XZY 为 0.5425/0.6633/0.6639；LoRA 与对比学习臂没有显示超出这一基座的明确增量。该负结果已接纳。

## 目的、对照与方法

- 目的：在统一升级配方下，分别测试 LoRA rank=2/8、中心／全局对比和空间继承是否提供超出纯回归的增益。
- 数据／方法：MPP2、既定划分、30 通路、种子 42；stage1 固定第5轮，stage2 最多60 epoch且从第6 epoch开始选择；学习率选择先看 r8 regression 第5轮内部等权 PCC，再以 zMSE 与较小学习率打平。XZY 为只读历史参考。该批共登记 40 个指标格，非三种子实验。
- 对照：`frozen_regression` 是解释基座；不同 stage 与 head 的格子不是可随意挑选的“最好结果”。

## 已登记结果与状态

| 代表性格子 | 内部等权 PCC | 内部整体展平 PCC | XZY等权 PCC | XZY整体展平 PCC |
|---|---:|---:|---:|---:|
| frozen_regression，第5轮 | 0.6649 | 0.8034 | 0.5425 | 0.6633 |
| stage1 外部等权最高：frozen centered contrastive | 0.664886 | 未在摘要中列出 | 0.543011 | 未在摘要中列出 |
| stage2 外部等权最高：r8 global contrastive/reset/spatial | 0.622299 | 0.776481 | 0.539291 | 0.659742 |

所有数值为单种子，SD 不适用。摘要格的“最高”仅定位对应分析输出，不能作为外部后选模型依据。

## 停止／转向、论文用途与评审

- **停止原因：** accepted 范围明确接受无增量结果，不把 LoRA 或对比学习解释为已带来有效增益。
- **论文用途：** 方法探索的负向证据；不与 Softlink local v2 的三种子四臂结果混排。
- **评审问题：** (1) 冻结回归基座是否在每个 stage 都清楚对应？(2) 多格试验怎样报告才不形成事后挑选？(3) 单种子负结果能支持到什么程度？
- **证据路径：** 包内 [`../../06_证据摘录/实验登记摘录.json`](../../06_证据摘录/实验登记摘录.json)；原仓库结果。

## 登记定位

实验 ID `phase2_softlink_contrastive_v3`；result ID `phase2-softlink-contrastive-v3-20260909-101623-result`；result root `experiments/results/phase2_softlink_contrastive_v3/20260909_101623_341_f0de1302`。可核对指标矩阵 `.../analysis/pcc_side_by_side.csv`、登记摘要 `.../analysis/registration_summary.json` 和外部汇总 `.../external_xzy/summary.json`。Registry 未登记单一脚本或 experiment package；勿将 raw 目录中的阶段文件当作统一入口。
