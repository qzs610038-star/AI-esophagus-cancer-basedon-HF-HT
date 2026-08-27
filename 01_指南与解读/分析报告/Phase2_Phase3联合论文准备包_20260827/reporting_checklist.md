# TRIPOD+AI / PROBAST+AI 论文准备检查表

> 该表仅用于稿件准备参考，不是正式 TRIPOD+AI 条目映射或 PROBAST+AI 四域逐题评估，也不代表低偏倚。状态：`complete`、`partial`、`missing`、`not_applicable_now`。

| 领域 | 检查项 | 状态 | 当前证据或缺口 |
|---|---|---|---|
| 标题/摘要 | 明确为模型开发、验证或探索性再分析 | complete | 标题与摘要写 evidence-constrained / 受限桥接 |
| 研究背景 | 明确目标人群、任务与拟使用场景 | partial | 已写 ESCC 新辅助治疗反应、78 位患者/307 张有效切片；中心、时段、拟用场景和行动阈值待作者补充 |
| 参与者 | 纳入/排除标准、来源时间、中心 | missing | 需作者补充完整队列流程图和时间范围 |
| 样本单位 | 区分 patient、slide、spot、fold、seed | complete | 方法与图表规则已明确 |
| 结局 | pCR/MPR 定义、阈值、评估者、盲法 | partial | 已展开病理完全缓解/主要病理缓解；阈值、事件数、评估者和盲法待作者补充 |
| 预测因子 | H&E、UNI2-h、30 通路的来源与时点 | partial | 代码映射已完成；标本治疗前后时点待补充 |
| 数据处理 | 缺失值、异常值、标准化 | partial | train-only z-score 边界明确；队列缺失值流程待补充 |
| 数据划分 | 患者级隔离与所有数据驱动选择 | missing for W004 | 当前切片级重复留出存在患者重叠，只能 compatibility-only |
| 样本量 | 说明样本量依据与事件数 | missing | 需 pCR/MPR 事件数和外部验证精度目标 |
| 模型 | 架构、超参数、优化器、选模 | complete for reviewed code | W007/W004 方法—代码映射已给出 |
| 可复现性 | 代码、模型、依赖、随机种子 | partial | W004 current；W007 仅 historical source-bound，main 缺完整训练树 |
| 判别 | AUROC 与 AUPRC | partial | Registry 臂级值仅在 COMPATIBILITY_ONLY 范围接纳；新 pooled/case 复算待审，患者级正式证据缺失 |
| 连续预测 | PCC、CCC、误差、R² | partial | W007 accepted 基础值；新逐通路复算待用户审核 |
| 校准 | 截距、斜率、曲线、Brier | partial | W004 可描述性复算；患者独立校准未完成 |
| 不确定性 | 患者级 95% CI / cluster bootstrap | missing | 现有 fold/spot 不能替代患者级区间 |
| 外部验证 | 多患者、独立中心/时间 | missing | 外部病例 E1 仅一名患者，不能称外部人群验证 |
| 模型比较 | 相同患者、seed、fold 的配对比较 | partial | W004 arm 可同 fit 配对，但训练/测试患者重叠 |
| 负对照 | 容量、随机、置乱与空间反事实 | partial | A004 为坐标反事实；PCA-30/random-30/patient shuffle 尚缺 |
| 亚组/公平性 | 预设亚组、敏感性与公平性 | missing | 样本量和临床变量待作者核查 |
| 临床效用 | 决策阈值与 DCA | not_applicable_now | 无预设临床行动阈值且样本不足，不做强行 DCA |
| 偏倚风险 | 参与者/预测因子/结局/分析四域 | partial | 已列主要风险，正式 PROBAST+AI 逐问答表待补 |
| 局限性 | 患者重叠、单 seed、单外部患者、同源分支 | complete | 中英文稿均逐项披露 |
| 数据与代码开放 | 可访问性、许可与限制 | partial | 本地路径已列；人类数据不得随公开代码发布，公开共享政策待项目负责人决定 |
| 伦理 | 审批号、同意/豁免、隐私 | missing | 需伦理委员会全称、批准号/日期、同意或豁免、去标识化与访问控制；不得虚构 |
| AI 使用 | 检索/起草/核验披露 | partial | 已披露用途与人工责任；机构对 AI 处理本地人类数据的合规判断待项目负责人确认 |

## 投稿前 HARD 门

1. 补全伦理与队列信息。
2. 明确是否将 W004 放主文或补充材料；若在主文，`COMPATIBILITY_ONLY` 必须紧邻结果。
3. 不得把 W007→W004 虚线改成已验证实线。
4. 正式主张若涉及性能增量，必须有患者级拆分、配对区间和容量匹配负对照。
5. 所有逐通路推断需预先定义多重比较控制；否则仅作描述。
