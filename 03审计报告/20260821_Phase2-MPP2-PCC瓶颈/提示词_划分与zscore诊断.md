# 提示词：划分与 z-score 只读诊断（复制到新对话）

> 硬门禁：在用户把 CA2 审批回传中的硬 Bug 结论接受为“全过”之前，不要开这个对话。  
> 开对话时，用户应把 CA2 报告路径和总体判定粘贴进下方「CA2 前置结果」。

---

## 复制起点

你是 PFMval 仓库里一次**只读划分与 z-score 诊断**对话。总管对话已经拆好任务。CA2 若已排除一般性硬 Bug，你只诊断现行 train / internal-val / external 划分和 z-score 是否足以解释外部 PCC 停在约 0.65，以及是否值得另开新协议讨论。

### CA2 前置结果（由用户粘贴）

- CA2 报告路径：
- CA2 总体判定：
- 硬 Bug 是否全过：是 / 否
- 用户接受日期：

若硬 Bug 未全过，立即停止，只回复：本对话不得开始，请先回到代码修复。

### 1. 本次任务的目的

在确认不是一般性硬 Bug 之后，回答：

> 现行 MPP2 协议下，internal-val 约 0.80 与 external XZY 约 0.65 的差距，有多少可归因于「患者内空间 10% 验证、同一患者身份进入 train 与 val」以及「train-only z-score 的目标/尺度」？是否值得另开患者级划分或 z-score 协议讨论稿？

本次是诊断，不是新实验。禁止生成新的可比较 PCC，禁止重生成受保护划分和 z-score，禁止用 XZY 选择更好的划分或标准化。

### 2. 操作规范与边界检测标准

#### 2.1 必读顺序

1. `AGENTS.md`（尤其：监督学习预处理必须在训练集拟合；不得用 XZY 拟合 z-score 或选 checkpoint；MPP 原始 ssGSEA、标准划分、z-score 参数、manifest 为受保护资产）
2. `.agents/skills/pfmval-audit/SKILL.md` 与 evidence-boundaries
3. `project_state/current_state.json` 相关字段
4. `experiments/experiment_registry.json` 中：
   - `mpp2_barcode_repair_v003_frozen_baseline_20260711`
   - `mpp2_std10val_xzy_ext_uni2h_mlp_20260706`（历史对照，不升格）
   - `mpp2_huber_loss_paired_v001_20260728`
   - `mpp2_cpgcr_probe_v001_20260811`
5. 现行 split / z-score 清单与参数文件（只读）
6. 仅 `lifecycle=active` 的 document_registry 记录

历史患者级 val（例如旧 V3bis：5 训练 + 1 患者 val + XZY）只作对照，证据类为 `historical`，不得写成当前协议或新结论。

```powershell
git status --short --branch
git rev-parse HEAD
python deploy/pfmval_ops.py agent start-check --strict
```

#### 2.2 工作树与写入

- 未绑定 `W###`。只读诊断。
- 本地 `explore` 不得访问服务器、不得生成可比较实验结论。本任务需要读受保护清单时只读、不改、不重建。
- 禁止写 `scripts/generate_standard_splits.py` 输出，禁止跑 `rebuild_zscore_from_manifest.py` 覆盖生产参数。
- 禁止起草已定版部署方案并绑定 W008。若诊断支持改协议，只在报告里写“建议总管下一步在 `01_指南与解读/部署方案/` 另开讨论稿”，你自己不要写那份讨论稿，除非用户在本对话里另外明确点名。
- 唯一写入：`03审计报告/20260821_Phase2-MPP2-PCC瓶颈/`

#### 2.3 必须核查的问题

1. 现行划分单位：患者、切片，还是空间 block？每位患者实际切片数？
2. 6 名患者是否都同时出现在 train 与 internal-val？XZY 是否仅在冻结后评估一次？
3. checkpoint / 早停信号是否来自与训练共享患者身份的 internal-val？
4. z-score 的 `fit_split`、患者集合、ddof、是否 clip；val 与 XZY 是否只 transform
5. 逐通路比较 train / internal-val / XZY 的均值、标准差、预测振幅（只用已有预测和受保护参数）
6. 能否在**不读 XZY 选方案**的前提下，说明 PCC 尚可而 raw R² 为负的机制；任何“若改划分/z-score 可能如何”只能标为假设

#### 2.4 判定标准

- 协议描述与代码/清单一致：记 `current` 事实，不是 Bug
- 发现实际拟合用了 val 或 XZY：这是硬 Bug，应立即 `NO-GO` 并请用户回到 CA2/修复，不要继续谈新协议
- 发现现行协议会让 internal 指标偏乐观：可建议“值得另开患者级划分讨论”，但**不**等于批准新实验
- 候选方向只列选项与代价，不锁定：
  - 患者留出 val（6 人留 1）
  - 训练患者重复 LOPO / nested CV，XZY 仍一次性
  - 维持空间划分、只改报告口径
  - 新划分下按折重拟合 train-only z-score
  - 患者内相对标准化/排名（会改任务定义）
- 明确不推荐：用 6+XZY 拟合 z-score；按 XZY 挑划分；覆盖旧 manifest

### 3. 必须回传的审批报告

#### 3.1 落盘

- 正式报告：`03审计报告/20260821_Phase2-MPP2-PCC瓶颈/划分与zscore诊断报告_20260821.md`
- 审批回传：`03审计报告/20260821_Phase2-MPP2-PCC瓶颈/审批回传_划分与zscore诊断_20260821.md`

正式报告用审计模板，总体判定只回答：“是否支持总管去写划分/z-score 部署讨论稿”，不是训练 GO。

必须分开写：已核验事实 / 假设 / 不得执行的下一步。

#### 3.2 聊天短回传

中文先给：划分和 z-score 各自能解释什么、不能解释什么；再给是否建议总管写部署讨论稿；最后给用户批准清单。

#### 3.3 停止点

不要生成新 split，不要重拟合 z-score，不要绑定 W008，不要 commit。写完把决定交回总管。
