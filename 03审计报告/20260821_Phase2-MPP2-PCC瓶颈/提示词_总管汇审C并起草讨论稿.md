# 提示词：总管汇审对话 C，并（仅当汇审同意）起草划分/z-score 讨论稿

> 把下方从「复制起点」到文末全部粘贴进**原总管对话**（不要开新的查找对话，也不要开修代码对话）。

---

## 复制起点

你是 PFMval 仓库里这次 Phase 2 / MPP2 PCC 瓶颈任务的**总管**。只做汇审、汇总和下一步拆解。不要复跑对话 C 的源码/预测诊断，不要修代码，不要训练，不要 import/accept，不要绑定 `W###`，不要写实施方案，不要 commit。

查找对话 C（划分与 z-score 只读诊断）已经结束。用户已在该对话接受判定，把结果交回给你。

### 用户交回的 C 结果（已接受，勿改写）

- 正式报告：`03审计报告/20260821_Phase2-MPP2-PCC瓶颈/划分与zscore诊断报告_20260821.md`
- 审批回传：`03审计报告/20260821_Phase2-MPP2-PCC瓶颈/审批回传_划分与zscore诊断_20260821.md`
- C 总体判定：`CONDITIONAL GO`
- 判定含义：支持总管去 `01_指南与解读/部署方案/` 写划分/z-score **讨论稿**；不是训练 GO，不是绑定 W008，没有锁定某一种新划分或新 z-score
- 硬 Bug（val/XZY 拟合 z-score）：未发现，不必回到 CA2 修复
- 用户接受日期：2026-08-21（用户原话「接收判定」）
- 查找对话已选下一动作：划分/z-score 诊断支持总管去 `01_指南与解读/部署方案/` 写讨论稿（仍不绑定工作树）

先读：

- `03审计报告/20260821_Phase2-MPP2-PCC瓶颈/00_总管任务包.md`
- `03审计报告/20260821_Phase2-MPP2-PCC瓶颈/总管汇总_A与B下一步_20260821.md`（A/B 已汇审；当时「写定版并绑定 W008」为 NO-GO，等的就是 C）
- 上面两份 C 报告/回传
- `.agents/skills/pfmval-audit/SKILL.md` 与 evidence-boundaries
- `AGENTS.md` 中「新实验部署方案与实施方案双阶段门」

```powershell
git status --short --branch
git rev-parse HEAD
python deploy/pfmval_ops.py agent start-check --strict
```

start-check 失败则本轮项目结论全部作废，只报告门禁，不写讨论稿。

### 1. 汇审范围（先做完再写任何讨论稿）

对 C 做总管抽查，不要重做逐通路诊断。至少核对：

1. C 的 locus / start-check 是否仍可复述为当前仓库事实（允许 HEAD 有新的治理提交，但须标明）
2. 证据类：C 报告与回传默认 `pending_review`；用户接受后，你可把「用户已接受该 CONDITIONAL GO」记为人工观察，**不要**把它升格为 `document_registry` 的 active 结论
3. 硬门抽查（只读打开文件，不重算全表）：`group_2/zscore_manifest.json` 的 `fit_split=train` 且排除 internal_val/external_test；`split_info.json` 的 `val_ratio_target=0.1` 且六患者都在 `fit_patients`；Registry 冻结基线内部 PCC≈0.797、外部 PCC≈0.655、raw R²≈−0.088
4. C 是否把「已核验事实 / 假设 / 不得执行」分开；oracle 均值对齐和 V3bis 是否被标成假设/historical
5. 超参数存疑（δ=1、Ridge λ=10、W007 对比与回归解耦）只作对照，不得用 XZY 反推新超参，不得重生成划分或 z-score

若抽查发现 C 把假设写成结论、用 XZY 选了方案、或实际用 val/XZY 拟合了 z-score：总体 `NO-GO`，停止，不要写讨论稿。

### 2. 必须落盘的总管汇总

写入：`03审计报告/20260821_Phase2-MPP2-PCC瓶颈/总管汇总_C与是否写讨论稿_20260821.md`

本文件默认 `pending_review`，不登记为 active 结论，不授权训练。

必须明确判定：

- 对「C 的 CONDITIONAL GO 是否成立」：`GO` / `CONDITIONAL GO` / `NO-GO`
- 对「现在修 MPP2 训练代码」：应为 `NO-GO`（除非抽查翻出硬 Bug）
- 对「现在开 Huber/W007/LoRA/Ridge 重训或改超参合同」：应为 `NO-GO`
- 对「现在写定版部署方案并绑定 W008 / 写实施方案」：应为 `NO-GO`
- 对「现在在 `01_指南与解读/部署方案/` 写一份划分/z-score **讨论稿**（非定版）」：仅当 C 汇审通过时为 `GO`

聊天里用短中文回传上述判定，并给用户批准清单。

### 3. 仅当第 2 步对「写讨论稿」为 GO 时，接着写讨论稿

路径只能是 `01_指南与解读/部署方案/`。建议文件名：

`01_指南与解读/部署方案/MPP2划分与zscore协议_讨论稿_20260821.md`

这是**讨论稿**，不是最终部署方案，不是实施方案，不创建/绑定 W###，不在工作树里建改动目录。

讨论稿必须：

- 写明来源：C 报告路径、总管汇总路径、用户接受日期
- 分开写：已核验事实 / 待讨论选项 / 明确不推荐 / 不得执行
- 选项只列代价，不锁定：
  - 患者留出 val（6 人留 1）
  - 训练患者重复 LOPO / nested CV，XZY 仍一次性
  - 维持空间划分、只改报告口径
  - 新划分下按折重拟合 train-only z-score
  - 患者内相对标准化/排名（会改任务定义）
- 明确不推荐：用 6+XZY 拟合 z-score；按 XZY 挑划分；覆盖旧 manifest
- 点名 C 的条件：当前 `split_manifest.csv` sha `1f7d1345…` ≠ 训练绑定 `babfa2b1…`；in-repo 旧 train CSV 不得当训练输入；不能把 0.80−0.65 写成已定量百分比
- 超参数存疑只附录对照，不写新 δ/λ/对比权重
- 结尾必须是「待用户审核讨论稿；批准最终方案并确认准备绑定实验之前，不得写实施方案、不得绑定 W###」
- 文首标明 lifecycle 为讨论稿 / `pending_review`

禁止把讨论稿写进 `project_state/plans/` 或 `project_state/implementation_plans/`。

### 4. 停止点

不要生成新 split，不要重拟合 z-score，不要绑定 W008，不要写 `W###-…_实施方案.md`，不要 commit，不要 push，不要 `git add` 后静默提交。写完把讨论稿路径和「请用户审核讨论稿」清单交给用户。
