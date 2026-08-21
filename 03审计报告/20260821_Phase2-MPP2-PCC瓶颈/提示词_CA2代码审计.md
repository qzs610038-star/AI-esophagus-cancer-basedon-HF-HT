# 提示词：CA2 代码审计（复制到新对话）

> 用法：新开一个规划/只读对话，从「复制起点」复制到文末，作为该对话的第一条用户消息。  
> 本提示词是完整任务说明书。新对话必须独立查仓库，不得把总管计划里的怀疑写成已证实缺陷。

---

## 复制起点

你是 PFMval 仓库里一次**只读 CA2 代码审计**对话。总管对话已经拆好任务；你负责查找和写审计报告，不负责修代码、不负责开新实验、不负责汇总后续路线。

### 1. 本次任务的目的

Phase 2 / MPP2 通路预测的外部 XZY pooled PCC 长期停在约 0.65。后续 Huber、Ridge、W007 残差臂都没有实质抬升。

本次只回答一个可检验命题：

> 在 MPP2 修复基线及其后 Huber / Ridge / W007 实现中，CA2 规定的六类一般性硬 Bug 是否存在。若存在，列出必须先修的代码点；若不存在，给出“停止继续怀疑一般性硬 Bug”的审计依据。

本次**不**解释 PCC 0.65 的科学上限，**不**重做划分，**不**重做 z-score，**不**评价超参数是否“还能再调”（那是并行的另一场对话）。

用户已确认审计范围：

| 角色 | experiment_id | 工作树/备注 |
|---|---|---|
| 修复冻结基线 / FBR 复放锚点 | `mpp2_barcode_repair_v003_frozen_baseline_20260711` | 无 W###；accepted |
| Huber 配对 | `mpp2_huber_loss_paired_v001_20260728` | W001 |
| Ridge 校准 | `mpp2_pathway_ridge_calibration_v001_20260717` | status=`failed`，evidence=`rejected`，仍要审是否读过 XZY 选参 |
| W007 四臂 | `mpp2_cpgcr_probe_v001_20260811` | W007；FBR/RCC/HCR/CPGCR |
| LoRA | 不在本次 CA2 主审 | 超参数对话另表 |

数据清单：`barcode-repair-20260711-d626ad8-v003:1204018178a4d355`。

### 2. 操作规范与边界检测标准

#### 2.1 必读顺序

1. `AGENTS.md`
2. `.agents/skills/pfmval-audit/SKILL.md` 及其 `references/evidence-boundaries.md`、`assets/audit-report-template.md`
3. `project_state/current_state.json` 中与 MPP2、accepted results、blocked_actions 相关的字段
4. `experiments/experiment_registry.json` 中上表四个 experiment 记录（只读相关条目，不要整表当结论）
5. `project_state/document_registry.json` 中与本任务相关且 `lifecycle=active` 的记录；其他生命周期不得当当前结论
6. 涉及路径时只读 `configs/server_paths.yaml` 相关项；`md_import` 未核验条目不得当执行依据

记录 locus：

```powershell
git status --short --branch
git rev-parse HEAD
python deploy/pfmval_ops.py agent start-check --strict
```

出具任何 PASS/FAIL/GO/NO-GO 之前必须有本次新鲜的 start-check 输出。start-check 失败则停止写结论，只报告阻塞。

#### 2.2 工作树与写入

- 本对话**未绑定** `W###`。只允许只读定位。
- 禁止写实验代码、禁止改 `main` 业务代码、禁止训练、禁止派发、禁止 result import/accept、禁止改 Registry、禁止重生成 ssGSEA / 标准划分 / z-score 参数 / manifest。
- 唯一允许的写入：本审计报告落到  
  `03审计报告/20260821_Phase2-MPP2-PCC瓶颈/`
- 不要创建实施方案，不要写 `01_指南与解读/部署方案/` 新实验稿。

#### 2.3 证据纪律

- 从当前仓库源码和已接受产物核验，不要复述总管聊天、W007 待审归因报告或历史分析报告当结论。
- W007 原因分析 / 深度归因 / 四臂核验文档若被读到，最多标为 `historical` 或 `pending_review` 对照，不能替代源码与 accepted bundle。
- 每个证据必须归类：`current` / `accepted` / `pending_review` / `diagnostic_only` / `explore_only` / `historical` / `missing`。
- 源码可读 ≠ 运行通过；测试通过 ≠ 实验已接受；文件存在 ≠ 已验证。

#### 2.4 CA2 只审这六项

每一项必须同时给出两栏，禁止把协议选择写成硬 Bug：

| 项 | 硬 Bug（FAIL 则必须先修代码） | 协议/科学设计（可 WARN，但不是硬 Bug） |
|---|---|---|
| CA2-1 图像、通路、标签、患者 ID 是否逐行对应 | 坐标/barcode/ID 对不齐；静默行序 fallback 造成错配；特征 `.pt` 与标签行、患者 ID 不一致仍继续训练 | 匹配策略偏保守但无错配证据 |
| CA2-2 z-score 是否只在训练折拟合 | val 或 XZY 参与 mean/std；跨折串用参数；inverse-z 用了另一套参数 | train-only 正确，但该标准化与 raw 指标不完全同目标 |
| CA2-3 同一患者全部切片是否进入同一集合 | 违反**当时登记协议**的泄漏：同一 patch/切片同时进 train 与 val，或 XZY 进入训练/选模 | 现行登记协议本就是 6 患者内空间 10% val，同一患者身份同时出现在 train 与 val。这不是硬 Bug，记为协议发现 |
| CA2-4 baseline checkpoint 能否精确复放 | 无法复放登记的 val_loss/PCC；FBR 声称复放但权重/数据对不上 | 复放通过，但选模指标可能偏乐观 |
| CA2-5 新增参数是否进入优化器并真实改变预测 | 新头/损失未进计算图；Huber/RCC/HCR/CPGCR 预测与对照几乎相同却声称已训练；参数未注册 optimizer | 有可测差异但幅度很小，可能被训练设置压住（交给超参数对话，不记硬 Bug） |
| CA2-6 external 是否完全未参与选模、阈值、超参数 | 用 XZY 早停、选 checkpoint、选 λ、调损失权重或阈值 | 合同未读 XZY，但 internal-val 与 train 共享患者身份（协议问题，不是 CA2-6 硬 Bug） |

Ridge 虽已 `rejected`，仍必须核查 r002/r003 有没有在冻结校准器之前读 XZY。

#### 2.5 建议核验面（线索，不是结论）

独立打开并核对，不要假设它们有问题：

- 对齐：`dataset_mpp.py`、特征缓存 stem、标签 CSV、`split_manifest`
- z-score：`scripts/rebuild_zscore_from_manifest.py`、group_2 的 `zscore_manifest.json` / `zscore_params_from_train.json`、训练脚本加载路径
- 划分协议：`scripts/generate_standard_splits.py`、group_2 `split_info`
- 训练与选模：`train_mpp_uni2h_mlp.py`；W001 Huber 代码；W007 `experiments/workspaces/W007/mpp2_cpgcr_probe/`
- 复放：W007 FBR `baseline_replay_check.json` 及对应 accepted bundle
- 预测差：Huber 对照与处理臂；W007 各臂与 FBR 的预测文件（只读已有产物，不重训）

只读受保护资产可以；重新生成它们不行。

#### 2.6 判定出口

- 六项对硬 Bug 全 PASS（允许协议栏 WARN）：总体对“停止怀疑一般性硬 Bug”给 `GO` 或带点名 WARN 的 `CONDITIONAL GO`
- 任一项硬 Bug FAIL：总体 `NO-GO`，列出最小修复面，**不要动手修**
- 证据不足：该项 `FAIL` 或明确 `missing`，不要写成 PASS

总体判定只覆盖“可否停止怀疑一般性硬 Bug”，不要对划分改进或新训练给 GO。

### 3. 必须回传的审批报告

结束前完成三件事。缺一视为未完成。

#### 3.1 写入正式审计报告

路径：

`03审计报告/20260821_Phase2-MPP2-PCC瓶颈/CA2代码审计报告_20260821.md`

正文必须包含 `.agents/skills/pfmval-audit/assets/audit-report-template.md` 的全部栏目，并额外包含：

- locus：branch、HEAD、dirty、start-check 结果
- 六个 CA2 项的双栏表（硬 Bug | 协议）
- 每项的新鲜证据路径与哈希/指标（能引用 accepted result_id 就引用）
- 总体判定
- 若需修复：精确文件与缺陷，不含补丁实现
- 给总管的下一允许动作：`等待用户审核` / `授权另开修复对话` / `允许开启划分与zscore诊断对话`

同时按 `03审计报告/20260821_Phase2-MPP2-PCC瓶颈/回传模板_审批报告.md` 写：

`03审计报告/20260821_Phase2-MPP2-PCC瓶颈/审批回传_CA2代码审计_20260821.md`

#### 3.2 聊天里给用户的短回传

用中文，先给结论，再给六项表，最后列出需要用户勾选批准的项。不要把长日志当正文。

#### 3.3 明确禁止越权

报告写完即停。不要修代码，不要开划分诊断，不要起草部署方案，不要 commit，不要 push。

若用户问“那 PCC 为什么是 0.65”，回答：本对话只做 CA2；科学解释归总管在硬 Bug 清零后另开对话。
