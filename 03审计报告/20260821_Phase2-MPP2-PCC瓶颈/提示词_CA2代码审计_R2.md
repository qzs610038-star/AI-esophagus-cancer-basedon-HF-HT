# 提示词：重开 CA2 代码审计（门禁已通过）

> 用法：新开规划/只读对话，从「复制起点」复制到文末。  
> 上一轮 CA2 因 start-check FAIL=2 停止，六项未出具。治理修复已由用户批准完成（revision 243，FAIL=0）。本轮必须真正核验六项。

---

## 复制起点

你是 PFMval 仓库里一次**只读 CA2 代码审计**对话。总管对话确认治理门禁已通，现重开查找。你负责六项核验和审计报告，不负责修代码、不开新实验、不汇总后续路线。

### 0. 与上一轮的关系（必读）

上一轮报告 `03审计报告/20260821_Phase2-MPP2-PCC瓶颈/CA2代码审计报告_20260821.md` 只是 **preflight 阻塞记录**，总体 `NOT APPLICABLE`，六项全是 `missing`。那份稿**不是** CA2 结论，不得复述为 PASS/FAIL，也不得据此停止怀疑硬 Bug。

治理修复摘要（已提交、可引用）：`01_指南与解读/分析报告/20260821_Phase2-MPP2审计未能正式推进_阻塞汇总.md`。用户已批准修复。修复对话回报：

- `start-check --strict`：`PASS=13 WARN=1 FAIL=0`
- 仅剩已知 barcode WARN，未重生成标签
- `state_revision=243`（`current_state` 与 `document_registry` 顶层一致）
- HEAD：`a5ce3bf`；`source_commit` 应对齐该 HEAD
- 未 commit 修复工作区文件以外的实验代码；未训练；未把 `03审计报告/` 写入登记

本轮开始时必须**自己再跑一遍** start-check，用新鲜输出，不要抄修复对话的数字。若本次又出现 FAIL：停止写六项结论，把新 FAIL 交回总管，不要自行 `state sync` / prune / `--with-docs`。

超参数审查报告可以当对照，但**不能替代** CA2 六项源码与产物核验。

### 1. 本次任务的目的

Phase 2 / MPP2 通路预测的外部 XZY pooled PCC 长期约 0.65。本次只回答：

> 在 MPP2 修复基线及其后 Huber / Ridge / W007 实现中，CA2 规定的六类一般性硬 Bug 是否存在。若存在，列出必须先修的代码点；若不存在，给出“停止继续怀疑一般性硬 Bug”的审计依据。

不解释 PCC 0.65 的科学上限，不重做划分，不重做 z-score，不评价超参数是否还能再调。

用户已确认范围：

| 角色 | experiment_id | 备注 |
|---|---|---|
| 修复冻结基线 / FBR 复放锚点 | `mpp2_barcode_repair_v003_frozen_baseline_20260711` | accepted |
| Huber 配对 | `mpp2_huber_loss_paired_v001_20260728` | W001；accepted |
| Ridge 校准 | `mpp2_pathway_ridge_calibration_v001_20260717` | `failed` / `rejected`；仍须审 r002/r003 冻结前是否读 XZY |
| W007 四臂 | `mpp2_cpgcr_probe_v001_20260811` | FBR/RCC/HCR/CPGCR；accepted |
| LoRA | 不在本次 CA2 主审 | 只存在超参数附表 |

数据清单：`barcode-repair-20260711-d626ad8-v003:1204018178a4d355`。

### 2. 操作规范与边界

#### 2.1 必读顺序

1. `AGENTS.md`
2. `.agents/skills/pfmval-audit/SKILL.md` 及其 `references/evidence-boundaries.md`、`assets/audit-report-template.md`
3. `project_state/current_state.json` 相关字段（确认 revision 243 与 source_commit）
4. `experiments/experiment_registry.json` 中上表四条（只读相关条目）
5. `project_state/document_registry.json` 中与本任务相关且 `lifecycle=active` 的记录
6. 涉及路径时只读 `configs/server_paths.yaml` 相关项

```powershell
git status --short --branch
git rev-parse HEAD
python deploy/pfmval_ops.py agent start-check --strict
```

预期：FAIL=0，允许 barcode WARN。FAIL≠0 则停止。通过后必须打开源码与 accepted 产物做六项，禁止再次因“谨慎”停在门禁。

#### 2.2 工作树与写入

- 未绑定 `W###`。只读定位。
- 禁止写实验代码、训练、派发、import/accept、改 Registry、重生成 ssGSEA/划分/z-score/manifest。
- 禁止 `git worktree prune`、`state sync`、`--with-docs`、commit、push。
- 禁止提交或 `git add` `03审计报告/`。
- 唯一允许的写入：本轮**新文件**落到  
  `03审计报告/20260821_Phase2-MPP2-PCC瓶颈/`  
  **不要覆盖**上一轮阻塞稿 `CA2代码审计报告_20260821.md` 与 `审批回传_CA2代码审计_20260821.md`。

本轮文件名：

- `CA2代码审计报告_20260821_R2.md`
- `审批回传_CA2代码审计_20260821_R2.md`  
  按同目录 `回传模板_审批报告.md` 填写。

#### 2.3 证据纪律

从当前仓库源码和已接受产物核验。不要把总管聊天、上一轮 `NOT APPLICABLE` 稿、W007 待审归因写成结论。W007 分析三份若读到，最多标 `historical` / `pending_review`。证据分类：`current` / `accepted` / `pending_review` / `diagnostic_only` / `explore_only` / `historical` / `missing`。

#### 2.4 CA2 只审这六项（双栏）

禁止把协议选择写成硬 Bug。

| 项 | 硬 Bug（FAIL 则必须先修代码） | 协议/科学设计（可 WARN，不是硬 Bug） |
|---|---|---|
| CA2-1 图像、通路、标签、患者 ID 逐行对应 | 坐标/barcode/ID 错配；静默行序 fallback 造成错配；`.pt` 与标签行、患者 ID 不一致仍训练 | 匹配策略偏保守但无错配 |
| CA2-2 z-score 只在训练折拟合 | val 或 XZY 参与 mean/std；跨折串用参数；inverse-z 用错参数 | train-only 正确，但该标准化与 raw 指标不完全同目标 |
| CA2-3 同一患者全部切片进入同一集合 | 违反**当时登记协议**的泄漏 | 现行协议本就是 6 患者内空间 10% val；同一患者身份同时在 train 与 val 是协议发现，不是硬 Bug |
| CA2-4 baseline checkpoint 精确复放 | 无法复放登记 val_loss/PCC；FBR 声称复放但对不上 | 复放通过，选模指标可能偏乐观 |
| CA2-5 新参数进入优化器并真实改变预测 | 新头/损失未进计算图；Huber/RCC/HCR/CPGCR 预测与对照几乎相同却声称已训练 | 有可测差异但很小，可能被训练设置压住（不记硬 Bug） |
| CA2-6 external 未参与选模、阈值、超参 | 用 XZY 早停、选 checkpoint、选 λ、调损失权重或阈值 | 合同未读 XZY，但 internal-val 与 train 共享患者身份（协议问题，不是本项硬 Bug） |

Ridge 虽已 rejected，仍必须核查 r002/r003 冻结校准器之前有没有读 XZY。

建议核验面（线索，不是结论）：`dataset_mpp.py`；`scripts/rebuild_zscore_from_manifest.py` 与 group_2 z-score 清单；`scripts/generate_standard_splits.py` 与 split_info；`train_mpp_uni2h_mlp.py`；W001 Huber；W007 `experiments/workspaces/W007/mpp2_cpgcr_probe/`；FBR `baseline_replay_check.json` 与 accepted bundle；各臂预测差。只读受保护资产，禁止重建。

#### 2.5 判定出口

- 六项对硬 Bug 全 PASS（允许协议栏 WARN）：对“停止怀疑一般性硬 Bug”给 `GO` 或点名 WARN 的 `CONDITIONAL GO`
- 任一项硬 Bug FAIL：总体 `NO-GO`，列出最小修复面，**不要动手修**
- 证据不足：该项 `FAIL` 或 `missing`，不要写成 PASS
- 总体判定只覆盖“可否停止怀疑一般性硬 Bug”，不对划分改进或新训练给 GO

### 3. 必须回传的审批报告

结束前三件事，缺一视为未完成。

1. 写入 `CA2代码审计报告_20260821_R2.md`：审计模板全部栏目 + locus + start-check 新鲜输出 + 六项双栏表 + 每项证据路径（能引用 result_id / sha256 就引用）+ 总体判定 + 若需修复则精确到文件但不含补丁。
2. 写入 `审批回传_CA2代码审计_20260821_R2.md`。
3. 聊天短回传：先结论，再六项表，再请用户勾选。下一允许动作只选一个：等待用户审核 / 授权另开修复代码对话 / 停止怀疑一般性硬 Bug（允许总管再开划分与 z-score 诊断）。

写完即停。不要修代码，不要开划分诊断，不要起草部署方案，不要 commit。
