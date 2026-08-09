# PFMval 本地代码 Git 治理与 W### 目录化执行方案

> plan_id: `local_code_git_governance_v001_20260809`  
> lifecycle: `approved`  
> execution_status: `COMPLETED`
> authority: `user_approved_execution_plan`  
> directives: `DIR-20260809-002`, `DIR-20260809-003`  
> source_state_revision: `188`
> source_commit: `cfafb3930f62475aa31d27aaa4991ae6430a38d1`  
> cutoff: `2026-06-30 23:59:59`  
> boundary: 用户已批准全部执行决定；允许本地迁移、取消 Git 跟踪、依赖修复、最简测试和分批提交。仍禁止远端推送、服务器操作、训练、结果导入以及数据或训练资产修改。

## 0. 一页结论

本方案解决两个问题：

1. 把 MPP1-5 正式序列前、已经退出活跃使用的旧代码集中到本地只读目录 `historical_code/pre_mpp1_5/`，避免后续工作树继续继承。
2. 从下一个新实验开始，把实验专属代码、配置、测试和结构化日志统一放入 `experiments/workspaces/W###/`，方便整包审核、合入和总结。

本轮初筛得到：

| 项目 | 数量 | 当前结论 |
|---|---:|---|
| Git 已跟踪代码 | 225 | 全量治理母集 |
| 最后一次代码提交早于 2026-07-01 | 102 | 仅为时间候选，不代表均可迁移 |
| `histogene/` 旧代码 | 10 | 用户已批准纳入历史代码候选 |
| `egnv1/` 旧代码 | 8 | 用户已批准纳入历史代码候选 |
| `egnv2/` 旧代码 | 8 | 用户已批准纳入历史代码候选 |
| 其他早期根目录代码 | 46 | 需结合调用关系逐批审核 |
| `tools/` 早期工具代码 | 12 | 高概率历史候选，仍需逐项审核 |
| 旧 `deploy/` 与 `scripts/` | 16 | 存在旧服务器通道与旧入口，必须专项审核 |
| `uni2h/` 早期代码 | 2 | 待确认是否仍有复现价值 |

重要发现：三旧目录虽然已解除特殊保护，但仍被根目录训练脚本、测试、资产登记和旧部署脚本引用。因此不能先移动目录、以后再修调用方；必须在同一迁移批次完成调用方收口。

## 1. 已确认的用户规则

### 1.1 旧代码边界

- 创建本地目录 `historical_code/pre_mpp1_5/`。
- 仅接收 MPP1-5 正式序列前的旧代码。
- 2026-07-01 以后退役的代码不再搬入历史目录，只使用 Git 历史管理。
- `histogene/`、`egnv1/`、`egnv2/` 已按 `DIR-20260809-002` 解除特殊保护并进入迁移候选。
- 解除保护不等于立即移动或删除；实际动作仍需本方案和精确清单通过审核。

### 1.2 W### 目录规则

- 每个实验继续绑定永久、不复用的 W###。
- 实验新增或修改的代码必须集中在 `experiments/workspaces/W###/code/`。
- 配置、最简测试、结构化日志和 closeout 同样位于该 W### 目录。
- 未修改的公共代码不复制；由 `shared_code_manifest.json` 记录其路径和 Git commit ID。
- 原始大日志、checkpoint、预测、缓存和中间产物继续放在工作树外、带 W### 的注册目录。

## 2. 事实依据与约束

主要依据：

- [`AGENTS.md`](../../AGENTS.md)
- [`项目 Git 文件追踪与未追踪结构全览`](../../01_指南与解读/部署方案/项目Git文件追踪与未追踪结构全览_20260809.md)
- [`服务器治理分步执行方案 1`](server_workspace_sync_step1_execution_plan_20260808.md)
- [`服务器 W### 同步与排障治理草案`](server_workspace_sync_troubleshooting_governance_review_draft_v001_20260808.md)
- [`编号工作树协议`](gitee_numbered_workspace_protocol_v001_20260726.md)
- [`workflow governance v3`](workflow_governance_v3_20260726.md)

继续有效的边界：

- 禁止 `git clean -fd`。
- 不自动删除 checkpoint、缓存、训练结果、MPP 数据或未跟踪产物。
- 服务器日常同步只允许 Gitee。
- GitHub `origin` 仅用于用户逐次明确触发的备份。
- MPP 原始 ssGSEA、标准划分、z-score 参数和 manifest 的保护不受本次旧代码解禁影响。
- 本方案不启用新的文件内容哈希治理；如需补齐登记一致性，另行申请单独批准，仍只限少量已修改文本文件，不扩展到代码树、数据、checkpoint、缓存或历史目录。

## 3. 目标目录

### 3.1 MPP1-5 前历史代码

```text
historical_code/
└─ pre_mpp1_5/
   ├─ legacy_models/
   │  ├─ histogene/
   │  ├─ egnv1/
   │  └─ egnv2/
   ├─ legacy_entrypoints/
   ├─ legacy_tools/
   ├─ legacy_deploy/
   ├─ legacy_scripts/
   ├─ README.md
   └─ migration_manifest.json
```

规则：

- `historical_code/` 为本地只读目录，并由 Git 精确忽略。
- `migration_manifest.json` 只记录原路径、归档路径、归档原因、原 Git commit ID 和用户审核批次，不计算额外文件哈希。
- Git 中保留迁移前基线提交，因此即使本地历史目录损坏，仍可从 Git 历史恢复。
- `histogene/results_vis/`、旧预测 CSV、图片和分析文本属于非代码资料，不进入 `historical_code/`；继续由非代码治理处理。

### 3.2 新实验 W### 包

```text
experiments/
└─ workspaces/
   └─ W006/
      ├─ code/                         # 实验新增或修改代码
      ├─ configs/                      # 实验配置
      ├─ tests/                        # 最简必要测试
      ├─ logs/
      │  ├─ diagnostic_summary.md
      │  ├─ training_summary.md
      │  └─ log_index.json
      ├─ shared_code_manifest.json     # 未复制的公共代码清单
      ├─ README.md
      └─ CLOSEOUT.md
```

外部运行目录：

```text
pfmval_runs/W006/A001/
pfmval_returns/W006/A001/R001/
pfmval_diagnostics/W006/D001/
```

`shared_code_manifest.json` 中的每条公共代码记录至少包含：

```text
path
source_commit
purpose
modified_in_workspace
promotion_candidate
```

## 4. 初步筛选结果

### 4.1 已授权的直接迁移候选：26 个代码文件

| 目录 | 代码文件数 | Git 最后提交日期范围 | 当前风险 |
|---|---:|---|---|
| `histogene/` | 10 | 2026-04-15 至 2026-04-29 | `utils.py`、dataset/model/train 仍被旧入口和测试引用 |
| `egnv1/` | 8 | 2026-04-22 | 旧运行脚本仍直接调用 `egnv1/train.py` |
| `egnv2/` | 8 | 2026-04-22 | 多个根目录训练入口仍 import `egnv2` |

明确影响面：

- `train_cross_patient_histogene.py`、`train_cross_patient_histogene_uni.py`、`train_cross_patient_egnv2.py`。
- `train_egnv2_uni.py`、`egnv2_uni_dataset.py`、`egnv2_uni_model.py`。
- 多个 `train_histogene_*`、`train_online_*` 入口仍引用 `histogene.utils`。
- `tests/test_metrics_compatibility.py` 仍导入 `histogene.utils`。
- `scripts/pfmval_assets.py` 仍把 `histogene/utils.py` 登记为 shared dependency（共享依赖）。
- `scripts/pfmval_execution_roundtrip.py` 仍把三目录列为禁止进入执行 bundle 的前缀。
- 旧 `deploy/pack_deploy.sh`、`pull.sh`、`sync_list.sh` 仍引用三目录。

### 4.2 其他时间候选：76 个代码文件

这些文件最后一次代码提交早于 2026-07-01，但尚未获得整体迁移批准。

#### 根目录：46 个

```text
analyze_cross_validation.py
compare_datasets.py
compare_single_vs_multi.py
compare_token_encoder_runs.py
data_distribution_analysis.py
dataset_graph_tokens.py
dataset_online.py
dataset_uni_tokens_augmix.py
dataset_uni_tokens.py
egnv2_uni_dataset.py
egnv2_uni_model.py
ensemble_late_fusion.py
extract_omiclip_features.py
extract_per_pathway_pcc.py
extract_uni_features_3st.py
extract_uni_tokens_augmented.py
extract_uni_tokens.py
extract_virchow2_tokens.py
graph_utils_gat.py
lora_utils.py
model_gfnet.py
model_online_cls.py
model_online_tokens.py
model_uni_tokens_gat.py
model_uni_tokens.py
notify_utils.py
organize_existing_results.py
run_tv_sweep.sh
spatial_tv_loss.py
split.py
train_cross_patient_egnv2.py
train_cross_patient_histogene_uni.py
train_cross_patient_histogene.py
train_egnv2_uni.py
train_histogene_omiclip.py
train_histogene_uni_tokens_augmix.py
train_histogene_uni_tokens_gat.py
train_histogene_uni_tokens.py
train_histogene_virchow2_tokens.py
train_online_cls.py
train_online_tokens.py
visualize_all_models.py
visualize_cross_patient.py
visualize_model_comparison.py
visualize_results.py
zscore.py
```

#### `tools/`：12 个

```text
tools/FINAL_DIAGNOSIS_REPORT.py
tools/analyze_gat_fold1.py
tools/analyze_intersection.py
tools/analyze_stats.py
tools/check_csv_patient.py
tools/debug_dataset.py
tools/debug_dataset_loading.py
tools/diagnose_csv_mismatch.py
tools/final_report.py
tools/generate_clean_csv.py
tools/simulate_train_uni_loading.py
tools/verify_clean_csv.py
```

#### `deploy/`：9 个

```text
deploy/audit_server.sh
deploy/config.sh
deploy/launch_nightly.sh
deploy/pack_deploy.sh
deploy/pull.sh
deploy/push.sh
deploy/run.sh
deploy/setup_server.sh
deploy/sync_list.sh
```

#### `scripts/`：7 个

```text
scripts/explore_server_data.ps1
scripts/prepare_jfx_fixed_data.py
scripts/run_egnv1_LMZ12939.ps1
scripts/run_egnv1_multi_3st.ps1
scripts/run_train_hyz15040_fixed.ps1
scripts/spatial_smooth_post.py
scripts/zscore_labels.py
```

#### `uni2h/`：2 个

```text
uni2h/infer.py
uni2h/train.py
```

### 4.3 明确不进入本轮历史代码目录

- 2026-07-01 以后新增或修改的 MPP1-5、MPP2 LoRA、校准、Phase 3 和治理代码。
- `reference_implementations/` 中 2026-08-01 后纳入的 CLAM 等参考实现；它们不满足时间边界。
- `.agents/`、`.claude/`、`project_state/`、`tests/` 中仍承担当前治理职责的代码。
- 数据、checkpoint、缓存、结果图片、预测 CSV 和文献资料。

## 5. 执行步骤

### CG1 — 约束治理（Constraint Governance）

目标：解除三目录特殊保护，但不提前执行迁移。

1. 记录 `DIR-20260809-002`。
2. 仅更新本方案及其明确引用的治理入口说明，不扩大到其他状态文件。
3. 将三目录生命周期从 `protected` 迁移方向调整为 `legacy_archive_candidate`。
4. 旧审计回执和历史方案保持原文，不回写历史结论。

退出条件：当前规范不再宣称三目录绝对禁止修改；同时明确真正迁移仍待清单批准。

### BI2 — 基线清点（Baseline Inventory）

目标：形成不会混入上一对话非代码变更的代码治理基线。

1. 等待当前非代码治理变更完成独立提交。
2. 保存代码治理前本地 Git 基线提交，不自动推送。
3. 生成完整代码清单：路径、最后提交日期、调用方、测试、所属实验、拟处理类别。
4. 将 102 个时间候选分为：直接迁移、联动迁移、保留、待确认。

退出条件：每个候选都有唯一分类，工作区无未说明的代码变化。

### LA3 — 历史归档（Legacy Archive）

目标：迁移经用户批准的 MPP1-5 前旧代码。

1. 创建 `historical_code/pre_mpp1_5/`。
2. 先复制到目标目录并核对数量与相对路径；不做内容哈希。
3. 核对通过后，再从当前活跃树取消对应 Git 跟踪。
4. 添加精确 `.gitignore` 规则。
5. 生成 `migration_manifest.json` 和人类可读 `README.md`。
6. 每一批独立提交；禁止批量 `git clean` 或模糊删除。

退出条件：目标目录完整、原 Git 历史可恢复、主线不再跟踪已批准旧代码。

### DR4 — 依赖修复（Dependency Repair）

目标：旧目录退出后，当前主线仍可运行和验证。

1. 将仍需要的通用指标实现统一指向 `pfmval_core.metrics`。
2. 与旧模型绑定的根目录训练入口一起归档，或明确保留兼容入口；不得留下失效 import。
3. 更新 `scripts/pfmval_assets.py` 的旧资产生命周期和路径登记。
4. 更新 `scripts/pfmval_execution_roundtrip.py`、旧部署脚本及其测试。
5. 删除或改写仅用于证明旧保护边界的测试断言。
6. 只运行受影响的最简测试，不做全量防御性测试。

建议最小测试：

```powershell
& 'D:\conda_envs\pfmval_py310\python.exe' -m pytest `
  tests/test_metrics_compatibility.py `
  tests/test_asset_registry.py `
  tests/test_execution_roundtrip_v1.py `
  -q
```

退出条件：无指向已迁移路径的活跃 import；最简相关测试通过。

### WT5 — 工作树模板化（Workspace Templating）

目标：以后所有实验都具备统一 W### 文件夹。

1. 在下一个新 W### 做无训练试点，不改造正在推进的 W004/W005。
2. 创建 `experiments/workspaces/W###/` 标准目录。
3. 增加 `shared_code_manifest.json`、日志索引和 closeout 模板。
4. 限制实验新增/修改代码只能进入当前 W### 目录。
5. 原始日志和训练产物继续写入外部注册目录。

退出条件：一个本地无训练试点可独立导入、执行最小入口并生成日志索引。

### MI6 — 主线合入（Mainline Integration）

目标：既能整体回收 W###，又保持后续主线精简。

1. 实验结束后，先把完整 W### 目录作为一个独立审核提交合入 main。
2. 从 W### 中提取真正可复用代码，单独晋升到公共目录并补最简测试。
3. 把 `CLOSEOUT.md`、公共代码清单和晋升结果保留在 main 的治理索引中。
4. W### 完整目录可在后续整理提交中从 main 当前树移除；完整内容仍保留在 Git 合入历史中。
5. 新工作树只继承 main 当前树，不继承已从当前树退出的旧 W### 实验包。

退出条件：main 中公共代码可直接复用；完整实验包可从对应 Git 提交恢复。

### VR7 — 验证与回退（Verification and Rollback）

目标：证明治理没有误删活跃代码。

1. 检查 Git 状态、候选清单、迁移清单和残余 import。
2. 运行 DR4 中列出的最简测试。
3. 运行严格门禁；已有历史 WARN 单独记录，不冒充新失败。
4. 回退只使用本批次独立 Git 提交恢复，不使用 `git reset --hard` 或 `git clean -fd`。

退出条件：严格门禁不新增由本次迁移造成的 FAIL，且每个历史文件均可定位到原 Git 提交。

## 6. 分批审核建议

| 批次 | 内容 | 推荐 |
|---|---|---|
| A | 约束解除、清单与方案文件 | 本轮完成 |
| B | `histogene/`、`egnv1/`、`egnv2/` 的 26 个代码文件及直接调用方 | 优先审核 |
| C | 12 个 `tools/` 和 2 个 `uni2h/` 旧文件 | 第二批审核 |
| D | 46 个根目录旧代码 | 按模型/数据/训练/分析四组审核 |
| E | 9 个旧 deploy 和 7 个旧 scripts | 与服务器治理方案联动审核 |
| F | 新 W### 目录模板与无训练试点 | 历史代码迁移后实施 |

## 7. 风险与控制

| 风险 | 控制方式 |
|---|---|
| 移走 `histogene.utils` 导致旧入口或测试失效 | 与调用方同批归档或切换到 `pfmval_core.metrics` |
| 把时间早但仍活跃的代码误归档 | 日期只作候选，必须检查调用方、测试和 active 方案 |
| 本地历史目录未被远端备份 | 保留迁移前 Git 基线和路径清单；是否另行备份由用户决定 |
| W### 内复制公共代码导致双份维护 | 公共代码只写 manifest，不复制未修改版本 |
| W### 整包长期堆积在 main | 合入后晋升公共代码，完整包保留在 Git 历史，当前树只留 closeout 索引 |
| 原始日志污染 Git | Git 只保留摘要与索引，原始日志放外部 W### 目录 |

## 8. 验收清单

- [x] 用户审核并批准精确迁移批次。
- [x] 非代码治理先形成独立基线提交。
- [x] 102 个时间候选全部获得唯一分类：101 个迁移，`lora_utils.py` 保留。
- [x] 三旧目录不再被 active import 直接引用。
- [x] 旧代码目录只包含批准的 MPP1-5 前代码。
- [x] 没有数据、checkpoint、缓存、结果或文献混入历史代码目录。
- [x] W### 模板包含代码、配置、测试、日志索引和 closeout；模板不占用编号。
- [x] 模板明确原始日志和训练产物位于工作树外。
- [x] 当前共用指标实现保留在 `pfmval_core.metrics`，未复制进 W### 模板。
- [x] 最简相关测试通过。
- [x] 严格门禁未新增由本次治理造成的 FAIL。
- [x] 未执行任何未经明确批准的远端推送。

## 9. 当前门禁状态

执行前严格门禁曾出现以下登记漂移；用户随后批准仅刷新四个 active 小型文本文件的登记指纹，并已完成修复：

- PASS：6
- WARN：6，均为既有 legacy result envelope 或受保护 MPP 数据问题
- FAIL：6
  - `document_registry.json` 比当前状态落后超过一个 revision；
  - `.agents/skills/pfmval-audit/SKILL.md` 登记指纹过期；
  - `.agents/skills/pfmval-governance/SKILL.md` 登记指纹过期；
  - `AGENTS.md` 登记指纹过期；
  - `project_state/plans/workflow_governance_v3_20260726.md` 登记指纹过期；
  - 绝对工作区分支与 Registry 不一致。

前五项与本轮用户授权修改 active 文档及状态 revision 前进直接相关。若要恢复严格门禁，需要刷新上述四个小型文本文件在 `document_registry.json` 中的既有 SHA-256 内容指纹，并同步派生状态。

哈希治理申报：

- 原因：active 文档正文已按用户指令变化，登记表仍指向旧版本。
- 必要性：不刷新就无法证明当前执行规范与登记事实一致，严格门禁会持续 FAIL。
- 范围：仅四个已修改的小型 active 文本文件；不扫描代码树、数据、checkpoint、缓存或历史目录。
- 负面影响：会计算四个文件的 SHA-256，并修改 `document_registry.json` 及相关派生状态；若随后继续编辑这些文件，需要再次刷新。
- 当前状态：已按 `DIR-20260809-003` 执行；未对其他文件、代码树、数据或历史目录计算内容哈希。

迁移前与执行后严格门禁均为 PASS 7、WARN 7、FAIL 0。执行后 7 个 WARN
仍为五项旧结果 envelope 不完整、`PROJECT_GUIDE.md` 过期和 MPP 标准标签
重复/冲突，未新增本次治理导致的 FAIL。

## 10. 用户审核决定

用户于 2026-08-09 明确决定以上五项全部为“是”，登记为 `DIR-20260809-003`。
