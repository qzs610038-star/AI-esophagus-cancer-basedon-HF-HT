# W007-mpp2_cpgcr_probe 实施方案

> 状态：`approved_pending_workspace_binding`（用户已批准，待 W007 绑定）  
> 创建日期：2026-08-11  
> 批准指令：`DIR-20260811-002`  
> 当前授权：仅允许本地 experiment 登记、W007 创建/绑定和修改前配置；不得填写实验代码、推送 Gitee、操作服务器或训练。

## 1. 绑定对象

| 项 | 值 |
|---|---|
| 来源部署方案 | `01_指南与解读/部署方案/MPP2连续通路几何软对比残差_初步探索部署方案_20260811.md` |
| 来源 revision | `v0.2`，用户于 2026-08-11 批准 |
| 拟登记 experiment | `mpp2_cpgcr_probe_v001_20260811` |
| 拟绑定工作树 | `W007`（永久编号，不复用 W005/W006） |
| 工作树短名 | `mpp2_cpgcr_probe` |
| 首批实验 | FBR、RCC、HCR、CPGCR，seed=42 |
| 外部评估 | 每臂内部 checkpoint 冻结后评估一次 XZY；XZY 不参与选择或调参 |
| 传输 | Gitee-only；禁止 SSH/SCP/HTTP remote command/Tunnel |

## 2. 修改前配置清单

- [ ] main 的 branch、HEAD、dirty 状态及未跟踪文件清单。
- [ ] `project_state/workspace_registry.json` 中 `next_workspace_number=7`，且 W007 未占用。
- [ ] `experiments/experiment_registry.json` 中拟用 experiment id 未占用。
- [ ] accepted B0 checkpoint、active repaired manifest、30 通路顺序、train-only z-score 与标准 split 均可读且只读。
- [ ] `configs/server_paths.yaml` 中服务器解释器、W### 工作树根、run/return/diagnostic 根和 XZY 路径仍有效。
- [ ] 治理 CLI 可运行严格 start-check；当前本机 Python 缺少 PyYAML，修复前不得登记 W007。
- [ ] 来源部署方案已纳入受跟踪文档与 active Registry；若该流程使用 SHA-256，须先取得用户对既有哈希治理的明确批准。

## 3. 代码边界

实施方案获批并绑定 W007 后，只在以下目录新增实验代码：

```text
experiments/workspaces/W007/mpp2_cpgcr_probe/
├─ configs/
│  ├─ protocol_v001.yaml
│  └─ server_v001.yaml
├─ mpp2_cpgcr/
│  ├─ model.py
│  ├─ losses.py
│  ├─ batch_plan.py
│  ├─ train_eval.py
│  ├─ result_contract.py
│  └─ one_click_runner.py
├─ tests/
│  ├─ test_model_contract.py
│  ├─ test_contrastive_contract.py
│  ├─ test_external_boundary.py
│  └─ test_one_click_runner.py
├─ W007_一键测试.cmd
└─ W007_一键训练.cmd
```

允许只读复用：`train_mpp_uni2h_mlp.py`、`dataset_mpp_manifest.py`、现有 result bundle 与 Gitee roundtrip 工具。首轮不修改公共训练代码；确需公共修改时先追加实施方案影响面并等待用户批准。

禁止修改：历史 `histogene/`、`egnv1/`、`egnv2/`，W005/W006，受保护 ssGSEA、z-score、manifest、split、embargo、accepted checkpoint 与缓存。

## 4. 边界行为

1. B0 参数 `requires_grad=False`，不进入优化器，永久 `eval()`，前向使用 `torch.no_grad()`。
2. RCC/HCR/CPGCR 共用图像分支初始化、batch plan、训练预算和 checkpoint 规则；HCR/CPGCR 共用通路编码器初始化与完整有效候选集合。
3. 训练、早停和 checkpoint 选择只能读取 train/internal-val；`external_benchmark` 只能读取冻结 checkpoint，不能写模型状态或训练配置。
4. 四臂配置在第一次 XZY 推理前冻结。逐臂 XZY 结果不得增删、改序、调参、重选 checkpoint 或触发重跑。
5. 性能不足只记 WARN；数据、身份映射、防泄漏、审批、source_commit、job、传输和结果合同失败为 HARD，并立即停止。
6. 每个成功臂回传 internal-val 与 XZY 逐样本预测、训练历史、指标和诊断；失败臂回传失败阶段原始日志。

## 5. 一键工具

### One-Click Test Controller（单击测试控制器，OCTC-1）

`W007_一键测试.cmd` 只调用固定 Python 入口，不接收任意 shell 参数。执行：身份核验 → 严格预检 → 配置/路径/资产只读检查 → 最小单元测试 → 非证据 dry-run → 诊断包回传。任一步非零退出即停止。

### One-Click Training Controller（单击训练控制器，OCTC-2）

`W007_一键训练.cmd` 先验证批量批准文件、全部 job_id、attempt 预算、source_commit 与协议 revision；缺一项即在 GPU 工作前停止。通过后按 FBR→RCC→HCR→CPGCR 顺序执行，每臂完成内部冻结后立即运行 XZY 推理并回传；不自动重试。

### Fail-Fast and Log Return（故障即停与日志回传，FLR-3）

Python 控制器使用固定阶段状态机。每阶段分别保存原始 `stdout.log`、`stderr.log`、`events.jsonl`、`resolved_config.json`。发生故障时：

1. 立即停止当前及后续阶段；
2. 写入 `failure.json`，包含 stage、exit_code、job_id、source_commit 与日志相对路径；
3. 调用现有 Gitee return 工具回传原始日志和失败清单；
4. 若 Gitee 本身故障，不重跑训练，保留服务器绝对路径并输出“仅回传重试”入口。

可行性：高。工作量为中等，核心是一个薄 Python 状态机、两个双击入口及四个最小测试文件；不新增服务器守护进程，也不建立绕过 Gitee 的远程控制通道。

## 6. 分步实施

### Workspace and Experiment Binding（工作树与实验绑定，WEB-1）

- 严格预检通过后登记 experiment 与 W007；创建本地持久工作树和实验目录。
- 记录 branch、HEAD、绝对路径、Registry 绑定和修改前配置。

### Model and Loss Construction（模型与损失构建，MLC-2）

- 实现冻结 B0、零初始化残差、RCC/HCR/CPGCR、患者均衡 batch、`tau_y` 与双向软目标。

### Evaluation Boundary Construction（评估边界构建，EBC-3）

- 隔离 `internal_only` 与 `external_benchmark`；实现四臂 internal-val/XZY 原始预测合同和内部—外部归因表。

### One-Click Controller Construction（单击控制器构建，OCCC-4）

- 实现两个 `.cmd` 与一个 Python 状态机；命令、配置和阶段均固定，不接受任意 shell。

### Failure and Return Construction（失败与回传构建，FRC-5）

- 实现非零即停、原始日志保留、失败包和 Gitee 回传；回传失败只允许重试 return，不重跑实验。

### Minimal Verification Test（最小验证测试，MVT-6）

- 完成 §7 的最小测试；不运行真实训练数据。

### Gitee Server Diagnostic（Gitee服务器诊断，GSD-7）

- 冻结 source_commit，经 Gitee 执行 allowlisted dry-run、路径/解释器/资产/输出合同检查；服务器修改只返回诊断或 patch，由本地单写修复。

### Batch Job Envelope（批量作业封装，BJE-8）

- 生成四个 job_id、顺序、三次训练 attempt、FBR 复放、四臂 XZY 推理及统一 return 合同；交用户另行批准正式批次。

## 7. 最小代码测试

统一入口：

```powershell
python -m pytest experiments/workspaces/W007/mpp2_cpgcr_probe/tests -q
```

必须覆盖：

1. 零残差输出等于 accepted B0；B0 在复合模型 `train()` 下仍确定、无梯度且不在优化器。
2. HCR/CPGCR 候选集合一致；双向软目标分别逐行归一化；`tau_y` 只用训练折。
3. `internal_only` 无法读取 XZY；外部模式不能改变 checkpoint、权重或配置。
4. 模拟任一阶段失败时后续阶段不执行，原始 stdout/stderr 原样进入失败包。
5. 无批准文件时一键训练在 GPU 工作前失败；有完整 fixture 时按固定四臂顺序运行。
6. 成功与失败回传清单均含必需原始预测或原始故障日志。

## 8. 服务器与结果过程

1. 本地完成全部代码与测试后冻结 source_commit。
2. 仅经 Gitee 同步到服务器 `D:\AIPatho\qzs\pfmval_automation\W007`。
3. 用户在服务器双击 `W007_一键测试.cmd`；故障自动停止并回传日志。
4. 诊断通过后生成批量批准文件，等待用户单独批准正式 job/attempt。
5. 用户在服务器双击 `W007_一键训练.cmd`；按固定四臂顺序运行，任一工程 HARD 故障停止后续臂。
6. 每臂结果分别回传；本地 import 验证后，四臂统一分析，不根据中途 XZY 改实验。

## 9. 证据路径

- 实施方案：`project_state/implementation_plans/W007-mpp2_cpgcr_probe_实施方案.md`
- 代码与测试：`experiments/workspaces/W007/mpp2_cpgcr_probe/`
- 服务器源码：`D:\AIPatho\qzs\pfmval_automation\W007`
- 服务器 attempt：`D:\AIPatho\qzs\pfmval_experiment_runs\W007\<job_id>\<attempt_id>`
- 服务器 return：`D:\AIPatho\qzs\pfmval_result_returns\W007\<job_id>\<attempt_id>`
- 服务器 diagnostic：`D:\AIPatho\qzs\pfmval_diagnostics\W007\<diagnostic_id>`

## 10. 进度

| 项目 | 状态 | commit | 证据 |
|---|---|---|---|
| WEB-1 工作树与实验绑定 | `pending_user_review` | — | 等待本实施方案二次批准 |
| MLC-2 模型与损失 | `pending_user_review` | — | — |
| EBC-3 评估边界 | `pending_user_review` | — | — |
| OCCC-4 单击控制器 | `pending_user_review` | — | — |
| FRC-5 失败与回传 | `pending_user_review` | — | — |
| MVT-6 最小测试 | `pending_user_review` | — | — |
| GSD-7 服务器诊断 | `pending_user_review` | — | — |
| BJE-8 批量作业封装 | `pending_user_review` | — | — |

## 11. 二次批准门

用户批准本文件后，才允许执行 WEB-1。正式服务器训练仍需后续绑定全部 job_id、attempt、source_commit 和预算的独立批量批准文件。
