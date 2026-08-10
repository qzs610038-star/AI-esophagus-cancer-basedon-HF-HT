# PFMval 本地状态与 Gitee 服务器维护机制

> lifecycle: active
> state scope: `server_maintenance`
> effective date: 2026-07-11

## 当前边界

- 服务器只通过配置好的 Gitee Git remote 接收代码/作业清单并回传状态、错误和小型结果。
- SSH、SCP、HTTP 远程命令、Remote Tunnel 等旧方案只保留为历史排障资料。
- 服务器不直接修改 Registry、Dashboard、`CURRENT_STATE.md` 或用户指令记录。
- 本地验证服务器回传包后，才更新 accepted 结果和当前状态。
- 作业清单绑定已提交的 `source_commit`；服务器在 path id `server_experiment_worktrees` 下按 `W###` 建立一个实验一个持久工作树。job/attempt 不再创建新源码工作树；复用前核对 workspace、branch、HEAD 和工作区清洁度。
- MPP 作业的输入、缓存、标准划分和输出根只能由已登记 path id 注入，作业参数不得覆盖路径；标准划分固定读取 pinned worktree 内的受 Git 跟踪资产。
- 作业参数 key 必须原样映射为 Python `--<key>`，不得自动在下划线与连字符之间转换；历史拼写兼容只能在训练脚本的 `argparse` 中显式声明别名并由回归测试覆盖。
- 启动器清理 stdout/stderr 事件时必须使用显式、非空的 SourceIdentifier；清理异常不得覆盖 Python 原始 stderr 或退出码。
- 旧作业 `mpp2-repair-v003-frozen-20260711` 在 Python 参数解析阶段失败、未开始训练，不得打包结果或复用；修复后必须绑定新提交并生成新 job_id。

## 状态链

`directives.jsonl` → `current_state.json` → `CURRENT_STATE.md`
服务器运行目录 → Gitee result envelope → 本地 inbox → `result import` → Registry → Dashboard/Current State

## 诊断与探索的分级边界

- `agent start-check --task diagnostic` 仅用于 allowlisted 的非证据性故障排查。它仍要求状态包可读、schema 有效、无未完成事务/锁以及 Gitee-only transport；文档新鲜度与无关 MPP 路径问题仅 WARN，不得阻断只读诊断。
- 诊断请求必须由 `diagnostic request` 生成，并由 `diagnostic record` 记录回传输出的路径、大小、UTF-8/LF 文本规范和 Git tree 闭包；诊断文件不逐文件计算 SHA-256。请求固定 source commit、source/return branch 和 command id，不含任意 shell 文本、训练参数、experiment id 或 result import。诊断回传分支固定在 `automation/diagnostics*` 下，不与 smoke/formal 结果流混用。
- 本地 `explore` 仅限 `scripts/explorations/` 与 `experiments/explorations/`。它不得访问服务器、训练数据或输出可比较指标；候选清理只报告，不执行删除。探索脚本升级只生成候选，仍须 directive、提交、experiment 登记与独立 smoke/formal 调度才能成为正式证据链的一部分。
- smoke/formal 的固定 source commit、干净 detached worktree、path/data manifest、Gitee 回传和结果导入规则不因 diagnostic/explore 放宽而改变。

## 指令生命周期

- 指令可从 `active` 显式追加转换为 `completed`、`superseded` 或 `cancelled`。完成不同于被替代，所有历史事件保留在 `directives.jsonl`。
- `state directives --check-stale` 只按 `review_after`、年龄阈值和关联实验终态输出人工复核候选，绝不自动归档或关闭。
- 新指令可携带 `related_experiment_ids`、`review_after` 和 `completion_evidence`；转换必须提供理由，替代还必须指向新的 directive ID。

## 路径与产物

- 服务器绝对路径通过 `configs/server_paths.yaml` 的稳定 path id 引用，不直接删除。
- `server_governance_checkout` 是治理与派发 checkout；`server_experiment_worktrees` 是持久 `W###` 源码根；`server_experiment_runs` 是外置 attempt 输出根；`server_result_returns` 是回传包暂存根；`server_diagnostics` 是非证据排障根；`server_runtime_bundles` 是不可变运行包根。
- `server_config` 与 `server_governance_config` 仅是 retired_server_config_pointer：保留作兼容核对或 fail-closed 历史指针，零训练试点与后续 agent 均不得把它们当作机器配置事实源；唯一机器配置事实源是治理仓库中的 `configs/server_paths.yaml`。
- `server_repo_worktree`（`pfmval_deploy_git`）及其中旧 config/standard-split path id 已降为 legacy，不得写入新训练输出、回传包或诊断输出。仍位于其中的两类特征 cache 与冻结 baseline checkpoint 仅保留只读输入资格；兼容 `server_mpp_results` 已重定向到外置 `server_experiment_runs`。旧 path id `server_automation_worktrees` 仅作弃用兼容，不得用于新实现。
- 新路径首次在服务器使用前必须做存在性、目录类型和边界只读核验；`server_diagnostics` 当前尤其需要现场确认或创建批准，路径登记本身不代表已执行服务器变更。
- MPP 原始 ssGSEA、标准划分、train-only z-score 参数、manifest、group 3/5 embargo 审计为受保护资产。
- 当前路径索引确认五组 train 标签均有冲突重复 barcode；新 MPP 训练在独立数据修复任务完成前硬阻止，禁止调度器自动去重或重建。
- 特征缓存、embedding/中间特征、checkpoint/权重等大型资产可留在服务器，但必须登记服务器路径、大小、SHA-256 及保留或复算策略。
- 对会产生预测结果的训练或校准作业，每个实际执行的评估 split 都必须通过 Gitee 结果包回传逐样本原始预测表；至少绑定稳定样本标识、空间分组、pathway、`y_true`（真实值）和 `y_pred`（预测值）。原始预测表属于 critical artifact，不得以“文件较大”为由降级到 `large_artifacts` 仅登记服务器路径。
- 原始预测回传清单必须显式列出 `split_id`（评估数据划分标识）与对应 `artifact_id`（结果包内文件标识）；缺失任一已评估 split、文件、SHA-256 或 split 绑定时，结果打包/验证直接 FAIL，不得进入 import/accepted。
- stdout/stderr、环境探测等诊断日志只在 Agent 自行分析或排障确有需要时回传；它们保持 diagnostic、非证据，不得替代原始预测表，也不得阻断原始预测表的回传。
- 每个训练 attempt 的回传包必须包含原始训练记录：至少一个机器可读 CSV 和一个原始 TXT 日志。CSV/TXT 必须登记 artifact id、相对路径、类型、证据角色、大小和来源 attempt；若它们承载 accepted 指标或逐样本预测则为 critical 并做 SHA-256，否则作为 supporting 只做精确清单、大小与 Git tree 闭包校验。
- 回传结果必须匹配原始 `automation/jobs/<job_id>/job.json` 的实验、提交、阶段、数据版本和正式批准；失败打包不留半包，导入中断由事务备份恢复。

## HT1 哈希分层（Hash Tiering 1）

- critical：输入合同、选择证明、accepted 指标来源、逐样本原始预测、受保护 manifest 使用 SHA-256 硬校验。
- supporting：普通训练 CSV/TXT、摘要和辅助表使用必传清单、精确字节数、Git tree 闭包；只有被提升为指标来源时才升级为 critical。
- diagnostic：环境探测、stdout/stderr 与排障记录为非证据，使用 allowlist、路径边界、大小、UTF-8/LF 和 Git tree 闭包，不再逐文件计算 SHA-256。
- bundle 身份只覆盖 critical 内容及规范化 manifest；不得因为 supporting/diagnostic 文本换行变化而阻塞回传。任何角色升级必须重新打包和校验，不能绕过 critical 哈希。

## RT2 快速排障与回传（Rapid Troubleshooting and Return 2）

- 服务器同步统一使用 `git fetch` 后校验目标 ref/commit；禁止在治理 checkout 或实验工作树执行隐式 `git pull`/merge。
- 诊断只使用固定 command id，从治理 checkout 创建短期 detached source worktree，输出落入 `server_diagnostics/<diagnostic_id>`，经 Gitee `automation/diagnostics/*` 回传；不得创建 experiment/result envelope。
- 结果先在 `server_result_returns/<result_id>` 构建并本地闭包验证，再对该精确目录执行 force-add，确保被 `.gitignore` 命中的必传 CSV/TXT 仍进入提交；提交前必须用 Git index 列表与回传 manifest 对账。
- 成功、失败和 incomplete attempt 都要回传终态 JSON、原始训练 CSV/TXT；成功且产生预测的 attempt 还必须回传每个 evaluated split 的原始预测表。

## 后续自动排障

自动轮询与 CLI 修复闭环仍是独立后续任务。首版用 Codex CLI 验证，适配器接口保留给 Claude Code 和其他已确认支持无交互模式的 CLI。正式训练始终需要显式用户批准。

## 新实验部署方案与实施方案双阶段门

本节是新实验从讨论进入工作树代码编辑前的执行顺序，适用于本地开发、服务器诊断准备和正式实验，不改变既有 Gitee-only、W###、批准、保护资产和结果导入门禁。

1. **部署讨论阶段**：新实验部署方案讨论稿只能放在 `01_指南与解读/部署方案/`。`project_state/plans/` 及其他历史 Agent 自建 plan 目录仅保留既有治理/历史方案，不承接新的实验部署讨论稿。
2. **部署方案批准门**：只有用户明确批准最终部署方案，并确认准备进入实验绑定，才可以进入下一步；批准前不得创建实验工作树或实验代码目录。
3. **实施方案生成**：依据最终批准的部署方案创建 `project_state/implementation_plans/W###-<工作树名>_实施方案.md`，文件至少绑定部署方案路径与 revision、experiment、拟用 W###、代码目录、修改前配置清单、任务分解、验证方式、证据位置和进度状态。
4. **实施方案批准门**：实施方案生成后必须单独经过用户审核和明确批准。实施方案未获批时，不得绑定/创建 W### 工作树，不得创建本次改动代码目录，不得写入实验代码或执行配置。
5. **工作树准备阶段**：实施方案获批后，才按该文件绑定/创建 W###，先建立本次改动代码放置目录并固化修改前配置；随后等待用户在后续对话显式切换到对应工作树。
6. **代码编辑与进度回写**：用户切换到对应工作树后，Agent 只能依据已批准实施方案编辑代码。每完成一个条目，必须在同一实施方案中追加进度记录并标记 `completed_pending_user_review`，写明 commit、测试和文件证据，供用户审核；不得以聊天摘要替代文件更新。
