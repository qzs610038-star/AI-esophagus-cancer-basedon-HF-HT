# PFMval 本地状态与 Gitee 服务器维护机制

> lifecycle: active
> state scope: `server_maintenance`
> effective date: 2026-07-11

## 当前边界

- 服务器只通过配置好的 Gitee Git remote 接收代码/作业清单并回传状态、错误和小型结果。
- SSH、SCP、HTTP 远程命令、Remote Tunnel 等旧方案只保留为历史排障资料。
- 服务器不直接修改 Registry、Dashboard、`CURRENT_STATE.md` 或用户指令记录。
- 本地验证服务器回传包后，才更新 accepted 结果和当前状态。
- 作业清单绑定已提交的 `source_commit`；服务器在 path id `server_automation_worktrees` 下建立只读语义的干净 detached worktree，复用前同时核对 HEAD 和工作区清洁度。
- MPP 作业的输入、缓存、标准划分和输出根只能由已登记 path id 注入，作业参数不得覆盖路径；标准划分固定读取 pinned worktree 内的受 Git 跟踪资产。
- 作业参数 key 必须原样映射为 Python `--<key>`，不得自动在下划线与连字符之间转换；历史拼写兼容只能在训练脚本的 `argparse` 中显式声明别名并由回归测试覆盖。
- 启动器清理 stdout/stderr 事件时必须使用显式、非空的 SourceIdentifier；清理异常不得覆盖 Python 原始 stderr 或退出码。
- 旧作业 `mpp2-repair-v003-frozen-20260711` 在 Python 参数解析阶段失败、未开始训练，不得打包结果或复用；修复后必须绑定新提交并生成新 job_id。

## 状态链

`directives.jsonl` → `current_state.json` → `CURRENT_STATE.md`
服务器运行目录 → Gitee result envelope → 本地 inbox → `result import` → Registry → Dashboard/Current State

## 诊断与探索的分级边界

- `agent start-check --task diagnostic` 仅用于 allowlisted 的非证据性故障排查。它仍要求状态包可读、schema 有效、无未完成事务/锁以及 Gitee-only transport；文档新鲜度与无关 MPP 路径问题仅 WARN，不得阻断只读诊断。
- 诊断请求必须由 `diagnostic request` 生成，并由 `diagnostic record` 记录回传输出的路径、大小和 SHA-256；请求固定 source commit、source/return branch 和 command id，不含任意 shell 文本、训练参数、experiment id 或 result import。诊断回传分支固定在 `automation/diagnostics*` 下，不与 smoke/formal 结果流混用。
- 本地 `explore` 仅限 `scripts/explorations/` 与 `experiments/explorations/`。它不得访问服务器、训练数据或输出可比较指标；候选清理只报告，不执行删除。探索脚本升级只生成候选，仍须 directive、提交、experiment 登记与独立 smoke/formal 调度才能成为正式证据链的一部分。
- smoke/formal 的固定 source commit、干净 detached worktree、path/data manifest、Gitee 回传和结果导入规则不因 diagnostic/explore 放宽而改变。

## 指令生命周期

- 指令可从 `active` 显式追加转换为 `completed`、`superseded` 或 `cancelled`。完成不同于被替代，所有历史事件保留在 `directives.jsonl`。
- `state directives --check-stale` 只按 `review_after`、年龄阈值和关联实验终态输出人工复核候选，绝不自动归档或关闭。
- 新指令可携带 `related_experiment_ids`、`review_after` 和 `completion_evidence`；转换必须提供理由，替代还必须指向新的 directive ID。

## 路径与产物

- 服务器绝对路径通过 `configs/server_paths.yaml` 的稳定 path id 引用，不直接删除。
- MPP 原始 ssGSEA、标准划分、train-only z-score 参数、manifest、group 3/5 embargo 审计为受保护资产。
- 当前路径索引确认五组 train 标签均有冲突重复 barcode；新 MPP 训练在独立数据修复任务完成前硬阻止，禁止调度器自动去重或重建。
- 权重和大型特征留在服务器；Gitee 只回传路径、大小、SHA-256 与限额内的小型指标/日志。
- 回传结果必须匹配原始 `automation/jobs/<job_id>/job.json` 的实验、提交、阶段、数据版本和正式批准；失败打包不留半包，导入中断由事务备份恢复。

## 后续自动排障

自动轮询与 CLI 修复闭环仍是独立后续任务。首版用 Codex CLI 验证，适配器接口保留给 Claude Code 和其他已确认支持无交互模式的 CLI。正式训练始终需要显式用户批准。
