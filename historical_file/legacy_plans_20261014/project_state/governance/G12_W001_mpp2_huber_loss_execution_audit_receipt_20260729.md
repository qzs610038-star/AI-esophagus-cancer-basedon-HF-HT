# G12 独立审核回执：MPP2 paired MSE vs Huber delta=1

审核时间：2026-07-29

## 唯一结论

**NO-GO（尚未启动 A001/A002；不消耗 run unit）**。

G12 的本地治理门禁、W001 定位、G11 已回传环境与两份 prepared job 均可复核；但固定的 experiment source commit `a04319a1d6aeafcd35d6adc7e2893f1d7b683bda` 不包含本实验的 registry/governance/job 记录。现有 `job run` 会在该 pinned source 的 `experiments/experiment_registry.json` 中按 experiment id 查找记录，查找失败后才可能进入训练入口。因此不能以“已验证 manifest”替代“可执行 server runner”，也不得改用本地治理提交 `0bf271622e145a7791359ccd11febd711769a96c` 作为 source commit。

## 已核验身份与门禁

| 项目 | 当前可核验值 | 判定 |
| --- | --- | --- |
| W001 locus | `D:/AI空间转录病理研究/PFMval_new_governed_workspaces/W001`；branch `codex/w001-mpp2-huber-loss-20260728-bound`；HEAD `a04319a...` | PASS |
| source remote ref | `refs/heads/codex/w001-mpp2-huber-loss-20260728-bound` -> `a04319a...` | PASS |
| approval / contract | `APR-mpp2-huber-loss-paired-v001-20260728-r001` / `569bb6d83dcc45ef77f900d92f58645211d05702aad6cb189e00131ae08d9962` | PASS |
| strict start-check | 治理工作树：`PASS=7 WARN=6 FAIL=0`；6 项 WARN 为既有 legacy 债务 | PASS |
| workspace locus check | `workspace check-locus --workspace-id W001 --cwd <W001>` | PASS |
| prepared manifests | `g10_A0_job_prepared.json` 与 `g10_A1_job_prepared.json` 各自 `job validate` PASS | PASS |
| paired parity | 除 arm/dataset 与 loss 元数据外一致；A001=MSE，A002=Huber `delta=1.0,reduction=mean` | PASS |
| G11 环境 | Python 3.13.5；torch 2.6.0+cu124；CUDA 12.4；RTX 4080 x1；D 盘 free 172,476,178,432 B；exit 0 | PASS（复用回传，不重跑） |
| result refs | `automation/server/W001/A001`、`automation/server/W001/A002` 当前均不存在 | PENDING |

## 阻塞证据

1. `git grep <experiment_id> a04319a... -- experiments project_state automation` 无匹配；该 pinned source 不含 G12 experiment/governance/job 记录。
2. `job run` 的 runner 固定读取 pinned worktree 的 `experiments/experiment_registry.json`，并以 `next(...)` 查找 `mpp2_huber_loss_paired_v001_20260728`；缺失时会在 pre-training 阶段失败。
3. prepared manifests 在远端治理 ref `refs/heads/codex/w001-mpp2-huber-loss-20260728` -> `45071ebc822c8eab859c4269d56391e610605a75` 可读取，但其本身不是 approved experiment source；G11 UX 改进提交 `0bf271...` 尚未推送且不得替换 source。

## 预算与边界

- approval `run_limit=4`；当前 `run_consumed=0`；A001、A002 均仅为 `ATTEMPT_PREPARED`。
- 没有 `EXPERIMENT_STARTED` 或 terminal attempt event；未创建 result bundle，未执行 result import/accept，未启动 G13。
- 未使用 SSH、SCP、HTTP remote command 或 Tunnel；本次只执行了本地只读 `git ls-remote`/`git fetch` 核验。

## 从配置解析的 Gitee-only 服务器操作卡（当前仅验证并停止）

服务器事实源 `configs/server_paths.yaml`：

- server repo：`D:\AIPatho\qzs\pfmval_deploy_git`
- server automation root：`D:\AIPatho\qzs\pfmval_automation`
- server config：`D:\AIPatho\qzs\pfmval_deploy_git\configs\config.server.yaml`

在服务器 PowerShell 中仅执行以下核验卡；任一检查失败即停止，**不得**运行 `job run`、训练脚本、`job pack` 或推送 result ref：

```powershell
$Repo = 'D:\AIPatho\qzs\pfmval_deploy_git'
$SourceRef = 'refs/heads/codex/w001-mpp2-huber-loss-20260728-bound'
$SourceSha = 'a04319a1d6aeafcd35d6adc7e2893f1d7b683bda'
$GovRef = 'refs/heads/codex/w001-mpp2-huber-loss-20260728'
$GovSha = '45071ebc822c8eab859c4269d56391e610605a75'
$Experiment = 'mpp2_huber_loss_paired_v001_20260728'

git -C $Repo fetch gitee $SourceRef $GovRef
if ((git -C $Repo rev-parse FETCH_HEAD) -ne $GovSha) { throw 'Gitee governance ref SHA drift' }
git -C $Repo cat-file -e "$SourceSha^{commit}"
git -C $Repo show "$SourceSha`:experiments/experiment_registry.json" |
  Select-String -SimpleMatch $Experiment
if ($LASTEXITCODE -ne 0) { throw 'STOP: pinned source lacks registered G12 experiment; do not start A001/A002' }
```

### 继续条件（需要新的明确治理修复/批准后才可执行）

只有在一个新的、可审计的 source ref 同时满足下列条件时，才可重新出具正式运行卡：

1. source ref 固定为允许替代 `a04319a...` 的新 explicit approval 所绑定 SHA；不得隐式使用 `0bf271...`。
2. 该 source 自身含 experiment registry、approval、critical contract、A001/A002 manifests，且 `job validate --require-head` 与 server `job run --dry-run` 均 PASS。
3. 新卡必须按 A001 -> A002 单 GPU 串行执行；每一臂开始后才记录/消耗 1 unit；工程故障立即打包 failed/incomplete attempt 后停止，禁止自动重试。
4. 每个 result ref 仅可返回 `automation/server/W001/A001` 或 `automation/server/W001/A002`，并含 source commit、approval、contract、loss 元数据、artifact size/SHA-256 与 terminal attempt event；本地后续仅 fetch/审核，不 import/accept。

