# G15 项目级服务器执行往返 v1 实现说明

> 授权：`DIR-20260730-001`  
> 实现归属：`W002` / `codex/w002-server-execution-roundtrip-v1-20260729`  
> TDD 提交：RED `089232b` → GREEN `1d33977`  
> 边界：仅本地与 bare remote 演练；未操作真实服务器、未 dispatch、未训练、未 import/accept 结果。

## 1. 公共入口

```powershell
# 本地构建并严格验证最终 execution bundle
python deploy/pfmval_ops.py governance execution-bundle-v1 build `
  --job <job.json> `
  --output <empty-bundle-directory>

# 任意主机复核 bundle
python deploy/pfmval_ops.py governance execution-bundle-v1 validate `
  --bundle <bundle-directory>

# 服务器聚合预检；失败时仍返回全部可完成的只读检查
python deploy/pfmval_ops.py job preflight-v2 `
  --bundle <bundle-directory> `
  --report <preflight-v2.json>

# 单入口执行或恢复
python deploy/pfmval_ops.py job execute-v1 `
  --bundle <bundle-directory>
```

bundle 内同时生成 `run_server.ps1`。操作员只需执行该脚本或把一个 bundle 路径传给稳定 CLI；不再填写 source/governance SHA、解释器候选、staging、result ref 或 commit message。

## 2. `execution_bundle_v1`

实现位于 `scripts/pfmval_execution_roundtrip.py`，Schema 位于
`project_state/schemas/execution_bundle_v1.schema.json`。

bundle 只包含：

- immutable job；
- experiment、approval、critical contract、attempt/budget、workspace 快照；
- job 引用的 path-id 快照；
- runner 及直接依赖；
- 必要 Schema；
- 内容寻址的 `bundle_manifest.json`；
- 单入口 `run_server.ps1`。

明确排除历史 `automation/results/**`、Dashboard、长文档、无关 experiment、
`histogene/`、`egnv1/`、`egnv2/`、数据、cache 和 checkpoint。

`bundle_id` 由以下不可变内容共同计算：

```text
job_id + source_commit + critical_contract_sha256 + every bundled file path/size/SHA-256
```

`created_at`、branch tip、worktree 形态、dirty main 和无关换行不参与身份。相同输入重复构建得到相同 `bundle_id` 与 `bundle_sha256`。

## 3. `server_fingerprint_v1`

Schema 位于 `project_state/schemas/server_fingerprint_v1.schema.json`。fingerprint 记录：

- OS、PowerShell、Git；
- Python candidate 及 `jsonschema` / PyTorch 能力；
- Python 与 PowerShell 严格 Schema backend；
- PyTorch、CUDA、GPU 数量与空闲显存；
- 磁盘空闲空间；
- bundle 冻结的 path-id；
- Git `core.autocrlf` 表现；
- `generated_at`、`expires_at` 与能力内容 SHA-256。

未过期且内容 SHA 有效时复用；过期后原子刷新。任意层级出现 credential、password、secret、token 或 private key 字段均拒绝写入。

## 4. 聚合 `preflight-v2`

preflight 一次汇总 18 个独立检查，统一输出：

```json
{
  "status": "ready | blocked",
  "safe_to_start": false,
  "passes": [],
  "failures": [],
  "warnings": []
}
```

覆盖 bundle、source HEAD/clean/object/entrypoint、approval/contract/job/data/split、
attempt root、lease、budget、Schema backend、GPU、磁盘、fast-forward 和 remote identity。
E0/E1 失败为 `HARD_FAIL`；Git 换行等 E3 主机表现只记 `WARN`，不把无关历史材料升级为 blocker。

## 5. 可恢复状态机

```text
VERIFY_BUNDLE
→ FINGERPRINT
→ AGGREGATE_PREFLIGHT
→ EXPERIMENT_STARTED
→ RUN
→ EXPERIMENT_TERMINAL
→ RESULT_BUNDLE
→ FF_PUSH
→ REMOTE_SHA_VERIFY
```

关键恢复语义：

- preflight blocked：不写 started，不消耗 run unit；
- started 存在但 terminal 不存在：返回 `RUNNING`，拒绝重复启动；
- terminal 已存在：跳过 executor，只恢复 result packaging 与 publish；
- publish receipt 已存在且 bundle 相同：返回 `ALREADY_PUBLISHED`；
- 同 job ID 绑定不同 bundle SHA：HARD FAIL；
- 非 fast-forward 或远端 SHA 漂移：停止，禁止 force-push；
- result packaging/publish 调用既有 `result_bundle_v1`，没有新建结果协议。

## 6. W002 治理维护绑定

workspace registry v1 原先把所有绝对工作树 HEAD 固定为 experiment source commit，
无法表达会持续前进的治理维护分支。新增可选
`workspace_kind=governance_maintenance`：

- 路径、branch 与 clean 仍为硬门禁；
- experiment workspace 继续严格固定 HEAD；
- governance maintenance 允许同一已登记分支在提交后前进；
- `W002` 仍永久不复用，保留初始基线 commit 供审计。

该兼容扩展不改变 W001 或已有 experiment workspace 的固定 source-commit 语义。
