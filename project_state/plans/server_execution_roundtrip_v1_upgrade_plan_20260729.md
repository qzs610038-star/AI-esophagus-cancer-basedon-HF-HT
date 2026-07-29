# 项目级服务器执行往返 v1 升级计划

> 状态：`proposed_for_new_goal`  
> 范围：项目公共治理能力；W001/G12 仅作为回归证据，不作为实现归属。  
> 禁止：训练、真实 dispatch、结果接纳、SSH/SCP/HTTP 远程命令、修改不可变 result refs。

## 目标

把 G12 的 14 个 blocker 固化为回归用例，使后续同类实验达到：

- 一次 Gitee push；
- 一次服务器入口调用；
- 一次 preflight 返回全部 blocker；
- 终态已存在时只恢复封装/推送，绝不重训；
- 相同内容重放不新增 attempt、commit 或 run unit。

## 实施顺序

### P0：最小 `execution_bundle_v1`

- 从 immutable job 推导执行闭包。
- 仅包含 job、experiment 快照、approval、critical contract、attempt/budget、
  runner 及直接依赖、必要 Schema、path-id 快照和 bundle manifest。
- 排除历史 results、Dashboard、长文档、无关 experiment 和整仓 archive。
- 身份固定为 `bundle_id + source_commit + critical_contract_sha256`，不得绑定 branch tip、
  worktree 形态或无关文件换行。

### P1：`server_fingerprint_v1`

- 记录 OS、PowerShell、Git、Python candidates、Schema backend、PyTorch、CUDA/GPU、
  磁盘、path-id 和 Git 换行行为。
- 记录内容 SHA、生成时间和过期时间。
- 不记录凭据，不作为科学证据；未过期时由 preflight 复用。

### P2：聚合 `preflight-v2`

- 一次检查 bundle、source、approval、contract、attempt、lease、budget、数据、split、
  cache、入口 SHA、输出目录、resolved argv、运行环境和 Git 发布条件。
- 独立检查不得 fail-fast；统一输出 `failures[]`、`warnings[]` 和 `safe_to_start`。
- E0 科学合同和 E1 执行治理失败为 HARD FAIL；E2 历史材料和 E3 主机表现不得误阻断。

### P3：单入口可恢复状态机

```text
VERIFY_BUNDLE
→ FINGERPRINT
→ AGGREGATE_PREFLIGHT
→ ACQUIRE_LEASE
→ EXPERIMENT_STARTED
→ RUN
→ EXPERIMENT_TERMINAL
→ RESULT_BUNDLE
→ FF_PUSH
→ REMOTE_SHA_VERIFY
```

- preflight 未通过：不启动、不计费。
- started 且仍运行：拒绝重复启动。
- terminal 已存在：只恢复结果封装与发布。
- 相同 bundle：返回 `ALREADY_PUBLISHED`。
- 同 ID 不同 SHA：HARD FAIL。
- 禁止 force-push 和隐式 retry。

### P4：迁移与推广

- 在独立、干净的项目治理维护工作树实现，不继续堆叠 W001 专用补丁。
- 提取 `result_bundle_v1` 公共核心，不携带 W001 的 experiment/result 状态。
- 新 job 默认启用新协议；legacy v1/v2 保持只读兼容。
- 至少用两个不同类型实验完成无训练的本地/bare-remote 演练后再推广。

## 验收矩阵

- G12 已知 14 blocker 全部成为 fixture。
- dirty 主仓库、普通 governance 目录、source/governance 分离。
- `core.autocrlf=true/false/input`，LF/CRLF/mixed/binary。
- source object 缺失、attempt root 已存在、approval/contract/job 漂移。
- fingerprint 过期、Schema backend 缺失、GPU/磁盘不足。
- 并发 ref 前进、非 fast-forward、重复 publish。
- 全量测试、严格门禁和本地 bare Gitee 模拟均通过。

## 完成定义

- 服务器操作员只提供一个 execution bundle 路径。
- 操作卡不再手填多个 SHA、解释器、staging、ref 或 commit message。
- 一次 preflight 返回全部 blocker。
- 无关历史文件、dirty main 和 CRLF 不阻断 E0/E1 正常合同。
- 不操作真实服务器、不运行训练，也能在本地完整证明状态机行为。

