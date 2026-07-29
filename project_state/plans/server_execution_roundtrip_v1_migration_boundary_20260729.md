# 服务器执行往返 v1 后续迁移边界

> 当前实现：G15 / W002  
> 本文只定义迁移门槛，不授权真实服务器运行、训练、结果 import/accept 或旧资产清理。

## 可立即使用

- 本地为 governed job 构建并验证 `execution_bundle_v1`；
- 本地/服务器用 bundle 路径执行聚合 `preflight-v2`；
- 复用或刷新 `server_fingerprint_v1`；
- 在本地 bare remote 演练 fast-forward、幂等重复和并发 parent 漂移；
- 对 terminal 已存在的 attempt 恢复既有 `result_bundle_v1` 封装与发布。

## 新 job 的迁移要求

新 job 切换为默认 `execution_bundle_v1` 前必须满足：

1. job 使用完整 `source_commit`、approval、critical contract、attempt、budget 和 path-id；
2. `resolved_argv` 或登记 adapter 能在 bundle path snapshot 下确定性解析；
3. runner 或登记 adapter 必须在 `PFMVAL_RESULT_SOURCE_BUNDLE` 指定目录生成
   可由既有 `result_bundle_v1` 消费的 source bundle；不得在服务器卡中手工拼装；
4. 两种实验类型均完成无训练 build/validate/preflight 演练；
5. 任务专用 Gitee ref 完成首次 push、幂等 replay、非 fast-forward 和 remote SHA 复核；
6. 实际服务器 fingerprint 只经 Gitee 回传并完成一次人工核准；
7. 正式训练仍需 job/source 绑定的独立显式用户批准。

## Legacy 保留

- legacy result-v1/v2：继续只读 validate/import 兼容，不重写 envelope；
- legacy job-v1/v2：继续只读 validate 和历史审计；不自动迁移、重跑或生成新 attempt；
- W001/G12 result refs、source commit、approval、contract、A003/A004 和 revision 均保持不可变；
- `result_bundle_v1` 是唯一结果封装协议，本次没有 `result_bundle_v2`。

## 尚未运行

- 真实服务器 fingerprint；
- 真实 Gitee 任务分支上的服务器 preflight receipt 往返；
- 任何 dispatch、训练、run-unit 消耗；
- 结果 import/accept；
- legacy worktree/archive/result ref 清理。

以上项目必须在新的显式服务器执行批准中逐项核验；不得把本地 fixture PASS 表述为真实服务器 PASS。
