# G12 服务器往返排障复盘与治理简化建议

> 日期：2026-07-29  
> 工作树：W001  
> experiment：`mpp2_huber_loss_paired_v001_20260728`  
> 证据边界：本文是运行流程复盘与后续改进建议，不是实验结果，不执行 result import/accept，不改变 G12 科学合同。  
> 当前运行状态：G12 的 A003（MSE）与 A004（Huber δ=1）均已完成并经 Gitee 回传；G12-R 又生成并核验了 A003/R004、A004/R002 两个不可变“仅封装修订”。本地尚未执行 result import/accept，Registry 仍为 `pending`，本文不作科学比较或结论。

## 1. 结论

本次最耗时的步骤不是训练代码、数据或 GPU 环境，而是：

**在服务器上构造并验证“不可变治理执行包”的 locus（执行位置）与 identity（内容身份）。**

- 窄口径：从放弃脏治理 worktree、改用 Gitee archive 开始，连续 **4 次**卡在同一条执行包链路：
  1. archive 不是 Git worktree；
  2. source commit 被错误地在 archive 中查询；
  3. archive 全树逐文件验证受 Windows CRLF 影响；
  4. 在此之前，脏治理 worktree 本身已阻断一次。
- 宽口径：把 runner SHA、governance SHA、dirty checkout 等同类 locus/identity 错误计入，服务器与本地之间共出现 **6 轮**可辨识阻塞，均发生在正式训练启动前。
- 这 6 轮均未进入 `EXPERIMENT_STARTED`，因此没有浪费模型拟合预算，但消耗了大量人工复制命令、服务器执行、错误回传、本地修复、测试、提交和再次 fetch 的时间。
- 首次结果回传出现 **3 次本地打包阻塞** 与 **1 次跨平台完整性核验歧义**；G12-R 的权威封装修订又暴露 **4 轮**服务器兼容问题。它们都发生在正式训练之后，未触发重训、未额外消耗 run unit。
- 全链路可辨识阻塞共 **14 个**：训练启动/执行包 6 个、首次结果回传 4 个、G12-R 封装修订 4 个。最大单一簇仍是 execution bundle 的 locus/identity（6 个）；若按阶段合并，结果封装与回传共 8 个，但根因分散在协议类型、路径唯一性、换行完整性、Git ref 语义和 Schema 运行时能力。

总体判断：

**安全目标正确，但实现把“安全身份”过度绑定到了 Git worktree 形态、整个仓库历史和主机特定文本转换，导致低价值的串行失败。应保留关键科学与预算门禁，同时把治理执行包改为最小、内容寻址、一次性聚合校验的独立对象。**

## 2. 可审计的阻塞序列（训练启动前）

| 轮次 | 服务器表现 | 实际卡点 | 触发的安全机制 | 是否发现了有价值风险 |
| ---: | --- | --- | --- | --- |
| 1 | `runner SHA mismatch` | 把 dirty 主仓库 HEAD 当成 runner identity | runner 必须固定到批准的治理提交 | 有价值：防止旧 runner 启动正式训练 |
| 2 | checkout 被 dirty `experiment_dashboard.md` 阻断，随后 `governance SHA mismatch` | 操作卡依赖错误/不存在的路径，并试图让 dirty 主仓库承担治理 checkout | clean locus + exact governance SHA | 有价值：避免覆盖服务器现有未提交内容 |
| 3 | `governance worktree is dirty; stop` | 复用的 G11 return worktree 含残留修改 | 治理执行目录必须 clean | 有价值，但暴露设计不应依赖共享 worktree |
| 4 | `git ... worktree list ... exit 128`，strict start-check FAIL | Gitee archive 是普通目录，validator 仍假设它是 Git worktree | workspace locus validation | 低价值误报：host/locus 模型未区分 local worktree 与 server archive |
| 5 | `job source commit is unavailable locally: a04319...` | job schema/approval 在 governance archive 校验，但 source Git 对象错误地也在 archive 查询 | source commit availability | 低价值误报：双绑定存在，但 Git 对象库选错 |
| 6 | `governance archive file hash mismatch: ...best_epoch.txt` | 对整个仓库 archive 做逐 blob 验证，历史结果文本发生 LF→CRLF | full-tree Git blob integrity | 低价值误报：被检查文件与本次 A003 执行无关，差异仅为文本换行 |

### 2.1 A003 完成后的结果包与回传链路

以下事件来自当时的操作输出；其最终落点已由 Gitee result ref、result-v2 envelope 和本地独立回传回执反向核验。它们是**回传工程事件**，不是额外训练 attempt。

| 序号 | 表现 | 根因 | 被保护的边界 | 修正/结果 |
| ---: | --- | --- | --- | --- |
| R1 | A003 terminal event 被操作卡判为“非成功” | runner 终态枚举为 `completed`，而 result envelope 才使用 `success`；操作卡把两层状态混为一谈 | 不得把非零/失败运行包装成成功结果 | 显式要求 `EXPERIMENT_TERMINAL + completed + returncode=0`，再构建 `status=success` 的 result-v2 |
| R2 | `duplicate return artifact basename: metrics.json` | 手工回传卡同时处理根级 metrics 与 artifact 副本，artifact 以 basename 作为 bundle 内路径，发生命名碰撞 | artifact path 必须唯一，避免 silent overwrite | 仅生成一份 metrics 文件并以 `metrics` artifact 登记；A003 的实际提交为 `R003` |
| R3 | `dictionary update sequence element #0 has length 1; 2 is required` | PowerShell 对单个 `large_artifacts` 记录序列化为 JSON object，而 result-v2 builder 要求 JSON array of mappings | large artifact 的路径、size、SHA-256 与 retention 必须结构化可验 | 使用 `ConvertTo-Json -InputObject $Records.ToArray()` / 数组输入，确保单元素仍输出 `[...]` |
| R4 | 远端 artifact SHA 与 result envelope 原始 SHA 表面不一致 | Git 规范化了文本/CSV 的 LF，而服务器 envelope 记录原始 CRLF 字节 | 防止回传文件被替换或静默改写 | 逐文件只允许完整 `LF→CRLF` 物化；6 个 JSON 逐字节一致，另 17 个文本/CSV 均只存在这一种可复现差异 |

最终回传事实：

| attempt | result id | result ref / commit | artifact 核验 |
| --- | --- | --- | --- |
| A003 / MSE | `W001-A003-result-R003` | `automation/server/W001/A003` / `d59221d650821796a561f65fc98561334f914757` | 11/11 通过 |
| A004 / Huber δ=1 | `W001-A004-result-R001` | `automation/server/W001/A004` / `445f72bd12c4409d775b8e147ff6d218c9711244` | 12/12 通过 |

两份 return commit 均以 `6c0f9cad173068ef194f3c1558a78e7fe8ceb313` 为唯一父提交。两次训练合计 `run_consumed=2`，没有 A005 或隐式重试。

### 2.2 G12-R 不可变“仅封装修订”链路

G12-R 不覆盖 A003/R003 或 A004/R001，不重跑训练，只从历史结果内容构建下一可用 revision，并要求 build、validate、governance result-v2 validate 与 record-import 共用同一权威完整校验入口。

| 序号 | 服务器表现 | 实际卡点 | 安全意图 | 最终修正 |
| ---: | --- | --- | --- | --- |
| RR1 | `implementation SHA mismatch` | 操作卡把“实现提交”错误地等同于持续前进的治理分支 tip；分支加入审计/测试提交后必然不相等 | 固定实际执行的构建器版本 | 改为验证 implementation commit 存在且为远端 tip 的祖先，再从 exact implementation SHA 建 detached worktree |
| RR2 | `A003/R004 build failed` | 操作卡使用裸 `python` 且只回传总括错误，隐藏了实际解释器与 stderr | 构建失败必须停止，不能发布半包 | 使用显式解释器、保留原始错误输出，并为每次执行生成新的空 staging 子目录 |
| RR3 | 固定 venv 报 `ModuleNotFoundError: jsonschema` | 操作卡把严格 Schema 校验绑定到一个未登记依赖能力的 Python 路径 | 不得因解释器漂移而换用宽松校验 | 改为 capability-based 探测，并完整回传每个 candidate 的路径、退出码和 stderr |
| RR4 | `no server Python candidate can import jsonschema` | capability 探测证明服务器没有任何可用的 Python Schema 后端，继续换路径无法解决 | 不得安装临时依赖、不得放宽 `result_envelope_v2.schema.json` | 增加 PowerShell 7 `Test-Json -SchemaFile` 零安装严格后端；再以 `PFMVAL_FORCE_POWERSHELL_SCHEMA=1` 和跨后端回归证明 build、validate、result-v2 validate、record-import 判定同构 |

G12-R 最终事实：

| attempt / revision | result id | result ref commit / parent | artifact / bundle SHA-256 |
| --- | --- | --- | --- |
| A003/R004 | `W001-A003-result-R004` | `83a7362e30cdcae871c638f03733815db501d963` / `d59221d650821796a561f65fc98561334f914757` | 12 / `f962f9a845e500648d147be6e2ea365be7de266fd0adb10dcedd20ad760c45e1` |
| A004/R002 | `W001-A004-result-R002` | `a34f257f50bf1bf68a93571a37bb7dfc11b6236b` / `445f72bd12c4409d775b8e147ff6d218c9711244` | 13 / `a2bfca4b0140e90c3c8ce7c90a34b0ad3d3266c7e1f5b9d885c4876494dc2c78` |

两份新 revision 均保留原 job、attempt、source commit、approval、contract、指标与 artifact 内容，仅修正封装元数据和包结构。`run_consumed` 仍为 `2/4`，没有 result import/accept 或 G13。

对应本地治理提交链：

| 提交 | 作用 |
| --- | --- |
| `f1babe4` | 新增 source + governance bundle 双绑定的 job runner v2 |
| `9b0aa41` | 生成 protocol revision 2 的 A003/A004 |
| `530c32d` | 允许治理 bundle 使用 baseline 的后继提交 |
| `190e4ae` | 固定运行路径、外部 run root 与 terminal event |
| `bd7da76` | 允许普通 Gitee archive 作为治理 bundle |
| `4ccadad` | server host scope 下不再把 archive 当本地 worktree |
| `cc25bff` | source commit 改在 pinned source worktree 的 Git 对象库校验 |
| `6c0f9ca` | archive 完整性验证容许严格限定的文本 LF→CRLF |
| `9840efe` | 新增权威 `result_bundle_v1` build/validate/publish 工作流 |
| `3009587` | 将 implementation 身份从“tip 相等”修正为“祖先关系 + exact checkout” |
| `9da0fae` / `6b04ef2` | 显式 Python 与 capability-based Schema 解释器探测 |
| `78e01dd` / `50fb973` | 增加零安装 PowerShell Schema 后端及强制同构演练 |
| `8a44cbe` | 服务器卡默认使用零安装严格 Schema 校验 |
| `31e9993` | 纳入最终审核回执并闭合 G12-R 审计链 |

从 `bd7da76` 到 `6c0f9ca` 的四个补丁，本质上都在补同一个缺口：**没有在第一次下发前对“Windows 服务器上的普通 archive + 独立 source worktree”做完整端到端演练。**

## 3. 核心安全机制是什么

造成最多阻塞的是以下组合机制：

### 3.1 source + governance 双绑定

正式训练实际依赖两个不可混淆的对象：

- source：`a04319a...`，承担训练入口与科学实现；
- governance bundle：承担 experiment、approval、critical contract、job、attempt budget、路径映射和 runner。

它的作用是防止：

- 为了让训练“能跑”而暗中替换 experiment source commit；
- 使用未经批准的新 loss、数据、split、seed 或训练参数；
- 用旧 runner 误读 job v2；
- 在服务器 dirty checkout 上混入未审查变更；
- 通过新 job、worktree 或隐式 retry 绕过 run budget。

### 3.2 clean locus 与完整性验证

clean worktree、exact SHA、archive blob 校验用于证明服务器执行的不是一个被临时修改的治理副本。

该机制本身应保留，因为正式训练的审批与预算不能只依赖一张人工复制的命令卡。

### 3.3 当前实现过度的部分

问题不在“校验太严格”，而在“校验对象选得过大、交付形态假设错误”：

- 把普通 archive 当 Git worktree；
- 把 governance archive 当 source Git 对象库；
- 校验整个 repository tree，包括与 A003 无关的历史 `automation/results/**`；
- 依赖服务器 Git 的 `core.autocrlf`/attribute 行为解释 archive 字节；
- fail-fast，一次只暴露一个错误，导致必须多轮往返才能看到下一个阻塞。

因此，安全机制的有效部分与摩擦部分可以拆开。

## 4. 哪些门禁必须保留

以下门禁直接保护科学有效性、预算或可追溯性，不建议削弱：

1. `source_commit` 必须是完整 SHA，source worktree HEAD 必须精确匹配且 clean。
2. approval、protocol revision、critical contract、job、attempt 必须互相绑定。
3. 数据 manifest、split、训练入口和关键输入必须校验 SHA-256。
4. actual argv 必须来自不可变 job；运行期只能注入已登记路径和外部 output root。
5. `EXPERIMENT_STARTED` 之后立即消耗 run unit；工程故障不得隐式重试。
6. A003/A004 必须单 GPU 顺序执行。
7. external XZY 只能在 checkpoint 冻结后评估一次。
8. started/terminal event、关键结果文件 size/SHA-256、result branch 与 result envelope 必须保留。
9. Gitee 仍是唯一服务器传输通道。

## 5. 推荐的简化方案

### 5.1 用最小 execution bundle 替代整仓库 archive

新增独立的 `execution_bundle_v1`，只包含本次运行实际消费的治理闭包：

- exact job manifest；
- experiment registry 中对应 experiment 的冻结快照；
- active approval；
- workspace/attempt/budget 记录；
- critical contract；
- active MPP repair registry 与 canonical audit manifest；
- `server_job_v2` 等必要 schema；
- runner 及其直接依赖模块；
- `server_paths.yaml` 的冻结快照；
- `bundle_manifest.json`。

明确排除：

- `automation/results/**` 历史结果；
- checkpoints、cache、训练数据；
- 无关实验 job；
- Dashboard、长文档和与执行无关的治理回执；
- `histogene/`、`egnv1/`、`egnv2/`。

`bundle_manifest.json` 对“实际打包后的字节”记录 size/SHA-256，并记录：

- `bundle_id`
- `source_commit`
- `governance_commit`
- `job_id`
- `approval_id`
- `critical_contract_sha256`
- `runner_version`
- `created_at`
- 每个文件的 role、path、size、SHA-256

这样仍有不可篡改身份，但不会让无关历史文件阻断训练。

### 5.2 本地先构建并验证最终交付物

增加单一命令：

```text
pfmval_ops.py governance bundle-v1 build-and-verify --job <job.json> --output <bundle>
```

该命令应：

1. 从 Git canonical blob 或明确源文件构建 bundle；
2. 在本地解包；
3. 使用与服务器完全相同的 verifier 校验；
4. 模拟 `host_scope=server`；
5. 运行 job dry-run 到“即将创建 attempt root”为止；
6. 生成 bundle SHA 与服务器操作卡。

只有这一命令通过才允许 push。不要再把跨平台 archive 兼容性留到服务器第一次发现。

### 5.3 服务器改为一次聚合 preflight

当前是 fail-fast：每次只看到第一个失败。建议增加：

```text
pfmval_ops.py job preflight-v2 --bundle ... --source-worktree ... --run-root ... --report preflight.json
```

它一次完成并汇总：

- remote ref/SHA；
- bundle SHA 和每个 execution-critical 文件；
- source HEAD/clean/object availability；
- workspace/experiment/approval/contract/job/attempt/budget；
- Python/PyTorch/CUDA/GPU/磁盘（允许复用未过期的 G11 probe fingerprint）；
- 数据、split、缓存、入口 SHA；
- output root 是否已存在；
- resolved argv；
- A003/A004 parity。

即使其中一项失败，也继续完成所有只读检查，最终统一返回：

```json
{
  "status": "blocked",
  "failures": [...],
  "warnings": [...],
  "safe_to_start": false
}
```

这样一次服务器回传即可暴露全部问题。

### 5.4 操作卡降为一个稳定入口

服务器不应再手工维护几十个 PowerShell 变量。bundle build 后生成一条固定调用：

```text
run-governed-job-v2.ps1 -Bundle <bundle> -SourceWorkspace W001 -Attempt A003 -DryRun
```

正式运行只去掉 `-DryRun`。脚本内部解析已登记路径，人工不再拼接 SHA、manifest 路径与 archive 目录。

### 5.5 将安全层级拆开

| 层级 | 内容 | 失败处理 |
| --- | --- | --- |
| E0 科学合同 | source、loss、数据、split、seed、epoch、selection policy、XZY policy | HARD FAIL，需新批准 |
| E1 执行治理 | approval、budget、job、runner、路径、attempt event、bundle manifest | HARD FAIL；语义不变适配可走已有 adaptation policy |
| E2 支撑历史 | 历史结果、Dashboard、说明文档、旧回执 | 不进入 execution bundle；不得阻断新训练 |
| E3 主机表现 | CRLF、路径分隔符、console encoding | 规范化后校验；记录 WARN/adapter evidence |

本次第 6 轮失败是典型的 E2/E3 问题错误阻断 E0/E1。

### 5.6 已落地：单一、类型稳定的 `result_bundle_v1`

G12-R 已把此前建议落地为权威入口：

1. 在空 staging 目录构建，固定 `completed + returncode=0 → success`。
2. 强制 `artifact_id`、`retention`、唯一相对路径及 artifact 闭包；`artifacts` 与 `large_artifacts` 始终为数组。
3. 同时校验 Schema、语义字段、size、SHA-256、原始换行与 Git 规范化完整性。
4. build、validate、governance result-v2 validate 与 record-import 使用同一入口，不再允许局部 validator 与完整 Schema 分叉。
5. 发布使用不可变 revision、fast-forward 与远端 commit SHA 复核；重复执行不得覆盖历史 revision。
6. Schema 后端支持 Python `jsonschema` 与 PowerShell 7 零安装严格验证，并有强制后端同构回归。

因此 R1--R4 对应的状态、命名、数组和换行歧义已形成代码与 fixture 防线。仍待简化的是服务器操作卡本身：解释器/后端能力探测、staging 分配、发布与回传应由一个生成式入口封装，不再由用户维护多段 PowerShell。

### 5.7 跨阶段共性根因

14 个 blocker 的共性不是“门禁太多”，而是：

- 正确的不变量被错误绑定到易变化的载体，例如 branch tip、worktree 形态、整仓 archive 字节或某个 Python 环境；
- 协议字段依赖 PowerShell 管道基数、Git 换行配置等隐式主机行为；
- preflight 采用 fail-fast，一轮只暴露一个 blocker；
- 本地下发前没有用服务器等价能力矩阵演练最终操作卡。

优化原则应是：**保留不变量，替换载体；保留严格校验，去除主机偶然性；一次聚合诊断，不再串行试错。**

## 6. 推荐实施顺序

### P0：已完成

1. `result_bundle_v1` 权威 build/validate/publish 入口。
2. A003/R003、A004/R001 历史 fixture，以及 retention、artifact_id、数组退化、重名路径、未登记 sidecar、LF/CRLF、重复推送、远端 SHA 不一致回归。
3. 严格 Schema 的零安装 PowerShell 后端和 Python 后端同构演练。
4. A003/R004、A004/R002 不可变“仅封装修订”及远端 ref/parent/bundle SHA 复核。

### P0：仍待实施

1. 实现最小 execution bundle schema、builder、verifier。
2. 实现聚合 `preflight-v2`，一次报告所有只读 blocker。
3. 生成单入口服务器操作卡，并自动选择已验证的 Schema 后端。
4. 将服务器 runtime capability 固化为可复用、带时效的 fingerprint，而不是在操作卡内临时猜测路径。
5. 在 Windows 上覆盖以下端到端场景：
   - dirty 主仓库；
   - 不创建新 worktree；
   - governance 是普通目录；
   - source 与 governance 分离；
   - `core.autocrlf=true/false/input`；
   - archive 含无关历史结果；
   - source object 缺失；
   - attempt root 已存在；
   - approval/contract/job 任一字段漂移。
6. 将完整“构建→严格验证→提交→fast-forward 推送→远端 SHA 复核”封装成一个稳定命令；操作卡只传 bundle/ref，不再手填实现 SHA、解释器候选和 staging 路径。

### P1：后续正式实验默认启用

1. job dispatch 直接产出 bundle 和操作卡。
2. 服务器只执行一次 preflight；全部 PASS 后启动。
3. preflight receipt 与 result package 均通过既定 Gitee ref 回传。
4. 相同 bundle SHA 的重复传输幂等，不创建新 attempt、不消耗预算。

### P2：达到稳定后清理兼容路径

保留旧 v1/v2 job 的只读 import 能力，但停止为新实验生成“整仓库 archive + 多段 PowerShell”操作卡。不要自动删除旧 worktree、archive 或结果。

## 7. 验收指标

后续治理改进至少达到：

- 以本次 **14 个可辨识 blocker** 为基线，下一次同类双臂实验的服务器往返 blocker 目标为 **0**；
- 从本地批准完成到服务器正式启动：**1 次 Gitee push + 1 次服务器粘贴执行**；
- preflight 失败：**1 次回传暴露全部 blocker**；
- 不因无关历史结果、Dashboard、CRLF 或 dirty 主仓库阻断；
- 不创建 attempt/job 专属 worktree；
- 不降低 source/approval/contract/input/budget/result 的 E0/E1 门禁；
- 相同 bundle 重传不训练；
- 工程故障不隐式重试；
- 操作卡不再要求用户手工维护多个 SHA 和路径。

## 8. 本次判断

| 检查 | 判定 |
| --- | --- |
| 多轮排障主要由训练代码或环境导致 | FAIL：主要由治理执行包 locus/identity 导致 |
| 核心安全目标有必要 | PASS |
| 当前整仓库、worktree 形态相关校验是否过度 | PASS：存在明确过度绑定 |
| 是否可以在不削弱科学门禁的前提下简化 | PASS |
| G12 是否已完成运行与回传边界 | PASS：双臂 completed、两个 result ref 已核验 |
| G12-R 是否已完成权威封装修订 | PASS：A003/R004、A004/R002 均经完整 Schema、artifact 闭包、bundle SHA 与远端 SHA 核验 |
| `result_bundle_v1` 是否已落地 | PASS |
| execution bundle / 聚合 preflight / 单入口服务器卡是否已落地 | FAIL：仍是后续治理优化的主要缺口 |
| 是否已 result import/accept 或启动 G13 | PASS：均未执行 |
| 是否可立刻实施上述 P0 改进 | CONDITIONAL GO：须以独立批准任务实施，不得追溯改写本次不可变 result refs |

总体结论：**CONDITIONAL GO（针对后续治理改进）**。G12 与 G12-R 的运行、回传和不可变封装修订已闭环，但结果仍处于未 import/未 accept 状态。`result_bundle_v1` 已解决 return path 的核心分叉；下一优先级应转向最小 execution bundle、聚合 preflight 和单入口服务器卡。后续改进必须作为独立、明确批准的工程治理任务执行，不得改写本次 source commit、approval、critical contract、attempt 或 result ref。
