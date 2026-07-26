# Workflow Governance v3 Revision 005 最终独立审核包

## Claim

- 可检验主张：`DIR-20260727-002` 授权的本地结构性重构已按 G00–G06
  顺序实现，P0A/P0B/P0C、R0–R11 的必需设计与实现证据通过，且没有越权执行
  服务器、训练、结果接纳、外部平台接入、资产迁移、目录解禁、close execute
  或清理。
- 请求决策：是否接受当前工作树中的已授权本地重构实现。
- 基线 commit：`be7ac120642fd3ee627239e4d829b651eff3de65`。
- 最终 commit：`NOT CREATED`。本任务未授权 Git stage/commit/push；最终实现由
  上述 HEAD 加当前 55 个变更路径共同定义。
- 指令：`DIR-20260727-002`。

## 总结论

- Overall: `GO`
- Reason：全部必需 D/I 项通过；完整本地测试为 130 passed；严格状态门禁为
  PASS=7、WARN=6、FAIL=0；v1 样本与 job validator 兼容；七个生成视图/状态
  文件连续两轮字节稳定；受保护目录无变更；所有未获授权的 O/S 项均按
  `NOT RUN` / `N/A` 保留。
- Evidence scope：只接受当前未提交工作树中的本地结构实现，不表示真实
  workspace/experiment/attempt/result、服务器运行、训练、迁移、清理、工作流
  扫描或科学结论已经运行或通过。

## Sources inspected

| Source | Lifecycle | What it proves |
|---|---|---|
| `AGENTS.md` | current / normative | 固定安全边界与事实源顺序 |
| `CURRENT_STATE.md` | current / derived | 当前指令、active plans、状态摘要 |
| `project_state/current_state.json` | current / authoritative | 当前机器可读状态与 source hashes |
| `project_state/directives.jsonl` | current / append-only | `DIR-20260727-002` 的授权与排除项 |
| `experiments/experiment_registry.json` | current / authoritative | 31 个既有实验；本轮未新增或接纳结果 |
| `project_state/document_registry.json` | current / authoritative | active 文档与派生视图生命周期 |
| `configs/server_paths.yaml` | current / authoritative | Gitee-only 与路径策略；本轮未操作服务器 |
| `project_state/governance_receipts/workflow_governance_v3/G00.json`–`G05.json` | current stage evidence | 阶段基线、命令、边界和 NOT RUN |
| 当前源文件、schema、测试与空 registry | current implementation | 本地结构实现、兼容层和负向门禁 |

## 模块验收矩阵

| Module | D | I | O | S | 边界 |
|---|---|---|---|---|---|
| R0 | PASS | PASS | PASS | N/A | 基线与 v1 fixture |
| R5b | PASS | PASS | PASS | N/A | Document Registry/状态稳定性 |
| P0A | PASS | PASS | NOT RUN | N/A | 未创建真实实验闭环 |
| R1 | PASS | PASS | NOT RUN | N/A | 空 workspace shadow registry |
| R2 | PASS | PASS | NOT RUN | N/A | 未创建/迁移真实工作树 |
| R3 | PASS | PASS | NOT RUN | N/A | 未签发真实训练批准 |
| R4 | PASS | PASS | NOT RUN | N/A | 未运行真实 critical contract gate |
| R5 | PASS | PASS | NOT RUN | N/A | 未导入或接纳真实 result v2 |
| P0B | PASS | PASS | PASS | N/A | 用户导航与简洁实验视图已生成 |
| R6 | PASS | PASS | PASS | N/A | metadata-only asset shadow |
| R7 | PASS | PASS | PASS | N/A | 中性 metrics 兼容层及两个 active caller |
| R8 | PASS | PASS | PASS_SHADOW_ONLY | N/A | 只做策略/迁移 preview |
| R9 | PASS | PASS | NOT RUN | N/A | close execute/清理未授权 |
| P0C | PASS | PASS | NOT RUN | N/A | 未写真实科研推理记录 |
| R10 | PASS | PASS | NOT RUN | N/A | 未触发真实事实/文档运行 |
| R11 | PASS | PASS | NOT RUN | N/A | 用户未提供真实扫描清单 |
| G06 | PASS | PASS | 见完整矩阵 | N/A | 集成、兼容、回退设计与独立审计 |

`D` 表示设计证据，`I` 表示实现/集成证据，`O` 表示真实运行证据，`S`
表示科学结论证据。

## Fresh checks

| Result | Check | Fresh evidence | Boundary |
|---|---|---|---|
| PASS | 完整本地测试 | `python -m pytest -q tests --basetemp C:\t\g06_final` → `130 passed, 2 warnings in 46.31s` | 测试覆盖，不等于真实训练 |
| WARN | Windows 长路径对照 | 长 basetemp 首次为 129 PASS/1 FAIL；同一失败用 `C:\t\p6` 复跑为 1 PASS | 环境路径长度，不是代码回归 |
| PASS | 严格状态门禁 | `agent start-check --strict` → PASS=7 WARN=6 FAIL=0 | 当前本地状态 |
| PASS | v1 兼容 fixture | R0 七类样本 hash/边界测试 → 1 PASS | 只证明冻结样本兼容 |
| PASS | v1 job validator | 既有 Ridge r003 job manifest valid | 不执行 job |
| PASS | 生成视图幂等 | current state、CURRENT_STATE、Document Registry、Dashboard、Progress、Project Guide、README 两轮 hash 均稳定 | 本地生成视图 |
| PASS | schema/CLI | strict schema gate、`workflow validate`、py_compile、`git diff --check` | 静态与本地接口 |
| PASS | 保护边界 | `git diff/status -- histogene egnv1 egnv2` 均为空 | 三个受保护目录 |
| PASS | 零真实运行记录 | approvals、attempt events、scientific records、project facts 均不存在；workspace/asset/workflow registry entry 均为 0 | O 保持 NOT RUN |
| PASS | 离线边界 | R11 离线负向测试通过，未接入外部平台或网络 | 本地 fixture |

## 变更面

- 运行与治理：`deploy/pfmval_ops.py`、`scripts/pfmval_state.py`，新增
  governance/assets/views/science/knowledge/workflows 模块。
- schema/registry：新增 workspace、approval、attempt、critical contract、
  job/result v2、asset、science、fact、document profile、workflow catalog
  schema，以及空的 workspace/asset/workflow registry。
- 兼容提取：新增 `pfmval_core.metrics`；仅将两个 active MPP caller 的 import
  指向中性模块，受保护 `histogene/utils.py` 未修改。
- 用户视图：新增 `PROJECT_GUIDE.md` 与
  `experiments/experiment_progress.md`，并纳入既有生成链。
- 测试/证据：新增 R0 fixture、阶段测试和 G00–G06 收据。
- Git locus：10 个 tracked 文件修改、45 个 untracked 文件新增；没有 stage、
  commit 或 push。

## 兼容与回退证据

- v1/v2 job/result 采用 dual-read；冻结的 preflight、smoke、formal、success、
  failed、incomplete 与 legacy accepted 样本 hash 未漂移。
- 新 registry 均为空 shadow 状态；移除新增层不会要求恢复真实 workspace、
  attempt、asset、workflow 或 accepted result。
- 生成视图由 registry/state 重新生成，连续两轮 byte-stable。
- 可回退顺序：移除新增模块/schema/空 registry/视图/测试/收据；反向应用
  `deploy/pfmval_ops.py`、`scripts/pfmval_state.py`、两个 active MPP import
  和生成状态文件的限定补丁；再运行 docs scan、views refresh、state sync 与
  全量测试。
- 未实际执行破坏性回退，因为用户明确排除了清理/删除；回退证据为 additive
  边界、冻结兼容样本和可枚举的 working-tree diff。

## WARN

- 当前实现未提交，`source_commit` 仍是基线 HEAD；提交、推送及远端 SHA
  核验需要用户另行授权。
- 五个 2026-07-06 legacy accepted result 缺完整 result envelope；这是既有
  WARN，本轮未伪造修复。
- 标准化 MPP 标签仍有已知重复/冲突 barcode；重新生成属于受保护资产任务，
  本轮未执行。
- `pfmval_py310` 目录不存在；测试实际使用 Python 3.13.1。
- codebase-memory 重建曾崩溃且图谱陈旧；本轮以实际文件、CLI 和测试回退验证，
  记为非阻塞 WARN。

## 明确 NOT RUN / N/A

- NOT RUN：服务器访问、Gitee dispatch/fetch/return、训练、真实 experiment/
  attempt、result import/acceptance、资产物理迁移、目录解禁、Hook enforce、
  close execute、工作树/缓存/checkpoint/数据清理、外部平台/云/网络接入、
  真实事实/文档/科研记录、真实 workflow scan/activation。
- N/A：本地结构重构不产生新的科学结论，因此所有 S 项均为 `N/A`。

## Next authorized action

当前本地实现可接受。若需要形成不可变交付点，下一步应由用户明确授权后再对
这 55 个变更路径进行限定 staging、commit 和可选 push；在此之前不得把基线
HEAD 表述为包含本次重构。
