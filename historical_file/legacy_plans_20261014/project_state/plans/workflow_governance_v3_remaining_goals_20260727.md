# 治理 v3 重构剩余 Goal — 执行计划

> 制定日期：2026-07-27
> 状态：`draft_pending_user_review`
> 当前基线：`main@17c202da02ca007de08192f89be9f9c2789519b3`
> 当前结论：G08 本地结构重构已经完成并通过独立复核；后续目标是验证真实
> 使用效果，而不是继续扩建框架。
> 固定边界：服务器只经 Gitee；不得直接 SSH/SCP/远程命令；不得隐式修改
> `histogene/`、`egnv1/`、`egnv2/` 或删除资产。

## 一、距离彻底收口还有什么

### 已完成

- R0–R11、P0A/P0B/P0C 的本地 D/I 已接受。
- G08 已完成 canonical `pfmval-governance` Skill、六个 active workflow、
  七条技术路由、manifest register、测试和本地 Catalog。
- 当前新鲜验证：`141 passed`；严格门禁 `PASS=7 WARN=6 FAIL=0`；
  Workflow Catalog `6/6 active@0.1.0`、均 `standardized=false`。

### 尚未完成

- 真实本地知识/文档工作流尚未用真实用户输入运行，P0C/R10/R11 仍为
  `O=NOT RUN`。
- 真实 experiment、W###、approval、run budget、critical contract、job
  manifest 尚未完成一次本地预派发闭环，R1–R4 仍为 `O=NOT RUN`。
- Gitee/server 往返、真实 attempt、result envelope/import 尚未按新治理链路
  运行，P0A/R2–R5 仍为 `O=NOT RUN`。
- R8 仅 `PASS_SHADOW_ONLY`；R9 close/cleanup 未运行，S7 未授权。
- 若干治理 directive 仍为 active；总方案尚未从“重构实施”转入“稳定维护”。

因此，按完整运行证据口径还剩 **G09–G14 六个 Goal**。其中 G09 可独立进行，
G10→G11→G12→G13 必须顺序执行，最后由 G14 统一收口。

## 二、依赖图

```text
G09 真实本地维护工作流 ──────────────────────────┐
                                                ├─→ G14 最终审核与生命周期收口
G10 实验本地预派发 → G11 Gitee诊断往返 → G12真实闭环 → G13生命周期/关闭 ┘
```

## 三、Goal 清单

### G09：六个维护工作流真实本地首轮运行

**完成目标**

使用用户提供的最小真实输入，各运行一次：

- project fact preview 与逐条确认；
- meeting brief；
- research path；
- learning guide 接口；
- document lifecycle preview；
- workflow discovery；
- paper-output 材料索引并停在 `template_pending`。

**用户需提供**

- 一条可确认的真实项目事实；
- 一段组会周期与近期工作简述；
- 一条重要结论、阴性路线或开放假设；
- 一个现有学习指南及关联源码；
- 一个待做 freshness 检查的 active 文档；
- 一组明确的 tracked discovery 路径；
- 一组论文材料引用；无需提供论文模板。

**验收**

- 六个用户入口、七条技术路由均产生可审计的真实本地输出。
- project fact 未经该条事实的显式确认不得写入。
- learning guide 保持 `interface_reserved`；paper 保持 `template_pending`。
- lifecycle 只生成 preview；新发现 workflow 最多为 candidate。
- Experiment/Workspace/Asset Registry 零变化。
- 全量测试通过，严格门禁无新增 WARN/FAIL。

**阻塞退出**

缺少真实内容、来源引用冲突或事实表述有歧义时停止对应写入；不得用 fixture
冒充真实 O 证据。

**新对话 Goal 文本**

> 创建 Goal G09：使用我提供的最小真实项目输入完成六个治理工作流的首轮本地
> O 验证；逐条预览并只在明确确认后写项目事实，其他输出保持既定接口和候选
> 边界。以七条路由可审计、零越权写入、全量测试和严格门禁通过为完成条件。

---

### G10：真实实验的本地预派发闭环

**完成目标**

选择一个真实待开展实验，在不访问服务器、不开始训练的前提下完成：

```text
experiment register
→ W### workspace 初始化/绑定
→ protocol approval 与结果性执行次数预算
→ critical contract digest
→ attempt/job manifest
→ 本地 validate/dry-run
```

**用户需提供**

- experiment ID、科研问题和假设；
- 指定本对话 `W###`；
- 变量、数据/模型引用、证据等级和停止规则；
- 允许的结果性执行次数；
- 是否允许创建对应本地 worktree。

**验收**

- experiment、W###、approval、budget、contract、attempt/job 身份唯一绑定。
- critical contract 重放摘要稳定。
- job manifest 绑定当前 source commit，validate/dry-run 通过。
- 没有 Gitee dispatch、服务器访问、训练或 run unit 消耗。
- 生成视图与 Registry 一致；测试和严格门禁通过。

**阻塞退出**

未指定 W###、关键合同不完整、预算不明确、工作区 dirty/路径冲突或需要修改
受保护资产时停止，不创建伪实验。

**新对话 Goal 文本**

> 创建 Goal G10：为我指定的真实实验完成本地预派发治理闭环，覆盖 experiment
> 注册、W### 绑定、协议批准、次数预算、critical contract、attempt/job
> manifest 和 dry-run；不得访问服务器或启动训练。以身份绑定、幂等、视图一致、
> 全量测试和严格门禁通过为完成条件。

---

### G11：Gitee-only 非证据性服务器诊断往返

**依赖**：G10。

**完成目标**

使用 G10 的 source commit 与 W###，只通过 Gitee 完成一次不训练、不产生科学
证据的诊断 request→server record→return→local import 往返，验证：

- exact commit fetch；
- 单写者/attempt ref；
- 状态与日志回传；
- remote/local SHA 一致；
- 诊断不会进入 Experiment Result 或消耗 run unit。

**用户需批准**

- 精确 W###、source commit、diagnostic command ID；
- 一次 Gitee 往返；
- 明确禁止训练和结果接纳。

**验收**

- 全程无 SSH/SCP/HTTP remote command/tunnel。
- 返回包有 commit、时间、输出 hash 和允许命令 ID。
- 诊断记录可审计但不成为实验结果。
- 冲突、远端 SHA 不一致或服务器不可用时零重试、零覆盖。

**阻塞退出**

远端状态不可验证、Gitee 分支冲突、出现未授权命令或需要训练时立即停止。

**新对话 Goal 文本**

> 创建 Goal G11：基于 G10 的 W### 和 source commit，经 Gitee-only 完成一次
> allowlisted、非证据性服务器诊断往返；不得训练、不得写实验结果、不得使用
> 直接远程通道。以 SHA/日志/命令 ID 可审计且零 run-unit 消耗为完成条件。

---

### G12：一次真实实验的 P0A 全生命周期闭环

**依赖**：G10、G11。

**完成目标**

在单独明确批准下，对 G10 实验执行一次最小受控 attempt：

```text
dispatch
→ server exact-commit run
→ pack/return
→ quarantine/verify
→ result import
→ accepted/rejected/failed 的正确生命周期
→ 用户/Agent 视图刷新
```

**用户需批准**

- job ID、source commit、W###、attempt ID；
- 训练/运行类型、资源和结果性次数预算；
- 科学/工程门禁、停止条件和是否允许一次实际运行；
- 结果回传与 import 范围。

**验收**

- 启动前所有批准、路径、contract digest 与 manifest 一致。
- 不发生隐式重试；E1 适配失败不冒充训练次数。
- 结果 envelope、critical/supporting/diagnostic 分层可验证。
- Registry 与双视图原子一致。
- 科学指标失败不等于治理失败：只要失败/拒绝被正确记录，生命周期闭环仍可
  通过；不得为了得到好指标自动重跑。

**阻塞退出**

任一 hash/批准/预算不匹配、服务器路径漂移、输入无法验证或结果包不完整时
停止在 quarantine/pending，不接纳、不重跑。

**新对话 Goal 文本**

> 创建 Goal G12：对 G10 已批准实验执行一次受控 P0A 全生命周期 attempt，
> 只经 Gitee，严格绑定 W###、job、source commit、critical contract 和一次
> 结果性预算；完成结果回传、验证、import 与视图一致。门禁失败时正确记录并
> 停止，不自动重试或调参。

---

### G13：生命周期迁移与工作树关闭的精确目标试运行

**依赖**：G12 已完成结果回传与 import。

**完成目标**

- 对 G12 产生的一个低风险、非保护资产执行一次可回退的 lifecycle 迁移；
- 对该精确 W### 先执行 close preview；
- 用户确认精确目标、保留清单和恢复策略后，才允许 close execute；
- 验证 R8/R9，不要求解除 `histogene/egnv1/egnv2` 的保护。

**用户需批准**

- 唯一 W###、experiment ID 和解析后的绝对路径；
- 单个资产 ID 及目标 lifecycle；
- preservation tag；
- run root/checkpoint 的 retention；
- 明确的删除清单；未列明内容一律保留。

**验收**

- preview 与 execute 权限分离。
- 无运行进程、未回传结果、dirty 未提交内容或未知路径。
- 每项迁移 append-only；无全局 prune、无递归扩大删除范围。
- 用户视图不再显示 running，且保留资产可恢复。

**阻塞退出**

目标不唯一、路径变化、仍有运行进程/未回传结果、恢复策略缺失或涉及受保护
目录时禁止 execute，只交付 preview。

**新对话 Goal 文本**

> 创建 Goal G13：仅针对我明确指定的 W###、experiment 和单个非保护资产，
> 完成 lifecycle migration 与 workspace close 的 preview；只有目标路径、
> 保留清单和恢复策略完全一致时才执行已批准的 close。禁止全局 prune、范围
> 扩大、受保护目录解禁或未列明删除。

---

### G14：治理 v3 最终运行效果审核与生命周期收口

**依赖**：G09、G10、G11、G12、G13。

**完成目标**

- 汇总 P0A/P0B/P0C、R0–R11 的最终 D/I/O/S 矩阵；
- 更新 active plan 中的 O 状态与真实证据；
- 清理已决定但仍显示 `[ ]` 的设计期复选项；
- 逐条复核治理相关 active directives，只有证据齐全且用户明确允许时才
  append-only 标记 completed/superseded；
- 将“重构实施方案”转为稳定维护/历史参考，并保留回退入口；
- 仅将真实使用、schema、测试、示例和稳定版本均满足条件的 workflow 标为
  `standardized=true`，其余保持 active/non-standardized。

**用户需确认**

- 是否接受最终 D/I/O/S 矩阵；
- 哪些治理 directive 可以关闭；
- active plan 是转入 maintenance 还是 historical/reference；
- 哪些 workflow 达到 standardized 门槛。

**验收**

- 所有模块只有 `PASS`、有理由的 `N/A` 或明确 deferred，不保留含糊完成声明。
- directives、Document Registry、current state、Catalog 与生成视图一致。
- 全量测试、双轮幂等、严格门禁通过，无新增 WARN。
- 最终独立 `pfmval-audit` 给出唯一 GO/CONDITIONAL GO/NO-GO。

**阻塞退出**

任一先决 Goal 无回执、真实 O 证据缺失、directive 仍有有效未来义务或用户未
批准生命周期转换时，不关闭计划、不强制 standardized。

**新对话 Goal 文本**

> 创建 Goal G14：基于 G09–G13 的已验收回执，对 workflow governance v3 做
> 最终运行效果审核和生命周期收口；更新 D/I/O/S 矩阵、计划、directive、
> Registry、Catalog 与生成视图。只关闭证据齐全且我明确批准的事项，只标准化
> 满足稳定门槛的 workflow，并交付一份最终独立审核包。

## 四、风险矩阵

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 用 fixture 冒充真实 O | 中 | 错误宣称重构完成 | G09/G10 必须使用用户指定真实输入 |
| 实验闭环验证变成模型搜索 | 中 | 消耗资源并污染科研结论 | G12 固定一次预算，失败即停 |
| Gitee 冲突导致覆盖/重跑 | 中 | 证据和次数失真 | exact commit、单写者、零 force push |
| 为验收 R8/R9 误删资产 | 低但高影响 | 不可恢复损失 | G13 精确 ID/路径、preview、保留清单 |
| 强制所有 workflow 标准化 | 中 | 虚假成熟度 | G14 逐项门槛，不满足则保持 0.1.0 |
| 历史 WARN 被误当重构失败 | 高 | 扩大任务范围 | 旧 envelope/条码问题另立专项，不阻塞治理收口 |

## 五、不计入本轮重构完成的后续专项

- 五个 legacy accepted result envelope 补全；
- 受保护 MPP 标准化标签重复/冲突的重新生成；
- 学习指南正文体系化与云端备份；
- 收到学长模板后的论文正式写作；
- 自动 repair watcher；
- 任意外部平台接入。

这些是数据债务或功能采用任务，不应为了把治理重构“做完”而混入 G09–G14。
