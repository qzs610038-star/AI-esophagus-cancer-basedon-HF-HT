# PFMval 治理分层与部署方案/学习指南联动设计

- Date: 2026-08-07
- Status: approved-design
- Decision bundle: H2 + L2 + T2 + M2
- Scope: local governance, state validation, document freshness, user-facing navigation
- Out of scope: server operation changes, training approval changes, result import policy changes, automatic document rewriting

## 1. 背景与问题

当前项目已经建立了较强的治理闭环：状态包、实验 Registry、文档 Registry、工作树 Registry、路径 Registry、结果包和关键合同哈希共同承担可追溯性与可审计性职责。相关约束可见于 `AGENTS.md`、`CURRENT_STATE.md`、`project_state/current_state.json`，以及治理总方案 `project_state/plans/workflow_governance_v3_20260726.md`。

本轮扫描确认了两个现实问题：

1. **哈希与一致性校验存在过宽覆盖。**
   当前 `scripts/pfmval_state.py` 会统一计算 `source_hashes`，再在 `validate_state()` 中按同一阻断级别进行比对。虽然 `document_registry` 已经排除了部分派生字段以避免自循环，但生成视图、导航视图和学习材料相关变化仍容易进入同一紧张链路。
2. **部署方案与学习指南缺少受治理的同步提醒机制。**
   仓库里已有大量人工交叉引用，但尚未形成“部署方案变化 → 对应学习指南进入待复核状态”的稳定机制。当前最接近的能力是 `scripts/pfmval_knowledge.py` 中的 `review_document_freshness()`，但 learning guide 路由仍停留在 `interface_reserved`，尚未真正接入 active 文档联动。

## 2. 设计目标

本设计的目标不是削弱治理，而是把治理从“广覆盖统一阻断”调整为“关键证据保持硬门禁，知识导航保持可见但柔性”。

### 2.1 目标

1. 保留实验与治理事实根的严格校验。
2. 将学习指南、导航页、生成视图等知识层文档从阻断链中分离。
3. 建立 active 部署方案与 active 学习指南之间的显式关系。
4. 当部署方案变化时，自动把相关学习指南标为 `review_due`，并以 WARN 暴露，而非直接 FAIL。
5. 尽量复用现有 Registry / freshness / view 体系，避免新建第二套文档治理系统。

### 2.2 非目标

1. 不自动改写学习指南正文。
2. 不要求一次性为所有历史文档补齐关系元数据。
3. 不改变结果包、checkpoint、manifest、critical contract 的硬校验政策。
4. 不改变服务器 Gitee-only 边界、训练批准边界、受保护资产边界。

## 3. 选定方案

本设计采用以下组合：

- **H2**：哈希校验分为 Evidence/Governance HARD 与 Knowledge/Navigation SOFT 两层。
- **L2**：部署方案与学习指南建立关联，并以 dependency fingerprints 驱动 freshness 复核。
- **T2**：联动提醒以 WARN 形式出现在检查与导航输出中，不直接阻断。
- **M2**：Registry 作为权威；核心 active 文档允许逐步增加轻量头部元数据，但不要求一次性清洗全部旧文档。

## 4. 架构设计

### 4.1 两层治理模型

#### A. Evidence / Governance Layer（HARD）

该层继续承担阻断性校验，只覆盖会影响实验结论、治理身份、输入输出绑定和审计可信度的对象：

- `experiments/experiment_registry.json`
- `project_state/document_registry.json` 中 `lifecycle=active` 且 `authority=normative` 的文档
- `project_state/workspace_registry.json`
- `project_state/mpp_repair_registry.json`
- `configs/server_paths.yaml`
- `mpp_standard_splits/path_index.json`
- 实验 `critical_contract_sha256`
- split / entrypoint / checkpoint / result bundle / large artifact 的哈希与绑定

这些对象继续在 `start-check --strict`、result import、experiment approval、attempt prepare 等关键链路中保持 HARD 语义。

#### B. Knowledge / Navigation Layer（SOFT）

该层保留可追溯性与 freshness 监控，但默认不触发阻断：

- `CURRENT_STATE.md`
- README 内嵌 state block
- `PROJECT_GUIDE.md`
- active 学习指南的 freshness
- active 部署方案与 active 学习指南之间的联动状态
- 其他导航性、说明性、教学性文档的关系更新提醒

该层以 `WARN`、`review_due`、导航视图标注为主，不成为实验/治理主闭环的统一失败原因。

### 4.2 文档关系模型

在 `project_state/document_registry.json` 中为 active 文档增加轻量关系字段，最小支持：

- `doc_role`: `deployment_plan` | `learning_guide` | 其他
- `related_docs`: 关联文档 `doc_id` 列表
- `source_refs`: 学习指南引用的上游文档 `doc_id` 列表
- `dependency_fingerprints`: `doc_id -> fingerprint` 映射
- `freshness`: `fresh` | `review_due`
- `freshness_reason`: 可选，供视图展示

其中：

- active 部署方案通常声明 `doc_role=deployment_plan`
- active 学习指南通常声明 `doc_role=learning_guide`
- 学习指南通过 `source_refs` 指向部署方案，通过 `dependency_fingerprints` 固化上次核验时的依赖指纹

### 4.3 指纹策略

本设计不要求学习指南直接承载重型证据哈希，而是承载**用于文档 freshness 的依赖指纹**。建议第一阶段仅支持：

1. 上游 active 部署方案的 `content_sha256`
2. 可选的关键源码文件存在性或当前 HEAD commit 指纹
3. 可选的 experiment / plan reference 指纹

依赖指纹的用途是判断“学习指南是否需要人工复核”，不是证明科研结果有效，因此它们属于 SOFT 层。

### 4.4 freshness 行为

当 `source_refs` 中任一上游部署方案的当前指纹与学习指南记录的 `dependency_fingerprints` 不一致时：

- 学习指南不自动改正文
- `freshness` 变为 `review_due`
- 记录 `changed_dependencies`
- 在检查命令和导航视图中输出 WARN

这一逻辑直接扩展现有 `scripts/pfmval_knowledge.py` 中的 `review_document_freshness()` 机制，而不是另造一套并行系统。

## 5. 命令与行为设计

### 5.1 `scripts/pfmval_state.py`

#### `compute_source_hashes()`

保留对事实根文件的哈希计算，但明确将 Knowledge/Navigation 层对象排除在 `source_hashes` 的 HARD 阻断责任之外。继续保持现有对 `document_registry` 派生字段的语义哈希消噪逻辑，并将其定位为“规范内容变化检测”，不是导航视图变化检测。

#### `validate_state()`

调整校验输出语义：

- 对 Evidence/Governance Layer 保持 FAIL。
- 对 Knowledge/Navigation Layer 改为 WARN 或通过专门 freshness 检查输出 `review_due`。
- `CURRENT_STATE.md`、README state block、`PROJECT_GUIDE.md` 不再作为统一 FAIL 入口；它们的失步最多触发 WARN，并建议重新生成。

#### 工作树漂移场景

`validate_governance_v3_state()` 中与 workspace HEAD 严格绑定相关的校验，继续在 experiment / training / import 上保持 HARD；但在纯知识治理或文档维护上下文中，应允许通过 host/task 语义降为 WARN，避免阻断治理扫描或学习材料维护。

### 5.2 `deploy/pfmval_ops.py`

`agent start-check --strict` 继续保留，但输出应分层：

- HARD failures: 影响实验和治理真实性的失败
- SOFT warnings: 学习指南待复核、导航视图失步、部署方案变更后知识材料未复核

必要时可新增或扩展一个知识侧 freshness 子命令，但第一阶段不强制引入新 CLI，只要现有输出能稳定表达即可。

### 5.3 `scripts/pfmval_knowledge.py`

扩展 `learning_guide` 相关逻辑，使其不再只停留在 `interface_reserved` 演示层，而能读取 active 文档关系并计算 freshness。核心行为：

- 根据 `source_refs` 找到上游部署方案
- 生成当前依赖指纹
- 与 `dependency_fingerprints` 比较
- 输出 `fresh` 或 `review_due`
- 返回 `changed_dependencies`

该扩展仍不直接写正文，不关闭 directive，不自动改变文档 lifecycle。

### 5.4 `scripts/pfmval_views.py`

在用户导航中增加轻量可见性：

- active 部署方案列表
- active 学习指南列表
- `review_due` 的学习指南及其触发来源

这样即使用户不跑深层检查，也能在导航视图中发现“哪份指南应该更新了”。

## 6. 文档元数据落点

本设计采用 M2：**Registry 为主，文档本体轻量补充**。

### 6.1 Registry 作为权威

权威关系、freshness、source refs、dependency fingerprints 均以 `project_state/document_registry.json` 为准。

### 6.2 文档本体轻量补充

对少量核心 active 部署方案/学习指南，可逐步增加头部信息或 frontmatter，例如：

- `related_docs`
- `last_verified_commit`
- `audience`

但这不是第一阶段的硬前提，不要求一次性重写所有旧文档。

## 7. 影响文件

### 7.1 代码与 schema

1. `scripts/pfmval_state.py`
2. `deploy/pfmval_ops.py`
3. `scripts/pfmval_knowledge.py`
4. `scripts/pfmval_views.py`
5. `scripts/update_doc_registry.py`
6. `project_state/schemas/document_registry.schema.json`
7. `project_state/schemas/document_profile.schema.json`

### 7.2 数据与文档

1. `project_state/document_registry.json`
2. `PROJECT_GUIDE.md`（生成）
3. `CURRENT_STATE.md`（生成，但不再作为知识层失步的统一 FAIL）
4. 核心 active 部署方案/学习指南的少量头部元数据（可选、渐进）

### 7.3 测试

1. `tests/test_pfmval_state.py`
2. `tests/test_pfmval_knowledge_documents.py`
3. `tests/test_user_project_guide.py`
4. `tests/test_pfmval_governance_skill.py`

必要时增加针对以下行为的新测试：

- deployment plan 变化触发 learning guide `review_due`
- navigation/derived view 失步只产生 WARN
- normative active 文档 hash 失步仍 FAIL
- 纯知识治理上下文不因 workspace HEAD 漂移而统一阻断

## 8. 分阶段实施建议

### Phase 1：分层与 schema 打底

- 为 document registry 增加 `doc_role`、`source_refs`、`dependency_fingerprints`、`freshness` 等字段
- 调整 `scripts/update_doc_registry.py` 与 schema
- 保持旧数据兼容，允许缺省字段

### Phase 2：freshness 计算与 WARN 暴露

- 扩展 `scripts/pfmval_knowledge.py`
- 在 `scripts/pfmval_state.py` / `deploy/pfmval_ops.py` 中引入 knowledge-layer WARN
- 补导航视图展示

### Phase 3：核心文档接入

- 先为 active 部署方案与 active 学习指南建立第一批关系
- 只覆盖当前 active 文档，不追溯清洗全历史
- 观察 `review_due` 噪音是否可接受，再决定是否扩大覆盖面

## 9. 风险与缓解

### 风险 1：依赖指纹过粗，导致频繁 review_due

缓解：第一阶段只绑定文档 `content_sha256` 与少量必要指纹，不把无关导航变化纳入学习指南依赖。

### 风险 2：依赖指纹过细，漏掉真正重要的学习材料过期

缓解：对核心部署方案先人工指定 `source_refs`，逐步总结稳定规则。

### 风险 3：WARN 可见但被忽略

缓解：同时在 start-check 输出与 `PROJECT_GUIDE.md` 导航中暴露 `review_due`，减少其只存在于深层命令里的风险。

### 风险 4：旧文档改造成本被放大

缓解：仅覆盖 active 文档；历史、superseded、reference-heavy 旧文档不作为第一阶段必改对象。

## 10. 验收标准

### 10.1 HARD 层验收

1. active normative 文档 hash 失步仍 FAIL。
2. result / checkpoint / manifest / contract 绑定仍 FAIL on mismatch。
3. workspace / server path / repaired manifest 的关键治理边界不削弱。

### 10.2 SOFT 层验收

1. `CURRENT_STATE.md` / README state block / `PROJECT_GUIDE.md` 失步不再统一导致 FAIL。
2. 部署方案变化后，对应 active 学习指南会被标记为 `review_due`。
3. `review_due` 在检查输出或导航视图中以 WARN 形式可见。
4. 学习指南不会被自动改写。

### 10.3 用户体验验收

1. 用户能清楚区分“实验真实性失败”和“知识材料待复核”。
2. 日常治理扫描不会因为纯文档导航失步而被过度阻断。
3. active 部署方案与 active 学习指南之间的关系可查、可追踪、可逐步维护。

## 11. 结论

本设计通过 H2 + L2 + T2 + M2 组合，将当前项目从“统一高压一致性校验”调整为“关键证据强校验、知识材料软联动提醒”的分层治理模式。它保留了实验与治理核心闭环，同时为 active 部署方案与 active 学习指南建立了一条受治理但不过度自动化的同步路径：**上游方案变化会触发下游学习指南待复核，但不会未经确认自动改写或阻断主流程。**
