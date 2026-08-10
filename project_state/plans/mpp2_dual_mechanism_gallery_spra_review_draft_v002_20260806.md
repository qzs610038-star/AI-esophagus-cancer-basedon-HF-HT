# MPP2 双机制独立改进完整方案 v2：Gallery 跨域对齐 + SPRA 结构化空间建模

> 文档状态：`review_draft_v2`（待用户审核）  
> 证据状态：`NOT RUN / pending_review`  
> 审核层面：本稿替代 2026-08-05 的单轴 SPRA 草稿，但不删除、不改写旧稿；未通过 document lifecycle 正式激活  
> 建议候选工作树：`W005`（仅建议；尚未创建、登记或占用）  
> W004 边界：只读核验；不修改代码、配置、分支、Registry、输入或结果，不向 W004 注入新 MPP2  
> 编制日期：2026-08-06

> **弃用补充（2026-08-10，`DIR-20260810-001`）**：用户已决定不再使用 W005 在服务器执行本方案训练。
> W005 编号永久保留且不得复用；现有 `codex/w005-mpp2-gspra-20260806-bound` 分支、Gitee ref 与本地
> worktree 仅作只读历史候选保留，不构成 experiment 登记、服务器排障或训练入口。后续新实验必须重新登记
> experiment，并使用 W006 或更高的新永久编号。下文所有“建议 W005”与未来执行表述均已失效，仅保留作历史方案参考。

## 1. 总体裁决

建议将原“单一 SPRA 路线”改为两个正交机制轴：

1. **G 轴：Training-Gallery 跨域对齐**——解决患者/切片间形态域、预测中心、尺度和截距偏移；
2. **S 轴：SPRA 结构化空间建模**——解决普通 30D MLP 未显式利用通路身份、通路共享结构与真实二维空间上下文的问题。

推荐工作名称：

> **G-SPRA：Gallery-aligned Spatial–Pathway Residual Adapter**  
> 中文：**图库对齐的空间—通路结构化残差适配器**

但 Gallery、通路 identity token、零初始化 residual 和简单 kNN 空间平滑已经在 W004 中存在实现或方案，因此它们在本路线中只能作为**共享机制、低成本探针或对照**，不能单独作为论文创新点。Phase 2 独立线的核心新增价值应限定为：

- 直接优化真实 30 通路回归，而不是 pCR/MPR 分类；
- 用患者级 LOPO 检验跨患者泛化，而不是 W004 的切片级 repeated holdout；
- 将 Gallery 与结构残差拆成可归因的 2×2 机制实验；
- 在独立工作树中同时完整实现 W004 尚未实现的 TopoSSM 与 masked 2D spectral field，并在预登记内部批次中分别评估；空间诊断用于解释结果，不在批次中途删减队列。

当前执行裁决：**NO-GO**。本文仍未创建/绑定 W005、登记 experiment、生成批量批准文件、启动训练、操作服务器、执行 XZY 或接纳结果。`DIR-20260806-002` 已授权未来新对话在用户显式指定并核验对应工作树后，按本方案一次性完成全部代码与本地测试，并经 Gitee 开展服务器 dry-run/兼容排障准备；正式训练仍须使用绑定全部 `job_id + attempt + source_commit + budget` 的批量批准文件。当前 strict start-check 仍因 W004 live HEAD 与 Registry 绑定不一致而 `FAIL`。

提升性门控规则（`DIR-20260806-001`）：本方案所有预测性能、跨患者稳定性、模型比较、候选晋级、空间机制选择和分支退出阈值均为 **WARN-only**。未达到预设提升时，只报告指标、证据缺口、风险、建议和继续实验的可能后果；不得自动停止、拒绝、关闭、回退或晋级，下一步由用户显式决定。该规则不降低数据完整性、防泄漏、训练折拟合、W### 绑定、正式训练/新增 attempt 批准、次数预算、W004 隔离、Gitee 传输、结果 import/accept 与证据等级等治理硬约束。

批处理执行规则（`DIR-20260806-002`）：内部实验采用“前置一次性工程闭环 → Gitee 服务器排障 → 冻结 commit/批量批准 → 逐个顺序执行 → 全部回传 → 统一科学分析”。单个内部 arm 的提升性 WARN 不得改变预登记队列；只有治理、数据、泄漏、结果合同或基础设施 HARD 才能暂停。每个作业回传后立即做包完整性/校验值验证，但在预登记内部批次全部到齐前不作模型淘汰、晋级或追加决策。

## 2. 两条机制轴回答什么问题

### 2.1 G 轴：跨域对齐

假设：现有 MPP2 的主要误差包含患者/切片级偏移。训练折中与待预测 patch 形态相似的参考 patch，其已知预测残差可用于对新患者做受限修正。

对查询 patch (i)：

\[
\hat y_i^{G}=\hat y_i^{base}+\lambda_G
\frac{1}{K}\sum_{j\in\mathcal N_G(i)}(y_j-\hat y_j^{base})
\]

- \(\mathcal N_G(i)\)：仅从训练患者 gallery 中按 morphology feature 检索的 K 个邻居；
- \(y_j-\hat y_j^{base}\)：训练折残差；
- \(\lambda_G\)：修正幅度，必须在内部训练/验证层确定；
- 目标患者真实 ssGSEA 标签不能参与 gallery、近邻、修正或选择。

该分支是 CHRep 启发的训练图库后校准探针，不是完整 CHRep 复现。CHRep 本身还包含结构感知表示学习、图像—表达对齐、坐标拓扑正则和受约束校准模块。

### 2.2 S 轴：通路结构与空间场

假设：即使患者级中心/尺度已对齐，当前 MLP 仍可能因为缺少通路共享结构和跨 patch 上下文而留下结构化误差。

第一层只增加低秩 pathway-query residual：

\[
u_i=W_hh_i,\qquad
\Delta y_{ik}=q_k^\top u_i+b_k
\]

\[
\hat y_i^{S}=\hat y_i^{base}+\alpha g_i\odot\Delta y_i
\]

- \(h_i\)：frozen UNI2-h 1536D CLS；
- \(q_k\)：第 k 条通路的查询向量；
- \(\alpha\)：零初始化 residual 系数；
- \(g_i\)：置信门控；
- residual 关闭或 \(\alpha=0\) 时，输出严格等于该 fold 的基线。

第二层只有在残差诊断通过后，才将同切片 patch 场送入 TopoSSM 或 masked 2D spectral residual。

### 2.3 G+S 组合回答互补性

组合模型：

\[
\hat y_i^{GS}=\mathcal G_{train}\bigl(\hat y_i^{S},h_i\bigr)
\]

其中 \(\mathcal G_{train}\) 是只由训练患者拟合并冻结的 gallery 修正。必须先分别证明 G-only 和 S-only，再测试组合；不得直接以组合胜出掩盖单个机制无效。

## 3. accepted 锚点与新协议边界

### 3.1 工程锚点

当前 accepted 锚点保持：

| 字段 | 值 |
|---|---|
| experiment id | `mpp2_barcode_repair_v003_frozen_baseline_20260711` |
| model | frozen UNI2-h CLS + `1536→1024→30` MLP |
| manifest | `barcode-repair-20260711-d626ad8-v003:1204018178a4d355` |
| internal PCC | `0.7971` |
| XZY pooled PCC | `0.6549` |
| XZY raw MAE | `1176.2114` |
| XZY raw R² | `-0.0880` |

accepted checkpoint 只用于：身份/hash 审计、输出 parity 和最终外部比较锚点；不得直接用于开发患者 LOPO 科学比较。

### 3.2 为什么 LOPO 不能直接冻结 accepted checkpoint

accepted checkpoint 已利用当前开发患者训练。若在“留一患者验证”中继续使用它，held-out 患者可能已进入其训练历史，构成协议泄漏。

因此新 LOPO 中必须为每个 outer fold：

1. 保持 UNI2-h encoder 冻结；
2. 仅用该 fold 的训练患者重新拟合同结构 MLP；
3. 在训练患者内部确定 checkpoint 和所有 G/S 参数；
4. held-out 患者只 transform/evaluate。

该 fold-specific MLP 是 `protocol-matched control`，不是 accepted 结果的数值复现，也不替代 accepted checkpoint。

## 4. 数据、划分与身份合同

### 4.1 开发和外部边界

- 开发域：当前 MPP2 的 6 名非 XZY 训练患者；
- outer evaluation：6-fold patient-level LOPO；
- XZY：冻结外部集，所有架构、K、rank、频带、扫描、epoch 和阈值锁定后才允许一次性评估；
- 若未来使用 XZY 少量 ST 真值锚点，必须另立 `target-patient few-shot` 协议，不能纳入零样本主结果。

### 4.2 每个 batch/slide 必须保留

```text
features: [N,1536]
targets: [N,30]
coords: [N,2]
valid_mask: [N]
patient_id, slide_id, sample_id, patch_stem
pathway_names: 长度 30，顺序与 SHA-256 固定
```

### 4.3 硬失败

- 重复 `patient_id+patch_stem`；
- patient 跨 outer train/test；
- gallery 包含 held-out 或 XZY 行；
- 缺坐标、坐标与 feature/target 行错位；
- 30 pathway 名称或顺序漂移；
- z-score 不是在 outer training patients 上拟合；
- 证据性运行使用 `allow_missing=True`；
- raw prediction 缺少 `sample_id, spatial_cluster_id, pathway_id, y_true, y_pred`。

## 5. 模块定义与对照

### 5.1 B0：fold-specific baseline

每个 outer fold 训练一个与 accepted MLP 同结构的控制模型。所有候选共用相同 UNI2-h cache、训练患者、内部验证、epoch budget、optimizer 和 checkpoint 规则。

### 5.2 G0：global residual control

只用训练折的逐通路平均残差修正：

\[
\hat y^{G0}=\hat y^{base}+\operatorname{mean}_{train}(y-\hat y^{base})
\]

它回答：KNN gallery 是否真的需要 morphology-conditioned reference，而不是普通截距修正。

### 5.3 G1：KNN training-gallery

- reference space：预注册为 frozen UNI2-h CLS；
- K：首轮只允许一个值，建议 `K=5`；
- distance：标准化后的 Euclidean 或 cosine 二选一并预注册；
- residual magnitude：设置 \(\lambda_G\) 或范数上限，避免远邻导致过校正；
- negative controls：reference row shuffle、residual row shuffle、随机等容量 gallery。

### 5.4 S0：generic equal-parameter residual

与 pathway-query residual 参数量尽量匹配，但不使用 30 pathway identity。它回答改进是否只是增加容量。

### 5.5 S1：pathway-query residual

- rank 首轮固定建议 `r=32`；
- fold-specific baseline 先训练并冻结；
- 只训练 residual branch；
- final projection 或 \(\alpha\) 零初始化；
- 必须提供 residual-off exact parity 测试；
- pathway identity permutation 是必做反事实。

注意：W004 已实现 30-pathway identity-token residual，因此 S1 本身不能作为独立新颖性主张。它在本路线中的作用是通往 Phase 2 空间场模型的最小结构基线。

### 5.6 S2A：TopoSSM 候选

只有 frozen residual 显示跨患者稳定的长程、方向性或非局部空间相关时启用：

1. 真实坐标构图；
2. 固定的多方向/图遍历扫描；
3. 正反向共享权重轻量 SSM；
4. 局部 KNN 聚合补充短程关系；
5. 坐标打乱、扫描顺序打乱和 simple KNN/GCN 等参数对照。

禁止把文件枚举顺序当作组织序列，禁止直接搬用 MambaIR。

### 5.7 S2B：masked 2D spectral 候选

只有 frozen residual 的二维频谱存在跨患者稳定频带时启用：

1. 按坐标 scatter 到稀疏网格；
2. tissue mask 与值分离，缺失位置不能当真实零值；
3. masked 2D FFT/DCT token mixing；
4. gather 回原 patch 顺序；
5. 固定 Gaussian、depthwise 2D conv、随机 frequency mask 和 coordinate shuffle 对照。

旧 `model_gfnet.py` 沿 token 序列做 1D rFFT，不是本假设的已验证实现。

### 5.8 S2 双候选前置实现、批后比较

TopoSSM 与 masked 2D spectral 均须在服务器正式运行前完成代码、配置、单元测试、显存 dry-run 和反事实接口。内部批次分别执行 S2A 与 S2B；诊断结果只用于统一分析时解释其适配性：

- 长程/方向性拓扑证据更稳定：支持 S2A 的机制解释；
- 频带证据更稳定：支持 S2B 的机制解释；
- 两者均弱：在批后触发“空间结构证据不足” WARN；
- 两者相近：批后同时报告效果、参数量、显存、时延和代码复杂度，由用户决定确认对象。

## 6. 分阶段实验矩阵

### Stage E：工作树内一次性完成全部工程，0 次训练

新对话必须先由用户指定永久工作树（当前候选 W005），核验 experiment/path/branch/HEAD/dirty/Registry 绑定后，才允许写代码。不得只实现下一 arm；正式服务器排障前必须一次性完成：B0、G0/G1/Gcf、S0/S1/Sid、S2A TopoSSM、S2B masked 2D spectral、Seq、Scf、2×2 组合、指标统计、统一分析脚本、job 配置生成器和 result bundle 打包/验证器。

| 编号 | 检查 | 通过条件 |
|---|---|---|
| E0 | accepted checkpoint/manifest/hash | 与 Registry 一致 |
| E1 | 数据身份、坐标、通路顺序 | 0 泄漏、0 冲突、0 未解释缺失 |
| E2 | accepted parity | residual off 逐元素一致 |
| E3 | fold-specific z-score/fit scope | outer test 只 transform |
| E4 | W004 防串线 | W004 path/branch/id 出现即失败 |
| E5 | prediction contract | 五列长表 + provenance/hash |
| E6 | 8GB GPU dry-run | 无 NaN/Inf/OOM；不改变 batch 证据单位 |
| E7 | 全 arm 配置冻结 | 每个 arm/fold 有唯一 config hash、job template 和预期输出 |
| E8 | 批量本地测试 | 单测、集成、synthetic smoke、失败注入和打包回读全部通过 |

### Stage F：Gitee 服务器排障与冻结

1. 本地工作树是唯一代码写入源；完成测试后提交精确 `source_commit` 并经 Gitee 下发；
2. 服务器只执行 allowlisted dry-run、真实资产 preflight、依赖/路径/显存兼容检查和最小非证据性 forward，不产生可比较训练结论；
3. 服务器问题通过 Gitee 回传诊断、日志或 patch 建议，由本地工作树修改、复测、提交新 commit，再经 Gitee 下发；禁止两端并行改同一分支；
4. 直到服务器 preflight、结果打包回读和全部 HARD 检查通过，才冻结最终 `source_commit`；
5. 为全部内部 arms 生成一份批量批准清单，逐项列出 `job_id、attempt_id、arm、fold、顺序、预算、source_commit、input/output hash`；该批准文件缺失时不得训练。

### Stage D0：只读失效模式诊断

在可追溯、身份完整的开发预测上计算：

- 每患者/每切片 residual mean 与 scale；
- morphology distance 与 residual similarity 的关系；
- residual Moran's I、半变异函数、距离分层误差；
- masked 2D 频谱能量及其跨患者稳定性；
- 通路 residual covariance 与有效秩。

若原始预测缺少稳定样本身份，本阶段为 `GAP`，禁止按 DataLoader 顺序或文件枚举顺序补身份。

### 内部批次执行顺序

冻结并批准后，服务器按 `D0 → D1 → D2 → D3 → D4` 顺序逐个执行。前一作业完成后只检查进程退出、数据/结果合同和包完整性；性能数值不用于暂停、删减、改序或追加队列。所有结果经 Gitee 回传到 inbox，批次齐套后统一 import/比较。若某作业触发 HARD，暂停后续队列并先完成治理修复；修复不自动获得重试资格。

### Stage D1：B/G 轴筛选

每个 outer fold 只训练一次 B0；G0/G1 是其训练折拟合的 post-hoc transform，不新增神经网络训练 attempt。

| Arm | 内容 | 训练 attempt |
|---|---|---:|
| B0 | fold-specific MLP | 6 |
| G0 | global residual | 0 |
| G1 | KNN training-gallery | 0 |
| Gcf | gallery/residual shuffle | 0 |

G1 的预期门控（仅在内部批次全部回传后统一判断）：

1. 患者平衡 raw MAE 相对 B0 改善至少 `2%`；
2. pooled PCC `Δ >= -0.005`；
3. 至少 `4/6` held-out patients 同方向；
4. G1 优于 G0，证明 morphology-conditioned gallery 超过普通截距修正；
5. shuffle control 失去增益；
6. XZY 未参与任何选择。

未通过时记录 WARN/阴性结果；不影响本批次已预登记的 S 轴执行。

### Stage D2：S1 通路结构筛选

在相同 6 folds、seed=42 下：

| Arm | 内容 | 新增训练 attempt |
|---|---|---:|
| S0 | generic equal-parameter residual | 6 |
| S1 | pathway-query residual | 6 |
| Sid | pathway identity permutation | 可作为固定模型反事实；若需重训另审 |

S1 的预期门控（仅在内部批次全部回传后统一判断）：

1. patient-balanced raw MAE 相对 B0 改善至少 `2%`；
2. pooled PCC 非劣；
3. 至少 `4/6` patients 同方向；
4. 至少 `18/30` pathways raw MAE 不劣；
5. S1 优于 S0，或 pathway identity 破坏后增益消失；
6. 改善不能完全由患者级均值修正解释。

### Stage D3：空间分支

运行两个已前置完成的空间模块和一个共享等参数/复杂度对照：

| Arm | 内容 | 新增训练 attempt |
|---|---|---:|
| S2A | TopoSSM | 6 |
| S2B | masked 2D spectral | 6 |
| Seq | simple KNN/GCN 或 Gaussian/Conv 等共享参数对照 | 6 |
| Scf | 坐标/扫描/频率破坏 | 首选冻结推理反事实；需要重训时单独审核 |

若 S2A/S2B 仅达到 raw MAE/PCC 同等、空间指标未改善或反事实对照未破坏增益，则在批后统一分析中触发机制证据不足 WARN；不得据中途结果取消另一空间 arm。

### Stage D4：2×2 机制归因

对 S1、S2A、S2B 的冻结输出分别运行 Gallery on/off 组合；G0/G1 为训练折 post-hoc transform 时原则上不新增神经网络训练：

| 结构轴 | Gallery off | Gallery on |
|---|---|---|
| S off | B0 | G1 |
| S on | S* | GS* |

以 benefit \(B=-\mathrm{rawMAE}\) 定义交互：

\[
I_{GS}=B_{GS}-B_G-B_S+B_{B0}
\]

- \(I_{GS}\approx0\)：近似可加；
- \(I_{GS}>0\)：互补协同；
- \(I_{GS}<0\)：两种修正可能重复或相互破坏。

必须同时报告 G 和 S 的主效应，不能只展示 GS 最佳数值。

### Stage C：确认和 XZY 一次性外部评估

Stage C 是内部批次统一分析和用户选择之后的**独立第二批**，不得与内部候选全量并行执行：

1. 用户根据完整内部结果选择确认候选，并单独批准确认批次；
2. 用预注册额外 seeds 对 B0 与最终候选成对确认；
3. 在 6 名开发患者上按冻结协议 refit；
4. 固定代码、配置、checkpoint、gallery、路径顺序和 hashes；
5. 单独申请一次 XZY 评估；
6. XZY 不得反向选择 G/S、K、rank、SSM 扫描、频带或 checkpoint，也不得对全部内部候选逐一评估。

建议外部成功条件：

**绝对性能成功**：

- XZY raw MAE 相对 `1176.2114` 改善至少 `3%`；
- PCC 相对 `0.6549` 的 `Δ >= -0.005`；
- raw R² 不低于 `-0.0880`；
- 不由少数 pathway 或局部区域独占。

**性能保持且机制增强**：

- raw MAE 恶化不超过 `1%`，PCC `Δ >= -0.005`；
- 空间真实性指标预注册改善；
- gallery/identity/coordinate/frequency 破坏对照按预期失效；
- 参数量、显存、时延和训练成本完整报告。

## 7. 指标与统计合同

### 7.1 一级指标

- patient-balanced raw MAE；
- pooled PCC 与 macro-patient PCC；
- patient-balanced raw R²（解释性指标，不单独选模）。

### 7.2 机制指标

- G 轴：residual mean/scale、邻居距离—修正误差关系、G1-G0 差值；
- S1：逐 pathway 改进宽度、residual covariance 有效秩；
- S2：truth/residual Moran's I、半变异函数、热点重合、频带能量恢复；
- GS：2×2 主效应和 interaction \(I_{GS}\)。

### 7.3 稳定性与成本

- fold/seed 方向计数和 bootstrap CI；
- 30 pathway 全量结果，不筛最好子集；
- 参数量、峰值显存、训练和推理时延；
- 所有 unsuccessful/negative arms 保留并写停止原因。

## 8. 实现边界

建议在正式 W005 中新增独立包：

```text
phase2_gspra/
  dataset.py
  baseline.py
  gallery.py
  pathway_decoder.py
  topology.py
  spectral.py
  model.py
  diagnostics.py
  metrics.py
  analysis.py
  job_matrix.py
  result_bundle.py
train_mpp2_gspra.py
server_preflight_mpp2_gspra.py
tests/test_mpp2_gspra_*.py
```

禁止事项：

- 不从 W004 运行时 import `phase3_pipeline`；
- 不复制后再修改 W004 文件；
- 不向 W004 cherry-pick；
- 不共享可写 output、checkpoint、cache lock 或 job id；
- 不把 W004 synthetic smoke 当作 Phase 2 模块有效证据。

可复用的是**接口思想和对照定义**。若以后确需抽取公共库，应另立治理重构任务，在 W004 与 W005 协议冻结后实施，不能在本实验中顺手重构。

## 9. 与当前 W004 的重叠审计

### 9.1 当前 W004 新鲜事实

只读核验时：

- live branch：`codex/w004-phase3-dual-baseline-transfer-20260804-bound`；
- live HEAD：`5e89ca909377e9a9cfa43d63f77189ca613df59c`；
- 工作树干净；
- 配置为 `server_preflight_only_no_training`；
- `real_training=false`、`job_creation=false`、`result_import=false`；
- pathway route 为 `legacy_zscore_compatibility`，证据 `pending_review`，`raw_mpp2_claim=false`；
- 当前实现已包含 gallery、空间平滑、四视图通路 residual 和 pathology-pathway residual。

因此“W004 已写代码”是 current implementation 事实；“这些模块在真实 Phase 3 上有效”仍是 `NOT RUN`。

### 9.2 重叠矩阵

| 对象 | W004 当前内容 | 新 Phase 2 内容 | 重叠等级 | 裁决 |
|---|---|---|---:|---|
| accepted MPP2 30D | 作为 Phase 3 输入依赖 | 作为工程锚点和最终比较 | 中 | 共享只读事实源，不共享可写产物 |
| training-gallery | 已有 global/KNN residual calibrator | G0/G1 跨患者回归校准 | 高 | 算法机制重叠；在新稿中降为探针/对照，不宣称独创 |
| 空间 kNN 平滑 | 已有 `spatial_smooth` 和坐标打乱 | S2 的 simple comparator | 高 | 直接作为对照，不能作为新路线核心 |
| pathway identity token | 已有四视图 identity-token residual | S1 pathway-query residual | 中高 | 概念重叠；输出端和训练目标不同，仍不能单独主张创新 |
| 零初始化 residual | 已用于 Phase 3 30D refinement 和 H&E feature injection | 用于 Phase 2 预测 residual | 高 | 共享安全设计模式，不是创新点 |
| 真实 SpT prototype/LOPO | W004 训练折 prototype 与 LOPO diagnostic | 新路线主要做预测误差和泛化 | 中 | 数据来源与部分方法重叠，科学终点不同 |
| aligned patch+coords | W004 `PatchBag` 含 pathology/pathway/coordinates | 新 dataset 还需 true pathway 和 LOPO identity | 中 | 可借鉴合同，不运行时依赖 W004 |
| TopoSSM | 未实现 | S2A 候选 | 无/低 | 新 Phase 2 独立方向 |
| masked 2D spectral field | 未实现 | S2B 候选 | 无/低 | 新 Phase 2 独立方向 |
| 主要任务 | pCR/MPR 切片分类 | 30 pathway patch/空间回归 | 低 | 不同科学终点 |
| 主要指标 | AUC/AUPRC/校准/患者聚类统计 | raw MAE/PCC/raw R²/空间真实性 | 低 | 不同决策指标 |
| 数据划分 | 307 slides、4 seeds×5 folds repeated holdout，患者重叠需 WARN | 6 development patients LOPO + XZY one-shot | 低 | 不得跨协议排名 |
| Phase 3 迁移 | W004 正在准备该问题 | 若把新 MPP2 接入 Phase 3 | 极高/冲突 | 当前禁止；未来另立实验，不回填 W004 |

### 9.3 总体重叠结论

- **代码/机制层面：有明显重叠，等级中高。** Gallery、空间 kNN、通路 identity token 和零初始化 residual 已在 W004 实现。
- **科学问题层面：重叠较低。** W004 预测 pCR/MPR；新路线预测真实通路值并检验 Phase 2 跨患者泛化。
- **数据依赖层面：部分重叠。** 二者共享 accepted MPP2、UNI2-h、30 pathway 顺序和 Phase 2 reference evidence，但 cohort/split/endpoint 不同。
- **执行层面：在严格隔离下可做到不重叠。** 新 W005、独立分支、输出和 job；W004 优先。
- **论文创新层面：Gallery/SPRA 基础部件不能重复计为两项创新。** 新路线真正可独立主张的候选是 Phase 2 的患者级 2×2 机制归因，以及 TopoSSM/二维频域对真实通路空间场的增量。

## 10. 防止与 W004 互相污染

1. 新路线不得读取 W004 的 pCR/MPR label、fold、checkpoint 或 model selection 结果；
2. W004 不得读取 G-SPRA 的开发/XZY结果来选择当前 branch；
3. W004 继续使用其已锁定 pathway route，不因新路线产生候选而替换；
4. 新路线的 XZY 不参与 W004 的任何阈值或融合选择；
5. 两边共享的 30 pathway order/checkpoint/hash 必须只读引用同一权威记录；
6. GPU/服务器时间冲突时 W004 优先，新路线等待；
7. 新 Phase 2 若通过，只能在 W004 完成或冻结后提出独立 downstream migration experiment；
8. 未来比较必须固定同一个 Phase 3 主干和 split，成对比较旧 MPP2 与新 MPP2，但不能修改历史 W004 结果。

## 11. 治理硬门控与提升性 WARN

| Gate | 期望条件 | 未满足时的处理 | 类型 |
|---|---|---|---|
| G0 治理 | strict start-check 无 FAIL；新 experiment/W###/approval/source commit 齐全 | 不执行，先修复或取得相应治理授权 | HARD |
| G1 数据 | LOPO、identity、coords、train-only z-score 全通过 | 泄漏、错位或训练折污染时停止并修复 | HARD |
| G2 工程 | baseline parity、W004 防串线、result contract 通过 | 触碰 W004、无法复现输入合同或结果合同失败时停止 | HARD |
| G3 Gallery | G1 超过 B0/G0 且 shuffle 破坏增益 | 报告 Gallery 提升/机制证据不足 WARN，由用户决定是否继续 G、补证或停止 | WARN |
| G4 Pathway | S1 超过 B0/S0 且 identity 有机制证据 | 报告 pathway-query 提升/机制证据不足 WARN，由用户决定是否进入 S2 | WARN |
| G5 Spatial | D0 支持至少一个空间假设 | 两者均弱或均相近时报告 WARN，由用户决定停止、单开或双开 | WARN |
| G6 External | 全部冻结并获 XZY one-shot 批准 | 未冻结、未批准或存在 XZY 调参意图时不执行 | HARD |
| G7 Downstream | Phase 2 accepted + W004 冻结/完成 + 新迁移批准 | 不修改、不回填 W004，另立迁移实验 | HARD |

G3-G5 只在预登记内部批次全部回传后统一判读，不得在中途改变队列。其中 `WARN` 不消耗或自动扩张训练预算；用户在批后选择补证或追加实验时，仍需为新增 attempt 完成批准和绑定。

## 12. 运行预算建议

以下只是上限草案，不构成授权：

| 阶段 | 神经网络训练 attempt 上限 |
|---|---:|
| D1 B0 | 6 |
| D1 Gallery transforms | 0 |
| D2 S0 + S1 | 12 |
| D3 S2A + S2B + 共享等参数对照 | 18 |
| D4 2×2 组合 | 0 或仅冻结 transform |
| 额外 seeds 确认 | 另审；建议只确认 B0 与最终 winner |

首轮完整内部批次最大 `36` 次训练 attempt：B0 6 次、S0/S1 12 次、S2A/S2B/Seq 18 次；Gallery 与2×2组合优先使用冻结 post-hoc 推理。批量批准后按顺序全部执行，不因提升性 WARN 中止。增加重训型反事实、重试、额外 seed 或超出 36 次仍须用户显式决定并形成新批准；治理 HARD 未满足时不得执行。

## 13. 用户需要审核的决策

1. 是否接受从单轴 SPRA 改为 G/S 双机制方案；
2. 是否同意 Gallery、简单空间平滑、pathway token、零初始化 residual 只作为共享探针/对照，不重复计为创新；
3. 是否接受 Phase 2 独立新颖性核心限定为患者级机制归因 + TopoSSM/二维频域双候选批后比较；
4. 是否同意正式实施时使用新永久工作树（当前候选 W005）；
5. 已确定未来采用完整代码前置实现、Gitee 服务器排障、冻结 commit 后批量顺序执行；
6. 是否接受首轮完整内部批次最多 36 次训练 attempt，并以一份列举所有 job 的批量批准文件统一授权；
7. 是否确认新 MPP2 无论结果如何都不自动替换 W004。

在以上决策明确前，正确状态保持：`review_draft_v2 / NOT RUN / pending_review / NO-GO execution`。

## 14. 当前阻塞、WARN 与证据限制

- 2026-08-06 strict start-check：`PASS=5, WARN=6, FAIL=1`；FAIL 为 live workspace HEAD 与 Registry 不一致。
- W004 live HEAD `5e89ca9...`，Registry 仍是较早绑定；必须通过既有治理流程处理，不能在本方案中顺手修 Registry。
- W004 当前 pathway route 为 `legacy_zscore_compatibility / pending_review / raw_mpp2_claim=false`，不能写成已使用 accepted raw MPP2 完成真实训练。
- W004 的 Gallery/SPRA 类代码通过存在性和静态检查只能证明实现，真实 Phase 3训练仍 `NOT RUN`。
- accepted/历史 prediction envelope 可能缺失完整样本身份；D0 前需重新审计，不允许按行号恢复。
- 6 位开发患者使模型选择不稳定；patch 数不能替代患者有效样本量。
- TopoSSM/二维网格在 8GB GPU 上有显存风险，必须先进行 chunk/sparse-mask dry-run。

## 15. 可核验来源

- [当前状态](../../CURRENT_STATE.md)
- [机器可读 current state](../current_state.json)
- [实验 Registry](../../experiments/experiment_registry.json)
- [工作树 Registry](../workspace_registry.json)
- [W004 Phase 3 分阶段方案](Phase3_双线基线与跨队列空间通路迁移分阶段实验方案_20260803.md)
- [旧单轴 SPRA 审核稿](mpp2_spatial_pathway_residual_adapter_review_draft_20260805.md)
- [MPP2 训练实现](../../train_mpp_uni2h_mlp.py)
- [manifest dataset](../../dataset_mpp_manifest.py)
- [旧 1D GFNet](../../model_gfnet.py)
- [prediction artifact schema](../schemas/prediction_artifact_contract_v1.schema.json)
- [编号工作树协议](gitee_numbered_workspace_protocol_v001_20260726.md)
- [CHRep](https://arxiv.org/abs/2604.21573)
- [MamMIL topology-aware scanning](https://arxiv.org/abs/2403.05160)
- [SpaHGC cross-slice transfer](https://arxiv.org/abs/2603.22821)
- [AFNO](https://arxiv.org/abs/2111.13587)
