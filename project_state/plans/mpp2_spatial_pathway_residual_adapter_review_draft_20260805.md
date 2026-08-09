# MPP2 独立改进路线审核稿：空间—通路结构化残差适配器

> 文档状态：`review_draft`（待用户审核）  
> 证据状态：`NOT RUN`；本文不构成 experiment 登记、W### 绑定、训练批准、服务器作业或结果接纳  
> 建议候选工作树：`W005`（仅建议，尚未创建或占用；永久编号以正式登记时为准）  
> 与 Phase 3 的关系：W004 继续使用 accepted repaired frozen MPP2；本文不得修改 W004 的代码、分支、输入合同、实验 Registry 或结果  
> 编制日期：2026-08-05

## 1. 审核结论

建议新增一条**受限、串行门控、低于 W004 优先级**的 MPP2 上游改进线，方法暂名：

> **SPRA：Spatial–Pathway Residual Adapter（空间—通路结构化残差适配器）**

不建议直接复跑 MambaIR、ConvSSM、旧 GFNet，或开放 LoRA、dropout、rank、loss、学习率的自由搜索。推荐顺序为：

1. 固定 accepted MPP2 为不可替换锚点；
2. 先验证低秩 pathway-query residual（通路查询残差）；
3. 只有发现稳定空间结构化残差后，才二选一进入 TopoSSM 或 masked 2D spectral residual；
4. 候选模型必须先在开发患者上冻结，之后才允许一次性读取 XZY；
5. 即使 Phase 2 候选通过，也不得自动注入 W004；下游迁移需另立实验、工作树与用户批准。

整体裁决为：**方案可进入用户审核，但当前执行状态为 NO-GO**。阻塞项不是科学可行性，而是尚未取得用户对新实验、工作树、协议和运行预算的批准，且严格前置检查仍有一个工作树 HEAD/Registry 不一致的治理 FAIL。

## 2. 当前事实与证据边界

### 2.1 accepted 锚点

实验 Registry 中当前应继续使用的锚点为：

| 字段 | 当前值 |
|---|---|
| experiment id | `mpp2_barcode_repair_v003_frozen_baseline_20260711` |
| evidence | `accepted` |
| 模型 | frozen UNI2-h CLS 1536D + 两层 MLP + 30 pathway 输出 |
| 数据 manifest | `barcode-repair-20260711-d626ad8-v003:1204018178a4d355` |
| internal val PCC | `0.7971` |
| XZY pooled PCC | `0.6549` |
| XZY raw MAE | `1176.2114` |
| XZY raw R² | `-0.0880` |
| 当前动作 | `closed_use_accepted_repaired_result_for_reporting` |

这些数值只描述既有 accepted protocol，不代表新 patient-disjoint/LOPO 协议的数值基线。

### 2.2 已关闭或不能直接扩张的路线

- LoRA dropout=0.10：internal-first gate 未通过；XZY PCC `0.6438`、raw MAE `1210.0846`、raw R² `-0.1293`，不得据此继续调参。
- MSE vs Huber δ=1：已 `closed_no_retry`；不能把换 loss 重新包装为本路线核心。
- 旧 GFNet：代码在 token 序列维执行 1D `rFFT`，之后对 token 求均值；它没有检验真实组织坐标上的 masked 2D 空间频率建模。
- rejected `CalibratedMPP2`、XZY 驱动的校准或模型选择不得进入本路线。

### 2.3 本地代码给出的真实改造边界

1. [`MPPMLPHead`](../../train_mpp_uni2h_mlp.py) 是 `1536 → 1024 → 30` 的 Linear–GELU–Dropout–Linear 头。
2. [`ManifestMPPDataset`](../../dataset_mpp_manifest.py) 会将 `[265,1536]` token 缓存压成 CLS `[1536]`，且 `__getitem__` 只返回 `feature,target`；虽然 manifest 含 `patient,x,y,patch_stem`，这些身份和坐标没有进入当前训练 batch。
3. [`OnlineMPPModel`](../../model_mpp_uni2h_lora.py) 同样只取 UNI2-h CLS 后送入 accepted MLP。
4. [`model_gfnet.py`](../../model_gfnet.py) 的频域操作沿 token 序列维做 1D FFT，不等价于组织坐标网格上的 2D FFT/DCT。
5. 当前通用指标已覆盖 MSE、MAE、R²、PCC；raw-scale 反变换逻辑可复用，但本路线还需患者平衡、空间一致性和校准指标。

因此，首个必要改动不是 Mamba 层，而是**新建隔离的数据/batch 合同**。

## 3. 研究问题与非目标

### 3.1 主研究问题

在不改变 accepted MPP2、受保护 MPP 资产与 W004 输入的前提下，显式建模：

- 30 条通路之间的共享低秩结构；
- 同一空间转录组切片上 patch 的真实二维邻接；
- 可关闭、可归因的预测残差；

能否改善跨患者绝对预测，并降低空间结构化误差？

### 3.2 非目标

- 不追求“用了 Mamba/频域”本身的架构新颖度。
- 不重新开放 UNI2-h 全量微调或 LoRA 超参数搜索。
- 不修改或重新生成 MPP 原始 ssGSEA、标准划分、z-score 参数和 embargo 资产。
- 不使用 XZY 选择模块、阈值、epoch 或 checkpoint。
- 不在本路线中执行 Phase 3 训练，也不替换 W004 的 frozen MPP2 30D 输入。
- 不把代码可运行、单元测试通过或本地 explore 结果称为 accepted 实验证据。

## 4. W004 隔离合同

| 隔离项 | 强制规则 |
|---|---|
| 工作树 | 新路线仅可在正式登记后的新永久 W### 中实施；当前建议 W005，不得复用 W001–W004 |
| 分支 | 新建独立 `codex/w005-mpp2-spra-*` 候选分支；不得向 W004 branch 写入或 cherry-pick |
| 文件 | 不改 W004 路径下任何代码、配置、缓存、checkpoint、输出或未跟踪文件 |
| Registry | 新 experiment id；不得修改 `phase3_dual_baseline_spatial_pathway_transfer_v001_20260804` 的 protocol revision 或绑定 |
| 输入 | W004 始终保留 accepted frozen MPP2 raw 30D；新模型输出不得覆盖同名文件或特征路径 |
| 资源 | W004 运行优先；若 GPU、I/O、缓存锁或服务器工作树冲突，SPRA 自动等待，不抢占 |
| 下游 | Phase 2 通过后也只能提出单独迁移实验审核，不得回填或重写 W004 结果 |

建议新增一条自动防串线测试：trainer 启动时若解析到 `workspace_id=W004`、W004 branch、W004 绝对路径或 W004 experiment id，立即失败。

## 5. 目标架构

### 5.1 不可变基线

对第 (i) 个 patch：

\[
h_i=\mathrm{UNI2h}_{frozen}(x_i),\qquad
\hat y_i^{base}=f_{MLP}^{accepted}(h_i)
\]

其中 (h_i\in\mathbb{R}^{1536})，\(\hat y_i^{base}\in\mathbb{R}^{30}\)。accepted checkpoint 只读加载并固定哈希。

### 5.2 SPRA 统一输出

\[
\hat y_i=\hat y_i^{base}+\alpha\,g_i\odot\Delta y_i
\]

- `α`：零初始化的可训练残差系数，初始输出严格等于 baseline；
- `g_i`：范围为 `[0,1]` 的置信门控；
- `Δy_i`：候选结构模块产生的 30D 残差；
- `⊙`：逐通路相乘。

必须提供 `residual_enabled=false` 的硬关闭开关，并通过逐元素输出一致性测试。

### 5.3 第一优先：低秩 pathway-query residual

为 30 条通路建立查询向量 (q_k\in\mathbb{R}^{r})，共享投影得到：

\[
u_i=W_hh_i,\qquad
\Delta y_{ik}=q_k^\top u_i+b_k
\]

其中 `r` 是低秩维度。第一轮只允许预注册一个 `r`，建议 `r=32`；不得把 `r` 变成 sweep。

必须加入等参数 generic low-rank residual，回答增益是否来自“通路身份建模”，而非仅仅增加参数。

### 5.4 第二优先候选 A：TopoSSM

仅当开发集残差显示稳定空间自相关时启用。输入单位是**同一切片的 patch 场**，不是单个 patch 内的 265 个 UNI2-h token。

建议最小实现：

1. 按真实坐标构建 tissue graph/KNN；
2. 使用固定、可复现的多方向扫描（例如 x/y 正反向或图遍历），保留 mask；
3. 共享权重的轻量双向 SSM 聚合长程上下文；
4. 与局部 KNN mean/GCN 等参数对照；
5. 做坐标打乱与扫描顺序反事实检验。

禁止按文件枚举顺序直接输入 Mamba，也不建议搬用为低层图像恢复设计的 MambaIR 全模型。

### 5.5 第二优先候选 B：masked 2D spectral residual

仅当残差频谱存在跨患者稳定的频带结构时启用：

1. 按坐标将 patch 特征或 baseline pathway field scatter 到稀疏二维网格；
2. 同步构建 tissue mask，缺失位置不得按真实零值处理；
3. 在 mask-aware 网格上做 2D FFT/DCT 残差；
4. 输出重新 gather 到原 patch 顺序；
5. 与固定高斯平滑、depthwise 2D convolution、随机频率 mask 对照。

这条路线与旧 1D token-GFNet 是不同假设，文档中不得宣称“复现 GFNet/AFNO/FECAM”。

## 6. 新数据与软件接口

建议只在新工作树增加独立包，不修改 accepted trainer：

```text
phase2_spra/
  dataset.py          # 按 patient/slide 成组，保留 patch_stem、x、y、mask
  baseline.py         # 只读加载 accepted head/checkpoint，输出一致性校验
  pathway_decoder.py  # pathway-query 与 generic 等参数对照
  topology.py         # KNN、扫描及可选 TopoSSM
  spectral.py         # masked 2D scatter/FFT(DCT)/gather
  model.py            # 零初始化残差、门控、硬关闭开关
  metrics.py          # 患者平衡、raw、空间和校准指标
train_mpp2_spra.py
tests/test_mpp2_spra_*.py
```

新 dataset 每个样本至少返回：

```text
features: [N,1536]
targets: [N,30]
coords: [N,2]
valid_mask: [N]
patient_id, slide_id, sample_id, patch_stem
pathway_names: 长度 30 且顺序哈希固定
```

任何缺坐标、重复 `patient_id+patch_stem`、跨 split 身份重复、通路列漂移或 manifest/hash 不一致都应硬失败，不允许 `allow_missing=True` 进入证据性运行。

## 7. 串行实验矩阵与运行预算

### Stage E：仅工程实现，0 次训练

| 编号 | 内容 | 通过条件 |
|---|---|---|
| E0 | accepted checkpoint/manifest/hash 只读解析 | 与 Registry 一致 |
| E1 | 新 dataset 身份、坐标、顺序审计 | 0 重复、0 跨 split 泄漏、0 未解释缺失 |
| E2 | `α=0` 或 residual off 的输出 parity | 与 accepted baseline 逐元素一致；阈值预注册 |
| E3 | CPU 小张量前后向与 8GB GPU 内存 dry-run | 无 NaN/Inf；不访问 W004 |
| E4 | raw prediction contract | 至少包含 `sample_id, spatial_cluster_id, pathway_id, y_true, y_pred` |

Stage E 通过只证明接口与工程安全，不证明模型有效。

### Stage D1：通路结构筛选

采用开发患者级外层 LOPO（leave-one-patient-out，逐一留一患者验证）新协议；它不是 accepted patch-manifest 数值复现。XZY 全程封存。

| Arm | 模型 | 必要性 |
|---|---|---|
| D1-I0 | protocol-matched frozen/固定结构 baseline | 新协议成对控制 |
| D1-Ieq | generic 等参数 low-rank residual | 参数量反事实 |
| D1-I1 | pathway-query residual | 核心候选 |

建议首轮固定 seed=42，6 个 outer folds，共 `18` 次 attempt 上限。禁止失败后自动重试；基础设施失败是否重跑需单独审核。

### Gate D1：进入空间分支的建议条件

以下阈值为**待用户批准的预注册建议**：

1. I1 相对 I0 的患者平衡 raw MAE 至少改善 `2%`；
2. pooled PCC 非劣，`ΔPCC >= -0.005`；
3. 至少 `4/6` 外层患者方向一致；
4. 至少 `18/30` 通路 raw MAE 不劣；
5. I1 优于 Ieq，或有明确 pathway-specific 机制证据；
6. 不含任何 XZY 驱动选择。

若未通过：关闭 SPRA 主路线，不进入 Mamba/频域。若总性能未通过但残差空间结构强，只能记录为 `explore_only` 假设，不能自动扩张。

### Stage D2：空间模块二选一

先在 D1 冻结预测上做只读残差诊断：Moran's I、半变异函数、距离分层误差、2D 频谱能量稳定性。根据预注册规则只选择一个分支：

- 长程/方向性拓扑更稳定：选 I2 `I1 + TopoSSM`；
- 稳定频带结构更清晰：选 I3 `I1 + masked 2D spectral residual`；
- 两者均不稳定：停止，不凭架构偏好选择。

入选模块必须同时跑一个等参数空间对照和一个破坏坐标/频率的反事实对照。建议上限仍为 6 folds × 3 arms = `18` 次 attempt。

### Stage C：确认与一次性外部评估

只有 D1/D2 胜者先在开发患者上以预注册额外 seeds 做成对确认并冻结全部配置后，才可提出 XZY 一次性评估批准。XZY 不参与 checkpoint、branch、rank、频带、邻居数或门控阈值选择。

外部成功建议分为两类：

**绝对性能成功**：

- XZY raw MAE 相对 `1176.2114` 改善至少 `3%`；
- PCC 不劣：相对 `0.6549` 的 `ΔPCC >= -0.005`；
- raw R² 不低于 accepted baseline；
- 改善不是由少数通路或单一区域支配。

**性能保持且机制更强**：

- raw MAE 相对恶化不超过 `1%`，PCC `Δ >= -0.005`；
- 同时显著改善预注册空间真实性指标，且坐标/频率破坏对照失去增益；
- 参数量、显存、时延与收敛成本完整报告；
- 只能表述为“空间结构机制证据增强”，不能仅凭使用 Mamba/频域宣称创新成功。

## 8. 指标与统计合同

### 8.1 一级指标

- 患者平衡 raw MAE：每位患者先求均值，再跨患者平均；
- pooled PCC 与 macro-patient PCC 同时报；
- raw R² 同时报，但不得单独用于小样本模型选择。

### 8.2 二级指标

- 30 通路逐通路 PCC、raw MAE、raw R²；
- calibration slope/intercept（校准斜率/截距）；
- fold/seed 方差、改进方向计数和 bootstrap 置信区间；
- Moran's I 真值差、残差 Moran's I、邻域热点重合、距离分层误差；
- 参数量、峰值显存、训练/推理时延。

### 8.3 选择规则

- checkpoint 与架构只依据开发患者；
- 先确定单一主指标和非劣阈值，再运行；
- 不用 30 个 pathway 中的最佳子集回报总体结论；
- 不跨不同 split/protocol 直接按绝对数值排名；
- 所有 raw predictions 必须进入受治理结果包，而不是只给路径。

## 9. GO/STOP 门槛

| 门槛 | GO | STOP/回退 |
|---|---|---|
| G0 治理 | strict start-check 无 FAIL；新 experiment、W###、source commit、approval 均绑定 | 任一缺失即不训练 |
| G1 数据 | 身份、坐标、通路顺序、train-only z-score 全部通过 | 泄漏、缺坐标或列漂移即停止 |
| G2 工程 | residual-off parity、CPU/GPU 测试、结果合同通过 | 无法重现 baseline 或触碰 W004 即停止 |
| G3 pathway | I1 通过 D1 预注册门槛并超过 Ieq | 未通过则关闭，不上 Mamba/频域 |
| G4 spatial | 仅一个空间假设有稳定诊断依据 | 两者均不稳定则停止 |
| G5 external | 模型已完全冻结且获得一次性 XZY 批准 | 任何 XZY 调参意图即停止 |
| G6 downstream | Phase 2 结果完成 import/accept，另获迁移批准 | 不修改、不回填 W004 |

## 10. 审核后仍需用户明确决定的事项

1. 是否接受“开一条低优先级、受限 MPP2 独立线”的总方向；
2. 是否将下一永久工作树候选定为 W005；
3. 是否批准先做 Stage E（仅实现和单元测试，0 次训练）；
4. 是否接受 D1 的 18-attempt 上限与建议阈值；
5. D2 是否坚持“诊断后二选一”，而不是同时铺开 Mamba 与频域；
6. 是否接受“任何 Phase 2 胜者都不自动进入 W004”的硬隔离条款。

在这些事项获批前，正确状态保持：`review_draft / NOT RUN / pending_review`。

## 11. 当前已知阻塞与风险

- 2026-08-05 新鲜 strict start-check：`PASS=5, WARN=6, FAIL=1`；FAIL 为绝对工作树 HEAD 与 Registry 不一致，必须在正式登记/执行前解决。
- 主工作区已有与本路线无关的用户修改和未跟踪文件；未来创建实验工作树前必须按规则核验并保护，不得清理或覆盖。
- 现有 MPP dataset 不返回坐标和样本身份；空间模块不能直接接入当前 trainer。
- 8GB GPU 对整张切片数千 patch 的 SSM/二维网格可能形成显存压力，需 chunking、稀疏 mask 和 dry-run；不得为适配显存而静默改变证据协议。
- 患者数少、patch 数多，patch-level 显著性会夸大有效样本量；统计单位必须以患者为主。
- 新 LOPO 是新协议，不能把其 I0 叫作 accepted baseline 的“数值复现”。

## 12. 可核验来源

- [当前项目状态](../../CURRENT_STATE.md)
- [机器可读 current state](../current_state.json)
- [实验 Registry](../../experiments/experiment_registry.json)
- [工作树 Registry](../workspace_registry.json)
- [W004 Phase 3 分阶段方案](Phase3_双线基线与跨队列空间通路迁移分阶段实验方案_20260803.md)
- [accepted MPP2 训练实现](../../train_mpp_uni2h_mlp.py)
- [manifest dataset](../../dataset_mpp_manifest.py)
- [LoRA 在线模型与 raw-scale 指标](../../train_mpp_uni2h_lora.py)
- [旧 1D GFNet 实现](../../model_gfnet.py)
- [raw prediction schema](../schemas/prediction_artifact_contract_v1.schema.json)
- [编号工作树协议](gitee_numbered_workspace_protocol_v001_20260726.md)

