# MPP2 + LoRA 实验构建方案（Codex 草案）

> 日期：2026-07-11  
> 状态：**draft / 尚未批准正式训练**  
> 审核：Codex 基于 state revision 34、实验注册表、现有 LoRA 实现和 Gitee job 包编制  
> 作用：指导代码构建、隔离测试、3 epoch smoke 与后续正式实验；不得替代显式训练批准文件

## 1. 当前训练状态快照

### 1.1 已验收的 MPP2 repaired baseline

- 实验：`mpp2_barcode_repair_v003_frozen_baseline_20260711`
- 数据证据：`barcode-repair-20260711-d626ad8-v003`
- 数据 manifest：`barcode-repair-20260711-d626ad8-v003:1204018178a4d355`
- External MPP2/XZY：PCC `0.6549`，raw MAE `1176.2114`，raw R² `-0.0880`
- 三项安全门均通过，当前状态允许准备 LoRA r=8，并允许在既定门禁下执行最多 3 epoch smoke。

### 1.2 MPP1/3/4/5 repaired frozen 重跑

| MPP | job_id | dispatch commit | 本轮可审计状态 |
|---:|---|---|---|
| 1 | `mpp1-repair-v003-recheck-20260711` | `cc979d4` | 已派发；用户报告服务器正在运行；Gitee 尚无 result branch |
| 3 | `mpp3-repair-v003-recheck-20260711` | `cc979d4` | 已派发；用户报告服务器正在运行；Gitee 尚无 result branch |
| 4 | `mpp4-repair-v003-recheck-20260711` | `cc979d4` | 已派发；用户报告服务器正在运行；Gitee 尚无 result branch |
| 5 | `mpp5-repair-v003-recheck-20260711` | `cc979d4` | 已派发；用户报告服务器正在运行；Gitee 尚无 result branch |

边界说明：`project_state/current_state.json` 当前仍为 `active_training_jobs=[]`，不能反证服务器空闲。资源调度以用户的实时说明“正在运行”为准；本方案准备阶段不得启动任何服务器 GPU/训练进程。

## 2. 不干扰当前重跑的硬边界

1. **现在只做本地代码设计、静态检查和 CPU 单元测试。** 不在服务器执行模型加载、forward、cache parity 或 smoke。
2. MPP2+LoRA 服务器动作必须等待以下任一条件：
   - 四个重跑均出现并导入有效结果包，且服务器资源确认释放；或
   - 用户另行明确确认服务器资源已空闲、允许执行指定的 LoRA job。
3. 使用独立 job、分支、worktree 和输出根：
   - dispatch branch：`automation/local/mpp2-repair-v003-lora-r8-smoke-<date>`
   - result branch：`automation/server/mpp2-repair-v003-lora-r8-smoke-<date>`
   - output root：`checkpoints/mpp_uni2h_lora/`
4. 不修改或复用 MPP1/3/4/5 的 `automation/jobs/<job_id>/job.json`、结果目录、进程事件或 result branch。
5. 不删除、不覆盖、不重新生成原始 ssGSEA、标准 split、train-only z-score 参数、manifest、MPP3/5 embargo 资产和现有缓存。
6. 服务器交互仍只走 Gitee；禁止 SSH/SCP/HTTP command/tunnel。
7. `histogene/`、`egnv1/`、`egnv2/` 保持只读。

## 3. Gemini 指南审查与采用边界

已读取：`01_指南与解读/学习指南/LoRA在线微调与MPP2迁移学习指南.md`。总体判定为：**方向可参考，但伪代码和若干论证不能直接作为执行依据**。本方案吸收其在线训练、manifest Dataset、参数分组、显存预检和过拟合监控思路，同时用当前代码、repaired manifest 和状态门修正其实现细节。

### 3.1 可直接吸收的内容

- LoRA 必须在原始图像经过 UNI2-h backbone 的在线 forward 中训练，不能继续只读 frozen CLS cache。
- 继续使用现有手工 `LoRALinear` 的零增量初始化、qkv/proj 注入、参数分组和 checkpoint 能力，暂不引入新的 PEFT 依赖。
- 使用 AdamW、梯度裁剪、AMP、显存峰值记录和可选 gradient checkpointing。
- 在线 Dataset 必须由当前 fixed manifest 驱动，并沿用 train-only z-score 与 external XZY 隔离。
- 监控 train/internal-val gap，避免复用旧三患者 LoRA 的“很快过拟合”结论作为新实验性能证据。

### 3.2 必须修正的问题

| 级别 | 指南位置 | 问题 | 本方案修正 |
|---|---|---|---|
| 高 | 141-147 | 把当前 split 描述成“Patch 随机洗牌 90/10”。实际 `group_2/split_manifest.csv` 带 `block_id`，由固定空间 block 划分；`split_info.json` 记录 `block_size=672`、`leakage_pairs=0`。 | 在线 Dataset 原样消费 manifest 的 `split`，不得重新随机切分。 |
| 高 | 191-213 | 伪代码用单一 `barcode/stem` 作为全患者 label key，可能重现跨患者同名 barcode 覆盖；同时静默跳过缺图/缺标签。 | 统一使用 `(mpp_id, patient, patch_stem)`；重复、缺失、错配全部 hard fail。 |
| 高 | 208-213 | 假定图像为 `<root>/<patient>/<stem>.jpg`。实际注册路径是 `<mpp_root>/2/<patient>/patch_images/<stem>.png`。 | 通过 path registry 的 `mpp_data_root` 构造 `.png` 路径，不接受隐式扩展名 fallback。 |
| 高 | 101-104、242-251 | 把 `qkv+proj+fc1+fc2` 作为 MPP2 P0 强推荐，会同时扩大可训练容量、显存和过拟合风险，且没有本项目新数据证据。 | P0 固定历史可解释的 `qkv+proj, r=8`；MLP LoRA 仅在 P0 通过后作为单因素消融。 |
| 高 | 63-92 | 多阶段 merge + 解冻是旧在线脚本能力，不是已验收的 MPP2 方案；若直接带入会把“LoRA 效果”和“末层全参微调效果”混在一起。 | MPP2 P0 只做 Stage 1 LoRA；stage2/stage3 不进入本轮实验矩阵。 |
| 高 | 123-129 | 建议立即搜索 `r={8,16,32}`，与当前门禁“r=8、最多 3 epoch smoke”冲突，也容易利用 external XZY 调参。 | smoke 只跑 r=8；rank 消融需新批准，且只按 internal val 预注册选择。 |
| 高 | 154-161 | 断言 6 患者会让过拟合显著推迟、允许更多 epoch、提高验证 PCC，属于未经实验支持的预测；同患者空间 block val 也不等于跨患者泛化。 | 写成待验证假设；仍按短 smoke 和 external XZY 最终评估，不预先放宽 epoch。 |
| 中 | 107-120、254-261 | 把 LoRA weight decay `0.01-0.1` 写成必须/通用最佳值，缺少本项目消融证据；当前历史实现默认 `1e-4`。 | P0 先用 `1e-4` 保持可比；后续只在 internal val 上比较 `1e-4` 与 `1e-2`，不默认上到 `0.1`。 |
| 中 | 131-135、238-240 | 把 gradient checkpointing 写成必须；旧 qkv/proj LoRA 在 8GB 卡上已有约 4.5GB 峰值记录，且 API 需运行时探测。 | Phase 1 先测显存；仅在支持 `set_grad_checkpointing` 且确有需要时启用，并记录速度代价。 |
| 中 | 142-143 | 将“旧三患者 JFX 修复”和“MPP 标签重建时跨患者 barcode 污染”合并为同一原因，事实链不准确。 | 两者分开记录；本实验只绑定已验收的 repaired MPP manifest。 |
| 中 | 159-160 | 称“一个 Batch 会同时看到多患者”不一定成立，尤其 P0 计划物理 batch=1；shuffle/梯度累积也不等于患者均衡 batch。 | 不用该说法论证泛化；输出患者采样占比，后续患者均衡 sampler 需单独消融。 |
| 中 | 232-240 | `validate_state(...)` 的导入存在，但伪代码缺少当前 frozen 入口已有的 `host_scope="server"`、active repaired manifest 路径绑定和 `verify_mpp_repair_server_assets`。 | 新 trainer 完整复用现有三层状态/资产校验，不只检查 `report.ok`。 |
| 低 | 39-43 | 称 B=0 时 A、B 在第一步都得到非零梯度不准确；初始时通常 B 先得到梯度，A 的梯度因 B=0 为零，后续才开始更新。 | 修正文档解释，不影响现有实现。 |
| 低 | 18、22 | 使用 `file:///` 绝对链接，不利于仓库迁移和 Markdown 渲染。 | 后续指南维护改为仓库相对链接。 |
| 低 | 134、280 | 将 24 blocks、dim=1536 的 UNI2-h 简称为 ViT-Large 不严谨。 | 以实际加载模型 snapshot/config 为准，不在方案中硬写 ViT-Large。 |

### 3.3 继续采用的项目事实源

除 Gemini 指南外，本方案仍以以下实时工件为准：

- `lora_utils.py`
- `train_online_cls.py`
- `model_online_cls.py`
- `dataset_online.py`
- `train_mpp_uni2h_mlp.py`
- `dataset_mpp.py`
- `UNI2h_LoRA冒烟测试报告_20260530.md`
- 当前 repaired MPP2 baseline、状态包和 Gitee job/result envelope

Gemini 指南是学习与设计参考，不进入 active document authority chain；与 `project_state/current_state.json`、`experiments/experiment_registry.json` 或实际代码冲突时，以后三者为准。

## 4. 为什么必须新增独立训练入口

当前 `train_mpp_uni2h_mlp.py` 从 `.pt` 缓存读取已冻结的 UNI2-h CLS 特征，只训练 `MPPMLPHead`。LoRA 必须位于 backbone 的 forward 路径中，缓存特征绕过 backbone，因此无法产生 LoRA 梯度。

推荐新增而非改写 frozen 主线：

- `dataset_mpp_online.py`：由 fixed manifest 构建原始 patch 图像样本；严格绑定 repaired labels。
- `model_mpp_uni2h_lora.py`：`UNI2-h backbone -> CLS[1536] -> 原 MPPMLPHead`，不引入坐标嵌入或频域旁路。
- `train_mpp_uni2h_lora.py`：在线加载图像，支持 `frozen`/`lora` 两种模式，复用现有 LoRA 注入工具、评估输出和结果 envelope。

不得直接复用 `OnlineCLSModel` 作为主实验模型，因为它额外包含投影层、坐标嵌入和不同的回归头；那会同时改变 backbone 适配与下游结构，无法与 repaired frozen baseline 公平比较。

## 5. 数据协议

### 5.1 固定数据

- `train_mpp_id=2`
- 六个训练患者：从 repaired split manifest 读取，不在代码中另写新划分
- internal validation：同一 `split_manifest.csv` 中的固定 val 行
- external test：`MPP2/XZY`，只在 best checkpoint 固定后评估一次
- repaired label root：必须由 job 的 `data_manifest_id` 解析，禁止回退到旧 `mpp_standard_splits` 标签
- z-score：仅使用已经由训练 split 拟合并随 repaired evidence 验证的参数；不重新拟合
- split 类型：固定空间 block split，不是运行时 patch 随机切分
- 样本数：train `9472`、internal val `1078`、external XZY `1039`
- MPP2 overlap policy：`none_100pct_stride`；固定 `block_size=672`，`leakage_pairs=0`

### 5.2 在线 Dataset 的硬校验

1. manifest 每行必须唯一解析到 `(mpp_id, patient, patch_stem, split, block_id)`。
2. 每个训练/验证样本必须一对一匹配 patch 图像与 repaired label。
3. 任一重复键、跨 split 重叠、患者错配、缺图、缺标签均默认 hard fail；正式实验禁止 `allow_missing`。
4. XZY 不得出现在训练、internal val、z-score fit 或 checkpoint selection 中。
5. 输出样本清单及 SHA-256，记录 train/val/external 的患者级数量。
6. 图像路径严格解析为 `<mpp_data_root>/2/<patient>/patch_images/<patch_stem>.png`；禁止指南伪代码中的 `.jpg` 猜测和静默跳过。

## 6. 模型与 LoRA 配置

### 6.1 公平对照模型

```text
image[3,224,224]
  -> UNI2-h backbone
  -> CLS[1536]
  -> Linear(1536,1024)
  -> GELU
  -> Dropout(0.3)
  -> Linear(1024,30)
```

回归头必须与 `MPPMLPHead` 完全一致，不添加 LayerNorm、位置编码、token pooling 或额外支路。

### 6.2 P0 LoRA 配置

| 参数 | 值 |
|---|---|
| rank | `8` |
| alpha | `16` |
| scale | `2.0` |
| target blocks | `0-23`（全部 24 层） |
| target modules | `attn.qkv` + `attn.proj` |
| LoRA dropout | `0.0` |
| backbone base weights | 全冻结 |
| gradient clipping | `1.0` |
| AMP | 开启 |
| seed | P0=`42`；仅通过后追加 `43/44` |

LoRA 注入顺序必须是：加载固定 UNI2-h 权重 -> 冻结 base backbone -> 注入 LoRA -> 构建优化器。初始 `lora_B=0`，因此注入瞬间输出应与未注入 backbone 一致。

Gemini 建议的 `fc1/fc2` 扩展不进入 P0。若 r=8 qkv+proj 正式实验通过，才新增一个保持 rank、seed、split、epoch 和 optimizer 不变的 `qkv+proj+fc1+fc2` 单因素消融；不得与 rank 或 weight decay 同时变化。

## 7. 迁移一致性门禁（训练前必过）

这是本方案最关键的实现门禁，用于证明“在线 LoRA 路径”和现有 frozen 缓存协议一致。

### Gate A：LoRA 零增量单元测试

- 对固定随机输入，plain backbone 与刚注入 LoRA 的 backbone 输出一致。
- base backbone 参数全部 `requires_grad=False`。
- 只有 LoRA A/B 与 MLP head 可训练。
- optimizer 参数集合与 `requires_grad=True` 参数集合完全一致且无重复。
- checkpoint 保存/加载后输出一致；本实验不执行 merge-to-original。

### Gate B：真实样本 cache parity

服务器资源释放后，从每个非 XZY 患者固定抽取少量 manifest 样本：

1. 用正式 UNI2-h transform 在线计算 CLS。
2. 与 repaired frozen baseline 使用的 `.pt` CLS 缓存逐样本比较。
3. 输出 cosine similarity、mean/max absolute error、dtype、模型 snapshot 和 transform 摘要。

建议门槛：FP32 cache 的 cosine `>=0.99999`；低精度 cache 的 cosine `>=0.999`。未通过时停止 LoRA，先查模型权重、transform、CLS 选择和 cache 版本，不得把差异解释成 LoRA 效果。

### Gate C：数据与输出隔离

- `agent start-check --strict`、`paths validate`、job schema 校验通过。
- job 绑定 state revision、source commit、正式 phase、批准对象和 repaired manifest。
- 输出目录不存在或为空；不得复用 frozen baseline/MPP1/3/4/5 目录。

## 8. 分阶段实验矩阵

### Phase 0：本地构建（可立即进行）

- 新增在线 dataset/model/trainer 和 CPU mock tests。
- 只用 dummy backbone/临时小图测试 manifest 过滤、LoRA 注入、checkpoint roundtrip、external 隔离和 argparse。
- 不加载服务器数据，不启动训练，不创建正式结果。

### Phase 1：服务器资源释放后的最小 preflight

- 运行 Gate B 的小样本 cache parity。
- 单 batch forward/backward，检查显存、NaN/Inf、梯度非零和输出形状。
- 先关闭 gradient checkpointing 测量真实峰值；仅在显存不足或安全余量不足时启用，并用 `hasattr(backbone, "set_grad_checkpointing")` 确认 API。
- 此阶段不计为正式性能证据。

### Phase 2：配对 3 epoch smoke

为避免“MLP 头多训练了 3 epoch”被误判为 LoRA 收益，建议从同一 repaired frozen head checkpoint 出发做配对 smoke：

已验收结果包记录的服务器 checkpoint 为 `D:\AIPatho\qzs\pfmval_deploy_git\checkpoints\mpp_uni2h_mlp\mpp2_barcode_repair_v003_frozen_baseline_20260711\best_checkpoint.pth`，大小 `6420688` bytes，SHA-256 `c69191d4a67939724988bc3656c3cd2e0b173d9c871456a5fb900ab446e86a98`。加载前必须重新校验路径、大小和哈希；只提取 `MPPMLPHead` 权重，不把 frozen cache 当作 LoRA 输入。

| 组别 | 初始化 | 可训练参数 | epoch |
|---|---|---|---:|
| S0 frozen-continue | accepted MPP2 head + frozen backbone | MLP head | 3 |
| S1 LoRA-r8-continue | 同一 accepted MPP2 head + 零增量 LoRA | MLP head + LoRA | 3 |

两组使用相同 seed、batch 顺序、internal val、optimizer 预算和 checkpoint 规则。S1-S0 的差值才是 smoke 阶段较可信的 LoRA 增量。若资源预算只允许一个组，S1 可先用于工程 smoke，但不得据此宣称 LoRA 有效。

建议起始优化器：AdamW；head lr `1e-4`，LoRA lr `1e-4`，P0 weight decay `1e-4`。若显存未知，先 `batch_size=1`、`grad_accum_steps=8`；以 Phase 1 实测后再提高物理 batch，不以 OOM 试错影响在跑任务。患者采样分布必须写入训练摘要，不能把梯度累积表述为“一个 batch 同时含多患者”。

### Phase 3：正式 r=8 主实验（需新的显式批准）

- 只有 3 epoch smoke 完整回传、导入和审计通过后才提出正式 job。
- 固定 r=8，不立即展开 rank/dropout/block 范围矩阵。
- seed 42 通过后再追加 43/44；报告均值、标准差和 95% bootstrap CI。
- external XZY 只用于最终一次评估，不能选择 seed、epoch 或超参。
- 若需要后续消融，顺序固定为：LoRA target modules -> weight decay -> rank；每次只改变一个因素，候选选择只看 internal val。

## 9. Checkpoint 与评估规则

- checkpoint selection：internal `val_loss` 最小。
- early stop：正式实验沿用短窗口，具体 patience 在 smoke 曲线后冻结；不得根据 external XZY 调整。
- 必交付：training history、best epoch、internal predictions、external predictions、per-pathway PCC、raw-scale MAE/R²、参数统计、显存峰值、运行时长、manifest/sample hashes、result envelope。
- 附加诊断：train-val gap、val-external gap、各患者 internal val 切片、逐通路增减排名。

## 10. 判定门

相对 accepted repaired baseline（PCC `0.6549`、raw MAE `1176.2114`、raw R² `-0.0880`）：

### 候选有效

- External PCC `>=0.6749`（至少 `+0.02`）；
- raw MAE `<=1235.02`（恶化不超过 5%）；
- raw R² `>=-0.1380`（下降不超过 0.05）；
- internal 提升不能伴随明显扩大的 val-external gap；
- 多 seed 方向一致后才称为稳定增益。

### 边界结果

- PCC 有提升但不足 `+0.02`，或 raw MAE/R² 超出候选门但未触发安全线：只记为探索信号，不升级主结论。
- PCC 上升而 raw 指标恶化：只能称为排序增益，不能称为部署增益。

### 安全暂停

相对 repaired baseline 若 PCC 下降 `>=0.05`、raw MAE 增加 `>=10%`（`>=1293.83`）或 raw R² 下降 `>=0.10`（`<=-0.1880`），任一触发即暂停 LoRA 路线并审计实现/数据；不得继续扩大超参搜索。

## 11. 推荐实施顺序

1. 仅在本地新增独立在线 MPP2+LoRA 代码与测试。
2. 审核代码不改动 frozen 重跑入口和 protected assets。
3. 等待 MPP1/3/4/5 服务器资源释放并确认无在跑进程。
4. 创建新的 source commit、job ID 和正式批准对象。
5. 先过真实样本 cache parity 与单 batch preflight。
6. 执行 S0/S1 配对 3 epoch smoke，Gitee 回传并验证导入。
7. 根据预注册判定门决定是否申请正式 r=8、多 seed 实验。

## 12. 当前结论

MPP2+LoRA 已具备数据门禁上的推进条件，但尚不具备“立即占用服务器训练资源”的条件。最安全且信息增益最高的做法是：**现在构建独立在线训练链和 CPU 测试；待 MPP1/3/4/5 重跑释放资源后，先验证 cache parity，再做同 checkpoint 的 frozen-continue 与 LoRA-r8-continue 配对 3 epoch smoke。**

## 13. Phase 0 实施状态（2026-07-11，Codex）

用户已批准本方案；项目指令记录为 `DIR-20260711-007`。本地 Phase 0 已实现：

- `dataset_mpp_online.py`：复合键严格匹配、固定 manifest split、PNG 在线加载、缺失/重复 hard fail。
- `model_mpp_uni2h_lora.py`：复用现有 `MPPMLPHead`，P0 只允许 qkv+proj、r=8、alpha=16。
- `train_mpp_uni2h_lora.py`：绑定 active repaired staging、accepted head checkpoint 哈希、S0/S1 配对优化器、最多 3 epoch、external 延迟评估、raw-scale 指标与 trainable-only checkpoint。
- `scripts/check_mpp_online_cache_parity.py`：Phase 1 在线 CLS/cache 比较工具；必须显式提供 `MPP1_3_4_5_IDLE` 资源释放确认才可运行。
- `tests/test_mpp_uni2h_lora.py`：覆盖患者作用域、重复/缺失硬门、零增量、参数组、checkpoint 兼容、CPU train/eval 和 parity 路径解析。

本轮验证：`python -m pytest tests -q` 为 `56 passed`；Python 编译检查通过。

尚未执行且不得冒充已完成的事项：

- 未在服务器加载 UNI2-h；
- 未执行真实样本 cache parity；
- 未执行单 batch GPU preflight；
- 未注册或派发 LoRA smoke job；
- 未执行 S0/S1 3 epoch smoke 或正式训练。

下一状态门：确认 MPP1/3/4/5 已释放服务器资源，再运行 parity 工具；parity 未通过不得创建 smoke job。

## 14. Phase 1 派发状态（2026-07-11，Codex）

- 用户确认 MPP1/3/4/5 已释放服务器资源；记录为 `DIR-20260711-008`。
- 四项 repaired frozen recheck 均已通过 Gitee 结果包导入并进入 accepted 状态。
- parity 预检实验：`mpp2_online_cache_parity_v003_20260711`。
- source commit：`35d718b8312ceb9d5291e43ac7561466e3a7d256`。
- job：`mpp2-online-cache-parity-v003-20260711-r001`。
- dispatch commit：`3ce4fe620bf5bc4ea5d7c0366af7bb1354d87583`。
- Gitee 分支：`automation/local/mpp2-online-cache-parity-v003-20260711-r001`。
- job phase/command：`preflight / cache_parity`；每患者固定抽取 8 个样本，设备为 CUDA。
- 本地验证：完整测试 `58 passed`，strict state gate `PASS=6 WARN=6 FAIL=0`，job manifest 校验通过。

项目当前未启用自动服务器轮询；服务器侧必须手动 fetch、validate、dry-run、run 和 pack/result-branch push。结果分支尚未出现，因此 **LoRA smoke 仍未创建或派发**。
