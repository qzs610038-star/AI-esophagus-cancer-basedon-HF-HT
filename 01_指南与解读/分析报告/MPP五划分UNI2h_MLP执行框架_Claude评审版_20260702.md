# MPP 五划分 UNI2-h MLP 执行框架（Claude 评审版）

更新时间：2026-07-02

用途：给 Claude 评审和执行 MPP 五划分实验入口改造。本文只定义执行框架，不替代完整结果分析。

## 1. 本轮固定口径

目标：比较 5 种 MPP/步长 patch 划分对外部患者 XZY 的跨患者泛化影响。

统一模型：`UNI2-h 冻结特征 + 两层 MLP 回归头`。

禁止混入：LoRA、Token+GFNet、频域分支、渐进解冻、旧 3 患者 fold 逻辑。

外部验证集固定：

```text
D:\AIPatho\Patch\visiumhd_patch\2\XZY
```

每个 MPP 训练集：

```text
D:\AIPatho\Patch\visiumhd_patch\{1-5}\HYZ15040
D:\AIPatho\Patch\visiumhd_patch\{1-5}\JFX
D:\AIPatho\Patch\visiumhd_patch\{1-5}\LMZ12939
D:\AIPatho\Patch\visiumhd_patch\{1-5}\TGC
D:\AIPatho\Patch\visiumhd_patch\{1-5}\XSL
D:\AIPatho\Patch\visiumhd_patch\{1-5}\ZHZ
```

实验 ID：

| MPP | Experiment ID |
|---|---|
| 1 | `mpp_v1_xzy_external_uni2h_mlp_20260701` |
| 2 | `mpp_v2_xzy_external_uni2h_mlp_20260701` |
| 3 | `mpp_v3_xzy_external_uni2h_mlp_20260701` |
| 4 | `mpp_v4_xzy_external_uni2h_mlp_20260701` |
| 5 | `mpp_v5_xzy_external_uni2h_mlp_20260701` |

## 2. 标签标准化

MPP 数据根里的 `{patient}_ssGSEA.csv` 是原始 ssGSEA 分数，不能直接训练。

标准化要求：

1. 只用训练集六例患者拟合 z-score 参数。
2. 用同一套参数转换训练集和外部 XZY。
3. XZY 不参与 z-score 参数拟合。
4. 保存每个 MPP 的 `zscore_params.csv/json` 和标准化后标签。
5. 训练前检查 NaN/Inf、patch stem 与标签交集。

## 3. 内部验证集决策

### 方案 A：不划分内部验证集（推荐默认）

含义：六例非 XZY 患者全部用于训练，固定训练预算，训练完成后只用 `MPP-2/XZY` 做一次外部评估。

优点：

- 最贴合本轮目标：直接检验跨患者泛化到 XZY。
- 最大化训练样本量，尤其适合 MPP-4 小样本方案。
- 避免 MPP-3/MPP-5 重叠 patch 的内部 train/val 去重负担。
- 避免把内部随机验证结果误读为单患者内部预测能力。
- 执行更简单，减少入口开发复杂度。

缺点：

- 不能用 `val_loss` 选择 best epoch。
- 不能 early stopping，只能固定 epoch 或固定 step。
- 若训练中后期过拟合，只能从训练 loss 和最终外部 XZY 指标事后判断。
- 需要提前锁定 epoch、学习率、seed，不允许看 XZY 结果后反复调参。

执行框架：

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HOME = "D:\AIPatho\shared\.cache\huggingface"
$env:TORCH_HOME = "D:\AIPatho\shared\.cache\torch"
$env:HF_HUB_OFFLINE = "1"

python -u train_mpp_uni2h_mlp.py `
  --mpp_root "D:\AIPatho\Patch\visiumhd_patch" `
  --train_mpp_id 3 `
  --train_patients HYZ15040,JFX,LMZ12939,TGC,XSL,ZHZ `
  --external_mpp_id 2 `
  --external_patient XZY `
  --val_strategy none `
  --num_epochs 50 `
  --batch_size 32 `
  --lr 1e-4 `
  --seed 42 `
  --dataset_name "mpp_v3_xzy_external_uni2h_mlp_20260701"
```

输出要求：

- `training_history.csv`：记录每 epoch train loss。
- `training_summary.txt`：写明 `val_strategy=none`，final checkpoint 评估外部 XZY。
- `predictions_external_xzy.csv`：外部 XZY 预测。
- `per_pathway_pcc_external_xzy.csv`：逐通路 PCC。
- `zscore_params.*`：训练集拟合的标准化参数。

结果解释：

- 使用 final checkpoint，不声明 best epoch。
- 报告用语写为：`Fixed-budget training; final checkpoint evaluated once on external XZY.`
- XZY 不能用于 epoch 选择、学习率选择、重跑筛选或早停。

### 方案 B：划分内部验证集

含义：从六例训练患者中切出一小部分内部 val，用于 early stopping / best epoch / 训练崩坏监控；外部 XZY 仍只做最终测试。

优点：

- 可以保留项目既有 `val_loss` 选 best checkpoint 的训练控制方式。
- 更容易发现 NaN、标签错位、明显过拟合或学习率异常。
- 有助于比较同一 MPP 内不同 epoch 的稳定性。

缺点：

- 内部 val 不代表真正跨患者泛化，容易被误读。
- MPP-3/MPP-5 有 50% 重叠 patch，不能随机 patch split；否则高度相似 patch 会跨 train/val 泄漏。
- 需要按空间 block、父 spot 或重叠组做 group split，入口开发更复杂。
- 会减少训练样本量，尤其对 MPP-4 不友好。

推荐内部 val 方式：

```text
每个训练患者按空间 block 或重叠 group 划分。
抽 8%-10% group 作为 internal val。
同一重叠来源的 patch 必须同去向：全进 train 或全进 val。
```

执行框架：

```powershell
python -u train_mpp_uni2h_mlp.py `
  --mpp_root "D:\AIPatho\Patch\visiumhd_patch" `
  --train_mpp_id 3 `
  --train_patients HYZ15040,JFX,LMZ12939,TGC,XSL,ZHZ `
  --external_mpp_id 2 `
  --external_patient XZY `
  --val_strategy spatial_group `
  --val_ratio 0.1 `
  --num_epochs 50 `
  --batch_size 32 `
  --lr 1e-4 `
  --seed 42 `
  --dataset_name "mpp_v3_xzy_external_uni2h_mlp_20260701"
```

输出要求：

- `split_manifest.csv`：每个 patch 的 train/val/external 归属。
- `group_overlap_audit.csv`：证明重叠 group 没有跨 train/val。
- `training_history.csv`：记录 train loss、internal val loss、internal val PCC。
- `predictions_internal_val.csv` 和 `predictions_external_xzy.csv`。
- `training_summary.txt`：明确 internal val 只用于训练控制，不作为最终科学结论。

结果解释：

- 可以用 internal val loss 选 best epoch。
- 最终排序仍只看 external XZY Test PCC / MAE / R2 / 逐通路 PCC。
- 若 internal val 与 external XZY 排序冲突，以 external XZY 为最终比较依据，但不得用 XZY 反向重选 epoch。

## 4. 推荐给 Claude 的执行顺序

1. 先审计服务器数据目录和 patch 数。
2. 新增或改造 MPP 专用入口 `train_mpp_uni2h_mlp.py`。
3. 实现统一数据读取：训练 MPP id 可变，外部 XZY 固定为 MPP-2。
4. 实现 z-score 标准化：只用训练集拟合，保存参数。
5. 先实现方案 A：`--val_strategy none`。
6. 对 MPP-3 跑 1 epoch smoke test。
7. 若 smoke test 正常，再扩展到 5 个 MPP。
8. 若评审认为必须保留 best epoch，再实现方案 B 的 `spatial_group` 内部验证。

## 5. Claude 评审重点

- 是否彻底移除了 LoRA 参数和 `cls_lora_r8` 命名。
- 是否没有复用旧 `--cross_patient --fold 1/2/3` 逻辑。
- XZY 是否固定为 `D:\AIPatho\Patch\visiumhd_patch\2\XZY`。
- XZY 是否未参与训练、z-score 拟合、epoch 选择或调参。
- MPP-3/MPP-5 若启用内部 val，是否有重叠 group 防泄漏审计。
- 输出文件是否足够让 Codex 复核：日志、summary、history、predictions、per-pathway PCC、z-score params、split manifest。

## 6. 当前建议

默认采用 **方案 A：不划分内部验证集**。

理由：本轮问题是 MPP 划分对外部 XZY 泛化的影响，不是单患者内部预测能力。无内部 val 可以最大化训练数据、简化入口，并避免 MPP-3/MPP-5 的内部重叠去重成本。代价是必须严格固定训练预算，不能看 XZY 后反复调参。

---

## 7. Claude 评审补充（2026-07-02，来源：Claude Code 会话评审）

> 评审依据：本框架全文 + `01_指南与解读/部署方案/服务器路径索引_20260701.md` + `.qoder/experience.md` + `CLAUDE.md` 核心铁律。V3（`mpp_v3_xzy_external_uni2h_mlp_20260701`）为本次评审对象，以下问题分级按对 V3 执行的影响排序。

### 🔴 必须修正（阻断 V3 执行）

**P1. 执行顺序漏了 UNI2-h 特征预提取/缓存步骤**
原 §4 第 2 步直接"改造入口脚本"，未把特征预提取列为独立阶段。MPP-3 是五划分里**样本量最大**的训练集：

```text
HYZ15040 5807 + JFX 8116 + LMZ12939 13376 + TGC 4476 + XSL 6093 + ZHZ 4257 ≈ 42,125 train patches
（MPP-3 训练合计；外加 MPP-2/XZY 1039 外部 patch）
```

若不在训练前一次性预提取 UNI2-h 特征到缓存，每次训练/重跑都会重复跑 backbone 推理。应在执行顺序里补一步：特征预提取 → `mpp_uni2h_cache/{mpp_id}/{patient}/`。

**P2. 未遵守 registry 铁律（CLAUDE.md #10）**
方案给出 `mpp_v3_xzy_external_uni2h_mlp_20260701` experiment ID，但未声明需在 `experiments/experiment_registry.json` 注册 `status: planned`。按项目铁律：启动训练前必须先注册对应 experiment id，训练完成后执行 `python scripts/finalize_experiment.py`。禁止绕过 registry 手动散落状态信息。

**P3. NaN 保护未写进框架**
原 §2 第 5 步仅"训练前检查 NaN/Inf"。本项目已知 **LMZ12939 含极端 z-score 值**（`.qoder/experience.md` 与 `CLAUDE.md` 训练要点反复记录），z-score 标准化与训练 loss 计算必须显式 `np.nan_to_num + clip ±100`，否则 numpy `reduce` 溢出 → loss 变 NaN。框架应作为硬性步骤写明，而非"检查"。

### 🟡 V3 特别注意（非阻断，建议处理）

**P4. MPP-3 是最大训练集 + Strategy A 固定 50 epoch → 最易过拟合且无早停**
方案 §3 方案 A 自认"不能 early stopping"。V3 训练样本 ~42K 是 V4(~2.8K) 的 15 倍，同样固定 50 epoch 对 V3 偏向过拟合。建议 V3 除固定 epoch 外，**额外记录每 epoch train loss**，事后判断是否在 50 epoch 前已收敛；或将 V3 epoch 预算调小（如 30）。最终取值待用户拍板，不在框架内写死。

**P5. external_mpp_id 固定为 2 的方法论说明缺失**
方案 §1 用 `--external_mpp_id 2` 给所有 MPP 当外部测试集，但未解释"为何固定 MPP-2 而非各 MPP 自身对应的 XZY"。这是**有意为之的受控比较**（同一外部测试集，变换训练集），应在文档写明，否则 Codex 复核会质疑训练(MPP-3)与测试(MPP-2)patch 提取参数不一致是否公平。

### 🟢 方案已做对的地方（确认无需改）

- 彻底排除 LoRA / Token / GFNet / 频域分支 / 渐进解冻 / 旧 3 患者 fold 逻辑 ✓
- XZY 固定不参与训练、z-score 拟合、epoch 选择或调参 ✓
- MPP-3/MPP-5 在 Strategy B 下要按重叠 group 划分（V3 用 Strategy A，不涉及，无泄漏风险）✓
- JFX 命名按 MPP 根 `JFX`（非 Phase2 的 `JFX0729`），与服务器路径索引一致 ✓
- 输出文件清单足以让 Codex 复核（日志 / summary / history / predictions / per-pathway PCC / zscore params / split manifest）✓
- 推荐默认 Strategy A，理由成立 ✓

### 建议补充的执行步骤（落实时使用）

| 步骤 | 动作 | 产出 |
|:---:|------|------|
| 0 | 服务器数据预检：核对 MPP-3 各患者 `patch_images/` 数 + `{patient}_ssGSEA.csv` 30 通路列结构；核对 MPP-2/XZY 同样 | 预检日志 |
| 1 | 注册实验到 `experiments/experiment_registry.json`：`mpp_v3_..._20260701`，`status: planned`，标注 strategy=A、external=MPP-2/XZY | registry 更新 |
| 2 | 特征预提取（MPP-3 六例 + MPP-2/XZY 一次性）到 `mpp_uni2h_cache/{3|2}/{patient}/*.pt`，`HF_HUB_OFFLINE=1` 指向 `D:\AIPatho\shared\.cache` | 特征缓存 |
| 3 | z-score 标准化：只在 MPP-3 六例拟合 mean/std → `zscore_params.json` → 同参数应用到六例 + 外部 XZY；含 `nan_to_num + clip ±100` | zscore_params + 标签 |
| 4 | 新建训练入口 `train_mpp_uni2h_mlp.py`：UNI2-h frozen 特征 + 2 层 MLP 回归头，`--val_strategy none`，内置 `--num_threads 8` / `HF_HUB_OFFLINE` / `PYTHONIOENCODING` / `dataset_name` 防碰撞 / NaN 保护 | 训练脚本 |
| 5 | Smoke test：MPP-3 跑 1 epoch，确认 loss 下降、无 NaN、输出齐全 | smoke 日志 |
| 6 | 正式训练：固定 seed 42，epoch 预算待定（50 或 30） | checkpoint + `training_history.csv` |
| 7 | 外部评估：final checkpoint 一次性评估 MPP-2/XZY → `predictions_external_xzy.csv` + `per_pathway_pcc_external_xzy.csv` | 评估产物 |
| 8 | 收尾：`python scripts/finalize_experiment.py` 更新 registry + dashboard | registry done |

### 待用户拍板项

- (a) V3 epoch 预算：维持框架 **50**，还是因 MPP-3 样本最大调到 **30**？
- (b) 特征提取与训练是否均在**服务器**执行？（MPP 数据在 `D:\AIPatho\Patch`，本机无此数据）
