# 给 phase3 队友：方案 A「经典空间残差」交接说明

> **读者**：phase3 / Phase 3 下游疗效预测同学  
> **选定方案**：方案 A = 本包四臂中的 `spatial`（经典中间残差 / 经典空间残差）  
> **本文只回答两件事**：① 模型权重在哪；② 如何接到当前 Phase 3（输入为 **30 维 z-score 预测分数**）。

> **推荐直接使用的开箱包**：`团队项目进度与结论/qzs/Phase3_方案A经典空间残差_开箱包_20260912/`。该目录已包含正式权重、两种 Phase 3 输入接口、配置模板、CLAM 输出和调试记录；下文保留为模型合同说明。

---

## 先和同目录另外两份说明区分开


| 文件                                   | 给谁                      | 用途                                  |
| ------------------------------------ | ----------------------- | ----------------------------------- |
| **本文件** `给phase3队友的交接说明_方案A经典中间残差.md` | phase3 队友               | 取方案 A 正式权重，生成 30 维 z-score，接入疗效预测管线 |
| `给队友的任务说明.md`                        | Phase 2 消融（任务 2 / 任务 3） | 用同一份权重做基因→通路、隔点训练消融                 |
| `队友同步说明.md`                          | 同上消融同学 / 其智能体           | 消融包改哪些代码、加载哪些 state 键               |


phase3 **不要**按任务 2/3 文档去改基因头或抽稀训练；也不要重训本包四臂。

---

## 1. 模型权重放在哪里

### 1.1 要用哪一份

已接纳批次：`20260908_005325_810_8d306c10`  
实验臂：`spatial`（方案 A）  
默认交付种子：**42**（正式轮次 30；内部验证选出）  
文件名：`formal_best.pt`（不要用同目录 `warmup_best.pt` / `last.pt`）

### 1.2 随包本地权重（Phase 3 默认使用）

```text
团队项目进度与结论/qzs/Phase3_方案A经典空间残差_开箱包_20260912/weights/formal_best_seed42.pt
```

这是下述服务器 `formal_best.pt` 的本地回传副本。开箱包配置默认引用它，Phase 3 队友不需要再从旧 `runs` 目录拼装权重和代码。

**服务器绝对路径（权威）：**

```text
D:\AIPatho\qzs\runs\phase2_softlink_local_v2\20260908_005325_810_8d306c10\train_42_spatial\checkpoints\formal_best.pt
```

批次端点清单（只取 `arm=spatial`、`seed=42`、`checkpoint_kind=formal` 那一行）：

```text
D:\AIPatho\qzs\runs\phase2_softlink_local_v2\20260908_005325_810_8d306c10\formal_endpoints.json
```

注意同批次还有种子 43、44 的 `train_43_spatial` / `train_44_spatial`，属于方案 A 本身的三种子复现。phase3 默认只接 **种子 42**；若要做多种子敏感性，再另行约定，不要静默换种子。

### 1.3 上机核对（可选）

```powershell
$p = 'D:\AIPatho\qzs\runs\phase2_softlink_local_v2\20260908_005325_810_8d306c10\train_42_spatial\checkpoints\formal_best.pt'
Test-Path $p
& 'C:\Users\AIPatho1\pfmval_env\Scripts\python.exe' -c "import torch; d=torch.load(r'$p', map_location='cpu', weights_only=False); print(d['arm'], d['kind'], d['seed'], d.get('selection'))"
```



注：本命令要在账号AIPatho1内运行，否则会报错，因为有部分环境以前不小心放到此账号目录下了



应看到：`arm=spatial`，`kind=formal`，`seed=42`，正式选出轮次为 30。缺文件就停，不要用 warmup/last 顶替，也不要重开训本包「补一个权重」。

若不使用开箱包内副本，也可在配置中引用上述服务器绝对路径。无论采用哪一份，都不要覆盖原批次权重。

### 1.4 checkpoint 里有什么

`torch.load(..., weights_only=False)` 得到的 dict 至少含：

- 元数据：`arm`、`seed`、`kind`、`epoch`、`selection`
- `model_state_dict`：`shared.*`（H）、`point_head.*`（C）、`spatial_head.weight`（B，**无 bias**）
- **没有** `relation_head`

---

## 2. 如何接入 Phase 3

### 2.1 接口对齐（与当前 phase3 一致）


| 项目             | 约定                                                            |
| -------------- | ------------------------------------------------------------- |
| Phase 3 当前通路输入 | **30 维 z-score 预测分数**（连续向量，不是预先 Top-k）                        |
| 方案 A 网络输出      | 同样是 **30 维训练标准化 z 分数**（`y = p + Bu`）                          |
| 推荐接入方式         | 把方案 A 在目标切片上的 **`pred_z`（30 维）** 当作基因/通路分支特征，替换或对比旧 MPP2 通路特征 |
| 不要默认做的事        | **不要**再做一次逆变换再喂给 phase3；当前管线要的是 z-score，不是 raw ssGSEA         |


若某条旧流水线仍读「raw scores」目录，那是历史 MPP2 raw 口径，与本次交接不一致。本次交付以 **z-score 30 维** 为准。

通路顺序必须与开箱包 `assets/pathway_manifest.json` 的 `pathway_names` 一致（共 30 条，从 `tls` … 到 `ECM_Organization`）。列顺序错了等于换特征。

### 2.2 方案 A 前向（推理时必须构图）

```text
UNI2-h 冻结缓存 1536 维 + 同切片、同划分坐标
  → 固定图（半径 1.5×原生步长，最多 8 邻居；边权 = 距离高斯 × 特征余弦，不学习）
  → H: Linear 1536→256 + GELU
  → C: 中心 Dropout 仅训练时；推理关闭 → Linear 256→30 → p
  → B: Linear 256→30，无偏置 → 残差 Bu
  → y = p + Bu   ← 这就是交给 Phase 3 的 30 维 z-score
```

只把 `formal_best.pt` 灌进「1536→1024→30 的旧 MLP」或只对中心特征做 `model(features)`、不传邻居，会 silent 退化成无空间模型，**不能**称为方案 A。

图超参须与本包 `config.json` 的 `graph` / `model` 一致：`hidden_dim=256`，`dropout=0.3`，`spatial_bias=false`，`radius_in_native_steps=1.5`，`max_neighbors=8` 等。

### 2.3 推荐工程步骤

优先复制并调整上述开箱包中的 `configs/clam_slide.example.json` 或 `configs/per_patch.example.json`。包默认宽容运行：单张坏切片或个别坏 patch 记录警告后继续；只有权重/模型完全不可用，或整个任务没有任何有效输出时停止。

1. **代码**：默认直接运行开箱包 `src/phase3_package.py`，不依赖服务器旧代码目录。如需对照原实验实现，只读查看随包 `reference_phase2_source/`。
2. **图像特征**：目标 ESCC 切片使用与训练一致的 **UNI2-h 1536 维**缓存；推理不现场加载整网 UNI2-h。
3. **点表**：每个 patch / spot 需有身份键、坐标、特征路径；空间图按 **同 `slide_id` + 同划分** 构建，**不要用 `patient_id` 冒充物理切片**。
4. **加载权重**：整网 `strict=True` 加载种子 42 的 `formal_best.pt`（`shared` + `point_head` + `spatial_head`）。
5. **导出**：保存逐点 **`pred_z`，shape = `[N, 30]`**，并带上 `pathway_names`、slide/spot 对齐键。本包外部预测 npz 里同时有 `pred_z` 与 `pred_raw`；**phase3 请读 `pred_z`**。
6. **接到 CLAM / 融合管线**：用新的 30 维 z-score 特征目录，按你们现有 `convert_patch_scores_to_clam.py` →（可选）`concat_clam_features.py` 流程，替换原先 MPP2 通路特征侧；`gene_dim` 仍为 **30**。病理侧仍是 UNI2-h 1536，拼接后 `embed_dim` 仍为 1566（若走直接拼接）。

### 2.4 标度边界（避免接错）

- `pred_z`：训练标签标准化空间；**不保证**在新切片上均值为 0、方差为 1。
- 仅当下游明确要求原始 ssGSEA 量级时，才用本包训练集参数做 **一次**逆变换：  
`raw = z × Std_train + Mean_train`（见开箱包 `assets/zscore_params_from_train.json`）。  
**禁止**用 Phase 3 测试患者或外部队列重估 mean/std。
- 当前 phase3 已采用 z-score 输入 → **默认跳过逆变换**。
- 通路筛选、加权、选 checkpoint 只能用 Phase 3 训练患者及内部验证；不能用盲测标签调权。

### 2.5 参考性能（稠密 Phase 2，不是 pCR/MPR 成绩）

下列数字来自已接纳批次，便于核对是否加载了正确权重；**不证明**疗效预测会更好。

种子 42、`spatial`：


| 划分     | 逐通路平均 PCC ↑ | 展平 PCC ↑ | zMSE ↓ |
| ------ | -----------: | --------: | ------: |
| 内部验证   | 0.6670      | 0.8058   | 0.3617 |
| 外部 XZY | 0.5541      | 0.6714   | 0.6133 |


报告 Phase 3 结果时请写清：「方案 A / softlink_local_v2 / 批次 `20260908_005325_810_8d306c10` / 种子 42 / `spatial` / `formal_best.pt` / 输入为 30 维 z-score」。

---

## 3. 请遵守的边界

- 只交付方案 A（`spatial`），不要把 relation/joint 或 β=1 平滑权重混进同一套「方案 A」目录而不标注。
- 不要删除或覆盖 `...\20260908_005325_810_8d306c10\` 下任何文件；替换你们自己代码时只动 `code/` 下对应文件夹。
- 新推理/训练结果写到你们自己的 `runs/<实验名>/<运行编号>/`，并登记所用 `formal_best.pt` 的绝对路径。
- 文件存在或能跑通 ≠ Phase 3 结论成立；疗效指标须按你们既定协议单独报告。

开箱包完整推理与适配说明见包内 `README.md`；本包不提供或触发训练。消融细节见同目录另两份「给队友」文档；**phase3 以本文件为准。**
