# Phase2 任务三全视野密度消融 v1

状态：代码包与本地审计材料已生成，**尚未启动正式训练**。本包只实现任务三；任务二继续暂停，只保留真实标签差异审计。

当前代码版本：`1.0.1`。该版本将空间残差头偏置固定为关闭（`spatial_bias=false`），与已验收 `phase2_fullfov_hpo_v1` 及冻结seed45 checkpoint结构一致；`1.0.0`不得用于formal。

服务器部署、输入检查、smoke、formal、完整性核对和结果回传请直接按根目录的 [`服务器操作说明卡.md`](服务器操作说明卡.md) 执行。

## 实验边界

- 底座来自已验收的 `phase2_fullfov_hpo_v1`：full-FOV 224×224 bicubic、无裁剪/增强、UNI2-h CLS 1536维、`1536→256→30`、精确GELU、dropout 0.3、空间分支零初始化。
- 空间图固定为半径2、最多12邻居、σ=0.5、图像温度0.7716396069760084、自身原始权重0.5、有向一跳图。
- 优化器固定 Adam、学习率1e-4、weight decay 0、batch 32、恒定学习率、float32。
- seed45 已验收 checkpoint 只做冻结参考评估，不用于三臂初始化、继续训练或调参。
- 三臂均从零训练：稠密9,472点、固定相位规则2×2隔点2,383点、逐患者随机等量2,383点。
- 种子45/46/47。相同种子的三臂重新用同一种子构造模型，初始参数逐张量精确核对；随机等量掩码使用独立随机流。
- 内部验证1,078点和外部XZY 1,039点始终稠密。训练图只含该臂保留的训练点，抽稀后半径不扩大。
- 每臂标准器只拟合本臂训练点的raw真值；所有评价统一回到稠密训练集定义的z尺度。

## 训练与选择

- 固定14,800次参数更新，关闭早停。
- 主选择器：每296次更新验证一次，共50次，以患者—通路等权PCC选择，患者—通路等权z-MSE打破并列。
- 敏感性选择器：同一训练轨迹的前50个完整epoch每轮验证一次，共50次。
- 两个选择器分别保存 checkpoint。只有二者都冻结后才构建并评价外部XZY图。
- `formal` 先评估冻结seed45参考，再执行3臂×3种子共9次从零训练；没有任务二训练入口。

## 运行入口

服务器使用：

```powershell
.\run.ps1 -Action check-inputs
.\run.ps1 -Action render-label-audit
.\run.ps1 -Action smoke
.\run.ps1 -Action formal
```

`check-inputs` 会核对本包输入文件、已验收full-FOV缓存、冻结checkpoint、标签和服务器输出根，并在CPU上以 `strict=True` 实际加载冻结checkpoint核对模型结构。任何服务器检查未通过时命令返回非零；`smoke`/`formal` 不会静默回退到旧缓存或旧标签。

`smoke` 每臂只运行2次更新并标记为非正式证据。`formal` 才执行冻结参考和9次正式训练。正式结果写入 `D:\AIPatho\qzs\runs\phase2_task3_fullfov_density_ablation_v1\<批次>`，新权重写入 `D:\AIPatho\qzs\weights\phase2_task3_fullfov_density_ablation_v1\<批次>`；回传结果只登记权重与特征缓存路径，默认不复制其本体。

## 任务二标签审计

`src/label_audit.py` 是独立只读脚本，不接入任务三训练。它只加载连接键、`true_*` 和 `true_raw_*`，不加载任何 `pred_*` 或 `pred_raw_*`。本地材料仅覆盖内部验证和外部XZY，不推断训练集。

产物位于 `analysis/`：

- `task2_true_label_comparison_30_pathways.csv`
- `task2_true_label_comparison_30_pathways.png`
- `task2_true_label_comparison_summary.json`
- `task2_true_label_source_manifest.json`

解释与对接清单见 `docs/task2_ssgsea_truth_gap.md`。

## 非破坏性约束

LJQ原实验包、原始结果和已验收底座均为只读输入。本包不移动或修改它们，不生成任务二新标签，不修改任务二Registry状态，不计算哈希，不建立工作树，不压缩或远端同步。代码完成不等于实验完成，也不等于结果已接受。
