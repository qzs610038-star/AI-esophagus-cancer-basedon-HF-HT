# Phase2 软连接对比学习 v3（设计 v4）

这是一个可手动复制的独立实验包。包名保持 `phase2_softlink_contrastive_v3`，设计版本为 **v4**。模型、损失、逐折拟合、训练编排、缓存、指标和续跑实现均在包内；代码完成不代表真实训练已经运行、结果已经登记或结论已经确认。

当前代码版本为 **v3.0.5**：保留 v3.0.4 的批内唯一性、冻结 CLS 缓存和旧快照梯度检查点兼容修复；新增独立的 XZY 只读外部推理入口。该入口只读取已完成运行、服务器权重、XZY 原始标签和图像，不训练、不选择模型，也不改写原运行状态。

本轮服务器排障、问题拆分和新对话交接入口见 [`docs/debugging_20260909/README.md`](docs/debugging_20260909/README.md)。

## 运行入口

服务器上把整个包复制到：

```text
D:\AIPatho\qzs\code\phase2_softlink_contrastive_v3
```

服务器解释器固定记录为 `C:\Users\AIPatho1\pfmval_env\Scripts\python.exe`。依赖须预先安装；启动器不联网、不自动安装依赖。

原划分入口先跑种子 42：

```powershell
.\run_original_split.ps1
```

它只执行 `original`：六患者 9472 训练点 / 1078 内部验证点的原空间块划分，不会启动 LOPO。可显式传入种子、批次、路径覆盖及最小验证模式：

```powershell
.\run_original_split.ps1 -Seeds 42,43,44 -Batch v4_full_diagnostic
.\run_original_split.ps1 -PlanOnly
```

完整六折入口单独运行：

```powershell
.\run_lopo6.ps1
```

默认折为 `HYZ15040,JFX,LMZ12939,TGC,XSL,ZHZ`，默认种子为 `42,43,44`。新运行可用 `-Folds` 选择一个或多个独立折，并用 `-Seeds` 选择种子；42 负责本折学习率选择，43/44 依赖同一运行中已完成的 42。两个入口都支持 `-Resume` 和 `-PlanOnly`；续跑严格复用原运行快照中的折与种子，跳过已完成任务，不把原入口转换为六折。

续跑必须显式指定既有的 `<run>` 目录；`run.json` 中的 `experiment_id`、`protocol`、`batch_id`、`run_directory` 和 `weight_directory` 会先校验。续跑复用原 `run.json`、`config.json`、`package.json`、运行目录和权重目录；训练引擎只会按已完成 checkpoint 增量刷新 `model_weights.json`。实际使用的新代码版本及本次退出状态追加写入 `logs/resume_history.jsonl`，不把原始尝试的版本覆盖掉。原配置快照中的种子/折也会被复用。因此续跑时不要再传 `-Seeds`、`-Folds`、`-Batch`、`-RunsRoot`、`-WeightsRoot` 或 `-PythonInterpreter`：

```powershell
.\run_original_split.ps1 -Resume -ResumeFrom "D:\AIPatho\qzs\runs\phase2_softlink_contrastive_v3\v4_full_diagnostic\20260908_231946_516_67144155"
.\run_lopo6.ps1 -Resume -ResumeFrom "D:\AIPatho\qzs\runs\phase2_softlink_contrastive_v3\v4_full_diagnostic\20260908_231950_069_eb329779"
```

单独传 `-ResumeFrom` 或只传 `-Resume` 都会在创建任何新目录前失败。

### XZY 只读外部推理

XZY 使用独立入口，专门评估本次已完成的 `original/seed_42` 固定五轮与内部验证最佳轮 checkpoint：

```powershell
.\run_xzy_external.ps1 -RunDirectory "D:\AIPatho\qzs\runs\phase2_softlink_contrastive_v3\v4_full_diagnostic\20260909_101623_341_f0de1302" -PlanOnly
.\run_xzy_external.ps1 -RunDirectory "D:\AIPatho\qzs\runs\phase2_softlink_contrastive_v3\v4_full_diagnostic\20260909_101623_341_f0de1302"
```

`-PlanOnly` 先只读核验运行完成状态、XZY 1039 个标签/图像身份、30 通路顺序、坐标网格和全部 checkpoint，不创建输出目录。XZY 默认原生步长为 224，并要求所有坐标位于同一 224 网格；若服务器事实不同，应先查证后用 `-NativeStep` 显式覆盖，不能为通过检查而随意修改。推理中断后保留现有产物，修复问题后使用同一命令加 `-Resume` 续跑。

所有新增文件严格位于：

```text
<RunDirectory>\external_xzy\
```

其中包括只读输入清单与核验记录、任务计划和状态、冻结 UNI CLS、各 checkpoint 的预测/标签与完整指标、`metrics_summary.csv` 和完成摘要。两种 PCC 分别记录为逐通路平均 `patient_macro_pathway_pcc` 与整体展平 `pooled_pcc`。XZY 历史上已看过，只能作为外部参考评估，且结果不参与学习率、epoch、checkpoint 或 v4 模型选择。

如系统限制脚本执行，可只对本次进程放宽策略：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_original_split.ps1 -PlanOnly
```

也可把 `-PythonInterpreter`、`-RunsRoot`、`-WeightsRoot` 和 `-Config` 指向本地临时目录做接口验收；正式服务器路径仍以 `config.json` 和 `configs/server_paths.yaml` 的活动主表为准。

## 输出与路径合同

每次调用都新建时间加随机标识的 `<run>` 目录，不使用哈希。默认布局为：

```text
D:\AIPatho\qzs\runs\phase2_softlink_contrastive_v3\<batch>\<run>
D:\AIPatho\qzs\weights\phase2_softlink_contrastive_v3\<batch>\<run>
```

两棵树绝不混用：运行目录保存日志、原始预测/标签/元数据、指标和配置快照；模型权重只保存到 `weights` 树。运行目录必须包含 `model_weights.json`，登记服务器权重目录以及 `warmup`、`formal`、`last` checkpoint 的绝对路径和状态。手动回传时不带权重文件或大缓存，只复制需要的运行目录到：

```text
D:\AI空间转录病理研究\PFMval_new\experiments\results\phase2_softlink_contrastive_v3\<run>
```

服务器替换代码时只替换 `code\phase2_softlink_contrastive_v3`，保留既有 `runs` 和 `weights`。本包不自动压缩、同步、提交或发送消息。

## v4 设计摘要

- 五主格：`frozen_regression`、`frozen_centered_contrastive`、末四块 `r8_regression`、`r8_global_contrastive`、`r8_centered_contrastive`；另有同一末四块范围的 `r2_regression` 容量诊断格。
- 三档 LoRA 学习率候选 `{1e-5, 3e-5, 1e-4}` 只在每协议/折的种子 42、末四块 r=8 回归格上试点；用第 5 轮内部患者—通路等权 PCC 选择，并以患者等权 zMSE、再以较小学习率打平。选定值供全部 r=8/r=2 格使用。
- H/C 学习率固定 `3e-5`，教师通路投影 T 学习率 `3e-4`；共同预热的 H/C 为 `3e-4`。主轮数固定 5 轮，另作第 1–5 轮的内部验证最佳轮敏感性表，不能把敏感性结果与固定端点混称。
- 真实 batch 只在训练数据短测 64/128：最多 10 个预热更新加 30 个计时更新，统一使用 BF16 阶段一精度，记录吞吐、峰值已分配/保留显存、设备余量和评估耗时。batch=128 必须保留至少 2 GiB 显存且每图吞吐不低于 64，否则回退到 64；短测不选择预测模型，正式格从共同起点重置。
- 教师使用训练块拟合的全局 `z` 或中心化 `z−m`；`rho=0.25`、`tau_z=0.1`，`tau_y` 从 `{0.03,0.1,0.3,1,3}` 按 20 个均衡训练批次的非对角归一化熵接近 0.7 选择，同距取较大值，并逐协议/折重算。教师停止梯度，双向分别归一化。
- 空间图单跳、半径 1.5 倍原生步长、最多 8 邻居、形态温度 0.2、中心原始权重 1，患者之间不连边。末四块只更新独立 Q/V，K 增量为零；r=8/r=2 的 `alpha/r` 均为 2。
- 完整诊断版会保存固定诊断点的 CLS/H 位移、Q/V 实际 ΔW、u/v 分布、前两次更新的分项梯度、教师熵/有效候选数/质量、抽样次数和缓存一致性；并输出双 PCC（逐患者逐通路平均 PCC 与整体展平 pooled PCC）、CCC/zRMSE/zMAE/rawMAE/R²/bias、去均值误差分解及可计算的空间残差指标。EVA、标签 queue、activation centering、患者内激活中心化和提前 ZHZ 单折均禁用。

## PlanOnly 最小验证

`PlanOnly` 只应解析配置、协议、种子/折、路径合同和兼容续跑信息，写入启动日志/计划摘要后退出；不加载 UNI、不读大数据、不训练、不创建 checkpoint，也不选择模型。例如：

```powershell
.\run_original_split.ps1 -PlanOnly -RunsRoot "$PWD\_plan_runs" -WeightsRoot "$PWD\_plan_weights"
.\run_lopo6.ps1 -PlanOnly -RunsRoot "$PWD\_plan_runs" -WeightsRoot "$PWD\_plan_weights"
```

对既有运行做最小续跑计划检查：

```powershell
.\run_original_split.ps1 -Resume -ResumeFrom "D:\AIPatho\qzs\runs\phase2_softlink_contrastive_v3\v4_full_diagnostic\<run>" -PlanOnly
```

包内 `tests/test_entrypoints_config.py` 只解析 JSON、做静态入口隔离检查并 mock 共享编排器；它禁止训练、联网和真实数据访问。

## 科研表述限制与待核验项

本入口的原划分结果只能表述为“同患者空间块留出条件下的候选增益，待患者留出复核”；完整六折是患者留出复核，不是全新盲测队列。六名患者均参与项目开发，XZY 历史上已看过，只能作参考评估。任何结果在 Registry 明确为 `accepted` 前只能称探索结果或待审结果。

LOPO 标签必须从活动主表的原始模板 `D:\AIPatho\Patch\visiumhd_patch\{group}\{patient}\{patient}_ssGSEA.csv` 读取，并在每个协议/折的训练块上重新拟合 z 标准化；既有 `ssGSEA_zscore` 仅作参考，不能替代逐折拟合。UNI2-h 使用 `configs/server_paths.yaml` 登记的精确快照目录；该目录及其中唯一完整的 `pytorch_model.bin` 仍须在服务器现场核验，加载器不会联网下载。

XZY 不会被拼入开发清单；v3.0.5 仅在训练和模型选择全部结束后，从原始标签首列重建点位身份，并对图像集合及坐标网格做只读、失败即停的核验。真实服务器上的 XZY 输入、权重加载和推理仍须以 `-PlanOnly` 与正式运行输出验证。配置中的大数据、原始标签模板和 UNI2-h 精确快照是绝对路径记录；包内只有六患者的小型划分清单。不要把代码存在、PlanOnly 通过或静态测试通过写成 XZY 推理完成或科研结论。
