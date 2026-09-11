# Phase 2完整继承单点模型与空间修正

**状态：代码包已实现并通过本地合成验收，尚未在服务器核验输入或启动真实训练。**

本轮仅种子42、三条轨迹：`spatial_residual_only`（固定H/C只训B，优先）、`point_continue`（续训H/C）和`spatial_joint`（续训H/C并训练B）。全部严格继承阶段一冻结纯回归第5轮H/C；B从零开始；不使用LoRA（低秩适配）、对比学习、子图采样或外部测试选模。

## 上传与服务器预检

将整个目录复制到：

```text
D:\AIPatho\qzs\code\phase2_spatial_warmstart_v1
```

先在该目录检查已登记的阶段一checkpoint和两类缓存。检查只读取训练/内部验证，不读取XZY标签：

```powershell
& 'C:\Users\AIPatho1\pfmval_env\Scripts\python.exe' -X utf8 .\src\main.py check --config .\config.json
```

服务器路径来自既有回传登记，本机未验证。预检必须确认：来源`epoch_5.pt`存在、H/C四键完整、缓存字段与维度正确、预测缓存与冻结图缓存的患者顺序及CLS值一致、点位能按`patient + patch_stem`映射到包内划分和坐标。`slide_mapping_status.json`仍未核实独立slide_id；当前v1.1图合同按“患者且划分内”分组，预检会明确报告该警告，不会把patient_id伪装为slide_id。

## 训练与恢复

按预定顺序运行三臂：

```powershell
.\run.ps1
```

只运行一臂：

```powershell
.\run.ps1 -Arm spatial_residual_only
```

从用户明确指定的同一轨迹`last.pt`恢复。恢复会创建新的run目录，并登记`parent_checkpoint`，不会覆盖父运行：

```powershell
.\run.ps1 -Mode resume -Arm spatial_joint -ResumeCheckpoint 'D:\AIPatho\qzs\weights\phase2_spatial_warmstart_v1\warmstart_no_lora_v1\<parent_run_id>\spatial_joint\seed_42\last.pt'
```

输出分别写入：

```text
D:\AIPatho\qzs\runs\phase2_spatial_warmstart_v1\warmstart_no_lora_v1\<run_id>
D:\AIPatho\qzs\weights\phase2_spatial_warmstart_v1\warmstart_no_lora_v1\<run_id>
```

每臂按`raw/<arm>/seed_42`与`weights/<arm>/seed_42`保存历史、逐点预测、图重建数组、step0一致性报告及`warmup.pt`、`formal.pt`、`last.pt`、`model_bundle.pt`。`last.pt`包含模型、优化器、更新计数、最佳候选、早停参照/计数及Python、NumPy、PyTorch、CUDA和专用dropout随机状态。顶层输出`selection.json`、`metrics.json`、`model_weights.json`；训练入口不会读取或评价XZY。

## 独立推理与外部评价

推理输入NPZ必须提供：`features`、`graph_features`、`patient`、`patch_stem`、`x`、`y`、`native_step`；可选`partition`。点模型不建图，空间模型按bundle中的固定图参数建图。推理不接收标签：

```powershell
python .\src\main.py predict --bundle '<model_bundle.pt>' --input '<inference_input.npz>' --output '<predictions.npz>'
```

方法与checkpoint固定后，外部评价输入再增加已按来源训练统计标准化的`target`：

```powershell
python .\src\main.py external-eval --bundle '<model_bundle.pt>' --input '<xzy_standardized_input.npz>' --output '<external_output_dir>'
```

本次XZY外评可直接复用既有阶段一固定E5缓存；其中`target_z`已使用来源训练集统计标准化，不会用XZY重新拟合参数。将更新后的代码包复制到服务器后运行：

```powershell
.\run_external_xzy.ps1
```

脚本只评价内部验证已选定的`spatial_joint/seed_42/model_bundle.pt`，不训练、不选模。结果写入原训练运行目录下的`external_xzy/spatial_joint/`。若服务器路径不同，可通过脚本参数`-SourceCache`、`-Bundle`、`-Output`覆盖。

若把`spatial_residual_only/seed_42/warmup_bundle.pt`作为bundle运行同一命令，入口会额外生成来源step0的固定β=1平滑诊断；它仍标记为不参与选择。外评输出同时保存`graph_reconstruction.npz`，供本地精确复核。

XZY输入由脚本从既有`stage1_fixed_e5/frozen_regression_e5.npz`现场转换；该缓存来自同一来源冻结H/C第5轮，已包含训练集z-score尺度的`prediction_z`与`target_z`。脚本不会读取XZY来拟合参数。若需原始尺度反变换，仍须从来源运行补回训练集mean/std；当前bundle如实记录`parameters_embedded=false`。

固定β=1预测平滑诊断由`src.predict.fixed_beta_one_smoothing`提供，严格包含中心自权重。它不训练、不搜索β、不参与选模；Ridge（岭回归）保持关闭。

回传运行目录后，在本地生成该派生诊断：

```powershell
python .\src\analyze_local.py --run-dir '<experiments/results/phase2_spatial_warmstart_v1/run_id>'
```

## 本地轻量验收

```powershell
python -m pytest .\tests -q -p no:cacheprovider
python .\src\main.py --plan
```

合成测试覆盖严格H/C加载、B=0逐点一致、图隔离与孤点、双PCC、三臂冻结和学习率策略、配对dropout、更新预算、完整恢复等价、标签无关推理、固定平滑及启动器batch目录。测试不启动真实训练、不访问网络、不计算哈希。

详细科研边界见[部署方案](docs/部署方案.md)，历史实现要求见[实现交接](docs/implementation_handoff.md)。当前没有新增真实实验结果，不将本实验登记为`done`或`accepted`。
