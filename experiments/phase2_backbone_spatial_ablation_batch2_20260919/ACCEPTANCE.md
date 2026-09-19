# 代码包验收记录

结论：2026-09-19 部署方案对应的独立代码包已完成本地工程验收。服务器模型下载、特征提取、正式训练和科研结果验收尚未开始。不能把本记录解释为实验结果已经完成或 accepted。

验收日期：2026-09-19。代码版本：`v002`。协议：`fullfov-spatial-backbone-batch2-v1`。

## 已完成范围

- 独立包位于 `experiments/phase2_backbone_spatial_ablation_batch2_20260919/`，从模板建立，运行时不从上级目录导入仓库公共代码。
- 三个新编码器的固定 repo / revision / loader / output_mode / 维数 / 原生归一化已写入 `inputs/model_manifest.json`；交付时 `snapshot_path` 为 `null`，这是正确的未下载状态。
- 已接纳 9 个全视野空间基线任务写入只读 `inputs/baseline_reference_manifest.json`，不重训；包内 accepted 指标是运行输入，原始绝对路径仅作来源追溯。
- 全视野几何与 `imagenet_rgb_v1` / `hoptimus_rgb_v1` 分字段实现；Phikon 路径不使用 `AutoImageProcessor`。
- 冻结 `spatial-11`：Adam 1e-4、wd 0、batch 32、hidden 256、dropout 0.3、图参数与选模窗口（formal 从第 1 轮、早停计数从第 41 轮、patience 20）。
- `run.ps1` 默认 `check-inputs`。下载使用独立 `download_models.ps1`，支持按模型失败继续和 `-RegisterOnly`；执行阶段支持 `-Models` 定向运行及失败后继续。
- 新特征缓存键包含模型与预处理配置；主缓存只有选定单 patch 特征 + `COMPLETE` + `metadata.json`。

## 本地验证证据

在代码包根目录执行：

```powershell
python -m pytest tests -q --tb=short
python -m compileall -q src runner.py
```

结果：

- pytest：40 项通过；无失败、错误或跳过。
- Python 编译：`src` 与 `runner.py` 全部通过。
- 独立性：复制到临时目录并清除 `PYTHONPATH` 后，`runner.py --help` 成功；`src` 没有上级仓库 import、`AutoImageProcessor` 或 `search` 入口。
- 适配器：二维 embedding、三维 token CLS、Phikon `last_hidden_state[:,0,:]`、错误 rank / 维数 / NaN 均按合同失败。
- 变换：边缘标记方图四边保留；非方图失败；两套归一化数值不同。
- 缓存：缺 `COMPLETE`、身份或 revision 不一致不得复用；拒绝 `mean_patch`。
- 训练头：同维同种子 H-optimus-0/1 与 UNI2-h、Phikon-v2 与 UNI 的初始 H/C/B state dict 精确相等；B 零初始化且无 bias；H/C/B 有限梯度；编码器不进入 Adam；孤立点退化为中心预测；图边随当前特征变化；早停从第 41 轮计数。
- 调度：完整范围仍为 9 个新空间任务；`-Models` 只允许选择完整三种子子集；一个模型的特征、训练或外部评估失败不会阻止后续模型；执行阶段 `partial` 返回非零，`analyze-local` 的不完整报告可正常落盘。
- 基线：缺臂、种子或包内 accepted 指标会报错；原始来源路径在当前机器不存在时不阻断 `check-inputs` 或分析，且不会触发基线重训。
- 下载：401/403 不当作可重试网络失败；一个下载失败不阻止其余模型尝试；H-optimus-0 固定 revision 使用实际存在的 `pytorch_model.bin`。

## 服务器待完成项

1. `run.ps1 -Action check-environment`
2. 按 `docs/服务器HuggingFace登录与三模型下载操作卡.md` 完成授权、固定 revision 下载及严格加载形状检查
3. 显式安装并通过预检的 `transformers` 版本（如缺失）；不要改 torch
4. `run.ps1 -Action prepare-features`
5. `run.ps1 -Action train-spatial`
6. 正式检查点齐全后 `run.ps1 -Action external-eval`
7. 回传含 `model_weights.json` 与 `feature_caches.json` 的运行目录，本地 `analyze-local`

未核实事项（不阻断代码交付）：服务器快照存在性、transformers 兼容版本、GPU 显存与提取吞吐、H-optimus 官方 0.5 µm/px 与当前 patch 物理尺度、来源组是否等于真实切片范围。

本次未下载权重、未创建服务器特征缓存、未运行正式训练、未修改实验 Registry、未提交 Git，也未打包压缩或远端同步。
