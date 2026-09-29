# 最新 Phase 2 基线参考

本目录复制自已验收的 `phase2_fullfov_hpo_v1 / 20260916_231813_101_7b1b9a79` 交接材料。

- `weights/formal_best.pt`：UNI2-h、全视野、spatial、seed45、formal、第 28 轮。
- `assets/model_provenance.json`：权重身份。
- `assets/seed_sources.json`：seed45、46、47 的来源与成绩。
- `assets/pathway_manifest.json`、`zscore_params_from_train.json`：30 维顺序和训练集标准化口径。
- `baseline_metrics.json`：seed45 与三种子摘要。
- `training_reference/`：有效配置和训练复现代码；已排除 `__pycache__`/`.pyc`。

模型结构是 1536→256→30，最新图参数和训练超参数以 `training_reference/effective_config.json` 为准。`training_reference/reproduce_seed45.py` 是复现训练入口，保留是为了队友需要时按另行授权复现；本次聚合复核不得调用它。

该 checkpoint 只用来核对当前基线和严格加载，不得作为任务二任何一臂的初始化、续训或微调来源。服务器原路径见根目录 `server_locations.json`；本包没有远程验证服务器原件。

