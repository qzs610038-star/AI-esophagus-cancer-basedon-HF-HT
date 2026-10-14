# 当前 Phase 2 主线标签

本目录是当前 `barcode-repair-20260711-d626ad8-v003` 的30维主线标签与 MPP2/group_2 冻结元数据副本，供审查原始30维处理、身份/划分/通路顺序，以及与任务二重构真值的独立比较；不得被候选聚合结果覆盖。任务二默认用核查后的真实基因一次性生成、两臂共用的重构真值，不以 V003 逐值复现为开跑门槛。

## 文件

- `labels/`：训练、内部验证和 XZY 外部标签。
- `split_manifest.csv`：开发患者训练/内部验证的点位选择清单，共 9,472 + 1,078 行。
- `split_info.json`：患者、划分和数量说明。
- `zscore_manifest.json`：30 通路顺序、训练集拟合范围、`ddof=1` 和裁剪口径。
- `zscore_params_from_train.json/.csv`：训练集逐通路均值与标准差。

`split_manifest.csv` 不包含 XZY；XZY 使用 `labels/external/XZY/` 中 1,039 个点位。训练和内部验证 CSV 可能包含同一患者的更大点位集合，正式划分必须通过 manifest 的 `patient + patch_stem + split` 选择，不能把整个 CSV 当作目标 split。

固定列顺序是 `barcode` 加 `zscore_manifest.json` 中的30条 `pathway_names`。重构目标/预测输出保持同一名称顺序；是否与本目录的 V003 数值等价需另做诊断。新任务二两臂不得在评估分支分别重算真实基因真值。
