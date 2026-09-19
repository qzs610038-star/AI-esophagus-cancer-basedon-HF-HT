# 训练实现与复现入口

`effective_config.json` 是实际训练交付权重的生效配置，`frozen_selection.json` 记录仅依据内部验证冻结的配方。`src/model.py`、`train.py`、`graph.py`、`selection.py`、`search.py`、编码器与变换实现保留原实验代码。本次未修改训练科学行为、配方或正式权重。

新训练仅通过用户显式执行 `reproduce_seed45.py` 启动，具体命令见上级 README。它先核对缓存、标签和身份，再从头训练种子 45，并写入新的运行/权重目录。原始图像、大缓存和编码器权重不随包分发。多阶段入口用于追溯，运行时也需要原始大数据资产。

交接版本 1.1.0 对三个便携读取位置作了定向修复，因此不能将整个目录描述成逐字未改动的历史快照：

- `reproduce_seed45.py`：使用 `--feature-caches-root` 时，从原配置构造共同身份清单的旧位置到新位置映射，并在新运行保存 `reproduction_inputs.json`。
- `src/stage_runtime.py`：向缓存读取器传入该明确位置映射。
- `src/feature_cache.py`：只接受已声明共同身份清单的位置变化；身份顺序、有限值、维度、模型版本和预处理仍须匹配，且不改写原缓存。

缓存根参数须指向包含 `phase2_fullfov_hpo_v1/` 的父目录；共同身份清单与 CLS 缓存必须一起保留。只移动缓存文件夹而遗失身份清单不能复现训练。
