# 旧任务二聚合实现参考（禁止用于正式启动）

> 本目录不是可运行的新实验包。不得从这里启动 smoke 或正式训练，也不得把 `configs/` 复制为新配置。

本目录只保留旧任务二与 319 基因/ssGSEA 聚合直接相关的配置和源码，不含旧 checkpoint、训练输出或压缩包。

- `configs/01_task2_direct_common.json`：直接 30 维臂。
- `configs/02_task2_gene_reconstruct.json`：319 基因后重构臂。
- `src/pathway_reconstruction.py`：预生成真值、预测后重构和 GSEApy 调用。
- `src/pathway_scale.py`：通路尺度处理。
- `src/targets/`：319 基因读取、标准化和 ssGSEA 目标定义。
- `src/main.py`、`preflight.py`：旧调用链和输入预检。
- `修改位置索引.md`：建议重点审查和重构的位置。

这些文件是历史证据，不是已批准的新实现。旧配置使用 seed42、旧缓存、半径1.5/8邻居、温度0.2、自身权重1、14,800 updates 和旧 MSE 选模，与现行合同冲突；旧代码还会在分支内重新聚合真实基因。正式任务二须先核实输入/算法、生成两臂共同重构真值，并采用 `给智能体的交接合同.md` 的最新口径；V003 只作主线参照。
