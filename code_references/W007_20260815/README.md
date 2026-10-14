# W007 历史源码参考

这里保存 W007 四臂残差探针的历史实现，供交接阅读和核对。来源为既有提交 `0f81be3cd542baa19ef3a2fe97bb146122a055bb`，原目录 `experiments/workspaces/W007/mpp2_cpgcr_probe`；恢复日期为 2026-09-20。

- `mpp2_cpgcr/model.py`：冻结基线与残差头。
- `mpp2_cpgcr/losses.py`：四臂损失。
- `mpp2_cpgcr/train_eval.py`：训练、内部选模及外部评价。
- `mpp2_cpgcr/one_click_runner.py`、`batch_plan.py`、`result_contract.py`：原运行编排、批次及结果合同。
- `configs/protocol_v001.yaml`：原协议。

文件按原字节恢复，清单见[来源.json](来源.json)。其中服务器、回传与自动化细节保留历史实现；本目录用途为源码参考，当前项目运行方式见根目录 AGENTS.md。

这份参考目录尚待用户随 GitHub 仓库同步。交接包中的实验代码索引已预留对应仓库路径；科研状态沿用原实验登记。
