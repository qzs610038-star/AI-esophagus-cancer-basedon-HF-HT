# PFMval 本地状态与 Gitee 维护实施日志

> 日期：2026-07-11  
> 性质：维护工作快照与后续交接参考，不是可变状态事实源  
> 当前事实源：`project_state/current_state.json`、`experiments/experiment_registry.json` 及其 active 方案

## 一、本轮目标

本轮维护用于解决项目中用户指令、实验结论、服务器路径和历史方案容易相互覆盖的问题，并在服务器只能通过 Gitee 通信的条件下，为后续训练、结果回传和自动排障建立可审计基础。

## 二、已完成工作

1. 建立受 Git 管理的状态包：`current_state.json`、追加式指令记录、文档注册表和 JSON Schema。
2. 生成 `CURRENT_STATE.md`，并将 `AGENTS.md`、`CLAUDE.md`、README 和本地 agent 视图收缩到统一事实源。
3. 将 Registry 保持为已验收实验事实源，Dashboard 改为派生视图；补充实验提交、作业、结果和数据版本溯源字段。
4. 在 `deploy/pfmval_ops.py` 建立状态、文档、路径、作业和结果统一维护命令。
5. 新增 `configs/server_paths.yaml` 和 `mpp_standard_splits/path_index.json`，统一维护服务器绝对路径、原始 ssGSEA、标准化标签、z-score 参数与 embargo 审计。
6. 建立 Gitee-only 作业/回传协议：作业绑定固定提交和干净 worktree，正式训练绑定显式用户批准，结果必须匹配原始作业清单。
7. 结果打包采用临时目录，导入采用带备份的事务更新；中断后可完成清理或回滚。
8. 停用旧 HTTP 远程命令入口；SSH、SCP、Remote Tunnel 仅保留为历史资料。
9. 将 MPP 训练门禁同时放入统一调度入口和 `train_mpp_uni2h_mlp.py`，直接运行脚本也不能绕过。

自动轮询 Gitee、Codex CLI 自动修复和正式训练等待器仍按原计划留作后续独立任务，本轮未实现。

## 三、验证结果

- 状态版本：revision 11。
- 自动化测试：28 项全部通过。
- 严格通用状态检查：5 PASS、6 WARN、0 FAIL。
- 本地项目检查：40 PASS、9 WARN、0 FAIL。
- 文档登记：134 份，其中 active 13、superseded 11、historical 95、missing 15。
- MPP 路径索引：102 个文件，共 79,816,901 字节。
- 新 MPP 训练调度返回阻断；直接运行训练脚本同样返回阻断。
- 现有 checkpoints、MPP 数据、缓存和用户未跟踪结果均未删除或覆盖。

## 四、发现的主要问题

### 1. MPP 标准化训练标签存在跨患者 barcode 污染

五组训练标签均出现同一 barcode 对应不同通路值的冲突：

| Group | 标签行数 | 唯一 barcode | 冲突重复组 |
|---:|---:|---:|---:|
| 1 | 35,942 | 19,640 | 11,354 |
| 2 | 11,158 | 9,472 | 1,686 |
| 3 | 45,151 | 35,227 | 8,457 |
| 4 | 3,043 | 2,553 | 490 |
| 5 | 11,936 | 9,166 | 2,372 |

根因位于 `scripts/rebuild_zscore_from_manifest.py`：六名患者数据合并为 `train_all` 后，写回单患者文件时仅按 barcode 成员关系筛选，没有同时按患者筛选。不同患者可能共享相同坐标格式的 barcode，因而其他患者的标签行被写入当前患者文件。

这不能通过保留第一行、最后一行或简单 `drop_duplicates` 修复，因为这些操作无法证明哪一行属于目标患者。当前 `labels_validated=false`，新 MPP 训练被硬阻止。

### 2. 历史结果证据边界

五个 2026-07-06 标准化 MPP 结果仍按原 Registry 状态保留，但缺少新协议要求的完整结果 envelope。发现当前标签生成缺陷后，这些结果不应直接作为修复后 LoRA 实验的同批次基线；需要先确认其实际输入版本，否则应在修复标签后重跑 MPP2 frozen baseline。

本轮没有擅自删除或改写历史实验结论，后续应通过单独的数据修复/结果迁移任务调整其证据状态。

### 3. 本地与服务器资产索引需要区分

原始 ssGSEA 主要位于服务器，且服务器只允许通过 Gitee 通信。修复后的大型标签不应伪装成本地已验证资产。后续任务需要增加服务器资产 manifest，记录服务器路径、大小、SHA-256 和生成提交，再由本地导入审计结果。

## 五、建议的后续顺序

1. 新开“MPP 标签跨患者污染修复”任务，禁止覆盖现有资产，先输出到版本化 staging 目录。
2. 为每行保留患者身份，以 `patient + barcode` 联合键生成标签；增加 raw、manifest、train、val 一对一断言。
3. 在 `dataset_mpp.py` 增加重复 barcode 硬失败，禁止隐式选择多行标签。
4. 服务器通过 Gitee 接收修复提交，重建 MPP1–5，并回传旧/新哈希、行数、唯一性、z-score 和 embargo 审计。
5. 本地导入修复证据，更新服务器资产 manifest、路径索引和状态包，严格检查通过后解除训练门禁。
6. 使用修复后的同批次数据重跑 MPP2 frozen baseline。
7. 再进行最多 3 epoch 的 MPP2 LoRA r=8 smoke；smoke 通过后向用户报告并等待正式训练批准。

在标签修复完成前，可以开发 LoRA 代码、注册 draft 实验和编写测试，但不得启动任何新的 MPP smoke 或正式训练。

## 六、交接状态

- 本轮代码和状态文件仍位于脏工作区，尚未由本轮维护操作暂存、提交或推送。
- 在推送 Gitee 前，应由用户审查改动范围并统一形成可复现提交。
- 后续 agent 必须先读取 `CURRENT_STATE.md`，不得仅依据本日志恢复状态。
- 本轮 agentmemory 召回为空；上述结论均来自当前仓库、状态包、代码和标签文件的实时检查。
