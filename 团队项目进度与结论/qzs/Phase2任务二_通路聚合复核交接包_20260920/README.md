# Phase 2 任务二通路聚合复核交接包

本包帮助原负责同学核查两段流程：最初原始基因如何得到30维 ssGSEA CSV；已有 raw30 CSV 如何经冻结划分和训练集标准化成为 V003。随后按核实后的真实基因/聚合合同完成任务二两臂消融。任务二目前仍为 `planned / pending_not_run`；历史319基因候选与主线 V003 不一致，不能未经审查直接当作已更正真值。

## 当前执行规则

开跑前确认基因列/尺度、基因集映射、聚合算法/参数、患者+点位连接和训练集专用标准化；由真实基因只生成一份静态30维重构真值，两个分支都读取它。纠正后，队友可在自己的完整目录直接开跑，无需先回传或再次批准。

- 与 V003 逐值比较是定位差异的诊断，**不是统一开跑门槛**。相同时可在可比口径下参照主线 seed45；不同时按“核实后的基因重构目标”独立报告两臂，不直接与主线分数比优劣。
- 基因臂仅对预测基因再次聚合；禁止两臂在评估分支分别重算真实基因真值。重构目标的通路 mean/std 来自该目标的训练集，不能在候选/预测值或 XZY 上重拟合；只有与 V003 的 raw 分数确实对齐时才沿用 V003 的参数。
- 原始 raw30 CSV→V003 有部分可核证处理记录；更上游原始基因→raw30 合同仍待队友核实。聚合合同先在开发数据固定，再看 XZY；XZY 不参与参数选择、早停或选模。

## 从哪里开始

- 队友先读：[给队友的交接说明.md](给队友的交接说明.md)
- 智能体先读：[给智能体的交接合同.md](给智能体的交接合同.md)
- 输入与全流程规范：[Phase2原始30维通路分数处理全流程.md](Phase2原始30维通路分数处理全流程.md)
- 实验结束后只需参考：[最简结果说明模板](return_template/最简结果说明模板.md)
- 本地证据边界和自检见：[audit_summary.md](audit_summary.md)

## 包内内容

| 目录/文件 | 用途 |
|---|---|
| `Phase2原始30维通路分数处理全流程.md` | raw30 CSV→划分/标准化→V003 的已知流程、服务器路径及上游待核事项 |
| `aggregation_review/current_phase2_labels/` | V003 30维标签、MPP2/group_2 划分、冻结顺序和训练集标准化参数；只读主线参考 |
| `aggregation_review/historical_319_candidate/` | 历史 319 基因候选及尺度；只用于查错 |
| `aggregation_review/comparison/` | 历史逐通路比较、图、摘要和专项审计材料 |
| `return_template/` | 可选本地合同模板和最简结果说明；完整数据不要求回传 |
| `baseline_reference/` | 最新已验收 UNI2-h 全视野空间模型 seed45 权重、有效配置和三种子成绩；权重只作参考 |
| `legacy_task2_reference/` | 旧聚合源码和旧配置；禁止作为正式运行入口或新实验配置 |
| `tests/validate_package.py` | 只读核对包内状态、标签身份/顺序、冻结配置、旧配置隔离和参考权重 |
| `tests/check_raw30_server.py` | 可选服务器只读核对 raw30→随包 V003 数值；不验证更上游基因→ssGSEA |

## 运行边界

根目录 `run.ps1` 仅支持 `check-package` 和 `check-server-env`，不会训练或重建标签。本包不包含正式任务二训练入口，因为原始基因、mapping 和更正后的聚合实现仍由负责同学在其工作环境中维护；“直接开跑”指输入/算法核实、共同真值冻结后在该完整工作目录运行。

服务器登记解释器为 `C:\Users\AIPatho1\pfmval_env\Scripts\python.exe`，正式全视野缓存根目录为 `D:\AIPatho\qzs\feature_caches\phase2_fullfov_hpo_v1`。不要使用裸 `python`，不要从 `legacy_task2_reference/configs/` 启动实验。

```powershell
.\run.ps1 -Mode check-server-env
.\run.ps1 -Mode check-package
```

本包未压缩、未计算哈希、未写服务器。正式开跑前 Registry 保持 `planned / pending_not_run`。
