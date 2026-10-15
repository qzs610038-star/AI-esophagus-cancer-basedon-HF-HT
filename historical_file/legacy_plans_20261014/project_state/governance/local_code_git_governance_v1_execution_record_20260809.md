# 本地代码 Git 治理 v1 执行记录

## 1. 批准与边界

- 批准指令：`DIR-20260809-003`
- 指令状态：`completed`（state revision 188）
- 时间截止点：2026-06-30 23:59:59（含）
- 迁移前可恢复基线：`225acf96ff4bdf29c1004c8da0049a44a2cc0def`
- 旧代码退出提交：`240e8b1`
- W### 模板提交：`586a93d`
- 未执行：远端推送、服务器操作、训练、结果导入、数据或训练资产修改。

## 2. LA3 — 历史归档（Legacy Archive）

- 102 个时间候选完成唯一分类：101 个迁移，`lora_utils.py` 因 MPP2 LoRA 仍在使用而保留。
- 本机只读副本位于 `historical_code/pre_mpp1_5/`，由 Git 忽略。
- 本地 `migration_manifest.json` 记录 101 个归档路径、路径映射、批准批次和恢复提交；未计算文件内容哈希。
- 三旧模型目录迁移代码数：`histogene/` 10、`egnv1/` 8、`egnv2/` 8。
- 旧 CSV、图片、预测、checkpoint、分析文本等非代码资料未迁移。

## 3. DR4 — 依赖修复（Dependency Repair）

- 当前 MPP 训练指标统一使用 `pfmval_core.metrics`。
- `scripts/pfmval_assets.py` 不再把三旧代码目录作为固定保护规则或运行依赖登记。
- 活跃 Python 文件对 `histogene`、`egnv1`、`egnv2` 的 import 扫描结果：0。

## 4. WT5 — 工作树模板化（Workspace Templating）

- 新增 `experiments/workspaces/_template/`；模板本身不占用 W 编号。
- 模板包含 `code/`、`configs/`、`tests/`、结构化日志索引、`shared_code_manifest.json` 和 `CLOSEOUT.md`。
- 无训练入口已成功读取模板配置并输出 `{"mode": "no_training", "workspace_id": "W###"}`。
- 新 experiment 只有在 Registry 正式分配永久编号后，才复制为对应 `W###/`。

## 5. 最小验证

- `tests/test_asset_registry.py`、`tests/test_execution_roundtrip_v1.py`、模板测试：13 passed。
- `tests/test_metrics_compatibility.py`：5 passed，另有单样本 R² 不定义的预期 warning。
- `tests/test_pfmval_asset_policy.py`：1 passed。
- 迁移清单与归档实物：101/101 一致。
- 最终严格门禁：PASS 7、WARN 7、FAIL 0。
- 7 个 WARN 均为既有事项：五项旧结果 envelope 不完整、`PROJECT_GUIDE.md`
  过期、MPP 标准标签重复/冲突；本次治理未新增 FAIL。

## 6. 恢复方法

如需恢复旧代码，只从迁移前提交 `225acf96ff4bdf29c1004c8da0049a44a2cc0def` 按精确路径取回；禁止使用 `git reset --hard`、`git clean -fd` 或对工作区做模糊批量恢复。
