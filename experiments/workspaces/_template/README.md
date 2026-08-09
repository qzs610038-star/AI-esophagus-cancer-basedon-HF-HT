# W### 实验代码包模板

创建新实验时，按 `project_state/workspace_registry.json` 分配的永久编号，将本目录复制为 `experiments/workspaces/W###/`。模板本身不占用编号。

目录职责：

- `code/`：本实验新增或修改的全部实验专属代码。
- `configs/`：本实验专属小型配置。
- `tests/`：只覆盖本实验变更的最简测试。
- `logs/`：Git 仅保留摘要和结构化索引；原始大日志写入工作树外的已注册 W### 运行目录。
- `shared_code_manifest.json`：记录使用但未修改的公共代码及其 Git commit ID，不复制公共代码。
- `CLOSEOUT.md`：记录整包合入、公共代码晋升和当前树退出结果。

关闭顺序：完整 W### 包独立提交合入 main → 晋升可复用代码 → 更新 closeout 与治理索引 → 经审核后让完整包退出 main 当前树；Git 历史保留完整包。
