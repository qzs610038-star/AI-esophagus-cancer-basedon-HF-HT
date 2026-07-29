# G15 主线整合与 G16 启动准备回执

> 日期：2026-07-30
> 范围：G15 本地公共治理能力的主线接纳，以及 G16 新对话启动条件准备。
> 边界：未操作真实服务器、未 dispatch、未训练、未消耗 run unit、未 import/accept 结果。

## 1. G15 主线整合

- 来源工作树：`W002`
- 来源分支：`codex/w002-server-execution-roundtrip-v1-20260729`
- 来源及 Gitee SHA：`05df5debe1ece3305604310463643c2f0a3c3f61`
- 主线原 HEAD：`e19ac19c54c3c5caf19122b5da896d18e53529ad`
- 整合方式：`git merge --ff-only`
- 变更范围：19 个 G15 源码、Schema、fixture、测试、Registry/状态和审计文档文件。
- 用户原有未跟踪文件
  `project_state/plans/workflow_governance_v3_remaining_goals_20260727.md`
  保持未跟踪，未进入整合或暂存范围。

## 2. 独立验证

- W002 全量：`204 passed, 2 warnings`
- 整合主线受控测试域：`184 passed, 2 warnings`
- strict start-check：`PASS=7 WARN=6 FAIL=0`
- paths validate：`PASS=0 WARN=1 FAIL=0`
- 公共 CLI 冒烟：
  - 从 `automation/jobs/W001-A003/job.json` 构建 `execution_bundle_v1`
  - build 与独立 validate 均返回 `status=valid`
  - `bundle_id=3067ded018d4c50f5002c29e43e324bfecb8132ddbc374b42f82d4305db663f9`
  - `file_count=22`
  - 仅构建/校验，没有 dispatch 或训练

两条 pytest warning 为既有少样本 R² 未定义提示；strict 的六条 warning 为既有
legacy result envelope 与受保护重复 barcode 债务。

## 3. 治理状态

- `DIR-20260730-001` 已通过 CLI append-only 转为 `completed`。
- 新增 active directive：`DIR-20260730-002`，用于 G16 启动准备。
- G16 详细计划：
  `project_state/plans/scientific_records_lifecycle_v1_upgrade_plan_20260729.md`
- 该详细计划在 Document Registry 中仍为 `historical/reference`，不能单独作为执行授权。
- G16 的当前执行权威是 `DIR-20260730-002` 加新对话中的明确用户指令；详细计划只提供实施细节。
- `workspace_registry.next_workspace_number=3`，因此 G16 应绑定永久治理维护工作树 `W003`。

## 4. G16 启动门槛

新对话开始时必须：

1. 明确写出 `本对话工作树：W003`。
2. 读取 `DIR-20260730-002`、本回执和科学记录升级计划。
3. 核对 W003 路径、branch、HEAD、dirty 和 workspace registry 后才写代码。
4. 保持 G15 的 execution bundle/preflight 扩展接口，不回退或复制 W002 专用实现。
5. 继续禁止训练、真实服务器操作、结果改写和受保护资产修改。

以上条件满足后，可以开始 G16 的 TDD 实现。
