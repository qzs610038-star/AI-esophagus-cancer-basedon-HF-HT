# MPP2 pathway Ridge calibration r002：启动、门禁失败与 Gitee 回传排障记录

> 日期：2026-07-17
> 性质：historical operational record；仅记录工程排障和证据回传，不替代 `CURRENT_STATE.md`、active plan、experiment registry 或主对话的实验结论。
> Job：`mpp2-pathway-ridge-calibration-20260717-r002`
> Experiment：`mpp2_pathway_ridge_calibration_v001_20260717`
> Dispatch commit：`e2d583e5276958e995e717450cdee4c49367ff90`
> Bound source commit：`1cb75a33999dda9ed942107c068f8b5cdbc89a40`
> Returned-result commit：`b2d60a1fecf936376c215f87c8ecec876427c1ea`

## 1. 范围与安全边界

本记录覆盖从 r001 启动契约修复、r002 的内部稳定性门禁停机，到失败结果以 Gitee-only 方式回传的排障流程。全程遵守以下边界：

- 服务器与本地只通过配置的 `gitee` Git remote 交换代码和小型结果；未使用 SSH、SCP、HTTP 或 tunnel。
- 校准在内部验证集上完成嵌套 LOPO；稳定性门禁失败后立即停止，未以放宽阈值、跳过门禁或访问 XZY 的方式继续。
- 回传的是失败诊断证据，不是 accepted result；只有经本地 `job import` 及后续门禁审查后才可能更新 governed state。
- 不改写 checkpoint、MPP 原始 ssGSEA、标准划分、z-score 参数、manifest 或 embargo 审计。

## 2. 时间线与根因

| 阶段 | 现象 | 已确认根因 | 最小修复/处理 | 验证 |
|---|---|---|---|---|
| r001 启动 | `run_experiment.ps1` 向实验脚本追加 `--num_threads 8` 后解析失败 | `fit_mpp2_pathway_ridge_calibration.py` 未声明该 launcher 契约参数 | 为脚本加入 `--num_threads`，验证为正整数并传给 `torch.set_num_threads` | 针对 launcher 参数的回归测试通过 |
| r001 直接脚本执行 | 直接运行脚本时找不到项目根模块 | Python 以 `scripts/` 作为首个导入路径，项目根目录不在 `sys.path` | 在项目导入前显式插入 `PROJECT_ROOT` | `python scripts/fit_mpp2_pathway_ridge_calibration.py --help` 成功；回归测试通过 |
| r002 训练 | 内部六患者数据加载完成后，抛出 `RuntimeError: internal nested-LOPO stability gate failed; calibrator was not frozen and XZY was not read` | 这是预注册的安全门禁，而非可通过重试规避的启动异常；诊断文件在失败前已写出 | 保持 `failed` 状态，收集内部诊断；不冻结 calibrator，不读 XZY | 控制台明确确认 XZY 未被读取 |
| 首次结果打包回传 | `git diff --cached --check` 非零，脚本把诊断文件格式告警误当作不可回传错误 | 回传脚本将 Git whitespace 检查设为硬失败；若格式化原始证据，会破坏 `result.json` 中的 SHA-256 | 保留 `diff --check` 输出为 WARN，不改写产物，继续提交 | 结果包保留原始字节并成功提交 |
| 回传续传 | 对已存在工作树报“不是预期 Git 工作树” | Git 在 Windows 上的 `rev-parse --show-toplevel` 可能返回 `/`，而 PowerShell 比较对象使用 `\\`；字符串相等判断误报 | 改用 `rev-parse --is-inside-work-tree` 加上当前分支精确校验；不依赖路径分隔符字符串相等 | 复用原工作树后成功推送 |

## 3. 失败结果的最小回传包

本次在失败前只生成并回传了以下 allowlisted 文件：

- `nested_lopo_metrics.json`
- `pathway_decisions.csv`
- `result.json`

返回分支为 `automation/server/mpp2-pathway-ridge-calibration-20260717-r002`，远端已核验指向提交 `b2d60a1fecf936376c215f87c8ecec876427c1ea`。该提交新增 3 个文件、915 行，推送大小约 8.55 KiB。

未包含 `predictions_external_xzy.csv`，因为门禁失败时 XZY 尚未读取；这既是安全属性，也是本次失败回传完整性核验的一部分。

## 4. 可复用的服务器回传模式

### 4.1 先打包，再创建独立结果分支

1. 从派发分支创建独立结果 worktree，分支名必须等于 job manifest 的 `result_branch`。
2. `job pack --status failed` 只传入已存在、且位于 allowlist 的诊断文件。
3. `result.json` 与 artifact 的大小、SHA-256 绑定后，不得格式化、去尾随空格、换行转换或重新序列化 artifact。
4. 暂存时仅允许 `automation/results/<job_id>/` 有改动；发现其他文件即停止。
5. 仅将该结果分支推送至 `gitee`，由本地 fetch、校验、`job import` 完成接收。

### 4.2 Windows/Git worktree 兼容检查

不要使用以下形式判断工作树身份：

```powershell
(git -C $ReturnRoot rev-parse --show-toplevel).Trim() -eq $ReturnRoot
```

在 Windows 上，Git 与 PowerShell 可能分别返回 `/` 和 `\\` 分隔符，造成假阴性。应使用语义检查：

```powershell
$Inside = (git -C $ReturnRoot rev-parse --is-inside-work-tree).Trim()
$Branch = (git -C $ReturnRoot branch --show-current).Trim()
if ($Inside -ne 'true' -or $Branch -ne $ResultBranch) {
    throw 'Return worktree identity mismatch.'
}
```

### 4.3 `git diff --check` 的正确处置

`git diff --cached --check` 仍应保留为可见性检查，但对已打包、由 SHA-256 保护的结果 artifact，它不能触发自动格式化，也不应阻止失败证据回传。推荐模式：

```powershell
git -C $ReturnRoot diff --cached --check
if ($LASTEXITCODE -ne 0) {
    Write-Warning 'Artifact formatting warning retained; do not rewrite SHA-bound evidence.'
}
```

随后仍必须执行结果 envelope 校验和 job-binding 校验；Git 格式告警不替代，也不削弱这些完整性校验。

## 5. 未来同类故障的决策树

1. **启动前 argparse 或导入失败**：先复现并检查 launcher 向脚本注入的固定参数，再检查脚本直接执行时的 `sys.path`；为两者各保留一个回归测试。
2. **统计/稳定性 gate 失败**：将其视为实验结果状态，不是运行时异常。先收集内部诊断，禁止跳过 gate、放宽阈值或提前访问 external XZY。
3. **已有失败输出需要回传**：使用 `job pack --status failed`；仅收集 allowlisted、实际存在的文件，避免复制大模型或受保护资产。
4. **Git whitespace 告警**：保留告警与原始字节，不执行修复性格式化；依赖结果 envelope 的 SHA-256 保护。
5. **结果 worktree 续传失败**：校验“是否 Git worktree + 是否正确分支 + 是否存在 `result.json`”，不要用 Windows 路径字符串直接比较。
6. **本地接收**：先 fetch 专用 `automation/server/<job_id>` 分支，验证 bundle 与 job binding；只有验证通过才执行 `deploy/pfmval_ops.py job import`，并由 state sync/门禁更新派生状态。

## 6. 交付状态

- Gitee 回传：完成。
- 失败结果包：已在专用分支保存，尚不在本日志中宣告 accepted 或 promoted。
- 实验的统计学结论、内外部门禁判定及任何后续协议变更：由主对话基于返回 bundle 独立处理；本日志不重复或覆盖该结论。
