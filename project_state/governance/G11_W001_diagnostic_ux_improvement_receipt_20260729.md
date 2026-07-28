# G11 W001 诊断往返体验改进回执

## 结论

G11 复核为 **PASS**；本次只完善诊断治理工具，G12、服务器训练、attempt 启动、run unit 消耗和 result import/accept 均为 **NOT RUN**。

## G11 反向核验

- 远端治理分支：`codex/w001-mpp2-huber-loss-20260728` → `45071ebc822c8eab859c4269d56391e610605a75`
- 远端诊断回传：`automation/diagnostics/w001-g11-environment-probe` → `486a9af5c76df81cdb2b7b7c14be8d85f54ee123`
- 回传文件：`environment_probe.json`，`1316` bytes，SHA-256=`86c836713d66e8f82d54ae3c145084e9b881db8e253cb8bb44da485bfeb397cd`
- 诊断门禁：`PASS=5 WARN=0 FAIL=0`
- 严格门禁：`PASS=7 WARN=6 FAIL=0`；六项均为既有历史债务

## 已实现的体验改进

1. 新增 `diagnostic run-allowlisted`：当前仅允许固定 `environment_probe` 收集器，不接收任意 shell 文本；其他虽在 request allowlist、但没有固定 runner 的命令会明确阻塞。
2. 新建诊断 request 时自动生成 `operation_cards.md`，主动解析 `configs/server_paths.yaml` 中的服务器仓库和自动化工作树路径，并给出本地发布、服务器受限执行/回传、本地取回/验证三张操作卡及停止条件。
3. `environment_probe.json` 固定以 UTF-8 无 BOM、LF 换行写入；runner 写入后立即校验字节契约并计算 size/SHA-256。
4. 本地 `diagnostic record` 在登记前再次校验唯一输出路径、UTF-8/LF、diagnostic ID、source commit 与 `exit_code=0`，从入口阻断 G11 遇到的 CRLF→LF 哈希漂移。
5. server runner 只在注册的 `server_automation_worktrees` 下创建 detached、clean、exact-source-commit 工作树；不删除或重置现有工作树。

## 验证

- 诊断定向测试：`2 passed`
- 除既有 local-only 文档链接用例外的回归：`169 passed, 1 deselected, 2 warnings`
- 严格门禁：`PASS=7 WARN=6 FAIL=0`
- 未过滤全量测试仍有 1 个既有工作树完整性失败：`PROJECT_GUIDE.md` 链接了在该 governed worktree 中未复制的 `availability=local_only` 文档（首个为 `CLAUDE.md`）；这不是本次诊断改动引入，未通过复制用户本地文档或放宽测试来掩盖。

## 修改范围

- `scripts/pfmval_state.py`
- `deploy/pfmval_ops.py`
- `tests/test_pfmval_state.py`

