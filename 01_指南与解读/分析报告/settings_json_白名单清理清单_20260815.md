# .claude/settings.json 白名单清理清单（待用户审核）

> 生成日期：2026-08-15 | 生成依据：P1-8（P0/P1 治理落地）| 状态：**待用户审核，未修改 settings.json**

## 背景

`.claude/settings.json` 的 `permissions.allow` 共 **139 条**精确命令白名单，其中大量条目来自历史阶段的
SSH 直连排障、模型下载、旧 workflow 清理等操作，与现行治理边界（Gitee-only 传输、禁止 SSH/SCP/HTTP 远程命令/Tunnel、
GitHub origin 仅手动备份）**并存但不一致**。本清单按风险分类，供用户逐项批准后执行清理。

## 统计


| 类别           | 数量     | 建议                               |
| ------------ | ------ | -------------------------------- |
| 冲突项（建议移除）    | **31** | 与 Gitee-only / 禁 SSH / 禁自动推送边界冲突 |
| Skill 授权（保留） | 8      | `Skill(...)` 条目，保留               |
| 中性项（保留或评估）   | 100    | 本地 Python/Git 只读命令等，保留           |


## 冲突项明细（31 条，建议移除）

### A. SSH 直连 / 探测（21 条）——与"SSH 非 active 通道"直接冲突

1. `Bash(ssh -o ConnectTimeout=5 -T git@github.com)`
2. `Bash(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -T git@gitee.com)`
3. `Bash(ssh -vvv -i ~/.ssh/pfmval_server -o ConnectTimeout=15 ... AIPatho1@117.68.10.96)`（多条变体）
4. `Bash(ssh-keygen -lf ~/.ssh/pfmval_server.pub)`
5. `Bash(ssh-keygen -t ed25519 -f ~/.ssh/pfmval_new_test ...)`
6. `Bash(ssh-keygen -t ed25519 -f ~/.ssh/pfmval_server_nopass ...)`
7. `Bash(powershell ... Test-NetConnection -ComputerName 117.68.10.96 -Port 22/2222/22330 ...)`（多条）
8. `Bash(powershell ... Test-WSMan -ComputerName 117.68.10.96 ...)`
9. `Bash(powershell ... tracert ... 117.68.10.96)`
10. `Bash(netstat -ano)`

### B. 直连下载 / 推送（4 条）——与 Gitee-only、GitHub 仅手动备份冲突

11. `Bash(git clone *)`
12. `Bash(curl -L -C - -o .../model.safetensors ... huggingface.co ...)`（2 条，含 hf-mirror 变体）
13. `Bash(GIT_LFS_SKIP_PUSH=1 git push origin main)`
14. `Bash(git push *)`

### C. 旧机制清理命令（4 条）——目标文件已归档/删除，命令已失效

15. `Bash(rm -f ".../workflow_pending_test-session-002.json" ...)`（workflow 旧机制，2026-08-15 已归档）
16. `Bash(rm -f ".../workflow_pending_*.json" ...)`（2 条变体）
17. `Bash(rm -f ".../workflow_pending_plan-test-*.json" ...)`

### D. 其他（2 条）

18. `Bash(GIT_CURL_VERBOSE=0 git -c http.postBuffer=... fetch origin main --depth=1)`（origin 深度拉取，历史一次性）
19. `Bash(sed -i 's/<HF_TOKEN>/<HF_TOKEN>/g' .../settings.json && git add ... && git rebase --continue)`（历史一次性令牌清理）

## 建议执行方式

1. 用户审核本清单（可勾选移除/保留）；
2. 批准后由 Agent 编辑 `settings.json` 删除勾选项（保留 `block-git-clean.py` hook 与 Skill 授权）；
3. 删除后运行 `python -c "import json; json.load(open('.claude/settings.json'))"` 验证 JSON 有效性；
4. 提交本地 commit（不推送）。

## 审核记录

- [x] 2026-08-15：待用户审核（尚未修改 settings.json）

