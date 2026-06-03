---
name: git-rescue
description: Use when git push fails — SSL errors, proxy issues, rejected non-fast-forward, divergent histories, secret scanning blocks, or "Git LFS locking API" errors. Covers China network environment (Clash proxy/TUN) and GitHub Push Protection remediation.
---

# Git Rescue — 推送故障诊断与修复

## 诊断流程

```
git push 失败
  ├─ SSL/EOF 错误？
  │   ├─ git ls-remote 能通？→ 代理截断流式响应（TUN 嫌疑）
  │   ├─ 换协议：HTTPS → SSH
  │   └─ 都不通？→ 检查代理/防火墙
  ├─ "non-fast-forward" rejected？
  │   └─ 历史分叉 → cherry-pick 方案（见下）
  ├─ "GH013: Push cannot contain secrets"？
  │   └─ GitHub Push Protection → rebase 清除（见下）
  └─ "LFS locking API" 错误？
      └─ git config lfs.<url>.locksverify false
```

## 症状 → 根因速查

| 症状 | 根因 | 修复 |
|------|------|------|
| `ls-remote` 通但 `fetch/push` 断 | Clash TUN 虚拟网卡截断 HTTP/2 流 | 换 SSH 协议 |
| `schannel: server closed abruptly` | Windows TLS 后端与代理不兼容 | `git config --global http.sslBackend openssl` |
| `OpenSSL SSL_read: unexpected eof` | 代理/TUN 截断大响应 | 关 TUN 或换 SSH |
| `Connection timed out` 无代理 | 直连被墙 | 配置代理或换 SSH |
| `non-fast-forward` + 相同 commit 消息不同 SHA | 两处独立 init 导致历史分叉 | cherry-pick 到远程基准 |
| `GH013: Push cannot contain secrets` | commit 中含 token/密钥 | rebase 清除全部涉事 commit |
| `LFS locksverify` 代理连接拒绝 | LFS 锁验证走 HTTPS 代理 | `locksverify false` |

## 核心修复模式

### 1. HTTPS → SSH 切换（代理干扰时首选）

```bash
# 前提：已有 SSH key 并添加到 GitHub
ssh -T git@github.com                              # 验证连通性
git remote set-url origin git@github.com:user/repo.git
git fetch origin                                    # SSH 不受 TUN 影响
```

**TUN 干扰特征**：`ls-remote` 正常（小响应），`fetch`/`push` 必断（大流式响应），且 `curl` 直连 GitHub 正常。

### 2. 历史分叉合并（cherry-pick 法）

适用场景：本地和远程 commit 消息相同但 SHA 完全不同（独立 init 导致）。

```bash
git branch backup-main                    # 安全备份
git fetch origin                          # 确保远程最新
git reset --hard origin/main              # 以远程为基准
# 从 backup-main 找出本地独有 commit，按时间从旧到新
git cherry-pick <oldest> <...> <newest>   # 拣选到远程之上
git push origin main
```

**核心原则**：以远程历史为基准，本地独有 commit 拣选到顶部 → 线性历史，无需 force push。

### 3. Secret 清除（GitHub Push Protection 阻断后）

GitHub 检测到 token 出现在**任一 commit** 中即阻断推送。必须改写所有涉事 commit。

```bash
# 步骤 1：编辑文件移除 secret
sed -i 's/<REAL_TOKEN>/<PLACEHOLDER>/g' sensitive_file

# 步骤 2：创建 fixup commit 指向涉事最早的 commit
git add sensitive_file
git commit --fixup <first_offending_sha>

# 步骤 3：autosquash rebase
git stash -u   # 确保工作树干净
GIT_SEQUENCE_EDITOR=true git rebase -i --autosquash <base_before_offending>

# 步骤 4：冲突时用 --theirs 然后重做 sed
# （每个涉事 commit 都可能冲突，需逐次修复）
git checkout --theirs sensitive_file
sed -i 's/<REAL_TOKEN>/<PLACEHOLDER>/g' sensitive_file
git add sensitive_file && git rebase --continue

# 步骤 5：验证所有 commit 无残留
for sha in $(git log --oneline <base>..HEAD --format="%h"); do
  git show $sha:sensitive_file | grep "<REAL_TOKEN>" && echo "FOUND in $sha"
done

# 步骤 6：推送
git push origin main
```

**关键教训**：
- `--ours` 会保留原始 token（HEAD 是原始涉事 commit）→ **必须用 `--theirs` 然后手动 sed**
- pre-commit hook 可能回退修改，需在 rebase 过程中逐次确认
- `git filter-branch` 如被拦截，用 `rebase -i --autosquash` 等效

### 4. LFS 锁验证禁用

```bash
git config lfs.https://github.com/USER/REPO.git/info/lfs.locksverify false
GIT_LFS_SKIP_PUSH=1 git push origin main   # 或直接跳过 LFS push
```

## 通用排查命令

```bash
# 连通性诊断
git ls-remote origin                    # 最小请求测试
GIT_CURL_VERBOSE=1 GIT_TRACE=1 git fetch origin 2>&1 | tail -80  # 详细追踪

# 历史对比
git log --oneline main...origin/main --left-right   # 分叉可视化

# SSH 验证
ssh -Tv git@github.com                  # 详细 SSH 诊断
```

## 预防措施

1. **Token/密钥不进仓库**：用环境变量或 `.env`（确保在 `.gitignore`）
2. **统一 init 流程**：避免多地点独立 `git init`，应从远程 clone
3. **SSH 作为备用**：HTTPS 代理不稳定时立即可切换
4. **推送前自检**：`git log -p origin/main..HEAD | grep -i "token\|secret\|key\|password"`
