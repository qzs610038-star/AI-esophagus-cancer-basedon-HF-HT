# diagnostic-20260810-gitee-rt-pilot-pathprobe-r001 操作卡

> `path_probe` 已在请求 allowlist 中，但当前没有固定 runner。
> 固定传输通道：Gitee；不得把请求解释为任意 shell 授权。

## 已解析参数

- 治理请求分支：`codex/w006-gitee-roundtrip-pilot-20260810-bound`
- 源码分支：`codex/w006-gitee-roundtrip-pilot-20260810-bound`
- 源码提交：`42b26431262efdfe93766f2e3bd3d54d26999472`
- 回传分支：`automation/diagnostics/diagnostic-20260810-gitee-rt-pilot-pathprobe-r001/return`
- 服务器仓库：`D:\AIPatho\qzs\pfmval_governance`
- 诊断工作树根：`D:\AIPatho\qzs\pfmval_diagnostics`

## 卡 1：本地发布请求

```powershell
git push gitee HEAD:codex/w006-gitee-roundtrip-pilot-20260810-bound
git push gitee codex/w006-gitee-roundtrip-pilot-20260810-bound:codex/w006-gitee-roundtrip-pilot-20260810-bound
```

## 卡 2：服务器受限执行与回传

`BLOCKED`：当前版本没有 `path_probe` 的固定收集器；请另开治理任务实现并测试，不得手写服务器命令替代。

## 卡 3：本地取回与验证

`NOT APPLICABLE`：没有固定 runner 输出时，不得登记完成事件。
