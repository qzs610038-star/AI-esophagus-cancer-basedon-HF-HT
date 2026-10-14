# 服务器 Hugging Face 登录与模型下载操作卡

目标：把两个已固定版本的 gated model（需授权模型）下载到 `D:\AIPatho\shared\.cache\huggingface`，随后由本包进行严格本地加载验证。下载和训练完全分开；本卡操作不会启动特征提取或训练。

当前固定版本：

| 模型 | 仓库 | revision（固定 commit） | 必需文件 |
|---|---|---|---|
| UNI | `MahmoodLab/UNI` | `b55a5ec6cade1a39edfe6534189a9b8ca7a022f0` | `config.json`、`pytorch_model.bin`、`README.md` |
| Virchow2 | `paige-ai/Virchow2` | `3158645804b69e3f3bc4439d4116edddf0840a72` | `config.json`、`model.safetensors`、`README.md` |

Hugging Face 官方 CLI（命令行工具）文档：[CLI 指南](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli)；固定 revision 和选择文件由官方 [`snapshot_download`](https://huggingface.co/docs/huggingface_hub/package_reference/file_download) 接口支持。

## 首次授权与登录

1. 在浏览器登录自己的 Hugging Face 账户，分别打开 [UNI](https://huggingface.co/MahmoodLab/UNI) 与 [Virchow2](https://huggingface.co/paige-ai/Virchow2)，阅读条款并申请/确认访问。两者当前模型卡均标为 CC-BY-NC-ND-4.0；Virchow2 页面还要求研究用途和个人授权。

2. 在 Hugging Face 的 token 设置页创建只读 token。不要把 token 写入 PowerShell 历史、配置、脚本、报告或聊天。

3. 在服务器交互式登录：

   ```powershell
   $env:HF_HOME = 'D:\AIPatho\shared\.cache\huggingface'
   $env:HF_HUB_OFFLINE = '0'
   hf --version
   hf auth login
   hf auth whoami
   ```

   `hf auth login` 会在终端内安全提示输入 token。若服务器仍提供旧版命令，可用 `huggingface-cli login`，但后续以 `hf auth whoami` 能通过为准。

4. 若 `hf` 命令不存在，只对项目指定解释器安装/升级 `huggingface_hub`，不要升级整套 PyTorch：

   ```powershell
   & 'C:\Users\AIPatho1\pfmval_env\Scripts\python.exe' -m pip install 'huggingface_hub>=0.30,<2'
   hf auth whoami
   ```

## 在线下载与严格验证

在代码包目录执行：

```powershell
.\download_models.ps1 -Report D:\AIPatho\qzs\inspection\phase2_model_download.json
```

脚本依次执行以下动作：

1. 按清单中的固定 revision 调用 `snapshot_download`；
2. 只请求每个模型的配置、权重和 README；
3. 把真实 `snapshots\<revision>` 绝对路径写入本包 `inputs\model_manifest.json`；
4. 立即切回离线模式；
5. 用 `pretrained=False` 在本地构造模型并 `strict=True` 加载权重；
6. 用一个全零 224×224 样本验证 CLS 与 mean-patch 形状；
7. 退出，不调用 `prepare_features.ps1` 或任何训练入口。

`inputs\model_manifest.json` 会区分 `files_present_unverified`、严格加载 `passed` 或 `failed`。若传入 `-Report`，成功或失败都会写出机器可读状态；失败返回非零退出码，不继续下载下一模型，也不会进入特征提取。

如需逐个处理：

```powershell
.\download_models.ps1 -Model uni
.\download_models.ps1 -Model virchow2
```

## 服务器无网络时离线复制

在一台已获授权且可联网的 Windows 机器上设置同类 `HF_HOME`，用本包或官方 `hf download --revision <commit>` 完成下载。然后复制以下完整缓存目录，保留 `snapshots\<commit>` 层级：

```text
<HF_HOME>\hub\models--MahmoodLab--UNI
<HF_HOME>\hub\models--paige-ai--Virchow2
```

放到服务器：

```text
D:\AIPatho\shared\.cache\huggingface\hub\
```

之后只登记并严格验证，不发网络请求：

```powershell
.\download_models.ps1 -RegisterOnly
```

不要通过邮件、网盘或代码仓库转发模型给未单独接受条款的其他账户；是否允许复制和共享以模型页面当前条款为准。

## 常见故障

| 现象 | 核查与处理 |
|---|---|
| HTTP 401 | 运行 `hf auth whoami`；确认 token 未过期、权限至少为 read；重新用交互式 `hf auth login`，不要把 token 放在命令参数里。 |
| HTTP 403 / gated repo | 浏览器确认两个模型都已批准，账户邮箱和用途符合模型页面要求；仅登录成功不等于已获模型访问权。 |
| `hf` 子命令不同 | 先看 `hf --help` 和版本；旧环境可用 `huggingface-cli login`。本包实际下载通过同一解释器内的 Python API，不依赖 `hf download` 的参数差异。 |
| 提示离线或无法 HEAD | 下载阶段确保 `$env:HF_HUB_OFFLINE='0'`；提取和训练阶段必须为 `1`。不要在严格离线加载失败后自动改成联网。 |
| 超时或代理中断 | 保留现有 cache 后重跑，Hugging Face 会复用已完成文件。按现场网络策略设置 `HTTPS_PROXY`；必要时提高 `HF_HUB_DOWNLOAD_TIMEOUT`。不要把代理凭据写进报告。 |
| 磁盘不足 | 用 `Get-PSDrive D` 检查可用空间；同时预留 UNI、Virchow2 权重和临时下载空间。不要把权重改写到 runs。 |
| 权限不足 | 核对当前账户对 `D:\AIPatho\shared\.cache\huggingface` 的创建和写入权限；不要以管理员权限绕过研究环境管理约束。 |
| Windows symlink 警告 | Hugging Face 可能退化为复制文件；这通常影响空间而非数值。可启用 Windows Developer Mode，或接受复制模式并确认磁盘空间。 |
| `.incomplete`、0 字节或 LFS pointer | 不登记、不训练。保留现场文件后重跑下载；严格加载仍失败时回传错误和文件大小，不手工拼接权重。 |
| config 与权重不匹配 | 确认二者来自同一 `snapshots\<revision>`。本包禁止 `strict=False`、删键或随机初始化继续。 |
| 本地已有 Virchow2 权重但来源版本不明 | 未记录 revision 的散落文件不能直接登记为本轮正式权重；重新下载固定版本，或从保留完整 snapshot 结构的授权缓存离线复制。 |

登录/下载失败时，回传终端错误、`hf --version`、`hf auth whoami` 的成功/失败状态和目录文件大小即可；不要回传 token。
