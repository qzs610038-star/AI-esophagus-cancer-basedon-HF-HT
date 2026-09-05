# PFMval Windows 服务器最简加载检查

这是诊断包：不训练、不下载、不修改数据或模型缓存、不计算哈希。依据本项目 `configs/server_paths.yaml` 和实际读取代码；不使用其他项目服务器配置。

## 运行

直接复制本地 `deliverables/server_load_check_20260905/` 文件夹，放到：

```text
D:\AIPatho\qzs\code\server_load_check_20260905
```

双击 `run.cmd` 即可。也可打开该目录的 PowerShell：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run.ps1
```

默认使用 `C:\Users\AIPatho1\pfmval_env\Scripts\python.exe`，不会自行切换到其他 Python。若此位置已变，在 `config.json` 的 `python_interpreter` 中填写实际解释器。

## 检查内容

- Windows、Python、PyTorch/torchvision/timm/NumPy/pandas/Pillow 等依赖的实际导入。
- CUDA 是否可用，并在默认 GPU 0 做一次 16×16 小矩阵乘法，不训练、不占用大量显存。
- UNI2-h 缓存权重能否在 CPU 上反序列化；只检查权重可读取和部分张量，不构造完整模型、不运行完整推理。
- MPP2 七位患者的图像、原始标签、1536 维特征缓存各抽取一份读取。原始标签读取不代表允许直接用于训练。
- 已登记旧 Phase2、划分、冻结基线、Phase3、历史 token 缓存和 Torch Hub 权重抽样。历史快照条目单独标为 `md_import`，由此次运行帮助核对其现状。
- 不读取 Hugging Face token 或服务器秘密文件；不会把真实图像、预测、标签或权重复制进报告，仅记录路径、形状、版本和错误。

普通检查只抽样；权重读取可能使用数 GB 内存并花费几分钟。数据目录最多向下查四层、300个目录；不是全盘扫描。某项失败后继续检查后续项。

## 回传

日志和结果自动写到：

```text
D:\AIPatho\qzs\runs\server_load_check_20260905\<运行编号>\
```

运行结束后，终端会显示需要复制的运行文件夹完整路径。将该文件夹直接复制回本地交给 Agent；不自动创建压缩文件，需要时由用户自行压缩。

复制时保留完整运行目录，包含 `REPORT.md`、`report.json`、`run.json`、`config.json`、`package.json` 和 `logs/`。解释器不存在也会落启动日志；如果连输出目录都无法建立，回传终端错误。

`PASS` 表示该项抽样读取成功；`WARN` 表示可选/历史资源未通过；`FAIL` 表示本包选定的 MPP2 核心检查失败。FAIL 会返回非零退出码，但不会提前跳过后续项目。运行记录的 `failed` 可能是诊断发现环境问题，不代表发生训练失败。

这不是全数据完整性、图像标签对应关系或完整模型推理验收。特征样本可读不能证明全部训练数据就绪。若 checkpoint 的安全读取方式不兼容，只记录错误，不自动改用任意 pickle 执行。新结果不自动写入科研 Registry。

代码版本 v002：只改变文件夹交付和回传提示，移除自动压缩；诊断检查内容与 v001 相同。正式服务器状态以回传报告为准，本地制作和测试不能证明服务器已通过。
