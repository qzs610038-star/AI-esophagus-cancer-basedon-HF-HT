# 代码包验收记录

结论：部署方案 v1.1 对应的独立代码包已完成本地工程验收；服务器模型下载、11,589 点特征提取、正式训练和科研结果验收尚未开始，不能把本记录解释为实验结果已经完成或 accepted。

验收日期：2026-09-15。

## 已完成范围

- UNI2-h 仅作为 `phase2_softlink_local_v2/20260908_005325_810_8d306c10` 的历史匹配 point 参照导入，不重提取、不重训。
- UNI 与 Virchow2 使用固定官方 revision、严格离线加载、共同图像预处理、独立 CLS/mean-patch 缓存和原生维数 point 头。
- 训练合同、配对输出层初始化、PCG64 点位顺序、独立 Dropout 随机流、formal 选模、早停及 XZY 隔离均由冻结配置和测试约束。
- 默认入口仅执行 seed 42；43/44 使用独立且需显式确认的入口。两个服务器批次可在本地合并，但只有三个模型的 42/43/44 全部齐备才产生样本标准差。
- 本地分析输出双 PCC、误差及一致性指标、逐患者/逐通路明细和资源字段；未冻结的空间指标与 Top-k overlap 显式记为未预定义。

## 本地验证证据

在代码包根目录执行：

```powershell
python -m pytest -q --tb=short
python -m compileall -q src runner.py
```

结果：

- pytest：74 项通过；无失败、错误或跳过。
- Python 编译：`src` 与 `runner.py` 全部通过。
- PowerShell：包内 9 个 `.ps1` 均通过 AST 语法解析。
- 独立性：代码包复制到临时目录、清除 `PYTHONPATH` 后，`runner.py --help` 成功；源码没有上级仓库 import 或父目录路径注入。
- CLI：总入口和下载、环境核查、历史参照核查、特征准备、本地分析子入口的 `--help` 均成功。

真实历史附件只读核验结果：

| seed | formal epoch | 已保存训练轮次 | PCG64(`seed+100000`) 全轮逐元素一致 | batch/末批合同 |
|---:|---:|---:|---|---|
| 42 | 31 | 41 | 是 | 256 / 保留 |
| 43 | 39 | 42 | 是 | 256 / 保留 |
| 44 | 24 | 34 | 是 | 256 / 保留 |

三个 seed 的历史内部/外部附件、formal checkpoint、point 状态、形状、有限值及无外部 `target_z` 合同均通过验证。冻结 split manifest 含 10,550 个开发点（train 9,472、internal_val 1,078），其身份顺序已经是历史 point 训练所用的排序；30 通路和训练集标准化参数均可读取。

固定模型版本已于 2026-09-15 对照官方仓库元数据核对：

- [UNI 官方模型卡](https://huggingface.co/MahmoodLab/UNI)：`b55a5ec6cade1a39edfe6534189a9b8ca7a022f0`；
- [Virchow2 官方模型卡](https://huggingface.co/paige-ai/Virchow2)：`3158645804b69e3f3bc4439d4116edddf0840a72`。

版本值属于官方 Hub commit 标识；本包没有新增文件哈希治理。

## 服务器待完成项

当前 `inputs/model_manifest.json` 的两个 `snapshot_path` 仍为 `null`，下载状态为 `not_downloaded`，这是交付时的正确状态。服务器须依次完成：

1. `check_environment.ps1`；
2. 按[登录与下载操作卡](docs/服务器HuggingFace登录与模型下载操作卡.md)完成授权、固定版本下载及严格加载；
3. `inspect_existing_uni2h.ps1`，确认 11,589 个历史 CLS 身份完整；
4. `prepare_features.ps1`，顺序提取 UNI、Virchow2；
5. 再次环境检查后运行 `run_seed42.ps1`；
6. 回传首批并按[本地分析操作卡](docs/本地分析与回传操作卡.md)审核；
7. 只有用户明确决定继续时，才运行 `run_remaining_seeds.ps1 -ConfirmAfterSeed42Review`。

本次未下载权重、未创建服务器特征缓存、未运行正式训练、未修改实验 Registry、未提交 Git，也未打包压缩或远端同步。
