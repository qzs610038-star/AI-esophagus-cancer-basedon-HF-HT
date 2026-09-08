# Phase2 软连接升级 v2.1 独立代码包

服务器代码目录：`D:\AIPatho\qzs\code\phase2_softlink_local_v2`  
服务器结果目录：`D:\AIPatho\qzs\runs\phase2_softlink_local_v2\<批次>\<运行>`
服务器权重目录：`D:\AIPatho\qzs\weights\phase2_softlink_local_v2\<批次>\<运行>`  

从下一次运行起，checkpoint（模型权重）与常规结果分开保存。回传 `runs` 批次时不复制 `weights`，但必须保留批次根目录的 `model_weights.json`，其中逐臂、逐种子登记服务器权重路径。

本包含完整代码入口，但没有执行正式训练。唯一人工编辑配置是 `config.json`；每个批次启动时会冻结 `config_snapshot.json`，不计算哈希。

## 入口

| 命令 | 用途 |
|---|---|
| `.\prepare_inputs.ps1` | 从服务器现有特征目录与完整坐标生成空间分组、步长和XZY无标签点表 |
| `.\run_all.ps1`（或 `.\run.ps1`） | 预检 → 4臂×3种子 → 内部冻结 → 12个XZY外部预测 |
| `.\run_point.ps1` | point（匹配回归）3种子 |
| `.\run_relation.ps1` | relation（关系辅助）3种子 |
| `.\run_spatial.ps1` | spatial（空间修正）3种子 |
| `.\run_joint.ps1` | joint（关系+空间）3种子 |
| `.\run_seed.ps1 -Seed 42` | 一个种子的4个实验臂 |
| `.\run_precheck.ps1` | 只做训练前核对，不更新优化器 |
| `.\run_external.ps1 -EndpointManifest <正式端点清单>` | 只做冻结端点外部预测 |
| `.\analyze_local.ps1 -BatchDir <回传批次目录>` | 本地派生指标、15/25轮、固定β=1平滑和Fable排查 |

训练入口可用 `-PythonInterpreter`、`-RunsRoot`、`-WeightsRoot`、`-Device` 覆盖运行位置，不会回写配置。`prepare_inputs.ps1` 会生成三份小型输入并把 `data.external_point_table` 写入 `config.json`。正常失败会令本批剩余任务标为 `not_run`，再次执行产生新批次，不覆盖旧结果。

### 本机回传分析解释器

`analyze_local.ps1` 默认按本机 Conda 已登记环境名 `pfmval_py310` 定位 `python.exe`：优先查询 `conda env list --json`，当 Conda 命令未加入当前 PowerShell 的 PATH（环境变量搜索路径）时读取用户级 Conda 环境登记。它不会回退到全局 Python，也不会把 `config.json` 中服务器专用的 `runtime.python_interpreter` 当作本机解释器。

如需覆盖，可显式指定本机解释器：

```powershell
.\analyze_local.ps1 -BatchDir <回传批次目录> -PythonInterpreter <本机 python.exe 路径>
```

可先运行 `.\resolve_local_python.ps1` 查看默认会使用的解释器路径。

## 服务器执行顺序

```powershell
.\prepare_inputs.ps1
.\run_precheck.ps1
.\run_all.ps1
```

准备入口使用已有的 MPP2 完整划分表和 UNI2-h 特征缓存：内部患者必须各自只命中一个完整特征来源组；XZY 必须命中一个1039点的完整无标签特征目录。它生成：

- `inputs/slide_mapping.csv`：以实际特征来源目录建立可审计空间分组ID，不把 `patient_id` 直接冒充物理切片条码。
- `inputs/slide_geometry.csv`：内部步长必须与v2.1固定值一致；XZY从完整规则坐标的x/y相邻差众数推断。
- `inputs/external_point_table.csv`：只含XZY身份、坐标和特征路径，不读取外部标签。

若同一患者命中多个不同来源组、坐标网格不一致、内部/外部点数不符或特征维数不是1536，准备入口会停止。仍需保证配置中的 labels 与特征缓存服务器路径确实存在，但不再需要手写映射或XZY点表。

## 关键合同

- 身份键固定为 `(patient_id, slide_id, spot_id)`；按键关联，不依赖文件行序。
- 训练、内部验证和外部点集不混合；外部XZY不参与调参、选点或早停。
- `point/relation/spatial/joint` 四臂共享逐种子初始公开参数、中心顺序和Dropout（随机失活）流。
- 第1–5轮关系系数为0，第6–15轮线性升至0.05；warmup（预热）检查点不能作为正式端点。
- 内部选择为患者等权通路PCC，容差内以患者等权z-MSE破同分；正式端点从第6轮起。
- 外部预测接口不接收标签，只接受 `formal`（正式）检查点；z分数只逆变换一次。
- 第15/25轮逐点预测随训练保存；本地分析缺少已定义窗口、mask或Top-k时记录“不适用”，不填假值。

## 最小验收

在包目录执行：

```powershell
$python = .\resolve_local_python.ps1
& $python -m pytest tests -q --tb=short
& $python -m compileall -q src
```

以上只验证代码合同，不产生正式模型或科研结果。规格与决策副本位于 `docs/`。
