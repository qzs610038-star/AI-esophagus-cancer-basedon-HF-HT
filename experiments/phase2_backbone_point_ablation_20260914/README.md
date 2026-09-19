# Phase2 UNI / Virchow2 基础模型替换消融代码包

本包已经实现部署方案 v1.1 要求的独立流程：复用 UNI2-h 历史 point（单点）参照，新提取并训练 UNI、Virchow2，默认只执行种子 42，完成后停止。代码包不包含预训练权重、大图、特征缓存或新训练结果；这些资产由用户复制到服务器后按明确步骤生成。

## 实验边界

- 主比较统一使用 CLS（全图汇总向量）：UNI2-h 1536 维、UNI 1024 维、Virchow2 1280 维。
- 回归头固定为 `d → 256 → GELU(exact) → Dropout(0.3) → 30`，只训练回归头。
- UNI、Virchow2 同时保存 mean-patch（图像 token 均值）作为辅助特征，但训练器只读取 `cls.npy`。
- UNI2-h 只导入 `phase2_softlink_local_v2/20260908_005325_810_8d306c10` 的匹配历史 point 结果，不重提取、不重训。
- XZY 只在内部正式检查点锁定后推理；服务器外部预测不接收或保存真值。
- 默认入口只运行种子 42。种子 43、44 必须在审阅首批结果后，通过独立入口显式启动。

详细科学合同见 `docs/protocol.md`。

## 服务器部署顺序

将整个目录复制到：

```text
D:\AIPatho\qzs\code\phase2_backbone_point_ablation_20260914
```

以下命令都在该目录执行，启动器不会自动安装依赖，也不会自动修改科学超参数。

1. 首次只读检查环境。模型尚未下载时，模型 snapshot 与候选特征登记项报错属于待处理前置条件；其余数据、解释器、GPU 和目录项应通过。

   ```powershell
   .\check_environment.ps1
   ```

2. 登录 Hugging Face（模型托管平台），确认两个 gated model（需授权模型）均已批准，再下载固定版本。完整步骤和故障处理见 `docs/服务器HuggingFace登录与模型下载操作卡.md`。

   ```powershell
   .\download_models.ps1
   ```

   该步骤只下载、登记并严格加载验证模型，不提取特征、不训练。

3. 只读核查全部既有 UNI2-h 特征路径和种子 42 历史附件。建议把报告写到代码目录之外。

   ```powershell
   .\inspect_existing_uni2h.ps1 -OutputDir D:\AIPatho\qzs\inspection\phase2_backbone_uni2h_seed42
   ```

   必须达到 11,589 个身份均命中。缺少 mean-patch 不会触发 UNI2-h 重提取。

4. 顺序提取 UNI、Virchow2 特征。脚本先构建共同身份清单，再为两个模型写独立版本缓存；CUDA 显存不足时按 16→8→4→2→1 降级。

   ```powershell
   .\prepare_features.ps1 -InspectionReport D:\AIPatho\qzs\inspection\phase2_backbone_uni2h_seed42\inspection_uni2h.json
   ```

   正式准备始终按 UNI→Virchow2 的固定顺序处理两个模型。若中途失败，修复后重跑同一命令；已带 `COMPLETE` 且合同一致的缓存会复用，只有两个候选均完整时 `feature_caches.json` 才标记完成。

5. 再次运行环境检查，确认全部项通过。

   ```powershell
   .\check_environment.ps1
   ```

6. 启动首批种子 42。以下两个命令等价；`run.ps1` 只委托 `run_seed42.ps1`。

   ```powershell
   .\run.ps1
   # 或
   .\run_seed42.ps1
   ```

7. 训练成功后立即停止，不自动运行其他种子。将新批次目录从服务器复制到本地：

   ```text
   D:\AIPatho\qzs\runs\phase2_backbone_point_ablation_20260914\<batch_id>
   ```

   不需要回传服务器训练权重或特征矩阵；批次中的 `model_weights.json` 与 `feature_caches.json` 已登记绝对路径。

8. 本地运行统一分析。未提供 XZY 真值时，仍会完成内部验证指标，并把 XZY 指标标成未计算。

   ```powershell
   .\analyze_local.ps1 `
     -BatchDir <回传批次目录> `
     -OutputDir <新建分析目录> `
     -PythonInterpreter <本地python.exe>
   ```

   若有按身份整理的 XZY 真值 NPZ，再追加 `-ExternalTargets <文件>`；格式见 `docs/本地分析与回传操作卡.md`。

   派生目录包含模型—种子总表、逐患者和逐通路 CSV、完整指标明细 JSON，以及单种子或三种子审核报告。空间指标和 Top-k overlap 若尚无冻结定义，会明确登记为未预定义，不填造数值。

9. 只有在用户审阅种子 42 并明确决定继续后，才运行：

   ```powershell
   .\run_remaining_seeds.ps1 -ConfirmAfterSeed42Review
   ```

   43/44 会形成另一个批次。两批均回传后，可把两个目录作为 `-BatchDir` 数组一次交给 `analyze_local.ps1`；详见本地分析操作卡。工具只在三个模型的 42/43/44 全部齐备时生成三种子样本标准差。

## 路径与输出

| 类型 | 服务器位置 |
|---|---|
| 代码 | `D:\AIPatho\qzs\code\phase2_backbone_point_ablation_20260914` |
| 批次结果 | `D:\AIPatho\qzs\runs\phase2_backbone_point_ablation_20260914\<batch_id>` |
| 新训练权重 | `D:\AIPatho\qzs\weights\phase2_backbone_point_ablation_20260914\<batch_id>\<model>_seed<seed>` |
| 新特征 | `D:\AIPatho\qzs\feature_caches\phase2_backbone_point_ablation_20260914\<model>\<feature_version>` |
| Hugging Face 缓存 | `D:\AIPatho\shared\.cache\huggingface` |

新特征缓存包含：

```text
cls.npy                 # N×d float32，唯一训练输入
mean_patch.npy          # N×d float32，辅助输出，不进入回归头
identities.jsonl        # 每行身份和固定行号
cache_meta.json         # 模型版本、token 布局、预处理、精度、软件和资源记录
COMPLETE                # 最后生成的完整提交标记
```

批次目录包含历史参照副本、候选内部预测、无标签 XZY 预测、训练历史、点位顺序、配置和代码快照。检查点不写入批次结果目录。

## 失败语义

- 缺文件、身份不一致、非有限值、维数不符、严格权重加载失败或正式端点缺失都会返回非零退出码。
- 调度器串行执行；首个失败单元记为 `failed`，其后单元记为 `not_run`，不会跨过故障继续形成不完整比较。
- 不支持 resume（断点续训）或覆盖已有运行/缓存。修复后重新启动，生成新目录；旧证据保留。
- 不使用文件哈希校验；模型 `revision` 是为可复现下载固定的官方 commit 标识，不是本包新增的文件哈希治理。

## 本地代码验收

在本包目录使用具备依赖的本地 Python：

```powershell
python -m pytest tests -q --tb=short
python -m compileall -q src runner.py
```

测试只使用人工 token、假编码器和小型数组，不下载模型、不读取服务器数据、不开展正式训练。
