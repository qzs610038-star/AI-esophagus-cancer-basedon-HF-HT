# Phase2 第二批全视野空间头：病理基础编码器替换消融

本包实现 2026-09-19 部署方案：在冻结的全视野空间头 `spatial-11`、数据、图、训练与选模规则下，比较 H-optimus-0、H-optimus-1、Phikon-v2 相对已接纳 UNI2-h / UNI / Virchow2 全视野空间臂的变化。代码版本 `v002`，协议 `fullfov-spatial-backbone-batch2-v1`。

本交付只包含独立代码包和本地零训练测试。未下载模型、未提取特征、未训练、未修改 Registry 或团队结论。任何真实下载与训练须由用户在服务器显式启动。

## 实验边界

- 新训练：3 个编码器 × 种子 45/46/47 × 仅空间臂，共 9 次。任务 ID：`backbone-batch2__{hoptimus0|hoptimus1|phikonv2}__full_fov_224_bicubic_v1__spatial__{45|46|47}__frozen_spatial`。
- 外部评估：上述 9 个正式检查点各对 XZY 推理一次；外部数据不参与选模。
- 基线：只读已接纳的 9 个全视野空间任务（`phase2_fullfov_hpo_v1` / `20260916_231813_101_7b1b9a79`），不重训。包内 `accepted_*_metrics` 是可移植的运行输入；原始绝对路径仅用于来源追溯，在服务器不存在时不阻断检查或分析。
- 几何协议：`full_fov_224_bicubic_v1`（完整方形 patch 精确缩放到 224×224）。
- 图像归一化按编码器原生接口：H-optimus-0/1 用 `hoptimus_rgb_v1`，Phikon-v2 与三个基线用 `imagenet_rgb_v1`。比较对象是“编码器 + 配套输入归一化”，不是强制统一像素归一化后的纯权重比较。
- 输出：H-optimus 使用 `model(x)` 的二维 `[B,1536]`；Phikon-v2 使用 `last_hidden_state[:,0,:]` 的 `[B,1024]`。不做公共投影，主缓存不保存 mean-patch。
- 数据：`MPP2/group_2`、`barcode-repair-v003`、30 通路；train 9472 / internal_val 1078 / external_test 1039。
- 主选模指标：内部验证 `patient_macro_pathway_pcc`。报告时必须并列整体展平 `pooled_pcc`，二者不能互证优越。
- 没有 `search` / `freeze` 阶段，也没有单点臂或去 B 臂。

科学合同见 `docs/protocol.md`。Hugging Face 登录与三模型下载见 `docs/服务器HuggingFace登录与三模型下载操作卡.md`。

## 服务器部署顺序

将整个目录复制到：

```text
D:\AIPatho\qzs\code\phase2_backbone_spatial_ablation_batch2_20260919
```

以下命令都在该目录执行。启动器不安装依赖、不改科学超参数、默认不训练。

1. 只读检查环境（不下载、不训练）。模型尚未下载时，snapshot 项为未核实属于预期。

   ```powershell
   .\run.ps1 -Action check-environment
   ```

2. 默认动作：只读核对包内输入、基线清单和模型路径登记。

   ```powershell
   .\run.ps1
   # 等价于
   .\run.ps1 -Action check-inputs
   ```

   记下终端打印的 `Run directory` 的**父目录**作为本批次 `$BatchDir`。后续动作必须显式传入该批次，否则会另开新批次。

   ```powershell
   $BatchDir = 'D:\AIPatho\qzs\runs\phase2_backbone_spatial_ablation_batch2_20260919\<batch_id>'
   ```

3. 按操作卡在浏览器完成 H-optimus-0 / H-optimus-1 条款同意（后者通常需机构邮箱人工审批），交互式 `hf auth login`，再下载固定 revision。该步骤**不是** `run.ps1` 的动作。

   ```powershell
   .\download_models.ps1 -Model all -Report D:\AIPatho\qzs\inspection\phase2_batch2_download.json
   ```

   一个模型失败不会取消已成功模型。H-optimus-1 未获访问时，可单独重跑 `-Model hoptimus1`；不得把 401/403 当网络抖动无限重试。下载后立即离线严格加载，并用一张合成 224×224 图检查形状：H-optimus `[1,1536]`，Phikon-v2 `[1,1024]`。不提取特征、不训练。

   Phikon-v2 需要 `transformers`。若环境检查显示缺失，只对该解释器显式安装，**不要改** torch / torchvision。兼容版本以本次严格加载预检通过为准，不要预先锁死未验证的旧版。

4. 三个模型就绪后，提取/复用特征。缓存键包含模型与预处理配置，例如 H-optimus 不得与 ImageNet 归一化缓存混用。

   ```powershell
   .\run.ps1 -Action prepare-features -BatchDir $BatchDir
   ```

   默认会依次处理三个模型；某个模型失败时仍继续后续模型，并以非零退出码和 `partial`/`failed` 状态结束。若 H-optimus-1 尚未获批，可显式只运行另外两个模型：

   ```powershell
   .\run.ps1 -Action prepare-features -BatchDir $BatchDir -Models hoptimus0,phikonv2
   ```

5. 训练冻结空间头。完整比较仍要求 3 模型 × 3 种子共 9 次；某模型缓存缺失时，该模型的任务会登记失败，后续模型继续。

   ```powershell
   .\run.ps1 -Action train-spatial -BatchDir $BatchDir
   ```

   H-optimus-1 尚未就绪时，也可定向运行已就绪模型；每个选定模型始终覆盖种子 45/46/47：

   ```powershell
   .\run.ps1 -Action train-spatial -BatchDir $BatchDir -Models hoptimus0,phikonv2
   ```

6. 对已有正式检查点执行 XZY 推理。默认核对全部 9 个任务，缺失项登记失败但不会阻止其他任务；也可用相同的 `-Models` 参数定向评估已完成模型。

   ```powershell
   .\run.ps1 -Action external-eval -BatchDir $BatchDir
   ```

   ```powershell
   .\run.ps1 -Action external-eval -BatchDir $BatchDir -Models hoptimus0,phikonv2
   ```

7. 回传运行目录到本地 `experiments/results/phase2_backbone_spatial_ablation_batch2_20260919/<运行编号>/`。常规回传必须包含 `model_weights.json` 与 `feature_caches.json`，**不复制**权重文件和特征缓存本体。然后本地合并分析：

   ```powershell
   .\run.ps1 -Action analyze-local -BatchDir <本地回传批次目录> -PythonInterpreter <本地python.exe> -RunsRoot <本地runs根> -WeightsRoot <任意占位权重根>
   ```

   若在服务器上分析，同样使用 `-Action analyze-local -BatchDir $BatchDir`。缺少任一新任务或基线时，报告必须标为不完整，不能称为三模型最终结论。结果在 Registry 接纳前只能称为待登记或探索结果。

## 路径

| 类型 | 位置 |
|---|---|
| 代码 | `D:\AIPatho\qzs\code\phase2_backbone_spatial_ablation_batch2_20260919` |
| 运行结果 | `D:\AIPatho\qzs\runs\phase2_backbone_spatial_ablation_batch2_20260919\<运行编号>` |
| 新训练权重 | `D:\AIPatho\qzs\weights\phase2_backbone_spatial_ablation_batch2_20260919\<批次>\<运行>` |
| 新特征缓存 | `D:\AIPatho\qzs\feature_caches\phase2_backbone_spatial_ablation_batch2_20260919\<model>\<geometry>__<norm_profile>\all\` |
| Hugging Face 缓存 | `D:\AIPatho\shared\.cache\huggingface` |
| 解释器（配置记录） | `C:\Users\AIPatho1\pfmval_env\Scripts\python.exe` |

特征缓存含 `features.npy`、`identities.jsonl`、`metadata.json` 和最后写入的 `COMPLETE`。中断目录、revision / 预处理 / 维数 / 身份不一致时不得复用。

## 失败语义

- 缺文件、身份不一致、非有限值、维数不符、严格加载失败或正式检查点缺失均记录到对应模型/任务；训练、特征或外部评估出现 `partial` 时返回非零退出码。`analyze-local` 可对部分结果正常结束，但报告保持 `partial`。
- 单个模型/训练任务失败会保留明确状态并继续后续模型，不得用其他种子补位；未完成 9+9+9 覆盖时不得称完整比较。
- 不支持完整续训或覆盖已有运行/缓存。修复后重新启动，生成新目录。
- 不使用文件哈希；模型 `revision` 是官方 commit 标识。

## 本地代码验收

在本包目录、具备依赖的本地 Python：

```powershell
python -m pytest tests -q --tb=short
python -m compileall -q src runner.py
```

测试只用合成图像、假编码器和小型数组，不下载模型、不读取服务器大数据、不开展正式训练。
