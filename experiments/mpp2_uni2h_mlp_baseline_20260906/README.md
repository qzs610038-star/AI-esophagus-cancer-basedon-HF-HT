# MPP2 UNI2-h + 两层 MLP 独立实验包

复制到服务器后双击即可开训。冒烟或满训只改一个文件。这不是已接纳基线的替代结果。

已接纳对照仍是 `mpp2_barcode_repair_v003_frozen_baseline_20260711`。默认 `run_kind=smoke` 只跑 2 个 epoch，不能当成正式复现。

## 你要做的事

1. 把整个本文件夹复制到：

```text
D:\AIPatho\qzs\code\mpp2_uni2h_mlp_baseline_20260906
```

不要带 `tests/work/`、`runs/`、`analysis/`。

2. 双击 `run.cmd`，或在该目录执行：

```powershell
.\run.ps1
```

不要改解释器、数据路径或输出根，也不要传 `-Config`。缺文件时程序会写明缺什么。

3. 需要切换时，只改包根目录的 `run_mode.json`：

```json
{ "run_kind": "smoke" }
```

| run_kind | 行为 |
|---|---|
| `smoke` | 训 2 个 epoch |
| `full` | 训 50 个 epoch，并用内部验证 `val_loss` 早停 |
| `predict` | 不训练，导出已接纳 checkpoint 的预测表 |

改完再双击 `run.cmd`。不要改 `config.json` 或 `package.json` 来切换轮数。

4. 回传终端打印的：

```text
D:\AIPatho\qzs\runs\mpp2_uni2h_mlp_baseline_20260906\<运行编号>\
```

放到本地 `experiments/mpp2_uni2h_mlp_baseline_20260906/runs/<运行编号>/`。不要自动打压缩包。

## 现役设置

- 模型：冻结 UNI2-h 1536 维特征 + `1536→1024→30` 两层 MLP
- 目标：`pathway_ssgsea`（训练读入和输出都是 30 通路）
- 采样：`dense`（训练集不抽稀）
- 划分：包内 `assets/group_2/split_manifest.csv`
- 通路标签：服务器修复台 `barcode-repair-20260711-d626ad8-v003`
- 基因标签：`inputs.gene_labels_root` 现在为空；任务 2 由队友填服务器上的原始基因路径

给队友的分工和步骤看 `给队友的任务说明.md`。输入、输出、换模型的接口细节看 `队友同步说明.md`。
