# Phase 2 任务2、任务3重新训练包

包名：`mpp2_task23_scratch_ablation_20260915`

本包执行已经确认的新实验口径。它不会修改服务器上的方案(a)代码、旧运行或旧权重，也不会把 `formal_best.pt` 或任何已训练下游权重加载进新模型。

## 一、这次实际运行的五组实验

| 顺序 | 组别 | 用途 |
|---|---|---|
| 1 | 任务2：直接预测共同重构通路 | 对照组；直接学习由真实319基因经固定ssGSEA生成的30通路 |
| 2 | 任务2：预测319基因后重构通路 | 实验组；先预测基因，再用完全相同的ssGSEA重构30通路 |
| 3 | 任务3：稠密训练 | 从头训练的稠密对照 |
| 4 | 任务3：规则隔点 | 只抽稀训练集，固定1.5倍原生步长图半径 |
| 5 | 任务3：随机等量 | 每位患者保留点数与规则隔点严格相同 |

五组都使用种子42。每个模型的 H/C 随机初始化，B全零初始化。任务3三组整网初始化逐参数相同；任务2输出维不同，但共享H初始化相同。

正式训练统一最多执行14,800次优化器更新，并每296次更新检查一次稠密内部验证集；五组都跑完固定预算，最终选择预算内内部验证损失最低的checkpoint。XZY只在checkpoint选定后评估一次。

## 二、上传位置

把整个文件夹上传为：

```text
D:\AIPatho\qzs\code\mpp2_task23_scratch_ablation_20260915
```

不要覆盖或改动：

```text
D:\AIPatho\qzs\code\phase2_softlink_local_v2
D:\AIPatho\qzs\runs\phase2_softlink_local_v2\20260908_005325_810_8d306c10
```

数据应保持在：

```text
D:\AIPatho\Patch\genes_tag
```

## 三、最简单运行方式

双击 `run.cmd`，依次执行：

1. 输入 `1`：服务器环境检查。结尾必须出现 `PREFLIGHT PASSED`。
2. 输入 `2`：五组冒烟测试。结尾必须出现 `ALL smoke ARMS SUCCEEDED` 和 `BATCH CONTRACT VERIFIED`。冒烟结果不能用于正式报告。
3. 输入 `3`：五组正式实验。程序会串行训练并在全部成功后自动生成回传压缩包。

选择3后请保持窗口开启。任务2需要多次执行ssGSEA，任务3稀疏组也要完成与稠密组相同的14,800次参数更新，因此总运行时间可能较长。

正式运行完成后，同一代码目录会出现：

```text
task23_return_full_日期时间_编号.zip
```

把这个zip发回即可。不要只发截图，也不要发送冒烟测试的zip。

## 四、结果保存位置

一批正式结果位于：

```text
D:\AIPatho\qzs\runs\mpp2_task23_scratch_ablation_20260915\full_日期时间_编号\
```

五组新权重位于：

```text
D:\AIPatho\qzs\weights\mpp2_task23_scratch_ablation_20260915\full_日期时间_编号\<组别>\best_checkpoint.pth
```

每组运行目录均包含 `model_weights.json`，记录权重绝对路径。旧目录不会被覆盖。

## 五、怎样判断正式实验有效

不能仅凭“窗口没报错”判断。程序会自动核对：

- 五组 `run.json` 都是 `status=succeeded`、`exit_code=0`、`run_kind=full`；
- 所有初始化记录都是 `fresh_random_h_c_zero_b`；
- `loaded_checkpoint=null` 且 `loaded_keys=[]`；
- 任务3三组初始化哈希完全一致；
- 任务2两组共享H初始化哈希一致；
- 五组都实际完成14,800次优化器更新；
- 任务3规则隔点与随机组按患者等量；
- 内部验证和XZY保持稠密；
- 每组都生成两种PCC、zMSE、rawMAE及逐患者逐通路明细。

全部成立后，批次目录里的 `batch_summary.json` 才会写成 `status=verified`。

## 六、如果运行失败

环境检查失败时，把完整窗口内容发回。若只提示缺少 `gseapy`，可在服务器PowerShell中执行：

```powershell
& 'C:\Users\AIPatho1\pfmval_env\Scripts\python.exe' -m pip install gseapy
```

某一组训练失败时，不要换种子、不要加载旧权重。把窗口最后约50行，以及显示的该组结果目录中的这些文件发回：

```text
run.json
config.json
logs\stdout.log
logs\errors.log
logs\console.log
```

## 七、评价口径

任务2两组共同真值都是“真实319基因经固定GSEApy ssGSEA得到的30通路”，因此可以直接比较算法改变是否提升。

任务3首先用原数据随附的参数把已修复、已对齐的标签无损还原到原始通路单位；每组模型的均值和标准差仅使用该组实际保留的训练点重新拟合。输出再还原到原始通路尺度，并使用预先固定的稠密训练评估标准器计算统一的展平PCC和zMSE。固定评估标度不作为稀疏模型的训练标准化。

最终同时报告：逐患者逐通路平均PCC、整体展平PCC、pooled zMSE和rawMAE。内部验证与外部XZY分开报告。
