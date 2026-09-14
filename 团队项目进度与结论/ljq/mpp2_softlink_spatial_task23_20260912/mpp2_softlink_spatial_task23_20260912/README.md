# mpp2_softlink_spatial_task23_20260912

> **2026-09-12 状态更正**：本包仍是加载稠密训练权重的旧微调实现，不能直接用于更正后的任务2/3从头训练。新方向是复用经典空间残差结构、重新初始化下游 H/C/B；见[交接更正说明](../../Phase2任务2与任务3_重新训练与交接更正说明_20260912.md)。先完成入口、标准化、构图与评价合同调整，再按新说明运行；以下步骤保留作旧实现参考。

这是任务2、任务3的独立空间模型消融包。它只读取服务器上原有方案(a)代码目录中的切片映射/几何文件，以及原有运行目录中的正式权重；不会修改或覆盖它们。

## 上传位置

把整个文件夹复制到：

```text
D:\AIPatho\qzs\code\mpp2_softlink_spatial_task23_20260912
```

不要覆盖：

```text
D:\AIPatho\qzs\code\phase2_softlink_local_v2
D:\AIPatho\qzs\runs\phase2_softlink_local_v2\20260908_005325_810_8d306c10
```

运行结果会分别新建在：

```text
D:\AIPatho\qzs\runs\mpp2_spatial_task2_gene_dense_20260912\<运行编号>
D:\AIPatho\qzs\runs\mpp2_spatial_task3_dense_finetune_20260912\<运行编号>
D:\AIPatho\qzs\runs\mpp2_spatial_task3_stride_20260912\<运行编号>
D:\AIPatho\qzs\runs\mpp2_spatial_task3_random_equal_20260912\<运行编号>
```

## 最简单运行顺序

所有命令都可以直接双击。

1. 双击 `00_服务器环境检查.cmd`。
   - 最后必须显示 `PREFLIGHT PASSED`。
   - 如果失败，不要改路径凑数；把窗口内容或报错发回。
2. 双击 `01_冒烟测试_全部.cmd`。
   - 会顺序跑四组，每组2轮。
   - 最后必须显示 `ALL SMOKE RUNS SUCCEEDED`。
   - 冒烟结果不能写入正式报告。
3. 依次双击正式任务：
   - `02_正式任务2.cmd`
   - `03_正式任务3_稠密续训对照.cmd`
   - `04_正式任务3_空间隔点.cmd`
   - `05_正式任务3_随机等量.cmd`
4. 四组都显示 `SUCCESS` 后，双击 `06_打包正式结果.cmd`。
5. 将同目录生成的 `task23_return_日期时间.zip` 发回分析。

服务器资源足够时，四个正式任务也可以分别开四个终端运行；如果GPU显存或CPU内存不确定，按上面顺序串行运行最稳妥。

## 成功判据

不能只看窗口没有报错。每个正式结果目录都必须满足：

- `run.json`：`status=succeeded`、`exit_code=0`、`run_kind=full`。
- `raw/initialization.json`：checkpoint身份为 `spatial/formal/seed42/epoch30`。
- 任务2：`load_mode=shared_only_new_gene_heads`，输出维数319。
- 任务3：`load_mode=full_spatial_state`，输出维数30。
- `metrics.json`、`raw/required_pathway_metrics.json` 存在。
- 内部验证和外部预测表存在。
- `logs/errors.log` 没有 traceback。

## 四套固定配置

| 配置 | 任务 | target | sampling |
|---|---|---|---|
| `config_task2_gene_dense.json` | 任务2 | 319基因→30通路 | dense |
| `config_task3_dense.json` | 任务3控制额外训练 | 30通路 | dense |
| `config_task3_stride.json` | 任务3空间隔点 | 30通路 | spatial_stride |
| `config_task3_random.json` | 任务3随机等量 | 30通路 | random_equal |

不需要手工修改这些JSON。具体科学设计和结论边界见 `实验设计.md`。

## 如果失败，发回什么

如果环境检查失败：发回完整窗口截图。

如果某次运行失败：发回窗口最后50行，以及窗口最后显示的 `Copy this return folder:` 对应目录中的：

```text
run.json
config.json
logs\startup.log
logs\stdout.log
logs\errors.log
logs\console.log
```

不要反复更换权重或改用43/44种子。
