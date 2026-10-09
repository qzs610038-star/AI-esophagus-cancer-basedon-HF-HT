# Phase 2 第二批静态图片 v2

本目录包含两张使用项目实际汇总表绘制的科研图。PNG 适合快速审阅，SVG 保留可编辑文字，PDF 用于论文排版。首批图片 v1 保持原样。

## 图片

| 图 | 内容 | 文件 |
|---|---|---|
| Fig P2-3S | 固定空间臂下六种编码器；内部验证和外部 XZY 的两种 PCC | [PNG](exports/Fig_P2_3S_six_encoders.png) · [SVG](exports/Fig_P2_3S_six_encoders.svg) · [PDF](exports/Fig_P2_3S_six_encoders.pdf) |
| Fig P2-5 | UNI2-h 既有通路目标基线与任务二基因路线的描述性结果 | [PNG](exports/Fig_P2_5_gene_route.png) · [SVG](exports/Fig_P2_5_gene_route.svg) · [PDF](exports/Fig_P2_5_gene_route.pdf) |

[打开本地静态图集](图集.html)可查看两张 PNG 缩略图并下载各格式。图集不使用远程脚本或交互式仪表板。

## 数据和复现

data 文件夹保存本次绘图使用的四份 CSV 副本；原始来源及逐表用途见 source_manifest.json。source/draw_figures.py 读取这些副本，不访问模型权重或预测文件。脚本核对六模型汇总均值、样本 SD 与逐种子值，并交叉核对 UNI2-h 基线及任务二来源表后再输出图和 render_checks.json。

在已具备 Matplotlib、NumPy 和 pandas 的 Python 环境中，从本目录运行：

    python source/draw_figures.py

图 1 的六种编码器各有三次运行，图上显示均值、样本 SD 和逐种子点。图 2 的 UNI2-h 基线显示三次运行的均值与样本 SD；任务二路线显示一次运行值，不提供样本 SD。患者—通路等权 PCC 与整体展平 pooled PCC 分开标示；同一 PCC 的内部验证和外部 XZY 使用相同横轴范围。

脚本使用现有绘图库生成 PNG、SVG、PDF，未安装新依赖。数值检查和最终 PNG 目视检查记录在 render_checks.json。
