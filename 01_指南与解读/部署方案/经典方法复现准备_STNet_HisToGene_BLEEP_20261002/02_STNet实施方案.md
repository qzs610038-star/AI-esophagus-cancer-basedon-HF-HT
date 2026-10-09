# STNet 独立复现包实施方案

日期：2026-10-02。**设计文档，未实现/未训练**。共同数据、指标、路径、阶段合同见[总览](00_总览与共同实验合同.md)；原文/源码链接见[索引STNet段](01_论文与官方代码资料索引.md#2-stnet-的一次资料入口)。候选未来包：`experiments/stnet_reproduction_v1/`，当前未创建。

## 1. 原论文做什么

He等2020，Nature Biomedical Engineering，DOI `10.1038/s41551-020-0578-x`。目标是用spot周围H&E形态预测局部基因表达，分析肿瘤内异质性。主数据为23乳癌患者、68切片、30612spots，观测26949genes；原ST spot直径约100µm、中心距200µm。输入224×224像素约150×150µm²，预测全队列平均表达最高250genes（原文PDF1–2、6页）。

完整方法：ImageNet DenseNet121，共享1024维视觉表示→250维线性多输出头；所有参数微调，平方误差回归。官方直接替换classifier为带bias的Linear；helper先清零weight/bias，完整gene训练入口随后把bias设为训练标签均值，最终是**weight=0、bias=train_mean**，不加额外MLP/激活/dropout。tumor/normal病理注释不作为额外模型输入或主监督。训练随机D4旋转/反射，测试八种对称预测求平均。不是冻结UNI特征加MLP，也没有项目经典空间残差分支。

外层留一患者，22人训练/1人测试；内层22训练患者四折择epoch。原配置SGD `lr=1e-6,momentum=0.9,batch=32,max_epochs=50`；源码按四折总SSE择epoch，**不是PCC**。另有单组织两Visium切片234匹配genes的独立外测，以及TCGA pseudo-bulk验证；不能混成同一种spot指标。Methods和图表脚本的slide/患者合并粒度也有差异：源码汇同患者切片spots后逐患者×gene算PCC，展示患者中位数/正相关患者数；不是项目患者×通路等权算术均值。原10x平均PCC0.33是234genes逐基因相关平均，不能当作项目验收阈值。（PDF6–7页与官方分析脚本）

## 2. 官方代码和依赖状态

固定commit `43022c1cb7de1540d5a74ea2338a12c82491c5ad`（2021-09-08），[官方代码](https://github.com/bryanhe/ST-Net/tree/43022c1cb7de1540d5a74ea2338a12c82491c5ad)。完整tree无LICENSE/COPYING；setup.py只有MIT classifier，GitHub license=null，**完整许可尚未确认**。未来可先独立实现论文机制并引用方法；直接复制源码前须明确许可依据，不能自行添加MIT文本冒充上游许可。

`setup.cfg`为历史Python~=3.7、torch~=1.0.0、torchvision~=0.2.1、numpy~=1.16.1、pandas~=0.24.1、scipy~=1.2.1、scikit-image~=0.14.0、argcomplete~=1.9.4、tqdm~=4.26.0，以及未锁PyYAML/matplotlib。OpenSlide/Pillow、ImageMagick等依赖不完整。当前项目历史torch2.6/torchvision0.21不是该环境的已验证替代，当前不安装旧环境。

静态问题：setup metadata/入口仍histonet；setup.py调用缺失`refresh_version.py`；eager Ensembl模块需要tree未提供的`ensembl.tsv/pkl`；Unix shell、affinity、symlink、旧PIL/torchvision调用和作者路径不直接兼容Windows。图表/random forest分支不进入核心包。旧CV脚本删除输出/部分checkpoint，未来必须改成新run、非覆盖存储，**不直接执行上游脚本**。这些问题未通过实际import验证；不能称上游成品可运行。

## 3. 精确方法合同与项目差异

官方标签变换是先取全基因计数分母，再截取目标：

`y_g = ln((c_g + 1) / (Σ_all_genes c + G_all_genes))`。

不是top250内归一，也不是log1p(CPM)。全队列基因并集补零及平均表达筛选出现在源码，论文Reporting Summary也写across all patients；可执行原数据线只能用外层训练集筛选，并为内层模型选择严格控制其拟合范围。若每内折筛选得不同gene列表，内折SSE尺度/通道不能直接无说明相加：须预定独立于评价队列的目标列表，或内折自身任务的验证误差汇总规则，并保留所选列表与有效数。不能用外层test或内层val拟合列表后称完全独立。此处是原协议修订，逐数值复现不成立。

| 因素 | 原论文/官方 | 本项目B线设计 |
|---|---|---|
| 数据/组织 | 乳癌ST/spot | ESCC Visium HD派生patch；物理尺度待核 |
| 输入 | 原图中心224px局部视野 | 共同身份对应的完整原patch缩放224，保留FOV；称项目输入适配，不能冒称150µm |
| 视觉模型 | ImageNet DenseNet121，全参训练 | 保留；离线权重、来源和版本登记，不换UNI |
| RGB标准化 | 源码从train loader约12batches估计mean/std | 用训练图像的确定性统计清单拟合并冻结；若为确定性抽样，记录seed/身份/样本数。与原抽样实现差异明确记录 |
| 目标 | normalized-log top250gene | 原V003 30通路训练z，线性头30；绕过gene公式 |
| split/selection | LOPO+内部四折SSE | 固定6人block内部+XZY；选模沿用项目内部macro PCC，tie用患者等权zMSE |
| 输入增强/推理 | D4；8对称TTA | 保留D4和8-TTA，全部方法输入管线区别记录；不使用跨split标签平滑 |

原论文3×3邻居平滑用于额外评价，不是主监督替代；B线不在边界两侧平滑训练/验证真值。若之后算平滑指标，单列明确集合和邻域，不能把更有利值替换主表。

## 4. 两条路线的包内设计

A线首先只设计主乳癌数据机制复现，原Visium/TCGA额外任务单独列待办；不一次扩张成所有论文应用。保留原网络、计数分母、D4/TTA、外层患者/内层CV及原指标；train-only基因筛选与原数据发布版本变化放入`deviations`。原Mendeley v5是否与原论文数据完全一致、基因映射资产待确认，不能假定。严格A线需要选模数据处理的内折隔离，不能因为原作者全队列筛选就照搬。

B线首轮完整CNN直接30，使用共同9472/1078/XZY1039及V003；不能用top250表达估计主线ssGSEA。两线配置、标签变换、选择指标和输出尺度隔离。A线原SSE与B线项目PCC均有明确名称，不能取两者中有利规则混用。

共同骨架外拟定模块：`image_dataset.py`（身份+图像）、`densenet.py`（native backbone+head）、`gene_preprocessing.py`（仅A线）、`rgb_stats.py`（train-only）、`augment.py`（D4）、`tta.py`、`original_cv.py`（仅A线计划）、`project_train.py`、`export.py`。实施优先独立机制实现/明确许可源码，保留精确层、权重加载和输出合同。

接口：`model(images[B,3,224,224])→pred[B,K]`；训练保留源码`loss = sum_B sum_K (pred-target)^2 / K`，即按通道平均、按batch求和，**不除以B**。它与常见MSELoss(mean)的梯度尺度不同；减batch时须记录有效步长变化，不能沿用lr后说训练完全等价。TTA将同身份八预测均值后导出。图像预训练权重不含本项目监督历史；不能热启主线回归head。B线所有DenseNet层均进优化器，不能意外只训练classifier。新head weight清零、bias取训练z均值（应约0但仍保存真实统计），合法导致首步backbone梯度为零，检查至少第二步的传播，而非把首步无更新判错。

| 参数 | A线依据 | B线首版建议/待生效 |
|---|---|---|
| 网络/输出 | DenseNet121/250；实际目标表按train-only方案冻结 | DenseNet121/30，线性无输出非负限制 |
| SGD | lr1e-6、momentum0.9、weight_decay0，batch32，最多50轮 | 首轮同配方；源码parser默认wd0且主README无覆盖 |
| loss/head | sum squared error / K；Linear weight0、bias训练均值 | V003训练z；保留归约和最终初始化，不加对比/TV/空间辅助loss |
| selection | 四折SSE择epoch，外层重新训练 | 项目macro、zMSE tie；建议formal从第1轮，早停计数16轮/耐心10/min_delta1e-4，训练上限50；未来冻结前确认 |
| 图像统计/增强 | train-only RGB、D4 | 全FOV resize224；train-only统计；8-TTA只用于预测 |
| seed | 原始seed细节未完整锁定 | 建议42/43/44，不按XZY挑选 |

B线第1轮开始选模不继承主线关系warmup；这是新方法选择窗口建议，不能声称既有主线所有超参照搬。正式日志同时保存学习率、优化器参数集、训练参数数和被选epoch。

## 5. 环境、测试和命令模板

拟移植核心PyTorch/torchvision，不为实现核心强制引入OpenSlide/Ensembl/HistomicsTK。A线整片读取需要时才引入明确缺失的I/O依赖；B线现有patch无需将JPG先转tiled TIFF。保留ImageNet权重映射和模型数值差异记录。新版本requirements须根据实际兼容测试锁定，不把历史区间当安装完成。

S0验收：合成边缘标记图验证FOV和D4/TTA；全基因分母小计数例、拟合keys排除val/XZY；CPU小batch forward/至少两步，核对weight0/bias训练均值的首步行为及随后backbone/head更新、源码loss归约、save/load一致；shuffle后的身份连接、双PCC与SSE各自定义；独立目录启动及失败不删除旧输出。方案阶段未执行这些测试。

S1：训练侧少量块验证短训练可行性，不加载XZY；量batch32吞吐/峰值显存，如需减batch记录变更。S2/S3按共同阶段完整点位/多seed、全部失败矩阵；原论文指标不作为性能门禁。

以下是**未来设计的接口，文件及参数目前未实现，不可直接执行**：

```powershell
# 未来上传后的 D:\AIPatho\qzs\code\stnet_reproduction_v1 内
.\run.ps1 -Config .\configs\project30.json -Action check-inputs
.\run.ps1 -Config .\configs\smoke.json -Action smoke
.\run.ps1 -Config .\configs\project30.json -Action train -Seed 42
.\run.ps1 -Config .\configs\project30.json -Action predict-internal
# 单独显式动作，仅在配置/checkpoint冻结后
.\run.ps1 -Config .\configs\project30.json -Action external-eval
# A线仅在原数据和预算明确后独立启动
.\run.ps1 -Config .\configs\original_data.json -Action original-cv
```

回传CNN checkpoint登记、原始预测、RGB统计、TTA与图像视野版本、输入键和实际参数；不回传训练图像/大权重。新缓存只放feature_caches，不复用UNI缓存。

## 6. 资源、先后和未确认事项

CNN训练比现有冻结head计算重；未测GPU时间。原完整主数据CV最多23×(4内折+1外层)=115次模型拟合，不适合作为近期B线的必要条件；8-TTA增加推理计算约8倍。B线单seed先测再估三seed预算，无GPU任务已启动。

未确认：完整源码许可、原数据版本与现存位置、HGNC映射、ImageNet离线权重、真实patch物理支持域、train-only筛基因的内外折合同、B线新选模窗口/资源预算。A线可以继续补资料，不要求下载；B线可在下一次实现授权后先推进完整CNN。

## 7. 方案最小自检

| 模块 | 判断及证据范围 |
|---|---|
| 约定/参数 | PASS（设计覆盖）：原SSE与B线PCC、网络/训练/TTA分列；B线window和参数仍待冻结，不代表运行值 |
| 数据/输入 | WARN：train-only已约定；全队列top250历史偏离已揭示；物理域/原数据实体未验证 |
| 学习行为 | UNVERIFIED：已定义全参梯度和TTA检查，无实现/运行 |
| 选择/交付 | PASS（设计覆盖）：项目指标/外部冻结/全部seed、权重与缓存登记；新结果不存在 |

可支持方法资料与复现设计；不能支持完整许可确认、成品可运行或原论文分数复现完成。
