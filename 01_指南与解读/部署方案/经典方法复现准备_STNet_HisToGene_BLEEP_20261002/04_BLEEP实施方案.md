# BLEEP 独立复现包实施方案

日期：2026-10-02。**设计文档，未实现/未训练**。共同数据/标签/指标/路径见[总览](00_总览与共同实验合同.md)；资料见[BLEEP索引](01_论文与官方代码资料索引.md#4-bleep-的一次资料入口)。未来候选包`experiments/bleep_reproduction_v1/`，当前不存在。

## 1. 原始问题、数据、完整方法

BLEEP，NeurIPS2023，arXiv2306.01859。用bi-modal contrastive learning将H&E和表达向量映入共同空间，通过训练RNA参考库检索推断，避免高维gene直接回归的均值收缩。它不以空间邻域图为必要输入，也不是项目回归head上加一个辅助contrastive loss。

主数据GSE240429为四张连续人肝切片，QC后2378/2349/2277/2265spots，合计9269；切片#3外留、#1/#2/#4训练，**不是四患者LOPO**。spot中心224px patch约55µm边长。历史预处理逐spot total-count normalize/log1p，每切片top1000HVG的**并集**共3467genes；四片联合Harmony后才留#3。该流程使用测试RNA参与筛选/批次校正，官方preprocess notebook也如此；与项目train-only边界冲突。（原文PDF3–5页）

完整模型：ImageNet预训练且可训练ResNet50，global avg pooling出2048维；图像2048→256和RNA3467→256两个残差投影头：首Linear→GELU→Linear→dropout0.1，残差加首投影，再LayerNorm。RNA没有另一个额外RNA backbone。双向soft-target cross entropy训练后，**image-only query→训练RNA embedding bank→对应RNA值聚合**。原PCC是MG8/HEG50/HVG50逐gene Pearson的均值，非项目30通路患者等权或展平PCC；Table3的±是3次重复对均值的最大差，不能改称SD。（PDF8–9页）

## 2. 固定官方版本与当前可运行状态

commit `23959670c7407510c1716a173184bdcc18833a35`（2024-10-22），[官方源码](https://github.com/bowang-lab/BLEEP/tree/23959670c7407510c1716a173184bdcc18833a35)，Apache-2.0。当前提交晚于论文，不能证明就是原checkpoint训练版；保留许可证/来源和移植改动。

无完整requirements/environment/lock。实际依赖torch/torchvision/timm、numpy/pandas、OpenCV、Pillow、sklearn、scipy、scanpy、harmonypy、tqdm；绘图另matplotlib/seaborn。notebook显示历史预处理Python3.7.16、推理3.10.2、Scanpy1.9.3，只是线索。训练脚本依赖SLURM环境、NCCL/DDP、`.cuda()`，不能原样当Windows单卡运行入口；当前不安装/不验证GPU。

源码静态直接问题：`dataset.py`未初始化`self.is_train`；`TF.to_tensor`已CHW后又`permute(2,0,1)`得到WCH；OpenCV BGR直接送PIL；`~/`等硬编码路径未统一展开；推理notebook缺显式pandas导入，依赖`hvg_union.npy`但可见预处理未保存；NaN PCC过滤后用原gene index取子集可能错位。未来只做必要的RGB/CHW/路径/入口/身份修复并记录，不能静默混成方法改良。**当前代码未运行，不宣称这些已修好。**

## 3. 主文、补充、源码差异的裁决设计

| 项目 | 主文 | 补充S3 | 当前源码/示例 | 首版建议 |
|---|---|---|---|---|
| 训练 | AdamW，lr1e-3，batch512，150epoch，4V100 | lr1e-4、wd1e-5、batch512 | lr1e-3、wd1e-3、batch256、默认4epoch | 建议原数据线选补充完整配方`lr1e-4/wd1e-5/150epoch`并明确主文冲突；B线同作为起始建议，实际对比batch按资源锁定。未获确认不称唯一原版 |
| top-k聚合 | Table3及论述默认top50算术均值；weighted是比较项 | 写weighted | inference默认average、另weighted | 首版K50算术均值，weighted另名可选，不自动多跑 |
| 检索距离 | Algorithm1 Euclidean | 未说明归一化 | normalize后cosine dot/topk | 保留源码cosine；等价于normalized vector Euclidean排序，不等于raw Euclidean |
| soft-target温度 | Algorithm1乘tau | 未消除歧义 | ResNet50除tau，其他分支乘tau | 选ResNet50源码、tau1，记录公式，禁调后悄悄统一 |
| key符号 | Algorithm1中Hv符号似image bank | 不完全澄清 | 实际key=训练RNA embeddings | 明确RNA key，payload=训练表达/通路；不混为image bank |

总览的“确认训练版本”仍待用户决定；建议上表使用补充作为有完整wd依据的起始版本，**不因预期好分选择**。也可选择当前源码完整超参并命名`official_source_recipe`，不能把代码默认4epoch称论文150epoch完成。主文有lr而缺明确wd时，不从别处拼凑成“主文唯一设置”。

ResNet50源码用：

`logits = H_rna @ H_img.T / tau`

`T = softmax((H_img @ H_img.T + H_rna @ H_rna.T) / (2*tau), dim=-1)`

对logits与logits.T分别做soft-target CE(T/T.T)，双向平均再batch均值。T按行softmax后其转置不保证每行和为1，源码也未对T.T重新归一；复现时不自动“修正”此行为。训练embedding不显式L2 normalize；T没有detach。不同分支的温度乘/除及targets梯度路径需保留和检查，不能替换为hard身份标签CLIP或自动detach教师。tau=1默认下乘/除数值相同，但以后变化不等价。

推理先把训练RNA key和query image embedding做L2 normalize，再cosine topK，默认训练payload算术平均。weighted分支用距离指数权重，作为单独配置保留，不用XZY挑aggregation/K。整条链无query RNA输入。

## 4. 原数据路线与30通路适配的边界

A线保留原肝数据、ResNet双塔/soft targets/训练RNA bank、gene预测与原指标；历史全四片HVG/Harmony仅作资料。可执行`leakage_safe_original_data`用训练#1/#2/#4筛gene并冻结训练统计；若内部holdout用于选模，筛gene/预处理更需只用训练部分。目标维度可能不再3467，须实际清单决定，不能凑到3467。

Harmony不能靠“先拟合后冻结”一句话假设有可靠unseen-query transform。首轮防泄漏原数据线建议省略联合Harmony，并明确是预处理偏离；若后来要批次校正，必须提供只训练拟合、可固定映射新数据的实现和验证，不能把query RNA连训练RNA重跑Harmony。历史已处理全四片矩阵不能包装成新项目无泄漏预处理证据。A线方法机制可以保留，逐数值复现无法承诺。

B线直接30通路：图像ResNet50完整训练保留，RNA侧输入改为**训练V003 z向量[30]**，残差投影30→256；bank key由同checkpoint的训练通路encoder产生，payload为同训练30z，query只图像。训练以双模态soft target为主要目标，不自动加MSE直回归。检索聚合输出30z→按训练参数一次逆变换；此版称“BLEEP完整机制30通路适配”，不是官方3467gene任务。

输入原项目完整patch FOV缩放224、官方预训练ResNet使用明确RGB/ImageNet normalization；修复官方颜色/轴问题必须登记，与原论文55µm关系未验证。不能用UNI缓存/冻结ResNet中间特征代替完整可训练图像塔。若之后比较共同PFM骨干，单独叫“骨干控制变体”，不替换本主比较。

基因预测后评分另列探索：HVG未必覆盖原30通路成员/背景；`ssGSEA(mean(gene))`通常不等于`mean(ssGSEA(gene))`。参考payload选gene还是原通路会改变算法输出，不能混称同一种目标。须遵守共同目标合同，不能因XZY展平PCC高就忽略标签不等价。

## 5. 严格参考库合同及模块

当前官方训练把#1/#2/#4随机spot80/20做内部loss选择，但inference又用三片全部RNA进bank，包含该内部holdout。项目内部评价须改变为**仅训练9472**的bank；内部1078/XZY1039皆不得入库、拟合或对比训练。未来LOPO若另行设计，每折须重训标签塔与库，不能共用六患者监督库。

拟定模块：`paired_dataset.py`、`image_encoder.py`、`projection.py`、`soft_target_loss.py`、`train.py`、`bank.py`、`retrieve.py`、`predict.py`、`gene_preprocess.py`（A线）、`environment_check.py`和共同I/O/指标模块。单卡launcher去掉SLURM依赖，DDP不是首轮必要条件；不继承作者路径或依赖包外notebook变量。

接口及返回：

- 训练 `forward(image[B,3,224,224], target[B,K])→loss/embeddings`，B≥2；不能将eval真值装入训练Dataset。
- `build_bank(selected_checkpoint, train_manifest, train_labels)→Bank`，key `[Ntrain,256]`，payload `[Ntrain,K]`；二者同identity行。
- `predict(query_images, checkpoint, bank)→pred[Nquery,K]`，接口没有query标签；检索允许分块，数学结果与小矩阵topk一致。
- 预测后scorer独立连接真值；query无标签或替换为错误标签时推理值不变，阻止无意走“有RNA shortcut”。

`bank_manifest.json`与`feature_caches.json`保存服务器绝对路径、checkpoint/模型与预处理版本、label_version/pathway_order、训练split/患者/来源组/spot键、RNA输入/投影/payload维度、n_rows、dtype、K/距离/归约；实际keys严格等于训练允许keys，禁止只写`source=train`却不检查内容。建库用selected checkpoint eval模式；不能沿用其他epoch训练embedding或同路径旧模型库。权重/库新run保存，不覆盖历史。

回传训练库身份索引、路径登记、检索设置以及少量diagnostic topk/key/similarity（可配置采样）；不回传库本体或患者图像。train bank约9472×256×float32=9.25MiB，30维payload约1.08MiB（仅数组，非磁盘/显存总量）。query×bank检索分块，不把全部查询矩阵无条件放GPU。

## 6. 参数、环境与分阶段验收

| 参数 | 首版建议 | 依据/未验证 |
|---|---|---|
| 图像/投影 | ResNet50 ImageNet可训练；2048→256残差头；dropout0.1 | 官方modules；离线权重存在性待核 |
| B线label投影 | 30→256残差头，输入训练z | 完整机制目标适配；不改变主线标签 |
| loss/tau | ResNet50源码soft CE，tau1，不detachT | models.py；两方向归约待实际检查 |
| optimizer/lr/wd/maxepoch | 建议AdamW/1e-4/1e-5/150 | 补充S3+主文epoch；版本裁决仍待确认，实际对象必须日志 |
| 对比batch | 原512；B线初测256，必要时128/64并固定 | 单卡资源适配；batch变化会改软目标和候选集合，梯度累积不能补回 |
| 检索 | 训练RNA key cosine、K50、算术平均 | 主文Table3与官方默认；小bank仅冒烟用min(50,N)，标非正式 |
| B线选择 | 内部检索预测macro、患者zMSE tie；建议formal1、计数16/耐心10/min_delta1e-4 | 项目端点公平性适配；官方按内部contrastive loss选模，两者分列 |
| seed | 建议42/43/44 | 配对项目重复；原文±不是本项目seedSD |

B线训练每epoch生成/更新当前**训练**bank并给内部图像检索预测，用内部项目指标选端点；可以缓存该epoch eval bank降低重复成本，缓存必须绑定epoch且选定后明确final bank。不能用contrastive loss最优与检索macro最优中外部更好的一套选择。完整ResNet训练期不能把图像embedding当固定特征永久复用。

S0 CPU小batch≥2：RGB/CHW与预处理边缘图、投影残差/LN、loss双向/温度/target梯度路径；实际优化器覆盖两塔；bank严格训练key、所有query-label去除/扰动后推理不变；topk cosine手算和分块一致，tie规则固定；K越界/零范数显式处理；保存加载同checkpoint一致。没有实施时不宣称这些测试已通过。

S1开发小样本短训练，量ResNet显存、有效对比batch和每epoch选模bank成本；至少检查loss有限、参数更新及检索预测不只是误连标签。资源不足降低对比batch须作为配方差异记录，不宣称等价复现原512。S2锁定配方/K/bank后全训练内部，S3配对multi-seed及冻结后的外部；全部成功/失败保留。

**下列是拟定接口，文件/参数当前未实现，不可直接运行**：

```powershell
# 未来上传后的 D:\AIPatho\qzs\code\bleep_reproduction_v1 内
.\run.ps1 -Config .\configs\project30.json -Action check-inputs
.\run.ps1 -Config .\configs\smoke.json -Action smoke
.\run.ps1 -Config .\configs\project30.json -Action train -Seed 42
.\run.ps1 -Config .\configs\project30.json -Action build-train-bank
.\run.ps1 -Config .\configs\project30.json -Action predict-internal
.\run.ps1 -Config .\configs\project30.json -Action external-eval
# A线另获数据/预算授权后，绝不调用全四片联合Harmony
.\run.ps1 -Config .\configs\original_data.json -Action original-data-train
```

新bank属于服务器feature_caches，原始预测/配置/路径登记属于runs；本地补全指标，不要求服务器跑所有图表/Registry。库payload是训练数据派生物，不能上传到第三方。

## 7. 资源、待决策与最小自检

原4V100/batch512/150是历史资源，未测Windows4080单卡耗时。主要成本为ResNet更新、B²soft-target矩阵、每epoch训练bank和检索验证。单卡小batch可研究可行性，但训练候选关系改变是实质偏离；当前不以复制4GPU为前置，不开GPU任务。S1后用真实epoch吞吐、bank生成/检索时间估单seed和3seed工作量。

需决定：主文/补充/源码训练版本（建议补充完整配方）、top50均值是否接受、直接30路线是否先行、共同seed/预算；需核离线ResNet权重和图像物理FOV。原肝数据线train-only且不做联合Harmony的偏离已明确，不以原分数设门槛。

| 模块 | 判断及限界 |
|---|---|
| 约定/参数 | WARN：版本冲突已列、建议明确但未裁决；soft-target梯度/公式与检索保留，无实际运行值 |
| 数据/输入 | PASS（设计边界）/UNVERIFIED（资产）：RNA bank训练键、query image-only、train-only预处理；历史全队列HVG/Harmony不照搬 |
| 学习/关系 | UNVERIFIED：无实现；双塔、soft targets、B≥2与gradient/axis已定义定向检查 |
| 选模/交付 | PASS（设计覆盖）：内部retrieval macro选模、无XZY反馈、K固定、库/权重/失败路径登记；新预测不存在 |

当前只支持资料和独立包设计；不能称官方当前代码可运行、原肝结果数值已复现、项目检索效果成立或BLEEP标签更优。
