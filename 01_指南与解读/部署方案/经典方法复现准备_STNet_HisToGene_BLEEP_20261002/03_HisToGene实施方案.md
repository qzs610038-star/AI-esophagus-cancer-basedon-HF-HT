# HisToGene 独立复现包实施方案

日期：2026-10-02。**设计文档，未实现/未训练**。共同数据、指标、路径及分阶段验收见[总览](00_总览与共同实验合同.md)；来源见[HisToGene索引](01_论文与官方代码资料索引.md#3-histogene-的一次资料入口)。未来候选包 `experiments/histogene_reproduction_v1/`，当前不存在。

## 1. 核心机制、原始任务与协议

Pang、Su、Li，2021-11-28 bioRxiv v1，DOI `10.1101/2021.11.28.470212`。本地31页PDF与官方metadata对应预印本；本次API `published=NA`，不冒称正式期刊版。

模型从H&E和spot坐标预测gene，并以密集采样推断更细空间图。关键单位是**整片spots的序列**：每spot RGB112×112全部像素展平，再线性投影1024维；x/y学习编码相加；同片N个spot构成N×N全局attention。没有单spot内小patch token序列、CLS或UNI预训练骨干。论文与tutorial采用8层、16头，每头64维，dropout0.1、lr1e-5、100epochs；FFN为2×dim（源码）。全随机参数训练，多输出MSE。（PDF6–8页、Figure1/PDF27页）

HER2+原资源36切片/8患者，保留至少180spots的32片/7患者，9612spots/785genes。原评估32-fold留一**切片**：31片训练、1片测试；可能同患者切片跨训练测试，**不是LOPO**。cSCC扩展GSE144240为12片/4人、6630spots、134genes，亦留一切片。原PCC在每切片每gene的spots间算，展示785gene分布；病理标注6片的聚类ARI和top100gene的fgsea是辅助分析，不是本项目ssGSEA监督。（PDF9–11、14–15、18、20–21、Figure2/PDF28页）

原超分辨率用密集112patch及重叠区域四预测平均，形成四倍面积分辨率；HisToGene*再把四相邻子块求和恢复spot级。没有更密分子测量真值时只能称模型密集推断，不能声称实测分子超分辨率。近期B线只评价原既定点位，不把超分辨率作为额外必做任务。

## 2. 官方版本、依赖和已查差异

固定commit `44ff75b9b68949796f27f1b4121d36ea4cba8bf3`（2021-11-30），[官方源码](https://github.com/maxpmx/HisToGene/tree/44ff75b9b68949796f27f1b4121d36ea4cba8bf3)，MIT许可证；保存版权/许可和移植改动。原文CC-BY-NC-ND4.0与代码MIT分别登记，不把论文图当MIT代码资产。

README只列torch≥1.8、pytorch-lightning≥1.4、scanpy≥1.8，没有完整lock/environment。实际还导入torchvision、torchmetrics、einops、scprep、anndata、numpy/pandas、Pillow、sklearn、matplotlib/seaborn/tqdm。旧Trainer `gpus=...`与旧accuracy API不保证在当前库兼容；不安装、不宣称当前环境可运行。未来建议保留模型主体，迁移最小训练循环到明确版本PyTorch，模型forward/MSE/Adam等行为定向对齐；是否保留旧Lightning为独立环境选项，未经验证不能填成锁定依赖。

| 事项 | 论文/示例/源码核验 | 实施影响 |
|---|---|---|
| 深度/lr默认 | 论文/tutorial8层、lr1e-5；README展示4层；构造默认4层、lr1e-4 | 必须显式8层/lr1e-5，不从构造默认倒推论文 |
| 输入shape | README省略batch维；model用`centers[:,:,0]` | 实际`[B,N,37632]`和`[B,N,2]`，官方DataLoader B=1 |
| attention/padding | 16头、QKV无bias、pre-LN、attention/FFN残差；无padding mask | 首版B=1，不混片；若加padding须mask，不能影响有效点 |
| 图像数值 | ViT_HER2ST RGB float保留0–255；轴转换后HWC展平；声明的transform未实际应用 | 不套ImageNet标准化，不误用单patchViT；展平顺序/裁剪需小图对齐 |
| 原训练 | tutorial仅fold5，batch1、100轮，无独立val loader/early stop，保存last | 单fold例子不是完整32fold复现；A线固定100不能用test挑epoch |
| 切片列表 | Dataset按目录排序`names[1:33]`选择，再leave-one | 包内显式固定32片名单及真实ID，不能依赖目录排序 |
| 密集推断 | `sr_predict`输出密集网格，公开函数未完成论文重叠平均/回聚合 | 不把函数存在写成超分辨率完整流程 |
| 启动/资产 | `predict.main`含未定义类/参数等残留；cSCC return在填充循环内、缺静态gene表、硬编码作者路径 | 先HER2机制及B线，不默认承诺cSCC/原main可运行 |

HisToGene论文中的ST-Net基线是作者自己的替代实现，不能把其`STModel`复制到另一包当官方STNet。

## 3. 标签与公平性关键差异

论文PDF6页描述按**全genes**总UMI归一、乘1e6、取自然对数；零值pseudocount没有清楚定义。源码先选静态785genes后调用无参数scprep归一/log；可查scprep v1.1.0及后续默认值对应：

`log10(1 + 10000 * c_g / Σ_selected_785 c)`。

分母、乘数、对数底和零值处理均不同；历史实际scprep版本未锁定，不能凭当前默认证明作者当时标签。A线配置分别命名`paper_label_contract_pending`、`official_label_train_only`，记录差异；不能挑更好的一种却写唯一原版。若选论文路线需先解决零值定义，未解决时不能产生伪造公式。

论文PDF6页说训练ST筛gene，9页又说所有32片HVG/表达≥1000spot筛选；静态785列表每fold来源未公开。当前项目禁止外测/内部选模集参与拟合，A线只在允许训练片筛选，或采用真正独立预定义目标清单；静态官方清单可以作历史比较事实，不能当无泄漏证据。内折若引入开发选模，同样须隔离筛选。原论文切片留出也不证明跨患者泛化。

B线绕过原gene变换，直接读共同V003标签，train-only9472均值/SD；输出30维，无非负约束。保留8层spot attention、像素投影与坐标编码；不能用UNI-CLS/token替代原始像素投影。785HVG或原文fgsea不够证明可还原30ssGSEA；gene→pathway另受共同标签缺口约束。

## 4. B线的输入、坐标、序列和资源设计

共同原patch完整FOV缩放到112×112供原像素投影，与224全FOV/原论文裁剪的像素和物理尺度差异都登记。首轮采用官方RGB0–255、HWC展平布局，不自动AugMix/ColorJitter/MixUp/TV/关系loss；图像transform必须通过有标记合成图核对轴与中心，不能凭shape正确判断对应组织。

严格按`patient_id + slide_id + split`组sequence：训练仅训练点、内部仅验证点、XZY仅外部点，labels只在训练loss/独立验证scorer出现。同患者跨split仍是项目原协议；组内全局attention可读取该查询组全部**无标签H&E**，不能带训练标签或拼其他片。每组点排序固定并随预测还原identity；shuffle仅改变sequence访问顺序，不能打散输入/标签的配对。首版batch_size=1避免padding。

当前每个来源组视作一序列时训练最大N=3014（LMZ），内部最大337，外部1039；数字来自split与已接纳外测，不证明来源组就是一个物理切片。不能把六人9472点合成一序列。每epoch按六训练sequence各访问一次，更新次数与spot minibatch方法差异如实登记，epoch数不能直接证明计算预算相等。

官方`Embedding(n_pos=64,dim=1024)`合法索引0–63，输入源是ST网格坐标，不是本项目上万像素坐标。设计`coordinate_adapter`仅依据训练数据或已有固定采集几何声明原点、方向、stride和量化单位；保存不可变规则，禁止用XZY标签或全评价坐标拟合min-max缩放。可用无标签、预先已登记的采集几何作确定性单位转换，但不能冒充监督拟合。

S0先核原点/整数网格以及各split越界数。`n_pos`若按已知采集尺寸扩展，会增加新嵌入参数、未知位置泛化风险，应记录为坐标容量适配；不能按观察外部分数扩表，不能把越界悄悄clamp到63。无法定义既定映射时明示缺合同，继续模型合成验证，真实完整训练未就绪。连续位置编码、每片min-max归一或局部窗口均改变原方法，另名实验而非静默修改。

资源：8层/785gene/n_pos64静态参数估算约106.65M；30维头仍约106M，非轻量head。FP32单份attention score为`16*N²*4bytes`，N3014约554MiB/层；八层再加softmax、激活、梯度、Adam状态，16GB可否承载未实测。原论文V10032GB、31片训练平均11分钟不能外推本项目。优先减少一次并行序列数、逐组I/O、验证无梯度及保持同attention上下文的内存优化；任何SDPA/数值精度迁移必须小例核对，不能假设等价已证实。必须窗口化时，作为`window_context_adapter`另版本记录窗口/边界/聚合，**不能称全片官方attention已忠实复现**。

## 5. 独立包模块、配置和输出

共同骨架之外：`pixel_sequence.py`（图像+身份）、`coordinate_adapter.py`、`spot_transformer.py`、`sequence_loader.py`、`gene_labels.py`（A线）、`original_slice_protocol.py`（A线）、`project_protocol.py`、`dense_predict.py`（后续可选、不首轮实现）。MIT源码可明确复制及最小兼容修改；声明原始来源和补丁，保留官方基因与像素变换的独立配置。

接口：`model(flat_pixels[1,N,3*S*S], coords[1,N,2])→pred[1,N,K]`，每点保留对应key；无CLS汇总，不把全片压成一个患者预测。loss是valid spot×target的MSE均值；变长序列B1不pad。B线选模用共同内部macro，tie患者等权zMSE；A线原演示fixed100last，补完整留一切片runner而非用held-out切片选checkpoint。

| 参数 | 建议显式值 | 来源/生效情况 |
|---|---|---|
| S/dim/depth/heads/dim_head/FFN | 112/1024/8/16/64/2048 | 论文+源码；尚未构造运行 |
| 位置表 | 原64，B线待geometry核定固定容量 | 不默认沿用64给像素坐标；不得测试集调容量 |
| 训练 | Adam,lr1e-5,dropout0.1,batch_sequence=1,max100 | tutorial/论文；Adam其他默认及有效值实施时登记 |
| 输入尺度/输出 | RGB0–255、官方展平顺序；A=实际gene表、B=30 | B线图像FOV resize适配；B线用V003 z |
| B线选择 | formal第1轮；建议计数16轮/耐心10/min_delta1e-4 | 新选择窗口建议，未来冻结；与A固定100分开 |
| seeds | 建议42/43/44 | 共同项目重复，不按外部成绩选 |

输出除共同合同外，保存sequence group、每组N、identity排序、coordinate规则/范围/容量、有效mask（若有）、裁剪与展平版本、完整/窗口context状态。每spot都有预测，不能只回传高PCC基因/切片图。不跨线拼z分数报告一个flat PCC。

## 6. 最小冒烟到正式实验

S0：CPU合成小N核查像素轴/颜色、坐标合法/越界、8层与优化器参数集合、单步像素投影/position/attention/head更新；输入/标签一致排列时输出相应排列，无遗漏spot；batch1变长、save/load及含padding扩展时mask；train/val/XZY split隔离。只验证真实变化部分，不跑整个旧训练链。

S1：仅训练/内部小组短训练与loss诊断、测最大sequence资源，避免跨split上下文；若选小N不能据此证明N3014可运行，额外对最大N无梯度/训练内存做明确profile后再决定完整配置。没有GPU授权前不执行。

S2/S3：锁定坐标和context、完整单seed再配对多seed。XZY query只给图像和已定义坐标，预测后独立scorer连真值；不能按XZY越界/表现临时换模型再保留旧“外测”称谓。完整失败保留，部分结果按部分报告。

**拟定命令接口尚未实现，目前不可运行**：

```powershell
# 未来上传后的 D:\AIPatho\qzs\code\histogene_reproduction_v1 内
.\run.ps1 -Config .\configs\project30.json -Action check-inputs
.\run.ps1 -Config .\configs\smoke.json -Action smoke
.\run.ps1 -Config .\configs\project30.json -Action profile-sequences
.\run.ps1 -Config .\configs\project30.json -Action train -Seed 42
.\run.ps1 -Config .\configs\project30.json -Action predict-internal
.\run.ps1 -Config .\configs\project30.json -Action external-eval
# 原HER2路线另行授权数据/预算后
.\run.ps1 -Config .\configs\original_data.json -Action original-slice-cv
```

新权重/缓存/原始结果及本地analysis按共同路径合同；此包不自动导出超分辨率或Phase3消费端，不替换历史HisToGene目录。

## 7. 未确认与方案最小自检

待核：HER2原数据是否已在本机、论文/源码标签路线和零值规则、785gene每fold来源、坐标原点/表容量与真实切片、112图像物理支持域、最大N训练资源、旧Lightning兼容或最小迁移。建议B线完整8层直接30先行，A线以HER2为主，cSCC/密集推断暂列单独后续，不自动启动。

| 模块 | 判断及限界 |
|---|---|
| 约定/参数 | PASS（方案）：原8层与4层默认分开，整片spot序列主体保留；坐标/context尚待锁定 |
| 数据/输入 | WARN：原全切片筛选歧义、gene公式冲突明示；B线同V003/split；物理几何未验证 |
| 学习/空间 | UNVERIFIED：无实现；约定每group/split独立及梯度/坐标/最大N检查，不称运行通过 |
| 选模/交付 | PASS（方案）：A固定epoch与B内部选模分列、完整spot覆盖/失败披露；预测未产生 |

当前只支持实施设计完整性，不支持官方入口可运行、全片注意力已承载、密集分子表达已复现或新实验结论。
