# 最小审查记录

> **当前状态（2026-09-28）**：v006 第二批的 9 项成功训练、9 份内部预测和 9 份 XZY 外部预测已在 Registry 登记为 `accepted`。本批用户回传的服务器报告、结果路径及接纳范围见 `experiments/experiment_registry.json` 中 `phase2_backbone_spatial_ablation_batch2_20260919` 条目，以及对应回传目录的 `analysis/复核接纳记录_20260927.md`。下文 v002–v006 的阶段性审查文字保留为各阶段当时记录；较早段落里的“尚待服务器验证／尚未完成”不代表当前状态。服务器相关事实仍以用户回传材料为据，本次更新未实时连接服务器。

日期／方案或代码版本：2026-09-19 / `package.json` v002 / `fullfov-spatial-backbone-batch2-v1`  
本次改变与结论范围：修复 H-optimus-0 权重文件合同、基线证据可移植性、分模型续跑及部分完成退出语义；结论范围限于本地工程行为，不含服务器运行或科研结果。  
阶段：受影响部分复查

| 模块／检查 | 状态 | 实际证据或适用的既有证据 | 未核实部分／影响范围 |
|---|---|---|---|
| 实验约定与超参数 | PASS | `config.json` 与 `src/config.py` 拒绝 search；`tests/test_train_graph_selection.py` 从实际 Adam 参数组读到 lr=1e-4、wd=0、两组 H_C/B 且 B 倍率 1.0；formal_start=1、早停计数从 41、patience=20 由合成选模状态验证。`run.ps1` 默认 `check-inputs`。 | 真实服务器更新次数、早停轮次与 GPU 数值待正式运行。 |
| 数据与输入边界 | PASS / WARN | 包内 z-score `fit_split=train` 且 `fit_patients` 不含 XZY；30 通路名称与标准化参数顺序一致。合成不存在的原始基线路径下，9 个包内 accepted 指标仍可读取；缺失/非有限指标会失败。共同身份在提取路径按 `patient_id\|source_group\|spot_id` 写入缓存。 | 本机无服务器图像/标签；`slide_geometry.csv` 的 `patch_coverage_size` 为空；来源组不是物理切片证明。 |
| 学习行为 | PASS | 合成空间前向反向：H/C/B 均有有限非零梯度；B 无 bias 且零初始化；exact GELU；优化器不含编码器参数。同维同种子 H-optimus-0/1↔UNI2-h、Phikon-v2↔UNI 的完整初始 state dict 逐张量相等。 | 未加载真实大编码器；不能证明服务器完整模型已验证。 |
| 选模与结果交付 | PASS | 合成历史：1–40 轮早停计数保持 0，第 41 轮开始计数；同分容差内改用更低 z-MSE。缺 H-optimus-1 时，特征准备、训练及外部评估均继续到 Phikon-v2并逐项登记失败；训练/特征/外评 `partial` 返回 2，本地分析 `partial` 返回 0 且保持不完整标识。分析同时列出 `patient_macro_pathway_pcc` 与 `pooled_pcc`。 | 无真实运行结果，两类 PCC 尚无可报告数值。 |
| 输入与缓存补充 | PASS / WARN | H-optimus-0 固定 revision 的合同已改为 `pytorch_model.bin`，下载清单、快照验证与加载器一致。边缘标记 256 方图经实际 `FrozenTransform` 后四边保留；非方图失败。H-optimus 与 ImageNet 归一化张量不相等。适配器拒绝 H-optimus 三维输出和缺失 `last_hidden_state`。缓存缺 COMPLETE / 身份 / revision 不得复用；拒绝 mean_patch。缓存路径含 `hoptimus_rgb_v1`，并登记 `project_mpp_status=unverified`。`src` 无 `AutoImageProcessor`。 | 三模型 snapshot 尚未下载；官方 0.5 µm/px 与项目 patch 尺度关系未核实。 |
| 空间补充 | PASS / WARN | 小图只连同 slide、同 split；边权余弦随当前特征变化；孤立点 degree=0 且前向退化为中心 C。图参数由冻结配置校验。 | 物理 patch 覆盖未知；不得将来源组当作真实切片条码。 |
| 历史与恢复补充 | PASS | `train.py` 拒绝 resume；任务失败另记 attempt，不覆盖已有运行目录。基线只读包内 accepted 指标，原始绝对路径仅作来源追溯，不重训。 | 不提供优化器/随机状态完整续训。 |
| 入口与包内测试 | PASS | `python -B -m pytest tests -q --tb=short`：40 passed；`python -B -m compileall -q src runner.py` 通过；真实 `run.ps1` 默认入口与 `-Models hoptimus0,phikonv2` 参数入口均退出 0。401/403 下载失败不阻止其余模型尝试。 | 下载、真实编码器加载、特征提取、正式训练按授权未执行。 |

## 关键参数（只列本次相关项）

| 参数 | 设定值 | 生效值或待验证 | 作用位置 | 设置依据 |
|---|---|---|---|---|
| 几何协议 | `full_fov_224_bicubic_v1` | 合成四边标记与非方图拒绝 PASS | `src/transforms.py` | 与已接纳全视野基线对齐 |
| 图像归一化 | H-optimus `hoptimus_rgb_v1`；其余 `imagenet_rgb_v1` | 同图两套 mean/std 得到不同张量 | `src/transforms.py`、`inputs/model_manifest.json` | 各编码器预训练接口 |
| 输出模式 | H-optimus 二维 embedding；Phikon CLS `[:,0,:]` | 无权重适配器测试 PASS；真实形状待服务器严格加载 | `src/model_adapters.py` | 官方前向合同 |
| 优化器 | Adam，lr 1e-4，wd 0，B 倍率 1.0，constant | 实际 param_groups 与设定一致 | `src/train.py` | 冻结 spatial-11 |
| 容量／正则 | hidden 256，dropout 0.3，exact GELU | 构建对象与梯度测试 PASS | `src/model.py` | 冻结 spatial-11 |
| 图 | r=2.0，k=12，σ=0.5，T=0.7716396069760084，self=0.5 | 小图构图 PASS；边权用当前特征 | `src/graph.py` | 冻结 spatial-11 |
| 选模窗口 | formal 1；计数 41；patience 20 | 合成 EarlyStopState PASS | `src/selection.py` | 冻结 spatial-11 |
| 头部宽度 | 随 d 变化：1536→408862；1024→277790 | `count_parameters` 与公式一致 | `src/config.py`、`src/model.py` | 保留原生维数，无公共投影 |

## 结论

- 已证实偏差及处理结果：四项审计偏差均已修复并完成定向回归；详情见 `修复日志_20260919_v002.md`。本批提取入口拒绝 `timm_tokens_cls`，该模式仅保留在适配器测试中。
- 条件性风险／未核实：H-optimus 官方 0.5 µm/px 与当前 MPP2 patch 尺度；服务器权重、transformers 兼容版本、GPU 显存；H-optimus-1 人工审批与更严许可；来源组≠物理切片；外部仅 XZY 单患者。
- 当前可支持的结论：v002 独立代码包已通过本地零训练复查，默认入口不会训练；完整三模型运行和定向部分运行的状态语义均已落地。不可作出的声明：三个新编码器已经可比、已经优于基线、服务器模型已验证、或任何 Registry 已确认结果。下一步由用户按 README 与下载操作卡在服务器显式启动。

## 2026-09-27：v003 离线登记定向复查

- **已证实偏差**：服务器首次执行 `download_models.ps1 -RegisterOnly` 时，H-optimus-0 报 H-optimus-1 缺少 `snapshot_path`，H-optimus-1 报 Phikon-v2 缺少 `snapshot_path`。实际原因是单模型验证调用 `load_model_specs(require_local_files=True)`，提前校验尚未登记的其他模型。三个路径已由首次调用逐项登记，但严格加载状态均未通过。
- **修复与范围**：`src/download_models.py` 改为解析全清单固定合同、只在 `load_local_encoder` 中严格校验当前模型快照。未改模型结构、输入变换、图、训练或选模。Phikon-v2 的 `transformers` 缺失是独立的服务器依赖问题，代码修复不代替安装。
- **定向证据**：新增从三个空 `snapshot_path` 逐项登记的合成回归测试；修复前该测试返回 `partial`，修复后返回 `complete`。相关测试 `test_dispatch_baseline_download.py` 与 `test_adapters.py` 共 19 项通过，未加载真实权重或运行训练。
- **模块边界**：实验约定与超参数、学习行为、选模和结果交付未变，沿用上表证据；本次仅复查输入快照登记与加载入口。服务器三个真实模型的严格加载及 Phikon-v2 依赖兼容性仍为 `UNVERIFIED`，须重跑 `-RegisterOnly` 后判定。
- **环境补救**：本地下载了供 Windows Python 3.13 使用的 `transformers==4.57.1` 离线 wheel 集合并完成离线解析预演；在隔离环境中离线加载真实 Phikon-v2 权重，以一张合成图得到 `[1,1024]` 且数值有限。未在服务器安装，H-optimus-0/1 真实加载及服务器 GPU 兼容性仍待 `-RegisterOnly` 验证。该版本要求 `huggingface_hub<1`。

## 2026-09-27：v004 H-optimus 结构定向复查

- **已证实偏差**：服务器 H-optimus-0 严格加载报告 `reg_token` 多余及 1536/1408 维不匹配。包内两款 H-optimus 清单均写 `vit_giant_patch14_224`，而下载快照自带 `config.json` 均写 `vit_giant_patch14_reg4_dinov2`；后者默认图像网格大于 224，需显式传入 `img_size=224`。
- **修复与范围**：两款 H-optimus 清单改为快照实际 architecture；本地构造器核对清单与快照一致并设置 224 输入。未改变编码器权重、特征维数、图像变换、训练头、超参数或模型选择。
- **定向证据**：以本地 timm 1.0.26 在 meta 设备构造新结构，分别比对 H-optimus-0 的 `.bin` 与 H-optimus-1 的 safetensors：两者各 567 个参数，名称和形状均完全一致；相关定向测试 20 项通过。未在服务器实际加载/前向，服务器 timm 1.0.27 与 CUDA 结果仍待 `-RegisterOnly` 验证。用户终端粘贴仅显示 H-optimus-0 的首段错误，不能据此推断其余模型的最新状态。

## 2026-09-27：v005 Phikon-v2 离线依赖与严格加载定向复查

- **服务器已证实状态**：三款模型的配置和权重文件检查均为 `True`。H-optimus-0/1 在服务器 Python 3.13.5、torch 2.6.0+cu124、timm 1.0.27 下严格加载和合成图前向通过，均为 `[1,1536]` 有限 float32。Phikon-v2 唯一报告失败是 `transformers: null` 与 `ImportError`，尚未进入权重加载；整批状态为 `partial`。
- **根因与交付缺口**：v004 代码包之外另存有离线依赖 ZIP，而 `-RegisterOnly` 按设计不安装依赖；用户仅复制代码包并登记必然仍缺 `transformers`。v005 将 ZIP 纳入包内，新增显式 `install_phikon_offline.ps1`，只对 `config.json` 指定的解释器离线安装。安装需由服务器用户执行；不会由登记或训练启动器隐式触发。
- **严格性修复**：此前 Phikon-v2 的 `AutoModel.from_pretrained` 默认未返回权重加载差异，前向成功可能误标为 `strict_load: passed`。v005 要求 `output_loading_info=True` 且 `missing_keys`、`unexpected_keys`、`mismatched_keys`、`error_msgs` 全为空；固定使用本地 safetensors，任一差异即失败。H-optimus 的 `strict=True` 不变。
- **定向证据与边界**：本地 Python 3.13 隔离环境执行包内离线安装脚本成功，`transformers 4.57.1`、`huggingface_hub 0.36.2` 可导入，torch/timm 版本保持原值；真实 Phikon-v2 权重加载的四项差异均为空，合成图前向得 `[1,1024]` 且有限。相关定向测试 21 项及包内全部 43 项测试通过。服务器安装、三模型在变更后的依赖环境中重登记仍为 `UNVERIFIED`。实验约定、划分、头部、超参数、训练和选模未改；正式特征提取前须服务器报告 `status: complete`、三项严格加载通过、失败列表为空。

## 2026-09-27：v005 服务器登记回报

- 用户回传的服务器输出显示离线依赖安装与导入成功：Python 3.13.5、torch 2.6.0+cu124、timm 1.0.27、`transformers 4.57.1`、`huggingface_hub 0.36.2`。
- 同一回报中的三模型登记报告为 `status: complete`、`failures: []`，三个 `strict_load: passed`；H-optimus-0/1 输出 `[1,1536]`，Phikon-v2 输出 `[1,1024]`，均为 float32。代码入口在 `complete` 时返回退出码 0；报告中的 `note` 是固定提示，不能解读为仍有失败。
- **范围判断**：模型文件、依赖导入、严格加载及合成图前向已据服务器回报通过；尚未看到 `check-environment`、`check-inputs`、正式特征提取或训练的输出。下一步先完成环境与输入检查，不把登记通过等同于实验完成。

## 2026-09-27：v006 训练路径修复

- **服务器证据**：用户回传 `prepare-features` 为 `cache_count: 3`、`overall_cache_status: complete`；首次 `train-spatial` 为 `completed: 0`、`failed: 9`。首项任务记录在 `train.py:214` 写 `optimizer_effective.json` 时出现 `FileNotFoundError`。本地核对相应服务器输出路径长 265 字符；`train.py` 已在写入前创建 `raw` 目录，因此原因是 Windows 长路径边界，并非目录创建遗漏。
- **修改范围**：`stage_runtime.py` 为每个模型／种子／尝试使用短物理目录名，如 `hoptimus0_s45_a02`；完整逻辑 `task_id` 和 `attempt_id` 继续写入批次任务记录。权重与原始输出仍分目录，已失败的 attempt01 不覆盖。模型、特征、图、优化器、超参数、选模及外部评估算法未改。
- **定向复验**：模拟真实任务入口写 `raw/optimizer_effective.json` 成功，完整任务编号仍保留；按用户服务器根目录计算九个新尝试目标路径互异且最长短于 240 字符。`test_partial_execution.py` 共 3 项通过。真实服务器九项训练是否完成仍待 v006 部署后回报，不能根据本地测试宣称训练成功。
- **四模块范围**：实验约定与超参数、数据／输入边界、学习行为均未变，沿用前述证据；本次只复查选模与结果交付中的尝试目录和失败记录。服务器旧批次可保留并重试，重登模型快照后复用已有特征缓存。

## 2026-09-27：v006 服务器训练回报

- 用户回传同一批次 `20260927_154318_475_2242f53b` 的训练重试报告：`status: completed`、`planned: 9`、`completed: 9`、`failed: 0`、`failed_task_ids: []`、`complete_comparison: true`。这确认服务器训练动作的九任务覆盖；首次 v005 失败记录应继续保留，不把两次尝试混成两套成功种子。
- 外部 XZY 推理、本地双 PCC 分析及 Registry 结果接纳尚未完成。本回报不能支持新编码器优于基线的科研结论；下一步同批次执行 `external-eval`，再回传运行目录。

## 2026-09-28：v006 外部预测与结果接纳补记

- Registry 条目 `phase2_backbone_spatial_ablation_batch2_20260919` 记录 `status: done`、`evidence_status: accepted`、`code_version: v006`，接纳时间为 `2026-09-28T00:01:47+08:00`。接受范围为三个编码器各 45/46/47 种子的 9 项成功训练、9 份内部验证预测及 9 份 XZY 外部预测；首次失败的 `attempt01` 保留，正式结果使用 `attempt02`。
- 回传 `external-eval` 动作报告为 `status: completed`、`formal_prediction_count: 9`、`failed: 0`、`complete_comparison: true`；`analysis/local_review.json` 记录了 9 项新训练、9 份新内部预测、9 份新外部预测及 3 组特征缓存，并标记复核裁决 `accepted`。
- 接纳结论限共享冻结 `spatial-11` 配方与当前评价集合。H-optimus-1 的 XZY 双 PCC 均值在六个空间臂中最高；H-optimus-0 对 UNI2-h 指标方向不一；Phikon-v2 内部略高于 UNI、外部较低。XZY 为单患者，只作描述性观察，不改变预先指定的 UNI2-h 下游模型。
- 服务器模型严格加载、训练及外部预测的状态来自用户回传报告；本地 Registry 和回传产物可复核登记与指标，本次补记不声称实时核验服务器权重或特征缓存实体。
