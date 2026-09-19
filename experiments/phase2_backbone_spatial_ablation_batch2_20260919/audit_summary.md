# 最小审查记录

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
