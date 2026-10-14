# 冻结实验合同与实现映射

本文件把部署方案 v1.1 转换为代码可检查的固定合同。若共同预处理、标签、训练或选模口径变化，应创建新协议版本和完整匹配组，不能把变更后的种子与本协议结果混表。

## 数据与身份

- 划分：`MPP2/group_2`；标签：`barcode-repair-v003`；通路顺序由包内 `zscore_manifest.json` 固定为 30 列。
- 计数：train 9,472、internal_val 1,078、XZY external_test 1,039，合计 11,589。
- 身份键：`patient_id|source_group|spot_id`。`source_group=MPP2_<患者>_source01` 是来源组标识，不宣称为已验证物理切片编号。
- 共同身份清单按身份解析图像路径，并读取每个 PNG 的实际像素宽高。缺图、重复、坐标不一致或各模型身份顺序不同均停止。
- 服务器训练标签加载器在任何文件 I/O 前拒绝 external_test 行。

## 模型接口

| 模型 | 角色 | 固定版本 | 完整 token | 主 CLS | mean-patch |
|---|---|---|---|---|---|
| UNI2-h | 历史匹配参照 | 历史版本未完整登记 | 仅完整 `[B,265,1536]` 才可识别 | `[:,0,:]`，1536 维 | 仅完整 token 的 `[:,9:,:]`；缺失不重提取 |
| UNI | 新候选 | `b55a5ec6cade1a39edfe6534189a9b8ca7a022f0` | `[B,197,1024]` | `[:,0,:]`，1024 维 | `[:,1:,:]` 的 196 token 均值 |
| Virchow2 | 新候选 | `3158645804b69e3f3bc4439d4116edddf0840a72` | `[B,261,1280]` | `[:,0,:]`，1280 维 | `[:,5:,:]` 的 256 token 均值；排除 4 个 register token |

UNI 按官方 ViT-L/16 参数在本地构造；Virchow2 从同一 snapshot 的 `config.json` 构造，并显式使用 `SwiGLUPacked` 和 `SiLU`。权重只从登记的本地文件加载，`strict=True`；禁止 `hf-hub:` 在线构造、`pretrained=True` 联网回退、`strict=False` 或随机权重继续。

官方依据：[UNI 模型卡](https://huggingface.co/MahmoodLab/UNI)、[Virchow2 模型卡](https://huggingface.co/paige-ai/Virchow2)。两个模型均为 gated model，且模型卡登记为 CC-BY-NC-ND-4.0；使用者需自行确认访问资格和研究用途符合条款。

## 共同预处理

所有新候选调用同一个函数：

```text
Pillow convert("RGB")
→ 短边 Resize(256, BICUBIC, antialias=True)
→ CenterCrop(224)
→ float tensor 0..1
→ ImageNet mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)
```

无随机增强，不使用各模型默认 transform。编码器 `eval()`、全部参数冻结；特征提取与训练均为 FP32，AMP=false，TF32=false。

历史 UNI2-h 的完整预处理与权重来源证据仍不齐全，因此标为匹配历史参照，而不是新协议重跑结果。`inspect_existing_uni2h.ps1` 负责记录实际命中路径、形状和缺项，不因记录不足自动重提取。

## 回归头、初始化与随机流

三个模型的比较头结构相同，仅原生输入维数不同：

```text
Linear(d,256,bias=True)
→ GELU(approximate="none")
→ Dropout(p=0.3)
→ Linear(256,30,bias=True)
```

参数量：UNI2-h 401,182；UNI 270,110；Virchow2 335,646。不做 PCA、补零、截断、L2 或逐维特征标准化。

对同一 seed，新候选模型先按 `seed+0` 在 CPU 构造，再只复制历史 `initial_weights.pt` 的 `point_head.weight` 与 `point_head.bias` 到输出层。禁止复制训练后权重、relation head 或 spatial head。

- 点位顺序：NumPy `PCG64(seed+100000)`。
- Dropout：CPU `torch.Generator(seed+200000)`，逐批生成 `[B,256]` mask，再传到训练设备。
- 两个随机流相互独立，状态随检查点记录。

## 训练与选模

- AdamW：lr `3e-4`、weight decay `1e-4`、betas `(0.9,0.999)`、eps `1e-8`。
- batch 256，保留尾批；最多 60 epoch；无 scheduler；`num_workers=0`；CPU threads=8。
- loss 是全部点×30 通路平方误差之和除以 `batch_points×30`。
- epoch 1–5 只记录 warmup；epoch 6 起才有 formal checkpoint。
- formal 比较先看患者—通路等权 PCC；差值在 `1e-6` 内时选患者等权 z-MSE 更低者；仍相同保留较早 epoch。
- 常量预测的选择 PCC 记为 -1，但测量 PCC 保留 NA；惩罚不进入 MSE。
- epoch 6–15 建立早停参考；epoch 16 起计数；min delta `1e-4`、patience 10。
- 无 formal 端点时失败，不回退 warmup、last 或外部指标。

## 外部隔离与报告

`predict_external_arrays` 的公开接口没有 label、target、score 或 metric 参数；只接受同模型同 seed 的 `formal_best.pt`。XZY 输出包含 `pred_z`、一次逆变换后的 `pred_raw`、身份与检查点元数据，不包含 `target_z`。

本地分析分别报告：

- 患者—通路等权 PCC（含测量值、选择惩罚值和不可定义计数）；
- 全部点×通路整体展平 PCC；
- 合并点位后逐通路平均 PCC（单独命名）；
- z-MSE、z-RMSE、z-MAE、Spearman、CCC（一致性相关）、raw-MAE、raw-R² 与训练标准差归一化 RMSE；
- 最佳轮次、预测头参数量，以及可取得的训练/提取耗时和显存峰值；
- 逐患者、逐通路和患者—通路选择明细。

没有身份对齐的原始真值时，raw 指标只能使用 z 真值逆变换结果，并明确标成“裁剪 z 真值的逆变换”，不能冒充原始未裁剪 ssGSEA。若提供原始真值，报告同时给出它与 z 真值逆变换之间的平均和最大绝对差异，不静默覆盖。首批只有 seed 42 时不生成三种子标准差。

空间指标及 Top-k 通路重叠只有在窗口、有效 mask、k 值和计算规则均冻结后才计算；否则显式登记 `not_computed_not_predefined`。当前算术等权聚合不套用 Fisher-z（相关系数变换），并在报告中单独说明。三种子齐备后，内部验证和 XZY 分集合、先逐 seed 评价，再计算均值和样本标准差；不先平均预测。
