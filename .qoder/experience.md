# PFMval 经验索引

> 🔒 **必读文件**：本文件在每次会话开始时必须读取。
> 内容仅保留指向性说明，详细经验请按需阅读对应skill文件。

---

## 📂 Skill文件索引

| 文件 | 适用场景 | 路径 |
|------|---------|------|
| 数据处理经验 | 数据加载、路径配置、CSV格式 | `.qoder/skills/数据处理经验.md` |
| 训练工程经验 | 模型训练、调参、环境配置 | `.qoder/skills/训练工程经验.md` |
| 可视化与输出规范 | 结果可视化、报告生成 | `.qoder/skills/可视化与输出规范.md` |
| 组会汇报撰写指南 | 撰写组会汇报文档 | `.qoder/skills/组会汇报撰写指南.md` |

---

## ⚠️ 关键提醒（每次必看）

1. **EGN-v1已淘汰** — 所有对比分析排除EGN-v1
2. **受保护目录禁改** — `histogene/`、`egnv1/`、`egnv2/` 只能新增独立文件适配
3. **路径typo** — `patch_noov_spilt` 是原始拼写，不修正
4. **predictions.csv** — 列名必须为 `true_xxx`/`pred_xxx` 格式
5. **val_loss选模型** — 最佳epoch以val_loss最小为准，非val_pcc最大

---

## 📊 当前最佳性能基线

| 模型 | 单患者Val PCC | 跨患者Test PCC |
|------|-------------|---------------|
| HisToGene-UNI Token（方案B） | 0.5336 | 0.4095 (Fold1) |
| HisToGene-UNI Token + GAT | — | 0.4068 (Fold1, 不显著) |
| EGN-v2+UNI | — | 0.1950 |
| 方案B 三折平均 | — | 0.3812 |

---

## 🔬 下一步方向

P0-2 多模态融合（GenePT通路嵌入 + 交叉注意力）为最高优先级。

---

## OmiCLIP (Loki) 部署经验（2026-05-19）

### 架构探查结论
- 模型：coca_ViT-L-14（CoCa架构），通过 open_clip 2.26.1 加载
- 加载方式：`model = open_clip.create_model('coca_ViT-L-14'); model.load_state_dict(ckpt['state_dict'])`
- checkpoint key 是 `state_dict`（非 `model`）
- `loki.predex.OmiCLIP_Predictor` 在 loki==0.0.1 中不存在
- vision encoder：`model.visual`（307M参数）
- token输出：`model.visual(x)[1]` → [B, 255, 768]
- CLS输出：`model.encode_image(x)` → [B, 768]
- 预处理：224×224，OPENAI_DATASET_MEAN/STD

### 关键坑点
1. PyTorch 2.0.1 无法处理中文路径写文件 → 用 io.BytesIO + Python open() 绕过
2. total_mem属性 → 应为 total_memory
3. HuggingFace Python库国内卡住 → curl.exe + hf-mirror.com
4. dataset_uni_tokens.py 有1536维硬编码断言 → 自定义OmiCLIPTokensDataset

### 文件位置
- 提取脚本：extract_omiclip_features.py（loki_env运行）
- 训练脚本：train_histogene_omiclip.py（Python313运行）
- 权重：pretrained_omiclip/checkpoint.pt（7.14GB）
- 缓存：omiclip_cache/{patient}/{train|val}/*.pt
