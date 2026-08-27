# 图表计划

> 所有图表默认 `pending_user_review`。图中必须同时显示证据状态和统计单位，避免将 spot、slide、fold 或 seed 误当独立患者。

## 主图

### Figure 1 — 双阶段研究框架与证据边界

- 左：Phase 2 配对 H&E/空间转录组 → 30 维通路预测与 W007 四臂。
- 右：Phase 3 H&E/pCR-MPR → W004 四种通路表示与 MIL 分类。
- W007→W004 只画虚线，标 `future integration; not data-bound in current evidence`。
- 下方画证据层级：W007 accepted；W004 accepted/COMPATIBILITY_ONLY；新复算 pending review。
- 当前草案：`figures/figure1_evidence_bridge.svg`。

### Figure 2 — W007 统一架构与四臂归因

- FBR：冻结 B0；RCC：图像残差；HCR：硬配对 InfoNCE；CPGCR：连续通路几何软目标。
- 标出训练期真实通路支路和部署期 image-only 推理。
- 图注写明 single seed=42，CPGCR 为方法候选而非性能赢家。

### Figure 3 — Phase 2 多指标与一致性诊断

- internal/external 分面，四臂相对 FBR 的 z-MSE、PCC、CCC、z-RMSE、raw R²、Spearman。
- 30 通路配对散点/森林图：PCC−CCC 与 PCC²−R²。
- MSE 分解为均值、尺度和相关结构误差。
- 外部病例 E1 只有一名患者，不绘制“外部患者总体”置信区间。
- 当前草案：`figures/figure3_w007_patient_balanced_metrics.png`；完整 30 通路表见 `reanalysis_data_index.md`。

### Figure 4 — W004 四臂桥接结果

- pCR/MPR 分面，20 个 seed-fold 的配对 AUC：A002−A001、A003−A004。
- 辅助面板：per-fit AUPRC/Brier；pooled repeated-row ROC/PR 仅作描述。
- 校准面板若可算，标题必须写 `repeated slide-level holdout; patient-nonindependent`。
- case_id 聚合结果标 `post hoc`，不得写 OOF。
- 当前草案：`figures/figure4a_w004_fixed_auc_contrasts.png` 与 `figures/figure4b_w004_test_diagnostics.png`。

### Figure 5 — 文献新颖性与可信度矩阵

- 横轴：直接回归↔参考库检索；纵轴：基因↔通路↔临床结局。
- 颜色：patient-held-out/slide-held-out/跨队列。
- 突出 PEaRL、DeepPathway、HE-to-SGE benchmark、TANGLE、PathGen、ENLIGHT-DeepPT。

## 主表

| 表 | 内容 | 状态 |
|---|---|---|
| Table 1 | W007 四臂 internal/external accepted 指标，赢家按指标分别加粗 | accepted，single seed |
| Table 2 | W004 四臂 mean per-fit AUC 与固定四个差值 | accepted，COMPATIBILITY_ONLY |
| Table 3 | 新颖性矩阵：已有工作、项目差异、证据强度、允许创新点 | external context |
| Table 4 | 方法—代码—证据映射与超参数 | implementation evidence |
| Table 5 | TRIPOD+AI/PROBAST+AI 缺口清单 | pending review |

## 补充材料

- Supplementary Table S1：W007 30 通路 PCC/CCC/z-RMSE/raw R²/Spearman 全值。
- Supplementary Table S2：W007 MSE 分解和四臂配对差值。
- Supplementary Table S3：W004 2 endpoints×4 arms×20 fits 的 AUC/AUPRC/Brier。
- Supplementary Table S4：W004 case_id post hoc 聚合。
- Supplementary Table S5：44 篇官方来源的完整新颖性矩阵。
- Supplementary Figure S1：W007 truth/prediction 均值、标准差与振幅压缩。
- Supplementary Figure S2：W004 描述性 ROC/PR/校准。

## 出图规则

1. 不使用星号或显著性标签，除非预先定义患者级推断并完成多重比较控制。
2. pCR 与 MPR 分开显示；MPR 标 secondary/exploratory。
3. AUC 用三位小数，固定差值正文用六位小数以对应治理记录。
4. 每个 caption 均写统计单位、数据划分、证据状态与不可外推边界。
5. 颜色与线型不暗示未被证据支持的优胜排序。
