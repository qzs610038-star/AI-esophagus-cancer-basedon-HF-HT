# 独立反方审稿意见与修复矩阵

> 状态：`pending_user_review`。本文件用于主动暴露拒稿风险，不升级任何证据。

## 总评

若论文以“CPGCR 性能领先、同源 H&E 双分支产生融合协同、W004 已验证临床转化”为主线，当前证据足以支持拒稿。若定位为“受控方法探针、结果中性、受限 Phase 2→3 桥接，并预先规定下一步患者级验证”，可形成可信的大修稿基础。

| # | 最可能的拒稿理由 | 当前立即修复 | 必须留给 future work |
|---|---|---|---|
| 1 | 同源 H&E 双分支被误写为独立多模态 | 全文改称 H&E 衍生形态—通路双表征；Figure 1 画单一 H&E 根节点 | 独立实测组学验证或同源容量匹配实验 |
| 2 | W004 切片级拆分造成伪重复 | 显示 78 cases、患者重叠与 `COMPATIBILITY_ONLY`；不做切片独立显著性 | 患者分组 CV/LOPO/真正外部验证 |
| 3 | W004 被误写为 W007/CPGCR 下游验证 | W007→W004 只画 `future integration` 虚线 | 直接冻结并使用 W007 输出的配对 Phase 3 研究 |
| 4 | CPGCR 不是性能赢家 | 完整并列四臂指标；写“small and metric-dependent” | 多 seed 稳定性和预注册主指标 |
| 5 | W007 只有 seed 42 | 表题写 `n_seed=1`，删除稳定/显著措辞 | 至少三 seed 同协议复现 |
| 6 | 外部病例 E1 只有一名患者 | 改称 single-patient external case study；禁止 spot-level 人群 CI | 多患者、多中心外部队列 |
| 7 | PCC 较高但 raw R² 为负 | PCC 与 CCC、MAE/RMSE、R²、校准并列 | 外部重校准与患者级一致性验证 |
| 8 | 多指标、30 通路与四臂造成事后择优 | 明确 checkpoint 指标；完整展示方向；探索项不作显著主张 | 多重比较策略与预注册分析计划 |
| 9 | 训练期真值通路编码器易被误解为测试泄漏 | 画训练/推理双流程，说明部署时无真实 ssGSEA | 目标打乱、几何和梯度机制消融 |
| 10 | 临床转化、SOTA 与空间机制过度 | ordinal 仅称待验证信号；空间结果保留为阴性；删除 SOTA/临床效用 | 临床阈值、DCA、前瞻性和新鲜外部验证 |

## 即使数值好看也不得进入主张

- CPGCR 个别误差指标略优，不得写“最佳”或“显著优于”。
- 外部病例 E1 的 spot-level PCC、p 值或窄区间不得写跨患者外部泛化。
- W004 A002 pCR AUC 0.861818 不得写患者独立提升、融合协同或临床效益。
- W004 AUPRC/Brier/校准不得写成 W007 下游验证或 raw MPP2 结果。
- A003/A004 微小差值不得写空间信息有效。
- pooled PCC 不得写绝对准确、校准良好或临床可用。
- 同源 H&E 两支路不得写独立组学模态。
- ssGSEA 派生分数不得写成直接实测的通路活性或因果机制。
- attention、embedding 分离、训练损失或漂亮热图不得替代预测性能和患者级验证。
- 最佳 fold、seed、epoch 或事后挑选的通路数不得升级为正式主结论。

## 结论门

当前允许：`methodologically motivated candidate`、`small and metric-dependent differences`、`compatibility-only bridge`、`supports further patient-independent validation`。
当前禁止：`best-performing`、`significantly outperformed`、`validated multimodal synergy`、`patient-independent generalization`、`clinical utility`、`state of the art`。
