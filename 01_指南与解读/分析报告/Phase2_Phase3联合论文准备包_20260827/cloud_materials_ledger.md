# Google Drive 最新材料台账（只读）

> `internal_audit_only`：本文件保留私有 Drive 路由供用户审核，不进入公开投稿包；公开版只保留日期、材料类别与“设计建议、非项目结果”的证据边界。
> 检索日期：2026-08-27；来源文件夹：[AI空转云端方案管理](https://drive.google.com/drive/folders/1_vXNo5fr3uSIhVU3CFQ505DoX6z_71YO)。
> 治理边界：以下材料均作为 `design_recommendation`（设计建议）使用，不是项目实验结果，不写回 Google Drive。

| 材料 | 修改时间（UTC） | 本次采用内容 | 禁止外推 |
|---|---|---|---|
| [最终推荐 Phase 2 指标体系与后续补充实验方向建议](https://docs.google.com/document/d/1uj4k-4WLgqby13vYYY-ERsTjbrvljmmq8WTwycbK3wc/edit) | 2026-08-26 09:07:31 | PCC/CCC 共同主指标；z-RMSE、raw R² 护栏；Spearman；PCC−CCC、PCC²−R² 与 MSE 分解；LOPO 建议 | 文中约值不是本次新实验结果；LOPO 尚未执行 |
| [Phase 3 指标推荐和后续建议](https://docs.google.com/document/d/1GKOiohCvCewKD6sAN0eMLohqQmM3Aud8EOTZ-5C1I98/edit) | 2026-08-26 09:08:24 | 患者是统计单位；AUROC+AUPRC；Brier 与校准；患者级配对区间；容量/置乱负对照 | 不把当前 W004 切片级重复留出称为患者 OOF；不声称临床效用 |
| [推荐文献阅读](https://docs.google.com/document/d/1MbmQCFSgu9EGz0J_kS4b6S6j2JdzEZgDv0of5vX5o88/edit) | 2026-08-26 09:10:18 | 文献阅读顺序和任务映射；强调通路级与基因级不可直接排名 | 文档中的文献状态需以本次官方来源复核为准 |
| [原 Markdown 文件评价分析](https://docs.google.com/document/d/1xkDr7MHMdOnn1gsl5a2iNK9IVA2nXqBMBDvVWswTTYU/edit) | 2026-08-26 09:06:44 | 将既有数值、解释与未来建议分层；强调外部单患者与绝对校准限制 | 不把 `CONDITIONAL GO` 解释为达到投稿或临床部署标准 |

## 本次来源校正

- PEaRL 已正式发表于 WACV 2026，不再标为预印本。
- HEtoSGEBench 是 Nature Communications 2025 基准论文的代码仓库名，论文题名为 *Benchmarking the translational potential of spatial gene expression prediction from histology*。
- DeepPathway 已于 2026-08-26 以 Bioinformatics accepted manuscript 上线；尚未独立复现，且官方摘要不足以验证所有具体损失实现细节。
- HESCAPE 提供反例：跨模态对比预训练可能改善某些下游分类，却降低直接表达预测；因此不能写“对比学习必然改善回归”。

## 采用后的论文规则

1. Phase 2 不用单一 PCC 决定胜负。
2. Phase 3 同时报告判别、概率误差与校准，但当前 W004 仅能作受限描述。
3. W007 与 W004 在总图中并列，中间用虚线标注 `future integration`。
4. 患者独立 LOPO、新鲜外部队列、容量匹配负对照均进入 future work，不写成已完成。
