# Phase 2 / Phase 3 本地探索性复算待审报告

- analysis_id: `phase2_phase3_reanalysis_v002`
- analysis_class: `exploratory_reanalysis`
- evidence_status: `pending_user_review`
- Registry: 未读取、未修改、未晋级。
- 输入边界：仅现有 W007 8 张 prediction CSV 与 W004 四个预测包的 320 张 prediction CSV。

## W007 pathway-wise 汇总

口径：每个 patient×pathway 先独立计算指标，再对 patient 做等权算术平均；internal_val 为 6 人等权，XZY 为 1 人。下表再对 30 个 pathway 做等权汇总；患者平衡的逐 pathway 数值见 `w007_pathway_metrics.csv`，逐患者明细见 `w007_patient_pathway_metrics.csv`（共 `840` 行）。

| arm | split | pcc | ccc | z_rmse | raw_r2 | spearman | pcc_minus_ccc | pcc_squared_minus_raw_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CPGCR | XZY | 0.5190 | 0.3993 | 0.7657 | -0.0801 | 0.4279 | 0.1197 | 0.3768 |
| CPGCR | internal_val | 0.6503 | 0.6103 | 0.5945 | 0.1990 | 0.6083 | 0.0400 | 0.2557 |
| FBR | XZY | 0.5194 | 0.3989 | 0.7672 | -0.0880 | 0.4262 | 0.1205 | 0.3858 |
| FBR | internal_val | 0.6485 | 0.6078 | 0.5960 | 0.1427 | 0.6075 | 0.0406 | 0.3108 |
| HCR | XZY | 0.5194 | 0.3990 | 0.7659 | -0.0827 | 0.4270 | 0.1204 | 0.3805 |
| HCR | internal_val | 0.6487 | 0.6087 | 0.5951 | 0.1702 | 0.6075 | 0.0400 | 0.2833 |
| RCC | XZY | 0.5165 | 0.3978 | 0.7675 | -0.0876 | 0.4265 | 0.1186 | 0.3816 |
| RCC | internal_val | 0.6504 | 0.6112 | 0.5941 | 0.2058 | 0.6088 | 0.0392 | 0.2491 |

W007 空间输出：Moran's I 可计算 `840` 行；局部象限热点为无显著性检验的描述性计数，共 `840` 行。

## W004 pooled 分类指标

| arm | task | split | n | auc | auprc | brier | calibration_intercept | calibration_slope | calibration_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ABS | MPR | test | 620 | 0.8714 | 0.8843 | 0.1456 | -0.0170 | 1.2407 | computed |
| ABS | MPR | validation | 620 | 0.8449 | 0.8615 | 0.1577 | -0.0342 | 1.1050 | computed |
| ABS | pCR | test | 620 | 0.8506 | 0.8132 | 0.1458 | 0.1666 | 1.5097 | computed |
| ABS | pCR | validation | 620 | 0.8497 | 0.8189 | 0.1452 | 0.2115 | 1.5812 | computed |
| ORD | MPR | test | 620 | 0.8746 | 0.8883 | 0.1439 | 0.0093 | 1.2475 | computed |
| ORD | MPR | validation | 620 | 0.8526 | 0.8677 | 0.1544 | -0.0128 | 1.1362 | computed |
| ORD | pCR | test | 620 | 0.8659 | 0.8236 | 0.1361 | 0.1032 | 1.3517 | computed |
| ORD | pCR | validation | 620 | 0.8661 | 0.8418 | 0.1340 | 0.1584 | 1.4225 | computed |
| SPATIAL_IDENTITY | MPR | test | 620 | 0.8722 | 0.8856 | 0.1451 | 0.0263 | 1.2490 | computed |
| SPATIAL_IDENTITY | MPR | validation | 620 | 0.8452 | 0.8602 | 0.1578 | 0.0212 | 1.1097 | computed |
| SPATIAL_IDENTITY | pCR | test | 620 | 0.8530 | 0.8162 | 0.1450 | 0.1572 | 1.5199 | computed |
| SPATIAL_IDENTITY | pCR | validation | 620 | 0.8510 | 0.8204 | 0.1444 | 0.2030 | 1.5844 | computed |
| SPATIAL_SHUFFLE | MPR | test | 620 | 0.8724 | 0.8855 | 0.1451 | 0.0175 | 1.2506 | computed |
| SPATIAL_SHUFFLE | MPR | validation | 620 | 0.8456 | 0.8600 | 0.1577 | 0.0179 | 1.1111 | computed |
| SPATIAL_SHUFFLE | pCR | test | 620 | 0.8491 | 0.8112 | 0.1464 | 0.1551 | 1.5115 | computed |
| SPATIAL_SHUFFLE | pCR | validation | 620 | 0.8495 | 0.8183 | 0.1452 | 0.2021 | 1.5901 | computed |

## W004 固定 AUC 差值

| contrast | task | n_paired_fits | mean_per_fit_auc_difference | pooled_auc_difference |
| --- | --- | --- | --- | --- |
| ORD_MINUS_ABS | MPR | 20 | 0.0040 | 0.0033 |
| ORD_MINUS_ABS | pCR | 20 | 0.0232 | 0.0153 |
| SPATIAL_IDENTITY_MINUS_SPATIAL_SHUFFLE | MPR | 20 | -0.0006 | -0.0002 |
| SPATIAL_IDENTITY_MINUS_SPATIAL_SHUFFLE | pCR | 20 | 0.0030 | 0.0039 |

per-fit 共 `320` 行；case_id post hoc 描述共 `620` 行。case_id 结果仅用于事后描述，不作为预注册推断。

## 不可计算项

- W004 Moran's I / 热点：`not_computable`。
- 原因：`prediction_csv_has_no_coordinate_columns`。
- W007 热点象限仅为 `computed_descriptive`，不包含置换检验或多重比较校正。

## 审核边界

所有输出均为 `exploratory_reanalysis` / `pending_user_review`；不得直接写入论文结论或晋级为 accepted evidence。
