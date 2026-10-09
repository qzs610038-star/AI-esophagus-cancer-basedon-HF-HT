# Phase 2: Pathway-score reconstruction from full-field pathology images with spatial-residual modeling

> English draft v1 | 22 September 2026. This draft follows the original outline of Background–Results–Discussion–Methods for integration into the full manuscript. The embedded figures remain the first Chinese review version; the English section text and figure legends below are the English manuscript version. Items marked [To be completed] are author-editing placeholders. Hypotheses for incomplete experiments are isolated in the “Pending experiments writing area” and are not current findings. Phase 2 citation identifiers are provisional; see [proposed references](Phase2拟引用文献.md).

## Phase 2 paragraph for the abstract

We developed a spatial-residual model based on full-field H&E images and frozen pathology encoders to reconstruct location-level 30-dimensional pathway scores, and examined the effects of encoder choice, the spatial branch, and training density on reconstruction performance. The development data comprised 9,472 training locations and 1,078 internal-validation locations from 6 patients; evaluation was additionally performed at 1,039 locations from the previously evaluated external patient XZY. Across all three encoders, the complete spatial model achieved higher mean internal and external patient–pathway macro-averaged Pearson correlation coefficients (PCCs) than the matched model without the spatial branch. For the complete UNI2-h spatial model, the external patient–pathway macro-averaged PCC and pooled PCC were 0.5711 ± 0.0033 and 0.6837 ± 0.0015, respectively, reported as the mean ± sample standard deviation (SD) across three random seeds. On fixed dense evaluation sets, dense training yielded higher mean values for both PCC definitions than regular subsampling and random equal-count training. These results support the use of spatial context and dense training information for pathway-score reconstruction under the current data and training protocol; their transferability across patients and value for downstream treatment-response prediction require further validation.

## Background

Estimating spatial molecular phenotypes from routine H&E (hematoxylin and eosin) images provides a research route for linking tissue morphology to molecular measurements. ST-Net examined the feasibility of predicting spatial gene expression from tissue images in breast tumors, and Hist2ST further combined image, positional, and graph-structured information to model spatial expression. These studies provide methodological precedents for supervised learning from images to molecular measurements, but their results do not directly establish pathway-reconstruction capability in esophageal squamous cell carcinoma in the present study. [P2-R1](Phase2拟引用文献.md#p2-r1), [P2-R2](Phase2拟引用文献.md#p2-r2)

We used an existing set of 30-dimensional pathway scores as location-level prediction targets and sought image-derived pathway features that could be used in subsequent patient-level analyses. Rather than only comparing the absolute performance of different models, we addressed three related questions: whether replacing the pathology encoder changes reconstruction performance under consistent input and target definitions; whether adding a neighborhood spatial residual provides information beyond the central-location representation; and how performance changes on fixed dense evaluation sets after training locations are subsampled. The first comparison batch included UNI, UNI2-h, and Virchow2; their sources were the UNI paper, the official UNI2-h model documentation, and the Virchow2 preprint, respectively. [P2-R3](Phase2拟引用文献.md#p2-r3), [P2-R4](Phase2拟引用文献.md#p2-r4), [P2-R5](Phase2拟引用文献.md#p2-r5)

To address these questions, we fixed full-field image processing, pathway order, data partitioning, and evaluation definitions, and conducted a three-encoder, three-arm comparison and a separate training-density ablation. The comparison between direct pathway prediction and pathway reconstruction after gene prediction will be used to further examine the choice of prediction target; no results from that comparison can yet be included in the present conclusions.

## Results

### Data composition and pathway-reconstruction workflow

The current analysis used 10,550 locations from 6 development patients to construct the training and internal-validation sets, including 9,472 locations for training and 1,078 for internal validation; 1,039 locations from the previously evaluated patient XZY were additionally used for external evaluation (Table P2-1). Internal validation consisted of held-out locations from the development patients and therefore evaluates spatial holdout performance within the same development cohort, rather than performance in completely independent patients. XZY was evaluated after model freezing in this round and had been used for historical evaluation; it cannot be treated as a newly acquired blinded test cohort.

The complete square H&E image patch at each location was resized to 224 × 224, processed by a frozen encoder to extract image features, and then used by a central readout and spatial-residual branch to generate 30-dimensional pathway predictions. Pathway-normalization parameters were fitted only on the training set and carried forward to internal and external evaluation. Model selection used internal-validation information only (Figure P2-1).

**Table P2-1 | Phase 2 data composition and use.**

| Set | Patient scope | Number of locations | Use |
|---|---|---:|---|
| Training | 6 development patients | 9,472 | Model fitting and training-set pathway normalization |
| Internal validation | Held-out locations from the same 6 development patients | 1,078 | Recipe and checkpoint selection; internal evaluation |
| External XZY | 1 previously evaluated external patient | 1,039 | Evaluation after freezing |

![Figure P2-1 Data workflow and evaluation boundaries](/D:/AI空间转录病理研究/PFMval_new/团队项目进度与结论/方案共享/Phase2论文写作包_20260922/02_图表配置/首批图片_v1/exports/Fig_P2_1_data_flow.png)

**Figure P2-1 | Phase 2 data workflow and evaluation boundaries.** A, Six development patients contributed 9,472 training locations and 1,078 held-out internal-validation locations; external evaluation used 1,039 locations from the previously evaluated patient XZY. B, Complete square H&E image patches were resized to 224 × 224 by bicubic interpolation, processed by a frozen pathology encoder to extract features, and then used to predict 30-dimensional pathway z scores. C, Pathway-normalization parameters were fitted only on the training set, and recipe and checkpoint selection were based only on internal validation. D, The fixed pathway order and training-set normalization parameters form the downstream interface. Training and internal validation originate from the same development cohort, and external results are limited to the single patient XZY.

### Encoder comparison shows metric-dependent performance ordering

For the complete spatial arm, the internal patient–pathway macro-averaged PCCs for UNI, UNI2-h, and Virchow2 were 0.6618 ± 0.0010, 0.6770 ± 0.0004, and 0.6792 ± 0.0005, respectively; the corresponding pooled PCCs were 0.8012 ± 0.0004, 0.8105 ± 0.0002, and 0.8113 ± 0.0009 (Table P2-2A). In XZY, the patient–pathway macro-averaged PCCs were 0.5594 ± 0.0032, 0.5711 ± 0.0033, and 0.5778 ± 0.0110, respectively, whereas the pooled PCCs were 0.6568 ± 0.0001, 0.6837 ± 0.0015, and 0.6783 ± 0.0134 (Table P2-2B).

Accordingly, Virchow2 had the higher mean external patient–pathway macro-averaged PCC, whereas UNI2-h had the higher mean pooled PCC. The complete UNI2-h spatial arm also had a lower external pooled zMSE of 0.6330 ± 0.0069. Because the two PCCs use different aggregation schemes, these results indicate that relative encoder ordering depends on the statistical object being evaluated; a single metric cannot be used to summarize one encoder as comprehensively superior to the others. All comparisons were completed using recipes fixed in advance and do not represent a comparison of fully optimized performance after separate extensive tuning for each encoder.

**Table P2-2A | Three-encoder, three-arm comparison: internal validation.**

| Encoder | Experimental arm | Patient–pathway macro-averaged PCC | Pooled PCC | Pooled zMSE |
|---|---|---|---|---|
| UNI | Point (`point`) | 0.6525 ± 0.0008 | 0.7955 ± 0.0001 | 0.3783 ± 0.0006 |
| UNI | No spatial branch (`no_b`) | 0.6508 ± 0.0002 | 0.7952 ± 0.0005 | 0.3781 ± 0.0007 |
| UNI | Complete spatial (`spatial`) | 0.6618 ± 0.0010 | 0.8012 ± 0.0004 | 0.3686 ± 0.0003 |
| UNI2-h | Point (`point`) | 0.6660 ± 0.0005 | 0.8037 ± 0.0004 | 0.3645 ± 0.0007 |
| UNI2-h | No spatial branch (`no_b`) | 0.6649 ± 0.0004 | 0.8039 ± 0.0003 | 0.3637 ± 0.0005 |
| UNI2-h | Complete spatial (`spatial`) | 0.6770 ± 0.0004 | 0.8105 ± 0.0002 | 0.3528 ± 0.0004 |
| Virchow2 | Point (`point`) | 0.6687 ± 0.0009 | 0.8052 ± 0.0012 | 0.3625 ± 0.0032 |
| Virchow2 | No spatial branch (`no_b`) | 0.6681 ± 0.0005 | 0.8051 ± 0.0003 | 0.3627 ± 0.0006 |
| Virchow2 | Complete spatial (`spatial`) | 0.6792 ± 0.0005 | 0.8113 ± 0.0009 | 0.3518 ± 0.0016 |

**Table P2-2B | Three-encoder, three-arm comparison: external XZY.**

| Encoder | Experimental arm | Patient–pathway macro-averaged PCC | Pooled PCC | Pooled zMSE |
|---|---|---|---|---|
| UNI | Point (`point`) | 0.5147 ± 0.0108 | 0.6297 ± 0.0093 | 0.7347 ± 0.0164 |
| UNI | No spatial branch (`no_b`) | 0.5165 ± 0.0058 | 0.6371 ± 0.0052 | 0.7136 ± 0.0160 |
| UNI | Complete spatial (`spatial`) | 0.5594 ± 0.0032 | 0.6568 ± 0.0001 | 0.6808 ± 0.0137 |
| UNI2-h | Point (`point`) | 0.5362 ± 0.0043 | 0.6587 ± 0.0115 | 0.6793 ± 0.0101 |
| UNI2-h | No spatial branch (`no_b`) | 0.5367 ± 0.0032 | 0.6606 ± 0.0076 | 0.6769 ± 0.0029 |
| UNI2-h | Complete spatial (`spatial`) | 0.5711 ± 0.0033 | 0.6837 ± 0.0015 | 0.6330 ± 0.0069 |
| Virchow2 | Point (`point`) | 0.5459 ± 0.0103 | 0.6623 ± 0.0090 | 0.7092 ± 0.0143 |
| Virchow2 | No spatial branch (`no_b`) | 0.5442 ± 0.0124 | 0.6713 ± 0.0047 | 0.7005 ± 0.0159 |
| Virchow2 | Complete spatial (`spatial`) | 0.5778 ± 0.0110 | 0.6783 ± 0.0134 | 0.6834 ± 0.0343 |

All values in Table P2-2 are the mean ± sample SD for seeds 45/46/47. The patient–pathway macro-averaged PCC is the equally weighted average of correlations calculated within each patient and pathway; pooled PCC is calculated once after flattening z scores across all locations × pathways. Pooled zMSE is the mean squared error across all locations × pathways on the same z-score scale. The point arm uses the point recipe, whereas no_b and spatial use the same spatial recipe; all three experimental arms were trained from scratch.

### The spatial-residual branch increased mean correlation in the matched-recipe control

The complete spatial model adds a neighborhood residual B to the central prediction C, with final output C+B (Figure P2-2). To distinguish the spatial branch from the training recipe, we compared the complete spatial arm, spatial, with the no_b arm, which used the same spatial recipe but turned off B and retrained.

In the order UNI, UNI2-h, and Virchow2, the mean internal patient–pathway macro-averaged PCC gains of spatial over no_b were 0.0109, 0.0121, and 0.0111, and the pooled PCC gains were 0.0060, 0.0066, and 0.0062, respectively; the corresponding external gains were 0.0430, 0.0343, and 0.0335, and 0.0196, 0.0231, and 0.0070 (Figure P2-3E/F). The complete spatial arms for all three encoders also had lower mean internal and external pooled zMSE values than their respective no_b controls. The direction of these comparisons supports additional predictive information from the spatial branch in the current comparison, but variation across three seeds describes training randomness only and cannot establish statistically significant improvement in a patient population.

![Figure P2-2 Spatial-residual model](/D:/AI空间转录病理研究/PFMval_new/团队项目进度与结论/方案共享/Phase2论文写作包_20260922/02_图表配置/首批图片_v1/exports/Fig_P2_2_model_structure.png)

**Figure P2-2 | Full-field spatial-residual model and three-arm comparison.** A, Using UNI2-h as an example, the frozen encoder extracts a 1,536-dimensional CLS feature; the shared mapping H produces a 256-dimensional representation, and the central readout C generates a 30-dimensional base prediction. B, A same-slide directed one-hop neighborhood has a radius of 2 native grid steps and includes at most 12 neighbors; the spatial head B maps neighbor-minus-center H feature differences, weighted by distance and CLS similarity, to a 30-dimensional correction. C, The output is C+B; no spatial correction is made when no usable neighbor is available. D, point uses the point-07 recipe, whereas no_b and spatial use the same spatial-11 recipe and are trained from scratch; comparison of the latter two describes the increment associated with branch B. The neighborhood is a structural schematic only.

![Figure P2-3 Encoder and spatial-branch comparison](/D:/AI空间转录病理研究/PFMval_new/团队项目进度与结论/方案共享/Phase2论文写作包_20260922/02_图表配置/首批图片_v1/exports/Fig_P2_3_编码器与空间分支.png)

**Figure P2-3 | Pathway-reconstruction performance of three encoders and the spatial branch.** A/B, Patient–pathway macro-averaged PCC for internal validation and external XZY; C/D, the corresponding pooled PCC. E/F, spatial−no_b gains paired by the same seed, shown separately for the two PCC definitions. Open markers and error bars show the mean ± sample SD for seeds 45/46/47; pale points are individual seed values, and vertical offsets are used only to facilitate viewing. The SD of each gain was calculated from the three paired differences. Internal validation consists of 1,078 held-out locations from 6 development patients, and external evaluation consists of 1,039 locations from XZY. spatial and no_b use the same spatial recipe, whereas point uses a different recipe; the magnitudes of the two PCC definitions cannot mutually establish prediction superiority.

### Dense training achieved higher mean correlation on fixed dense evaluation sets

The training-density ablation compared dense training, regular 2×2 subsampling, and patient-wise random equal-count training, with 9,472, 2,383, and 2,383 training locations, respectively. All models were trained from scratch and evaluated on the same dense internal-validation set and XZY locations. In the primary analysis with 14,800 fixed updates, the dense arm had an internal patient–pathway macro-averaged PCC of 0.6770 ± 0.0004 and pooled PCC of 0.8105 ± 0.0002; the corresponding values were 0.6394 ± 0.0005 and 0.7895 ± 0.0004 for regular subsampling, and 0.6381 ± 0.0030 and 0.7886 ± 0.0013 for random equal-count sampling (Table P2-3).

In XZY, patient–pathway macro-averaged PCCs for dense, regular subsampling, and random equal-count sampling were 0.5711 ± 0.0033, 0.5554 ± 0.0049, and 0.5412 ± 0.0044, respectively; their pooled PCCs were 0.6837 ± 0.0015, 0.6715 ± 0.0030, and 0.6683 ± 0.0061. The dense arm also had lower mean zMSE and MAE on the original-score scale. These results show that, with the current model, fixed graph parameters, and computational budget, dense training achieved better mean predictive performance on both evaluation sets (Figure P2-4). Because subsampling simultaneously changes the number of training locations, spatial coverage, and graph connectivity, this result reflects their combined effects.

**Table P2-3 | Training-density ablation under equal numbers of updates.**

| Set | Training scheme | Number of training locations | Patient–pathway macro-averaged PCC | Pooled PCC | Pooled zMSE | MAE on original-score scale |
|---|---|---:|---|---|---|---|
| Internal validation | Dense | 9,472 | 0.6770 ± 0.0004 | 0.8105 ± 0.0002 | 0.3528 ± 0.0004 | 847.48 ± 1.07 |
| Internal validation | Regular 2×2 subsampling | 2,383 | 0.6394 ± 0.0005 | 0.7895 ± 0.0004 | 0.3878 ± 0.0005 | 905.10 ± 1.06 |
| Internal validation | Patient-wise random equal-count sampling | 2,383 | 0.6381 ± 0.0030 | 0.7886 ± 0.0013 | 0.3887 ± 0.0021 | 903.02 ± 2.04 |
| External XZY | Dense | 9,472 | 0.5711 ± 0.0033 | 0.6837 ± 0.0015 | 0.6330 ± 0.0069 | 1145.34 ± 18.41 |
| External XZY | Regular 2×2 subsampling | 2,383 | 0.5554 ± 0.0049 | 0.6715 ± 0.0030 | 0.6431 ± 0.0088 | 1163.57 ± 15.41 |
| External XZY | Patient-wise random equal-count sampling | 2,383 | 0.5412 ± 0.0044 | 0.6683 ± 0.0061 | 0.6573 ± 0.0320 | 1177.84 ± 46.42 |

Values are the mean ± sample SD for seeds 45/46/47. Internal and external location counts remained 1,078 and 1,039, respectively; error evaluation was returned to the z-score scale defined by the dense training set for all arms, whereas MAE was calculated on the original pathway-score scale. The dense arm and the main-experiment UNI2-h spatial arm retain their respective run sources and were not merged to increase the number of replicates.

![Figure P2-4 Training-density ablation](/D:/AI空间转录病理研究/PFMval_new/团队项目进度与结论/方案共享/Phase2论文写作包_20260922/02_图表配置/首批图片_v1/exports/Fig_P2_4_训练密度消融.png)

**Figure P2-4 | Training-density ablation on fixed dense evaluation sets.** A, Conceptual sampling schematic for dense training, regular 2×2 subsampling, and patient-wise random equal-count training; the dot grid does not correspond to coordinates from a real tissue section. B/C show internal and external patient–pathway macro-averaged PCC, and D/E show pooled PCC. Each arm was trained from scratch with seeds 45/46/47, 14,800 fixed updates, early stopping disabled, and internal validation every 296 updates, providing 50 internal selection opportunities. Internal and external evaluation sets remained dense. Markers and error bars show mean ± sample SD, pale points are individual seed values, and numerical labels retain three decimal places. Subsampling simultaneously changes the number of training locations, spatial coverage, and training-graph connectivity.

On the same training trajectories, when checkpoint selection was restricted to the first 50 complete training epochs for each arm, the dense arm still had higher mean values for both PCC definitions in internal and external evaluation than either sparse arm (Supplementary Figure P2-4S). This sensitivity result retained the direction of the primary comparison; it is an alternative checkpoint selection from the same training batch and cannot be treated as an independent replicate.

![Supplementary Figure P2-4S Equal-50-epoch sensitivity analysis](/D:/AI空间转录病理研究/PFMval_new/团队项目进度与结论/方案共享/Phase2论文写作包_20260922/02_图表配置/首批图片_v1/exports/Fig_P2_4S_等50轮敏感性分析.png)

**Supplementary Figure P2-4S | Equal-50-epoch sensitivity analysis for the training-density ablation.** The same nine training trajectories as in Figure P2-4 were used, with internal validation to select a checkpoint within the first 50 complete training epochs of each arm. A/B show patient–pathway macro-averaged PCC for internal validation and XZY, and C/D show pooled PCC. Points, error bars, and seed conventions are the same as in Figure P2-4. The underlying trajectories each had 14,800 updates; this figure does not add training batches or replicates.

## Discussion

The Phase 2 results indicate that, under fixed full-field input and 30-dimensional pathway targets, pathology-image representations can support location-level pathway-score reconstruction, and both the spatial-residual branch and training-sampling scheme are associated with predictive performance. The complete spatial arms of all three encoders had higher mean internal and external values for both PCC definitions than their matched-recipe no_b controls, and dense training outperformed both sparse schemes on both evaluation sets. These observations provide experimental support for the current model setting while limiting its applicability to this study’s data and comparison protocol.

The role of the spatial branch can be understood as supplementing the central prediction. The central readout uses only the image at the current location, whereas branch B receives neighbor-minus-center representation differences and can therefore incorporate local tissue context into the prediction. The joint modeling of image and spatial structure is consistent with the direction of prior spatial-expression prediction research, but this study did not directly validate specific cell-composition or molecular-mechanism explanations; the gain from the spatial branch should therefore remain a model-level observation. [P2-R2](Phase2拟引用文献.md#p2-r2) Because the point arm uses a different recipe from the spatial recipe, interpretation of branch B should rest on the paired spatial versus no_b comparison.

The encoder comparison shows that patient–pathway macro-averaged PCC and pooled PCC can yield different orderings. The former weights patients and pathways equally, whereas the latter evaluates the overall correlation structure of all standardized location–pathway entries. The different external ordering of Virchow2 and UNI2-h indicates that the two summaries do not answer exactly the same question. We therefore report both PCC definitions and error metrics in parallel rather than generalizing an advantage on one metric to all dimensions of prediction. The UNI2-h spatial-model seed 45 used for downstream delivery was specified before external evaluation; its use was not based on post hoc selection of the encoder or seed with the best external performance.

The density ablation supports retaining dense training information, but the observed difference cannot yet be attributed to the number of locations alone. The regular-subsampling and random-equal-count arms both reduce training samples and alter spatial coverage and training neighborhoods; a fixed graph radius also changes the available connections after subsampling. Although the equal-updates primary analysis and equal-50-epoch sensitivity analysis had consistent directions, they share the same data and training trajectories. Separately identifying the contributions of sample count, spatial coverage, and neighborhood connectivity will require specifically designed control experiments.

This study has several limitations. First, internal validation and training originate from the same development patients, and external evaluation includes only one previously evaluated patient; patient-to-patient or cross-center generalization uncertainty cannot be estimated from these data. Second, statistics across three seeds reflect training randomness only and are insufficient for significance inference in a patient population. Third, the fixed-recipe comparison did not optimize each encoder comprehensively; observed ordering is also influenced by recipe fit. The original pathway-scoring algorithm, gene-set membership, and physical image coverage remain to be completed in the source materials, and these details are relevant to biological interpretation of the target and the boundary of spatial evaluation. Finally, the gene-reconstruction comparison is incomplete, so this phase cannot conclude that direct pathway prediction is superior to the indirect gene route.

As an input interface for Phase 3, this phase delivers location-level 30-dimensional pathway z scores together with patient, slide, and location identities and training-set normalization parameters. Subsequent work will evaluate whether this representation provides information beyond pathology morphology for patient-level treatment-response prediction; until the corresponding results are available, conclusions from this phase concern pathway reconstruction only and do not address clinical-prediction benefit.

## Methods

### Study materials, identity linkage, and data partitioning

The study used spatial-transcriptomics locations matched to H&E images and their 30-dimensional pathway labels. [To be completed: study center, recruitment period, inclusion and exclusion criteria, number of slides per patient, ethics approval, and consent handling.] The development patients were HYZ15040, JFX, LMZ12939, TGC, XSL, and ZHZ, and the external patient was XZY. The current runs used the fixed MPP2/group_2 partition and the barcode-repair-v003 label version; “MPP2” is a project-scheme identifier and is not interpreted in this article as a physical micrometer/pixel value.

Image features, labels, and coordinates were linked through patient–slide–location identities. Training and internal-validation location identities were separated, but both sets came from the same development patients. Spatial graphs were constructed separately by slide within each data set, with no edges connecting locations from different splits or different slides. Graph construction used only coordinates and frozen image features, not pathway labels. The physical coverage of image patches, registration method, and physical distance corresponding to the partition buffer are [to be completed]; this draft therefore does not extend identity separation into a claim that image support regions have been proven completely non-overlapping.

### Pathway targets and normalization

The model regressed 30 pathway scores jointly, with the same pathway order used in all experiments. For pathway *k*, the raw score was converted to a z score using the training-set mean μ*k* and SD σ*k*, and the same parameters were applied to internal validation and external evaluation. The training target remained 30-dimensional throughout this phase; pathways were not selected on the basis of external performance, nor was the target changed to the top 20 pathways.

[To be completed: processing of the original expression matrix; gene identifiers and expression scale; gene-set database and version; members and background set for each pathway; algorithm, software version, parameters, and missing-gene handling for the original 30-dimensional pathway scores.] This draft does not describe ssGSEA or GSVA as a confirmed actual label-generation workflow; the relevant methodological references are listed only as conditional candidates.

### Full-field input and image encoders

All inputs were complete square image patches, converted to RGB and resized to 224 × 224 by bicubic interpolation without center cropping. Pixel values were converted to 0–1 floating-point values and normalized with ImageNet means and SDs; the current protocol used no random image augmentation. The scanning device, original pixel scale, patch size, and physical patch coverage are [to be completed]. UNI, UNI2-h, and Virchow2 encoders remained frozen and extracted CLS (image-summary token) features of 1,024, 1,536, and 1,280 dimensions, respectively, as inputs to the prediction head. This experiment used CLS features and did not concatenate mean-pooled image-token features with the CLS representation. Model identifiers, fixed versions, and input protocols were retained in the run records. [P2-R3](Phase2拟引用文献.md#p2-r3), [P2-R4](Phase2拟引用文献.md#p2-r4), [P2-R5](Phase2拟引用文献.md#p2-r5)

### Central prediction and spatial residual

For the frozen image feature *x*<sub>*i*</sub> at location *i*, a shared linear map followed by GELU (Gaussian error linear unit) activation produced the 256-dimensional representation *h*<sub>*i*</sub>, and the linear readout C generated a 30-dimensional central prediction. The spatial branch used representation differences between same-slide neighbors and the center to form a residual:

$$
h_i=\operatorname{GELU}(W_Hx_i+b_H),\qquad
u_i=\sum_{j\in\mathcal N(i)}a_{ij}(h_j-h_i),\qquad
\hat y_i=C(h_i)+B(u_i).
$$

The neighborhood radius was fixed at 2 native grid steps per slide, retaining at most the 12 nearest neighbors, and the graph was a directed one-hop graph. Let ρ<sub>*ij*</sub> be the distance between two locations divided by the native slide grid step, and let image similarity be the cosine similarity of frozen CLS features. The raw neighbor weight was:

$$
w_{ij}=\exp\left(-\frac{\rho_{ij}^2}{2\sigma^2}\right)
\exp\left(\frac{\cos(x_i,x_j)-1}{\tau}\right),\qquad
a_{ij}=\frac{w_{ij}}{w_{\mathrm{self}}+\sum_{j\in\mathcal N(i)}w_{ij}}.
$$

The frozen parameters were σ=0.5, τ=0.7716396069760084, and raw self-weight *w*<sub>self</sub>=0.5. B was a bias-free linear mapping with zero initialization; when no usable neighbor was available, *u*<sub>*i*</sub> was zero and prediction retained only the central output. During training, the central-readout input received recipe-specific dropout, while neighborhood differences used representations produced by the shared mapping; dropout was disabled during inference.

### Training recipes and the three-arm ablation

We first froze separate point and spatial recipes using internal validation, and then trained three experimental arms—point, no_b, and spatial—for each of the three encoders, with seeds 45, 46, and 47 for each arm, giving 27 final training runs. point used the point-07 recipe; no_b switched off B but retrained H/C using the same spatial-11 recipe as spatial. This setting makes the spatial−no_b comparison the increment associated with branch B under a matched recipe.

The point recipe used AdamW with a learning rate of 0.00015132861735491556, weight decay of 0.004910055796471723, and dropout of 0.5; the spatial recipe used Adam with a learning rate of 0.0001, weight decay of 0, and dropout of 0.3. Both used a batch size of 32, hidden dimension of 256, and a constant learning rate. Models were trained with mean squared error on normalized pathway scores. Formal checkpoints were selected by internal patient–pathway macro-averaged PCC; where that metric was tied within the prespecified tolerance, lower patient–pathway macro-averaged zMSE broke the tie. External data did not participate in recipe, checkpoint, or seed selection.

Final training allowed at most 120 epochs; formal checkpoint selection began at epoch 1, and early-stopping counting began at epoch 41 with a patience of 20 epochs. The actual stopping epoch varied by run, and the complete per-run table is [to be completed]. The downstream-delivered UNI2-h spatial model for seed 45 was trained through epoch 60 and used the formal checkpoint from epoch 28. Search-phase seeds 42/43/44 were not included in the final three-seed statistics.

### Training-density ablation

The density experiment fixed UNI2-h full-field features and the spatial recipe, and used dense, regular 2×2 subsampling, and patient-wise random equal-count arms. The random arm matched the number of training locations in the regular-subsampling arm for each patient; both sparse arms retained 2,383 training locations. A graph was constructed for each arm only from its retained training locations, without expanding the original neighborhood radius; internal and external evaluation sets and their dense structures remained unchanged.

All nine training runs across three seeds were initialized from scratch and did not use prediction-head weights that had seen the complete dense training set as their starting point. Pathway normalization in each arm was fitted only to that arm’s training truths; after the corresponding inverse transformation, predictions were returned to the z-score scale defined by the dense training set for comparison. Each training run used 14,800 fixed optimizer updates with early stopping disabled; the primary selector performed internal validation every 296 updates, yielding 50 selection opportunities. The sensitivity analysis used the best internal checkpoint within the first 50 complete training epochs on the same trajectories. The pre-existing frozen seed-45 model was used only as a separate reference, not as initialization or an additional replicate.

### Evaluation metrics and statistical presentation

We report two PCC definitions in parallel. Patient–pathway macro-averaged PCC is calculated by first correlating true and predicted location scores within every pathway of every patient and then averaging with equal weight across patients and pathways; pooled PCC is calculated once after separately flattening true and predicted z scores across all locations × pathways. The two definitions have different aggregation units, and their numerical magnitudes are not used to mutually validate superiority. Error metrics include pooled zMSE and, in the density ablation, MAE (mean absolute error) on the original pathway-score scale.

Internal and external results are presented separately. Repeated training was summarized as the arithmetic mean and sample SD of seeds 45/46/47. SD was calculated as the square root of the sample variance, using *n*−1 in the variance denominator, where *n*=3. Spatial-branch gains were calculated as spatial−no_b within the same seed before the three differences were summarized. This draft uses descriptive comparisons and does not report unperformed significance tests, patient confidence intervals, or post hoc selection of a winning seed.

## Pending experiments writing area—separate from completed results

The following content preserves manuscript locations and interpretive paths. It consists of current writing hypotheses, not obtained experimental results, and does not claim preregistration.

### Direct pathway prediction versus pathway reconstruction after gene prediction [To be completed]

We plan to compare direct pathway prediction with pathway aggregation after gene prediction using the same images, location partition, and frozen shared 30-dimensional truth. True genes will be used only to generate the shared truth once, and both routes will share that target; the gene-prediction arm will additionally feed predicted genes into the same aggregation workflow. If the reconstructed truth differs from the current V003 target, the comparison will be interpreted within the new shared truth and will not be ranked directly against the main-table scores.

**Hypothesis and expectation.** Direct supervision of pathways may reduce the propagation of gene-prediction error through aggregation. An ideal scenario would show internally and externally consistent directions for both PCC definitions and error metrics, with consistent findings across seeds. **Actual results:** [To be completed: formal inputs and aggregation algorithm; training budget and seeds; internal/external values for both PCC definitions, zMSE, mean ± SD, and run sources for both arms; Figure P2-5 and Table P2-4.] If no clear difference is observed, the report should state that no stable advantage was observed under the current protocol; if the gene route performs better, the discussion should be revised accordingly and examine gene-level supervision, aggregation, and capacity differences, without presupposing that the direct route must be superior.

### Second-batch encoder comparison [To be completed]

The second batch is planned to compare H-optimus-0, H-optimus-1, and Phikon-v2 under the fixed spatial recipe, running only the spatial arm with seeds 45/46/47 for each model, for a total of 9 training runs. H-optimus and Phikon-v2 use their respective paired input normalizations, so the comparison includes changes in both the encoder and its input specification; spatial results from the first batch serve as existing references.

**Hypothesis and expectation.** Replacing encoders may change reconstruction performance, but the second batch is not assumed to be superior to the first batch. An ideal scenario would show one new encoder with directionally consistent improvement in both PCC definitions and error metrics internally and externally. **Actual results:** [To be completed: run matrix for the three new encoders; internal/external values for both PCC definitions and error metrics, mean ± SD, and acceptance status; Table P2-5.] If there is no obvious improvement, or if internal and external directions disagree, conclusions should be limited to transferability under the fixed recipe and should not infer advantage from model size.

### Spatial heatmaps and UMAP [To be completed]

Figure P2-6 is intended to juxtapose true and predicted spatial pathway distributions, pending specification of the patient–slide–pathway selection rule, background image, coordinates, and scale. Figure P2-7 is intended to show UMAP (dimensionality-reduction visualization) of frozen representations, pending specification of the representation layer, training-fitted and fixed projection range, and pathway used for coloring. No observations are currently available for either figure type; this draft does not prewrite conclusions about spatial agreement, cluster separation, or biological mechanisms.

## Author checks and sources

Numbers in this draft derive from [accepted-results excerpts](../04_证据与对应/已接纳结果摘录.md); paired gains and sensitivity directions derive from [first-batch plotting data](../02_图表配置/首批图片_v1/data/source_manifest.json). Structure and training methods were checked against the current run-source snapshot, frozen recipes, and [model handoff materials](</D:/AI空间转录病理研究/PFMval_new/团队项目进度与结论/qzs/Phase2最终模型_Phase3交接包_20260918/README.md>); external papers are used only as background and method sources and do not replace this study’s run evidence.

Before full-manuscript integration, the following must be completed: clinical and ethics materials; raw scoring and gene-set materials; imaging and physical support domains; a per-run checkpoint summary; the two incomplete comparisons; and subsequent figures. Complete citation identities, purposes, and conditional citations are provided in [Phase 2 proposed references](Phase2拟引用文献.md).
