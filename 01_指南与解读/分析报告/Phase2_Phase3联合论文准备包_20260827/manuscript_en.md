# From H&E-Derived Spatial Pathway Representations to Treatment-Response Bridging: An Evidence-Constrained Two-Phase Computational Pathology Framework

> Status: `pending_user_review`; general-journal IMRaD draft.
> Accepted-evidence snapshot: local `main` commit `9859a3c`; exploratory reanalysis version: `3d0cad7`. The manuscript package itself remains pending commit and user review.
> This is not a submission-ready manuscript. Authors, affiliations, ethics identifiers, full eligibility criteria, and journal-specific formatting remain to be supplied.

## Abstract

### Background

Inferring spatial gene or pathway expression from routine hematoxylin-and-eosin (H&E) images may provide biologically structured representations for cohorts without spatial transcriptomics. However, correlation does not establish absolute agreement or cross-patient generalization. In the proposed end-to-end design, morphology and pathway branches derived from the same H&E image would not constitute independent image–omics modalities. We developed an evidence-constrained Phase 2-to-Phase 3 narrative rather than claiming a completed pipeline.

### Methods

Phase 2 comprised the accepted W007 four-arm experiment: frozen baseline replay (FBR), residual-only control (RCC; historical arm code retained), hard-pair contrastive residual learning (HCR), and continuous pathway-geometry contrastive residual learning (CPGCR). All arms shared a frozen baseline and a 30-dimensional pathway output. CPGCR constructed soft relational targets from continuous pathway geometry while retaining image-only inference. Phase 3 comprised the restricted W004 four-arm result group: absolute input (A001), within-slide ordinal input (A002), true-coordinate spatial smoothing (A003), and coordinate-shuffle spatial control (A004). Mean per-fit and pooled repeated-row estimands were kept separate. Diagnostics included Pearson correlation coefficient (PCC), concordance correlation coefficient (CCC), z-RMSE, raw R², Spearman correlation, AUROC, AUPRC, Brier score, and calibration. External work was assessed by targeted source verification rather than a systematic review.

### Results

All W007 single-seed arms were formally accepted, but differences were small and metric-dependent. RCC achieved the lowest internal patient-balanced z-MSE (0.378252); HCR achieved the highest pooled PCC in the single external case E1 (0.655982); and CPGCR achieved numerically favorable external z-MSE (0.656832), raw MAE (1173.902), and mean pathway raw R² (−0.080119). No arm dominated all metrics. In W004, the mean per-fit AUC differences for A002 versus A001 were +0.023182 for pathological complete response (pCR) and +0.003958 for major pathological response (MPR). The A003-versus-A004 differences were +0.002955 and −0.000625, respectively. This existing result group remains restricted to `COMPATIBILITY_ONLY`: it used legacy z-score inputs and repeated slide-level holdout with patient overlap, and therefore does not constitute patient-independent out-of-fold (OOF), fusion-synergy, or clinical-utility evidence.

### Conclusions

The authors position CPGCR as a Phase 2 methodological candidate to be validated, not as the best-performing model. Its 30-dimensional output is only shape-compatible with a future interface; no W007-to-W004 data binding has been demonstrated. The W004 ordinal signal warrants new patient-independent validation, whereas the spatial control comparison does not support a stable spatial gain. Patient-independent Phase 3 evaluation, calibration, capacity-matched negative controls, and fresh external validation are required before evidence upgrade.

**Keywords:** computational pathology; spatial transcriptomics; pathway expression; contrastive learning; treatment response; pCR; calibration; patient-level validation

## 1. Introduction

Spatial transcriptomics links tissue morphology to local molecular states, but cost, tissue consumption, and technical requirements limit broad clinical deployment. ST-Net, TRIPLEX, HEST-1k, and recent single-cell or generative studies show that H&E morphology contains signals associated with local expression. Yet a 2025 Nature Communications benchmark of 11 methods, five datasets, and 28 metrics demonstrated that a single correlation coefficient is insufficient to establish translational value and that rankings may change across datasets, downstream tasks, and evaluation dimensions ([official article](https://www.nature.com/articles/s41467-025-56618-y)).

Pathway-level targets offer two potential advantages: they aggregate sparse gene-level measurements into biological processes and can serve as a biologically structured bottleneck for H&E representations. PEaRL, formally published at WACV 2026, directly studied H&E-based gene and pathway-expression prediction ([CVF](https://openaccess.thecvf.com/content/WACV2026/html/Majumder_PEaRL_Pathway-Enhanced_Representation_Learning_for_Gene_and_Pathway_Expression_Prediction_WACV_2026_paper.html)). DeepPathway appeared as a Bioinformatics accepted manuscript in August 2026, reinforcing pathway-level contrastive modeling as an active direction. Conversely, HESCAPE reported that cross-modal pretraining improved some mutation-classification tasks while degrading direct expression prediction. Contrastive learning itself therefore cannot be treated as evidence of better regression.

A second question is whether an upstream molecular proxy supports clinical endpoints. ENLIGHT-DeepPT provides a precedent for a two-stage H&E-to-imputed-transcriptomics-to-treatment-response design in large multi-cancer and independent cohorts ([Nature Cancer](https://pubmed.ncbi.nlm.nih.gov/38961276/)). MCAT, SurvPath, and Pathomic Fusion combine independently measured omics with images. In the proposed future integration here, both morphology and pathway branches would originate from H&E, so the accurate term would be an H&E-derived morphology–pathway dual representation rather than an independent modality. The complete upstream provenance of the current W004 legacy input was not reverified in this task, so same-source derivation is not asserted as a fact about W004.

We organized the available evidence around three questions: (1) what are the implementation and performance boundaries of a continuous pathway-geometry contrastive method under matched four-arm comparisons; (2) how do PCC, CCC, error, and calibration distinguish trend recovery from absolute agreement; and (3) do absolute, ordinal, and spatial-counterfactual inputs provide a restricted signal for Phase 3 pCR/MPR prediction? The goal was not to establish clinical utility from a small cohort, but to create an auditable and reproducible two-phase manuscript-preparation framework.

## 2. Methods

### 2.1 Study design and evidence hierarchy

This work is a governed retrospective review and exploratory reanalysis of existing local experiments. No project server was contacted, no models or training data were downloaded, and no model was retrained. The provenance order for project facts was: accepted Experiment Registry results; local `pending_user_review` reports and new reanalyses; Google Drive design recommendations; and official external literature. This order concerns provenance and lifecycle, not clinical evidence strength; `accepted` means project-internal governance acceptance only. External studies were assessed separately by study-design level and source fitness.

W007 provided project-accepted single-seed Phase 2 evidence. The current W004 result group remained `COMPATIBILITY_ONLY`. All new computations remain `exploratory_reanalysis / pending_user_review`. Approval, attempt-lifecycle, and transaction details are retained in the internal review index rather than the scientific main text.

### 2.2 Phase 2: W007 four-arm representation learning

Each spot was represented by a 1,536-dimensional frozen pathology-foundation-model feature and mapped to 30 pathway z-score predictions. FBR replayed the accepted frozen B0 predictor. RCC added a 1,536-to-256 image projector and a zero-initialized 256-to-30 residual head; it is a residual-only protocol control and cannot by itself identify a pure capacity effect. HCR added bidirectional hard-pair InfoNCE to the same prediction architecture. CPGCR transformed pairwise mean-squared distances between the true 30-dimensional pathway vectors into soft within-batch relational targets, replacing the conventional one-positive/all-other-negative assumption. The true-pathway encoder was used only during HCR/CPGCR training; inference remained image-only.

The absolute regression objective was patient-balanced. Batches comprised six patients and five spots per patient. Adam was used with a learning rate of 1×10⁻⁴, a maximum of 50 epochs, and patience of 10. Checkpoints were selected exclusively by internal patient-balanced z-MSE; external case E1 was not used for selection. The full training implementation is bound to historical source commits, and the current `main` checkout does not contain the complete runnable W007 training tree.

### 2.3 Phase 2 metrics

PCC, CCC, z-RMSE, raw R², and Spearman correlation were calculated per pathway and patient before patient-equal aggregation. PCC quantifies linear association; CCC additionally penalizes location and scale disagreement. Diagnostic gaps were defined as `PCC−CCC` and `PCC²−raw R²`. MSE was decomposed into location, scale, and correlation-structure components. Moran’s I and hotspot analyses were computed only when complete coordinates were available; otherwise the output was explicitly marked `not_computable`, without reconstructing protected data.

### 2.4 Phase 3: W004 four-arm bridge

Project materials identify W004 as a neoadjuvant-treatment-response task in esophageal squamous cell carcinoma (ESCC). The critical contract records 78 patients and 307 effective slides. Endpoints were pathological complete response (pCR) and major pathological response (MPR), but thresholds, adjudicators, blinding, event counts, center, and study period remain author-supplied placeholders and are not inferred here.

W004 trained each endpoint over four seeds and five folds: 20 fits per endpoint and 40 fits per arm across pCR and MPR. Twenty fits are not 20 independent patients. Each slide-level patch bag contained 1,536-dimensional image features, 30-dimensional pathway inputs, and coordinates. The two branches were mapped to hidden dimension 128, combined by an adaptive gate, pooled by attention-based multiple-instance learning (MIL), and mapped to a binary logit. Training used patient-weighted binary cross-entropy (BCE), AdamW with a learning rate of 1×10⁻³, up to 20 epochs, patience of five, and validation BCE for selection.

A001 used absolute legacy z-score inputs. A002 converted each pathway to a within-slide percentile rank. A003 applied k-nearest-neighbor smoothing at the true coordinates (k=5, lambda=0.5). A004 deterministically shuffled coordinates before applying the same smoother. Prespecified contrasts were A002−A001 and A003−A004.

### 2.5 Phase 3 metrics and statistical boundaries

AUROC, AUPRC, and Brier score were computed per fit, with paired contrasts at identical task/seed/fold combinations. Pooled repeated-row ROC and precision–recall curves were descriptive and kept separate from mean per-fit estimands. Aggregation by `case_id` was post hoc and was not described as patient-independent OOF. Calibration intercept and slope were supplied only when labels and probabilities supported estimation, with patient non-independence disclosed in every label and caption.

The 20 seed-fold estimates were not treated as 20 independent patients and were not used in ordinary t tests. Because no valid patient-level OOF predictions were available, we did not make DeLong-test or decision-curve claims. Future evaluation should use patient-grouped splitting and paired patient-level cluster bootstrap.

### 2.6 Literature review and reporting framework

The review was conducted on 27 August 2026 as targeted source verification rather than a systematic review. Seventy-eight title/topic queries yielded 44 included sources: 32 from 2024–2026, 43 peer reviewed, and one explicitly downgraded preprint. Sources were limited to PubMed, journal sites, DOI records, CVF, NeurIPS Proceedings, and BMJ.

TRIPOD+AI ([BMJ](https://www.bmj.com/content/385/bmj-2023-078378)) and PROBAST+AI informed manuscript preparation only. A formal item-level TRIPOD+AI mapping and four-domain PROBAST+AI assessment have not yet been completed. Neither framework itself establishes low bias or clinical validity.

### 2.7 Ethics, privacy, and availability placeholders

The research ethics committee name, approval identifier and date, scope of retrospective secondary analysis, written-consent or waiver basis, de-identification procedure, and data-access controls have not been supplied by the authors and block submission. The public-facing draft uses “external case E1” rather than an internal patient code. No patient-level files were written to Google Drive or public Web services; institutional compliance for AI processing of local human data must still be confirmed by the project lead.

## 3. Results

### 3.1 No W007 arm was an overall performance winner

**Table 1. Accepted W007 four-arm metrics; single seed and one external case.**

| Arm | Internal patient-balanced z-MSE ↓ | Internal pooled PCC ↑ | External E1 z-MSE ↓ | External pooled PCC ↑ | External raw MAE ↓ | External mean pathway raw R² ↑ |
|---|---:|---:|---:|---:|---:|---:|
| FBR | 0.380743 | 0.797126 | 0.662077 | 0.654896 | 1176.211 | −0.088017 |
| RCC | **0.378252** | **0.798154** | 0.659534 | 0.653758 | 1177.107 | −0.087571 |
| HCR | 0.379641 | 0.797706 | 0.658109 | **0.655982** | 1174.598 | −0.082654 |
| CPGCR | 0.378847 | 0.798000 | **0.656832** | 0.654932 | **1173.902** | **−0.080119** |

RCC was numerically strongest on internal absolute error, HCR on pooled PCC in external case E1, and CPGCR on several E1 error-based metrics. Differences were small and E1 represented only one patient. The results therefore support metric-specific variation rather than overall superiority. The authors position CPGCR as a candidate because of its continuous pathway geometry and matched ablations; its 30-dimensional output is only shape-compatible with a future interface and is not empirically bound to Phase 3.

### 3.2 Role of the multi-metric reanalysis

The new reanalysis computed every metric within patient×pathway before taking an equal-weight patient mean, and wrote PCC/CCC, z-RMSE, raw R², Spearman correlation, PCC−CCC, PCC²−R², and MSE decomposition to `experiments/explorations/phase2_phase3_reanalysis_v002/`. Across 30 pathways and six equally weighted internal patients, RCC had the numerically highest PCC/CCC/raw R² (0.6504/0.6112/0.2058), compared with 0.6503/0.6103/0.1990 for CPGCR; the differences were small. In the single external case E1, the four-arm mean pathway PCC ranged from 0.5165 to 0.5194, CCC from 0.3978 to 0.3993, and raw R² remained negative (−0.0880 to −0.0801). These are `pending_user_review` exploratory derivatives; they neither modify the W007 Registry nor establish a new winner.

Coordinates were complete for W007, allowing Moran’s I in 840 patient×pathway×arm/split units. Hotspots were descriptive quadrant counts without permutation inference. They characterize within-patient spatial structure and do not estimate a population of external patients. The patient-balanced visualization is `figures/figure3_w007_patient_balanced_metrics.png`.

### 3.3 The W004 ordinal signal exceeded the spatial signal but remained compatibility-only

**Table 2. Fixed W004 four-arm AUROC contrasts; `COMPATIBILITY_ONLY`.**

| Contrast | Endpoint | Control mean AUC | Treatment mean AUC | Difference |
|---|---|---:|---:|---:|
| A002 ordinal − A001 absolute | pCR | 0.838636 | 0.861818 | **+0.023182** |
| A002 ordinal − A001 absolute | MPR | 0.871667 | 0.875625 | +0.003958 |
| A003 spatial identity − A004 shuffle | pCR | 0.837273 | 0.840227 | +0.002955 |
| A003 spatial identity − A004 shuffle | MPR | 0.872292 | 0.871667 | −0.000625 |

The pCR ordinal difference supports further patient-independent validation, whereas the MPR difference was weak. The spatial identity-versus-shuffle differences were near zero and changed direction across endpoints, providing no support for a stable spatial gain. Arm-level AUPRC and Brier summaries in the Registry were accepted within the group’s `COMPATIBILITY_ONLY` scope; newly generated pooled, case-aggregated, and calibration outputs remain exploratory and pending review. In the pooled test description, ORD had pCR AUPRC/Brier values of 0.8236/0.1361 versus 0.8132/0.1458 for ABS; the corresponding MPR differences were small. Because the 620 rows arise from repeated fits rather than 620 independent patients, they cannot support patient-level inference. W004 prediction CSVs lacked coordinate columns, so Moran’s I and hotspot analyses were registered as `not_computable` without reconstructing protected data.

### 3.4 Novelty lies in the combination of continuous geometry, constrained residual learning, and evidence bridging

The literature already covers pathway-level prediction (PEaRL and DeepPathway), cross-modal representations (BLEEP, TANGLE, and OmiCLIP), spatial-expression benchmarking (HEST-1k and the H&E-to-SGE benchmark), and predicted transcriptomics for clinical outcomes (ENLIGHT-DeepPT and PathGen). We therefore do not claim the first use of pathway-level contrastive learning. The testable distinction is the combination of: continuous 30-dimensional ssGSEA geometry; an accepted frozen B0 anchor with zero-start residual learning; matched FBR/RCC/HCR/CPGCR ablations; explicit separation of relative trend, absolute agreement, and treatment-response bridging; and governed stratification of negative and restricted evidence.

## 4. Discussion

The principal finding was not that one arm “won,” but that different evaluation dimensions exposed different limitations. The winners for W007 correlation, absolute error, and pathway-wise R² were not the same. PCC can remain high under global shifts or rescaling, making CCC, R², and error decomposition necessary to distinguish trend recovery from absolute calibration. Lin’s CCC and external-validation methodology for continuous outcomes support this framework ([Lin 1989](https://pubmed.ncbi.nlm.nih.gov/2720055/)).

The W004 ordinal pCR difference motivates the hypothesis that within-slide pathway rank may transfer better than absolute magnitude when cross-patient calibration is unstable. However, patient overlap in the current protocol creates risks of pseudoreplication and patient-specific memorization. The result is a numerical signal within a compatibility comparison, not evidence of new-patient benefit. The A003/A004 result also illustrates the value of preserving negative findings: true-coordinate smoothing did not produce endpoint-consistent gains over coordinate shuffling.

The two-phase narrative faces an additional same-source information problem. SurvPath, MCAT, and Pathomic Fusion use independently measured omics, whereas the proposed W007-to-Phase 3 pathway branch would be inferred from H&E. PathGen is a closer precedent for same-source synthetic expression. The current W004 legacy-input provenance was not reverified; any future claim of incremental pathway value requires a bound provenance chain plus image-only, pathway-only, direct-concatenation, learned-fusion, PCA-30, random-30, patient-shuffled-pathway, and pathway-identity-permutation controls.

TRIPOD+AI requires transparent reporting of participants, splits, performance, subgroups, and openness. PROBAST+AI further examines bias in participants/data, predictors, outcomes, and analysis. The main current high-risk features are patient-nonindependent splitting, a single external patient, limited calibration evidence, multiple comparisons, and the absence of the full W007 training tree from current `main`. Explicitly retaining these limitations supports a credible “method candidate + restricted bridge + prespecified validation path” rather than an overstated clinical claim.

## 5. Limitations

1. W007 has one formally accepted seed; internal evaluation contains six patients and external case E1 contains one patient.
2. W004 uses repeated slide-level holdout with patient overlap; case aggregation is post hoc.
3. W004 used legacy z-score compatibility input and was not bound to W007 CPGCR output.
4. Current results do not establish raw MPP2 validity, fusion synergy, patient-independent generalization, or clinical utility.
5. New reanalyses are derived exploratory evidence and do not enter the Registry before user review.
6. The current `main` checkout lacks the full W007 training implementation; a future source-bound restoration and independent reproducibility check are required.
7. Ethics approval identifiers, informed-consent or waiver details, specimen timing, scanner information, and batch metadata remain to be provided by the authors.

## 6. Conclusions and future work

The safest current conclusion is that the authors position CPGCR as a biologically motivated, implemented Phase 2 methodological candidate, without consistent cross-metric performance superiority. Its 30-dimensional output is shape-compatible with a possible future interface, but no downstream data binding has been shown. The W004 ordinal pCR signal justifies a new validation protocol, while the spatial counterfactual does not support a stable spatial gain. Multi-seed W007 evaluation, patient-level Phase 3 OOF predictions, capacity-matched negative controls, and fresh external validation are prerequisites for claims of performance increment or clinical value.

## Data, code, and AI-assistance disclosure

- This draft used only local registries, return bundles, prediction tables, code or historical Git objects, and read-only Google Drive documents. No project server was contacted.
- New scripts and outputs are under `scripts/explorations/phase2_phase3_reanalysis_v002/` and `experiments/explorations/phase2_phase3_reanalysis_v002/`.
- AI assisted with literature verification, static code review, reanalysis scripting, and drafting. Authors and the project lead retain responsibility for factual, ethical, and reporting review; compliance with institutional policy for AI processing of local human data remains to be confirmed.
- No Gitee/GitHub push or Google Drive write occurred, and no original W004 bundle or protected asset was modified.

## References

Formal entries for all 44 verified sources are provided in `references.bib`, with query, publication-status, and screening records in `literature_search_log.md`. Core sources include PEaRL (WACV 2026), DeepPathway (Bioinformatics 2026 accepted manuscript), HESCAPE (ICCV Workshops 2025), BLEEP (NeurIPS 2023), the H&E-to-SGE benchmark (Nature Communications 2025), TRIPOD+AI, and PROBAST+AI (BMJ).
