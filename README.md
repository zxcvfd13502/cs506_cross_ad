# CROSS-AD: Multimodal Learning for Explainable Alzheimer's Disease Prediction

**CAS CS 506 — Tools for Data Science, Boston University**  
**Status:** Project proposal. Data collection, implementation, and experiments are planned; no results are claimed yet.

## Project Description

Alzheimer's disease (AD) involves changes at multiple biological levels. Structural MRI measurements, EEG recordings, and molecular biomarkers offer different views of these changes. This project investigates whether combining complementary measurements improves diagnostic-group classification compared with using a single modality.

Our primary task is to classify subjects as **Cognitively Normal (CN)**, **Mild Cognitive Impairment (MCI)**, or **Alzheimer's Disease (AD)** using measurements associated with the same clinical visit. Here, prediction means classification of the recorded diagnostic group, rather than forecasting future disease onset.

The core analysis will compare **MRI-derived features**, **molecular biomarkers**, and **their combination** in subjects with matched measurements. EEG will be an exploratory component, analyzed independently unless genuine subject-level overlap is available. We will also examine which features contribute to model predictions.

To keep the project feasible within approximately eight weeks, we will prioritize existing regional MRI measurements and interpretable machine-learning baselines. Processing raw 3D MRI scans and developing deep-learning architectures are outside the core scope.

## Research Question and Measurable Goals

**Primary question:** Does combining MRI-derived measurements and molecular biomarkers improve CN/MCI/AD classification over either modality alone?

1. **Characterize the available data.** Report subject counts by diagnosis, modality availability, matched-subject counts, and feature-level missingness. Compare distributions of selected structural and molecular measurements across diagnostic groups.
2. **Establish unimodal baselines.** Train and evaluate MRI-only and molecular-biomarker-only classifiers, alongside a majority-class reference model. Evaluate EEG separately if time and data quality permit.
3. **Measure the contribution of multimodal information.** Compare MRI-only, molecular-only, and combined models on the same matched subjects and identical evaluation splits. Use macro-F1 as the primary metric and report the combined model's difference from the strongest unimodal model selected using training-side validation.
4. **Interpret model behavior.** Identify influential features using standardized model coefficients or permutation importance, and report class-specific errors with confusion matrices and per-class recall.

Our hypothesis is that complementary modalities will improve classification. A reproducible comparison showing no improvement is also a meaningful outcome; performance gains are not assumed.

## Data Collection Plan

### MRI and Molecular Biomarkers: ADNI

The planned primary source is the **Alzheimer's Disease Neuroimaging Initiative (ADNI)**. Its research data include clinical, imaging, and biofluid biomarker information. Access requires an application and approval; the data are not unrestricted public downloads. See the [official ADNI data access page](https://adni.loni.usc.edu/data-samples/adni-data/).

We plan to collect:

| Information | Candidate measurements | Purpose |
| --- | --- | --- |
| Diagnosis and visit metadata | Diagnostic category, subject identifier, visit identifier, assessment date | Define labels and align measurements |
| MRI-derived structural features | Hippocampal and ventricular volumes, regional gray-matter volumes, cortical thickness where available | Structural predictors |
| Molecular biomarkers | Available amyloid- and tau-related measurements; other markers only if coverage is adequate | Molecular predictors |
| Demographic and acquisition metadata | Age, sex, imaging site, and relevant measurement metadata where available | Describe the cohort and examine potential confounding |

**Collection method:**

1. Apply for ADNI access through the LONI Image and Data Archive (IDA) using our institutional affiliation and project description.
2. After approval, download the relevant clinical, MRI-summary, and biomarker tables and their data dictionaries through the IDA study-files interface. The [ADNI download guide](https://adni.loni.usc.edu/quick-start-guide-asset/getting_started.html) describes access to tabular data.
3. Record source filenames, download dates, available versions, units, and selected columns so the collection can be reproduced by authorized users.
4. Match tables using subject and visit identifiers, checking acquisition dates. Prefer same-visit measurements; if a date tolerance is necessary, define and document it before modeling and avoid using future measurements to predict an earlier diagnosis.
5. Audit class counts, modality overlap, missingness, and assay compatibility before finalizing the cohort and feature set. Sample size will be reported after this audit rather than estimated without access to the data.

Raw subject-level data will remain outside this public repository. The repository will document acquisition and processing steps and contain only materials permitted for public sharing.

### EEG: Exploratory Dataset

A candidate is [OpenNeuro dataset ds004504](https://openneuro.org/datasets/ds004504), titled *A dataset of EEG recordings from: Alzheimer's disease, Frontotemporal dementia and Healthy subjects*. Its [dataset repository](https://github.com/OpenNeuroDatasets/ds004504) provides metadata and dataset documentation.

We plan to download a documented dataset version through OpenNeuro's download interface, including EEG files, participant labels, and recording metadata. We will verify the available diagnoses and data quality before selecting the task. For this candidate, the initial analysis would focus on AD versus healthy controls; we will not assume an MCI group or relabel frontotemporal dementia as MCI.

Candidate EEG features include relative delta, theta, alpha, and beta power, spectral summaries, and selected connectivity measures if feasible. These participants will not be treated as matched ADNI participants. Performance from different cohorts or label sets will be reported separately and will not establish a multimodal benefit.

### Feasibility Decision

By the end of Week 2, we will document data-access status, cohort sizes, label distributions, and usable modality overlap. These findings will determine the final analysis scope. Three-modality fusion is an optional extension only if sufficiently matched data are actually available.

## Data Cleaning and Feature Extraction

The planned pipeline will:

- Resolve duplicate records and inconsistent identifiers or diagnostic labels.
- Check measurement units, missing-value codes, implausible values, and differences between assays or acquisition protocols.
- Select one eligible visit per subject for the initial analysis using a documented rule, such as the earliest visit with the required measurements.
- Extract regional MRI features and selected molecular measurements; consider intracranial-volume adjustment for regional volumes where appropriate.
- Summarize missingness and document exclusions. Fit any imputation, scaling, and feature selection using training data only.
- If EEG is included, inspect recording quality, apply documented preprocessing, and aggregate segment-level features to the subject level. Segments from the same subject will stay within the same evaluation partition.

Diagnostic labels, identifiers, and variables that directly encode the target diagnosis will not be predictors. Demographic variables, if used, will be handled consistently across modality comparisons.

## Tentative Modeling Plan

We will begin with logistic regression and compare a small selection of course-relevant methods such as k-nearest neighbors, decision trees, and support vector machines. Model choices and tuning ranges will be refined after inspecting the data.

The initial multimodal method will concatenate the available MRI and molecular feature vectors:

```text
MRI features ---------------------> Unimodal classifier
Molecular features ---------------> Unimodal classifier
MRI features + molecular features -> Multimodal classifier
```

All three experiments will use the same matched cohort and evaluation partitions. Additional unimodal experiments on larger available cohorts may be reported separately, with their sample sizes clearly identified.

## Evaluation and Test Plan

- **Split by subject:** Reserve approximately 20% of eligible subjects for final testing, with class stratification where feasible. Use stratified cross-validation within the remaining subjects to select models and hyperparameters. If the cohort is too small for a reliable holdout, use nested subject-level cross-validation and document the change before model selection.
- **Prevent leakage:** Fit preprocessing inside each training fold. Keep all visits or EEG segments belonging to a subject together, and do not use the final test set for tuning or feature selection.
- **Report complementary metrics:** Use macro-F1 as the primary metric, supported by balanced accuracy, accuracy, per-class precision and recall, and confusion matrices. Report ROC-AUC only when the class structure and model scores support it.
- **Make matched comparisons:** Report the macro-F1 difference between the combined model and the selected unimodal baseline on identical evaluation subjects. Where sample size permits, estimate uncertainty using paired subject-level bootstrap resampling.
- **Interpret limitations:** Examine class imbalance, sample size, missing-data selection, site effects, and correlated features. Feature importance will describe model associations rather than establish biological causation.

## Tentative Visualization Plan

Planned figures include diagnostic-group counts, modality-overlap summaries, missingness plots, biomarker distributions by group, correlation matrices, and exploratory PCA plots. Model analysis will include performance comparisons, confusion matrices, and feature-importance plots. Figures will identify the cohort and sample size used.

## Eight-Week Timeline

| Period | Planned work | Milestone |
| --- | --- | --- |
| Weeks 1–2 | Request data access; collect tables and metadata; audit labels, missingness, and modality overlap | Document the available cohort and finalize the feasible scope |
| Weeks 3–4 | Clean and match records; extract features; create exploratory figures; train an initial baseline | Prepare preliminary processing, visualizations, and results for the first check-in |
| Weeks 5–6 | Train additional unimodal models; implement feature fusion; compare models using shared validation splits | Prepare multimodal comparisons and modeling progress for the second check-in |
| Week 7 | Complete evaluation; examine errors and feature importance; assess limitations | Finalize quantitative comparisons and interpretation |
| Week 8 | Consolidate findings, figures, reproducibility instructions, and presentation | Complete the final report and presentation |

The schedule is approximate and will be aligned with the course's October and November check-ins.

## Fallback Plan

If matched MRI, molecular, and EEG data are unavailable, the core project will remain MRI-only versus molecular-only versus MRI + molecular biomarkers; EEG will remain independent or be omitted.

If the matched MRI–molecular cohort is too small, we will reduce the feature set and modeling complexity. If necessary, we will narrow the task to CN versus AD and explicitly report the change in scope. If access to ADNI is delayed or unavailable, we will discuss a revised scope with course staff, using the candidate EEG dataset for a unimodal classification and interpretation project. Such a fallback would not answer the original multimodal research question.

## Expected Deliverables

1. A documented data collection and cohort-selection process.
2. A reproducible cleaning and feature-extraction pipeline.
3. Exploratory visualizations of the available measurements.
4. Unimodal baselines and, if matched data permit, a controlled multimodal comparison.
5. Quantitative evaluation, feature interpretation, and a discussion of limitations.
6. A final report and presentation describing findings, including negative results.

This repository currently contains the proposal only. Implementation, environment setup, run instructions, and tests will be added as the project develops.

## Team and Course

- **Team members:** To be added.
- **Repository owner:** [zxcvfd13502](https://github.com/zxcvfd13502)
- **Course:** CAS CS 506 — Tools for Data Science, Boston University
- **Assignment:** [Final project requirements and proposal rubric](https://gallettilance.github.io/final_project/)
