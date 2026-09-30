# CROSS-AD: Two-Channel EEG–MRI Learning for Explainable Alzheimer’s Disease Prediction

**CAS CS 506 — Tools for Data Science, Boston University**  
**Status:** Project proposal. Data collection, implementation, and experiments are planned; no results are claimed yet.

## Project Description

Alzheimer’s disease (AD) affects both brain structure and neural function. Structural magnetic resonance imaging (MRI) measures anatomical changes such as hippocampal atrophy and ventricular enlargement, whereas electroencephalography (EEG) measures functional changes in brain activity and connectivity.

This project proposes a temporary two-channel version of CROSS-AD containing:

1. An **EEG channel** for functional brain measurements.
2. An **MRI channel** for structural brain measurements.

The molecular-biomarker channel from the original CROSS-AD proposal is deferred to future work.

The primary harmonized classification task will be **Alzheimer’s disease versus cognitively normal controls (AD vs. CN)** because this label definition is available in both candidate datasets. A secondary MRI-only analysis may classify subjects as **CN, Mild Cognitive Impairment (MCI), or AD** if the available ADNI cohort is sufficiently large.

Subject-level EEG–MRI fusion requires both modalities to be collected from the same individuals. The currently identified public EEG and MRI sources contain different participants. Therefore:

- EEG and MRI models will first be developed and evaluated independently.
- A fused two-channel model will be trained and evaluated only if a genuinely paired EEG–MRI cohort becomes available.
- Subjects from separate datasets will not be artificially paired.
- Results from different cohorts will not be used to claim a multimodal performance improvement.

To keep the project feasible within approximately eight weeks, the MRI channel will prioritize existing regional measurements rather than processing raw 3D scans. The EEG channel will use interpretable spectral and connectivity features rather than a large end-to-end neural network.

## Research Questions and Measurable Goals

### Primary Questions

1. How accurately can interpretable EEG features distinguish AD from cognitively normal controls?
2. How accurately can regional structural MRI features distinguish AD from cognitively normal controls?
3. If paired EEG–MRI data become available, does combining functional and structural measurements improve classification relative to either modality alone?

### Measurable Goals

1. **Characterize each dataset.**  
   Report subject counts, diagnosis distributions, demographic distributions, feature availability, recording characteristics, and missingness.

2. **Develop an EEG baseline.**  
   Extract spectral and functional-connectivity features and train classifiers for AD-versus-CN prediction.

3. **Develop an MRI baseline.**  
   Use regional volumetric and cortical measurements to train classifiers for AD-versus-CN prediction. A secondary CN/MCI/AD analysis will be considered if feasible.

4. **Implement a two-channel fusion module.**  
   If matched EEG–MRI subjects are available, combine the two modality representations using feature concatenation or calibrated late fusion.

5. **Interpret model behavior.**  
   Identify influential EEG frequency bands, electrode relationships, and MRI regions using model coefficients, permutation importance, and error analysis.

Our hypothesis is that EEG and MRI capture complementary functional and structural manifestations of AD. However, performance improvements are not assumed. A reproducible result showing no multimodal improvement would also be meaningful.

## Proposed Two-Channel Architecture

```text
EEG recording
    │
    ├── Signal preprocessing
    ├── Spectral features
    ├── Functional-connectivity features
    │
    ▼
EEG encoder/classifier ───────► EEG prediction
    │
    └──────────────┐
                   │
                   ▼
             Fusion module ───► AD/CN prediction
                   ▲
                   │
    ┌──────────────┘
    │
MRI regional measurements
    │
    ├── Volume adjustment
    ├── Scaling and imputation
    ├── Regional feature selection
    │
    ▼
MRI encoder/classifier ───────► MRI prediction


## Expected Deliverables
1. A documented data collection and cohort-selection process.
2. A reproducible cleaning and feature-extraction pipeline.
3. Exploratory visualizations of the available measurements.
4. Unimodal baselines and, if matched data permit, a controlled multimodal comparison.
5. Quantitative evaluation, feature interpretation, and a discussion of limitations.
6. A final report and presentation describing findings, including negative results.

## Team members:
 - Aoming Liu (amliu@bu.edu)
 - Runkun Guo (stsun@bu.edu)
