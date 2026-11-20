# CROSS-AD: Two-Channel EEG–MRI Learning for Explainable Alzheimer’s Disease Prediction

**CAS CS 506 — Tools for Data Science, Boston University**  
**Status:** In progress. Data collection, preprocessing, unimodal EEG and MRI baselines, the unpaired "Option 1" prototype and a **paired two-channel EEG + MRI model** are done. The two-channel model reaches macro-F1 0.840 on the 25 paired BrainLat participants. See [Current Progress](#current-progress-two-channel-eeg--mri-model) and `reports/two_channel_report.pdf`.

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

```

## Eight-Week Timeline

| Period | Planned Work | Milestone |
| --- | --- | --- |
| Weeks 1–2 | Obtain the EEG dataset; request or confirm ADNI access; audit diagnostic labels, sample sizes, missingness, and the availability of paired EEG–MRI data | Finalize the feasible project scope and cohort definitions |
| Weeks 3–4 | Implement EEG and MRI preprocessing; extract spectral, connectivity, and regional MRI features; create exploratory visualizations | Complete cleaned feature tables and initial unimodal baselines |
| Weeks 5–6 | Train and tune EEG-only and MRI-only models; implement the two-channel fusion interface; train the fusion model only if paired data are available | Complete unimodal comparisons and, if feasible, the EEG–MRI fusion analysis |
| Week 7 | Evaluate the final models; analyze classification errors, feature importance, and modality contributions | Finalize quantitative results and interpretation |
| Week 8 | Consolidate the code, figures, documentation, report, and presentation | Complete the GitHub repository, final report, and presentation |

The schedule is approximate and will be aligned with the course’s October and November check-ins.

## Expected Deliverables
1. A documented data collection and cohort-selection process.
2. A reproducible cleaning and feature-extraction pipeline.
3. Exploratory visualizations of the available measurements.
4. Unimodal baselines and, if matched data permit, a controlled multimodal comparison.
5. Quantitative evaluation, feature interpretation, and a discussion of limitations.
6. A final report and presentation describing findings, including negative results.

## Current Progress: Two-Channel EEG + MRI Model

BrainLat's MRI demographics table documents an `EEG_ID` for some MRI participants, so a
genuinely paired EEG + structural MRI cohort exists inside BrainLat: **25 AD/CN participants
(15 AD, 10 CN, site CLB)**. On this cohort we trained and compared single-channel and
two-channel models on identical cross-validation splits (10 repeats × 5 folds).

| Model | Channels | macro-F1 (mean ± SD) | ROC AUC |
|---|---|---|---|
| **Pretrained encoders → concat → logistic regression (main)** | EEG + MRI | **0.840 ± 0.039** | **0.905** |
| Pretrained MRI encoder → logistic regression | MRI | 0.784 ± 0.048 | 0.849 |
| Paired-only MLP fusion (concat) | EEG + MRI | 0.777 ± 0.036 | 0.862 |
| Paired-only MRI MLP | MRI | 0.748 ± 0.064 | 0.845 |
| Paired-only EEG MLP | EEG | 0.664 ± 0.042 | 0.745 |
| Pretrained EEG encoder → logistic regression | EEG | 0.641 ± 0.023 | 0.689 |
| Logistic regression on age + sex | none | 0.475 ± 0.044 | 0.493 |

- **Two channels beat one.** MRI and EEG misclassify different people (5 MRI errors, 9 EEG
  errors, 1 shared); fusion corrects 4 of the 5 MRI errors.
- **Pretrain-then-fuse.** Each channel's encoder is pretrained on all unpaired BrainLat CN/AD
  data (404 T1 scans, 72 EEG recordings), frozen, and only a 17-parameter logistic-regression
  head is fitted on the paired participants. Test participants are removed from the
  pretraining pools in every split.
- **Uncertainty.** With 25 test participants, the gain over EEG alone is significant
  (+0.199, 95% CI +0.068 to +0.344); the gain over MRI alone (+0.056, CI −0.066 to +0.192) is not.

Full methods, testing protocol, figures and interpretation: **`reports/two_channel_report.pdf`**.
Earlier reports: `reports/pretrain_fusion_report.html`, `reports/option1_results.pdf`
(unpaired "Option 1" prototype), `reports/data_introduction.pdf`.

**No subject-level ADNI or BrainLat data is in this repository.** Both datasets forbid
redistribution; get them from the sources below and run the pipeline. Participant-level
outputs (manifests, features, splits, predictions) are written to `paths.work` in
`configs/*.yaml`, outside the repo; only aggregate tables are committed.
`docs/github_upload_manifest.csv` lists every file and whether it may be shared.

### Repository layout

```text
crossad/                 Python package (run modules from the repo root with python -m)
  data/                  download and verify raw data (BrainLat via Synapse, ds004504)
  preprocess/            EEG harmonisation + spectral features, ADNI MRI cohort
  option1/               Option 1: shared embedding from unpaired cohorts
  paired/                paired cohort audit, EEG/MRI features, paired-only models
  pretrain/              pretrain-then-fuse two-channel model (main model)
  reports/               report generators (HTML/PDF)
configs/                 paired_config.yaml, pretrain_config.yaml (all paths and settings)
jobs/                    SCC batch jobs (*.qsub)
tests/                   smoke test on synthetic features
results/                 aggregate results: option1/, paired/, pretrain/
reports/                 generated reports and figures
docs/                    data-sharing manifest, proposal slides
processed/               ds004504-derived tables (CC0)
```

### Data sources

| Dataset | Modality | Access |
|---|---|---|
| BrainLat | Resting-state EEG (EEGLAB .set/.fdt) and T1 MRI (BIDS) | [Synapse syn51549340](https://www.synapse.org/#!Synapse:syn51549340) (conditions of use). Needs a Synapse token in `~/.synapseConfig` |
| ADNI | 3T structural MRI, UCSF FreeSurfer 7 regional measures (Option 1 only) | [IDA](https://ida.loni.usc.edu) (application required). Export `UCSFFSX7`, `DXSUM`, `PTDEMOG`, `DATADIC`, `Key_MRI` into `MRI/` |
| OpenNeuro ds004504 | Resting-state EEG (AD/FTD/CN), BIDS (Option 1 only) | [openneuro.org/datasets/ds004504](https://openneuro.org/datasets/ds004504), CC0 |

### Running the pipeline (BU SCC)

Job files in `jobs/` are written and running in the my personal work space in BU SCC; the training data requires access under NIH or NSF grant number. If you want to reproduce the training part, please contact the data owner on ADNI and Synapse.

```bash
module load python3/3.10.12 pytorch/1.13.1
pip install --user mne==1.6.1            


# 1. data: full BrainLat release (EEG + MRI, ~105 GB)
qsub jobs/fetch_brainlat_all.qsub

# 2. paired cohort: audit + pairing, EEG features, SynthSeg MRI features; then models
qsub jobs/paired_features.qsub           # paired_audit, paired_eeg_features, paired_mri_features
qsub jobs/paired_train.qsub              # smoke test, then 8 paired-only models

# 3. two-channel pretrain-then-fuse
python -m crossad.pretrain.pretrain_cohorts --config configs/pretrain_config.yaml
qsub jobs/pretrain_synthseg.qsub        
qsub -hold_jid pt_synthseg jobs/pretrain_features.qsub
python -m crossad.pretrain.pretrain_train --config configs/pretrain_config.yaml

```

| Stage | Module | Aggregate output (committed) |
|---|---|---|
| Paired audit | `crossad.paired.paired_audit` | `results/paired/audit_*` |
| EEG features | `crossad.paired.paired_eeg_features` (1–30 Hz, 2 s epochs, regional relative band power, θ/α, peak alpha) | `results/paired/eeg_qc_summary.csv` |
| MRI features | `crossad.paired.paired_mri_features` (SynthSeg `--parc --robust`, 16 AD-signature volumes / ICV) | `results/paired/mri_qc_summary.csv` |
| Paired-only models | `crossad.paired.paired_train` | `results/paired/metrics_*`, `fusion_vs_single.csv` |
| Pretraining pools | `crossad.pretrain.pretrain_cohorts` | `results/pretrain/pool_counts.csv` |
| Two-channel model | `crossad.pretrain.pretrain_train` | `results/pretrain/metrics_*`, `comparisons.csv`, `pretrain_encoders.csv` |

**Option 1 (unpaired prototype, October).** `jobs/run_pipeline.qsub` builds the ADNI MRI
cohort and BrainLat EEG features (`crossad.preprocess`), `jobs/fetch_ds004504.qsub` and
`crossad.preprocess.prepare_ds004504_eeg` add ds004504, `jobs/train_option1.qsub` trains the
model (`results/option1/`), and `crossad.reports.make_data_report` /
`make_option1_report` write the reports.

### Acknowledgments

Data collection and sharing for this project was funded by the Alzheimer's Disease
Neuroimaging Initiative (ADNI) (National Institutes of Health Grant U01 AG024904) and
DOD ADNI (Department of Defense award number W81XWH-12-2-0012). Data used in preparation
of this work were obtained from the ADNI database (adni.loni.usc.edu). As such, the
investigators within the ADNI contributed to the design and implementation of ADNI and/or
provided data but did not participate in analysis or writing of this work. A complete
listing of ADNI investigators can be found at
http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf

BrainLat: Prado P. et al., "The BrainLat project, a multimodal neuroimaging dataset of
neurodegeneration from underrepresented backgrounds", *Scientific Data* 10, 889 (2023).

ds004504: Miltiadous A. et al., "A dataset of scalp EEG recordings of Alzheimer's
disease, frontotemporal dementia and healthy subjects from routine EEG", *Data* 8(6), 95 (2023).

## Team members:
 - Aoming Liu (amliu@bu.edu)
 - Runkun Guo (stsun@bu.edu)
