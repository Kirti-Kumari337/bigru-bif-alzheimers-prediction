# A Dual-Phase Longitudinal Framework for Alzheimer's Disease Prediction

## Integrating Biological Irreversibility Filtering with Bidirectional GRUs

**Kirti Kumari, Abhay Chaudhary, Mansi Butola, Khushi Krishali**  
Department of Computer Science and Engineering  
Graphic Era Hill University, Dehradun, India

---

## Overview

This repository contains the complete implementation of a novel dual-phase framework for Alzheimer's Disease prediction using longitudinal clinical data from ADNI. The key contribution is the **Biological Irreversibility Filter (BIF)** — a preprocessing step that corrects biologically implausible diagnostic reversals before model training.

> **Core idea:** Alzheimer's Disease does not spontaneously reverse. Any record showing a patient improving from AD back to MCI or CN is a data error — not a clinical event. The BIF fixes these errors before the model ever sees the data.

---

## Key Results

| Phase | Task | AUC | Sensitivity |
|---|---|---|---|
| Phase 1 | CN vs AD Classification | **0.9877** | **96.37%** |
| Phase 2 | MCI→AD Conversion (24-month) | **0.8253** | **85.67%** |
| Site Holdout | External Site 130 (98 patients) | **0.8329** | **70.00%** |

**BIF Contribution:** +0.008 AUC and +4.67 percentage points sensitivity  
**No neuroimaging required** — only 6 routine clinical variables

---

## Repository Structure

```
📁 bigru-bif-alzheimers/
├── 01_Data_Preprocessing_Pipeline.ipynb     ← ADNI data cleaning & merging
├── 02_BIF_and_Phase1_CN_vs_AD.ipynb         ← BIF application + Phase 1 model
├── 03_Phase2_MCI_Conversion_Prediction.ipynb ← Phase 2 model + ablation + SHAP
├── 04_Site_Level_Holdout_Validation.ipynb   ← External site generalisation
└── README.md
```

---

## Notebook Guide

### 01 — Data Preprocessing Pipeline
**Source:** `Untitled45.ipynb`  
Loads 5 raw ADNI data streams (DXSUM, MMSE, ADAS-Cog, demographics, APOE4), harmonises VISCODE labels, removes 697 duplicates, imputes missing ADAS scores using visit-type median, and computes patient age dynamically from birth year.

**Output:** `FINAL_CLEAN_FULL_with_demo_apoe_age_fixed.csv` — 15,168 visits, 3,788 patients

---

### 02 — BIF + Phase 1 CN vs AD
**Source:** `Copy_of_ADNI_DATA_MERGE_AND_CLEAN__3_.ipynb`  
Applies BIF (corrects 527 reversals using `cummax()`), generates trajectory visualisation, and trains Bi-GRU for binary CN vs AD classification with SHAP explainability.

**Key results:**
- BIF fixed: **527 reversals**
- Post-BIF: CN=5,458, MCI=6,688, AD=2,981
- Phase 1 AUC: **0.9877** | Sensitivity: **96.37%**

---

### 03 — Phase 2 MCI Conversion Prediction
**Source:** `Untitled52_fixed.ipynb`  
Complete Phase 2 pipeline: 24-month conversion label construction with zero data leakage, 5-fold patient-level GroupShuffleSplit cross-validation, class-weighted training, ablation study (EXP-A/B/C), and SHAP feature importance.

**Key results:**
- Labelled MCI visits: **4,967** | Converters: **781 (15.7%)**
- Phase 2 AUC: **0.8253** | Sensitivity: **85.67%** at threshold 0.35
- SHAP ranking: MMSCORE > TOTAL13 > EDUCATION > AGE > APOE4 > SEX

---

### 04 — Site-Level External Holdout Validation
**Source:** `Untitled53.ipynb`  
Trains on 74 ADNI sites, evaluates on fully withheld site 130 (98 patients never seen during training). Three-class (CN vs MCI vs AD) task.

**Key results:**
- Training sites AUC: **0.8724**
- Holdout site AUC: **0.8329** (degradation of only 0.0395)

---

## Features Used

| Feature | Type | SHAP Rank |
|---|---|---|
| MMSCORE | Cognitive (MMSE score 0–30) | 1st (0.00834) |
| TOTAL13 | Cognitive (ADAS-Cog subscale) | 2nd (0.00530) |
| EDUCATION | Demographic (years of schooling) | 3rd (0.00255) |
| AGE | Demographic (computed per visit) | 4th (0.00209) |
| APOE4_carrier | Genetic (ε4 carrier status) | 5th (0.00142) |
| SEX_BIN | Demographic | 6th (0.00000) |

---

## Model Architecture

```
Phase 1 (CN vs AD):
BiGRUModel: GRU[64×2] → Linear(128→2) → CrossEntropyLoss
Validation: Patient-level 80/20 split (PTID-keyed, seed=42)

Phase 2 (MCI Conversion):
BiGRUClassifier: GRU[64×2] → BatchNorm1d(128) → Dropout(0.3) → Linear(128→1) → Sigmoid
Total params: 28,033
Validation: 5-fold GroupShuffleSplit (PTID-keyed, seed=42)
Loss: BCEWithLogitsLoss with inverse-frequency class weighting
```

---

## Requirements

```bash
pip install torch torchvision
pip install tensorflow
pip install scikit-learn pandas numpy matplotlib seaborn
pip install shap
pip install sympy==1.12
```

---

## Data Access

Data used in this study were obtained from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)** database (adni.loni.usc.edu).

**ADNI data is restricted** — to reproduce this work you must:
1. Register at [adni.loni.usc.edu](https://adni.loni.usc.edu)
2. Apply for data access
3. Download: `DXSUM_04Oct2025.csv`, `MMSE_04Oct2025.csv`, `ADAS_04Oct2025.csv`, `All_Subjects_PTDEMOG_04Oct2025.csv`, and APOE4 genotype file

**Raw data files are NOT included in this repository** in compliance with the ADNI data use agreement.

---

## Reproducibility

All notebooks use fixed random seeds for reproducibility:
```python
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## Citation

```
Kirti Kumari, Abhay Chaudhary, Mansi Butola, Khushi Krishali,
"A Dual-Phase Longitudinal Framework for Alzheimer's Disease Prediction:
Integrating Biological Irreversibility Filtering with Bidirectional GRUs,"
[Conference/Journal Name], 2025.
```

---

## Acknowledgement

Data used in preparation of this work were obtained from the ADNI database.
ADNI is funded by the National Institute on Aging (NIH Grant U01 AG024904),
DOD ADNI (award W81XWH-12-2-0012), and through contributions from private sector partners.

---

## Contact

Kirti Kumari — kirtikumaridoon@gmail.com  
Graphic Era Hill University, Dehradun, India
