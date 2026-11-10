# Chicago Traffic Crashes: Predicting Multiple-Injury Incidents

**ST 597 Final Project — Protocol Checkpoint & Data Pipeline**

## 📌 Project Overview

Traffic crashes involving multiple injuries account for a significant portion of overall traffic harm in Chicago. Although multiple-injury crashes constitute only **4.14%** of total reported collisions between 2023 and 2025 (13,707 out of 331,203 cleaned observations), they represent nearly half of all recorded injuries across the city.

The primary objective of this project is to assist the **Chicago Department of Transportation (CDOT)** by evaluating whether pre-crash roadway, environmental, and temporal features can reliably identify high-risk crash profiles.

Rather than focusing on causal inference or post-investigation factors, this repository establishes a leakage-free machine learning pipeline designed to compare **Logistic Regression**, **Random Forest**, and **Gradient Boosting** against a naive **Majority-Class Baseline**.

---

## 📁 Repository Structure

```text
├── data/
│   ├── raw/
│   │   └── Traffic_Crashes_-_Crashes_20261013.csv       # Original Chicago Open Data download
│   └── processed/
│       ├── chicago_crashes_2023_2025.csv               # Cleaned 3-year subset
│       ├── train_2023_2024.csv                         # Training set (2023–2024; n = 222,301)
│       ├── test_2025_DO_NOT_USE_FOR_TUNING.csv          # Held-out test set (2025; n = 108,902)
│       └── train_fold_assignments.csv                  # Grouped 5-fold CV mapping
├── notebooks/
│   ├── 01_preliminary_exploration.ipynb                # Initial data exploration & scope design
│   └── 02_protocol_checkpoint_pipeline.ipynb           # Data cleaning, leakage audit, CV split & baseline
├── reports/
│   ├── data_dictionary.csv                             # Complete predictor roles and specs
│   ├── excluded_variables.csv                          # Audit log of dropped post-crash fields
│   ├── baseline_grouped_cv.csv                         # Majority-class baseline performance by fold
│   ├── baseline_summary.csv                            # Averaged baseline metrics across 5 folds
│   └── model_protocol_config.json                      # Saved pipeline & hyperparameter grid configs
├── README.md                                           # Project documentation
└── requirements.txt                                    # Python dependencies
```

---

## 📓 Notebook Guide

### `notebooks/01_preliminary_exploration.ipynb`
* **Purpose:** Initial exploratory data analysis (EDA) conducted to evaluate dataset feasibility and establish project scope for the application.
* **Key Tasks:**
  * Analyzed target class imbalance (~4.13% positive rate).
  * Evaluated missingness across administrative and physical road variables (identified `LANE_CNT` as unusable due to >99.9% missingness).
  * Formulated the binary target `multiple_injuries` ($\text{INJURIES\_TOTAL} \ge 2$).

### `notebooks/02_protocol_checkpoint_pipeline.ipynb`
* **Purpose:** The core reproducible pipeline for data ingestion, cleaning, temporal splitting, grouped cross-validation setup, and baseline metric calculation.
* **Key Tasks:**
  1. **Filtering & Cleaning:** Subsetted records for 2023–2025 and extracted 16 pre-crash predictor features.
  2. **Data Leakage Audit:** Removed 11 variables recorded post-incident (e.g., `PRIM_CONTRIBUTORY_CAUSE`, detailed injury counts).
  3. **Temporal Train/Test Split:** Partitioned data into 2023–2024 Training ($n = 222,301$) and held-out 2025 Testing ($n = 108,902$).
  4. **Grouped 5-Fold Cross-Validation:** Built `GroupKFold` folds grouped by `STREET_NAME` (Seed `597`) to prevent spatial leakage across neighboring incidents.
  5. **Baseline Benchmark:** Evaluated a `DummyClassifier` (majority class) across CV folds to set benchmark metrics.
  6. **Model Protocol Configuration:** Pre-defined feature pipelines and hyperparameter grids for Logistic Regression, Random Forest, and Gradient Boosting (saved to `reports/model_protocol_config.json`).

---

## 🛠️ Data Pipeline & Methodology Summary

* **Target Variable (`multiple_injuries`):**
  * `1` = Crash resulted in 2 or more injured persons ($\text{INJURIES\_TOTAL} \ge 2$).
  * `0` = Crash resulted in 0 or 1 injured person.
* **Primary Evaluation Metric:** **Average Precision (PR-AUC)**. Standard accuracy is discarded due to the ~95.91% majority class, which causes naive classifiers to yield zero recall and F1-score.
* **Missing Value Strategy:** Categorical missingness in work zones (`WORK_ZONE_I`, `WORKERS_PRESENT_I`) and intersections (`INTERSECTION_RELATED_I`) is retained as an explicit `"Unknown"` category to preserve administrative reporting patterns.
* **Spatial Grouping:** 5-fold CV is strictly grouped by `STREET_NAME` to ensure crashes along the same roadway corridor remain within the same partition.

---

## 🚀 Instructions for Running the Pipeline

### 1. Environment Setup

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/chicago-crash-prediction.git
cd chicago-crash-prediction
pip install -r requirements.txt
```

### 2. Data Placement

Ensure the raw Chicago Traffic Crashes CSV file is placed inside `data/raw/`:

```text
data/raw/Traffic_Crashes_-_Crashes_20261013.csv
```

### 3. Running Notebooks

Launch Jupyter Lab to run the pipeline:

```bash
jupyter lab
```

Execute `notebooks/02_protocol_checkpoint_pipeline.ipynb`. The script will automatically verify whether processed files exist; if missing, it will generate all clean datasets, fold assignments, and reports inside `data/processed/` and `reports/`.

---

## 📊 Summary of Baseline Benchmark

| Metric | Fold Mean | Std Dev | Notes |
| :--- | :---: | :---: | :--- |
| **Accuracy** | 95.91% | ±0.22% | High accuracy due to ~95.9% majority class |
| **Precision** | 0.0000 | ±0.0000 | Fails to predict positive class |
| **Recall** | 0.0000 | ±0.0000 | Never identifies multiple-injury crashes |
| **F1-Score** | 0.0000 | ±0.0000 | Baseline performance target |
| **PR-AUC (Average Precision)** | 0.0409 | ±0.0022 | Benchmark baseline to outperform |
| **ROC-AUC** | 0.5000 | ±0.0000 | Equivalent to random guessing |

---

## 🔒 Data Leakage Audit Rules

To guarantee valid out-of-sample predictions, the following 11 variables are **permanently excluded** from modeling predictors:

* `INJURIES_TOTAL`
* `INJURIES_FATAL`
* `INJURIES_INCAPACITATING`
* `INJURIES_NON_INCAPACITATING`
* `INJURIES_REPORTED_NOT_EVIDENT`
* `INJURIES_NO_INDICATION`
* `INJURIES_UNKNOWN`
* `MOST_SEVERE_INJURY`
* `CRASH_TYPE`
* `PRIM_CONTRIBUTORY_CAUSE`
* `SEC_CONTRIBUTORY_CAUSE`

## Steps for pushing and pulling!!!

1. BEFORE changing anything or opening anything: use: "git pull" to grab the changes that were made before you

2. IF you MAKE changes, make sure to do "git status" and do what it says to commit your changes

3. ONLY THEN do you do "git push" to send up the changes you made so others can pull it



