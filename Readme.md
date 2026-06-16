# 💧 Water Pump Status Prediction

**Team Hydro Dominion** · ReDI School Data Circle, Berlin 2025  
Dataset: [DrivenData — Pump It Up](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/)

---

## Why we built this

Millions of people in rural Tanzania depend on water pumps that break down without warning. By the time anyone notices, entire communities can go weeks without clean water.

We wanted to know: what if you could predict which pumps are about to fail — before they do?

That question drove this project. Over three sprints, my teammate and I went from raw, messy data to trained models that can classify the operational status of 59,400 waterpoints across Tanzania. It's not a toy problem — the data is real, the class imbalance is brutal, and the stakes feel meaningful.

---

## The problem

Each waterpoint falls into one of three states:

| Status | Share |
|---|---|
| Functional | 54.3% |
| Non-functional | 38.4% |
| Functional needs repair | 7.3% |

That 7.3% minority class — pumps that are deteriorating but not yet broken — is the hardest and most important thing to catch. A lot of our effort went into not letting the model ignore it.

---

## Project structure

```
Water-Pump-Status-Prediction-ML/
├── EDA/                                         # Sprint 1: exploration notebooks
├── ML_Hydro_Dominion/                           # Sprint 1 team notes
├── Models/
│   ├── Finalized_model_Baseline.ipynb           # All models + tuning
│   ├── sprint_2_docx.md                         # Sprint 2 write-up
│   ├── best_rf_tuned_pipeline.pkl               # Best model (tuned Random Forest)
│   └── rf_balanced_new_features_pipeline.pkl    # Balanced RF for minority recall
├── Water_Pumps_Predictions_Sprint.../           # Prediction outputs
├── Training Set Values.csv
├── Training Set Labels.csv
└── Test Set Values.csv
```

---

## Sprint 1 — Exploring the data ✅

We started by merging three datasets (59,400 rows, 41 features) and quickly found that the raw data was a mess. Installer names alone had 2,000+ variants — "Government", "GOVT", "govt of tanzania", and so on — all meaning the same thing. We collapsed those into ~30 clean groups.

Some of what we found along the way:

- **Age matters a lot.** Older pumps fail more, and `pump_age` became one of our strongest features.
- **Where a pump is tells you a lot.** Failure rates differ significantly across regions and river basins.
- **Who installed it matters too.** Some installers have a track record of pumps that last; others don't.
- **That minority class is going to be a problem.** 7.3% is not a lot of training signal for something this important.

**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Power BI, Git

---

## Sprint 2 — Building and tuning models ✅

### Features we engineered

We ended up with 21 features, built from the raw columns:

**Numeric (11):** `gps_height`, `population`, `public_meeting`, `permit`, `pump_age`, `payment_binary`, `population_zero`, `log_population`, `gps_height_zero`, `is_dry`, `installer_freq`, `lga_freq`

**Categorical (10):** `installer`, `basin`, `lga`, `extraction_type_group`, `management`, `water_quality`, `quantity`, `source`, `waterpoint_type`, `pump_age_bin`

A few we're particularly happy with:
- `pump_age` — simple subtraction (`year_recorded - construction_year`), but surprisingly powerful
- `is_dry` — a binary flag for whether the water source is dry; turns out that's quite predictive
- `installer_freq` / `lga_freq` — frequency encoding instead of one-hot for high-cardinality columns, which kept the feature space manageable

### What we tried

We ran six baseline models, from Logistic Regression to AdaBoost, using `SelectKBest` with mutual information to pick the right features for each one.

| Model | Pipeline |
|---|---|
| Logistic Regression | Linear (scaled + one-hot) |
| K-Nearest Neighbors (k=5) | Linear (scaled + one-hot) |
| Decision Tree | Tree (ordinal encoded) |
| Random Forest (200 estimators) | Tree |
| Gradient Boosting (100 estimators) | Tree |
| AdaBoost (100 estimators) | Tree |

Random Forest pulled ahead early and we doubled down on it — running RandomizedSearchCV across 20 iterations with 5-fold cross-validation, tuning depth, estimators, split criteria, and more.

### Dealing with the class imbalance

We trained a second Random Forest with `class_weight='balanced'` and compared it head-to-head with the tuned version. The balanced model catches more "needs repair" pumps (better minority recall), but pays a small price in overall accuracy. Both are saved — which one you'd deploy depends on whether you'd rather miss a failing pump or flag a healthy one.

### Being careful about leakage

Stratified 80/20 split, median imputation fitted only on training data, learning curves to check for overfitting. We were paranoid about this stuff.

### Saved models

| File | What it is |
|---|---|
| `best_rf_tuned_pipeline.pkl` | Best overall accuracy (tuned RF) |
| `rf_balanced_new_features_pipeline.pkl` | Better minority class recall |

---

## 🔭 Sprint 3 — Insights & Deployment ✅

Sprint 3 was about making the model actually useful — not just accurate on paper, but something a field engineer could open and act on.

We used SHAP to explain what the model is doing under the hood. The biggest finding: **water quantity is the single strongest predictor of failure**. A dry source alone shifts the probability of a pump being non-functional by 0.4. GPS location, extraction type, management, and pump age round out the top five. Pump age is interesting — it's gradual until it isn't. Failure spikes sharply after 50 years.

A few other things the data confirmed: gravity and mono pumps outlast hand and rope pumps by a significant margin, NGO-managed hand pumps fail roughly twice as often as water-board-managed gravity pumps, and population served barely correlates with failure at all.

On the modelling side, we landed on **XGBoost + SMOTE** as the stronger final model — it generalizes better (88.4% train / 78.6% val) with a 0.69 macro F1. Random Forest + SMOTE overfits more (99.5% train) but has slightly better repair precision. We kept both, because running them together flags borderline pumps that neither catches alone.

For MLOps, we logged 24 runs across 7 model configs in MLflow and versioned all large files through DVC — so every result is reproducible and traceable.

The final deliverable is a **7-page Streamlit dashboard** built for field teams and government authorities. You can check a single pump or run batch analysis, and every prediction comes with a SHAP explanation so it's never just a black box.

---

## Tech stack

Python 3 · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn · XGBoost · joblib · Git

---

## The team

Built by **Hydro Dominion** as part of the ReDI School Berlin Data Circle (March–June 2025):

- [Vaibhav Koneti](https://github.com/Vaibhavkoneti)
- [HephzibahBehulah](https://github.com/HephzibahBehulah)
