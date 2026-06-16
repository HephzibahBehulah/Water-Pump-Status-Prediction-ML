# Sprint 2 — Baseline Model Development

## Project Overview

This sprint focuses on building and evaluating baseline classification models to predict the operational status of waterpoints (water pumps) in Tanzania. The target variable `status_group` has three classes: **functional**, **non functional**, and **functional needs repair**. The dataset comes from the Taarifa waterpoints platform and is part of the DrivenData "Pump it Up" competition.

The training data consists of 59,400 records with 40 features describing each waterpoint's characteristics — including geographic location, management details, water source type, extraction method, and more.

---

## Exploratory Data Analysis

Three CSV files (training values, training labels, test values) are loaded and merged on the `id` column into a single working dataframe of shape (59400, 41). Key findings from the initial exploration:

**Class imbalance** — The target distribution is roughly 54.3% functional, 38.4% non functional, and only 7.3% functional needs repair. This heavy minority class guided later decisions around stratified splitting and class-weighted models.

**Missing and zero values** — Several numeric columns contain suspicious zeros used as placeholders for missing data: `amount_tsh` (70% zeros), `gps_height`, `longitude`, and `population` all have significant zero rates. True nulls exist in `permit`, `public_meeting`, and `scheme_name` (48% missing).

**High cardinality** — Columns like `wpt_name` (37k unique), `subvillage` (19k), `installer` (2k), and `funder` (1.9k) have far too many categories to use directly. Additionally, `recorded_by` has only 1 unique value, making it useless.

**Redundant feature groups** — The dataset contains grouped versions of several features (e.g., `extraction_type` / `extraction_type_group` / `extraction_type_class`). Only the mid-level grouping is retained to balance cardinality and information content.

---

## Feature Engineering & Cleaning

### Column Removal

Seventeen columns are dropped in the first pass, falling into four categories:

- **Useless**: `id`, `recorded_by` (single value)
- **Ultra-high cardinality**: `wpt_name`, `subvillage`, `scheme_name`
- **Redundant groupings**: `extraction_type`, `extraction_type_class`, `management_group`, `payment_type`, `quality_group`, `quantity_group`, `source_type`, `source_class`, `waterpoint_type_group`
- **Geospatial (deferred for baseline)**: `longitude`, `latitude`, `amount_tsh`

Additional columns removed after further analysis: `funder` (limited value beyond installer), `pump_age_missing`, `payment`, `district_code`, `region_code`, `num_private`, `region`, `ward`, and `scheme_management`.

### Installer Consolidation

The `installer` column (2,000+ unique values) is cleaned through text standardization (lowercase, strip whitespace, remove special characters), manual grouping of known misspellings and aliases into ~30 canonical groups (e.g., "government", "gover", "gove" all map to `government`), and bucketing rare remaining values into `other`.

### Pump Age

A `pump_age` feature is derived from `year_recorded - construction_year`. Records with `construction_year == 0` are treated as missing, and any negative ages (data errors) are set to NaN. The raw date columns are dropped after derivation.

### Binary Features

- `permit` and `public_meeting` are mapped from True/False to 1/0
- `payment_binary` is created: 0 if `payment == 'never pay'`, 1 otherwise

### Engineered Features

Several additional features are created to capture non-linear signals:

- `population_zero` and `gps_height_zero` — binary flags for zero-valued records
- `log_population` — log-transformed population to handle skew
- `pump_age_bin` — pump age bucketed into new / mid / old / very_old
- `is_dry` — binary flag for `quantity == 'dry'`
- `installer_freq` and `lga_freq` — frequency encoding of high-cardinality categorical columns

---

## Train/Validation Split

The data is split 80/20 using stratified sampling on the target variable to preserve class proportions, yielding 47,520 training samples and 11,880 validation samples.

---

## Missing Value Handling

For `pump_age`, median imputation is applied using only the training set median to prevent data leakage. Remaining numeric columns use median imputation and categorical columns use most-frequent imputation, both handled within the preprocessing pipelines.

---

## Preprocessing Pipelines

Two preprocessing pipelines are built for different model families:

**Linear pipeline** — For logistic regression and KNN. Applies log transformation to `population` (highly skewed), standard scaling to all numeric features, and one-hot encoding to categorical features.

**Tree pipeline** — For tree-based models. Applies median imputation to numeric features and ordinal encoding (with unknown handling) to categorical features. No scaling needed since trees are scale-invariant.

---

## Baseline Model Comparison

Six model families are evaluated across a range of feature counts using `SelectKBest` with mutual information scoring:

| Model | Preprocessing | Notes |
|---|---|---|
| Decision Tree | Tree | Single tree, default hyperparameters |
| Random Forest | Tree | 200 estimators |
| Gradient Boosting | Tree | 100 estimators |
| AdaBoost | Tree | 100 estimators |
| Logistic Regression | Linear | max_iter=1000 |
| KNN | Linear | k=5 neighbors |

Each model is trained with varying feature counts (k=3 to total features), and the best k per model is selected by validation accuracy. **Random Forest consistently achieves the highest validation accuracy.**

---

## Feature Importance

Mutual information scores from SelectKBest reveal the most informative features. The top contributors include `quantity`, `waterpoint_type`, `extraction_type_group`, `water_quality`, and `pump_age` — all of which have intuitive domain relevance to pump functionality.

---

## Hyperparameter Tuning

### Random Forest

A `RandomizedSearchCV` with 20 iterations and 5-fold cross-validation is run over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`. The best model is saved as `best_rf_tuned_pipeline.pkl`.

### Gradient Boosting

A separate `RandomizedSearchCV` (20 iterations, 5-fold CV) tunes the Gradient Boosting model over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `min_samples_split`, and `min_samples_leaf`. Validation accuracy and classification reports are compared head-to-head with the Random Forest.

---

## Overfitting Analysis

The tuned Random Forest is evaluated for overfitting by comparing training accuracy, 5-fold CV accuracy, and validation accuracy. A learning curve analysis is also performed, plotting training and validation accuracy as a function of dataset size (20% to 100%) to diagnose whether more data or stronger regularization would help.

---

## Class Imbalance Handling

A `class_weight='balanced'` variant of the Random Forest is trained to address the severe underrepresentation of the "functional needs repair" class (only 7.3% of samples). This reweights the loss function inversely proportional to class frequency, trading some overall accuracy for improved recall on the minority class. Confusion matrices and classification reports are compared with and without balanced weights.

---

## Final Feature Set

**Numeric (11):** `gps_height`, `population`, `public_meeting`, `permit`, `pump_age`, `payment_binary`, `population_zero`, `log_population`, `gps_height_zero`, `is_dry`, `installer_freq`, `lga_freq`

**Categorical (10):** `installer`, `basin`, `lga`, `extraction_type_group`, `management`, `water_quality`, `quantity`, `source`, `waterpoint_type`, `pump_age_bin`

---

## Saved Artifacts

| File | Description |
|---|---|
| `best_rf_tuned_pipeline.pkl` | Tuned Random Forest pipeline (RandomizedSearchCV best) |
| `rf_balanced_new_features_pipeline.pkl` | Balanced-weight RF with engineered features |

---

## Key Takeaways & Next Steps

**What worked well:** Random Forest proved to be the strongest baseline. Feature engineering — especially pump age, payment binary, and frequency encoding — provided measurable gains. Class-weighted training improved minority-class recall.

**Remaining challenges:** The "functional needs repair" class remains difficult to predict due to its small size and overlapping characteristics with both other classes. Balanced weights help recall but reduce precision.

**Potential next steps:** Explore XGBoost with more aggressive tuning, try SMOTE or other oversampling for the minority class, reintroduce geographic features (latitude/longitude) with proper encoding, and investigate ensemble stacking of the best models.

---

## Tools & Libraries

- **Python 3** with pandas, numpy, matplotlib, seaborn
- **scikit-learn** for preprocessing, model training, evaluation, and hyperparameter search
- **XGBoost** (imported but primarily used Random Forest and Gradient Boosting from sklearn)
- **joblib** for model serialization
