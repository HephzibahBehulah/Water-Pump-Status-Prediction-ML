# 💧 Water Pump Status Prediction — Hydro Dominion

> A machine learning project built as part of the **ReDI School Data Circle program** (Berlin, 2025).
> > Team: **Hydro Dominion** | Dataset: [DrivenData — Pump It Up](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/)
> >
> > ---
> >
> > ## Why I Built This
> >
> > Access to clean water is one of the most critical global challenges. Across Tanzania, thousands of water pumps serve rural communities — but many fail silently, leaving people without water for weeks or months. This project asks a straightforward question: **can we predict which pumps are likely to fail before they do?**
> >
> > Working as part of a two-person team (Team Hydro Dominion) through the ReDI School Data Circle course, we tackled this as a real-world, end-to-end ML project — from raw data exploration all the way to trained, serialized models ready for deployment.
> >
> > ---
> >
> > ## 🎯 Problem Statement
> >
> > Predict the **operational status** of 59,400 waterpoints across Tanzania into one of three classes:
> >
> > | Class | Share |
> > |---|---|
> > | Functional | 54.3% |
> > | Non-functional | 38.4% |
> > | Functional needs repair | 7.3% (minority class) |
> >
> > The severe class imbalance — especially the 7.3% minority class — was one of the core challenges we addressed.
> >
> > ---
> >
> > ## 🗂️ Project Structure
> >
> > ```
> > Water-Pump-Status-Prediction-ML/
> > ├── EDA/                               # Sprint 1: Exploratory analysis notebooks
> > ├── ML_Hydro_Dominion/                 # Sprint 1 team documentation
> > ├── Models/
> > │   ├── Finalized_model_Baseline.ipynb # Sprint 2: All models + tuning
> > │   ├── sprint_2_docx.md               # Sprint 2 write-up
> > │   ├── best_rf_tuned_pipeline.pkl     # Best model (tuned Random Forest)
> > │   └── rf_balanced_new_features_pipeline.pkl  # Balanced-weight RF
> > ├── Water_Pumps_Predictions_Sprint.../  # Prediction outputs
> > ├── Training Set Values.csv
> > ├── Training Set Labels.csv
> > └── Test Set Values.csv
> > ```
> >
> > ---
> >
> > ## 🔬 Sprint 1 — Exploratory Data Analysis ✅
> >
> > ### What We Did
> > - Merged 3 datasets (59,400 rows × 41 features) on the `id` column
> > - - Cleaned and validated data: handled 70% zero-rate in `amount_tsh`, extracted `pump_age` from construction year, standardized 2,000+ installer name variants into ~30 canonical groups
> >   - - Investigated 12 research hypotheses about what drives pump failure
> >    
> >     - ### Key Findings
> >     - - **Older pumps fail more** — `pump_age` is among the top predictive features
> >       - - **Geography matters** — failure rates vary significantly across Tanzania's regions and basins
> >         - - **Installer quality is measurable** — certain installers consistently produce better-performing pumps
> >           - - **Class imbalance is severe** — "functional needs repair" at just 7.3% requires special handling in modeling
> >            
> >             - ### Tools Used
> >             - Python · Pandas · NumPy · Matplotlib · Seaborn · Power BI · Git
> >            
> >             - ---
> >
> > ## 🤖 Sprint 2 — Model Development & Tuning ✅
> >
> > ### Feature Engineering
> > Engineered 21 features from raw data:
> >
> > **Numeric (11):** `gps_height`, `population`, `public_meeting`, `permit`, `pump_age`, `payment_binary`, `population_zero`, `log_population`, `gps_height_zero`, `is_dry`, `installer_freq`, `lga_freq`
> >
> > **Categorical (10):** `installer`, `basin`, `lga`, `extraction_type_group`, `management`, `water_quality`, `quantity`, `source`, `waterpoint_type`, `pump_age_bin`
> >
> > Key engineered features:
> > - `pump_age` — derived from `year_recorded - construction_year`
> > - - `log_population` — log-transformed to handle right skew
> >   - - `installer_freq` / `lga_freq` — frequency encoding for high-cardinality columns
> >     - - `is_dry` — binary flag for `quantity == 'dry'`
> >       - - `payment_binary` — simplified payment to pay/never-pay
> >        
> >         - ### Models Evaluated (6 Baselines)
> >        
> >         - | Model | Pipeline |
> >         - |---|---|
> >         - | Logistic Regression | Linear (scaled + one-hot) |
> > | K-Nearest Neighbors (k=5) | Linear (scaled + one-hot) |
> > | Decision Tree | Tree (ordinal encoded) |
> > | Random Forest (200 estimators) | Tree |
> > | Gradient Boosting (100 estimators) | Tree |
> > | AdaBoost (100 estimators) | Tree |
> >
> > Used **SelectKBest** with mutual information scoring to select optimal feature subsets per model.
> >
> > ### Hyperparameter Tuning
> > - Ran **RandomizedSearchCV** (20 iterations, 5-fold CV) on both Random Forest and Gradient Boosting
> > - - Tuned: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `learning_rate`, `subsample`
> >   - - Best model: **Tuned Random Forest**
> >    
> >     - ### Class Imbalance Handling
> >     - - Trained a `class_weight='balanced'` variant of the Random Forest
> >       - - Compared confusion matrices and classification reports with/without balanced weights
> >         - - Balanced model improves **recall for the minority class** ("functional needs repair") at the cost of modest overall accuracy
> >          
> >           - ### Data Quality Checks
> >           - - Stratified 80/20 train/validation split to preserve class proportions (47,520 train / 11,880 validation)
> >             - - Median imputation fitted on training set only — **no data leakage**
> >               - - Learning curve analysis to diagnose overfitting vs. underfitting
> >                 - - 5-fold cross-validation accuracy compared against train and validation accuracy
> >                  
> >                   - ### Saved Artifacts
> >                   - | File | Description |
> >                   - |---|---|
> >                   - | `best_rf_tuned_pipeline.pkl` | Tuned Random Forest (RandomizedSearchCV best estimator) |
> >                   - | `rf_balanced_new_features_pipeline.pkl` | Balanced-weight RF for minority class recall |
> >                  
> >                   - ---
> >
> > ## 🔭 Sprint 3 — In Progress 🚧
> >
> > Planned next steps:
> > - Experiment with XGBoost with more aggressive tuning
> > - - Apply SMOTE oversampling for the minority class
> >   - - Reintroduce geographic coordinates (latitude/longitude) with proper encoding
> >     - - Investigate ensemble stacking of the best models
> >      
> >       - ---
> >
> > ## 🛠️ Tech Stack
> >
> > | Category | Tools |
> > |---|---|
> > | Language | Python 3 |
> > | Data Processing | Pandas, NumPy |
> > | Visualization | Matplotlib, Seaborn, Power BI |
> > | ML Framework | scikit-learn |
> > | Boosting | XGBoost |
> > | Model Serialization | joblib |
> > | Collaboration | Git, GitHub |
> >
> > ---
> >
> > ## 👥 Team
> >
> > **Hydro Dominion** — built as part of the ReDI School Berlin Data Circle course (March–June 2025)
> >
> > - [Vaibhav Koneti](https://github.com/Vaibhavkoneti)
> > - - [HephzibahBehulah](https://github.com/HephzibahBehulah)
