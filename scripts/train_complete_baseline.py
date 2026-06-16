"""
Complete Baseline Training - Matching Notebook Approach
Uses: SelectKBest, RandomizedSearchCV, Cross-Validation, Proper Preprocessing

Run: python scripts/train_complete_baseline.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from src.utils import setup_logger, get_config

logger = setup_logger(__name__, log_level="INFO", log_file="logs/training_complete.log")


def convert_to_str(X):
    """Convert to string for encoding."""
    return X.astype(str)


def load_and_engineer_data(config):
    """Load, clean, and engineer data."""
    logger.info("Loading data...")

    raw_dir = Path(config.get('data.raw_dir'))
    train_values_path = raw_dir / "training_values.csv"
    train_labels_path = raw_dir / "training_labels.csv"

    df_values = pd.read_csv(train_values_path)
    df_labels = pd.read_csv(train_labels_path)

    logger.info(f"Loaded {len(df_values)} records, {len(df_values.columns)} features")

    # Merge
    df = df_values.merge(df_labels, on='id')

    # ===== FEATURE ENGINEERING =====
    logger.info("Feature engineering...")

    # Drop unnecessary columns
    drop_cols = ['id', 'recorded_by', 'scheme_name', 'wpt_name', 'subvillage']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Date features
    if 'date_recorded' in df.columns:
        df['date_recorded'] = pd.to_datetime(df['date_recorded'])
        df['year_recorded'] = df['date_recorded'].dt.year
        df['month_recorded'] = df['date_recorded'].dt.month
        df = df.drop(columns=['date_recorded'])

    # Pump age
    if 'construction_year' in df.columns:
        df['pump_age'] = df['year_recorded'] - df['construction_year']
        df['pump_age'] = df['pump_age'].fillna(df['pump_age'].median())
        df['pump_age'] = df['pump_age'].clip(lower=0)
        df = df.drop(columns=['construction_year', 'year_recorded'])

    # Handle zero values (placeholders for missing)
    for col in ['gps_height', 'longitude', 'latitude', 'amount_tsh']:
        if col in df.columns:
            df.loc[df[col] == 0, col] = np.nan

    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        if df[col].isna().sum() > 0:
            df[col].fillna('unknown', inplace=True)

    # Convert booleans to strings
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(str)

    logger.info(f"After engineering: {df.shape}")
    return df


def prepare_features(df):
    """Separate features and target."""
    X = df.drop(columns=['status_group']).copy()
    y = df['status_group']

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    logger.info(f"Features: {len(numeric_features)} numeric + {len(categorical_features)} categorical")
    logger.info(f"Classes: {le.classes_.tolist()}")

    return X, y_encoded, numeric_features, categorical_features, le


def create_preprocessing_pipeline(numeric_features, categorical_features):
    """Create preprocessing pipeline."""

    numeric_transformer = 'passthrough'

    categorical_transformer = Pipeline(steps=[
        ('to_str', FunctionTransformer(convert_to_str)),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def main():
    """Complete baseline training with all techniques."""
    logger.info("="*60)
    logger.info("COMPLETE BASELINE TRAINING - WITH TUNING & FEATURE SELECTION")
    logger.info("="*60)

    cfg = get_config()
    logger.info(f"Using {cfg.environment.upper()} configuration\n")

    try:
        # ===== STAGE 1: DATA LOADING =====
        logger.info("[STAGE 1] Data Loading & Engineering...")
        df = load_and_engineer_data(cfg)

        # ===== STAGE 2: FEATURE PREPARATION =====
        logger.info("\n[STAGE 2] Feature Preparation...")
        X, y, numeric_features, categorical_features, label_encoder = prepare_features(df)

        # ===== STAGE 3: TRAIN/VAL SPLIT =====
        logger.info("\n[STAGE 3] Train/Validation Split...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

        # ===== STAGE 4: PREPROCESSING =====
        logger.info("\n[STAGE 4] Preprocessing Data...")
        preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
        preprocessor.fit(X_train)
        X_train_proc = preprocessor.transform(X_train)
        X_val_proc = preprocessor.transform(X_val)

        # ===== STAGE 5: FEATURE SELECTION =====
        logger.info("\n[STAGE 5] Feature Selection with SelectKBest...")
        selector = SelectKBest(score_func=mutual_info_classif, k='all')
        X_train_selected = selector.fit_transform(X_train_proc, y_train)
        X_val_selected = selector.transform(X_val_proc)

        # Get top features
        feature_scores = selector.scores_
        top_k = min(15, len(feature_scores))  # Use top 15 features
        top_indices = np.argsort(feature_scores)[-top_k:]

        logger.info(f"Using top {top_k} features out of {len(feature_scores)}")

        X_train_selected = X_train_selected[:, top_indices]
        X_val_selected = X_val_selected[:, top_indices]

        # ===== STAGE 6: HYPERPARAMETER TUNING =====
        logger.info("\n[STAGE 6] Hyperparameter Tuning with RandomizedSearchCV...")

        mlflow.set_tracking_uri(cfg.get('mlflow.tracking_uri', 'http://localhost:5000'))
        mlflow.set_experiment('water-pump-development')

        with mlflow.start_run(run_name="complete_baseline_with_tuning"):
            param_dist = {
                'n_estimators': [100, 150, 200, 250],
                'max_depth': [10, 15, 20, 25],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2'],
            }

            rf = RandomForestClassifier(
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            )

            # Randomized search
            logger.info("Running RandomizedSearchCV (20 iterations)...")
            random_search = RandomizedSearchCV(
                rf,
                param_distributions=param_dist,
                n_iter=20,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )

            random_search.fit(X_train_selected, y_train)
            best_model = random_search.best_estimator_

            logger.info(f"Best parameters: {random_search.best_params_}")
            logger.info(f"Best CV score: {random_search.best_score_:.4f}")

            # Log best params
            for param, value in random_search.best_params_.items():
                mlflow.log_param(param, value)

            # ===== STAGE 7: EVALUATION =====
            logger.info("\n[STAGE 7] Model Evaluation...")

            # Cross-validation score
            cv_scores = cross_val_score(best_model, X_train_selected, y_train, cv=5, scoring='accuracy')
            logger.info(f"5-Fold CV Scores: {cv_scores}")
            logger.info(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

            # Predictions
            train_pred = best_model.predict(X_train_selected)
            val_pred = best_model.predict(X_val_selected)

            # Metrics
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            val_precision = precision_score(y_val, val_pred, average='weighted', zero_division=0)
            val_recall = recall_score(y_val, val_pred, average='weighted', zero_division=0)
            val_f1 = f1_score(y_val, val_pred, average='weighted', zero_division=0)

            # Macro metrics (important for imbalanced data)
            val_precision_macro = precision_score(y_val, val_pred, average='macro', zero_division=0)
            val_recall_macro = recall_score(y_val, val_pred, average='macro', zero_division=0)
            val_f1_macro = f1_score(y_val, val_pred, average='macro', zero_division=0)

            # Log metrics
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("val_accuracy", val_acc)
            mlflow.log_metric("val_precision_weighted", val_precision)
            mlflow.log_metric("val_recall_weighted", val_recall)
            mlflow.log_metric("val_f1_weighted", val_f1)
            mlflow.log_metric("val_precision_macro", val_precision_macro)
            mlflow.log_metric("val_recall_macro", val_recall_macro)
            mlflow.log_metric("val_f1_macro", val_f1_macro)
            mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())

            logger.info(f"\nTrain Accuracy: {train_acc:.4f}")
            logger.info(f"Val Accuracy: {val_acc:.4f}")
            logger.info(f"Val Precision: {val_precision:.4f}")
            logger.info(f"Val Recall: {val_recall:.4f}")
            logger.info(f"Val F1: {val_f1:.4f}")

            # Classification report
            logger.info("\nClassification Report:")
            logger.info(classification_report(y_val, val_pred, target_names=label_encoder.classes_))

            # Confusion matrix
            cm = confusion_matrix(y_val, val_pred)
            logger.info(f"Confusion Matrix:\n{cm}")

            # ===== STAGE 8: SAVE ARTIFACTS =====
            logger.info("\n[STAGE 8] Saving Artifacts...")

            models_dir = Path("models/production")
            models_dir.mkdir(parents=True, exist_ok=True)

            import joblib
            joblib.dump(best_model, models_dir / "complete_baseline_rf.pkl")
            joblib.dump(preprocessor, models_dir / "baseline_preprocessor.pkl")
            joblib.dump(selector, models_dir / "baseline_selector.pkl")
            joblib.dump(label_encoder, models_dir / "label_encoder.pkl")

            logger.info("Models saved!")

            # ===== SAVE RESULTS =====
            results = {
                "model": "random_forest_complete_baseline",
                "best_params": random_search.best_params_,
                "cv_mean_accuracy": float(cv_scores.mean()),
                "cv_std_accuracy": float(cv_scores.std()),
                "train_accuracy": float(train_acc),
                "val_accuracy": float(val_acc),
                "val_precision": float(val_precision),
                "val_recall": float(val_recall),
                "val_f1": float(val_f1),
                "top_features_count": top_k,
                "feature_selection_method": "SelectKBest with mutual_info_classif"
            }

            with open("metrics_complete_baseline.json", "w") as f:
                json.dump(results, f, indent=2)

        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETED!")
        logger.info("="*60)
        logger.info(f"\nResults saved to: metrics_complete_baseline.json")
        logger.info(f"Models saved to: models/production/")
        logger.info(f"MLflow: http://localhost:5000\n")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
