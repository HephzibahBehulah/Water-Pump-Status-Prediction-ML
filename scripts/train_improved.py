"""
Improved training pipeline matching Sprint 2 baseline.
Uses proper feature engineering and preprocessing.

Run with: python scripts/train_improved.py
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
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils import setup_logger, get_config

logger = setup_logger(__name__, log_level="INFO", log_file="logs/training_improved.log")


def convert_to_str(X):
    """Convert columns to string."""
    return X.astype(str)


def load_and_clean_data(config):
    """Load and clean data with proper preprocessing."""
    logger.info("Loading data...")

    raw_dir = Path(config.get('data.raw_dir'))
    train_values_path = raw_dir / "training_values.csv"
    train_labels_path = raw_dir / "training_labels.csv"

    df_values = pd.read_csv(train_values_path)
    df_labels = pd.read_csv(train_labels_path)

    logger.info(f"Loaded {len(df_values)} records")

    # Merge
    df = df_values.merge(df_labels, on='id')

    # ========== FEATURE ENGINEERING ==========
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

    # Replace problematic zero values with NaN for imputation
    for col in ['gps_height', 'longitude', 'latitude', 'amount_tsh']:
        if col in df.columns:
            df.loc[df[col] == 0, col] = np.nan

    # Convert booleans to strings BEFORE imputation
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(str)

    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        if df[col].isna().sum() > 0:
            df[col].fillna('unknown', inplace=True)

    logger.info(f"Features after engineering: {df.shape[1]}")
    return df


def prepare_features(df):
    """Prepare features and target."""
    # Separate features and target
    X = df.drop(columns=['status_group']).copy()
    y = df['status_group']

    # Identify feature types
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    logger.info(f"Numeric features: {len(numeric_features)}")
    logger.info(f"Categorical features: {len(categorical_features)}")
    logger.info(f"Total features: {len(numeric_features) + len(categorical_features)}")

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    logger.info(f"Classes: {le.classes_.tolist()}")

    return X, y_encoded, numeric_features, categorical_features, le


def create_preprocessing_pipeline(numeric_features, categorical_features):
    """Create preprocessing pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    # Numeric: just pass through (RandomForest doesn't need scaling)
    numeric_transformer = 'passthrough'

    # Categorical: convert to string first, then ordinal encode
    categorical_transformer = Pipeline(steps=[
        ('to_str', FunctionTransformer(convert_to_str)),
        ('ordinal', OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ))
    ])

    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def main():
    """Main improved training pipeline."""
    logger.info("="*60)
    logger.info("IMPROVED TRAINING PIPELINE - WITH PROPER PREPROCESSING")
    logger.info("="*60)

    cfg = get_config()
    logger.info(f"Using {cfg.environment.upper()} configuration\n")

    try:
        # ========== STAGE 1: DATA LOADING & CLEANING ==========
        logger.info("[STAGE 1] Data Loading & Cleaning...")
        df = load_and_clean_data(cfg)

        # ========== STAGE 2: FEATURE PREPARATION ==========
        logger.info("\n[STAGE 2] Feature Preparation...")
        X, y, numeric_features, categorical_features, label_encoder = prepare_features(df)

        # ========== STAGE 3: TRAIN/VAL SPLIT ==========
        logger.info("\n[STAGE 3] Train/Validation Split...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

        # ========== STAGE 4: CREATE PIPELINE & TRAIN ==========
        logger.info("\n[STAGE 4] Training with MLflow...\n")

        mlflow.set_tracking_uri(cfg.get('mlflow.tracking_uri', 'http://localhost:5000'))
        experiment_name = cfg.get('mlflow.experiment_name', 'water-pump-development')
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="improved_random_forest"):
            # Create preprocessing pipeline
            preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)

            # Create full pipeline
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            )

            # Log parameters
            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("max_depth", 15)
            mlflow.log_param("min_samples_split", 5)
            mlflow.log_param("class_weight", "balanced")

            # Preprocess training data
            logger.info("Preprocessing data...")
            preprocessor.fit(X_train)
            X_train_processed = preprocessor.transform(X_train)
            X_val_processed = preprocessor.transform(X_val)

            # Train
            logger.info("Training Random Forest...")
            model.fit(X_train_processed, y_train)

            # Evaluate
            train_pred = model.predict(X_train_processed)
            val_pred = model.predict(X_val_processed)

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

            logger.info(f"Train Accuracy: {train_acc:.4f}")
            logger.info(f"Val Accuracy: {val_acc:.4f}")
            logger.info(f"Val Precision: {val_precision:.4f}")
            logger.info(f"Val Recall: {val_recall:.4f}")
            logger.info(f"Val F1: {val_f1:.4f}")

            # Save model and preprocessor
            logger.info("\nSaving model and preprocessor...")
            models_dir = Path("models/production")
            models_dir.mkdir(parents=True, exist_ok=True)

            import joblib
            joblib.dump(model, models_dir / "improved_rf_model.pkl")
            joblib.dump(preprocessor, models_dir / "preprocessor.pkl")
            joblib.dump(label_encoder, models_dir / "label_encoder.pkl")
            joblib.dump(numeric_features, models_dir / "numeric_features.pkl")
            joblib.dump(categorical_features, models_dir / "categorical_features.pkl")

            logger.info(f"Model saved to models/production/")

            # Save results
            results = {
                "model": "improved_random_forest",
                "train_accuracy": float(train_acc),
                "val_accuracy": float(val_acc),
                "val_precision": float(val_precision),
                "val_recall": float(val_recall),
                "val_f1": float(val_f1),
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
            }

            with open("metrics_improved.json", "w") as f:
                json.dump(results, f, indent=2)

        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"\nResults: metrics_improved.json")
        logger.info(f"MLflow: http://localhost:5000\n")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
