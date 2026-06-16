"""
SMOTE + XGBoost Training - Handle Class Imbalance

Applies SMOTE to balance the training data, then trains XGBoost
This helps the model learn the minority class better.

Run: python scripts/train_smote_xgboost.py
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from src.utils import setup_logger, get_config

logger = setup_logger(__name__, log_level="INFO", log_file="logs/training_smote.log")


def convert_to_str(X):
    return X.astype(str)


def load_and_engineer_data(config):
    logger.info("Loading data...")
    raw_dir = Path(config.get('data.raw_dir'))
    df_values = pd.read_csv(raw_dir / "training_values.csv")
    df_labels = pd.read_csv(raw_dir / "training_labels.csv")

    df = df_values.merge(df_labels, on='id')
    logger.info(f"Loaded {len(df)} records")

    # Feature engineering
    drop_cols = ['id', 'recorded_by', 'scheme_name', 'wpt_name', 'subvillage']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    if 'date_recorded' in df.columns:
        df['date_recorded'] = pd.to_datetime(df['date_recorded'])
        df['year_recorded'] = df['date_recorded'].dt.year
        df['month_recorded'] = df['date_recorded'].dt.month
        df = df.drop(columns=['date_recorded'])

    if 'construction_year' in df.columns:
        df['pump_age'] = df['year_recorded'] - df['construction_year']
        df['pump_age'] = df['pump_age'].fillna(df['pump_age'].median()).clip(lower=0)
        df = df.drop(columns=['construction_year', 'year_recorded'])

    for col in ['gps_height', 'longitude', 'latitude', 'amount_tsh']:
        if col in df.columns:
            df.loc[df[col] == 0, col] = np.nan

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        if df[col].isna().sum() > 0:
            df[col].fillna('unknown', inplace=True)

    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(str)

    return df


def prepare_data(df):
    X = df.drop(columns=['status_group']).copy()
    y = df['status_group']

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Check class distribution BEFORE split
    unique, counts = np.unique(y_encoded, return_counts=True)
    logger.info("Class distribution BEFORE split:")
    for cls, count in zip(unique, counts):
        logger.info(f"  Class {le.classes_[cls]}: {count} ({count/len(y)*100:.1f}%)")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    return X_train, X_val, y_train, y_val, numeric_features, categorical_features, le


def preprocess_and_select_features(X_train, X_val, y_train, numeric_features, categorical_features):
    logger.info("Preprocessing data...")

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

    preprocessor.fit(X_train)
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    logger.info("Feature selection with SelectKBest...")
    selector = SelectKBest(score_func=mutual_info_classif, k='all')
    X_train_sel = selector.fit_transform(X_train_proc, y_train)
    X_val_sel = selector.transform(X_val_proc)

    top_k = min(15, len(selector.scores_))
    top_indices = np.argsort(selector.scores_)[-top_k:]
    X_train_sel = X_train_sel[:, top_indices]
    X_val_sel = X_val_sel[:, top_indices]

    logger.info(f"Using top {top_k} features")
    return X_train_sel, X_val_sel


def apply_smote(X_train, y_train):
    """Apply SMOTE to balance training data"""
    logger.info("\n[SMOTE] Applying SMOTE to balance training data...")

    # Check class distribution before SMOTE
    unique, counts = np.unique(y_train, return_counts=True)
    logger.info("Class distribution BEFORE SMOTE:")
    for cls, count in zip(unique, counts):
        logger.info(f"  Class {cls}: {count} ({count/len(y_train)*100:.1f}%)")

    # Apply SMOTE
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Check class distribution after SMOTE
    unique, counts = np.unique(y_train_smote, return_counts=True)
    logger.info("\nClass distribution AFTER SMOTE:")
    for cls, count in zip(unique, counts):
        logger.info(f"  Class {cls}: {count} ({count/len(y_train_smote)*100:.1f}%)")

    logger.info(f"Training data size: {len(X_train)} → {len(X_train_smote)} samples")

    return X_train_smote, y_train_smote


def main():
    logger.info("="*60)
    logger.info("SMOTE + XGBoost - Class Imbalance Handling")
    logger.info("="*60)

    cfg = get_config()

    try:
        # Load & prepare data
        df = load_and_engineer_data(cfg)
        X_train, X_val, y_train, y_val, numeric_features, categorical_features, le = prepare_data(df)

        # Preprocess & select features
        X_train_sel, X_val_sel = preprocess_and_select_features(
            X_train, X_val, y_train, numeric_features, categorical_features
        )

        # ===== APPLY SMOTE =====
        X_train_smote, y_train_smote = apply_smote(X_train_sel, y_train)

        # Set MLflow
        mlflow.set_tracking_uri(cfg.get('mlflow.tracking_uri', 'http://localhost:5000'))
        mlflow.set_experiment('water-pump-development')

        # ===== TRAIN XGBOOST WITH SMOTE =====
        logger.info("\n[XGBoost] Training XGBoost with SMOTE-balanced data...")

        with mlflow.start_run(run_name="xgboost_smote_balanced"):
            param_dist = {
                'n_estimators': [100, 150, 200],
                'max_depth': [5, 7, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
            }

            xgb = XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )

            random_search = RandomizedSearchCV(xgb, param_dist, n_iter=15, cv=5, scoring='accuracy', n_jobs=-1)
            random_search.fit(X_train_smote, y_train_smote)

            best_xgb = random_search.best_estimator_
            logger.info(f"Best XGB params: {random_search.best_params_}")

            # Evaluate on ORIGINAL validation set (not SMOTE'd)
            y_pred = best_xgb.predict(X_val_sel)

            acc = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

            # Macro metrics (important for imbalanced data)
            precision_macro = precision_score(y_val, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(y_val, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)

            # Log metrics
            mlflow.log_metric("val_accuracy", acc)
            mlflow.log_metric("val_precision_weighted", precision)
            mlflow.log_metric("val_recall_weighted", recall)
            mlflow.log_metric("val_f1_weighted", f1)
            mlflow.log_metric("val_precision_macro", precision_macro)
            mlflow.log_metric("val_recall_macro", recall_macro)
            mlflow.log_metric("val_f1_macro", f1_macro)

            logger.info(f"\nXGBoost (SMOTE) Accuracy: {acc:.4f}")
            logger.info(f"Weighted - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            logger.info(f"Macro - Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")

            # ===== DETAILED METRICS FOR EACH CLASS =====
            logger.info("\n" + "="*60)
            logger.info("PER-CLASS PERFORMANCE (minority class is key!)")
            logger.info("="*60)

            report = classification_report(y_val, y_pred, target_names=le.classes_, zero_division=0)
            logger.info("\n" + report)

            cm = confusion_matrix(y_val, y_pred)
            logger.info(f"\nConfusion Matrix:\n{cm}")

            # Calculate per-class metrics
            for i, class_name in enumerate(le.classes_):
                tp = cm[i, i]
                fn = cm[i, :].sum() - tp
                fp = cm[:, i].sum() - tp
                tn = cm.sum() - tp - fn - fp

                class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0

                logger.info(f"\n{class_name}:")
                logger.info(f"  Recall: {class_recall:.4f} (minority improvement key!)")
                logger.info(f"  Precision: {class_precision:.4f}")
                logger.info(f"  F1: {class_f1:.4f}")

            # Save results
            import joblib
            models_dir = Path("models/production")
            models_dir.mkdir(parents=True, exist_ok=True)

            joblib.dump(best_xgb, models_dir / "xgboost_smote.pkl")

            results = {
                "model": "xgboost_with_smote",
                "val_accuracy": float(acc),
                "val_precision": float(precision),
                "val_recall": float(recall),
                "val_f1": float(f1),
                "smote_applied": True,
                "training_data_size_before": 47520,
                "training_data_size_after": len(X_train_smote)
            }

            with open("metrics_smote_xgboost.json", "w") as f:
                json.dump(results, f, indent=2)

        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETED!")
        logger.info("="*60)
        logger.info(f"Results: metrics_smote_xgboost.json")
        logger.info(f"Model: models/production/xgboost_smote.pkl")
        logger.info(f"View at: http://localhost:5000\n")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
