"""
Train Multiple Models: Random Forest + XGBoost with Comparison

Run: python scripts/train_multiple_models.py
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

from src.utils import setup_logger, get_config

logger = setup_logger(__name__, log_level="INFO", log_file="logs/training_multiple.log")


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


def train_random_forest(X_train, X_val, y_train, y_val):
    logger.info("\n[MODEL 1] Training Random Forest...")

    with mlflow.start_run(run_name="rf_tuned"):
        param_dist = {
            'n_estimators': [100, 150, 200],
            'max_depth': [15, 20, 25],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2'],
        }

        rf = RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42)
        random_search = RandomizedSearchCV(rf, param_dist, n_iter=15, cv=5, scoring='accuracy', n_jobs=-1)
        random_search.fit(X_train, y_train)

        best_rf = random_search.best_estimator_
        logger.info(f"Best RF params: {random_search.best_params_}")

        y_pred = best_rf.predict(X_val)
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision_weighted': precision_score(y_val, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_val, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_val, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_val, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_val, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_val, y_pred, average='macro', zero_division=0),
        }

        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        logger.info(f"RF Accuracy: {metrics['accuracy']:.4f}, F1 (weighted): {metrics['f1_weighted']:.4f}, F1 (macro): {metrics['f1_macro']:.4f}")

        return best_rf, metrics


def train_xgboost(X_train, X_val, y_train, y_val):
    logger.info("\n[MODEL 2] Training XGBoost...")

    with mlflow.start_run(run_name="xgb_tuned"):
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
        random_search.fit(X_train, y_train)

        best_xgb = random_search.best_estimator_
        logger.info(f"Best XGB params: {random_search.best_params_}")

        y_pred = best_xgb.predict(X_val)
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision_weighted': precision_score(y_val, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_val, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_val, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_val, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_val, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_val, y_pred, average='macro', zero_division=0),
        }

        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        logger.info(f"XGB Accuracy: {metrics['accuracy']:.4f}, F1 (weighted): {metrics['f1_weighted']:.4f}, F1 (macro): {metrics['f1_macro']:.4f}")

        return best_xgb, metrics


def main():
    logger.info("="*60)
    logger.info("TRAINING MULTIPLE MODELS: RF + XGBoost")
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

        # Set MLflow
        mlflow.set_tracking_uri(cfg.get('mlflow.tracking_uri', 'http://localhost:5000'))
        mlflow.set_experiment('water-pump-development')

        # Train models
        rf_model, rf_metrics = train_random_forest(X_train_sel, X_val_sel, y_train, y_val)
        xgb_model, xgb_metrics = train_xgboost(X_train_sel, X_val_sel, y_train, y_val)

        # Compare
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON")
        logger.info("="*60)
        logger.info(f"\nRandom Forest Accuracy: {rf_metrics['accuracy']:.4f}")
        logger.info(f"XGBoost Accuracy:       {xgb_metrics['accuracy']:.4f}")

        if xgb_metrics['accuracy'] > rf_metrics['accuracy']:
            logger.info(f"\nWINNER: XGBoost (+{(xgb_metrics['accuracy'] - rf_metrics['accuracy'])*100:.2f}%)")
        else:
            logger.info(f"\nWINNER: Random Forest (+{(rf_metrics['accuracy'] - xgb_metrics['accuracy'])*100:.2f}%)")

        # Save results
        import joblib
        models_dir = Path("models/production")
        models_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(rf_model, models_dir / "rf_multiple_comparison.pkl")
        joblib.dump(xgb_model, models_dir / "xgb_multiple_comparison.pkl")

        results = {
            "random_forest": {
                "accuracy": float(rf_metrics['accuracy']),
                "precision_weighted": float(rf_metrics['precision_weighted']),
                "precision_macro": float(rf_metrics['precision_macro']),
                "recall_weighted": float(rf_metrics['recall_weighted']),
                "recall_macro": float(rf_metrics['recall_macro']),
                "f1_weighted": float(rf_metrics['f1_weighted']),
                "f1_macro": float(rf_metrics['f1_macro'])
            },
            "xgboost": {
                "accuracy": float(xgb_metrics['accuracy']),
                "precision_weighted": float(xgb_metrics['precision_weighted']),
                "precision_macro": float(xgb_metrics['precision_macro']),
                "recall_weighted": float(xgb_metrics['recall_weighted']),
                "recall_macro": float(xgb_metrics['recall_macro']),
                "f1_weighted": float(xgb_metrics['f1_weighted']),
                "f1_macro": float(xgb_metrics['f1_macro'])
            },
            "best_model": "xgboost" if xgb_metrics['accuracy'] > rf_metrics['accuracy'] else "random_forest"
        }

        with open("metrics_multiple_models.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info("\nResults saved to: metrics_multiple_models.json")
        logger.info("View comparison at: http://localhost:5000\n")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
