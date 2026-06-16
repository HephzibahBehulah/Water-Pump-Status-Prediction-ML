"""
Complete training pipeline with DVC + MLflow integration.
This script demonstrates:
  1. Loading data (tracked with DVC)
  2. Feature engineering
  3. Model training (tracked with MLflow)
  4. Model evaluation
  5. Saving results (artifacts)

Run with: python scripts/train_pipeline.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from src.utils import setup_logger, get_config
from src.models import ModelPipeline

logger = setup_logger(__name__, log_level="INFO", log_file="logs/training.log")


def load_and_prepare_data(config):
    """
    Load raw data and prepare it for training.
    Data files should be tracked with DVC.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, feature_names)
    """
    logger.info("Loading data...")

    # Load data
    raw_dir = Path(config.get('data.raw_dir'))
    train_values_path = raw_dir / "training_values.csv"
    train_labels_path = raw_dir / "training_labels.csv"

    if not train_values_path.exists():
        logger.error(f"Training data not found at {train_values_path}")
        logger.info(f"Please download from: https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/")
        raise FileNotFoundError(f"Data not found: {train_values_path}")

    # Load CSVs
    df_values = pd.read_csv(train_values_path)
    df_labels = pd.read_csv(train_labels_path)

    logger.info(f"Loaded {len(df_values)} training records")

    # Merge
    df = df_values.merge(df_labels, on='id')
    logger.info(f"Merged training and labels")

    # Use all columns except id and target
    drop_cols = ['id', 'status_group', 'wpt_name', 'recorded_by', 'scheme_name']

    # Get feature columns
    feature_cols = [col for col in df.columns if col not in drop_cols]

    logger.info(f"Using {len(feature_cols)} features for training")
    logger.info(f"Features: {feature_cols[:5]}... (showing first 5)")

    # Prepare features and target
    X = df[feature_cols]
    y = df['status_group']

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    logger.info(f"Features: {X.shape[1]}, Classes: {len(le.classes_)}")
    logger.info(f"Classes: {le.classes_.tolist()}")

    # Train/val split
    train_size = config.get_float('data.train_split', 0.8)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded,
        test_size=1-train_size,
        stratify=y_encoded,
        random_state=42
    )

    logger.info(f"Train/Val split: {len(X_train)} / {len(X_val)}")

    return X_train, y_train, X_val, y_val, X.columns.tolist(), le


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("WATER PUMP STATUS PREDICTION - TRAINING PIPELINE")
    logger.info("="*60)

    # Load config
    cfg = get_config()
    logger.info(f"Using {cfg.environment.upper()} configuration")

    try:
        # ============ STAGE 1: DATA LOADING ============
        logger.info("\n[STAGE 1] Loading Data...")
        X_train, y_train, X_val, y_val, feature_names, label_encoder = load_and_prepare_data(cfg)

        # ============ STAGE 2: MODEL TRAINING ============
        logger.info("\n[STAGE 2] Training Models with MLflow...")

        # Set MLflow tracking
        mlflow_uri = cfg.get('mlflow.tracking_uri', 'http://localhost:5000')
        mlflow.set_tracking_uri(mlflow_uri)
        experiment_name = cfg.get('mlflow.experiment_name', 'water-pump-development')

        # Initialize pipeline
        pipeline = ModelPipeline(cfg.config)

        # Train Random Forest
        logger.info("\nTraining Random Forest...")
        rf_results = pipeline.train(
            X_train, y_train,
            X_val, y_val,
            model_name="random_forest",
            experiment_name=experiment_name
        )

        # ============ STAGE 3: SAVE MODELS ============
        logger.info("\n[STAGE 3] Saving Models...")

        models_dir = Path(cfg.get('data.raw_dir')).parent.parent / "models" / "production"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = models_dir / "random_forest_latest.pkl"
        pipeline.save_model(str(model_path))

        # Save label encoder
        le_path = models_dir / "label_encoder.pkl"
        import joblib
        joblib.dump(label_encoder, le_path)
        logger.info(f"Label encoder saved to {le_path}")

        # ============ STAGE 4: LOG RESULTS ============
        logger.info("\n[STAGE 4] Logging Results...")

        # Create results dictionary
        results = {
            "model_name": "random_forest",
            "timestamp": pd.Timestamp.now().isoformat(),
            "metrics": {
                "train_accuracy": float(rf_results['train_accuracy']),
                "val_accuracy": float(rf_results['val_accuracy']),
                "val_precision": float(rf_results['val_precision']),
                "val_recall": float(rf_results['val_recall']),
                "val_f1": float(rf_results['val_f1']),
            },
            "features_used": feature_names,
            "target_classes": label_encoder.classes_.tolist(),
            "data_shapes": {
                "train": str(X_train.shape),
                "val": str(X_val.shape)
            }
        }

        # Save results
        results_file = Path("metrics.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_file}")

        # ============ COMPLETION ============
        logger.info("\n" + "="*60)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"\nModel Performance:")
        logger.info(f"   Train Accuracy: {rf_results['train_accuracy']:.4f}")
        logger.info(f"   Validation Accuracy: {rf_results['val_accuracy']:.4f}")
        logger.info(f"   Validation F1 Score: {rf_results['val_f1']:.4f}")
        logger.info(f"\nSaved artifacts:")
        logger.info(f"   Model: {model_path}")
        logger.info(f"   Label Encoder: {le_path}")
        logger.info(f"   Metrics: {results_file}")
        logger.info(f"\nMLflow UI: {mlflow_uri}")
        logger.info(f"   Run: mlflow ui")
        logger.info("\n")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
