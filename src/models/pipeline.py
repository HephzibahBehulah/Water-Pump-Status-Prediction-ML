"""
Model pipeline for loading, training, and making predictions.
Integrates with MLflow for experiment tracking and model registry.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.utils import setup_logger

logger = setup_logger(__name__)


class ModelPipeline:
    """Handles model training, evaluation, and inference with MLflow integration."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model pipeline.

        Args:
            config: Configuration dictionary from Config class
        """
        self.config = config
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.feature_names = None
        logger.info("ModelPipeline initialized")

    def _prepare_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Simple data preparation: convert to numeric.

        Args:
            X: DataFrame to prepare

        Returns:
            Numeric DataFrame
        """
        X_prep = X.copy()

        # Convert object columns to numeric codes
        for col in X_prep.columns:
            if X_prep[col].dtype == 'object':
                X_prep[col] = pd.factorize(X_prep[col])[0]
            # Fill NaN with 0
            X_prep[col] = X_prep[col].fillna(0)

        return X_prep

    def create_preprocessing_pipeline(
        self,
        numeric_features: list,
        categorical_features: list
    ) -> ColumnTransformer:
        """
        Create scikit-learn preprocessing pipeline.

        Args:
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names

        Returns:
            ColumnTransformer for preprocessing
        """
        # Define transformers
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        logger.info(f"✅ Preprocessing pipeline created: {len(numeric_features)} numeric, {len(categorical_features)} categorical features")
        return preprocessor

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_name: str = "random_forest",
        experiment_name: str = "water-pump-development"
    ) -> Dict[str, Any]:
        """
        Train model with MLflow tracking.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_name: Which model to train (random_forest, xgboost)
            experiment_name: MLflow experiment name

        Returns:
            Dictionary with metrics and model info
        """
        # Set MLflow experiment
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"{model_name}_training"):
            logger.info(f"🚀 Starting training for {model_name}")

            # Get model config
            model_config = self.config.get(f'models.{model_name}', {})
            logger.info(f"Model config: {model_config}")

            # Create and train model
            if model_name == "random_forest":
                model = RandomForestClassifier(**model_config)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            # Log parameters
            for param_name, param_value in model_config.items():
                mlflow.log_param(param_name, param_value)

            # Convert to numeric (simple preprocessing)
            X_train_numeric = self._prepare_data(X_train)
            X_val_numeric = self._prepare_data(X_val)

            # Train
            logger.info("Training model...")
            model.fit(X_train_numeric, y_train)

            # Predictions
            y_train_pred = model.predict(X_train_numeric)
            y_val_pred = model.predict(X_val_numeric)

            # Calculate metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_precision = precision_score(y_val, y_val_pred, average='weighted')
            val_recall = recall_score(y_val, y_val_pred, average='weighted')
            val_f1 = f1_score(y_val, y_val_pred, average='weighted')

            # Log metrics
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("val_accuracy", val_acc)
            mlflow.log_metric("val_precision", val_precision)
            mlflow.log_metric("val_recall", val_recall)
            mlflow.log_metric("val_f1", val_f1)

            logger.info(f"✅ Training complete")
            logger.info(f"   Train Accuracy: {train_acc:.4f}")
            logger.info(f"   Validation Accuracy: {val_acc:.4f}")
            logger.info(f"   Validation F1: {val_f1:.4f}")

            # Log model
            mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
            logger.info(f"📦 Model logged to MLflow")

            self.model = model

            return {
                "model": model,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "confusion_matrix": confusion_matrix(y_val, y_val_pred).tolist()
            }

    def save_model(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, path)
        logger.info(f"💾 Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load model from disk.

        Args:
            path: Path to model file
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        self.model = joblib.load(path)
        logger.info(f"✅ Model loaded from {path}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Features for prediction

        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train() first.")

        predictions = self.model.predict(X)
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Features for prediction

        Returns:
            Array of probability distributions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train() first.")

        probabilities = self.model.predict_proba(X)
        return probabilities
