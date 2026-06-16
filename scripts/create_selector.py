"""
Create and save the SelectKBest selector for the SMOTE model
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import get_config


def convert_to_str(X):
    return X.astype(str)


def create_and_save_selector():
    """Create selector from training data"""
    print("Creating SelectKBest selector...")

    cfg = get_config()

    # Load data
    raw_dir = Path(cfg.get('data.raw_dir'))
    df_values = pd.read_csv(raw_dir / "training_values.csv")
    df_labels = pd.read_csv(raw_dir / "training_labels.csv")
    df = df_values.merge(df_labels, on='id')

    # Feature engineering (same as training)
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

    # Prepare features
    X = df.drop(columns=['status_group']).copy()
    y = df['status_group']

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    # Preprocess
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

    # Create and fit selector
    selector = SelectKBest(score_func=mutual_info_classif, k='all')
    selector.fit(X_train_proc, y_train)

    # Save
    models_dir = Path("models/production")
    joblib.dump(selector, models_dir / "selector.pkl")
    print(f"✅ Selector saved to: models/production/selector.pkl")


if __name__ == "__main__":
    create_and_save_selector()
