"""
Project Constants - Centralized magic numbers and configuration values
"""

# ===== MODEL CONSTANTS =====
FEATURE_SELECTION_K = 15  # Number of top features to select
RANDOM_STATE = 42  # For reproducibility
N_JOBS = -1  # Use all available cores

# ===== DATA CONSTANTS =====
TRAIN_TEST_SPLIT_RATIO = 0.2
STRATIFIED_SPLIT = True

# ===== FEATURE ENGINEERING =====
PUMP_AGE_DEFAULT = 5
YEAR_RECORDED_DEFAULT = 2023
MONTH_RECORDED_DEFAULT = 6
CONSTRUCTION_YEAR_DEFAULT = 2015
POPULATION_DEFAULT = 1000
REGION_CODE_DEFAULT = 1
DISTRICT_CODE_DEFAULT = 1
GPS_HEIGHT_DEFAULT = 1000
AMOUNT_TSH_DEFAULT = 50000
NUM_PRIVATE_DEFAULT = 0

# ===== CATEGORICAL DEFAULTS =====
MISSING_CATEGORICAL_VALUE = 'unknown'

# ===== CLASS LABELS =====
CLASS_NAMES = ['Functional', 'Functional needs repair', 'Non-functional']
CLASS_FUNCTIONAL = 0
CLASS_NEEDS_REPAIR = 1
CLASS_NON_FUNCTIONAL = 2

# ===== RISK THRESHOLDS =====
RISK_LOW_THRESHOLD = 0.3
RISK_MEDIUM_THRESHOLD = 0.6
RISK_HIGH_THRESHOLD = 1.0

RISK_LEVELS = {
    'LOW': (0.0, RISK_LOW_THRESHOLD),
    'MEDIUM': (RISK_LOW_THRESHOLD, RISK_MEDIUM_THRESHOLD),
    'HIGH': (RISK_MEDIUM_THRESHOLD, RISK_HIGH_THRESHOLD),
}

# ===== HYPERPARAMETERS =====
# Random Forest
RF_N_ESTIMATORS = [100, 150, 200]
RF_MAX_DEPTH = [15, 20, 25]
RF_MIN_SAMPLES_SPLIT = [2, 5, 10]
RF_MIN_SAMPLES_LEAF = [1, 2, 4]
RF_MAX_FEATURES = ['sqrt', 'log2']
RF_RANDOM_SEARCH_ITERATIONS = 15
RF_CV_FOLDS = 5

# XGBoost
XGB_N_ESTIMATORS = [100, 150, 200]
XGB_MAX_DEPTH = [5, 7, 9]
XGB_LEARNING_RATE = [0.01, 0.1, 0.3]
XGB_SUBSAMPLE = [0.7, 0.8, 0.9]
XGB_COLSAMPLE_BYTREE = [0.7, 0.8, 0.9]
XGB_RANDOM_SEARCH_ITERATIONS = 15
XGB_CV_FOLDS = 5

# SMOTE
SMOTE_K_NEIGHBORS = 5

# ===== FEATURE LISTS =====
NUMERIC_FEATURES = [
    'amount_tsh', 'gps_height', 'longitude', 'latitude', 'num_private',
    'region_code', 'district_code', 'population', 'month_recorded', 'pump_age'
]

CATEGORICAL_FEATURES = [
    'funder', 'installer', 'basin', 'region', 'lga', 'ward', 'public_meeting',
    'scheme_management', 'permit', 'extraction_type', 'extraction_type_group',
    'extraction_type_class', 'management', 'management_group', 'payment',
    'payment_type', 'water_quality', 'quality_group', 'quantity', 'quantity_group',
    'source', 'source_type', 'source_class', 'waterpoint_type', 'waterpoint_type_group'
]

# ===== PATHS =====
DATA_RAW_DIR = 'data/raw'
DATA_PROCESSED_DIR = 'data/processed'
MODELS_DIR = 'models/production'
LOGS_DIR = 'logs'
RESULTS_DIR = 'results'
METRICS_DIR = 'results/metrics'
PREDICTIONS_DIR = 'results/predictions'

# ===== MLFLOW CONSTANTS =====
MLFLOW_EXPERIMENT_NAME = 'water-pump-development'
MLFLOW_TRACKING_URI = 'http://localhost:5000'

# ===== LOGGING =====
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ===== FEATURE ENGINEERING DROP COLUMNS =====
COLUMNS_TO_DROP = ['id', 'recorded_by', 'scheme_name', 'wpt_name', 'subvillage']

# ===== ZERO-VALUE COLUMNS (replace with NaN) =====
ZERO_VALUE_COLUMNS = ['gps_height', 'longitude', 'latitude', 'amount_tsh']
