# File: model_training/config.py
"""
Configuration Settings: Model Training Pipeline
-----------------------------------------------
Defines default paths, data preprocessing parameters, optimization metrics,
and base model hyperparameters used in the training process.
"""

# --- Directory Settings ---
DEFAULT_DATA_DIR = "../data"            # Default location for input datasets
DEFAULT_OUTPUT_DIR = "../saved_artifacts" # Default location for saving trained models and artifacts

# --- Data Preprocessing Settings ---
TEST_SPLIT_SIZE = 0.2   # Proportion of data to use for the test set
RANDOM_STATE = 42       # Seed for reproducibility in splitting and model training

# --- Default Optimization Metrics ---
# Specifies the default metric to optimize during hyperparameter tuning
# based on the problem type. Can be overridden by command-line arguments.
DEFAULT_OPTIMIZATION_METRICS = {
    'binary': 'f1',       # Default for binary classification
    'regression': 'rmse'  # Default for regression
    # Example for future multiclass support:
    # 'multiclass': 'f1_macro'
}

# --- Base Model Hyperparameters (LightGBM) ---
# Core parameters for LightGBM. Specifics like 'objective' and 'metric'
# are set dynamically in train_utils.py based on the problem type.
# Parameters like 'n_estimators', 'learning_rate', etc., are often
# overridden during hyperparameter tuning.
LGBM_BASE_PARAMS = {
    'n_estimators': 200,        # Initial number of boosting rounds
    'learning_rate': 0.05,      # Initial learning rate
    'num_leaves': 31,           # Initial max number of leaves in one tree
    'max_depth': -1,            # No limit on tree depth by default
    'seed': RANDOM_STATE,       # For reproducibility within LGBM
    'n_jobs': -1,               # Use all available CPU cores
    'verbose': -1,              # Suppress LightGBM verbosity
    # Note: 'objective', 'metric', and 'scale_pos_weight' are set dynamically.
}
