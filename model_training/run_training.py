# File: model_training/run_training.py
"""
Main Script: Model Training Pipeline Execution
---------------------------------------------
This script orchestrates the model training process for binary classification
or regression tasks using LightGBM. It handles command-line argument parsing,
data loading, preprocessing (via data_utils), hyperparameter tuning (via Optuna
in train_utils), final model training, evaluation, and artifact saving
(model, encoder, scaler).
"""

# --- Imports ---
import argparse
import os
import pandas as pd
import joblib
import traceback # For detailed error logging

# Import project modules from the model_training directory
from . import data_utils
from . import train_utils
from . import config as train_config

# --- Main Training Pipeline Function ---
def main(args):
    """
    Runs the complete model training pipeline based on provided arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    print(f"\n--- Starting Model Training Run ---")
    print("Configuration:")
    print(f"  Data Path: {args.data_path}")
    print(f"  Target Column: {args.target_column}")
    print(f"  Problem Type: {args.problem_type}")
    if args.problem_type == 'binary':
        print(f"  Positive Label: {args.positive_label if args.positive_label else 'Auto-Infer'}")
        print(f"  Tune Threshold Metric: {args.tune_threshold_metric if args.tune_threshold_metric else 'Disabled'}")
    print(f"  Optimization Metric (Optuna): {args.optimization_metric}")
    print(f"  Columns to Drop: {args.drop_cols if args.drop_cols else 'None'}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Optuna Trials: {args.optuna_trials}")
    print(f"  Model Type: {args.model_type}")
    print("-" * 30)

    # --- Argument Validation ---
    # Basic checks for metric suitability
    if args.problem_type == 'binary' and args.optimization_metric not in ['f1', 'recall', 'precision', 'auc', 'accuracy']:
        print(f"Warning: Optimization metric '{args.optimization_metric}' might not be standard for binary classification.")
    if args.problem_type == 'regression' and args.optimization_metric not in ['rmse', 'mae', 'r2']:
        print(f"Warning: Optimization metric '{args.optimization_metric}' might not be standard for regression.")
    if args.problem_type == 'binary' and args.tune_threshold_metric and args.tune_threshold_metric not in ['f1', 'recall', 'precision']:
        print(f"Warning: Threshold tuning metric '{args.tune_threshold_metric}' not supported. Valid options: f1, recall, precision. Disabling threshold tuning.")
        args.tune_threshold_metric = None

    # --- Pipeline Steps ---
    try:
        # 1. Resolve Paths and Load Data
        script_dir = os.path.dirname(__file__)
        base_dir = os.path.dirname(script_dir) # Assumes model_training is one level down from project root
        abs_data_path = args.data_path if os.path.isabs(args.data_path) else os.path.abspath(os.path.join(base_dir, args.data_path))
        abs_output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.abspath(os.path.join(base_dir, args.output_dir))
        print(f"Resolved Data Path: {abs_data_path}")
        print(f"Resolved Output Dir: {abs_output_dir}")
        df = data_utils.load_data(abs_data_path, args.target_column, args.drop_cols)

        # 2. Infer Feature Types
        inferred_numerical_features, inferred_categorical_features = data_utils.infer_feature_types(df, args.target_column)

        # 3. Preprocess Data (Impute, Encode, Scale, Split)
        # This returns the fitted encoder and scaler along with split data
        X_train, X_test, y_train, y_test, fitted_encoder, fitted_scaler, \
        final_numerical_features, final_categorical_features = data_utils.preprocess_data_generic(
            df,
            target_column_arg=args.target_column,
            numerical_features_arg=inferred_numerical_features,
            categorical_features_arg=inferred_categorical_features,
            problem_type=args.problem_type,
            positive_label=args.positive_label
        )
        print(f"Preprocessing complete. Training data shape: {X_train.shape}")

        # 4. Tune Hyperparameters and Train Final Model
        trained_model = None
        best_hyperparams = None
        if args.model_type.lower() == 'lgbm':
            # Prepare base parameters, removing those to be tuned
            base_lgbm_params = train_config.LGBM_BASE_PARAMS.copy()
            if args.problem_type == 'binary': base_lgbm_params['objective'] = 'binary'; base_lgbm_params['metric'] = 'auc' # Default metric for LGBM internal eval
            elif args.problem_type == 'regression': base_lgbm_params['objective'] = 'regression'; base_lgbm_params['metric'] = 'rmse'
            else: raise ValueError(f"Problem type '{args.problem_type}' not supported by LGBM setup.")

            # Parameters to be tuned by Optuna (remove from base_params if present)
            params_to_tune = ['n_estimators', 'learning_rate', 'num_leaves', 'max_depth', 'reg_alpha', 'reg_lambda', 'colsample_bytree', 'subsample', 'subsample_freq', 'min_child_samples', 'scale_pos_weight']
            for param in params_to_tune: base_lgbm_params.pop(param, None)

            # Run tuning and final training
            trained_model, best_hyperparams = train_utils.tune_and_train_lgbm_model_generic(
                X_train, y_train, base_lgbm_params,
                args.problem_type, args.optimization_metric, # Pass Optuna optimization metric
                n_trials=args.optuna_trials
            )
        else:
            raise ValueError(f"Model type '{args.model_type}' not supported.")

        if trained_model is None:
            raise RuntimeError("Model training failed (returned None).")

        # 5. Evaluate Final Model on Test Set
        train_utils.evaluate_model_generic(
            trained_model, X_test, y_test, args.problem_type,
            tune_threshold_metric=args.tune_threshold_metric # Pass metric for threshold tuning if specified
        )

        # 6. Save Model Artifacts (Model, Encoder, Scaler, Params)
        train_utils.save_model_and_info(
            model=trained_model,
            output_dir=abs_output_dir,
            best_params=best_hyperparams,
            encoder=fitted_encoder, # Save the fitted encoder
            scaler=fitted_scaler    # Save the fitted scaler
        )

        print(f"\n--- Training Run Finished Successfully ---")
        print(f"Artifacts saved to: {abs_output_dir}")

    except Exception as e:
        print(f"\n--- Training Run FAILED ---")
        print(f"Error: {e}")
        traceback.print_exc()
        print("-----------------------------")

# --- Command Line Interface Setup ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a LightGBM model (Binary Classification/Regression) with preprocessing, Optuna tuning, and optional threshold optimization."
    )

    # Required Arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the input CSV dataset (relative to project root or absolute).")
    parser.add_argument("--target_column", type=str, required=True,
                        help="Name of the target variable column in the dataset.")
    parser.add_argument("--problem_type", type=str, required=True, choices=['binary', 'regression'],
                        help="Specify the type of problem: 'binary' classification or 'regression'.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory path to save trained model artifacts (relative to project root or absolute).")

    # Optional Arguments
    parser.add_argument("--model_type", type=str, default="lgbm", choices=["lgbm"],
                        help="Type of model to train (default: lgbm). Currently only supports LightGBM.")
    parser.add_argument("--positive_label", type=str, default=None,
                        help="Value representing the positive class in binary classification. Required if target isn't 0/1 and cannot be inferred (e.g., 'Yes', 'Paid').")
    parser.add_argument("--optimization_metric", type=str, default=None,
                        help="Metric for Optuna to optimize during hyperparameter tuning (e.g., 'f1', 'auc' for binary; 'rmse', 'r2' for regression). Defaults based on problem_type.")
    parser.add_argument("--drop_cols", nargs='*', default=None,
                        help="Space-separated list of column names to drop from the dataset before training (e.g., --drop_cols id user_id).")
    parser.add_argument("--optuna_trials", type=int, default=50,
                        help="Number of hyperparameter tuning trials for Optuna (default: 50).")
    parser.add_argument("--tune_threshold_metric", type=str, default=None, choices=['f1', 'recall', 'precision'],
                        help="Metric to optimize for probability threshold tuning in binary classification (e.g., 'f1'). If omitted, threshold tuning is skipped.")

    args = parser.parse_args()

    # --- Set Default Optimization Metric if Not Provided ---
    if args.optimization_metric is None:
        args.optimization_metric = train_config.DEFAULT_OPTIMIZATION_METRICS.get(args.problem_type)
        if args.optimization_metric is None:
            print(f"Error: Could not determine default optimization metric for problem type '{args.problem_type}'. Please specify using --optimization_metric.")
            exit(1) # Exit if no default is found
        print(f"Using default Optuna optimization metric for {args.problem_type}: {args.optimization_metric}")

    # --- Handle Optional Argument Logic ---
    if args.problem_type == 'binary' and args.positive_label is None:
        print("Info: --positive_label not provided for binary classification. Will assume target is 0/1 or attempt auto-inference.")
    if args.problem_type == 'regression' and args.tune_threshold_metric is not None:
        print("Warning: --tune_threshold_metric is only applicable for binary classification. Ignoring for regression.")
        args.tune_threshold_metric = None

    # --- Execute Main Pipeline ---
    main(args)
