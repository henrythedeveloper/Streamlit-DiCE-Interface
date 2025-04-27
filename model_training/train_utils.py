# File: model_training/train_utils.py
"""
Model Training Utilities
------------------------
Provides helper functions for the LightGBM training process, including:
- Optuna-based hyperparameter tuning with cross-validation.
- Final model training using best found parameters.
- Model evaluation using standard metrics (classification and regression).
- Optimal probability threshold finding for binary classification.
- Saving trained model artifacts (model, encoder, scaler, parameters).
Assumes input data (X) is already preprocessed (encoded categoricals, scaled numericals).
"""

# --- Imports ---
import lightgbm as lgb
import numpy as np
import pandas as pd
import os
import json
import joblib
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    average_precision_score, f1_score, recall_score, precision_score,
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_curve # Used for threshold tuning implicitly
)
from sklearn.model_selection import StratifiedKFold, KFold
import optuna
import traceback # For detailed error logging

# Configure Optuna logging level (optional: reduce verbosity)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Feature Range Calculation (Legacy/Reference) ---
# Note: This function calculates ranges but they are not currently used
# directly in the artifact saving or model training process here.
# It might be useful for analysis or setting constraints elsewhere.
def calculate_and_get_ranges(X_train, numerical_features):
    """
    Calculates feature ranges (1st to 99th percentile) for numerical features.

    Args:
        X_train (pd.DataFrame): The training feature set.
        numerical_features (list): List of numerical column names.

    Returns:
        dict: Dictionary mapping numerical feature names to [min, max] range.
    """
    print("\nCalculating feature ranges (1st-99th percentile) for numerical features...")
    ranges = {}
    if not numerical_features:
        print("No numerical features provided for range calculation.")
        return ranges

    for feature in numerical_features:
        if feature not in X_train.columns:
            print(f"Warning: Numerical feature '{feature}' not in X_train. Skipping range calc.")
            continue
        if pd.api.types.is_numeric_dtype(X_train[feature]):
            feature_data = X_train[feature].dropna()
            if not feature_data.empty:
                try:
                    min_val = np.percentile(feature_data, 1)
                    max_val = np.percentile(feature_data, 99)
                    # Basic sanity check
                    if min_val > max_val: min_val, max_val = max_val, min_val
                    ranges[feature] = [float(min_val), float(max_val)]
                except Exception as e:
                    print(f"  Error calculating range for '{feature}': {e}")
            else:
                print(f"  Feature '{feature}' has no non-NaN data. Skipping range calc.")
        else:
            print(f"  Feature '{feature}' is not numeric. Skipping range calc.")

    print(f"Feature ranges calculated for {len(ranges)} numerical features.")
    return ranges


# --- Optuna Objective Function ---
def _lgbm_objective_generic(trial, X, y, cv, base_params,
                            problem_type, optimization_metric):
    """
    Objective function for Optuna hyperparameter optimization using cross-validation.
    Evaluates a set of hyperparameters suggested by Optuna.

    Args:
        trial (optuna.Trial): An Optuna trial object.
        X (pd.DataFrame): Training features.
        y (pd.Series): Training target variable.
        cv (sklearn.model_selection._split): Cross-validation splitter object.
        base_params (dict): Fixed LightGBM parameters (e.g., objective, metric, seed).
        problem_type (str): 'binary' or 'regression'.
        optimization_metric (str): The metric to optimize (e.g., 'f1', 'rmse').

    Returns:
        float: The average score of the optimization metric across CV folds.
    """
    # --- Define Hyperparameter Search Space ---
    # Suggest values for hyperparameters within reasonable ranges
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 60),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True), # L1 regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True), # L2 regularization
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), # Feature fraction
        'subsample': trial.suggest_float('subsample', 0.6, 1.0), # Row fraction (bagging)
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 7), # Frequency for bagging
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50), # Min data needed in a leaf
    }
    # Add scale_pos_weight for binary classification if data is imbalanced
    if problem_type == 'binary':
        count_class_0_y = np.sum(y == 0)
        count_class_1_y = np.sum(y == 1)
        if count_class_1_y > 0 and count_class_0_y > 0: # Only suggest if both classes present
            # Suggest a range around the theoretical optimal for imbalance
            ratio = count_class_0_y / count_class_1_y
            params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', max(0.1, ratio*0.5), min(10.0, ratio*2.0), log=True)

    # Combine suggested params with fixed base params
    params.update(base_params)

    scores = []
    # --- Cross-Validation Loop ---
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Initialize model based on problem type
        model = None
        current_fold_params = params.copy() # Use a copy for fold-specific adjustments

        if problem_type == 'binary':
            # Check if both classes are present in the fold for scale_pos_weight
            count_class_0_fold = np.sum(y_train_fold == 0)
            count_class_1_fold = np.sum(y_train_fold == 1)
            if not (count_class_1_fold > 0 and count_class_0_fold > 0) and 'scale_pos_weight' in current_fold_params:
                del current_fold_params['scale_pos_weight'] # Remove if only one class in fold
            model = lgb.LGBMClassifier(**current_fold_params)
        elif problem_type == 'regression':
            current_fold_params.pop('scale_pos_weight', None) # Ensure it's removed for regression
            model = lgb.LGBMRegressor(**current_fold_params)
        else:
            raise ValueError(f"Unsupported problem_type in objective: {problem_type}")

        # --- Fit Model with Early Stopping ---
        try:
            # Note: categorical_feature parameter is NOT used as data is pre-encoded
            model.fit(X_train_fold, y_train_fold,
                      eval_set=[(X_val_fold, y_val_fold)],
                      eval_metric=base_params.get('metric'), # Use metric defined in base_params
                      callbacks=[lgb.early_stopping(15, verbose=False)]) # Stop if validation score doesn't improve
        except Exception as e:
            print(f"Warning: Error during model fitting in fold {fold + 1}: {e}. Skipping fold.")
            # Assign a score indicative of failure based on optimization direction
            direction = 'maximize' if optimization_metric in ['f1', 'recall', 'precision', 'auc', 'accuracy', 'r2'] else 'minimize'
            scores.append(float('-inf') if direction == 'maximize' else float('inf'))
            continue # Skip to next fold

        # --- Evaluate Fold based on Optimization Metric ---
        score = None
        try:
            if problem_type == 'binary':
                y_pred_val = model.predict(X_val_fold) # Predict classes (0/1)
                if optimization_metric == 'f1': score = f1_score(y_val_fold, y_pred_val, pos_label=1, zero_division=0)
                elif optimization_metric == 'recall': score = recall_score(y_val_fold, y_pred_val, pos_label=1, zero_division=0)
                elif optimization_metric == 'precision': score = precision_score(y_val_fold, y_pred_val, pos_label=1, zero_division=0)
                elif optimization_metric == 'auc':
                     # AUC requires probabilities
                     if hasattr(model, 'predict_proba'):
                         y_prob_val = model.predict_proba(X_val_fold)[:, 1]
                         if len(np.unique(y_val_fold)) > 1: # Check if both classes present in validation fold
                              score = roc_auc_score(y_val_fold, y_prob_val)
                         else: score = 0.5 # Assign neutral score if only one class present
                     else: score = 0.0 # Cannot calculate AUC
                elif optimization_metric == 'accuracy': score = accuracy_score(y_val_fold, y_pred_val)
                else: raise ValueError(f"Unsupported optimization_metric for binary: {optimization_metric}")
            elif problem_type == 'regression':
                y_pred_val = model.predict(X_val_fold)
                if optimization_metric == 'rmse': score = np.sqrt(mean_squared_error(y_val_fold, y_pred_val))
                elif optimization_metric == 'mae': score = mean_absolute_error(y_val_fold, y_pred_val)
                elif optimization_metric == 'r2': score = r2_score(y_val_fold, y_pred_val)
                else: raise ValueError(f"Unsupported optimization_metric for regression: {optimization_metric}")
        except Exception as e:
            print(f"Warning: Error calculating score in fold {fold + 1}: {e}")
            direction = 'maximize' if optimization_metric in ['f1', 'recall', 'precision', 'auc', 'accuracy', 'r2'] else 'minimize'
            score = float('-inf') if direction == 'maximize' else float('inf') # Assign failure score

        scores.append(score if score is not None else (float('-inf') if direction == 'maximize' else float('inf')))

    # Calculate mean score across folds, ignoring failed folds (inf/nan)
    valid_scores = [s for s in scores if np.isfinite(s)]
    mean_score = np.mean(valid_scores) if valid_scores else (float('-inf') if direction == 'maximize' else float('inf'))

    return mean_score


# --- Model Tuning and Training Function ---
def tune_and_train_lgbm_model_generic(X_train, y_train, base_params,
                                      problem_type, optimization_metric,
                                      n_trials=50, n_splits=5):
    """
    Tunes LightGBM hyperparameters using Optuna and cross-validation,
    then trains the final model on the full training set using the best parameters found.

    Args:
        X_train (pd.DataFrame): Full training feature set (preprocessed).
        y_train (pd.Series): Full training target variable.
        base_params (dict): Fixed LightGBM parameters (objective, metric, seed, etc.).
        problem_type (str): 'binary' or 'regression'.
        optimization_metric (str): Metric for Optuna to optimize.
        n_trials (int): Number of Optuna trials to run.
        n_splits (int): Number of folds for cross-validation during tuning.

    Returns:
        tuple: (trained_lgbm_model, best_hyperparameters_dict)
               Returns (None, None) if tuning or training fails.
    """
    print(f"\n--- Starting Optuna Hyperparameter Tuning ({n_trials} trials, {n_splits}-fold CV) ---")
    print(f"Optimizing for: {optimization_metric}")

    # Setup cross-validation strategy based on problem type
    if problem_type == 'binary':
        # Stratified K-Fold for classification to preserve class proportions
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=base_params.get('seed', 42))
    elif problem_type == 'regression':
        # Standard K-Fold for regression
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=base_params.get('seed', 42))
    else:
        raise ValueError(f"Unsupported problem_type for CV setup: {problem_type}")

    # Determine optimization direction based on metric
    direction = 'maximize'
    if optimization_metric in ['rmse', 'mae']:
        direction = 'minimize'

    # --- Run Optuna Study ---
    try:
        study = optuna.create_study(direction=direction)
        study.optimize(lambda trial: _lgbm_objective_generic(
                            trial, X_train, y_train, cv, base_params,
                            problem_type, optimization_metric
                            ),
                       n_trials=n_trials, show_progress_bar=True) # Show progress bar

        best_params_trial = study.best_trial.params
        print(f"\nOptuna Tuning Complete.")
        print(f"Best trial {optimization_metric} score (avg across folds): {study.best_trial.value:.4f}")
        print("Best hyperparameters found:")
        for key, value in best_params_trial.items(): print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error during Optuna optimization: {e}")
        traceback.print_exc()
        return None, None # Return None if tuning fails

    # --- Train Final Model ---
    print("\n--- Training Final LightGBM Model with Best Parameters ---")
    final_params = base_params.copy()
    final_params.update(best_params_trial) # Combine base params with best tuned params

    final_model = None
    try:
        if problem_type == 'binary':
            # Recalculate scale_pos_weight on the full training set if needed
            count_class_0_final = np.sum(y_train == 0)
            count_class_1_final = np.sum(y_train == 1)
            if count_class_1_final > 0 and count_class_0_final > 0:
                # Use tuned value if available, otherwise calculate
                if 'scale_pos_weight' not in final_params:
                    final_params['scale_pos_weight'] = count_class_0_final / count_class_1_final
                    print(f"Calculated scale_pos_weight for final model: {final_params['scale_pos_weight']:.4f}")
                else:
                    print(f"Using tuned scale_pos_weight for final model: {final_params['scale_pos_weight']:.4f}")
            elif 'scale_pos_weight' in final_params:
                del final_params['scale_pos_weight'] # Remove if only one class present

            final_model = lgb.LGBMClassifier(**final_params)
        elif problem_type == 'regression':
            final_params.pop('scale_pos_weight', None) # Ensure removed for regression
            final_model = lgb.LGBMRegressor(**final_params)
        else:
            raise ValueError(f"Unsupported problem_type for final model: {problem_type}")

        # Fit the final model on the entire training dataset
        # No early stopping here, use the tuned n_estimators
        final_model.fit(X_train, y_train)
        print("--- Final Model Training Finished ---")
        return final_model, final_params

    except Exception as e:
        print(f"Error during final model training: {e}")
        traceback.print_exc()
        return None, None # Return None if final training fails


# --- Optimal Threshold Finding ---
def find_optimal_threshold(y_true, y_pred_proba, metric='f1', steps=100):
    """
    Finds the optimal probability threshold for binary classification
    that maximizes a given metric (f1, recall, or precision).

    Args:
        y_true (pd.Series or np.ndarray): True binary labels (0/1).
        y_pred_proba (np.ndarray): Predicted probabilities for the positive class (class 1).
        metric (str): The metric to maximize ('f1', 'recall', 'precision').
        steps (int): Number of thresholds to check between 0.01 and 0.99.

    Returns:
        tuple: (best_threshold, best_score)
    """
    best_threshold = 0.5
    best_score = -1.0
    thresholds = np.linspace(0.01, 0.99, steps) # Generate candidate thresholds

    for threshold in thresholds:
        # Apply threshold to get binary predictions
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        score = 0
        # Calculate the score for the chosen metric
        try:
            if metric == 'f1': score = f1_score(y_true, y_pred_thresh, pos_label=1, zero_division=0)
            elif metric == 'recall': score = recall_score(y_true, y_pred_thresh, pos_label=1, zero_division=0)
            elif metric == 'precision': score = precision_score(y_true, y_pred_thresh, pos_label=1, zero_division=0)
            else: raise ValueError(f"Unsupported metric for threshold tuning: {metric}")

            # Update best score and threshold if current score is better
            if score > best_score:
                best_score = score
                best_threshold = threshold
        except Exception as e:
            print(f"Warning: Error calculating score for threshold {threshold}: {e}")
            continue # Skip this threshold if score calculation fails

    return best_threshold, best_score


# --- Model Evaluation ---
def evaluate_model_generic(model, X_test, y_test, problem_type, tune_threshold_metric=None):
    """
    Evaluates the trained model on the test set using standard metrics.
    Optionally performs threshold tuning for binary classification to find
    a threshold maximizing a specified metric (f1, recall, precision).

    Args:
        model: The trained LightGBM model object.
        X_test (pd.DataFrame): Test features (preprocessed).
        y_test (pd.Series): True test target values.
        problem_type (str): 'binary' or 'regression'.
        tune_threshold_metric (str, optional): Metric to optimize for threshold tuning
                                               ('f1', 'recall', 'precision'). If None,
                                               uses the default 0.5 threshold. Defaults to None.
    """
    print("\n--- Evaluating Model on Test Set ---")
    optimal_threshold = 0.5 # Default threshold

    try:
        if problem_type == 'binary':
            # --- Binary Classification Evaluation ---
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilities for class 1
                y_pred_default = (y_pred_proba >= 0.5).astype(int) # Predictions at default 0.5

                # Calculate metrics at default threshold
                accuracy = accuracy_score(y_test, y_pred_default)
                try: auc = roc_auc_score(y_test, y_pred_proba)
                except ValueError: auc = float('nan') # Handle cases with only one class in y_test
                try: avg_precision = average_precision_score(y_test, y_pred_proba)
                except ValueError: avg_precision = float('nan')

                # Get target names for classification report
                unique_labels = sorted(np.unique(y_test))
                target_names = [f"Class {label}" for label in unique_labels]
                if set(unique_labels) == {0, 1}: target_names = ["Class 0 (Negative)", "Class 1 (Positive)"]

                print("\n--- Metrics (Default Threshold: 0.5) ---")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"AUC: {auc:.4f}")
                print(f"Average Precision: {avg_precision:.4f}")
                print("Classification Report:")
                print(classification_report(y_test, y_pred_default, zero_division=0, target_names=target_names))

                # --- Optional Threshold Tuning ---
                if tune_threshold_metric:
                    print(f"\n--- Finding Optimal Threshold (Maximizing: {tune_threshold_metric}) ---")
                    print("Note: Tuning threshold on the test set provides an optimistic performance estimate.")
                    best_threshold, best_score = find_optimal_threshold(y_test, y_pred_proba, metric=tune_threshold_metric)
                    optimal_threshold = best_threshold
                    print(f"Optimal Threshold Found: {optimal_threshold:.4f} (Yields {tune_threshold_metric}: {best_score:.4f})")

                    # Evaluate using the optimal threshold
                    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
                    print(f"\n--- Metrics (Optimal Threshold: {optimal_threshold:.4f}) ---")
                    print("Classification Report:")
                    print(classification_report(y_test, y_pred_optimal, zero_division=0, target_names=target_names))
            else:
                # Handle models without predict_proba (e.g., some regressors used for classification)
                print("Warning: Model lacks 'predict_proba'. Evaluating based on 'predict' output only.")
                y_pred = model.predict(X_test)
                # Ensure predictions are binary if possible
                if set(np.unique(y_pred)).issubset({0, 1}):
                     accuracy = accuracy_score(y_test, y_pred)
                     print(f"Accuracy: {accuracy:.4f}")
                     print("\nClassification Report:")
                     print(classification_report(y_test, y_pred, zero_division=0))
                else:
                     print("Warning: 'predict' output is not binary (0/1). Cannot calculate classification metrics.")

        elif problem_type == 'regression':
            # --- Regression Evaluation ---
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print("\n--- Regression Metrics ---")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R^2 Score: {r2:.4f}")
        else:
            print(f"Warning: Evaluation metrics not implemented for problem type: {problem_type}")

    except AttributeError as ae:
        print(f"Evaluation Error: Model may lack required prediction methods. {ae}")
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred during model evaluation: {e}")
        traceback.print_exc()

    print("--- Model Evaluation Finished ---")


# --- Artifact Saving ---
def save_model_and_info(model, output_dir, best_params=None, encoder=None, scaler=None):
    """
    Saves the trained model, fitted encoder, and fitted scaler together
    in a single 'model.joblib' file. Optionally saves the best hyperparameters
    to 'best_params.json'.

    Args:
        model: The trained model object.
        output_dir (str): The directory path to save the artifacts.
        best_params (dict, optional): Dictionary of best hyperparameters found by Optuna.
        encoder: The fitted encoder object (e.g., OrdinalEncoder).
        scaler: The fitted scaler object (e.g., StandardScaler).
    """
    print(f"\n--- Saving Artifacts ---")
    print(f"Output directory: {output_dir}")
    try:
        os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist

        # --- Combine Model, Encoder, Scaler into a Dictionary ---
        artifacts_to_save = {
            'model': model,
            'encoder': encoder,
            'scaler': scaler
        }
        model_filename = "model.joblib"
        model_path = os.path.join(output_dir, model_filename)
        joblib.dump(artifacts_to_save, model_path)
        print(f"Model, Encoder, and Scaler saved together to: {model_path}")

        # --- Save Best Hyperparameters (if provided) ---
        if best_params:
            # Convert numpy types to standard Python types for JSON serialization
            serializable_params = {}
            for k, v in best_params.items():
                if isinstance(v, (np.integer, np.int_)): serializable_params[k] = int(v)
                elif isinstance(v, (np.floating, np.float_)): serializable_params[k] = float(v)
                elif isinstance(v, (np.bool_, bool)): serializable_params[k] = bool(v)
                else: serializable_params[k] = v # Keep other types as is

            params_filename = "best_params.json"
            params_path = os.path.join(output_dir, params_filename)
            with open(params_path, 'w') as f:
                json.dump(serializable_params, f, indent=4) # Pretty print JSON
            print(f"Best hyperparameters saved to: {params_path}")

        print("--- Artifact Saving Finished ---")

    except Exception as e:
        print(f"Error saving artifacts: {e}")
        traceback.print_exc()

