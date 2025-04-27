# File: src/evaluate.py
"""
Counterfactual Explanation Evaluation Module
-------------------------------------------
Provides functions to calculate quantitative metrics for evaluating the quality
of generated counterfactual explanations: Sparsity, Proximity, Diversity, and Validity.
Includes internal preprocessing to ensure metrics are calculated consistently
on appropriately encoded and scaled data, matching the model's input format.
"""

# --- Imports ---
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.exceptions import NotFittedError
import warnings
import traceback

# --- Internal Helper: Preprocessing for Evaluation ---
def _preprocess_for_eval(df, encoder=None, scaler=None,
                         numerical_features=None, categorical_features=None,
                         expected_features=None):
    """
    Preprocesses a DataFrame (original instance or counterfactuals)
    using provided encoder and scaler for consistent metric calculation.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
        encoder: Fitted encoder object (e.g., OrdinalEncoder).
        scaler: Fitted scaler object (e.g., StandardScaler).
        numerical_features (list): List of numerical feature names.
        categorical_features (list): List of categorical feature names.
        expected_features (list): The full list of feature names in the order
                                  the model expects.

    Returns:
        pd.DataFrame or None: The preprocessed DataFrame with numerical features
                              scaled and categorical features encoded, ready for
                              distance calculations or model prediction, or None
                              if preprocessing fails.
    """
    if df is None or df.empty:
        print("Error (_preprocess_for_eval): Input DataFrame is None or empty.")
        return None
    if not expected_features:
        print("Warning (_preprocess_for_eval): Missing expected_features list. Using input columns.")
        expected_features = df.columns.tolist() # Fallback

    # Ensure only expected features are present and in the correct order
    missing_cols = [f for f in expected_features if f not in df.columns]
    if missing_cols:
        print(f"Error (_preprocess_for_eval): Input DataFrame missing expected columns: {missing_cols}")
        return None
    X = df[expected_features].copy() # Work on a copy

    # --- Encoding ---
    if encoder and categorical_features:
        cats_present = [f for f in categorical_features if f in X.columns]
        if cats_present:
            try:
                # Ensure string type and handle NaNs before encoding
                for col in cats_present:
                    if X[col].isnull().any():
                        mode_val = X[col].mode()[0] if not X[col].mode().empty else 'missing'
                        X[col].fillna(mode_val, inplace=True)
                    X[col] = X[col].astype(str)
                # Apply transform using the fitted encoder
                encoded_values = encoder.transform(X[cats_present])
                X[cats_present] = encoded_values.astype(np.float64)
            except Exception as e:
                 print(f"Error applying encoder during eval preprocessing: {e}")
                 traceback.print_exc() # Print detailed traceback
                 return None # Fail preprocessing

    # --- Scaling ---
    if scaler and numerical_features:
        nums_present = [f for f in numerical_features if f in X.columns]
        if nums_present:
             try:
                  # Ensure numeric and handle NaNs before scaling
                  for col in nums_present:
                       if not pd.api.types.is_numeric_dtype(X[col]):
                            X[col] = pd.to_numeric(X[col], errors='coerce')
                       if X[col].isnull().any():
                            median_val = X[col].median()
                            X[col].fillna(median_val, inplace=True)
                  # Apply transform using the fitted scaler
                  scaled_values = scaler.transform(X[nums_present])
                  X[nums_present] = scaled_values
             except Exception as e:
                  print(f"Error applying scaler during eval preprocessing: {e}")
                  traceback.print_exc() # Print detailed traceback
                  return None # Fail preprocessing

    # --- Final Check: Ensure all columns are numeric after processing ---
    final_non_numeric = X.select_dtypes(exclude=np.number).columns
    if not final_non_numeric.empty:
        print(f"Error (_preprocess_for_eval): Result has non-numeric columns: {final_non_numeric.tolist()}")
        print("Dtypes after processing:")
        print(X.dtypes)
        return None

    return X


# --- Evaluation Metric: Sparsity ---
def calculate_sparsity(original_instance_df, cfs_df):
    """
    Calculates the average sparsity of counterfactuals relative to the original.
    Sparsity = 1 - (average fraction of features changed).
    Compares original feature values directly (no preprocessing needed).

    Args:
        original_instance_df (pd.DataFrame): DataFrame containing the single original instance.
        cfs_df (pd.DataFrame): DataFrame containing the generated counterfactual instances.

    Returns:
        float or None: Average sparsity score (higher is better, closer to 1.0 means fewer changes),
                       or None if input is invalid.
    """
    if cfs_df is None or cfs_df.empty or original_instance_df is None or original_instance_df.empty:
        print("Warning (Sparsity): Input DataFrame(s) are invalid.")
        return None
    try:
        common_cols = original_instance_df.columns.intersection(cfs_df.columns)
        if len(common_cols) == 0:
            print("Warning (Sparsity): No common columns found between original and counterfactuals.")
            return None

        # Extract values for comparison
        original_vals = original_instance_df[common_cols].iloc[0]
        cfs_vals = cfs_df[common_cols]
        n_features = len(common_cols)
        if n_features == 0: return 1.0 # No features to change

        # Count differences row-wise
        num_changed = cfs_vals.ne(original_vals, axis=1).sum(axis=1)
        avg_fraction_changed = num_changed.mean() / n_features
        sparsity = 1.0 - avg_fraction_changed
        return sparsity

    except Exception as e:
        print(f"Error calculating sparsity: {e}")
        traceback.print_exc()
        return None


# --- Evaluation Metric: Proximity ---
def calculate_proximity(original_instance_df, cfs_df, data_info, encoder=None, scaler=None):
    """
    Calculates the average proximity (distance) between the original instance
    and the generated counterfactuals. Uses L1 (Manhattan) distance on
    preprocessed (encoded/scaled) data.

    Args:
        original_instance_df (pd.DataFrame): DataFrame with the single original instance.
        cfs_df (pd.DataFrame): DataFrame with the generated counterfactuals.
        data_info (dict): Dictionary containing model feature lists.
        encoder: Fitted encoder object.
        scaler: Fitted scaler object.

    Returns:
        float or None: Average L1 distance (lower is better), or None on error.
    """
    if cfs_df is None or cfs_df.empty or original_instance_df is None or original_instance_df.empty:
        print("Warning (Proximity): Input DataFrame(s) are invalid.")
        return None

    try:
        # Get feature lists expected by the model from data_info
        num_feat = data_info.get('_model_num_features', [])
        cat_feat = data_info.get('_model_cat_features', [])
        expected_features = data_info.get('_model_expected_features', list(original_instance_df.columns))

        # Preprocess both original and counterfactuals for consistent comparison
        original_processed = _preprocess_for_eval(
            original_instance_df, encoder, scaler, num_feat, cat_feat, expected_features
        )
        cfs_processed = _preprocess_for_eval(
            cfs_df, encoder, scaler, num_feat, cat_feat, expected_features
        )

        if original_processed is None or cfs_processed is None:
            print("Error (Proximity): Preprocessing failed.")
            return None

        # Calculate Manhattan distances on processed data
        distances = manhattan_distances(original_processed.values, cfs_processed.values)

        # distances[0] contains L1 distances from the original to each counterfactual
        avg_proximity = np.mean(distances[0])
        return avg_proximity

    except Exception as e:
        print(f"Error calculating proximity: {e}")
        traceback.print_exc()
        return None


# --- Evaluation Metric: Diversity ---
def calculate_diversity(cfs_df, data_info, encoder=None, scaler=None):
    """
    Calculates the average diversity (pairwise distance) among counterfactuals.
    Uses L1 (Manhattan) distance between pairs of counterfactuals on
    preprocessed (encoded/scaled) data.

    Args:
        cfs_df (pd.DataFrame): DataFrame containing the generated counterfactuals.
        data_info (dict): Dictionary containing model feature lists.
        encoder: Fitted encoder object.
        scaler: Fitted scaler object.

    Returns:
        float or None: Average pairwise L1 distance (higher is better),
                       or None if fewer than 2 CFs or on error.
    """
    if cfs_df is None or len(cfs_df) < 2:
        # Diversity requires at least 2 counterfactuals to compare
        return None

    try:
        # Get feature lists expected by the model
        num_feat = data_info.get('_model_num_features', [])
        cat_feat = data_info.get('_model_cat_features', [])
        expected_features = data_info.get('_model_expected_features', list(cfs_df.columns))

        # Preprocess counterfactuals
        cfs_processed = _preprocess_for_eval(
            cfs_df, encoder, scaler, num_feat, cat_feat, expected_features
        )

        if cfs_processed is None:
            print("Error (Diversity): Preprocessing failed.")
            return None

        # Calculate pairwise Manhattan distances on processed data
        distances = manhattan_distances(cfs_processed.values)

        # Extract upper triangle (excluding the diagonal) to get unique pairs
        k = len(cfs_processed)
        num_pairs = k * (k - 1) / 2
        if num_pairs <= 0: return 0.0 # Should not happen if len(cfs_df) >= 2

        # Sum distances of unique pairs and calculate the average
        avg_diversity = np.sum(np.triu(distances, k=1)) / num_pairs
        return avg_diversity

    except Exception as e:
        print(f"Error calculating diversity: {e}")
        traceback.print_exc()
        return None


# --- Evaluation Metric: Validity ---
def check_validity(cfs_df, model, desired_class, data_info, encoder=None, scaler=None):
    """
    Checks the fraction of generated counterfactuals that the provided model
    predicts as the desired class, after appropriate preprocessing.

    Args:
        cfs_df (pd.DataFrame): DataFrame containing the generated counterfactuals.
        model: The trained machine learning model object.
        desired_class (int): The target class value (0 or 1) the counterfactuals
                             should predict.
        data_info (dict): Dictionary containing model feature lists.
        encoder: Fitted encoder object.
        scaler: Fitted scaler object.

    Returns:
        float or None: Fraction of valid counterfactuals (prediction == desired_class),
                       or None on error.
    """
    if cfs_df is None or cfs_df.empty:
        print("Warning (Validity): Input counterfactuals DataFrame is invalid.")
        return None
    if model is None:
        print("Warning (Validity): Model object is None.")
        return None

    try:
        # Get feature lists expected by the model
        num_feat = data_info.get('_model_num_features', [])
        cat_feat = data_info.get('_model_cat_features', [])
        expected_features = data_info.get('_model_expected_features', list(cfs_df.columns))

        # Preprocess counterfactuals to match model input format
        cfs_processed = _preprocess_for_eval(
            cfs_df, encoder, scaler, num_feat, cat_feat, expected_features
        )

        if cfs_processed is None:
            print("Error (Validity): Preprocessing failed.")
            return None

        # Make predictions using the preprocessed data
        predictions = None
        with warnings.catch_warnings(): # Suppress potential feature name warnings
            warnings.simplefilter("ignore", category=UserWarning)
            # Convert to NumPy for robustness with sklearn/lgbm models
            cfs_processed_np = cfs_processed.to_numpy()

            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(cfs_processed_np)
                # Assuming binary classification, class 1 is the second column
                # Use standard 0.5 threshold for prediction
                predictions = (probabilities[:, 1] > 0.5).astype(int)
            elif hasattr(model, 'predict'):
                predictions = model.predict(cfs_processed_np)
            else:
                print("Error (Validity): Model lacks 'predict_proba' and 'predict' methods.")
                return None

        # Calculate the fraction matching the desired class
        num_valid = np.sum(predictions == desired_class)
        validity_fraction = num_valid / len(cfs_df)
        return validity_fraction

    except Exception as e:
        print(f"Error checking validity: {e}")
        traceback.print_exc()
        return None