# File: src/data/feature_analyzer.py
"""
Data Feature Analysis Module
----------------------------
Provides functions for analyzing DataFrame columns to:
1. Infer feature types (numerical vs. categorical) based on data types and heuristics.
2. Calculate plausible value ranges for numerical features.
"""

# --- Imports ---
import streamlit as st # Used only for potential warnings/info messages
import pandas as pd
import numpy as np

# --- Constants ---
# Threshold for unique values: numeric columns with unique values <= this
# threshold are initially suspected to be categorical.
LOW_CARDINALITY_THRESHOLD = 15

# --- Feature Type Inference ---
def infer_feature_types(df, target_column):
    """
    Automatically infers numerical and categorical features for a DataFrame,
    excluding the specified target column. Uses pandas dtypes and heuristics,
    such as treating low-cardinality numeric columns and boolean columns
    as categorical.

    Args:
        df (pd.DataFrame): The input DataFrame (should be cleaned of major issues).
        target_column (str): The name of the target variable column to exclude.

    Returns:
        tuple: (list_of_numerical_features, list_of_categorical_features)

    Raises:
        ValueError: If the target column is not found or if no feature columns remain
                    after dropping the target.
    """
    print("\n--- Starting Feature Type Inference ---")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame columns: {df.columns.tolist()}")

    # Exclude target column to get features DataFrame
    X = df.drop(columns=[target_column], errors='ignore') # Use errors='ignore' for safety
    if X.empty:
        raise ValueError("DataFrame has no feature columns remaining after dropping the target.")

    # 1. Initial classification based on broad pandas dtypes
    numeric_cols_initial = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_initial = X.select_dtypes(exclude=np.number).columns.tolist()
    print(f"Initial check: {len(numeric_cols_initial)} numeric-like, {len(categorical_cols_initial)} non-numeric.")

    final_numerical = []
    final_categorical = list(categorical_cols_initial) # Start with definite non-numerics

    # 2. Refine numeric list using heuristics
    print(f"Applying heuristics (Low Cardinality Threshold = {LOW_CARDINALITY_THRESHOLD})...")
    for col in numeric_cols_initial:
        unique_count = X[col].nunique()
        # Heuristic: Low unique count suggests categorical nature (e.g., ratings, flags)
        if unique_count <= LOW_CARDINALITY_THRESHOLD:
            if col not in final_categorical:
                final_categorical.append(col)
                print(f"  Reclassified '{col}' as CATEGORICAL (low unique values: {unique_count})")
        else:
            # High unique count, keep as numerical
            if col not in final_numerical:
                 final_numerical.append(col)

    # 3. Ensure explicitly boolean types are always categorical
    # This overrides the numeric check if a column has dtype 'bool'
    bool_cols_explicit = X.select_dtypes(include='bool').columns.tolist()
    for col in bool_cols_explicit:
        if col in final_numerical: # If it was wrongly classified as numeric
            final_numerical.remove(col)
            print(f"  Moved explicit boolean '{col}' from numeric list to categorical.")
        if col not in final_categorical: # Add if not already present
            final_categorical.append(col)
            print(f"  Confirmed explicit boolean '{col}' as CATEGORICAL.")

    # 4. Final Output and Sanity Check
    print(f"--- Feature Type Inference Complete ---")
    print(f"Final Numerical Features ({len(final_numerical)}): {final_numerical}")
    print(f"Final Categorical Features ({len(final_categorical)}): {final_categorical}")

    # Sanity check: Ensure no feature is listed in both categories
    overlap = set(final_numerical) & set(final_categorical)
    if overlap:
        print(f"WARNING: Overlap detected between final lists: {overlap}. Prioritizing as categorical.")
        # Remove overlapping items from numerical list
        final_numerical = [f for f in final_numerical if f not in overlap]

    return final_numerical, final_categorical


# --- Feature Range Calculation ---
def calculate_feature_ranges(df, numerical_features):
    """
    Calculates plausible value ranges (1st to 99th percentile) for specified
    numerical features in a DataFrame. Used for setting constraints in DiCE.

    Args:
        df (pd.DataFrame): The DataFrame containing the data (original values).
        numerical_features (list): List of column names identified as numerical.

    Returns:
        dict: A dictionary mapping numerical feature names to their [min, max]
              range (as floats). Returns an empty dict if no numerical features
              are provided or found.
    """
    print("\nCalculating feature ranges (1st-99th percentile) for numerical features...")
    ranges = {}
    if not numerical_features:
        print("No numerical features provided to calculate ranges for.")
        return ranges

    for feature in numerical_features:
        if feature not in df.columns:
            print(f"Warning: Numerical feature '{feature}' not found in DataFrame. Skipping range calculation.")
            continue

        # Ensure the column is actually numeric before calculating percentiles
        if pd.api.types.is_numeric_dtype(df[feature]):
            feature_data = df[feature].dropna() # Exclude NaNs for percentile calculation
            if not feature_data.empty:
                try:
                    # Calculate 1st and 99th percentiles
                    min_val, max_val = np.percentile(feature_data, [1, 99])

                    # --- Sanity Checks and Adjustments for Range Validity ---
                    # Ensure min <= max
                    if min_val > max_val: min_val, max_val = max_val, min_val
                    # Handle constant columns or near-constant columns
                    if np.isclose(min_val, max_val):
                        # Add a small buffer based on value or a fixed amount
                        buffer = abs(min_val * 0.01) if not np.isclose(min_val, 0) else 0.01
                        # Use actual data min/max as bounds for the buffer
                        data_min, data_max = feature_data.min(), feature_data.max()
                        min_val = max(data_min, min_val - buffer)
                        max_val = min(data_max, max_val + buffer)
                        # Ensure range is not zero after buffering
                        if np.isclose(min_val, max_val): max_val = min_val + buffer
                    # Ensure range respects non-negativity if data is non-negative
                    if min_val < 0 and feature_data.min() >= 0: min_val = 0.0
                    # Ensure range respects non-positivity if data is non-positive
                    if max_val > 0 and feature_data.max() <= 0: max_val = 0.0

                    ranges[feature] = [float(min_val), float(max_val)]
                    print(f"  Range for '{feature}': [{ranges[feature][0]:.4f}, {ranges[feature][1]:.4f}]")
                except Exception as e:
                    print(f"  Error calculating range for '{feature}': {e}")
            else:
                 print(f"Warning: Numerical feature '{feature}' contains only NaN values. Skipping range calculation.")
        else:
            print(f"Warning: Column '{feature}' listed as numerical is not actually numeric (dtype: {df[feature].dtype}). Skipping range calculation.")

    print(f"Feature ranges calculated for {len(ranges)} numerical features.")
    return ranges
