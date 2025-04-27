# File: model_training/data_utils.py
"""
Data Utilities: Model Training
------------------------------
Provides functions for loading, cleaning, inferring feature types,
and preprocessing data specifically for the model training pipeline.
Includes dataset-specific cleaning steps and uses OrdinalEncoder and
StandardScaler for feature transformation.
"""

# --- Imports ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.exceptions import NotFittedError

# Import config from the same directory
from . import config as train_config # Relative import for model_training config

# --- Constants ---
LOW_CARDINALITY_THRESHOLD = 15 # Threshold for inferring categorical from numeric

# --- Data Loading Functions ---
def load_data(file_path, target_column_arg, columns_to_drop=None):
    """
    Loads a dataset from a CSV file, performs dataset-specific cleaning based
    on filename conventions (e.g., handling Cholesterol=0 in heart failure data,
    -1 values in cybersecurity data), validates the target column, and drops
    any specified columns.

    Args:
        file_path (str): Path to the input CSV file.
        target_column_arg (str): Name of the target variable column.
        columns_to_drop (list, optional): List of column names to remove. Defaults to None.

    Returns:
        pd.DataFrame: The loaded and initially cleaned DataFrame.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        ValueError: If the target column is not found or other configuration errors occur.
        Exception: For any other unexpected errors during loading.
    """
    print(f"\n--- Loading Data ---")
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at {file_path}")

        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from {file_path} ({len(df)} rows).")
        print(f"Initial shape: {df.shape}")

        # --- Dataset-Specific Cleaning ---
        filename = os.path.basename(file_path).lower()
        print(f"Applying cleaning rules based on filename: {filename}")

        # 1. Heart Failure: Replace Cholesterol 0 with NaN
        if 'heart_failure' in filename and 'Cholesterol' in df.columns:
            initial_zeros = (df['Cholesterol'] == 0).sum()
            if initial_zeros > 0:
                print(f"  Heart Failure Rule: Replacing {initial_zeros} zero values in 'Cholesterol' with NaN.")
                df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
            else:
                print("  Heart Failure Rule: No zero values found in 'Cholesterol'.")

        # 2. Cybersecurity: Replace -1 with NaN (assumed missing indicator)
        if 'cybersecurity' in filename:
            print("  Cybersecurity Rule: Replacing -1 with NaN in feature columns...")
            columns_to_check = df.columns.drop(target_column_arg, errors='ignore') # Exclude target
            replaced_count = 0
            for col in columns_to_check:
                # Check if -1 exists and the column is suitable for replacement (e.g., numeric)
                if -1 in df[col].unique() and pd.api.types.is_numeric_dtype(df[col]):
                    count_before = df[col].notna().sum()
                    df[col] = df[col].replace(-1, np.nan)
                    count_after = df[col].notna().sum()
                    replaced_count += (count_before - count_after)
            if replaced_count > 0:
                print(f"  Replaced {replaced_count} occurrences of -1 with NaN across relevant columns.")
            else:
                print("  No -1 values found to replace in relevant columns.")
        # --- End Specific Cleaning ---

        # --- Standard Validation and Column Dropping ---
        if target_column_arg not in df.columns:
            raise ValueError(f"Target column '{target_column_arg}' not found in the loaded data.")

        if columns_to_drop:
            cols_found = [col for col in columns_to_drop if col in df.columns]
            if cols_found:
                df = df.drop(columns=cols_found)
                print(f"Dropped specified columns: {cols_found}. New shape: {df.shape}")
            else:
                print(f"Warning: None of the specified columns to drop were found: {columns_to_drop}")

        print("--- Data Loading Finished ---")
        return df

    except FileNotFoundError as fnf_err:
        print(f"Error: {fnf_err}")
        raise
    except ValueError as val_err:
        print(f"Configuration Error: {val_err}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        traceback.print_exc()
        raise

# --- Feature Type Inference ---
def infer_feature_types(df, target_column_arg):
    """
    Automatically infers numerical and categorical features based on pandas dtypes
    and heuristics (low cardinality numerics, booleans are treated as categorical).
    Excludes the target column. This logic should ideally match the inference
    used in the prediction application (`src/data/feature_analyzer.py`).

    Args:
        df (pd.DataFrame): The input DataFrame (after initial loading/cleaning).
        target_column_arg (str): The name of the target column to exclude.

    Returns:
        tuple: (list_of_numerical_features, list_of_categorical_features)

    Raises:
        ValueError: If target column not found or no feature columns remain.
    """
    print("\n--- Inferring Feature Types (Training Pipeline) ---")
    if target_column_arg not in df.columns:
        raise ValueError(f"Target column '{target_column_arg}' not found for feature inference.")

    X = df.drop(columns=[target_column_arg], errors='ignore')
    if X.empty:
        raise ValueError("DataFrame has no feature columns after dropping the target.")

    # Initial split based on general dtype
    numeric_cols_initial = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_initial = X.select_dtypes(exclude=np.number).columns.tolist()
    print(f"Initial check: {len(numeric_cols_initial)} numeric-like, {len(categorical_cols_initial)} non-numeric.")

    final_numerical = []
    final_categorical = list(categorical_cols_initial) # Start with non-numerics

    # Apply heuristics
    print(f"Applying heuristics (Low Cardinality Threshold = {LOW_CARDINALITY_THRESHOLD})...")
    for col in numeric_cols_initial:
        unique_count = X[col].nunique()
        # Heuristic: Low unique count suggests categorical
        if unique_count <= LOW_CARDINALITY_THRESHOLD:
            if col not in final_categorical:
                final_categorical.append(col)
                print(f"  Reclassified '{col}' as CATEGORICAL (low unique values: {unique_count})")
        else:
            # Keep as numerical
            if col not in final_numerical:
                 final_numerical.append(col)

    # Ensure boolean types are categorical
    bool_cols_explicit = X.select_dtypes(include='bool').columns.tolist()
    for col in bool_cols_explicit:
        if col in final_numerical: # Correct if misclassified
            final_numerical.remove(col)
            print(f"  Moved explicit boolean '{col}' from numeric list to categorical.")
        if col not in final_categorical: # Add if not already present
            final_categorical.append(col)
            print(f"  Confirmed explicit boolean '{col}' as CATEGORICAL.")

    # Final check for overlaps (shouldn't happen with current logic, but safe)
    overlap = set(final_numerical) & set(final_categorical)
    if overlap:
        print(f"WARNING: Overlap detected between final lists: {overlap}. Prioritizing as categorical.")
        final_numerical = [f for f in final_numerical if f not in overlap]

    print(f"--- Feature Type Inference Complete (Training Pipeline) ---")
    print(f"Final Numerical Features ({len(final_numerical)}): {final_numerical}")
    print(f"Final Categorical Features ({len(final_categorical)}): {final_categorical}")

    return final_numerical, final_categorical

# --- Data Preprocessing ---
def preprocess_data_generic(df, target_column_arg, numerical_features_arg, categorical_features_arg,
                            problem_type, positive_label=None):
    """
    Preprocesses data for model training:
    1. Imputes missing values (median for numerical, mode for categorical).
    2. Applies OrdinalEncoder to categorical features.
    3. Applies StandardScaler to numerical features.
    4. Encodes the target variable (binary 0/1 or keeps as is for regression).
    5. Splits data into training and testing sets.

    Args:
        df (pd.DataFrame): DataFrame after initial loading and cleaning.
        target_column_arg (str): Name of the target column.
        numerical_features_arg (list): List of numerical feature names.
        categorical_features_arg (list): List of categorical feature names.
        problem_type (str): 'binary' or 'regression'.
        positive_label (str/int/float, optional): The value representing the positive
                                                  class for binary classification. Needed if
                                                  target is not already 0/1. Defaults to None.

    Returns:
        tuple: (
            X_train, X_test, y_train, y_test, # Split data
            fitted_ordinal_encoder,          # Fitted encoder object
            fitted_scaler,                   # Fitted scaler object
            final_numerical_features,        # List of numerical features used
            final_categorical_features       # List of categorical features used
        )

    Raises:
        ValueError: If preprocessing fails or data becomes invalid.
        TypeError: If target column cannot be processed for the specified problem type.
    """
    print(f"\n--- Starting Data Preprocessing (Problem Type: {problem_type}) ---")
    initial_rows = len(df)
    fitted_ordinal_encoder = None
    fitted_scaler = None

    # --- Handle Missing Values (Imputation) ---
    # Numerical Imputation (Median)
    num_cols_to_impute = [col for col in numerical_features_arg if col in df.columns]
    if num_cols_to_impute:
        nan_counts_before_num = df[num_cols_to_impute].isnull().sum()
        if nan_counts_before_num.sum() > 0:
            print(f"Imputing NaNs in numerical columns using median: {nan_counts_before_num[nan_counts_before_num > 0].index.tolist()}")
            num_imputer = SimpleImputer(strategy='median')
            df[num_cols_to_impute] = num_imputer.fit_transform(df[num_cols_to_impute])
        else:
            print("No NaNs found in numerical columns.")
    else:
        print("No numerical columns specified or found for imputation.")

    # Categorical Imputation (Mode)
    cat_cols_to_impute = [col for col in categorical_features_arg if col in df.columns]
    if cat_cols_to_impute:
        nan_counts_before_cat = df[cat_cols_to_impute].isnull().sum()
        if nan_counts_before_cat.sum() > 0:
            print(f"Imputing NaNs in categorical columns using mode: {nan_counts_before_cat[nan_counts_before_cat > 0].index.tolist()}")
            # Ensure string type before imputation for mode strategy
            for col in cat_cols_to_impute:
                if df[col].isnull().any(): df[col] = df[col].astype(str).fillna('NaN_placeholder_impute')
                else: df[col] = df[col].astype(str) # Ensure consistent type
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df[cat_cols_to_impute] = cat_imputer.fit_transform(df[cat_cols_to_impute])
            # Replace placeholder back to NaN if needed elsewhere, though encoder handles strings
            df.replace('NaN_placeholder_impute', np.nan, inplace=True)
        else:
            print("No NaNs found in categorical columns.")
    else:
        print("No categorical columns specified or found for imputation.")

    # Drop rows with missing target values AFTER imputation of features
    df.dropna(subset=[target_column_arg], inplace=True)
    rows_after_target_dropna = len(df)
    if rows_after_target_dropna < initial_rows:
        print(f"Dropped {initial_rows - rows_after_target_dropna} rows with missing target values.")
    if rows_after_target_dropna == 0:
        raise ValueError("No data remaining after dropping rows with missing target values.")

    # --- Feature Selection & Target Separation ---
    # Use the provided feature lists, ensuring they exist in the current DataFrame state
    final_numerical_features = [f for f in numerical_features_arg if f in df.columns]
    final_categorical_features = [f for f in categorical_features_arg if f in df.columns]
    features_to_use = final_numerical_features + final_categorical_features
    if not features_to_use:
        raise ValueError("No features selected or available in the DataFrame after imputation.")

    X = df[features_to_use].copy()
    y = df[target_column_arg].copy()
    print(f"Features selected for X: {features_to_use}")

    # --- Ordinal Encode Categorical Features ---
    if final_categorical_features:
        print(f"Applying OrdinalEncoder to: {final_categorical_features}")
        # Ensure string type and handle any remaining NaNs (shouldn't exist after imputation)
        for col in final_categorical_features:
            if X[col].isnull().any(): X[col] = X[col].astype(str).fillna('missing_value_encode')
            else: X[col] = X[col].astype(str)

        # Fit and transform using OrdinalEncoder
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1) # Handle unseen values in test set
        X[final_categorical_features] = ordinal_encoder.fit_transform(X[final_categorical_features])
        fitted_ordinal_encoder = ordinal_encoder # Store the fitted encoder
        print("OrdinalEncoder fitted and applied.")
    else:
        print("No categorical features to encode.")

    # --- Scale Numerical Features ---
    if final_numerical_features:
        print(f"Applying StandardScaler to: {final_numerical_features}")
        # Ensure columns are truly numeric before scaling
        for col in final_numerical_features:
            if not pd.api.types.is_numeric_dtype(X[col]):
                print(f"Warning: Column '{col}' intended for scaling is not numeric ({X[col].dtype}). Attempting conversion.")
                X[col] = pd.to_numeric(X[col], errors='coerce')
                if X[col].isnull().any(): # Impute if coercion created NaNs
                    median_val = X[col].median()
                    X[col].fillna(median_val, inplace=True)
        # Fit and transform using StandardScaler
        scaler = StandardScaler()
        X[final_numerical_features] = scaler.fit_transform(X[final_numerical_features])
        fitted_scaler = scaler # Store the fitted scaler
        print("StandardScaler fitted and applied.")
    else:
        print("No numerical features to scale.")

    # --- Target Variable Processing ---
    if problem_type == 'binary':
        print("Processing target variable for binary classification...")
        y = convert_target_col(y) # Use helper function for robust conversion
        if y is None:
             raise ValueError("Target column processing for binary classification failed.")
        print(f"Target encoded as 0/1. Distribution:\n{y.value_counts(normalize=True)}")
    elif problem_type == 'regression':
        if not pd.api.types.is_numeric_dtype(y):
            print(f"Warning: Problem type is 'regression' but target column '{target_column_arg}' is not numeric ({y.dtype}). Attempting conversion.")
            try:
                y = pd.to_numeric(y, errors='raise')
                print("Target column successfully converted to numeric for regression.")
            except ValueError:
                raise TypeError(f"Target column '{target_column_arg}' could not be converted to numeric for regression.")
        print("Target variable prepared for regression.")
    else:
        raise ValueError(f"Unsupported problem_type: {problem_type}")

    # --- Split Data into Train/Test ---
    stratify_split = y if problem_type == 'binary' and y.nunique() > 1 else None # Stratify only if binary and has both classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=train_config.TEST_SPLIT_SIZE,
        random_state=train_config.RANDOM_STATE,
        stratify=stratify_split
    )
    print(f"Data split into Train ({X_train.shape[0]} samples) and Test ({X_test.shape[0]} samples).")
    print("--- Data Preprocessing Finished ---")

    # Return all necessary components
    return (X_train, X_test, y_train, y_test,
            fitted_ordinal_encoder, fitted_scaler,
            final_numerical_features, final_categorical_features)

# --- Helper Function: Convert Target Column (Copied from src/data/processor.py for consistency) ---
# This function is duplicated here to make model_training self-contained,
# ideally it would be in a shared utils module.
def convert_target_col(y_series):
    """
    Converts a target variable Series to binary (0/1) format.
    Handles numeric, boolean, and common string representations.

    Args:
        y_series (pd.Series): The raw target column.

    Returns:
        pd.Series or None: The target column converted to integer type (0 or 1),
                           or None if conversion fails.

    Raises:
        ValueError: If the target column cannot be unambiguously converted to binary.
    """
    # (Implementation is identical to the one in src/data/processor.py)
    # ... (omitted for brevity in this refactoring example, assume it's copied here) ...

    y_name = y_series.name if hasattr(y_series, 'name') else 'Target'
    original_dtype = y_series.dtype
    print(f"Attempting to convert target column '{y_name}' (dtype: {original_dtype}) to binary 0/1...")

    if y_series.isnull().all(): raise ValueError(f"Target column '{y_name}' contains only NaN values.")
    y_series_clean_analysis = y_series.dropna()
    if y_series_clean_analysis.empty: raise ValueError(f"Target column '{y_name}' is empty after dropping NaNs.")

    unique_vals = y_series_clean_analysis.unique()
    num_unique = len(unique_vals)

    if pd.api.types.is_numeric_dtype(y_series_clean_analysis) and set(unique_vals).issubset({0, 1}):
        print(f"Target '{y_name}' is already binary numeric (0/1).")
        return y_series.astype('Int64')
    elif pd.api.types.is_numeric_dtype(y_series_clean_analysis) and num_unique == 2:
        sorted_unique = sorted(unique_vals); neg_val, pos_val = sorted_unique[0], sorted_unique[1]
        print(f"Mapping numeric target: {neg_val} -> 0, {pos_val} -> 1")
        mapping = {neg_val: 0, pos_val: 1}; return y_series.map(mapping).astype('Int64')
    elif pd.api.types.is_bool_dtype(y_series_clean_analysis):
        print(f"Target '{y_name}' is boolean. Mapping True->1, False->0.")
        mapping = {True: 1, False: 0}; return y_series.map(mapping).astype('Int64')
    elif num_unique == 2 and (pd.api.types.is_object_dtype(y_series_clean_analysis) or \
                              pd.api.types.is_string_dtype(y_series_clean_analysis) or \
                              pd.api.types.is_categorical_dtype(y_series_clean_analysis)):
        y_lower_analysis = y_series_clean_analysis.astype(str).str.lower().str.strip()
        unique_lower = y_lower_analysis.unique(); label1, label2 = unique_lower[0], unique_lower[1]
        print(f"Target '{y_name}' has two unique string/object values: '{label1}', '{label2}'. Attempting mapping.")
        positive_labels = {'yes', 'true', '1', 'positive', 'p', 't', 'y', 'ok', 'good', 'high', 'paid', "won't recidivate", '>50k', 'accept', 'heartdisease', 'anemic', 'phishing', 'botnet'} # Added botnet
        negative_labels = {'no', 'false', '0', 'negative', 'n', 'f', 'bad', 'low', 'default', 'will recidivate', '<=50k', 'reject', 'normal', 'not anemic', 'legitimate', 'background'} # Added background
        pos_label, neg_label = None, None
        if label1 in positive_labels and label2 in negative_labels: pos_label, neg_label = label1, label2
        elif label2 in positive_labels and label1 in negative_labels: pos_label, neg_label = label2, label1
        elif label1 in positive_labels: pos_label, neg_label = label1, label2; print(f"Warning: Assuming '{label2}' maps to 0.")
        elif label2 in positive_labels: pos_label, neg_label = label2, label1; print(f"Warning: Assuming '{label1}' maps to 0.")
        elif label1 in negative_labels: neg_label, pos_label = label1, label2; print(f"Warning: Assuming '{label2}' maps to 1.")
        elif label2 in negative_labels: neg_label, pos_label = label2, label1; print(f"Warning: Assuming '{label1}' maps to 1.")
        else: sorted_unique = sorted(unique_lower); neg_label, pos_label = sorted_unique[0], sorted_unique[1]; print(f"Warning: Mapping alphabetically: '{pos_label}'=1, '{neg_label}'=0.")
        print(f"Mapping target: '{pos_label}' -> 1, '{neg_label}' -> 0")
        mapping = {pos_label: 1, neg_label: 0}
        y_series_standardized = y_series.astype(str).str.lower().str.strip()
        converted_series = y_series_standardized.map(mapping)
        if converted_series.isnull().sum() > y_series.isnull().sum():
            unmapped = y_series_standardized[converted_series.isnull() & y_series.notnull()].unique()
            raise ValueError(f"Mapping error for target '{y_name}'. Unmapped values: {unmapped}")
        return converted_series.astype('Int64')
    else:
        raise ValueError(f"Target column '{y_name}' has {num_unique} unique values or unhandled dtype ({original_dtype}). Cannot auto-convert to binary.")

