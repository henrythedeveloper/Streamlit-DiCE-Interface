# File: src/data/processor.py
"""
Data Processing Module for Model and Explainer
----------------------------------------------
Provides functions to process the loaded dataset for two main purposes:
1. Preparing features for input into the trained machine learning model
   (applying encoding and scaling using loaded artifacts).
2. Preparing data for the DiCE explainer (maintaining original values where needed).
Also includes utilities for target variable conversion and extracting expected
features from a model object.
"""

# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
import traceback # For detailed error logging

# --- Main Data Processing Function ---
def process_uploaded_data(df_full, data_info, model=None, encoder=None, scaler=None):
    """
    Processes the full dataset using loaded encoder and scaler artifacts.
    Separates data into formats suitable for model prediction and DiCE.

    Args:
        df_full (pd.DataFrame): The loaded, cleaned DataFrame with original values.
        data_info (dict): Dictionary containing feature lists (user-defined and model-expected),
                          target column name, and model type.
        model: The trained model object (used to get expected features if needed).
        encoder: The fitted OrdinalEncoder object (loaded from training).
        scaler: The fitted StandardScaler object (loaded from training).

    Returns:
        tuple: (
            X_processed_for_model (pd.DataFrame or None): Features encoded & scaled for prediction.
            y_app (pd.Series or None): Processed target variable (binary 0/1).
            scaler (object or None): The scaler object that was used (passed in).
            dice_data_df (pd.DataFrame or None): Data with original values for DiCE Data object.
            None: Placeholder (previously fitted_encoder, now passed in).
            categorical_indices (list or None): Indices of categorical features in X_processed_for_model.
            None: Placeholder (previously encoded_categorical_values).
        ) Returns None for components if processing fails at any critical step.
    """
    print("\n--- Starting Data Processing for Model and DiCE ---")
    # Extract necessary info from data_info dictionary
    target_col = data_info.get('target_column')
    user_num_features = data_info.get('numerical_features', [])
    user_cat_features = data_info.get('categorical_features', [])
    model_expected_features = data_info.get('_model_expected_features', [])
    model_num_features = data_info.get('_model_num_features', [])
    model_cat_features = data_info.get('_model_cat_features', [])

    # --- Input Validation ---
    if not target_col: st.error("Target column name missing in data_info."); return None, None, None, None, None, None, None
    if not model_expected_features: st.error("Model expected features list missing in data_info."); return None, None, None, None, None, None, None
    if target_col not in df_full.columns: st.error(f"Target column '{target_col}' not found in input DataFrame."); return None, None, None, None, None, None, None

    # --- 1. Prepare DataFrame for DiCE (Original Values + Target) ---
    # Select features defined by user (numerical + categorical) + target column
    features_for_dice = user_num_features + user_cat_features
    cols_to_keep_dice = [f for f in features_for_dice if f in df_full.columns] + [target_col]
    missing_dice_cols = [c for c in cols_to_keep_dice if c not in df_full.columns]
    if missing_dice_cols:
        st.warning(f"Columns specified in config but missing from loaded data for DiCE: {missing_dice_cols}")
        cols_to_keep_dice = [c for c in cols_to_keep_dice if c in df_full.columns] # Adjust list

    try:
        dice_data_df = df_full[cols_to_keep_dice].copy().reset_index(drop=True)
        print(f"Created dice_data_df (for DiCE Data obj) with shape: {dice_data_df.shape}")
    except KeyError as ke:
        st.error(f"Error selecting columns for DiCE DataFrame: {ke}")
        return None, None, None, None, None, None, None

    # --- 2. Prepare Target Variable (Convert to 0/1) ---
    try:
        y_app_raw = df_full[target_col]
        y_app = convert_target_col(y_app_raw.copy()) # Convert to binary 0/1
        if y_app is None: raise ValueError("Target column conversion failed.")
        print("Target variable y_app prepared (binary 0/1).")
    except Exception as e:
        st.error(f"Error processing target column '{target_col}': {e}")
        return None, None, None, None, None, None, None

    # --- 3. Prepare Features for Model Prediction (Apply Encoding/Scaling) ---
    # Select only the features the model expects, in the correct order
    missing_model_features = [f for f in model_expected_features if f not in df_full.columns]
    if missing_model_features:
        st.error(f"Input DataFrame is missing features the model expects: {missing_model_features}")
        return None, None, None, None, None, None, None

    X_processed_for_model = df_full[model_expected_features].copy()
    print(f"Created initial X_processed_for_model with shape: {X_processed_for_model.shape}")

    # Apply Ordinal Encoding (using loaded encoder)
    if model_cat_features and encoder:
        print(f"Applying loaded OrdinalEncoder to features: {model_cat_features}")
        cats_to_encode = [f for f in model_cat_features if f in X_processed_for_model.columns]
        if cats_to_encode:
            try:
                # Handle NaNs and ensure string type before encoding
                for col in cats_to_encode:
                    if X_processed_for_model[col].isnull().any():
                        mode_val = X_processed_for_model[col].mode()[0] if not X_processed_for_model[col].mode().empty else 'missing'
                        X_processed_for_model[col].fillna(mode_val, inplace=True)
                    X_processed_for_model[col] = X_processed_for_model[col].astype(str)
                # Apply the transform method of the loaded encoder
                encoded_data = encoder.transform(X_processed_for_model[cats_to_encode])
                X_processed_for_model[cats_to_encode] = encoded_data.astype(np.float64) # Ensure float for consistency
                print(f"Successfully applied OrdinalEncoder.")
            except NotFittedError:
                st.error("Error: The loaded encoder is not fitted. Model artifact might be corrupted or from incomplete training.")
                return None, None, None, None, None, None, None
            except ValueError as ve:
                st.error(f"Error applying encoder: {ve}. Check if data contains values not seen during training.")
                traceback.print_exc()
                return None, None, None, None, None, None, None
            except Exception as e:
                st.error(f"Unexpected error during encoding: {e}")
                traceback.print_exc()
                return None, None, None, None, None, None, None
        else:
            print("No categorical features found in X_processed_for_model matching model's expected list.")
    elif model_cat_features and not encoder:
        st.warning("Model expects categorical features, but no encoder was loaded. Using data as-is for these columns.")
    else:
        print("No categorical features expected by model or no encoder provided.")

    # Apply Scaling (using loaded scaler)
    if model_num_features and scaler:
        print(f"Applying loaded StandardScaler to features: {model_num_features}")
        nums_to_scale = [f for f in model_num_features if f in X_processed_for_model.columns]
        if nums_to_scale:
            try:
                # Ensure columns are numeric and impute NaNs before scaling
                for col in nums_to_scale:
                    if not pd.api.types.is_numeric_dtype(X_processed_for_model[col]):
                        X_processed_for_model[col] = pd.to_numeric(X_processed_for_model[col], errors='coerce')
                    if X_processed_for_model[col].isnull().any():
                        median_val = X_processed_for_model[col].median()
                        X_processed_for_model[col].fillna(median_val, inplace=True)
                # Apply the transform method of the loaded scaler
                scaled_numerical = scaler.transform(X_processed_for_model[nums_to_scale])
                X_processed_for_model[nums_to_scale] = scaled_numerical
                print(f"Successfully applied StandardScaler.")
            except NotFittedError:
                st.error("Error: The loaded scaler is not fitted. Model artifact might be corrupted or from incomplete training.")
                return None, None, None, None, None, None, None
            except ValueError as ve:
                st.error(f"Error applying loaded scaler: {ve}. Check feature names, order, and dtypes.")
                traceback.print_exc()
                return None, None, None, None, None, None, None
            except Exception as e:
                st.error(f"Unexpected error applying loaded scaler: {e}")
                traceback.print_exc()
                return None, None, None, None, None, None, None
        else:
            print("No numerical features found in X_processed_for_model matching model's expected list.")
    elif model_num_features and not scaler:
        st.warning("Model expects numerical features, but no scaler was loaded. Using unscaled data for these columns.")
    else:
        print("No numerical features expected by model or no scaler provided.")

    # --- Final Checks and Return ---
    # Verify all columns in the processed DataFrame are numeric
    final_non_numeric = X_processed_for_model.select_dtypes(exclude=np.number).columns
    if not final_non_numeric.empty:
        st.error(f"Internal Error: Post-processing resulted in non-numeric columns: {final_non_numeric.tolist()}. Check encoding steps.")
        print("Dtypes of X_processed_for_model before returning:")
        X_processed_for_model.info(buf=open(os.devnull, 'w')) # Suppress print to console
        return None, None, None, None, None, None, None

    # Get indices of categorical features in the final processed DataFrame
    categorical_indices = [X_processed_for_model.columns.get_loc(col) for col in model_cat_features if col in X_processed_for_model.columns]

    print("--- Finished Data Processing ---")
    return X_processed_for_model, y_app, scaler, dice_data_df, None, categorical_indices, None


# --- Helper Function: Extract Model Features ---
def extract_model_features(model):
    """
    Attempts to extract the list of feature names the model expects from its attributes.

    Args:
        model: The trained model object.

    Returns:
        list or None: A list of expected feature names, or None if they cannot be determined.
    """
    model_expected_features = None
    try:
        if hasattr(model, 'feature_name_'):
            model_expected_features = list(model.feature_name_)
        elif hasattr(model, 'feature_names_in_'):
            model_expected_features = list(model.feature_names_in_)
        # Add checks for other framework-specific attributes if needed

        if model_expected_features:
            print(f"Extracted expected features from model attributes: {len(model_expected_features)} features.")
        else:
            print("Warning: Could not extract feature names from model attributes (e.g., 'feature_name_', 'feature_names_in_').")
    except Exception as e:
        print(f"Warning: Error extracting feature names from model: {e}")

    return model_expected_features


# --- Helper Function: Convert Target Column ---
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
    y_name = y_series.name if hasattr(y_series, 'name') else 'Target'
    original_dtype = y_series.dtype
    print(f"Attempting to convert target column '{y_name}' (dtype: {original_dtype}) to binary 0/1...")

    # Handle fully null target
    if y_series.isnull().all():
        raise ValueError(f"Target column '{y_name}' contains only NaN values.")

    y_series_clean_analysis = y_series.dropna()
    if y_series_clean_analysis.empty:
        raise ValueError(f"Target column '{y_name}' is empty after dropping NaNs.")

    unique_vals = y_series_clean_analysis.unique()
    num_unique = len(unique_vals)

    # Case 1: Already numeric 0/1
    if pd.api.types.is_numeric_dtype(y_series_clean_analysis) and set(unique_vals).issubset({0, 1}):
        print(f"Target '{y_name}' is already binary numeric (0/1).")
        # Convert to nullable integer type for consistency
        return y_series.astype('Int64')

    # Case 2: Numeric with two unique values (e.g., 1/2, -1/1)
    elif pd.api.types.is_numeric_dtype(y_series_clean_analysis) and num_unique == 2:
        sorted_unique = sorted(unique_vals)
        neg_val, pos_val = sorted_unique[0], sorted_unique[1]
        st.warning(f"Numeric target '{y_name}' has values {neg_val}/{pos_val}. Mapping {neg_val}->0, {pos_val}->1.")
        print(f"Mapping numeric target: {neg_val} -> 0, {pos_val} -> 1")
        mapping = {neg_val: 0, pos_val: 1}
        return y_series.map(mapping).astype('Int64')

    # Case 3: Boolean
    elif pd.api.types.is_bool_dtype(y_series_clean_analysis):
        print(f"Target '{y_name}' is boolean. Mapping True->1, False->0.")
        mapping = {True: 1, False: 0}
        return y_series.map(mapping).astype('Int64')

    # Case 4: String/Object/Categorical with two unique values
    elif num_unique == 2 and (pd.api.types.is_object_dtype(y_series_clean_analysis) or \
                              pd.api.types.is_string_dtype(y_series_clean_analysis) or \
                              pd.api.types.is_categorical_dtype(y_series_clean_analysis)):

        y_lower_analysis = y_series_clean_analysis.astype(str).str.lower().str.strip()
        unique_lower = y_lower_analysis.unique()
        label1, label2 = unique_lower[0], unique_lower[1]
        print(f"Target '{y_name}' has two unique string/object values: '{label1}', '{label2}'. Attempting mapping.")

        # Define common positive/negative labels (case-insensitive)
        positive_labels = {'yes', 'true', '1', 'positive', 'p', 't', 'y', 'ok', 'good', 'high', 'paid', "won't recidivate", '>50k', 'accept', 'heartdisease', 'anemic', 'phishing'}
        negative_labels = {'no', 'false', '0', 'negative', 'n', 'f', 'bad', 'low', 'default', 'will recidivate', '<=50k', 'reject', 'normal', 'not anemic', 'legitimate'}

        pos_label, neg_label = None, None

        # Attempt to identify positive/negative based on common labels
        if label1 in positive_labels and label2 in negative_labels: pos_label, neg_label = label1, label2
        elif label2 in positive_labels and label1 in negative_labels: pos_label, neg_label = label2, label1
        elif label1 in positive_labels: pos_label, neg_label = label1, label2; st.warning(f"Assuming '{label2}' maps to 0 for target '{y_name}'.")
        elif label2 in positive_labels: pos_label, neg_label = label2, label1; st.warning(f"Assuming '{label1}' maps to 0 for target '{y_name}'.")
        elif label1 in negative_labels: neg_label, pos_label = label1, label2; st.warning(f"Assuming '{label2}' maps to 1 for target '{y_name}'.")
        elif label2 in negative_labels: neg_label, pos_label = label2, label1; st.warning(f"Assuming '{label1}' maps to 1 for target '{y_name}'.")
        else:
            # Fallback: Map alphabetically if common labels aren't found
            sorted_unique = sorted(unique_lower); neg_label, pos_label = sorted_unique[0], sorted_unique[1]
            st.warning(f"Labels '{label1}'/'{label2}' not standard positive/negative. Mapping alphabetically: '{pos_label}'=1, '{neg_label}'=0.")

        print(f"Mapping target: '{pos_label}' -> 1, '{neg_label}' -> 0")
        mapping = {pos_label: 1, neg_label: 0}

        # Apply mapping case-insensitively
        y_series_standardized = y_series.astype(str).str.lower().str.strip()
        converted_series = y_series_standardized.map(mapping)

        # Check if mapping introduced new NaNs (shouldn't happen if num_unique was 2)
        if converted_series.isnull().sum() > y_series.isnull().sum():
             unmapped_values = y_series_standardized[converted_series.isnull() & y_series.notnull()].unique()
             raise ValueError(f"Mapping error during target conversion for '{y_name}'. Some values could not be mapped: {unmapped_values}")
        return converted_series.astype('Int64')

    # Case 5: Unhandled type or number of unique values
    else:
        raise ValueError(f"Target column '{y_name}' has {num_unique} unique values or an unhandled dtype ({original_dtype}). Cannot automatically convert to binary (0/1). Please ensure the target column has exactly two distinct values suitable for binary classification.")

