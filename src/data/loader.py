# File: src/data/loader.py
"""
Data Loading Module
-------------------
Handles loading data from a specified CSV file path for the Streamlit application.
Includes basic validation (file existence, non-empty, target column presence)
and optional dropping of specified columns. Also includes basic NaN handling.
"""

# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import os
import traceback # For detailed error logging

# --- Main Data Loading Function ---
def load_user_data(data_file_path, target_column_name, columns_to_drop=None):
    """
    Loads data from a specified CSV file path, performs basic validation,
    optionally drops columns, and handles rows with missing values.

    Args:
        data_file_path (str): The absolute or relative path to the CSV data file.
        target_column_name (str): Name of the target variable column (for validation).
        columns_to_drop (list, optional): List of column names to drop after loading.
                                          Defaults to None.

    Returns:
        pd.DataFrame or None: The loaded and cleaned DataFrame, or None if any
                              critical error occurs during loading or validation.
    """
    # --- Input Validation ---
    if not data_file_path or not isinstance(data_file_path, str):
        st.error("No valid data file path was provided.")
        print("Error (load_user_data): Invalid data_file_path.")
        return None

    if not os.path.exists(data_file_path):
        st.error(f"Data file not found at the specified path: {data_file_path}")
        print(f"Error (load_user_data): File not found at {data_file_path}.")
        return None

    # --- Data Loading ---
    try:
        print(f"Loading data from file: {data_file_path}")
        df = pd.read_csv(data_file_path)
        print(f"Data loaded successfully. Initial shape: {df.shape}")

        # --- Basic DataFrame Validation ---
        if df.empty:
            st.warning("The loaded data file appears to be empty.")
            print("Warning (load_user_data): Loaded DataFrame is empty.")
            return None # Return None for empty file
        if target_column_name not in df.columns:
            st.error(f"The specified target column '{target_column_name}' was not found in the loaded data file.")
            print(f"Error (load_user_data): Target column '{target_column_name}' not found.")
            return None
        if len(df.columns) < 2:
            st.warning("Loaded data must contain at least two columns (one feature and one target column).")
            print("Warning (load_user_data): DataFrame has fewer than 2 columns.")
            return None # Need at least one feature and target

        # --- Drop Specified Columns ---
        if columns_to_drop:
            initial_cols_count = df.shape[1]
            cols_found_to_drop = [col for col in columns_to_drop if col in df.columns]
            if cols_found_to_drop:
                df = df.drop(columns=cols_found_to_drop)
                print(f"Dropped specified columns: {cols_found_to_drop}. New shape: {df.shape}")
            else:
                # This is just a warning, not a fatal error
                print(f"Warning (load_user_data): None of the specified columns to drop were found: {columns_to_drop}")

        # --- Handle Missing Values (NaNs) ---
        # Simple strategy: drop rows with any NaN values.
        # More sophisticated imputation should happen during preprocessing if needed.
        initial_rows = len(df)
        df_cleaned = df.dropna() # Drop rows containing any NaN
        dropped_rows = initial_rows - len(df_cleaned)

        if dropped_rows > 0:
            st.warning(f"Dropped {dropped_rows} rows containing missing values (NaNs) during initial load.")
            print(f"Dropped {dropped_rows} rows with NaNs during loading.")

        if df_cleaned.empty:
            st.error("Data became empty after removing rows with missing values. Cannot proceed.")
            print("Error (load_user_data): DataFrame empty after dropping NaNs.")
            return None

        print(f"Data shape after dropping NaNs (if any): {df_cleaned.shape}")
        # Reset index after potentially dropping rows to ensure clean indexing downstream
        return df_cleaned.reset_index(drop=True)

    except FileNotFoundError:
        # This case should be caught by os.path.exists, but included for robustness
        st.error(f"Data file not found (FileNotFoundError): {data_file_path}")
        print(f"Error (load_user_data): FileNotFoundError at {data_file_path}.")
        return None
    except Exception as e:
        # Catch other potential errors during pandas read_csv or processing
        st.error(f"An unexpected error occurred while loading or cleaning the data file '{os.path.basename(data_file_path)}': {e}")
        print(f"Error (load_user_data): Unexpected exception: {e}")
        traceback.print_exc()
        return None
