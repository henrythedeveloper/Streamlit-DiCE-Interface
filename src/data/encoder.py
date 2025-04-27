# File: src/data/encoder.py
"""
Data Encoding Module (Legacy/Reference)
---------------------------------------
Provides functions for creating and applying encoders for categorical features.

Note: In the current application flow, encoding is typically handled by
loading a pre-fitted encoder saved during model training, rather than
fitting a new one within the application. This module might be used for
reference or specific edge cases.
"""

# --- Imports ---
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import streamlit as st # Used only for potential warnings/info messages

# --- Encoder Creation Function ---
def create_categorical_encoder(feature_values, method='label'):
    """
    Creates and fits an encoder for a given set of categorical feature values.

    Args:
        feature_values (pd.Series or np.ndarray): The unique values of the
                                                  categorical feature to encode.
        method (str): The encoding method to use ('label' or 'onehot').
                      Defaults to 'label'.

    Returns:
        tuple: (fitted_encoder_object, mapping_dictionary)
               The mapping dictionary shows the correspondence between original
               categories and their encoded representation.

    Raises:
        ValueError: If an unknown encoding method is specified.
    """
    print(f"Creating '{method}' encoder for feature values.")
    if method == 'label':
        encoder = LabelEncoder()
        # Ensure input is suitable for LabelEncoder (1D array-like)
        values_to_fit = pd.Series(feature_values).astype(str).unique()
        encoder.fit(values_to_fit)
        # Create a mapping from original classes to encoded integers
        mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
        print(f"  LabelEncoder fitted. Mapping: {mapping}")
        return encoder, mapping
    elif method == 'onehot':
        encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        # OneHotEncoder expects a 2D array-like structure
        values_to_fit = pd.Series(feature_values).astype(str).unique().reshape(-1, 1)
        encoder.fit(values_to_fit)
        # Create a mapping (more complex for one-hot, shows original categories)
        mapping = {cat: i for i, cat in enumerate(encoder.categories_[0])}
        print(f"  OneHotEncoder fitted. Categories: {encoder.categories_[0]}")
        return encoder, mapping
    else:
        raise ValueError(f"Unknown encoding method specified: {method}")

# --- Feature Encoding Application Function ---
def encode_categorical_features(df, categorical_features):
    """
    Applies label encoding to specified categorical features in a DataFrame.

    Note: This function fits a *new* encoder for each column. For consistent
          encoding matching a trained model, use a pre-fitted encoder loaded
          from training artifacts (handled in data/processor.py).

    Args:
        df (pd.DataFrame): The DataFrame containing the features to encode.
        categorical_features (list): A list of column names to be treated as
                                     categorical and encoded.

    Returns:
        tuple: (
            encoded_df (pd.DataFrame): DataFrame with specified columns encoded.
            encoders (dict): Dictionary mapping column names to fitted encoders.
            categorical_indices (list): List of integer indices for the encoded columns.
            encoded_categorical_values (dict): Dictionary mapping column names to
                                               their category-to-integer mappings.
        )
    """
    encoders = {}
    encoded_df = df.copy()
    encoded_categorical_values = {}
    categorical_indices = []

    print(f"Applying new label encoding to columns: {categorical_features}")

    # Iterate through columns and apply encoding if categorical
    for i, col in enumerate(df.columns):
        if col in categorical_features:
            categorical_indices.append(i)
            # Check if the column is not already numeric before encoding
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    # Create and fit a new LabelEncoder for this column
                    encoder, mapping = create_categorical_encoder(df[col].astype(str), method='label')
                    # Transform the column in the copied DataFrame
                    encoded_df[col] = encoder.transform(df[col].astype(str))

                    # Store the fitted encoder and its mapping
                    encoders[col] = encoder
                    encoded_categorical_values[col] = mapping

                    print(f"  Applied label encoding to '{col}'.")
                    # st.info(f"Encoded categorical feature '{col}' using label encoding.") # Optional UI feedback
                except Exception as e:
                    print(f"  Error encoding column '{col}': {e}")
                    # Handle error appropriately, maybe skip column or raise exception
                    st.warning(f"Could not encode column '{col}'. It might contain incompatible data types.")
            else:
                print(f"  Skipping encoding for '{col}' as it is already numeric.")

    print("Finished applying new label encoding.")
    return encoded_df, encoders, categorical_indices, encoded_categorical_values
