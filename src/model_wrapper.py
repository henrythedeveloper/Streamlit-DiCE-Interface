# File: src/model_wrapper.py
"""
Model Wrapper for DiCE Integration
----------------------------------
Provides a wrapper class compatible with the DiCE framework.
This class takes a trained ML model and its associated preprocessing
artifacts (encoder, scaler) and ensures that data passed to the model
for prediction within the DiCE generation process is correctly
preprocessed (encoded, scaled) first.
"""

# --- Imports ---
import numpy as np
import pandas as pd
import streamlit as st # For displaying errors/warnings if needed (optional)
from sklearn.exceptions import NotFittedError
import warnings
import traceback # For detailed error logging

# --- Model Wrapper Class ---
class ModelWrapper:
    """
    Wraps a trained machine learning model to handle preprocessing internally.
    Designed for use with the DiCE library. It applies encoding and scaling
    using artifacts loaded from the training phase.
    """

    def __init__(self, model, encoder=None, scaler=None, model_expected_features=None,
                 model_num_features=None, model_cat_features=None, **kwargs):
        """
        Initializes the ModelWrapper.

        Args:
            model: The underlying trained model object (e.g., LGBMClassifier).
            encoder: Fitted encoder object (e.g., OrdinalEncoder).
            scaler: Fitted scaler object (e.g., StandardScaler).
            model_expected_features (list): Full list of feature names in the
                                          order the model expects them.
            model_num_features (list): Names of numerical features expected by the model.
            model_cat_features (list): Names of categorical features expected by the model.
        """
        self.model = model
        self.encoder = encoder
        self.scaler = scaler
        self.model_expected_features = model_expected_features or []
        self.model_num_features = model_num_features or []
        self.model_cat_features = model_cat_features or []

        # --- Initialization Validation ---
        if not self.model_expected_features:
            print("Warning (ModelWrapper): Initialized without 'model_expected_features'. Preprocessing might be unreliable.")
        if self.model_cat_features and self.encoder is None:
            print("Warning (ModelWrapper): Expects categorical features but no encoder provided. Encoding step will be skipped.")
        if self.model_num_features and self.scaler is None:
            print("Warning (ModelWrapper): Expects numerical features but no scaler provided. Scaling step will be skipped.")

        print("ModelWrapper (with Preprocessing) initialized.")
        print(f"  Model expects features: {self.model_expected_features}")
        print(f"  Numerical features for scaling: {self.model_num_features}")
        print(f"  Categorical features for encoding: {self.model_cat_features}")

    def _preprocess(self, X_input):
        """
        Internal method to preprocess input data before prediction.
        Applies encoding and scaling using the stored artifacts.

        Args:
            X_input (pd.DataFrame or np.ndarray): Input data with original feature values.

        Returns:
            pd.DataFrame or None: Processed DataFrame ready for the model, or None on failure.
        """
        # --- Input Handling ---
        # Convert input to DataFrame if it's not already one
        if not isinstance(X_input, pd.DataFrame):
            print("Info (Wrapper._preprocess): Input is not a DataFrame. Converting using expected features.")
            try:
                # Ensure the number of columns matches if it's an ndarray
                if isinstance(X_input, np.ndarray) and X_input.shape[1] != len(self.model_expected_features):
                    print(f"Error (Wrapper._preprocess): Input ndarray columns ({X_input.shape[1]}) "
                          f"don't match expected features ({len(self.model_expected_features)}).")
                    return None
                X_input_df = pd.DataFrame(X_input, columns=self.model_expected_features)
            except Exception as e:
                 print(f"Error (Wrapper._preprocess): Could not convert input to DataFrame: {e}")
                 return None
        else:
             X_input_df = X_input.copy() # Work on a copy

        # --- Feature Alignment ---
        # Ensure columns are present and in the correct order
        missing_cols = [f for f in self.model_expected_features if f not in X_input_df.columns]
        if missing_cols:
            print(f"Error (Wrapper._preprocess): Input missing expected columns: {missing_cols}")
            return None
        # Reorder columns to match the model's expectation
        X = X_input_df[self.model_expected_features].copy()

        # --- Preprocessing Steps ---

        # 1. Ensure Numerical Columns are Numeric (Handle Potential Objects/NaNs)
        if self.model_num_features:
            nums_present = [f for f in self.model_num_features if f in X.columns]
            if nums_present:
                for col in nums_present:
                    if not pd.api.types.is_numeric_dtype(X[col]):
                        # Attempt conversion if not numeric
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                    # Impute NaNs that might result from coercion or were already present
                    if X[col].isnull().any():
                        median_val = X[col].median() # Use median for imputation
                        X[col].fillna(median_val, inplace=True)

        # 2. Apply Ordinal Encoding
        if self.encoder and self.model_cat_features:
            cats_present = [f for f in self.model_cat_features if f in X.columns]
            if cats_present:
                try:
                    # Handle potential NaNs before encoding (impute with mode)
                    for col in cats_present:
                        if X[col].isnull().any():
                            mode_val = X[col].mode()[0] if not X[col].mode().empty else 'missing'
                            X[col].fillna(mode_val, inplace=True)
                        # Ensure string type for encoder
                        X[col] = X[col].astype(str)
                    # Apply the fitted encoder
                    encoded_values = self.encoder.transform(X[cats_present])
                    # Overwrite columns with encoded values (ensure float type for consistency)
                    X[cats_present] = encoded_values.astype(np.float64)
                except NotFittedError:
                    print("Error (Wrapper._preprocess): The provided encoder is not fitted.")
                    return None
                except ValueError as ve:
                    # Handle cases where new, unseen categories appear in the data
                    print(f"Error applying encoder in wrapper: {ve}")
                    traceback.print_exc()
                    return None
                except Exception as e:
                     print(f"Unexpected error during encoding in wrapper: {e}")
                     traceback.print_exc()
                     return None

        # 3. Apply Scaling
        if self.scaler and self.model_num_features:
            nums_present = [f for f in self.model_num_features if f in X.columns]
            if nums_present:
                # Final check for non-numeric types before scaling
                non_numeric_in_num_list = X[nums_present].select_dtypes(exclude=np.number).columns
                if not non_numeric_in_num_list.empty:
                     print(f"Error (Wrapper._preprocess): Columns intended for scaling are non-numeric: {non_numeric_in_num_list.tolist()}")
                     return None
                try:
                    # Apply the fitted scaler
                    scaled_values = self.scaler.transform(X[nums_present])
                    # Overwrite columns with scaled values
                    X[nums_present] = scaled_values
                except NotFittedError:
                    print("Error (Wrapper._preprocess): The provided scaler is not fitted.")
                    return None
                except ValueError as ve:
                    print(f"Error applying scaler in wrapper: {ve}")
                    traceback.print_exc()
                    return None
                except Exception as e:
                     print(f"Unexpected error during scaling in wrapper: {e}")
                     traceback.print_exc()
                     return None

        # --- Final Check: Ensure all columns are numeric after processing ---
        final_non_numeric = X.select_dtypes(exclude=np.number).columns
        if not final_non_numeric.empty:
             print(f"Error (Wrapper._preprocess): Post-processing check failed. Non-numeric columns remain: {final_non_numeric.tolist()}")
             print("Final dtypes before returning from _preprocess:")
             print(X.dtypes)
             return None

        return X


    def predict(self, X_input):
        """
        Preprocesses input data and returns class predictions from the wrapped model.

        Args:
            X_input (pd.DataFrame or np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted class labels.
        """
        print("ModelWrapper: Received data for predict.")
        X_processed = self._preprocess(X_input)
        if X_processed is None:
            print("Error (Wrapper.predict): Preprocessing failed. Returning empty array.")
            # Return an empty array matching expected output dimensionality if possible
            return np.array([])

        print("ModelWrapper: Calling predict on the internal model...")
        try:
            # Convert final processed DataFrame to NumPy for model compatibility
            X_processed_np = X_processed.to_numpy()
            # Suppress potential warnings about feature names if model is picky
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                predictions = self.model.predict(X_processed_np)
            return predictions
        except Exception as e:
            print(f"Error during wrapped model predict: {e}")
            traceback.print_exc()
            raise e # Re-raise the error after logging

    def predict_proba(self, X_input):
        """
        Preprocesses input data and returns class probabilities from the wrapped model.

        Args:
            X_input (pd.DataFrame or np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted class probabilities (typically shape [n_samples, n_classes]).
        """
        print("ModelWrapper: Received data for predict_proba.")
        X_processed = self._preprocess(X_input)
        if X_processed is None:
            print("Error (Wrapper.predict_proba): Preprocessing failed. Returning empty array.")
            # Return an empty array matching expected output dimensionality if possible
            return np.empty((0, 2)) # Assuming binary classification

        print("ModelWrapper: Calling predict_proba on the internal model...")
        try:
            # Convert final processed DataFrame to NumPy
            X_processed_np = X_processed.to_numpy()
            # Suppress potential warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                probabilities = self.model.predict_proba(X_processed_np)
            return probabilities
        except Exception as e:
            print(f"Error during wrapped model predict_proba: {e}")
            traceback.print_exc()
            raise e # Re-raise the error after logging

    def __getattr__(self, attr):
        """
        Allows accessing other attributes or methods directly from the wrapped model
        if they are not defined in the wrapper itself.
        """
        if attr in self.__dict__:
            # If the attribute is explicitly defined in the wrapper, return it
            return getattr(self, attr)
        try:
            # Otherwise, try to get the attribute from the underlying model
            return getattr(self.model, attr)
        except AttributeError:
            # If neither the wrapper nor the model has the attribute, raise standard error
            raise AttributeError(f"'{type(self).__name__}' object (and its wrapped model '{type(self.model).__name__}') has no attribute '{attr}'")