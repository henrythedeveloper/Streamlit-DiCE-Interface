# File: src/model_loader.py
"""
Model Artifact Loading Module
-----------------------------
Handles loading of machine learning models and associated preprocessing artifacts
(like encoders and scalers) saved together in a single .joblib file.
It expects a dictionary structure {'model': ..., 'encoder': ..., 'scaler': ...}
but includes fallback logic if components are missing or the file contains only the model.
"""

# --- Imports ---
import streamlit as st
import joblib
import os
import warnings

# --- Main Loading Function ---
def load_user_model(model_artifact_path):
    """
    Loads a model, encoder, and scaler from a specified .joblib file path.

    Checks for the expected dictionary structure:
    {'model': model_obj, 'encoder': encoder_obj, 'scaler': scaler_obj}

    If the file doesn't contain this structure, it assumes the file contains
    only the model object and returns None for the encoder and scaler.

    Args:
        model_artifact_path (str): The absolute or relative path to the .joblib artifact file.

    Returns:
        tuple: (model_object, encoder_object, scaler_object, model_type_string)
               Returns (None, None, None, None) if loading fails.
               Encoder or scaler objects will be None if not found in the artifact.
    """
    model = None
    encoder = None
    scaler = None
    model_type = "unknown" # Default backend identifier for DiCE

    # --- Input Validation ---
    if not model_artifact_path or not isinstance(model_artifact_path, str):
        st.error("No valid model artifact path provided.")
        return None, None, None, None

    if not os.path.exists(model_artifact_path):
        st.error(f"Model artifact file not found at: {model_artifact_path}")
        return None, None, None, None

    # --- Artifact Loading ---
    try:
        file_suffix = os.path.splitext(model_artifact_path)[1].lower()
        if file_suffix != '.joblib':
            # This is just a warning; joblib might still load it.
            print(f"Warning: Model file does not have a .joblib extension: {model_artifact_path}")

        print(f"Attempting to load artifacts from: {model_artifact_path}")
        # Suppress potential UserWarnings during loading (e.g., version mismatches)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            loaded_object = joblib.load(model_artifact_path)

        # --- Artifact Structure Check ---
        # Check if the loaded object is the expected dictionary
        if isinstance(loaded_object, dict) and 'model' in loaded_object:
            model = loaded_object['model']
            encoder = loaded_object.get('encoder') # Safely get encoder, default None
            scaler = loaded_object.get('scaler')   # Safely get scaler, default None
            print("Successfully loaded artifacts from dictionary structure.")
            print(f"  Model loaded: {'Yes' if model else 'No'}")
            print(f"  Encoder loaded: {'Yes' if encoder else 'No'}")
            print(f"  Scaler loaded: {'Yes' if scaler else 'No'}")
        else:
            # Fallback: Assume the loaded object *is* the model
            print("Loaded object is not the expected dictionary. Assuming it's the model only.")
            model = loaded_object
            encoder = None
            scaler = None

        # --- Model Type Inference (for DiCE backend) ---
        if model is not None:
            model_type_str = str(type(model)).lower()
            # Check for scikit-learn compatibility (predict_proba is preferred)
            has_predict_proba = hasattr(model, 'predict_proba')
            has_predict = hasattr(model, 'predict')

            if has_predict_proba and has_predict:
                model_type = 'sklearn' # Common case for classifiers
                print(f"Inferred model type: 'sklearn' (has predict and predict_proba).")
            elif has_predict:
                model_type = 'sklearn' # Can still work for DiCE, might lack probability info
                print(f"Warning: Loaded model has 'predict' but not 'predict_proba'. DiCE might have limitations.")
            else:
                # Model is unusable for DiCE if it lacks basic prediction methods
                print("Error: Loaded model object lacks required 'predict' or 'predict_proba' methods.")
                st.error("The loaded model is incompatible. It needs a 'predict' or 'predict_proba' method.")
                return None, None, None, None # Fail loading
        else:
            # If model extraction failed
            st.error(f"Failed to extract a valid model object from the file: {model_artifact_path}")
            return None, None, None, None # Fail loading

    except FileNotFoundError:
        # Should be caught by os.path.exists, but included for robustness
        st.error(f"Model artifact file not found (FileNotFoundError): {model_artifact_path}")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading model artifact '{os.path.basename(model_artifact_path)}': {e}")
        print(f"Detailed loading error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None # Fail loading

    # Return loaded components
    return model, encoder, scaler, model_type