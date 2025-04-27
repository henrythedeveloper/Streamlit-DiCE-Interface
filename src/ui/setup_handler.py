# File: src/ui/setup_handler.py
"""
UI Module: Scenario Setup Handler
---------------------------------
Coordinates the multi-step process triggered when a user selects a
pre-defined scenario. It handles:
1. Loading the scenario configuration.
2. Loading the corresponding dataset.
3. Loading the pre-trained model and associated preprocessing artifacts
   (encoder for categorical features, scaler for numerical features).
4. Preparing a `data_info` dictionary containing feature lists, target name, etc.
5. Processing the loaded data using the encoder and scaler.
6. Running initial predictions on the processed data.
7. Initializing the DiCE explainer with the model and processed data.
Updates relevant session state variables throughout the process.
"""

# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import json
import joblib
import os
import traceback # For detailed error logging

# Project Modules
from ..data import load_user_data, process_uploaded_data, calculate_feature_ranges
from .. import model_loader
from .. import explain
from .. import config # To access PROJECT_SCENARIOS

# --- Main Setup Function ---
def handle_explainer_setup(scenario_key):
    """
    Manages the end-to-end setup process for a selected scenario.

    Loads data, model, preprocessing artifacts, processes data, runs initial
    predictions, and initializes the DiCE explainer. Updates session state
    accordingly.

    Args:
        scenario_key (str): The key identifying the selected scenario
                            in config.PROJECT_SCENARIOS.

    Returns:
        bool: True if the entire setup process completes successfully, False otherwise.
    """
    st.session_state.app_ready = False # Ensure app is marked as not ready during setup
    print(f"\n--- Starting Setup Handler for Scenario: {scenario_key} ---")

    # 1. Get Scenario Configuration
    # -----------------------------
    scenario_config = config.get_scenario_config(scenario_key)
    if not scenario_config:
        st.error(f"Configuration for scenario '{scenario_key}' not found in config.py.")
        print(f"Error: Scenario config not found for key '{scenario_key}'.")
        return False
    print(f"Loaded Scenario Config: {scenario_config}")

    # 2. Load Data
    # ------------
    try:
        data_path = scenario_config['data_path']
        # Resolve path relative to project root if needed
        if not os.path.isabs(data_path):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # Assumes src/ui structure
            data_path = os.path.join(project_root, data_path)
        print(f"Attempting to load data from resolved path: {data_path}")

        df_loaded = load_user_data(
             data_path,
             scenario_config['target_column'],
             scenario_config.get('columns_to_drop_on_load', []) # Safely get optional list
        )
        if df_loaded is None:
            # load_user_data should show specific error in UI
            raise ValueError("Data loading function returned None.")
        st.session_state.uploaded_df = df_loaded
        print(f"Data loaded successfully. Shape: {st.session_state.uploaded_df.shape}")
    except Exception as e:
        st.error(f"Failed to load data for scenario '{scenario_key}': {e}")
        print(f"Data loading failed: {e}")
        traceback.print_exc()
        return False

    # 3. Load Model Artifacts (Model, Encoder, Scaler)
    # -------------------------------------------------
    try:
        model_artifact_path = scenario_config['model_artifact_path']
        # Resolve path relative to project root
        if not os.path.isabs(model_artifact_path):
             project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
             model_artifact_path = os.path.join(project_root, model_artifact_path)
        print(f"Attempting to load model artifact from resolved path: {model_artifact_path}")

        # Load model, encoder, scaler, and inferred model type using the loader module
        model, encoder, scaler, model_type = model_loader.load_user_model(model_artifact_path)

        if model is None:
            # model_loader should show specific error in UI
            print(f"Error: Failed to load model from artifact: {model_artifact_path}")
            return False

        # Store loaded components in session state
        st.session_state.model = model
        st.session_state.encoder = encoder
        st.session_state.scaler = scaler
        st.session_state.model_type = model_type
        print(f"Model loaded. Type: {model_type}. Encoder loaded: {'Yes' if encoder else 'No'}. Scaler loaded: {'Yes' if scaler else 'No'}")
    except Exception as e:
        st.error(f"Failed to load model artifact for scenario '{scenario_key}': {e}")
        print(f"Model artifact loading failed: {e}")
        traceback.print_exc()
        return False

    # 4. Prepare Data Information Dictionary
    # --------------------------------------
    try:
        # Determine the features the loaded model actually expects from its attributes
        model_expected_features = None
        try:
            if hasattr(st.session_state.model, 'feature_name_'):
                model_expected_features = list(st.session_state.model.feature_name_)
            elif hasattr(st.session_state.model, 'feature_names_in_'):
                model_expected_features = list(st.session_state.model.feature_names_in_)
            else:
                print("Warning: Model lacks feature name attributes. Using features from config as fallback.")
                model_expected_features = scenario_config['numerical_features'] + scenario_config['categorical_features']
        except Exception as feat_err:
            print(f"Error retrieving feature names from model: {feat_err}. Using config fallback.")
            model_expected_features = scenario_config['numerical_features'] + scenario_config['categorical_features']

        # Store various feature lists in session state and data_info
        st.session_state.model_expected_features = model_expected_features
        st.session_state.user_numerical_features = scenario_config['numerical_features']
        st.session_state.user_categorical_features = scenario_config['categorical_features']
        # Filter expected features based on config types for consistency
        st.session_state.model_num_features = [f for f in model_expected_features if f in st.session_state.user_numerical_features]
        st.session_state.model_cat_features = [f for f in model_expected_features if f in st.session_state.user_categorical_features]

        # Calculate feature ranges (e.g., for DiCE constraints)
        feature_ranges = calculate_feature_ranges(st.session_state.uploaded_df, st.session_state.user_numerical_features)

        # Assemble the data_info dictionary for use by other modules
        st.session_state.data_info = {
            "target_column": scenario_config['target_column'],
            "numerical_features": st.session_state.user_numerical_features, # User's view of types
            "categorical_features": st.session_state.user_categorical_features, # User's view of types
            "feature_ranges": feature_ranges, # Calculated ranges for constraints
            "model_type": st.session_state.model_type, # Backend identifier for DiCE
            # Internal lists reflecting the model's actual input structure
            "_model_expected_features": st.session_state.model_expected_features,
            "_model_num_features": st.session_state.model_num_features,
            "_model_cat_features": st.session_state.model_cat_features
        }
        print("Data info dictionary prepared.")
    except Exception as e:
        st.error(f"Failed to prepare data info dictionary or calculate ranges: {e}")
        print(f"Data info preparation failed: {e}")
        traceback.print_exc()
        st.session_state.data_info = None
        return False

    # 5. Process Data for Model Input and DiCE
    # ----------------------------------------
    try:
        df_to_process = st.session_state.uploaded_df
        # Process data using loaded artifacts (encoder, scaler)
        X_processed_for_model, y_app, _, dice_data_df, _, cat_indices, _ = process_uploaded_data(
            df_full=df_to_process.copy(), # Pass a copy to avoid modifying original
            data_info=st.session_state.data_info,
            model=st.session_state.model, # Pass model for context if needed by processor
            encoder=st.session_state.encoder, # Pass loaded encoder
            scaler=st.session_state.scaler # Pass loaded scaler
        )

        if X_processed_for_model is None:
            st.error("Data processing step failed. Cannot proceed.")
            return False

        # Store processed data components in session state
        st.session_state.X_app = X_processed_for_model # Data ready for model prediction
        st.session_state.y_app = y_app # Processed target variable (e.g., 0/1)
        st.session_state.dice_data_df = dice_data_df # Original data for DiCE Data object setup
        st.session_state.categorical_indices = cat_indices # Indices needed by some models/explainers
        print("Data processed for model prediction and DiCE setup.")

    except Exception as e:
        st.error(f"Error during data processing: {e}")
        print(f"Data processing failed: {e}")
        traceback.print_exc()
        return False

    # 6. Setup DiCE Explainer
    # -----------------------
    try:
        # Initialize DiCE using the processed data and loaded artifacts
        dice_explainer, _ = explain.setup_dice_explainer(
            model_object=st.session_state.model,
            model_backend_name=st.session_state.data_info['model_type'],
            dice_data_df=st.session_state.dice_data_df, # Use original data for DiCE setup
            data_info=st.session_state.data_info,
            encoder=st.session_state.encoder, # Pass artifacts to ModelWrapper inside DiCE
            scaler=st.session_state.scaler,
            model_expected_features=st.session_state.model_expected_features,
            model_num_features=st.session_state.model_num_features,
            model_cat_features=st.session_state.model_cat_features
        )
        if dice_explainer is None:
            st.error("DiCE explainer setup failed. Check logs for details.")
            return False
        st.session_state.dice_explainer = dice_explainer
        print("DiCE explainer setup complete.")
    except Exception as dice_err:
        st.error(f"Failed during DiCE explainer setup: {dice_err}")
        print(f"DiCE setup failed: {dice_err}")
        traceback.print_exc()
        st.session_state.dice_explainer = None
        return False

    # 7. Make Initial Predictions on the Entire Dataset
    # -------------------------------------------------
    st.session_state.predictions_app = None
    st.session_state.probabilities_app = None
    if st.session_state.model and st.session_state.X_app is not None:
        try:
            print("Running initial predictions on the processed dataset (X_app)...")
            X_predict = st.session_state.X_app

            # Ensure column order matches model expectation before prediction
            # Retrieve expected order again for safety
            expected_order = st.session_state.model_expected_features
            if expected_order and list(X_predict.columns) != list(expected_order):
                print(f"Reordering columns to match model's expected order: {expected_order}")
                try:
                    X_predict = X_predict[expected_order]
                except KeyError as ke:
                    print(f"Error reordering columns for prediction: Missing columns {ke}. Prediction might fail.")
                    # Proceed, but prediction might fail later

            # Convert to NumPy array for prediction robustness
            X_predict_np = X_predict.to_numpy()
            print(f"Data shape for prediction: {X_predict_np.shape}, Dtype: {X_predict_np.dtype}")

            model_to_predict = st.session_state.model

            # Perform prediction using the NumPy array
            if hasattr(model_to_predict, 'predict_proba'):
                st.session_state.probabilities_app = model_to_predict.predict_proba(X_predict_np)
                # Derive predictions from probabilities using 0.5 threshold
                st.session_state.predictions_app = (st.session_state.probabilities_app[:, 1] > 0.5).astype(int)
                print("Predictions and probabilities obtained successfully.")
            elif hasattr(model_to_predict, 'predict'):
                st.session_state.predictions_app = model_to_predict.predict(X_predict_np)
                print("Predictions obtained (probabilities not available).")
            else:
                st.error("Loaded model lacks required 'predict_proba' or 'predict' method.")
                return False

            # Log prediction counts for verification
            if st.session_state.predictions_app is not None:
                unique_preds, counts = np.unique(st.session_state.predictions_app, return_counts=True)
                pred_counts_dict = dict(zip(unique_preds, counts))
                count_0 = pred_counts_dict.get(0, 0)
                count_1 = pred_counts_dict.get(1, 0)
                print(f"--- Initial Prediction Counts ---")
                print(f"  Predicted Class 0: {count_0} instances")
                print(f"  Predicted Class 1: {count_1} instances")
                print(f"-------------------------------")

        except ValueError as ve:
             # Catch specific errors like feature mismatch during predict
             st.error(f"Prediction Error: {ve}")
             print(f"Prediction Failed: {ve}")
             traceback.print_exc()
             return False
        except Exception as pred_err:
             st.error(f"An unexpected error occurred during initial prediction: {pred_err}")
             print(f"Initial Prediction Failed: {pred_err}")
             traceback.print_exc()
             return False
    elif st.session_state.X_app is None:
         st.error("Processed data (X_app) is not available for prediction.")
         return False

    # 8. Final Setup Completion Check
    # -------------------------------
    if st.session_state.dice_explainer and st.session_state.predictions_app is not None:
        st.session_state.app_ready = True # Mark app as ready for explanation generation
        # Reset state related to specific explanations from previous runs/scenarios
        st.session_state.selected_instance_index = None
        st.session_state.dice_results = None
        st.session_state.explained_instance_df = None
        print(f"--- Scenario '{scenario_key}' Setup Successful. App is Ready. ---")
        return True
    else:
        # If setup failed at any critical point
        print(f"--- Scenario '{scenario_key}' Setup Failed. App not ready. ---")
        st.session_state.app_ready = False
        # Provide feedback if specific components failed
        if not st.session_state.dice_explainer: st.error("Failed to initialize the DiCE explainer.")
        if st.session_state.predictions_app is None: st.error("Failed to generate initial predictions on the dataset.")
        return False
