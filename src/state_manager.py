# src/state_manager.py
"""
Streamlit Session State Manager
-------------------------------
This module handles the initialization and management of Streamlit session state
variables used throughout the application. The session state provides persistent
storage of data between reruns of the Streamlit app.

Session state variables manage:
1. File upload states
2. Data processing states
3. Model loading states
4. Counterfactual generation states
5. User interface states

This module helps maintain a consistent application state and prevents errors
from accessing undefined state variables.
"""

import streamlit as st

def initialize_session_state():
    """
    Initializes all required session state keys if they don't exist.
    
    This function ensures all necessary state variables are defined before
    they are accessed, preventing KeyError exceptions during app execution.
    
    The state variables are organized into logical categories based on their use:
    - File management
    - Data processing
    - Model loading
    - Explainer state
    - UI control state
    """

    # --- File Upload States ---
    # Track uploaded files and their processing status
    if 'model_file' not in st.session_state: st.session_state.model_file = None
    if 'data_file' not in st.session_state: st.session_state.data_file = None

    # --- Data Processing States ---
    # Store loaded data and configuration information
    if 'uploaded_df' not in st.session_state: st.session_state.uploaded_df = None
    if 'target_column_options' not in st.session_state: st.session_state.target_column_options = []
    if 'selected_target_column' not in st.session_state: st.session_state.selected_target_column = None
    if 'data_info' not in st.session_state: st.session_state.data_info = None

    # --- Model States ---
    # Store loaded model objects and related components
    if 'model' not in st.session_state: st.session_state.model = None
    if 'model_type' not in st.session_state: st.session_state.model_type = None
    if 'scaler' not in st.session_state: st.session_state.scaler = None
    if 'feature_encoders' not in st.session_state: st.session_state.feature_encoders = None  # NEW
    if 'categorical_indices' not in st.session_state: st.session_state.categorical_indices = None  # NEW

    # --- Processed Data States ---
    # Store the prepared data for the app and DiCE
    if 'X_app' not in st.session_state: st.session_state.X_app = None
    if 'y_app' not in st.session_state: st.session_state.y_app = None
    if 'X_app_scaled' not in st.session_state: st.session_state.X_app_scaled = None # Scaled numerical features
    if 'dice_data_df' not in st.session_state: st.session_state.dice_data_df = None # Full processed data for DiCE

    # --- Explainer States ---
    # Store DiCE explainer and results
    if 'dice_explainer' not in st.session_state: st.session_state.dice_explainer = None
    if 'dice_results' not in st.session_state: st.session_state.dice_results = None

    # --- Instance Selection & Constraints States ---
    # Track which instance is being explained and any constraints
    if 'selected_instance_index' not in st.session_state: st.session_state.selected_instance_index = None
    if 'immutable_features_selection' not in st.session_state: st.session_state.immutable_features_selection = []
    if 'explained_instance_df' not in st.session_state: st.session_state.explained_instance_df = None # Store the instance being explained

    # --- Prediction States ---
    # Store model predictions for all instances
    if 'predictions_app' not in st.session_state: st.session_state.predictions_app = None
    if 'probabilities_app' not in st.session_state: st.session_state.probabilities_app = None

    # --- Control Flag States ---
    # Track overall application state and readiness
    if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
    if 'app_ready' not in st.session_state: st.session_state.app_ready = False
    """
    Initializes all required session state keys if they don't exist.
    
    This function ensures all necessary state variables are defined before
    they are accessed, preventing KeyError exceptions during app execution.
    
    The state variables are organized into logical categories based on their use:
    - File management
    - Data processing
    - Model loading
    - Explainer state
    - UI control state
    """

    # --- File Upload States ---
    # Track uploaded files and their processing status
    if 'model_file' not in st.session_state: st.session_state.model_file = None
    if 'data_file' not in st.session_state: st.session_state.data_file = None

    # --- Data Processing States ---
    # Store loaded data and configuration information
    if 'uploaded_df' not in st.session_state: st.session_state.uploaded_df = None
    if 'target_column_options' not in st.session_state: st.session_state.target_column_options = []
    if 'selected_target_column' not in st.session_state: st.session_state.selected_target_column = None
    if 'data_info' not in st.session_state: st.session_state.data_info = None

    # --- Model States ---
    # Store loaded model objects and related components
    if 'model' not in st.session_state: st.session_state.model = None
    if 'model_type' not in st.session_state: st.session_state.model_type = None
    if 'scaler' not in st.session_state: st.session_state.scaler = None

    # --- Processed Data States ---
    # Store the prepared data for the app and DiCE
    if 'X_app' not in st.session_state: st.session_state.X_app = None
    if 'y_app' not in st.session_state: st.session_state.y_app = None
    if 'X_app_scaled' not in st.session_state: st.session_state.X_app_scaled = None # Scaled numerical features
    if 'dice_data_df' not in st.session_state: st.session_state.dice_data_df = None # Full processed data for DiCE

    # --- Explainer States ---
    # Store DiCE explainer and results
    if 'dice_explainer' not in st.session_state: st.session_state.dice_explainer = None
    if 'dice_results' not in st.session_state: st.session_state.dice_results = None

    # --- Instance Selection & Constraints States ---
    # Track which instance is being explained and any constraints
    if 'selected_instance_index' not in st.session_state: st.session_state.selected_instance_index = None
    if 'immutable_features_selection' not in st.session_state: st.session_state.immutable_features_selection = []
    if 'explained_instance_df' not in st.session_state: st.session_state.explained_instance_df = None # Store the instance being explained

    # --- Prediction States ---
    # Store model predictions for all instances
    if 'predictions_app' not in st.session_state: st.session_state.predictions_app = None
    if 'probabilities_app' not in st.session_state: st.session_state.probabilities_app = None

    # --- Control Flag States ---
    # Track overall application state and readiness
    if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
    if 'app_ready' not in st.session_state: st.session_state.app_ready = False

     # Add these new ones:
    if 'feature_encoders' not in st.session_state: st.session_state.feature_encoders = None
    if 'encoded_categorical_values' not in st.session_state: st.session_state.encoded_categorical_values = None


def reset_explanation_state():
    """Resets only the explanation-related state variables."""
    if 'dice_results' in st.session_state:
        st.session_state.dice_results = None
    if 'explained_instance_df' in st.session_state:
        st.session_state.explained_instance_df = None