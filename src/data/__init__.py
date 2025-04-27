# File: src/data/__init__.py
"""
Data Handling Package for Counterfactual Explainer
--------------------------------------------------
This package contains modules responsible for loading, analyzing,
and processing data for use within the application, particularly
for model prediction and counterfactual explanation generation.
"""

# --- Exports ---
# Expose key functions for easier access from other parts of the application.
from .loader import load_user_data
from .feature_analyzer import calculate_feature_ranges
from .processor import process_uploaded_data, convert_target_col, extract_model_features

# --- Package Initialization ---
# print("Initializing src.data package...") # Can be uncommented for debugging imports
