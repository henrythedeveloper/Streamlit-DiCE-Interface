# File: src/explain.py
"""
Counterfactual Generation Module using DiCE
-------------------------------------------
Handles the setup of the DiCE explainer and the generation of
counterfactual examples based on user queries and constraints.
It uses a ModelWrapper to ensure data passed to the underlying ML model
is correctly preprocessed (encoded/scaled).
"""

# --- Imports ---
import streamlit as st
import dice_ml
import pandas as pd
import numpy as np
import json
import inspect
import traceback # For detailed error logging

# Exception specific to DiCE for configuration issues
from raiutils.exceptions import UserConfigValidationException

# Project Modules
from . import config # For DiCE defaults (method, num_cfs)
from .model_wrapper import ModelWrapper # Handles preprocessing within DiCE

# --- JSON Serialization Utility ---
class NumpyEncoder(json.JSONEncoder):
    """ Custom JSON encoder for handling numpy data types during serialization. """
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# --- DiCE Explainer Setup ---
def setup_dice_explainer(
    model_object, model_backend_name, dice_data_df, data_info,
    encoder=None, scaler=None, model_expected_features=None,
    model_num_features=None, model_cat_features=None
    ):
    """
    Initializes the DiCE framework components: Data, Model (wrapped), and Explainer.

    Args:
        model_object: The trained machine learning model.
        model_backend_name (str): Backend identifier for DiCE (e.g., 'sklearn').
        dice_data_df (pd.DataFrame): DataFrame with original feature values, including the target.
        data_info (dict): Dictionary holding feature lists (numerical, categorical, target).
        encoder: The fitted encoder object (e.g., OrdinalEncoder).
        scaler: The fitted scaler object (e.g., StandardScaler).
        model_expected_features (list): Full list of features the model expects.
        model_num_features (list): Numerical features the model expects.
        model_cat_features (list): Categorical features the model expects.

    Returns:
        tuple: (dice_ml.Dice object, updated data_info dict) or (None, data_info) on failure.
    """
    print(f"\n--- Setting up DiCE for {model_backend_name} model ---")
    target_column = data_info['target_column']
    # Use feature lists relevant for the DiCE Data object (original user view)
    numerical_features_dice = data_info['numerical_features'][:]
    categorical_features_dice = data_info['categorical_features'][:]

    # --- Input Validation ---
    if model_object is None: print("Error (DiCE Setup): Model object is None."); return None, data_info
    if dice_data_df is None or dice_data_df.empty: print("Error (DiCE Setup): DataFrame for DiCE is None or empty."); return None, data_info
    if target_column not in dice_data_df.columns: print(f"Error (DiCE Setup): Target column '{target_column}' missing from DataFrame."); return None, data_info
    missing_num_dice = [f for f in numerical_features_dice if f not in dice_data_df.columns]
    missing_cat_dice = [f for f in categorical_features_dice if f not in dice_data_df.columns]
    if missing_num_dice: print(f"Error (DiCE Setup): Numerical features missing from DataFrame: {missing_num_dice}"); return None, data_info
    if missing_cat_dice: print(f"Error (DiCE Setup): Categorical features missing from DataFrame: {missing_cat_dice}"); return None, data_info

    try:
        # 1. Adjust Feature Types for DiCE Data Object (Handle Boolean Columns)
        # DiCE's Data object might prefer booleans treated as categorical
        potential_bool_cols = dice_data_df.select_dtypes(include='bool').columns.tolist()
        updated_features_for_dice = False
        temp_num_dice = list(numerical_features_dice)
        temp_cat_dice = list(categorical_features_dice)
        for b_col in potential_bool_cols:
            if b_col != target_column:
                if b_col in temp_num_dice:
                    temp_num_dice.remove(b_col)
                    if b_col not in temp_cat_dice: temp_cat_dice.append(b_col); updated_features_for_dice = True
                elif b_col not in temp_cat_dice: temp_cat_dice.append(b_col); updated_features_for_dice = True
        if updated_features_for_dice:
            print("Adjusted feature types for DiCE Data object initialization due to boolean columns.")
            numerical_features_dice = temp_num_dice
            categorical_features_dice = temp_cat_dice

        # 2. Create the dice_ml.Data object
        d = dice_ml.Data(dataframe=dice_data_df,
                         continuous_features=numerical_features_dice,
                         categorical_features=categorical_features_dice,
                         outcome_name=target_column)
        print("DiCE Data object created.")
        print(f"  DiCE Data Continuous Features: {d.continuous_feature_names}")
        print(f"  DiCE Data Categorical Features: {d.categorical_feature_names}")

        # 3. Create the Model Wrapper instance (passes preprocessing artifacts)
        print("Instantiating ModelWrapper...")
        wrapped_model = ModelWrapper(
            model=model_object, encoder=encoder, scaler=scaler,
            model_expected_features=model_expected_features,
            model_num_features=model_num_features, model_cat_features=model_cat_features
        )

        # 4. Create the dice_ml.Model object using the wrapper
        # This allows DiCE to call predict/predict_proba on our wrapped model
        model_dice = dice_ml.Model(model=wrapped_model, backend=model_backend_name, model_type='classifier')
        print(f"DiCE Model object created using ModelWrapper for backend: {model_backend_name}.")

        # 5. Instantiate the main DiCE explainer class
        dice_method = config.DICE_METHOD
        exp = dice_ml.Dice(d, model_dice, method=dice_method)
        print(f"DiCE explainer instantiated with method: {dice_method}.")
        print("--- DiCE Setup Complete ---")
        return exp, data_info # Return the explainer and potentially updated data_info

    except Exception as e:
         print(f"Unexpected error during DiCE initialization: {e}")
         traceback.print_exc()
         return None, data_info


# --- Counterfactual Generation ---
def generate_explanations(dice_explainer, query_df, data_info, desired_class_cf, immutable_features_list=[], num_cfs=config.NUM_COUNTERFACTUALS):
    """
    Generates counterfactual explanations for a given query instance.

    Args:
        dice_explainer (dice_ml.Dice): Initialized DiCE explainer object.
        query_df (pd.DataFrame): DataFrame containing the single instance to explain
                                 (with original feature values/dtypes).
        data_info (dict): Dictionary with feature information (user view, ranges, etc.).
        desired_class_cf (int): The target class (0 or 1) for the counterfactuals.
        immutable_features_list (list): Features that DiCE should not change.
        num_cfs (int): The number of counterfactual examples desired by the user.

    Returns:
        dice_ml.counterfactual_explanations.CounterfactualExplanations or None:
            The object containing DiCE results, or None if generation fails.

    Raises:
        UserConfigValidationException: If DiCE cannot find counterfactuals.
        RuntimeError: For other unexpected errors during generation.
    """
    if query_df is None or query_df.empty: print("Error (Generate): Query DataFrame is empty."); return None
    if dice_explainer is None: print("Error (Generate): DiCE explainer is not initialized."); return None
    if len(query_df) > 1: print("Warning (Generate): Received multiple query instances. Explaining only the first."); query_df = query_df.iloc[[0]]

    print(f"\n--- Generating Counterfactuals (User requested: {num_cfs}) ---")
    print(f"Desired class: {desired_class_cf}")

    # Prepare query instance for DiCE (using user-defined feature lists)
    all_user_features = data_info['numerical_features'] + data_info['categorical_features']
    missing_query_cols = [f for f in all_user_features if f not in query_df.columns]
    if missing_query_cols:
        print(f"Error (Generate): Query DataFrame missing expected user features: {missing_query_cols}")
        return None
    query_df_dice = query_df[all_user_features].copy() # Ensure correct columns

    # Determine features DiCE can modify
    features_that_can_vary = [f for f in all_user_features if f not in immutable_features_list]
    print(f"Immutable features: {immutable_features_list}")
    print(f"Features allowed to vary: {features_that_can_vary}")

    # Feature ranges constraint (REMOVED for broader search in this version)
    # feature_ranges = data_info.get('feature_ranges') # Get ranges if available
    print(f"Permitted feature ranges constraint: Not Applied")

    try:
        desired_class_val = int(desired_class_cf)

        # Increase the number of CFs DiCE searches for internally to improve chances
        # of finding `num_cfs` diverse and valid ones.
        total_cfs_to_generate = max(num_cfs * 50, 200)
        print(f"DiCE internal search: total_CFs={total_cfs_to_generate} (will return up to {num_cfs}).")

        # Call DiCE's generation method
        dice_exp_results = dice_explainer.generate_counterfactuals(
            query_instances=query_df_dice,
            total_CFs=total_cfs_to_generate,
            desired_class=desired_class_val,
            features_to_vary=features_that_can_vary
            # permitted_range constraint REMOVED
        )
        print("--- Counterfactual Generation Finished ---")

        # --- Post-Generation Checks and Handling ---
        # Check if DiCE found any valid counterfactuals at all
        if not dice_exp_results or \
           not dice_exp_results.cf_examples_list or \
           dice_exp_results.cf_examples_list[0].final_cfs_df is None or \
           dice_exp_results.cf_examples_list[0].final_cfs_df.empty:
             print("Warning: DiCE finished but found no valid counterfactuals.")
             # Raise the specific DiCE exception for clarity in the UI
             raise UserConfigValidationException(
                 "No counterfactuals found for the query point. The instance might be too far from the decision boundary, or constraints might be too restrictive."
             )

        # Trim results if DiCE returned more than the user requested
        num_found = len(dice_exp_results.cf_examples_list[0].final_cfs_df)
        if num_found > num_cfs:
            print(f"Trimming DiCE results from {num_found} to requested {num_cfs}")
            # Trim the DataFrame
            dice_exp_results.cf_examples_list[0].final_cfs_df = \
                dice_exp_results.cf_examples_list[0].final_cfs_df.head(num_cfs)
            # Also trim the internal list representation if it exists
            if hasattr(dice_exp_results.cf_examples_list[0], 'final_cfs_list') and \
               isinstance(dice_exp_results.cf_examples_list[0].final_cfs_list, list):
                 dice_exp_results.cf_examples_list[0].final_cfs_list = \
                     dice_exp_results.cf_examples_list[0].final_cfs_list[:num_cfs]

        return dice_exp_results

    except UserConfigValidationException as e:
        # Catch DiCE's specific configuration/validation error
        print(f"DiCE UserConfigValidationException during generation: {e}")
        print("Query Instance (passed to DiCE):")
        print(query_df_dice.head().to_string())
        raise e # Re-raise to be caught by UI

    except Exception as e:
        # Catch other potential errors during generation
        print(f"Unexpected error during counterfactual generation: {e}")
        print("Query Instance (passed to DiCE):")
        print(query_df_dice.head().to_string())
        traceback.print_exc()
        # Re-raise a generic error for the UI
        raise RuntimeError(f"Counterfactual generation failed: {e}") from e