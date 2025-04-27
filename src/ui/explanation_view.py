# File: src/ui/explanation_view.py
"""
UI Module: Explanation Display Area
-----------------------------------
Renders the main explanation section in the Streamlit application.
This includes displaying the selected instance's details, prediction,
triggering counterfactual generation, and presenting the results
(tables, visualizations, quality metrics).
"""

# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import traceback # For detailed error logging

# Project Modules
from .. import config
from .. import explain
from .. import evaluate
from .visualization import plot_feature_comparison

# --- Helper Function: DataFrame Styling for Differences ---
def highlight_diff(data, original_series_to_compare):
    """
    Applies CSS background styling to highlight cells in a DataFrame
    that differ from a corresponding comparison Series (original instance).

    Handles both numeric comparisons (with tolerance) and non-numeric
    (string-based) comparisons. Assumes inputs have original data types.

    Args:
        data (pd.DataFrame): The DataFrame to style (e.g., counterfactuals).
        original_series_to_compare (pd.Series): The original instance row.

    Returns:
        pd.DataFrame: DataFrame of CSS styles for differing cells.
                      Returns an empty style DataFrame on error or invalid input.
    """
    try:
        # --- Input Validation ---
        if not isinstance(data, pd.DataFrame):
            print(f"Error (highlight_diff): Input 'data' is type {type(data)}, expected DataFrame.")
            return pd.DataFrame('', index=data.index, columns=data.columns)
        if not isinstance(original_series_to_compare, pd.Series):
            print(f"Error (highlight_diff): Input 'original_series_to_compare' is type {type(original_series_to_compare)}, expected Series.")
            return pd.DataFrame('', index=data.index, columns=data.columns)

        # --- Alignment and Initialization ---
        common_cols = data.columns.intersection(original_series_to_compare.index)
        if not common_cols.any():
            print("Warning (highlight_diff): No common columns found for comparison.")
            return pd.DataFrame('', index=data.index, columns=data.columns)

        data_aligned = data[common_cols]
        original_aligned = original_series_to_compare[common_cols]

        style_df = pd.DataFrame('', index=data_aligned.index, columns=data_aligned.columns)
        highlight_style = 'background-color: #94a78633' # Light green accent

        # --- Column-wise Comparison Logic ---
        for col in data_aligned.columns:
            original_val = original_aligned[col]
            data_col = data_aligned[col]

            # Determine data types for appropriate comparison
            is_numeric_data = pd.api.types.is_numeric_dtype(data_col)
            is_numeric_original = pd.api.types.is_numeric_dtype(original_val)

            if is_numeric_data and is_numeric_original:
                # Numeric comparison: Use numpy.isclose for float tolerance
                original_val_float = float(original_val) if pd.notna(original_val) else np.nan
                data_col_numeric = pd.to_numeric(data_col, errors='coerce').fillna(np.nan)
                # Compare, treating NaNs as equal only if both are NaN
                diff_mask_col = ~np.isclose(data_col_numeric, original_val_float, equal_nan=True, atol=1e-6, rtol=1e-6)
            else:
                # Non-numeric/Mixed comparison: Use string representation
                original_val_s = str(original_val) if pd.notna(original_val) else 'NaN'
                data_col_s = data_col.apply(lambda x: str(x) if pd.notna(x) else 'NaN')
                diff_mask_col = data_col_s.ne(original_val_s)

            # Apply highlight style where differences exist
            style_df.loc[diff_mask_col, col] = highlight_style

        return style_df

    except Exception as e:
        print(f"Error during highlight_diff execution: {e}")
        traceback.print_exc()
        # Return empty styles on any unexpected error
        return pd.DataFrame('', index=data.index, columns=data.columns)


# --- Main UI Rendering Function ---
def render_main_explanation_area():
    """
    Renders the main content area of the Streamlit app, responsible for
    displaying the selected instance, triggering counterfactual generation,
    and presenting the explanation results.
    """
    instance_index = st.session_state.get('selected_instance_index')

    # Guide user if no instance is selected yet
    if instance_index is None:
        st.info("⬅️ Select an instance in the sidebar (Step 3) to explain.")
        return

    # Retrieve necessary data from session state
    dice_data_df = st.session_state.get('dice_data_df') # Original data for display/DiCE
    y_app = st.session_state.get('y_app') # Processed target (0/1)
    predictions_app = st.session_state.get('predictions_app') # Model predictions (0/1)
    probabilities_app = st.session_state.get('probabilities_app') # Model probabilities
    X_app = st.session_state.get('X_app') # Processed features for model input

    # Validate required data is available
    if dice_data_df is None or y_app is None or predictions_app is None or X_app is None:
        st.warning("Data or predictions not fully loaded. Please complete the setup process first.")
        return

    try:
        target_col_name = st.session_state.data_info['target_column']
        if target_col_name not in dice_data_df.columns:
            st.error(f"Internal Error: Target column '{target_col_name}' missing from display data (dice_data_df).")
            return

        # --- Display Selected Instance Details ---
        st.subheader(f"Selected Instance (Index: {instance_index})")
        try:
            # Extract the full original instance data using the selected index
            instance_to_display_full = dice_data_df.loc[[instance_index]]
        except KeyError:
            st.error(f"Error: Index {instance_index} not found in the display data (dice_data_df). It might have been filtered during loading.")
            st.session_state.selected_instance_index = None # Reset selection
            return

        # Align predictions and probabilities with the selected instance
        try:
            # Handle different prediction storage types (Series, DataFrame, ndarray)
            if isinstance(predictions_app, (pd.Series, pd.DataFrame)):
                if instance_index not in predictions_app.index:
                    raise KeyError(f"Index {instance_index} not found in predictions index.")
                predicted_outcome = predictions_app.loc[instance_index]
                # Safely access probability for class 1
                predicted_prob_val = probabilities_app.loc[instance_index, 1] if (
                    probabilities_app is not None and
                    isinstance(probabilities_app, pd.DataFrame) and
                    1 in probabilities_app.columns and
                    instance_index in probabilities_app.index
                ) else None
            elif isinstance(predictions_app, np.ndarray):
                # Map DataFrame index to numpy array position using X_app's index
                try:
                    numpy_pos = X_app.index.get_loc(instance_index)
                    if numpy_pos >= len(predictions_app):
                        raise IndexError("Mapped numpy position out of bounds for predictions array.")
                    predicted_outcome = predictions_app[numpy_pos]
                    # Safely access probability for class 1
                    predicted_prob_val = probabilities_app[numpy_pos, 1] if (
                        probabilities_app is not None and
                        probabilities_app.ndim == 2 and
                        probabilities_app.shape[1] > 1 and
                        numpy_pos < len(probabilities_app)
                    ) else None
                except KeyError:
                    st.error(f"Internal Error: Selected index {instance_index} not found in processed features index (X_app) for numpy mapping.")
                    traceback.print_exc()
                    return
                except IndexError as idx_err:
                    st.error(f"Internal Error: Index mapping error accessing numpy arrays. {idx_err}")
                    traceback.print_exc()
                    return
            else:
                raise TypeError(f"Unsupported type for predictions storage: {type(predictions_app)}")

            # Get the actual outcome (from processed target)
            actual_outcome = y_app.loc[instance_index]
            predicted_prob_str = f"{predicted_prob_val:.3f}" if predicted_prob_val is not None else "N/A"

        except (KeyError, IndexError, TypeError) as e:
            st.error(f"Internal Error: Could not align selected index {instance_index} with processed data/predictions. Error: {e}")
            traceback.print_exc()
            return

        # Store the features of the instance being explained (for DiCE and comparison)
        # Drop the target column
        st.session_state.explained_instance_df = instance_to_display_full.drop(columns=[target_col_name], errors='ignore')

        # Display the original instance features (convert to string for robust display)
        df_display_orig = st.session_state.explained_instance_df.copy()
        df_display_orig_str = df_display_orig.fillna('NaN').astype(str)
        st.write("Original Instance Features:")
        st.dataframe(df_display_orig_str, use_container_width=True)

        # Display outcome metrics
        col1, col2 = st.columns(2)
        actual_display = int(actual_outcome) if pd.notna(actual_outcome) else "N/A"
        col1.metric("Actual Outcome", actual_display)
        col2.metric("Predicted Outcome", f"{predicted_outcome} (Prob: {predicted_prob_str})")

    except KeyError as ke:
        st.error(f"Error accessing data for index {instance_index}: {ke}. Index mismatch or data filtering issue.")
        st.session_state.selected_instance_index = None # Reset selection
        traceback.print_exc()
        return
    except Exception as e:
        st.error(f"An unexpected error occurred displaying instance details: {e}")
        traceback.print_exc()
        return

    # --- Counterfactual Generation Section ---
    st.subheader("Generate Counterfactuals")
    # Determine the desired outcome (opposite of the predicted outcome)
    desired_class_cf = 1 - predicted_outcome
    immutable_features = st.session_state.get('immutable_features_selection', [])
    num_cfs_to_gen = config.NUM_COUNTERFACTUALS

    # Disable button if DiCE explainer isn't ready
    button_disabled = st.session_state.get('dice_explainer') is None
    button_tooltip = "Complete setup first (select scenario)" if button_disabled else f"Find instances similar to the selected one, but predicted as Class {desired_class_cf}."

    # Generation Button
    if st.button(f"Generate {num_cfs_to_gen} Counterfactuals (Target: Class {desired_class_cf})",
                 key="generate_cf_button", disabled=button_disabled, help=button_tooltip):
        st.session_state.dice_results = None # Clear previous results before generating new ones

        # Get the instance data (features only) to pass to DiCE
        query_instance_for_dice = st.session_state.get('explained_instance_df')
        if query_instance_for_dice is None or query_instance_for_dice.empty:
            st.error("Could not retrieve the instance features to generate counterfactuals.")
            return

        # Show spinner during generation process
        with st.spinner("Generating counterfactuals... This might take a moment."):
            try:
                # Call the explanation generation function from explain.py
                st.session_state.dice_results = explain.generate_explanations(
                    dice_explainer=st.session_state.dice_explainer,
                    query_df=query_instance_for_dice,
                    data_info=st.session_state.data_info,
                    desired_class_cf=desired_class_cf,
                    immutable_features_list=immutable_features,
                    num_cfs=num_cfs_to_gen
                )
            except Exception as gen_err:
                # Display errors encountered during the generation process
                st.error(f"Error during counterfactual generation: {gen_err}")
                print(f"Counterfactual generation failed: {gen_err}")
                traceback.print_exc()
                st.session_state.dice_results = None # Ensure results are cleared on error

    # --- Display Results Area ---
    if st.session_state.get('dice_results'):
        # If results exist in session state, call the display function
        display_dice_results(
            dice_results=st.session_state.dice_results,
            original_instance_df=st.session_state.explained_instance_df,
            desired_class_cf=desired_class_cf,
            encoder=st.session_state.get('encoder'),
            scaler=st.session_state.get('scaler'),
            data_info=st.session_state.get('data_info'),
            model=st.session_state.get('model')
        )
    elif button_disabled:
        # Guide user if setup isn't complete
        st.info("Setup not complete. Please ensure a scenario is selected and loaded in the sidebar.")


# --- Function to Display DiCE Results ---
def display_dice_results(dice_results, original_instance_df, desired_class_cf,
                         encoder, scaler, data_info, model):
    """
    Displays the generated counterfactuals from the DiCE results object,
    including tables, visualizations, and calculated quality metrics.

    Args:
        dice_results: The CounterfactualExplanations object returned by DiCE.
        original_instance_df (pd.DataFrame): DataFrame of the original instance features.
        desired_class_cf (int): The target class for the counterfactuals (0 or 1).
        encoder: Fitted encoder object used for evaluation preprocessing.
        scaler: Fitted scaler object used for evaluation preprocessing.
        data_info (dict): Dictionary containing feature lists and other info.
        model: The trained ML model object for validity checks.
    """
    try:
        # --- Extract Counterfactual DataFrame ---
        if not hasattr(dice_results, 'cf_examples_list') or not dice_results.cf_examples_list:
            st.info("DiCE results object is missing the expected 'cf_examples_list'. Cannot display results.")
            return

        cf_examples = dice_results.cf_examples_list[0] # Assuming explanation for one instance
        cfs_df = cf_examples.final_cfs_df # The DataFrame containing counterfactuals

        if cfs_df is None or cfs_df.empty:
            st.info("No counterfactual examples were generated for this instance with the current settings.")
            # Display DiCE summary message if available
            if hasattr(cf_examples, 'summary_str') and cf_examples.summary_str:
                 st.caption(f"DiCE Summary: {cf_examples.summary_str}")
            return

        # Validate original instance data is available for comparison
        if not isinstance(original_instance_df, pd.DataFrame) or original_instance_df.empty:
             st.error("Internal Error: Original instance data for comparison is invalid.")
             return

        st.subheader(f"Generated Counterfactuals (Target: Class {desired_class_cf})")
        st.write(f"Found {len(cfs_df)} counterfactual(s):")
        st.divider()

        # --- Display Full Counterfactuals Table with Highlighting ---
        st.markdown("#### Full Counterfactuals")
        st.caption("Highlighted cells show changes from the original input.")

        # Align columns between original and counterfactuals for accurate comparison
        common_cols = original_instance_df.columns.intersection(cfs_df.columns)
        if not common_cols.any():
             st.error("No common columns between original instance and counterfactuals for display.")
             return

        original_instance_aligned = original_instance_df[common_cols]
        cfs_df_aligned = cfs_df[common_cols].copy()

        if original_instance_aligned.empty:
             st.error("Original instance DataFrame became empty after aligning columns.")
             return
        original_series = original_instance_aligned.iloc[0] # Get the original instance as a Series

        # Apply styling using the helper function
        try:
            # Reset index before styling to avoid potential Styler index alignment issues
            cfs_df_aligned_reset = cfs_df_aligned.reset_index(drop=True)

            # Apply highlighting style
            styled_cfs = cfs_df_aligned_reset.style.apply(
                highlight_diff, original_series_to_compare=original_series, axis=None
            )
            # Display the styled DataFrame
            st.dataframe(styled_cfs, use_container_width=True)

        except Exception as style_err:
             st.error(f"Error applying highlighting styles to DataFrame: {style_err}")
             print(f"Styling Error Details: {style_err}")
             traceback.print_exc()
             # Fallback: Display as strings without styling if error occurs
             st.dataframe(cfs_df_aligned.fillna('NaN').astype(str), use_container_width=True)

        st.divider()

        # --- Display Changes Only Table ---
        st.markdown("#### Feature Changes Summary")
        st.caption("This table shows only the features that changed in each counterfactual.")

        # Calculate differences mask robustly (similar logic to highlight_diff)
        diff_mask_full = pd.DataFrame(False, index=cfs_df_aligned.index, columns=cfs_df_aligned.columns)
        for col in cfs_df_aligned.columns:
            original_val = original_series[col]
            data_col = cfs_df_aligned[col]
            is_numeric_data = pd.api.types.is_numeric_dtype(data_col)
            is_numeric_original = pd.api.types.is_numeric_dtype(original_val)
            if is_numeric_data and is_numeric_original:
                original_val_float = float(original_val) if pd.notna(original_val) else np.nan
                data_col_numeric = pd.to_numeric(data_col, errors='coerce').fillna(np.nan)
                diff_mask_full[col] = ~np.isclose(data_col_numeric, original_val_float, equal_nan=True, atol=1e-6, rtol=1e-6)
            else:
                original_val_s = str(original_val) if pd.notna(original_val) else 'NaN'
                data_col_s = data_col.apply(lambda x: str(x) if pd.notna(x) else 'NaN')
                diff_mask_full[col] = data_col_s.ne(original_val_s)

        # Identify columns that had at least one change
        changed_cols = diff_mask_full.any(axis=0)[lambda x: x].index.tolist()

        comparison_df = None # Initialize comparison DataFrame
        if changed_cols:
            # Select only the changed columns from original and counterfactuals
            cfs_changes_only = cfs_df_aligned[changed_cols]
            original_changes = original_instance_aligned[changed_cols]
            original_changes.index = ["Original"] # Set index label for clarity

            # Combine original changes and counterfactual changes
            comparison_df = pd.concat([original_changes, cfs_changes_only])

            # Convert to string for robust display
            comparison_df_str = comparison_df.fillna('NaN').astype(str)
            st.dataframe(comparison_df_str, use_container_width=True)
        else:
            st.info("No features were changed in the generated counterfactuals.")


        st.divider()

        # --- Display Visual Comparison Plot ---
        st.markdown("#### Visual Comparison")
        if changed_cols and comparison_df is not None:
            # Pass the changed columns and the comparison DataFrame to the plotting function
            plot_feature_comparison(changed_cols, comparison_df)
        else:
            st.info("No features were changed, so no comparison plot is available.")


        st.divider()

        # --- Display Explanation Quality Metrics ---
        st.markdown("#### Explanation Quality Metrics")

        # Calculate metrics using evaluate.py functions
        # Pass original data (aligned) and CF data (aligned) for calculations
        sparsity = evaluate.calculate_sparsity(original_instance_aligned, cfs_df_aligned)
        # Pass preprocessing artifacts for distance-based metrics
        proximity_val = evaluate.calculate_proximity(
            original_instance_aligned, cfs_df_aligned, data_info, encoder, scaler
        )
        diversity_val = evaluate.calculate_diversity(
            cfs_df_aligned, data_info, encoder, scaler
        )
        # Pass model and preprocessing artifacts for validity check
        validity = evaluate.check_validity(
            cfs_df_aligned, model, desired_class_cf, data_info, encoder, scaler
        )

        # Display Metrics in columns
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Sparsity", value=f"{sparsity:.3f}" if sparsity is not None else "N/A",
                      help="Higher=fewer changes (closer to 1.0 is better). Avg. fraction of features *not* changed.")
        col_m2.metric("Proximity", value=f"{proximity_val:.3f}" if proximity_val is not None else "N/A",
                      help="Lower=closer to original. Avg. L1 distance between original & CFs (after processing).")
        col_m3.metric("Diversity", value=f"{diversity_val:.3f}" if diversity_val is not None else "N/A",
                      help="Higher=more varied options. Avg. L1 distance between pairs of CFs (after processing).")
        col_m4.metric("Validity", value=f"{validity:.1%}" if validity is not None else "N/A",
                      help="% of generated CFs predicted as the desired class by the model.")

        st.caption("📊 **Interpreting metrics:** Sparsity: Higher=fewer changes. Proximity: Lower=closer to original. Diversity: Higher=more varied options. Validity: % that achieved target.")

    except AttributeError as ae:
        st.error(f"Error accessing DiCE results structure: {ae}. The format might be unexpected.")
        print(f"Error processing DiCE results object. Type: {type(dice_results)}, Content snippet: {str(dice_results)[:200]}")
        traceback.print_exc()
    except Exception as disp_err:
        st.error(f"An unexpected error occurred while displaying results: {disp_err}")
        traceback.print_exc()
