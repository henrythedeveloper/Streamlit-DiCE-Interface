# File: src/ui/instance_selector.py
"""
UI Module: Instance Selection and Constraints
---------------------------------------------
Provides the Streamlit UI components in the sidebar for:
1. Filtering instances based on the model's predicted outcome (Class 0 or Class 1).
2. Selecting a specific instance (by index) from the filtered list to explain.
3. Specifying features that should remain constant (immutable) during
   counterfactual generation.
"""

# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import traceback # For detailed error logging

# --- Main Rendering Function ---
def render_instance_selector_and_constraints():
    """
    Renders the instance selection dropdown and the immutable features
    multiselect widget in the Streamlit sidebar. Updates session state
    based on user selections.
    """
    st.header("3. Select Instance & Constraints")

    # --- Help Expander ---
    with st.expander("ℹ️ Help: Selecting an Instance", expanded=False):
        st.markdown("""
        **Purpose:** Choose the specific data point (instance) from your dataset that you want the application to explain. The counterfactuals generated will show how this particular instance could be modified to achieve a different prediction from the model.

        **Steps:**
        1.  **Filter by Prediction:** Select whether you want to see instances the model predicted as Class 0 or Class 1. This helps narrow down the list.
        2.  **Select Instance:** Choose the specific instance index from the dropdown menu.
        3.  **(Optional) Set Constraints:** Use the multiselect box below to specify any features that should *not* be changed by the counterfactual generation process (e.g., features like 'Age' or 'Race' that are often considered immutable).
        """)

    # --- Instance Selection Logic ---
    instance_index = None
    options_available = False

    # Retrieve necessary data from session state
    predictions_app = st.session_state.get('predictions_app')
    X_app = st.session_state.get('X_app') # Processed features (index is used)

    if predictions_app is not None and X_app is not None:
        try:
            # Validate alignment between predictions and processed data index
            if len(predictions_app) != len(X_app):
                st.error("Internal Error: Mismatch between number of predictions and processed data rows.")
                return False # Indicate failure

            # Create a pandas Series for easier filtering, using X_app's index
            predictions_series = pd.Series(predictions_app, index=X_app.index)

            # Display prediction counts for context
            pred_counts = predictions_series.value_counts().sort_index()
            count_0 = pred_counts.get(0, 0)
            count_1 = pred_counts.get(1, 0)
            st.caption(f"Model Predictions: Class 0: {count_0}, Class 1: {count_1}")

            # --- Filter Instances by Predicted Outcome ---
            explore_outcome = st.radio(
                "Filter instances by predicted outcome:", (0, 1), key="explore_outcome_filter",
                format_func=lambda x: f"Predicted Class {x}", index=0, horizontal=True,
                help="Show instances that the model predicted as either Class 0 or Class 1.",
                # Reset instance/results state when the filter changes
                on_change=lambda: st.session_state.update(selected_instance_index=None, dice_results=None, explained_instance_df=None)
            )

            # Get the indices matching the selected predicted outcome
            options_list = X_app.index[predictions_series == explore_outcome].tolist()

            # --- Instance Selection Dropdown ---
            if options_list:
                current_selection = st.session_state.get('selected_instance_index')
                # Reset selection if the previously selected index isn't in the new filtered list
                if current_selection not in options_list:
                    current_selection = None

                # Determine default selection (use current if valid, else first option)
                default_selection = current_selection if current_selection is not None else options_list[0]
                try:
                    # Find index of the default selection within the options list for the widget
                    default_idx = options_list.index(default_selection)
                except ValueError:
                    default_idx = 0 # Fallback to first item if default not found

                # Create the selectbox widget
                selected_idx = st.selectbox(
                    f"Select Instance Index (Predicted as {explore_outcome}):",
                    options=options_list,
                    key="instance_selector",
                    index=default_idx,
                    help="Choose the specific data point (by its original index) you want to explain.",
                    # Reset explanation results when the instance selection changes
                    on_change=lambda: st.session_state.update(dice_results=None, explained_instance_df=None)
                )
                # Update session state with the selected index
                st.session_state.selected_instance_index = selected_idx
                options_available = True # Flag that selection was successful
            else:
                # Handle case where no instances match the filter
                st.warning(f"No instances found where the model predicted outcome {explore_outcome}.")
                st.session_state.selected_instance_index = None # Clear selection

        except Exception as e:
            st.error(f"Error during instance selection setup: {e}")
            traceback.print_exc()
            st.session_state.selected_instance_index = None

    else:
        # Handle case where prerequisites (predictions/X_app) are missing
        st.warning("Model predictions or processed data not available. Complete setup first.")
        st.session_state.selected_instance_index = None


    # --- Immutable Features Selection ---
    st.markdown("**Feature Constraints (Optional)**")
    if st.session_state.data_info:
        # Get the list of all features (as seen by the user) from data_info
        all_user_features = st.session_state.data_info['numerical_features'] + st.session_state.data_info['categorical_features']

        # Ensure the default value for the multiselect only contains valid features
        current_immutable = st.session_state.get('immutable_features_selection', [])
        valid_default_immutable = [f for f in current_immutable if f in all_user_features]

        # Create the multiselect widget
        selected_immutable = st.multiselect(
             "Select features to keep CONSTANT (immutable):",
             options=sorted(all_user_features), # Sort options alphabetically
             default=valid_default_immutable,
             key="immutable_select",
             help="DiCE will not change these features when generating counterfactuals (e.g., Age, Race)."
        )
        # Update session state directly if the selection changes
        if set(selected_immutable) != set(st.session_state.immutable_features_selection):
            st.session_state.immutable_features_selection = selected_immutable
            # No rerun needed, state is just updated for the 'Generate' button logic

    else:
        # Handle case where feature list isn't available yet
        st.warning("Feature list not available (complete setup first).")

    return options_available # Return flag indicating if instance selection was possible

