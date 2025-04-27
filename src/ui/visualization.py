# File: src/ui/visualization.py
"""
UI Module: Visualization Components
-----------------------------------
Provides functions for creating visualizations related to counterfactual
explanations, primarily focusing on comparing feature values between the
original instance and the generated counterfactuals using Plotly.
"""

# --- Imports ---
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import traceback # For detailed error logging

# --- Main Plotting Function ---
def plot_feature_comparison(changed_cols, comparison_df):
    """
    Renders a plot or table comparing feature values between the original instance
    and counterfactual examples for a user-selected feature.

    Handles both numeric features (bar chart) and categorical/non-numeric features (table).

    Args:
        changed_cols (list): List of column names that had differing values
                             between the original and at least one counterfactual.
        comparison_df (pd.DataFrame): DataFrame containing the original instance
                                      (first row, index='Original') and counterfactuals
                                      (subsequent rows) for the changed columns only.
                                      Assumes original data types.
    """
    # --- Input Validation ---
    if not changed_cols:
        st.info("No features were changed, so no comparison visualization is available.")
        return
    if comparison_df is None or comparison_df.empty:
        st.warning("Comparison data is missing or empty, cannot generate visualization.")
        return

    st.markdown("**Compare Feature Values:**")

    # --- Feature Selection Dropdown ---
    # Add a placeholder option for user guidance and sort features alphabetically
    options_for_graph = ["-- Select a feature --"] + sorted(changed_cols)

    # Use a unique key for the selectbox to maintain state correctly across reruns
    # Incorporate instance index if available to reset selection when instance changes
    instance_idx_key = st.session_state.get('selected_instance_index', 'default_instance')
    plot_key = f"feature_plot_select_{instance_idx_key}"

    feature_to_plot = st.selectbox(
        "Select a changed feature to visualize:",
        options=options_for_graph,
        index=0, # Default to the placeholder
        key=plot_key,
        help="Choose which feature's values you want to compare across the original and counterfactuals."
    )

    # --- Plotting/Table Generation Logic ---
    # Proceed only if a valid feature (not the placeholder) is selected
    if feature_to_plot and feature_to_plot != "-- Select a feature --":
        if feature_to_plot not in comparison_df.columns:
             st.error(f"Selected feature '{feature_to_plot}' not found in the comparison data.")
             return

        try:
            # Extract the data series (column) for the selected feature
            values_to_plot = comparison_df[feature_to_plot]

            # --- Create Explicit String Labels for X-axis/Table Index ---
            # Ensures consistent labeling regardless of original DataFrame index
            labels = ["Original"] + [f"CF {i}" for i in range(len(comparison_df) - 1)]
            if len(labels) != len(values_to_plot):
                 print(f"Warning (Plot/Table): Mismatch between number of labels ({len(labels)}) and data rows ({len(values_to_plot)}).")
                 # Adjust labels or values if possible, or raise error
                 labels = labels[:len(values_to_plot)] # Simple truncation for now

            # --- Determine if Feature is Numeric for Plotting ---
            # Attempt numeric conversion; if all results are NaN, treat as non-numeric
            numeric_values = pd.to_numeric(values_to_plot, errors='coerce')
            is_numeric = not numeric_values.isna().all() # True if at least one value is numeric

            if is_numeric:
                # --- Numeric Feature: Create Bar Chart ---
                numeric_values_list = numeric_values.tolist() # Convert to list for Plotly

                # Define colors for bars (different color for 'Original')
                bar_colors = ['steelblue' if label == 'Original' else '#94a786' for label in labels]
                # Format text labels for bars (showing values)
                bar_text = [f"{v:.2f}" if pd.notna(v) else "N/A" for v in numeric_values_list]

                # Create Plotly Figure object
                fig = go.Figure(data=[
                    go.Bar(
                        x=labels,           # Use explicit string labels for categories
                        y=numeric_values_list, # Use numeric data for bar heights
                        marker_color=bar_colors,
                        text=bar_text,
                        textposition='auto', # Position text automatically on bars
                        name=feature_to_plot # Name for legend/hover
                    )
                ])
                # Configure layout properties
                fig.update_layout(
                    title=f"Comparison for '{feature_to_plot}'",
                    xaxis_title="Instance Type",
                    yaxis_title="Feature Value",
                    title_x=0.5, # Center title
                    height=400,
                    barmode='group', # Ensure bars are grouped side-by-side
                    xaxis_type='category', # Treat x-axis labels as distinct categories
                    # Explicitly set x-axis ticks to match labels for clarity
                    xaxis=dict(
                        tickmode='array',
                        tickvals=labels, # Define positions for the ticks
                        ticktext=labels  # Define the text labels for the ticks
                    )
                )
                # Display the plot in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            else:
                # --- Categorical/Non-Numeric Feature: Display Table ---
                st.write(f"Values for **{feature_to_plot}** (Categorical/Non-Numeric):")

                # Create a simple DataFrame for display using the explicit labels
                categorical_df = pd.DataFrame({
                    'Instance': labels,
                    'Value': values_to_plot.astype(str).fillna("N/A") # Ensure string type and handle NaNs
                })
                categorical_df = categorical_df.set_index('Instance') # Use 'Instance' column as index

                # Display the table
                try:
                    st.dataframe(categorical_df, use_container_width=True, hide_index=False)
                except Exception as display_err:
                     st.error(f"Error displaying categorical table: {display_err}")
                     print(f"Error displaying categorical table: {display_err}")
                     traceback.print_exc()

        except Exception as plot_err:
            # Catch-all for errors during plotting or table generation
            st.error(f"Could not generate visualization for '{feature_to_plot}': {plot_err}")
            print(f"Visualization Error Details: {plot_err}")
            traceback.print_exc()

    elif not feature_to_plot or feature_to_plot == "-- Select a feature --":
        # Guide user if no feature is selected from the dropdown
         st.caption("Select a feature above to see the comparison plot or table.")
