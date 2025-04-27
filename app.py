# File: app.py
"""
Streamlit Application: Counterfactual Explainer (Project Demo)
---------------------------------------------------------------
This application loads pre-defined scenarios (data + model combinations)
and allows users to generate and explore counterfactual explanations using
the DiCE (Diverse Counterfactual Explanations) framework.
"""

# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import json
import os

# Project Modules
from src import config
from src.ui import (
    render_instance_selector_and_constraints,
    render_main_explanation_area,
    handle_explainer_setup,
    show_progress_indicator
)
from src import state_manager
from src.data import load_user_data

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Counterfactual Explainer", page_icon="💡")

# --- Initialize Session State ---
state_manager.initialize_session_state()

# --- Streamlit UI: Main Title & Description ---
st.title("💡 Counterfactual Explainer (Project Demo)")
st.markdown("""
    This app demonstrates counterfactual explanations for specific project scenarios.

    **How it works:**
    1. Select a Scenario (Dataset + Model) from the sidebar.
    2. Choose an instance to explain.
    3. Explore counterfactual explanations showing alternative scenarios that would lead to a different outcome.
""")

# --- Sidebar Interface ---
with st.sidebar:
    # --- Workflow Progress Indicator ---
    show_progress_indicator()

    # --- 1. Scenario Selection ---
    st.header("1. Select Scenario")
    scenario_options = list(config.PROJECT_SCENARIOS.keys())
    options_with_placeholder = ["-- Select --"] + scenario_options
    selected_scenario_key = st.selectbox(
        "Choose a dataset and model to explain:",
        options=options_with_placeholder,
        index=0, # Default to placeholder
        key="scenario_selector"
    )

    # --- 2. Load & Setup Scenario ---
    # Store the previously selected scenario to detect changes
    previous_scenario = st.session_state.get('selected_scenario_key', None)

    if selected_scenario_key != "-- Select --":
        st.session_state.selected_scenario_key = selected_scenario_key # Store current selection
        # Trigger reload if the scenario changed OR if the app isn't ready for the current scenario
        if selected_scenario_key != previous_scenario or not st.session_state.app_ready:
            st.session_state.app_ready = False # Mark as not ready while loading
            st.session_state.dice_explainer = None # Reset explainer
            st.session_state.dice_results = None # Reset results
            st.session_state.selected_instance_index = None # Reset instance selection

            st.info(f"Loading scenario: {selected_scenario_key}...")
            with st.spinner("Loading data, model, and setting up explainer..."):
                # Call the setup handler which loads based on the selected key
                setup_successful = handle_explainer_setup(selected_scenario_key)

            if setup_successful:
                st.sidebar.success(f"Scenario '{selected_scenario_key}' loaded!")
                # Rerun to update the main panel now that app_ready is True
                st.rerun()
            else:
                st.sidebar.error(f"Failed to load scenario '{selected_scenario_key}'. Check logs.")
                st.session_state.app_ready = False # Ensure it stays not ready
                st.session_state.selected_scenario_key = None # Reset selection on failure
    else:
        # If placeholder "-- Select --" is chosen, reset the application state
        if st.session_state.get('app_ready', False):
            st.session_state.app_ready = False
            st.session_state.selected_scenario_key = None
            st.session_state.dice_explainer = None
            st.session_state.dice_results = None
            st.session_state.selected_instance_index = None
            st.rerun() # Rerun to clear the main panel

    # --- 3. Instance Selection & Constraints ---
    # Display only if a scenario is successfully loaded
    if st.session_state.app_ready and st.session_state.selected_scenario_key != "-- Select --":
        render_instance_selector_and_constraints()
    elif selected_scenario_key != "-- Select --" and not st.session_state.app_ready:
        # Indicate loading/failure state if a scenario was selected but not ready
        st.warning("Scenario setup in progress or failed.")

# --- Main Display Area ---
if st.session_state.app_ready and st.session_state.selected_scenario_key != "-- Select --":
    # Display counterfactual generation UI if setup is complete
    scenario_display_name = config.PROJECT_SCENARIOS[st.session_state.selected_scenario_key]['display_name']
    st.header(f"Explaining: {scenario_display_name}")
    render_main_explanation_area()
else:
    # Show initial instructions if no scenario is loaded
    st.info("⬅️ Please select a scenario in the sidebar to begin.")