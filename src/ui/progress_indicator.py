# File: src/ui/progress_indicator.py
"""
UI Module: Workflow Progress Indicator
--------------------------------------
Provides a simple visual indicator component displayed in the Streamlit
sidebar to show the user's current stage in the application workflow.
"""

# --- Imports ---
import streamlit as st

# --- Main Function ---
def show_progress_indicator():
    """
    Displays a visual indicator reflecting the current step in the
    application workflow based on relevant session state flags.
    Uses icons (✅, ⏳, ⬜) and bold text for the current step.
    """
    st.markdown("### Workflow Progress")

    # --- Determine Current Step ---
    # The current step is determined by checking the state flags sequentially.
    if not st.session_state.get('selected_scenario_key'):
        current_step = 1 # Step 1: No scenario selected
    elif not st.session_state.get('app_ready', False):
        current_step = 2 # Step 2: Scenario selected, but setup (loading/processing) is ongoing or failed
    elif st.session_state.get('selected_instance_index') is None:
        current_step = 3 # Step 3: Setup complete, waiting for instance selection
    elif st.session_state.get('dice_results') is None:
        current_step = 4 # Step 4: Instance selected, ready to generate counterfactuals
    else:
        current_step = 5 # Step 5: Counterfactuals generated, results are being explored

    # --- Define Workflow Steps ---
    # These labels correspond to the logical stages of using the application.
    steps = [
        "1. Select Scenario",
        "2. Load Scenario",    # Represents the loading/setup phase
        "3. Select Instance",
        "4. Generate CFs",     # Counterfactuals
        "5. Explore Results"
    ]

    # --- Display Step Indicators ---
    # Use Streamlit columns for a horizontal layout.
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        step_number = i + 1
        with col:
            if step_number < current_step:
                # Mark past steps as completed
                st.markdown(f"✅ <small>{step}</small>", unsafe_allow_html=True)
            elif step_number == current_step:
                # Highlight the current step
                st.markdown(f"⏳ **<small>{step}</small>**", unsafe_allow_html=True)
            else:
                # Mark future steps as pending
                st.markdown(f"⬜ <small>{step}</small>", unsafe_allow_html=True)

    st.divider() # Visual separator
