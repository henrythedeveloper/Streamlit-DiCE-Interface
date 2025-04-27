# File: src/ui/__init__.py
"""
Streamlit UI Components Package
-------------------------------
Exports the necessary UI rendering and handling functions
for the counterfactual explainer application, making them
available for import from the 'src.ui' namespace.
"""

# --- Exports ---
from .instance_selector import render_instance_selector_and_constraints
from .explanation_view import render_main_explanation_area, display_dice_results
from .setup_handler import handle_explainer_setup
from .visualization import plot_feature_comparison
from .progress_indicator import show_progress_indicator
