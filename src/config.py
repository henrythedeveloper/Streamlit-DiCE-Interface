# File: src/config.py
"""
Configuration Module: Counterfactual Explainer Application
----------------------------------------------------------
Stores application-wide settings, DiCE parameters, and definitions
for pre-defined project scenarios (data paths, model paths, feature types).
"""

# --- Imports ---
import os

# --- Counterfactual Generation Parameters ---
NUM_COUNTERFACTUALS = 4     # Default number of counterfactuals to display
DICE_METHOD = "genetic"   # DiCE method for generating counterfactuals

# --- Application Settings ---
RANDOM_STATE = 42           # Seed for reproducible results where applicable

# --- Pre-defined Project Scenarios ---
# Defines the datasets and models available in the application.
# Paths should be relative to the project root directory (where app.py is located).
PROJECT_SCENARIOS = {
    "Heart Failure": {
        "display_name": "Heart Failure Prediction",
        "data_path": "data/heart_failure_prediction.csv",
        "model_artifact_path": "saved_artifacts/heart_failure/model.joblib",
        "target_column": "HeartDisease",
        "numerical_features": ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"],
        "categorical_features": ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope"],
        "columns_to_drop_on_load": []
    },
    "Cybersecurity": {
        "display_name": "Phishing URL Detection",
        "data_path": "data/dataset_cybersecurity_michelle.csv",
        "model_artifact_path": "saved_artifacts/cybersecurity/model.joblib",
        "target_column": "phishing",
        # Feature types derived from data analysis during model training
        "numerical_features": [
            'qty_dot_url', 'qty_hyphen_url', 'qty_underline_url', 'qty_slash_url',
            'qty_equal_url', 'qty_and_url', 'qty_asterisk_url', 'qty_percent_url',
            'length_url', 'qty_dot_domain', 'qty_vowels_domain', 'domain_length',
            'qty_hyphen_directory', 'qty_underline_directory', 'qty_slash_directory',
            'qty_percent_directory', 'directory_length', 'qty_hyphen_file',
            'qty_underline_file', 'qty_percent_file', 'file_length', 'qty_dot_params',
            'qty_hyphen_params', 'qty_underline_params', 'qty_slash_params',
            'qty_equal_params', 'qty_and_params', 'qty_percent_params',
            'params_length', 'qty_params', 'time_response', 'asn_ip',
            'time_domain_activation', 'time_domain_expiration', 'qty_ip_resolved',
            'qty_nameservers', 'qty_mx_servers', 'ttl_hostname'
        ],
        "categorical_features": [
            'qty_questionmark_url', 'qty_at_url', 'qty_exclamation_url', 'qty_space_url',
            'qty_tilde_url', 'qty_comma_url', 'qty_plus_url', 'qty_hashtag_url',
            'qty_dollar_url', 'qty_tld_url', 'qty_hyphen_domain', 'qty_underline_domain',
            'qty_slash_domain', 'qty_questionmark_domain', 'qty_equal_domain',
            'qty_at_domain', 'qty_and_domain', 'qty_exclamation_domain',
            'qty_space_domain', 'qty_tilde_domain', 'qty_comma_domain',
            'qty_plus_domain', 'qty_asterisk_domain', 'qty_hashtag_domain',
            'qty_dollar_domain', 'qty_percent_domain', 'domain_in_ip',
            'server_client_domain', 'qty_dot_directory', 'qty_questionmark_directory',
            'qty_equal_directory', 'qty_at_directory', 'qty_and_directory',
            'qty_exclamation_directory', 'qty_space_directory', 'qty_tilde_directory',
            'qty_comma_directory', 'qty_plus_directory', 'qty_asterisk_directory',
            'qty_hashtag_directory', 'qty_dollar_directory', 'qty_dot_file',
            'qty_slash_file', 'qty_questionmark_file', 'qty_equal_file',
            'qty_at_file', 'qty_and_file', 'qty_exclamation_file', 'qty_space_file',
            'qty_tilde_file', 'qty_comma_file', 'qty_plus_file', 'qty_asterisk_file',
            'qty_hashtag_file', 'qty_dollar_file', 'qty_questionmark_params',
            'qty_at_params', 'qty_exclamation_params', 'qty_space_params',
            'qty_tilde_params', 'qty_comma_params', 'qty_plus_params',
            'qty_asterisk_params', 'qty_hashtag_params', 'qty_dollar_params',
            'tld_present_params', 'email_in_url', 'domain_spf',
            'tls_ssl_certificate', 'qty_redirects', 'url_google_index',
            'domain_google_index', 'url_shortened'
        ],
        "columns_to_drop_on_load": []
    },
    "Botnet": {
        "display_name": "Botnet Attack Detection",
        "data_path": "data/botnet_dataset.csv",
        "model_artifact_path": "saved_artifacts/botnet/model.joblib",
        "target_column": "class",
        "numerical_features": [
            "Duration", "AvgDuration", "PBS", "AvgPBS", "TBS", "PBR",
            "AvgPBR", "TBR", "Missed_Bytes", "Packets_Sent",
            "Packets_Received", "SRPR"
        ],
        "categorical_features": ["Transport_Protocol"],
        "columns_to_drop_on_load": []
    }
}

# --- Helper Function ---
def get_scenario_config(scenario_key):
    """
    Safely retrieves the configuration dictionary for a given scenario key.

    Args:
        scenario_key (str): The key identifying the desired scenario
                            (e.g., "Heart Failure").

    Returns:
        dict or None: The configuration dictionary if the key exists, otherwise None.
    """
    return PROJECT_SCENARIOS.get(scenario_key)