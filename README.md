# Implementing Counterfactual Explanations for Model Interpretability using DiCE

## Project Description

This project focuses on developing and evaluating a system for generating diverse counterfactual explanations for machine learning models to enhance model interpretability. Counterfactual explanations illustrate the minimal changes needed in input features to achieve a different prediction outcome, providing intuitive insights into model decisions. This system utilizes the DiCE (Diverse Counterfactual Explanations) framework integrated into a user-friendly Streamlit interface, allowing users to explore how changes in input features affect model predictions for pre-trained LightGBM models. The goal is to provide an effective tool for exploring model behavior and enhancing trust in automated decision-making, particularly for complex "black-box" models.

**Keywords:** Counterfactual Explanations, Explainable AI (XAI), Machine Learning Interpretability, DiCE, LightGBM, Model Trust ]

## Key Features

* **Counterfactual Generation:** Implements the DiCE framework (using the genetic algorithm) to generate diverse counterfactual explanations for classification models.
* **Model Support:** Demonstrates explanations for LightGBM models trained on various datasets (Heart Failure Prediction, Phishing URL Detection, Botnet Attack Detection)
* **Interactive Interface:** A Streamlit application (`app.py`) allows users to:
    * Select pre-configured scenarios (Dataset + Model).
    * Filter and select specific instances for explanation.
    * Specify immutable features (features to keep constant).
    * Generate and view counterfactual examples.
    * Visualize feature comparisons between the original instance and counterfactuals.
* **Explanation Quality Evaluation:** Calculates and displays metrics (Sparsity, Proximity, Diversity, Validity) to assess the quality of generated counterfactuals.
* **Preprocessing Integration:** Includes a custom model wrapper (`src/model_wrapper.py`) to handle necessary data preprocessing (ordinal encoding, standard scaling) consistently within the DiCE workflow, using artifacts saved during training.


## Installation

1.  **Clone the repository:**
    ```bash
    git clone [Your Repository URL]
    cd [Your Repository Name]
    ```
2.  **Create and activate a virtual environment:**

    * **Using Conda:**
        ```bash
        conda create --name counterfactual-env python=3.10  # Or choose your preferred Python 3.x version
        conda activate counterfactual-env
        ```
    * **Using `venv`:**
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows use `venv\Scripts\activate`
        ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(See `requirements.txt` for the list of packages)*

## Training the Models (Optional)

The pre-trained model artifacts (`.joblib` files containing the model, encoder, and scaler) are expected in the `saved_artifacts/` directory. If these are not present, you can train the models using the provided scripts. Ensure the datasets are in the `data/` directory.

Run the following commands from the root directory of the project:

* **Heart Failure:**
    ```bash
    python -m model_training.run_training \
        --data_path data/heart_failure_prediction.csv \
        --target_column HeartDisease \
        --problem_type binary \
        --output_dir saved_artifacts/heart_failure
    ```
* **Cybersecurity (Phishing URL):**
    ```bash
    python -m model_training.run_training \
        --data_path data/dataset_cybersecurity_michelle.csv \
        --target_column phishing \
        --problem_type binary \
        --output_dir saved_artifacts/cybersecurity
    ```
* **Botnet Attack:**
    ```bash
    python -m model_training.run_training \
        --data_path data/botnet_dataset.csv \
        --target_column class \
        --problem_type binary \
        --output_dir saved_artifacts/botnet
    ```

## Usage

1.  **Ensure Datasets and Models are Available:**
    * Place the required datasets (`.csv` files) in the `data/` directory (or update paths in `src/config.py`).
    * Ensure the trained model artifacts (`model.joblib`) are in the corresponding subdirectories within `saved_artifacts/` as specified in `src/config.py`. Run the training commands above if needed.
2.  **Activate your virtual environment** (if not already active):
    * `conda activate counterfactual-env` OR `source venv/bin/activate`
3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
4.  **Interact with the application:**
    * Select a scenario (e.g., "Heart Failure Prediction") from the sidebar.
    * Filter instances by the model's prediction (Class 0 or 1).
    * Select a specific instance index from the dropdown.
    * Optionally, choose features to keep constant (immutable).
    * Click "Generate Counterfactuals".
    * Explore the results, including the table of counterfactuals (changes highlighted), feature comparison plots, and quality metrics.

## References & Acknowledgements

* This project builds upon the concepts and framework presented in the following key papers:
    * Wachter, S., Mittelstadt, B. and Russell, C. (2017), 'Counterfactual explanations without opening the black box: Automated decisions and the GDPR', Harv. J. Law Technol. 31(2), 841-887.
    * Mothilal, R. K. R., Sharma, A. and Tan, C. (2020), Explaining machine learning classifiers through diverse counterfactual explanations, in 'Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency, FAT* '20', pp. 607-617.
    * Guidotti, R. (2024), 'Counterfactual explanations and how to find them: literature review and benchmarking', Data Min. Knowl. Discov. 38(6), 2770-2824.
* The DiCE (Diverse Counterfactual Explanations) library (`dice-ml`) is central to the explanation generation.