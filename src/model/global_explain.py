"""Global SHAP explanation, computed on a fixed sample of the training-encoded dataset.

Mirrors the notebook's explainability section exactly:
    X_test_sample = X_test_scaled.sample(n=100, random_state=42)
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_sample)

Here the sample is reconstructed from the saved, already-encoded
`model_dataset.csv` (same rows/columns the notebook produced before scaling)
using the same feature order and random_state, then scaled with the fitted
scaler from the deployment package before computing SHAP values.
"""

import pandas as pd
import shap
import streamlit as st

from src import config
from src.model.loader import load_deployment_package, load_model_dataset


@st.cache_resource(show_spinner="Computing SHAP explanations...")
def compute_global_shap():
    package = load_deployment_package()
    dataset = load_model_dataset()

    feature_names = package["feature_names"]
    scaler = package["scaler"]
    model = package["model"]

    features = dataset[feature_names]

    sample = features.sample(
        n=min(config.SHAP_SAMPLE_SIZE, len(features)),
        random_state=config.RANDOM_STATE,
    )

    scaled_sample = scaler.transform(sample)
    scaled_df = pd.DataFrame(scaled_sample, columns=feature_names, index=sample.index)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled_df)

    display_df = scaled_df.rename(columns=config.DISPLAY_NAME_OVERRIDES)

    return shap_values, display_df, explainer.expected_value
