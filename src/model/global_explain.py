"""Global SHAP explanation, computed on a sample of the training-encoded dataset.

Mirrors the notebook's explainability section (shap.TreeExplainer on a sample
of X_test_scaled), extended to work for any of the five deployed models, not
just the default Random Forest, using the same tree/non-tree branching as
src.model.predictor.explain(). The sample is reconstructed from the saved,
already-encoded `model_dataset.csv` (same rows/columns the notebook produced
before scaling), then scaled with the fitted scaler from the deployment
package before computing SHAP values.
"""

import numpy as np
import pandas as pd
import shap
import streamlit as st

from src import config
from src.model.loader import load_deployment_package, load_model_dataset
from src.model.predictor import TREE_MODEL_TYPES, background_sample, resolve_model


@st.cache_resource(show_spinner="Computing SHAP explanations...")
def compute_global_shap(model_name: str | None = None):
    """Return (shap_values, display_df, base_value) for the chosen model.

    Cached per model_name, so switching models recomputes and caches
    independently instead of reusing a stale explanation.
    """
    package = load_deployment_package()
    dataset = load_model_dataset()
    model = resolve_model(package, model_name)

    feature_names = package["feature_names"]
    scaler = package["scaler"]

    features = dataset[feature_names]
    sample = features.sample(
        n=min(config.SHAP_SAMPLE_SIZE, len(features)),
        random_state=config.RANDOM_STATE,
    )
    scaled_sample = scaler.transform(sample)
    scaled_df = pd.DataFrame(scaled_sample, columns=feature_names, index=sample.index)
    display_df = scaled_df.rename(columns=config.DISPLAY_NAME_OVERRIDES)

    if type(model).__name__ in TREE_MODEL_TYPES:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(scaled_df)
        base_value = np.asarray(explainer.expected_value).reshape(-1)[0]
    else:
        explainer = shap.Explainer(model.predict, background_sample(package))
        result = explainer(scaled_df)
        shap_values = result.values
        base_value = np.asarray(result.base_values).reshape(-1).mean()

    return shap_values, display_df, base_value
