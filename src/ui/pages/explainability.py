"""Model Explainability page: global SHAP summary and feature importance."""

import matplotlib.pyplot as plt
import numpy as np
import shap
import streamlit as st

from src.model.global_explain import compute_global_shap
from src.ui.components import section_title


def render_body() -> None:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    section_title("What is SHAP?")
    st.markdown(
        "SHAP shows how much each patient characteristic pushed a prediction "
        "above or below the model's average. See the **About** page for more "
        "detail on how it works."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    shap_values, display_df, _ = compute_global_shap()

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    section_title("SHAP Summary Plot")
    st.caption(
        "Each dot is one patient. Position on the horizontal axis shows whether "
        "that feature increased (right) or decreased (left) the predicted length "
        "of stay for that patient. Colour shows whether the feature's value was "
        "high (red) or low (blue) for that patient."
    )
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, display_df, show=False)
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.close("all")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    section_title("SHAP Feature Importance")
    st.caption(
        "Ranks features by their average absolute impact on predictions across "
        "all sampled patients: the higher the bar, the more that feature "
        "influences the model overall, regardless of direction."
    )
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.barh(np.array(display_df.columns)[order], mean_abs[order], color="#0F6E63")
    ax2.set_xlabel("Mean |SHAP value|")
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    section_title("How to Interpret These Plots")
    st.markdown(
        "Medical condition, Glucose, Stress Level and HbA1c drive most "
        "predictions. These plots describe the model's learned behaviour and "
        "should support, not replace, clinical judgement."
    )
    st.markdown("</div>", unsafe_allow_html=True)
