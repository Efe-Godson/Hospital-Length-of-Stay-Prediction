"""Model Explainability page: global SHAP summary and feature importance."""

import matplotlib.pyplot as plt
import numpy as np
import shap
import streamlit as st

from src.model.global_explain import compute_global_shap
from src.ui.components import page_header, section_title
from src.ui.icons import icon as get_icon


def render() -> None:
    page_header(
        "Model Explainability",
        "Understanding which patient factors drive the model's predictions.",
        icon=get_icon("search", 26),
    )

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    section_title("What is SHAP?")
    st.markdown(
        """
        SHAP (SHapley Additive exPlanations) is a method for explaining the
        output of a machine learning model. For every prediction, SHAP
        calculates how much each input feature pushed the prediction higher
        or lower compared to the model's average prediction. Adding up all
        of a patient's feature contributions, starting from that average,
        reconstructs the model's final predicted length of stay.

        This makes it possible to see **which patient characteristics matter
        most overall**, and **why the model made a specific prediction for an
        individual patient** (see the Prediction page for a patient-level
        example).
        """
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
        """
        - **Medical condition** (Cancer, Diabetes, Hypertension, Healthy) tends
          to have the largest influence on predicted length of stay, reflecting
          the importance of the underlying diagnosis.
        - **Glucose**, **Stress Level**, **HbA1c**, **Blood Pressure** and
          **Sleep Hours** are the strongest continuous clinical and lifestyle
          contributors.
        - Features near the bottom of the importance chart, such as
          **Smoking**, **Alcohol** and **Gender**, have comparatively small
          effects on individual predictions.
        - These plots describe the model's learned behaviour on this dataset;
          they should support, not replace, clinical judgement.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)
