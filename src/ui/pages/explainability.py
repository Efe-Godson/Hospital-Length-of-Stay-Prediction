"""Model Explainability page: global SHAP summary and feature importance."""

import matplotlib.pyplot as plt
import numpy as np
import shap
import streamlit as st

from src.model.global_explain import compute_global_shap
from src.ui.components import card, section_title


def render_body(model_name: str | None = None) -> None:
    st.caption(
        "SHAP is a way of showing which patient details the model leans on most, "
        "and whether each one tends to push predicted stays up or down. See the "
        "**About** page for more detail."
    )

    if not st.session_state.get("shap_tab_loaded"):
        st.info(
            "Computing this analysis takes a little while the first time it's "
            "requested. Once loaded, switching models above updates it automatically."
        )
        if st.button("Load SHAP Explanations", type="primary"):
            st.session_state["shap_tab_loaded"] = True
            st.rerun()
        return

    shap_values, display_df, _ = compute_global_shap(model_name)

    with card():
        section_title("SHAP Summary Plot")
        st.caption(
            "Each dot is one patient. A dot on the right means that detail made the "
            "stay longer for that patient; on the left means it made it shorter. "
            "Red dots are patients with a higher value for that detail, blue dots "
            "are lower (e.g. red on Glucose means a high glucose reading)."
        )
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, display_df, show=False)
        st.pyplot(plt.gcf(), width="stretch")
        plt.close("all")

    with card():
        section_title("SHAP Feature Importance")
        st.caption(
            "This ranks patient details by how much they matter to the model on "
            "average, regardless of whether they push a stay up or down: the "
            "longer the bar, the bigger its overall influence."
        )
        mean_abs = np.abs(shap_values).mean(axis=0)
        order = np.argsort(mean_abs)
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.barh(np.array(display_df.columns)[order], mean_abs[order], color="#0F6E63")
        ax2.set_xlabel("Mean |SHAP value|")
        st.pyplot(fig2, width="stretch")
        plt.close(fig2)
