"""Model Explainability page: global SHAP summary and feature importance."""

import matplotlib.pyplot as plt
import numpy as np
import shap
import streamlit as st

from src.model.global_explain import compute_global_shap
from src.ui.components import card, section_title


def render_body(model_name: str | None = None) -> None:
    st.caption(
        "SHAP explains which patient factors drive predictions overall. "
        "See the **About** page for how it works."
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
            "Each dot is one patient. Position on the horizontal axis shows whether "
            "that feature increased (right) or decreased (left) the predicted length "
            "of stay for that patient. Colour shows whether the feature's value was "
            "high (red) or low (blue) for that patient."
        )
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, display_df, show=False)
        st.pyplot(plt.gcf(), width="stretch")
        plt.close("all")

    with card():
        section_title("SHAP Feature Importance")
        st.caption(
            "Ranks features by their average absolute impact on predictions: the "
            "higher the bar, the more that feature influences the model overall, "
            "regardless of direction."
        )
        mean_abs = np.abs(shap_values).mean(axis=0)
        order = np.argsort(mean_abs)
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.barh(np.array(display_df.columns)[order], mean_abs[order], color="#0F6E63")
        ax2.set_xlabel("Mean |SHAP value|")
        st.pyplot(fig2, width="stretch")
        plt.close(fig2)
