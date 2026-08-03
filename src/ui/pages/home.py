"""Home / landing page."""

import streamlit as st

from src.ui.components import nav_card, page_header
from src.ui.icons import icon as get_icon


def render() -> None:
    page_header(
        "Hospital Length of Stay Predictor",
        "A clinical decision-support tool for estimating patient hospitalisation duration.",
        icon=get_icon("hospital", 26),
    )

    st.markdown(
        """
        <div class='app-card'>
            <span class='pill'>Machine Learning · Clinical Decision Support</span>
            <h3 style='margin-top: 0.9rem;'>Predict hospital length of stay from patient data</h3>
            <p style='color: var(--clr-text-muted); font-size: 0.98rem; line-height: 1.6;'>
                Enter a patient's demographic, clinical and lifestyle information
                to estimate how many days they are likely to remain hospitalised.
                See the <strong>About</strong> page for details on the underlying
                model and dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("#### Explore the application")

    cards = [
        ("stethoscope", "Prediction", "Enter patient details and generate an individual length-of-stay estimate with explanation."),
        ("chart-bar", "Model Insights", "Compare the five regression models evaluated and explore SHAP explainability."),
    ]

    cols = st.columns(2)
    for col, (icon_name, title, desc) in zip(cols, cards):
        with col:
            st.markdown(nav_card(get_icon(icon_name, 24), title, desc), unsafe_allow_html=True)
            if st.button("Open", key=f"nav_btn_{title}", use_container_width=True):
                st.session_state["requested_page"] = title
                st.rerun()

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.info(
        "Use the navigation menu in the sidebar, or click a card above, to "
        "move between pages.",
        icon=":material/arrow_back:",
    )
