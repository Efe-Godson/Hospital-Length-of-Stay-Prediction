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
                This application estimates how many days a patient is likely to remain
                hospitalised, based on demographic information, clinical measurements
                and lifestyle factors recorded at admission. It uses a Random Forest
                regression model, trained and validated on 30,000 patient records,
                selected after comparing five candidate algorithms on prediction
                accuracy and error.
            </p>
            <p style='color: var(--clr-text-muted); font-size: 0.98rem; line-height: 1.6;'>
                Reliable length-of-stay estimates support bed and staffing
                planning, discharge coordination and early identification of
                patients who may require extended care &mdash; helping hospitals
                allocate resources more effectively.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("#### Explore the application")

    cards = [
        ("stethoscope", "Prediction", "Enter patient details and generate an individual length-of-stay estimate with explanation."),
        ("chart-bar", "Model Performance", "Compare the five regression models evaluated and see why Random Forest was selected."),
        ("search", "Explainability", "Understand which patient factors most influence the model's predictions using SHAP."),
        ("info", "About", "Read about the project workflow, dataset and technologies used."),
    ]

    cols = st.columns(4)
    for col, (icon_name, title, desc) in zip(cols, cards):
        with col:
            st.markdown(nav_card(get_icon(icon_name, 24), title, desc), unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.info(
        "Use the navigation menu in the sidebar to move between pages. "
        "Start with **Prediction** to generate a length-of-stay estimate.",
        icon=":material/arrow_back:",
    )
