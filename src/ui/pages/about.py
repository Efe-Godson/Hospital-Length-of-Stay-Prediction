"""About page: project overview, workflow, technologies, dataset, author."""

import streamlit as st

from src.model.loader import load_raw_dataset
from src.ui.components import page_header, render_metric_row, section_title
from src.ui.icons import icon as get_icon


def render() -> None:
    page_header("About This Project", icon=get_icon("info", 26))

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    section_title("Project Overview")
    st.markdown(
        """
        This application predicts hospital length of stay using a machine
        learning model trained on a healthcare risk-factors dataset combining
        demographic, clinical and lifestyle information for 30,000 patients.
        It was developed as part of an MSc dissertation project exploring how
        predictive modelling and explainable AI can support clinical
        decision-making around hospital resource planning.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    section_title("Machine Learning Workflow")
    st.markdown(
        """
        1. **Data understanding** — exploratory analysis of 30,000 patient
           records and 20 variables, including missing-value and correlation
           analysis.
        2. **Data preparation** — removal of non-informative columns, median
           imputation for missing numerical values, mode imputation for
           missing categorical values, one-hot encoding of categorical
           variables, an 80/20 train-test split and feature standardisation
           with `StandardScaler`.
        3. **Model development** — five regression models were trained and
           evaluated on the same test set: Linear Regression, Decision Tree,
           Random Forest, Gradient Boosting and XGBoost.
        4. **Model selection** — Random Forest was selected based on the
           lowest RMSE and highest R² among the candidates.
        5. **Explainability** — SHAP (TreeExplainer) was used to generate
           global and local explanations of the selected model's predictions.
        6. **Deployment** — the fitted model, scaler and feature metadata
           were packaged for use in this Streamlit application, without any
           retraining or changes to preprocessing.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    section_title("Technologies Used")
    st.markdown(
        """
        - **Python** — core language for analysis and application development
        - **pandas / NumPy** — data manipulation
        - **scikit-learn** — preprocessing, model training and evaluation
        - **XGBoost** — gradient boosting model comparison
        - **SHAP** — model explainability
        - **Streamlit** — interactive web application
        - **Plotly / Matplotlib** — data visualisation
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    section_title("Dataset Summary")
    df = load_raw_dataset()
    render_metric_row(
        [
            {"label": "Patient Records", "value": f"{len(df):,}"},
            {"label": "Original Variables", "value": str(df.shape[1])},
            {"label": "Predictors Used", "value": "22"},
            {"label": "Target Variable", "value": "LengthOfStay"},
        ]
    )
    st.caption(
        "Source: Healthcare Risk Factors Dataset. Two non-informative columns "
        "(random text and synthetic noise) were identified and removed during "
        "preprocessing."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    section_title("Author")
    st.markdown(
        """
        **Author:** _Add your name here_
        **Programme:** _Add MSc programme / institution here_
        **Contact:** _Add contact email here_
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)
