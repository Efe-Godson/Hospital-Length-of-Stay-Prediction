"""Model Insights page: Model Performance and Explainability combined via tabs.

Both tabs share a single model selector, so switching the selected model
updates the metrics, charts and SHAP explanations together.
"""

import streamlit as st

from src.model.loader import load_deployment_package
from src.ui.components import card, page_header, section_title
from src.ui.icons import icon as get_icon
from src.ui.pages import explainability, performance


def render() -> None:
    page_header(
        "Model Insights",
        "Performance comparison and explainability for the evaluated models.",
        icon=get_icon("chart-bar", 26),
    )

    package = load_deployment_package()
    model_names = list(package.get("models", {}).keys()) or [package.get("default_model_name", "Random Forest")]
    default_name = package.get("default_model_name", model_names[0])

    with card():
        section_title("Model")
        selected_name = st.selectbox(
            "View performance and explainability for",
            model_names,
            index=model_names.index(default_name),
            help="Random Forest is the model deployed for prediction, but you can "
            "view how any of the five evaluated models performed and explore its "
            "SHAP explanations.",
        )

    tab_performance, tab_explainability = st.tabs(["Model Performance", "Explainability"])

    with tab_performance:
        performance.render_body(selected_name)

    with tab_explainability:
        explainability.render_body(selected_name)
