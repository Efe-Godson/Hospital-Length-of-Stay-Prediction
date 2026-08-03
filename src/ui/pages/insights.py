"""Model Insights page: Model Performance and Explainability combined via tabs."""

import streamlit as st

from src.ui.components import page_header
from src.ui.icons import icon as get_icon
from src.ui.pages import explainability, performance


def render() -> None:
    page_header(
        "Model Insights",
        "Performance comparison and explainability for the evaluated models.",
        icon=get_icon("chart-bar", 26),
    )

    tab_performance, tab_explainability = st.tabs(["Model Performance", "Explainability"])

    with tab_performance:
        performance.render_body()

    with tab_explainability:
        explainability.render_body()
