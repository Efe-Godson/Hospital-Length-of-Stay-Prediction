"""Entry point: page config, navigation and page dispatch."""

import streamlit as st

from src.ui import theme
from src.ui.icons import icon
from src.ui.pages import about, explainability, home, performance, prediction

st.set_page_config(
    page_title="Hospital Length of Stay Predictor",
    page_icon=":material/local_hospital:",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme.inject_theme()

PAGES = {
    "Home": home,
    "Prediction": prediction,
    "Model Performance": performance,
    "Explainability": explainability,
    "About": about,
}

with st.sidebar:
    st.markdown(
        "<div style='padding: 0.5rem 0 1rem 0;'>"
        "<span style='font-size:1.3rem; font-weight:700; color: var(--clr-primary); "
        "display:inline-flex; align-items:center; gap:0.4rem;'>"
        f"{icon('hospital', 22)} LOS Predictor</span>"
        "<div style='color: var(--clr-text-muted); font-size:0.85rem;'>Clinical Decision Support</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    selection = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Random Forest regression model · SHAP explainability")

PAGES[selection].render()
