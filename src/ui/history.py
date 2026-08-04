"""Sidebar prediction-history panel, shared across all pages via app.py."""

import streamlit as st

from src.utils.formatting import format_days


def render_sidebar_history() -> None:
    """Fills the sidebar's remaining space with a running log of predictions made this session."""
    st.markdown(
        "<div style='color: var(--clr-text-muted); font-size:0.78rem; font-weight:600; "
        "text-transform:uppercase; letter-spacing:0.05em; margin: 1.5rem 0 0.5rem;'>"
        "Prediction History</div>",
        unsafe_allow_html=True,
    )

    history = st.session_state.get("prediction_history")
    if not history:
        st.caption("No predictions yet this session. Results will appear here.")
        return

    for entry in history:
        st.markdown(
            f"<div style='padding: 0.5rem 0; border-bottom: 1px solid rgba(15, 110, 99, 0.15);'>"
            f"<div style='font-size:0.85rem;'>{entry['gender']}, {entry['age']:.0f} &middot; "
            f"{entry['medical_condition']}</div>"
            f"<div style='display:flex; justify-content:space-between; align-items:center; margin-top:0.15rem;'>"
            f"<span class='pill' style='font-size:0.7rem; padding:0.15rem 0.5rem;'>{entry['model_name']}</span>"
            f"<span style='font-weight:700; color:var(--clr-primary);'>{format_days(entry['prediction'])}</span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
