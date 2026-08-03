"""Reusable UI building blocks shared across pages."""

import streamlit as st


def page_header(title: str, subtitle: str = "", icon: str = "") -> None:
    icon_html = f"{icon} " if icon else ""
    st.markdown(f"## {icon_html}{title}")
    if subtitle:
        st.markdown(
            f"<p style='color: var(--clr-text-muted); margin-top: -0.5rem;'>{subtitle}</p>",
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)


def section_title(text: str) -> None:
    st.markdown(f"<div class='section-title'>{text}</div>", unsafe_allow_html=True)


def metric_tile(label: str, value: str, sub: str = "") -> str:
    sub_html = f"<div class='metric-sub'>{sub}</div>" if sub else ""
    return (
        "<div class='metric-tile'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"{sub_html}"
        "</div>"
    )


def render_metric_row(metrics: list[dict]) -> None:
    """metrics: list of {"label", "value", "sub"} dicts, rendered in equal columns."""
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.markdown(
                metric_tile(m["label"], m["value"], m.get("sub", "")),
                unsafe_allow_html=True,
            )


def nav_card(icon: str, title: str, description: str) -> str:
    return (
        "<div class='nav-card'>"
        f"<div class='nav-icon'>{icon}</div>"
        f"<h4>{title}</h4>"
        f"<p>{description}</p>"
        "</div>"
    )


def start_card() -> None:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)


def end_card() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def pill(text: str) -> str:
    return f"<span class='pill'>{text}</span>"
