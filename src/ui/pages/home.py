"""Home page: an executive Hospital Operations Dashboard summarising the training dataset."""

import plotly.graph_objects as go
import streamlit as st

from src import config
from src.model.loader import load_deployment_package, load_raw_dataset
from src.ui.components import card, page_header
from src.ui.icons import icon as get_icon

_CHART_HEIGHT = 320
_BLUE = "#2D6CDF"
_BLUE_LIGHT = "#7FA6EE"


def _chart_layout(fig: go.Figure, xaxis_title: str = "", yaxis_title: str = "") -> go.Figure:
    fig.update_layout(
        height=_CHART_HEIGHT,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        font=dict(family="Inter, sans-serif", color=config.COLORS["text"], size=12),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="rgba(15, 110, 99, 0.08)")
    fig.update_yaxes(gridcolor="rgba(15, 110, 99, 0.08)")
    return fig


def _kpi_card(icon_name: str, label: str, value: str, col) -> None:
    with col:
        with card():
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:0.5rem;'>"
                f"<span style='color:var(--clr-primary); display:inline-flex;'>"
                f"{get_icon(icon_name, 20)}</span>"
                f"<span style='font-size:0.78rem; font-weight:600; color:var(--clr-text-muted); "
                f"text-transform:uppercase; letter-spacing:0.04em;'>{label}</span>"
                f"</div>"
                f"<div style='font-size:1.8rem; font-weight:800; color:var(--clr-text); "
                f"margin-top:0.4rem;'>{value}</div>",
                unsafe_allow_html=True,
            )


def _length_of_stay_histogram(df) -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=df["LengthOfStay"],
            marker_color=_BLUE,
            nbinsx=19,
            hovertemplate="Length of Stay: %{x} days<br>Patients: %{y}<extra></extra>",
        )
    )
    return _chart_layout(fig, "Length of Stay (Days)", "Number of Patients")


def _condition_distribution_chart(df) -> go.Figure:
    counts = df["Medical Condition"].value_counts().sort_values()
    fig = go.Figure(
        go.Bar(
            x=counts.values, y=counts.index, orientation="h",
            marker_color=_BLUE,
            hovertemplate="%{y}: %{x:,} patients<extra></extra>",
        )
    )
    return _chart_layout(fig, "Number of Patients", "")


def _avg_los_by_condition_chart(df) -> go.Figure:
    avg_los = df.groupby("Medical Condition")["LengthOfStay"].mean().sort_values()
    fig = go.Figure(
        go.Bar(
            x=avg_los.values, y=avg_los.index, orientation="h",
            marker_color=_BLUE,
            hovertemplate="%{y}: %{x:.1f} days<extra></extra>",
        )
    )
    return _chart_layout(fig, "Average Length of Stay (Days)", "")


def _grouped_metric_chart(df, columns: list[str]) -> go.Figure:
    values = [df[col].mean() for col in columns]
    fig = go.Figure(
        go.Bar(
            x=columns, y=values,
            marker_color=_BLUE,
            text=[f"{v:.1f}" for v in values],
            textposition="outside",
            hovertemplate="%{x}: %{y:.1f}<extra></extra>",
        )
    )
    return _chart_layout(fig, "", "Average Value")


def _lifestyle_risk_chart(df) -> go.Figure:
    labels = ["Smoking", "Alcohol", "Family History"]
    percentages = [df[col].mean() * 100 for col in labels]
    fig = go.Figure(
        go.Bar(
            x=labels, y=percentages,
            marker_color=_BLUE_LIGHT,
            text=[f"{v:.0f}%" for v in percentages],
            textposition="outside",
            hovertemplate="%{x}: %{y:.0f}%% of patients<extra></extra>",
        )
    )
    return _chart_layout(fig, "", "% of Patients")


def render() -> None:
    page_header(
        "Hospital Operations Dashboard",
        "Executive overview of the patient dataset used to develop the Length of Stay prediction model.",
        icon=get_icon("hospital", 26),
    )

    df = load_raw_dataset()
    package = load_deployment_package()

    kpis = [
        ("users", "Total Patients", f"{len(df):,}"),
        ("calendar", "Avg. Length of Stay", f"{df['LengthOfStay'].mean():.1f} days"),
        ("user", "Average Age", f"{df['Age'].mean():.0f} yrs"),
        ("stethoscope", "Medical Conditions", str(df["Medical Condition"].nunique())),
        ("chart-bar", "Predictor Variables", str(len(package["feature_names"]))),
    ]
    cols = st.columns(5)
    for (icon_name, label, value), col in zip(kpis, cols):
        _kpi_card(icon_name, label, value, col)

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    row1 = st.columns(3)
    with row1[0]:
        with card():
            st.markdown("**Distribution of Length of Stay**")
            st.plotly_chart(_length_of_stay_histogram(df), width="stretch")
    with row1[1]:
        with card():
            st.markdown("**Patients by Medical Condition**")
            st.plotly_chart(_condition_distribution_chart(df), width="stretch")
    with row1[2]:
        with card():
            st.markdown("**Average Length of Stay by Condition**")
            st.plotly_chart(_avg_los_by_condition_chart(df), width="stretch")

    row2 = st.columns(3)
    with row2[0]:
        with card():
            st.markdown("**Average Clinical Measurements**")
            clinical_cols = [
                "Glucose", "Blood Pressure", "BMI", "Cholesterol",
                "Triglycerides", "HbA1c", "Oxygen Saturation",
            ]
            st.plotly_chart(_grouped_metric_chart(df, clinical_cols), width="stretch")
    with row2[1]:
        with card():
            st.markdown("**Lifestyle Risk Profile**")
            st.plotly_chart(_lifestyle_risk_chart(df), width="stretch")
    with row2[2]:
        with card():
            st.markdown("**Wellness Indicators**")
            wellness_cols = ["Physical Activity", "Diet Score", "Stress Level", "Sleep Hours"]
            st.plotly_chart(_grouped_metric_chart(df, wellness_cols), width="stretch")
