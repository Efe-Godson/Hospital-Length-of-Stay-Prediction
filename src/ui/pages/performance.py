"""Model Performance page: comparison table, metrics and charts."""

import plotly.graph_objects as go
import streamlit as st

from src import config
from src.model.loader import load_model_results
from src.ui.components import card, render_metric_row, section_title


def _bar_chart(df, metric: str, title: str, highlight: str, lower_is_better: bool = True) -> go.Figure:
    sorted_df = df.sort_values(metric, ascending=lower_is_better)
    colors = [
        config.COLORS["primary"] if model == highlight else config.COLORS["border"]
        for model in sorted_df["Model"]
    ]
    fig = go.Figure(
        go.Bar(
            x=sorted_df[metric],
            y=sorted_df["Model"],
            orientation="h",
            marker_color=colors,
            text=sorted_df[metric],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=metric,
        yaxis_title="",
        height=340,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color=config.COLORS["text"]),
    )
    return fig


def render_body(model_name: str | None = None) -> None:
    results = load_model_results()
    results_sorted = results.sort_values("RMSE").reset_index(drop=True)
    model_names = results_sorted["Model"].tolist()
    selected_name = model_name if model_name in model_names else model_names[0]

    selected = results_sorted[results_sorted["Model"] == selected_name].iloc[0]

    render_metric_row(
        [
            {"label": "Selected Model", "value": selected["Model"]},
            {"label": "MAE", "value": f"{selected['MAE']:.2f}"},
            {"label": "RMSE", "value": f"{selected['RMSE']:.2f}"},
            {"label": "R²", "value": f"{selected['R²']:.2f}"},
        ]
    )

    with card():
        section_title("Model Comparison Table")
        st.dataframe(
            results_sorted,
            width="stretch",
            hide_index=True,
        )

    with card():
        section_title("Prediction Error Comparison")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                _bar_chart(results, "RMSE", "RMSE by Model (lower is better)", selected_name),
                width="stretch",
            )
        with c2:
            st.plotly_chart(
                _bar_chart(results, "MAE", "MAE by Model (lower is better)", selected_name),
                width="stretch",
            )

    with card():
        section_title("Training Time Comparison")
        st.plotly_chart(
            _bar_chart(results, "Training Time (s)", "Training Time by Model (seconds)", selected_name),
            width="stretch",
        )

    st.caption("See the **About** page for why Random Forest was selected as the deployed model.")
