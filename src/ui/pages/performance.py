"""Model Performance page: comparison table, metrics and charts."""

import plotly.graph_objects as go
import streamlit as st

from src import config
from src.model.loader import load_model_results
from src.ui.components import page_header, render_metric_row, section_title


def _bar_chart(df, metric: str, title: str, lower_is_better: bool = True) -> go.Figure:
    sorted_df = df.sort_values(metric, ascending=lower_is_better)
    colors = [
        config.COLORS["primary"] if model == "Random Forest" else config.COLORS["border"]
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


def render() -> None:
    page_header(
        "Model Performance",
        "Comparison of regression models evaluated during development.",
        icon="📊",
    )

    results = load_model_results()
    best = results.sort_values("RMSE").iloc[0]

    render_metric_row(
        [
            {"label": "Selected Model", "value": best["Model"]},
            {"label": "MAE", "value": f"{best['MAE']:.2f}"},
            {"label": "RMSE", "value": f"{best['RMSE']:.2f}"},
            {"label": "R²", "value": f"{best['R²']:.2f}"},
        ]
    )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    section_title("Model Comparison Table")
    st.dataframe(
        results.sort_values("RMSE").reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    section_title("Prediction Error Comparison")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_bar_chart(results, "RMSE", "RMSE by Model (lower is better)"), use_container_width=True)
    with c2:
        st.plotly_chart(_bar_chart(results, "MAE", "MAE by Model (lower is better)"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    section_title("Training Time Comparison")
    st.plotly_chart(
        _bar_chart(results, "Training Time (s)", "Training Time by Model (seconds)"),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    section_title("Why Random Forest Was Selected")
    st.markdown(
        """
        Five regression models were trained and evaluated on the same held-out
        test set: **Linear Regression**, **Decision Tree**, **Random Forest**,
        **Gradient Boosting** and **XGBoost**.

        **Random Forest** achieved the lowest RMSE and one of the lowest MAE
        values among all candidates, alongside the highest R² score,
        indicating both the smallest average prediction error and the
        strongest ability to explain variation in hospital length of stay.
        Although it required the longest training time, this was a one-time
        cost during development and does not affect prediction speed at
        inference. Its measurable improvement in predictive accuracy over the
        other models justified its selection as the final deployed model.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)
