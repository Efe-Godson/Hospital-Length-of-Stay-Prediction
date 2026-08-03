"""Model Performance page: comparison table, metrics and charts."""

import plotly.graph_objects as go
import streamlit as st

from src import config
from src.model.loader import load_model_results
from src.ui.components import render_metric_row, section_title


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


def render_body() -> None:
    results = load_model_results()
    results_sorted = results.sort_values("RMSE").reset_index(drop=True)
    model_names = results_sorted["Model"].tolist()
    best_name = model_names[0]

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    section_title("Model")
    selected_name = st.selectbox(
        "View metrics for",
        model_names,
        index=model_names.index(best_name),
        help="Random Forest is the model deployed for prediction, but you can "
        "view how any of the five evaluated models performed.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    selected = results_sorted[results_sorted["Model"] == selected_name].iloc[0]

    render_metric_row(
        [
            {"label": "Selected Model", "value": selected["Model"]},
            {"label": "MAE", "value": f"{selected['MAE']:.2f}"},
            {"label": "RMSE", "value": f"{selected['RMSE']:.2f}"},
            {"label": "R²", "value": f"{selected['R²']:.2f}"},
        ]
    )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    section_title("Model Comparison Table")
    st.dataframe(
        results_sorted,
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    section_title("Prediction Error Comparison")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            _bar_chart(results, "RMSE", "RMSE by Model (lower is better)", selected_name),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            _bar_chart(results, "MAE", "MAE by Model (lower is better)", selected_name),
            use_container_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    section_title("Training Time Comparison")
    st.plotly_chart(
        _bar_chart(results, "Training Time (s)", "Training Time by Model (seconds)", selected_name),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    section_title("Why Random Forest Was Selected")
    st.markdown(
        "**Random Forest** achieved the lowest RMSE and the highest R² of the "
        "five models compared, so it was selected as the deployed model. See "
        "the **About** page for the full model development workflow."
    )
    st.markdown("</div>", unsafe_allow_html=True)
