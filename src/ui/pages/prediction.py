"""Prediction page: patient data entry, and a popup with the prediction result."""

import plotly.graph_objects as go
import streamlit as st

from src import config
from src.model.loader import load_deployment_package
from src.model.predictor import (
    PatientInput,
    all_contributing_features,
    explain,
    predict,
    top_contributing_features,
)
from src.ui.components import card, page_header, section_title
from src.ui.icons import icon as get_icon
from src.utils.formatting import format_days, format_signed_days

_SELECT_PLACEHOLDER = "— Select —"

# Maps a clinical-measurement feature name to the PatientInput attribute
# holding its raw (unscaled) value, for the Clinical Observations panel.
_FEATURE_TO_PATIENT_ATTR = {
    "Glucose": "glucose",
    "Blood Pressure": "blood_pressure",
    "BMI": "bmi",
    "Oxygen Saturation": "oxygen_saturation",
    "Cholesterol": "cholesterol",
    "Triglycerides": "triglycerides",
    "HbA1c": "hba1c",
}


def _entry_form(package: dict) -> tuple[str, PatientInput | None]:
    with card():
        section_title("Demographics")
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age (years)", min_value=0, max_value=120, value=0, step=1)
        gender = c2.selectbox("Gender", [_SELECT_PLACEHOLDER] + package["gender_options"])
        medical_condition = c3.selectbox(
            "Medical Condition", [_SELECT_PLACEHOLDER] + package["medical_conditions"]
        )

    with card():
        section_title("Clinical Measurements")
        c1, c2, c3 = st.columns(3)
        glucose = c1.number_input("Glucose (mg/dL)", min_value=0.0, max_value=500.0, value=0.0, step=1.0)
        blood_pressure = c2.number_input("Blood Pressure (mmHg)", min_value=0.0, max_value=300.0, value=0.0, step=1.0)
        bmi = c3.number_input("BMI", min_value=0.0, max_value=80.0, value=0.0, step=0.1)

        c1, c2, c3 = st.columns(3)
        oxygen_saturation = c1.number_input("Oxygen Saturation (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        cholesterol = c2.number_input("Cholesterol (mg/dL)", min_value=0.0, max_value=600.0, value=0.0, step=1.0)
        triglycerides = c3.number_input("Triglycerides (mg/dL)", min_value=0.0, max_value=800.0, value=0.0, step=1.0)

        c1, _, _ = st.columns(3)
        hba1c = c1.number_input("HbA1c (%)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)

    with card():
        section_title("Lifestyle Factors")
        c1, c2, c3 = st.columns(3)
        smoking = c1.toggle("Smoker", value=False)
        alcohol = c2.toggle("Alcohol Use", value=False)
        family_history = c3.toggle("Family History of Illness", value=False)

        c1, c2 = st.columns(2)
        physical_activity = c1.slider("Physical Activity (hrs/week)", -5.0, 15.0, 0.0, 0.1)
        diet_score = c2.slider("Diet Score (0-12)", -3.0, 13.0, 0.0, 0.1)

        c1, c2 = st.columns(2)
        stress_level = c1.slider("Stress Level (0-16)", -3.0, 16.0, 0.0, 0.1)
        sleep_hours = c2.slider("Sleep Hours (per night)", 0.0, 12.0, 0.0, 0.1)

    model_names = list(package.get("models", {}).keys()) or [package.get("default_model_name", "Random Forest")]
    default_name = package.get("default_model_name", model_names[0])
    c1, c2 = st.columns(2)
    with c1:
        model_name = st.selectbox(
            "Model",
            model_names,
            index=model_names.index(default_name),
            help="Random Forest is the model recommended in this project's evaluation, "
            "but any of the five compared models can be used to generate a prediction.",
        )
    with c2:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        submitted = st.button("Predict Length of Stay", type="primary", width="stretch")

    if not submitted:
        return model_name, None

    if gender == _SELECT_PLACEHOLDER or medical_condition == _SELECT_PLACEHOLDER:
        st.error("Please select a Gender and Medical Condition before predicting.")
        return model_name, None

    return model_name, PatientInput(
        age=age,
        gender=gender,
        medical_condition=medical_condition,
        glucose=glucose,
        blood_pressure=blood_pressure,
        bmi=bmi,
        oxygen_saturation=oxygen_saturation,
        cholesterol=cholesterol,
        triglycerides=triglycerides,
        hba1c=hba1c,
        smoking=smoking,
        alcohol=alcohol,
        physical_activity=physical_activity,
        diet_score=diet_score,
        family_history=family_history,
        stress_level=stress_level,
        sleep_hours=sleep_hours,
    )


def _describe(name: str) -> str:
    return config.FEATURE_PHRASES.get(name, name)


def _bare(name: str) -> str:
    """Strip a leading possessive/verb from a feature phrase, for use mid-list."""
    phrase = _describe(name)
    for prefix in ("not having ", "having ", "their ", "a ", "being "):
        if phrase.startswith(prefix):
            return phrase[len(prefix):]
    return phrase


def _join_naturally(phrases: list[str]) -> str:
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f" and {phrases[-1]}"


def _narrative_summary(contributors: dict, prediction: float) -> str:
    """A short flowing paragraph in the style of a clinical report summary."""
    increasing = [name for name, _ in contributors["increasing"]]
    decreasing = [name for name, _ in contributors["decreasing"]]
    days = format_days(prediction)

    if not increasing and not decreasing:
        return f"The model predicts an estimated hospital stay of **{days}**, close to its typical prediction."

    clauses = []
    if increasing:
        lead = _describe(increasing[0])
        clauses.append(f"{lead[0].upper()}{lead[1:]} increased the expected hospital stay")
    if decreasing:
        bare = _join_naturally([_bare(name) for name in decreasing])
        verb = "was" if len(decreasing) == 1 else "were"
        clause = f"{bare} {verb} associated with a shorter recovery period"
        clauses.append(clause if not clauses else f"while {clause}")

    body = clauses[0] if len(clauses) == 1 else f"{clauses[0]}, {clauses[1]}"
    return f"{body}, resulting in an estimated hospital stay of **{days}**."


def _clinical_observations(explanation, patient: PatientInput) -> list[dict]:
    """Reference-range status for the clinical measurements that most influenced this prediction."""
    values, names = explanation.values, explanation.feature_names
    ranked = sorted(
        (
            (name, abs(value))
            for name, value in zip(names, values)
            if name in config.CLINICAL_REFERENCE_RANGES
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    observations = []
    for name, _ in ranked[:4]:
        attr = _FEATURE_TO_PATIENT_ATTR[name]
        value = getattr(patient, attr)
        range_info = config.CLINICAL_REFERENCE_RANGES[name]

        if value < range_info["low"]:
            status, normal = "Low", False
        elif value > range_info["high"]:
            status, normal = "Elevated", False
        else:
            status, normal = "Normal", True

        unit = f" {range_info['unit']}" if range_info["unit"] else ""
        observations.append(
            {
                "label": range_info["label"],
                "status": status,
                "value_str": f"{value:g}{unit}",
                "normal": normal,
            }
        )
    return observations


def _contribution_chart(items: list[tuple[str, float]]) -> go.Figure:
    items = sorted(items, key=lambda item: item[1])
    names = [config.FEATURE_PHRASES.get(name, name).capitalize() for name, _ in items]
    values = [value for _, value in items]
    colors = [config.COLORS["danger"] if v > 0 else config.COLORS["accent"] for v in values]

    fig = go.Figure(
        go.Bar(x=values, y=names, orientation="h", marker_color=colors)
    )
    fig.update_layout(
        xaxis_title="Impact on predicted length of stay (days)",
        yaxis_title="",
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color=config.COLORS["text"]),
    )
    return fig


@st.dialog("Prediction Result", width="large")
def _show_result_dialog(state: dict) -> None:
    prediction = state["prediction"]
    explanation = state["explanation"]
    patient = state["patient"]
    baseline = float(explanation.base_values)

    st.markdown(
        f"<div style='text-align:center; margin: 0 0 1.25rem;'>"
        f"<span class='pill'>{state.get('model_name', 'Random Forest')}</span>"
        f"<div style='font-size:0.85rem; color:var(--clr-text-muted); text-transform:uppercase; "
        f"letter-spacing:0.05em; margin-top:0.75rem;'>Predicted Length of Stay</div>"
        f"<div style='font-size:3.4rem; font-weight:800; color:var(--clr-primary); line-height:1.15;'>"
        f"{format_days(prediction)}</div>"
        f"<div style='font-size:0.85rem; color:var(--clr-text-muted);'>"
        f"Model average: {format_days(baseline)} &middot; "
        f"{format_signed_days(prediction - baseline)} from average</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    contributors = top_contributing_features(explanation, n=3)

    section_title("Prediction Summary")
    st.markdown(_narrative_summary(contributors, prediction))

    observations = _clinical_observations(explanation, patient)
    if observations:
        section_title("Clinical Observations")
        for obs in observations:
            icon_name = "check" if obs["normal"] else "alert-triangle"
            color = config.COLORS["success"] if obs["normal"] else config.COLORS["warning"]
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:0.6rem; padding:0.3rem 0;'>"
                f"<span style='color:{color}; display:inline-flex;'>{get_icon(icon_name, 18)}</span>"
                f"<span><strong>{obs['label']}:</strong> {obs['status']} ({obs['value_str']})</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.caption(
        "The explanation below describes how this prediction was made. "
        "It should support, not replace, clinical judgement."
    )

    with st.expander("Patient Factors Contributing to the Prediction"):
        items = all_contributing_features(explanation, n=8)
        st.plotly_chart(_contribution_chart(items), width="stretch")
        st.caption("Red bars increased the predicted stay; blue bars decreased it.")


def render() -> None:
    page_header(
        "Length of Stay Prediction",
        "Enter patient information to generate an individual prediction.",
        icon=get_icon("stethoscope", 26),
    )

    package = load_deployment_package()

    model_name, patient = _entry_form(package)

    if patient is not None:
        prediction, scaled_row = predict(patient, package, model_name)
        explanation = explain(scaled_row, package, model_name)
        st.session_state["last_prediction"] = {
            "patient": patient,
            "prediction": prediction,
            "explanation": explanation,
            "model_name": model_name,
        }
        _show_result_dialog(st.session_state["last_prediction"])
