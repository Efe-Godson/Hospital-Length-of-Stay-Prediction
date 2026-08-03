"""Prediction page: patient data entry, prediction and local SHAP explanation."""

import matplotlib.pyplot as plt
import shap
import streamlit as st

from src.model.loader import load_deployment_package
from src.model.predictor import PatientInput, explain, predict, top_contributing_features
from src.ui.components import page_header, render_metric_row, section_title
from src.ui.icons import icon as get_icon
from src.utils.formatting import format_days, format_signed, yes_no


def _entry_form(package: dict) -> PatientInput | None:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    section_title("Demographics")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age (years)", min_value=0, max_value=120, value=55, step=1)
    gender = c2.selectbox("Gender", package["gender_options"])
    medical_condition = c3.selectbox("Medical Condition", package["medical_conditions"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    section_title("Clinical Measurements")
    c1, c2, c3 = st.columns(3)
    glucose = c1.number_input("Glucose (mg/dL)", min_value=0.0, max_value=500.0, value=110.0, step=1.0)
    blood_pressure = c2.number_input("Blood Pressure (mmHg)", min_value=0.0, max_value=300.0, value=138.0, step=1.0)
    bmi = c3.number_input("BMI", min_value=5.0, max_value=80.0, value=28.0, step=0.1)

    c1, c2, c3 = st.columns(3)
    oxygen_saturation = c1.number_input("Oxygen Saturation (%)", min_value=0.0, max_value=100.0, value=95.0, step=0.1)
    cholesterol = c2.number_input("Cholesterol (mg/dL)", min_value=0.0, max_value=600.0, value=212.0, step=1.0)
    triglycerides = c3.number_input("Triglycerides (mg/dL)", min_value=0.0, max_value=800.0, value=175.0, step=1.0)

    c1, _, _ = st.columns(3)
    hba1c = c1.number_input("HbA1c (%)", min_value=0.0, max_value=20.0, value=6.0, step=0.1)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    section_title("Lifestyle Factors")
    c1, c2, c3 = st.columns(3)
    smoking = c1.toggle("Smoker", value=False)
    alcohol = c2.toggle("Alcohol Use", value=False)
    family_history = c3.toggle("Family History of Illness", value=False)

    c1, c2 = st.columns(2)
    physical_activity = c1.slider("Physical Activity (hrs/week)", -5.0, 15.0, 3.8, 0.1)
    diet_score = c2.slider("Diet Score (0-12)", -3.0, 13.0, 4.0, 0.1)

    c1, c2 = st.columns(2)
    stress_level = c1.slider("Stress Level (0-16)", -3.0, 16.0, 5.9, 0.1)
    sleep_hours = c2.slider("Sleep Hours (per night)", 0.0, 12.0, 6.2, 0.1)
    st.markdown("</div>", unsafe_allow_html=True)

    submitted = st.button("Predict Length of Stay", type="primary", use_container_width=True)

    if not submitted:
        return None

    return PatientInput(
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


def _render_summary(patient: PatientInput) -> None:
    section_title("Patient Summary")
    rows = [
        ("Age", f"{patient.age:.0f} years"),
        ("Gender", patient.gender),
        ("Medical Condition", patient.medical_condition),
        ("Glucose", f"{patient.glucose:.1f} mg/dL"),
        ("Blood Pressure", f"{patient.blood_pressure:.1f} mmHg"),
        ("BMI", f"{patient.bmi:.1f}"),
        ("Oxygen Saturation", f"{patient.oxygen_saturation:.1f}%"),
        ("Cholesterol", f"{patient.cholesterol:.1f} mg/dL"),
        ("Triglycerides", f"{patient.triglycerides:.1f} mg/dL"),
        ("HbA1c", f"{patient.hba1c:.1f}%"),
        ("Smoking", yes_no(patient.smoking)),
        ("Alcohol Use", yes_no(patient.alcohol)),
        ("Family History", yes_no(patient.family_history)),
        ("Physical Activity", f"{patient.physical_activity:.1f} hrs/week"),
        ("Diet Score", f"{patient.diet_score:.1f}"),
        ("Stress Level", f"{patient.stress_level:.1f}"),
        ("Sleep Hours", f"{patient.sleep_hours:.1f} hrs"),
    ]
    left, right = st.columns(2)
    half = (len(rows) + 1) // 2
    for col, chunk in zip((left, right), (rows[:half], rows[half:])):
        with col:
            for label, value in chunk:
                st.markdown(
                    f"<div style='display:flex; justify-content:space-between; "
                    f"padding: 0.35rem 0; border-bottom: 1px solid rgba(15, 110, 99, 0.3); font-size:0.9rem;'>"
                    f"<span style='color:var(--clr-text-muted);'>{label}</span>"
                    f"<span style='font-weight:600;'>{value}</span></div>",
                    unsafe_allow_html=True,
                )


def _plain_english_explanation(contributors: dict, prediction: float, baseline: float) -> str:
    direction = "longer" if prediction > baseline else "shorter"
    lines = [
        f"The model predicts a length of stay **{format_days(prediction)}**, which is "
        f"{direction} than the model's average prediction of {format_days(baseline)}."
    ]
    if contributors["increasing"]:
        names = ", ".join(name for name, _ in contributors["increasing"])
        lines.append(f"Factors **increasing** the predicted stay: {names}.")
    if contributors["decreasing"]:
        names = ", ".join(name for name, _ in contributors["decreasing"])
        lines.append(f"Factors **decreasing** the predicted stay: {names}.")
    return "  \n".join(lines)


def render() -> None:
    page_header(
        "Length of Stay Prediction",
        "Enter patient information to generate an individual prediction.",
        icon=get_icon("stethoscope", 26),
    )

    package = load_deployment_package()

    patient = _entry_form(package)

    if patient is not None:
        prediction, scaled_row = predict(patient, package)
        explanation = explain(scaled_row, package)
        st.session_state["last_prediction"] = {
            "patient": patient,
            "prediction": prediction,
            "explanation": explanation,
        }

    state = st.session_state.get("last_prediction")
    if not state:
        return

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("### Prediction Result")

    baseline = float(state["explanation"].base_values)
    render_metric_row(
        [
            {"label": "Predicted Length of Stay", "value": format_days(state["prediction"])},
            {"label": "Model Average Prediction", "value": format_days(baseline)},
            {
                "label": "Difference From Average",
                "value": format_signed(state["prediction"] - baseline, 1) + " days",
            },
        ]
    )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("<div class='app-card'>", unsafe_allow_html=True)
        _render_summary(state["patient"])
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='app-card'>", unsafe_allow_html=True)
        section_title("Why this prediction?")
        contributors = top_contributing_features(state["explanation"])
        st.markdown(
            _plain_english_explanation(contributors, state["prediction"], baseline)
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    section_title("Local SHAP Explanation")
    st.caption(
        "This waterfall plot shows how each patient characteristic pushed the "
        "prediction above or below the model's average output."
    )
    shap.plots.waterfall(state["explanation"], show=False)
    fig = plt.gcf()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)
