"""Prediction page: patient data entry, prediction and local SHAP explanation."""

import matplotlib.pyplot as plt
import shap
import streamlit as st

from src import config
from src.model.loader import load_deployment_package
from src.model.predictor import PatientInput, explain, predict, top_contributing_features
from src.ui.components import card, page_header, render_metric_row, section_title
from src.ui.icons import icon as get_icon
from src.utils.formatting import format_days, format_signed_days

_SELECT_PLACEHOLDER = "— Select —"


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
    c1, c2 = st.columns([2, 1])
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


def _join_naturally(phrases: list[str]) -> str:
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f" and {phrases[-1]}"


def _plain_english_explanation(contributors: dict, prediction: float, baseline: float) -> str:
    rounded_prediction, rounded_baseline = round(prediction), round(baseline)

    if rounded_prediction == rounded_baseline:
        lines = [
            f"The model predicts a stay of **{format_days(prediction)}**, about the same "
            f"as its average prediction of {format_days(baseline)}."
        ]
    else:
        direction = "longer" if prediction > baseline else "shorter"
        lines = [
            f"The model predicts a stay of **{format_days(prediction)}**, which is "
            f"{direction} than its average prediction of {format_days(baseline)}."
        ]

    if contributors["increasing"]:
        phrases = _join_naturally([_describe(name) for name, _ in contributors["increasing"]])
        lines.append(f"This was pushed **up** mainly by {phrases}.")
    if contributors["decreasing"]:
        phrases = _join_naturally([_describe(name) for name, _ in contributors["decreasing"]])
        lines.append(f"This was pulled **down** mainly by {phrases}.")

    return "  \n".join(lines)


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
            "prediction": prediction,
            "explanation": explanation,
            "model_name": model_name,
        }

    state = st.session_state.get("last_prediction")
    if not state:
        return

    st.markdown(
        f"### Prediction Result "
        f"<span class='pill' style='vertical-align:middle; margin-left:0.5rem;'>"
        f"{state.get('model_name', 'Random Forest')}</span>",
        unsafe_allow_html=True,
    )

    baseline = float(state["explanation"].base_values)
    render_metric_row(
        [
            {"label": "Predicted Length of Stay", "value": format_days(state["prediction"])},
            {"label": "Model Average Prediction", "value": format_days(baseline)},
            {
                "label": "Difference From Average",
                "value": format_signed_days(state["prediction"] - baseline),
            },
        ]
    )

    with card():
        section_title("Why this prediction?")
        contributors = top_contributing_features(state["explanation"])
        st.markdown(_plain_english_explanation(contributors, state["prediction"], baseline))

    with card():
        section_title("How the Model Reached This Number")
        st.caption(
            "Starting from the model's average prediction, each bar shows how much "
            "one patient detail pushed the number up (red) or down (blue) to arrive "
            "at the final prediction."
        )
        shap.plots.waterfall(state["explanation"], show=False)
        fig = plt.gcf()
        st.pyplot(fig, width="stretch")
        plt.close(fig)
