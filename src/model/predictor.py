"""
Prediction and explanation logic.

This module is the single place that touches the fitted `model` and `scaler`
from the deployment package. It re-creates, for a single new patient, exactly
the same encoding and scaling steps the notebook applied to the training data:

    raw inputs -> one-hot encode (Gender, Medical Condition) -> order columns
    to match `feature_names` -> scaler.transform() -> model.predict()

No preprocessing logic is changed or re-fitted; the scaler and model are used
strictly in inference mode.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from src import config


@dataclass
class PatientInput:
    age: float
    gender: str
    medical_condition: str
    glucose: float
    blood_pressure: float
    bmi: float
    oxygen_saturation: float
    cholesterol: float
    triglycerides: float
    hba1c: float
    smoking: bool
    alcohol: bool
    physical_activity: float
    diet_score: float
    family_history: bool
    stress_level: float
    sleep_hours: float


def build_feature_row(
    patient: PatientInput,
    feature_names: list[str],
    medical_conditions: list[str],
) -> pd.DataFrame:
    """Assemble a single-row DataFrame matching the notebook's encoded feature order.

    Categorical variables are expanded into the same one-hot dummy columns
    produced by `pd.get_dummies(..., drop_first=True)` in the notebook, with
    the baseline categories (Gender="Female", Medical Condition="Arthritis")
    represented implicitly as all-zero dummies.
    """
    row = {
        "Age": patient.age,
        "Glucose": patient.glucose,
        "Blood Pressure": patient.blood_pressure,
        "BMI": patient.bmi,
        "Oxygen Saturation": patient.oxygen_saturation,
        "Cholesterol": patient.cholesterol,
        "Triglycerides": patient.triglycerides,
        "HbA1c": patient.hba1c,
        "Smoking": int(patient.smoking),
        "Alcohol": int(patient.alcohol),
        "Physical Activity": patient.physical_activity,
        "Diet Score": patient.diet_score,
        "Family History": int(patient.family_history),
        "Stress Level": patient.stress_level,
        "Sleep Hours": patient.sleep_hours,
        "Gender_Male": int(patient.gender == "Male"),
    }

    for condition in medical_conditions:
        column = f"Medical Condition_{condition}"
        if column not in feature_names:
            # Baseline category dropped by pd.get_dummies(drop_first=True);
            # represented implicitly by all other dummies being 0.
            continue
        row[column] = int(patient.medical_condition == condition)

    return pd.DataFrame([row], columns=feature_names)


def predict(patient: PatientInput, package: dict) -> tuple[float, pd.DataFrame]:
    """Scale the patient row with the fitted scaler and predict with the fitted model.

    Returns the predicted length of stay (days) and the scaled feature row
    (needed downstream for SHAP explanation).
    """
    feature_names = package["feature_names"]
    model = package["model"]
    scaler = package["scaler"]
    medical_conditions = package["medical_conditions"]

    raw_row = build_feature_row(patient, feature_names, medical_conditions)
    scaled_values = scaler.transform(raw_row)
    scaled_row = pd.DataFrame(scaled_values, columns=feature_names, index=raw_row.index)

    prediction = float(model.predict(scaled_row)[0])
    return prediction, scaled_row


def explain(scaled_row: pd.DataFrame, package: dict) -> shap.Explanation:
    """Generate a SHAP explanation for a single scaled patient row.

    Uses shap.TreeExplainer on the deployed Random Forest model, matching the
    notebook's explainability methodology exactly.
    """
    model = package["model"]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled_row)

    display_row = scaled_row.rename(columns=config.DISPLAY_NAME_OVERRIDES)
    base_value = np.asarray(explainer.expected_value).reshape(-1)[0]

    return shap.Explanation(
        values=shap_values[0],
        base_values=base_value,
        data=display_row.iloc[0].values,
        feature_names=display_row.columns.tolist(),
    )


def top_contributing_features(explanation: shap.Explanation, n: int = 3) -> dict:
    """Return the top-n features increasing and decreasing the prediction."""
    values = np.asarray(explanation.values)
    names = np.asarray(explanation.feature_names)

    order = np.argsort(values)

    decreasing = [(names[i], values[i]) for i in order[:n] if values[i] < 0]
    increasing = [(names[i], values[i]) for i in order[::-1][:n] if values[i] > 0]

    return {"increasing": increasing, "decreasing": decreasing}
