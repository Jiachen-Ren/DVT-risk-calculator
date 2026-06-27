from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="DVT Risk Calculator",
    page_icon="🩸",
    layout="centered",
)


@st.cache_resource
def load_bundle():
    return joblib.load("final_model_bundle.joblib")


bundle = load_bundle()

MODEL_NAME = bundle["model_name"]
SELECTED_VARS = bundle["selected_vars"]
CATEGORICAL_VARS = set(bundle.get("categorical_vars", []))
LOG_TRANSFORM_VARS = set(bundle.get("log_transform_vars", []))
VARIABLE_LABELS = bundle.get("variable_labels", {})
CATEGORY_LEVELS = bundle.get("category_levels", {})
MODELS = bundle["models"]
PREPROCESSORS = bundle["preprocessors"]
MODEL_CUTOFF = float(bundle.get("model_cutoff", 0.5))


DEFAULT_VALUES = {
    "age": 65,
    "hgb": 100.0,
    "plt": 200.0,
    "d_dimer": 1000.0,
    "at3": 80.0,
    "trauma": 0,
}

HELP_TEXT = {
    "age": "Age in years.",
    "trauma": "Trauma status before the prediction time point.",
    "hgb": "Hemoglobin. Use the same unit as the modeling dataset.",
    "plt": "Platelet count. Use the same unit as the modeling dataset.",
    "d_dimer": "D-dimer. Use the same unit as the modeling dataset; log transformation is applied automatically inside the calculator.",
    "at3": "Antithrombin III. Use the same unit as the modeling dataset.",
}


def label_of(var: str) -> str:
    return VARIABLE_LABELS.get(var, var)


def format_level(var: str, level):
    try:
        level_float = float(level)
        if var in {"trauma", "cancer", "vte_history", "surgery_history", "transfusion_history", "vasoactive"}:
            if level_float == 0:
                return "No"
            if level_float == 1:
                return "Yes"
        if var == "sex":
            return f"Code {level:g}"
        if level_float.is_integer():
            return str(int(level_float))
        return str(level)
    except Exception:
        return str(level)


def build_input_widgets() -> pd.DataFrame:
    st.header("Patient information")
    values = {}

    for var in SELECTED_VARS:
        label = label_of(var)
        help_msg = HELP_TEXT.get(var, "Use the same definition and unit as the modeling dataset.")

        if var in CATEGORICAL_VARS:
            levels = CATEGORY_LEVELS.get(var, [0, 1])
            clean_levels = []
            for x in levels:
                try:
                    xf = float(x)
                    clean_levels.append(int(xf) if xf.is_integer() else xf)
                except Exception:
                    clean_levels.append(x)

            display_mapping = {format_level(var, x): x for x in clean_levels}
            default_raw = DEFAULT_VALUES.get(var, clean_levels[0] if clean_levels else 0)
            default_display = format_level(var, default_raw)
            if default_display not in display_mapping and display_mapping:
                default_display = list(display_mapping.keys())[0]

            options = list(display_mapping.keys())
            chosen_display = st.selectbox(
                label,
                options=options,
                index=options.index(default_display) if default_display in options else 0,
                help=help_msg,
            )
            values[var] = display_mapping[chosen_display]
        else:
            default = float(DEFAULT_VALUES.get(var, 0.0))
            if var == "age":
                values[var] = st.number_input(label, min_value=18.0, max_value=120.0, value=float(default), step=1.0, help=help_msg)
            elif var == "d_dimer":
                values[var] = st.number_input(label, min_value=0.0, max_value=1000000.0, value=float(default), step=10.0, help=help_msg)
            elif var in {"hgb", "plt", "at3"}:
                values[var] = st.number_input(label, min_value=0.0, max_value=100000.0, value=float(default), step=1.0, help=help_msg)
            else:
                values[var] = st.number_input(label, value=float(default), step=1.0, help=help_msg)

    return pd.DataFrame([values], columns=SELECTED_VARS)


def prepare_model_input(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    for var in SELECTED_VARS:
        df[var] = pd.to_numeric(df[var], errors="coerce")

    for var in SELECTED_VARS:
        if var not in CATEGORICAL_VARS:
            continue
        levels = CATEGORY_LEVELS.get(var)
        if not levels:
            continue

        clean_levels = []
        for x in levels:
            try:
                xf = float(x)
                clean_levels.append(int(xf) if xf.is_integer() else xf)
            except Exception:
                clean_levels.append(x)

        mapping = {}
        for i, v in enumerate(clean_levels):
            try:
                mapping[float(v)] = i
            except Exception:
                pass

        df[var] = df[var].map(lambda x: mapping.get(float(x), np.nan) if pd.notna(x) else np.nan)

    for var in SELECTED_VARS:
        if var in LOG_TRANSFORM_VARS:
            x = pd.to_numeric(df[var], errors="coerce")
            df[var] = np.where(x >= 0, np.log1p(x), np.nan)

    return df[SELECTED_VARS]


def predict_probability(model_df: pd.DataFrame) -> float:
    probs = []
    for model, preprocessor in zip(MODELS, PREPROCESSORS):
        X = preprocessor.transform(model_df)
        probs.append(float(model.predict_proba(X)[0, 1]))
    return float(np.mean(probs))


st.title("Early Prediction Calculator for Lower-Extremity DVT")
st.markdown(
    """
This web-based calculator estimates the individualized risk of **new-onset lower-extremity deep vein thrombosis within 7 days**
after the first 24 hours of mechanical ventilation in ICU patients.
"""
)

st.warning(
    "This calculator is a research prototype for clinical decision support only. "
    "It should not replace clinical judgment, guideline-based thromboprophylaxis, or lower-extremity ultrasonography."
)

with st.expander("Model information", expanded=False):
    st.write(f"Final model: **{MODEL_NAME}**")
    st.write("Selected predictors:")
    st.write([label_of(v) for v in SELECTED_VARS])
    st.write(f"Model-derived high-risk cutoff: **{MODEL_CUTOFF:.3f}**")
    st.caption("All laboratory values must use the same units as the modeling dataset.")

raw_input = build_input_widgets()

if st.button("Calculate DVT risk", type="primary"):
    model_input = prepare_model_input(raw_input)

    if model_input.isna().any(axis=None):
        st.error("Some inputs could not be processed. Please check the selected categories and numeric values.")
    else:
        risk = predict_probability(model_input)

        st.subheader("Prediction result")
        st.metric(
            label="Predicted 7-day risk of new-onset lower-extremity DVT",
            value=f"{risk * 100:.1f}%",
        )

        if risk >= MODEL_CUTOFF:
            st.error(f"Risk category: High risk (≥ {MODEL_CUTOFF:.3f})")
        else:
            st.success(f"Risk category: Lower risk (< {MODEL_CUTOFF:.3f})")

        st.markdown("### Interpretation")
        st.markdown(
            f"""
The predicted probability of new-onset lower-extremity DVT within 7 days is **{risk * 100:.1f}%**.

This estimate was generated by the final model using the following predictors:
{", ".join([label_of(v) for v in SELECTED_VARS])}.
"""
        )

        st.info(
            "Patients classified as high risk may need closer DVT surveillance and individualized thromboprophylaxis assessment, "
            "according to clinician judgment and local protocols."
        )

        with st.expander("Show model input after preprocessing"):
            st.dataframe(model_input)

st.markdown("---")
st.caption("Prototype web-based calculator. For research use and clinical decision support only.")
