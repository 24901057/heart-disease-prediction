
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️"
)

st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient features to run the ML model.")

# Load saved model and preprocessing objects
@st.cache_resource
def load_artifact():
    return joblib.load("heart_disease_model.pkl")

try:
    artifact = load_artifact()
    model = artifact["model"]
    imputer = artifact["imputer"]
    scaler = artifact["scaler"]
    columns = artifact["columns"]
except FileNotFoundError:
    st.error("Please place heart_disease_model.pkl in this folder.")
    st.stop()

st.subheader("Patient information")

# Create inputs based on the dataset's feature columns
values = {}

for col in columns:
    values[col] = st.number_input(
        col.replace("_", " ").title(),
        value=0.0,
        step=1.0,
        key=col
    )

if st.button("Predict", type="primary"):
    # Prepare input in the same column order used for training
    input_df = pd.DataFrame([values], columns=columns)

    # Apply the same preprocessing used during training
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)

    # Predict
    prediction = int(model.predict(input_scaled)[0])

    st.subheader("Prediction result")

    if prediction == 1:
        st.warning(
            "The model predicts the positive class "
            "for heart disease."
        )
    else:
        st.success(
            "The model predicts the negative class "
            "for heart disease."
        )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_scaled)[0]

        st.write("Model class probabilities:")
        for label, probability in zip(
            model.classes_, probabilities
        ):
            st.write(f"Class {label}: {probability:.2%}")

st.caption(
    "Educational demonstration only. This application "
    "cannot diagnose or rule out heart disease."
)
