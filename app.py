import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Load model + scaler (same objects saved from your notebook, untouched)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("final_svm_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error(
        "Could not find `final_svm_model.pkl` and/or `scaler.pkl`. "
        "Place both files in the same folder as this app.py before running."
    )
    st.stop()

# This is the exact column order the scaler/model were fit on
# (pulled straight from scaler.feature_names_in_ — nothing invented).
FEATURE_ORDER = list(scaler.feature_names_in_)


# ----------------------------------------------------------------------------
# Build the model-ready feature vector from raw form inputs.
# This reproduces: drop customerID -> get_dummies(drop_first=True) -> scaler
# with the SAME reference (dropped) categories your training data had.
# ----------------------------------------------------------------------------
def build_feature_row(inputs: dict) -> pd.DataFrame:
    row = {col: 0 for col in FEATURE_ORDER}

    # Numeric features
    row["SeniorCitizen"] = int(inputs["SeniorCitizen"])
    row["tenure"] = float(inputs["tenure"])
    row["MonthlyCharges"] = float(inputs["MonthlyCharges"])
    row["TotalCharges"] = float(inputs["TotalCharges"])

    # Binary Yes/No categoricals (reference/dropped category = "No" or "Female")
    if inputs["gender"] == "Male":
        row["gender_Male"] = 1
    if inputs["Partner"] == "Yes":
        row["Partner_Yes"] = 1
    if inputs["Dependents"] == "Yes":
        row["Dependents_Yes"] = 1
    if inputs["PhoneService"] == "Yes":
        row["PhoneService_Yes"] = 1
    if inputs["PaperlessBilling"] == "Yes":
        row["PaperlessBilling_Yes"] = 1

    # MultipleLines (reference = "No")
    if inputs["MultipleLines"] == "No phone service":
        row["MultipleLines_No phone service"] = 1
    elif inputs["MultipleLines"] == "Yes":
        row["MultipleLines_Yes"] = 1

    # InternetService (reference = "DSL")
    if inputs["InternetService"] == "Fiber optic":
        row["InternetService_Fiber optic"] = 1
    elif inputs["InternetService"] == "No":
        row["InternetService_No"] = 1

    # Internet-dependent add-ons (reference = "No")
    for feat in [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]:
        val = inputs[feat]
        if val == "No internet service":
            row[f"{feat}_No internet service"] = 1
        elif val == "Yes":
            row[f"{feat}_Yes"] = 1

    # Contract (reference = "Month-to-month")
    if inputs["Contract"] == "One year":
        row["Contract_One year"] = 1
    elif inputs["Contract"] == "Two year":
        row["Contract_Two year"] = 1

    # PaymentMethod (reference = "Bank transfer (automatic)")
    pm = inputs["PaymentMethod"]
    if pm == "Credit card (automatic)":
        row["PaymentMethod_Credit card (automatic)"] = 1
    elif pm == "Electronic check":
        row["PaymentMethod_Electronic check"] = 1
    elif pm == "Mailed check":
        row["PaymentMethod_Mailed check"] = 1

    return pd.DataFrame([row], columns=FEATURE_ORDER)


# ----------------------------------------------------------------------------
# Sidebar - about
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About this app")
    st.write(
        "Predicts customer churn using your tuned **SVM (linear kernel, C=1)** "
        "model. The preprocessing (feature encoding + scaling) mirrors your "
        "training notebook exactly — no logic was changed."
    )
    st.divider()
    st.caption("Model file: `final_svm_model.pkl`")
    st.caption("Scaler file: `scaler.pkl`")
    st.caption(f"Features expected: {len(FEATURE_ORDER)}")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("📉 Customer Churn Prediction")
st.write(
    "Fill in the customer's details below and click **Predict Churn** to see "
    "whether the model expects this customer to leave."
)
st.divider()

# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
with st.form("churn_form"):
    st.subheader("👤 Demographics")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
    with c2:
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    with c3:
        partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])

    st.subheader("📞 Phone & Internet Services")
    c1, c2 = st.columns(2)
    with c1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        if phone_service == "Yes":
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
        else:
            multiple_lines = "No phone service"
            st.selectbox(
                "Multiple Lines", ["No phone service"], disabled=True
            )
    with c2:
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )

    st.subheader("🛡️ Add-on Services")
    has_internet = internet_service != "No"
    addon_cols = st.columns(3)
    addon_labels = [
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
    ]
    addon_keys = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    addon_values = {}
    for i, (label, key) in enumerate(zip(addon_labels, addon_keys)):
        with addon_cols[i % 3]:
            if has_internet:
                addon_values[key] = st.selectbox(label, ["No", "Yes"], key=key)
            else:
                st.selectbox(
                    label, ["No internet service"], disabled=True, key=key
                )
                addon_values[key] = "No internet service"

    st.subheader("📄 Contract & Billing")
    c1, c2, c3 = st.columns(3)
    with c1:
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
    with c2:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    with c3:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    st.subheader("💰 Charges & Tenure")
    c1, c2, c3 = st.columns(3)
    with c1:
        tenure = st.number_input(
            "Tenure (months)", min_value=0, max_value=100, value=12, step=1
        )
    with c2:
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0, value=70.0, step=0.5
        )
    with c3:
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0, value=840.0, step=1.0
        )

    submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if submitted:
    inputs = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": addon_values["OnlineSecurity"],
        "OnlineBackup": addon_values["OnlineBackup"],
        "DeviceProtection": addon_values["DeviceProtection"],
        "TechSupport": addon_values["TechSupport"],
        "StreamingTV": addon_values["StreamingTV"],
        "StreamingMovies": addon_values["StreamingMovies"],
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    X_row = build_feature_row(inputs)
    X_scaled = scaler.transform(X_row)

    prediction = model.predict(X_scaled)[0]

    st.divider()
    st.subheader("Result")

    if prediction == "Yes":
        st.error("⚠️ This customer is **likely to churn**.")
    else:
        st.success("✅ This customer is **likely to stay**.")

    # The model was trained with probability=False, so calibrated
    # probabilities aren't available. We show the decision_function
    # distance instead, as a rough confidence indicator (not a true probability).
    try:
        decision_score = model.decision_function(X_scaled)[0]
        confidence_pct = min(abs(decision_score) / 2.0, 1.0) * 100
        st.caption(
            f"Decision-function distance from boundary: `{decision_score:.3f}` "
            f"(rough confidence ≈ {confidence_pct:.0f}%). "
            "Note: this SVM was trained with `probability=False`, so this is "
            "not a calibrated probability — just how far this case sits from "
            "the decision boundary."
        )
        st.progress(confidence_pct / 100)
    except Exception:
        pass

    with st.expander("See the exact encoded feature vector sent to the model"):
        st.dataframe(X_row.T.rename(columns={0: "value"}))
