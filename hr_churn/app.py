import streamlit as st
import pandas as pd
import joblib

# ================== Page Config ==================
st.set_page_config(
    page_title="HR Churn Prediction",
    page_icon="🧑‍💼",
    layout="wide",
)

# ================== Load Model ==================
model = joblib.load("hr_churn_model.pkl")
encoders = joblib.load("hr_encoder.pkl")
columns = joblib.load("hr_columns.pkl")

# ================== Header ==================
st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stDataFrame {
        border-radius: 12px;
        border: 1px solid #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧑‍💼 HR Churn Prediction System")
st.caption("A simple ML-powered system to predict employee attrition risk.")

st.markdown("---")

# ================== Upload CSV ==================
uploaded_file = st.file_uploader("📂 Upload Employee Data (.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Encode categorical columns
    for col in df.columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col].astype(str))

    df = df[columns]  # Ensure column order

    # Predictions
    preds = model.predict(df)
    df["Attrition_Prediction"] = preds
    df["Attrition_Prediction"] = df["Attrition_Prediction"].map({1: "Yes", 0: "No"})

    # ================== Summary Metrics ==================
    total = len(df)
    churn_yes = (df["Attrition_Prediction"] == "Yes").sum()
    churn_no = total - churn_yes

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", total)
    col2.metric("Likely to Leave", churn_yes)
    col3.metric("Likely to Stay", churn_no)

    st.markdown("---")
    st.success("✅ Prediction Complete!")

    # ================== Display Data ==================
    st.subheader("Prediction Results")
    st.dataframe(df, use_container_width=True)

    # ================== Download Button ==================
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Predictions",
        data=csv_data,
        file_name="predictions.csv",
        mime="text/csv",
    )

else:
    st.info("👆 Please upload a CSV file to get predictions.")
