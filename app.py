import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="🏭",
    layout="wide"
)

model = joblib.load("machine_model.joblib")
df = pd.read_csv("cleaned_dataset.csv")
results = pd.read_csv("final_results.csv")

FEATURES = model.feature_names_in_.tolist()

st.markdown("""
<style>
[data-testid="stMetric"] {
    border: 1px solid #cccccc;
    padding: 18px;
    border-radius: 12px;
}
.stButton > button {
    background: #2563eb;
    color: white;
    width: 100%;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🏭 Machine Analytics")


page = st.sidebar.radio("Navigation", [
    "Dashboard",
    "Model Performance",
    "Predict Machine",
    "Upload New Dataset"
])

st.title("Industrial Machine Failure Prediction")
st.caption("Machine Learning Based Predictive Maintenance")

# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    a, b, c = st.columns(3)

    a.metric("Total Machines", len(df))
    b.metric("Failed Machines", int(df["Machine failure"].sum()))
    c.metric("Best Model", results.iloc[0]["Model"])

    col1, col2 = st.columns(2)

    col1.plotly_chart(px.pie(
        df, names="Machine failure",
        hole=0.45, title="Failure Distribution"
    ))

    col2.plotly_chart(px.histogram(
        df, x="Type", color="Machine failure",
        title="Machine Type Analysis"
    ))

    st.plotly_chart(px.scatter(
        df, x="Rotational speed [rpm]",
        y="Torque [Nm]", color="Machine failure",
        title="Rotational Speed vs Torque"
    ))

    st.dataframe(df.head(50))

# ---------------- MODEL PERFORMANCE ----------------

elif page == "Model Performance":

    comparison = pd.read_csv("model_comparison.csv")

    cols = st.columns(4)

    for col, metric in zip(cols, [
        "Accuracy", "Precision", "Recall", "F1-Score"
    ]):
        col.metric(metric, f"{results.iloc[0][metric]:.2%}")

    st.subheader("Model Comparison")

    st.dataframe(comparison)

    st.plotly_chart(px.bar(
        comparison, x="Model",
        y=["Accuracy", "Precision", "Recall", "F1-Score"],
        barmode="group"
    ))

    st.subheader("Cross-Validation Results")
    st.dataframe(pd.read_csv("cross_validation.csv"))

    st.subheader("Hyperparameter Tuning")
    st.dataframe(pd.read_csv("tuning_results.csv"))

# ---------------- INDIVIDUAL PREDICTION ----------------

elif page == "Predict Machine":

    st.subheader("Enter Machine Operating Conditions")

    a, b = st.columns(2)

    machine_type = a.selectbox("Machine Type", ["L", "M", "H"])
    air = b.number_input("Air Temperature (K)", 0.0, value=300.0)
    process = a.number_input("Process Temperature (K)", 0.0, value=310.0)
    speed = b.number_input("Rotational Speed (rpm)", 0, value=1500)
    torque = a.number_input("Torque (Nm)", 0.0, value=40.0)
    wear = b.number_input("Tool Wear (min)", 0, value=100)

    if st.button("Predict Machine Failure"):

        data = pd.DataFrame([{
            "Type": machine_type,
            "Air temperature [K]": air,
            "Process temperature [K]": process,
            "Rotational speed [rpm]": speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear
        }])

        pred = model.predict(data[FEATURES])[0]
        prob = model.predict_proba(data[FEATURES])[0, 1]

        if pred == 1:
            st.error("Machine Failure: YES")
        else:
            st.success("Machine Failure: NO")

        st.metric("Estimated Failure Probability", f"{prob:.2%}")
        st.progress(float(prob))

# ---------------- NEW CSV PREDICTION ----------------

else:

    st.subheader("Upload New Machine Dataset")

    uploaded = st.file_uploader("Upload CSV File", type="csv")

    if uploaded:

        try:
            data = pd.read_csv(uploaded)

            if data.empty:
                raise ValueError("CSV file is empty")

            if not all(c in data.columns for c in FEATURES):
                raise ValueError("Required feature columns are missing")

            data["Type"] = data["Type"].str.strip().str.upper()

            if not data["Type"].isin(["L", "M", "H"]).all():
                raise ValueError("Invalid Machine Type")

            for col in FEATURES:
                if col != "Type":
                    data[col] = pd.to_numeric(data[col], errors="raise")

                    if not np.isfinite(data[col]).all() or (data[col] < 0).any():
                        raise ValueError("Invalid values in " + col)

            data["Predicted Failure"] = model.predict(data[FEATURES])

            data["Failure Probability (%)"] = (
                model.predict_proba(data[FEATURES])[:, 1] * 100
            ).round(2)

            data["Status"] = data["Predicted Failure"].map({
                0: "No Failure", 1: "Failure"
            })

            st.dataframe(data)

            st.plotly_chart(px.pie(
                data, names="Status",
                hole=0.4, title="Prediction Results"
            ))

            st.download_button(
                "Download Prediction Results",
                data.to_csv(index=False),
                "machine_predictions.csv",
                "text/csv"
            )

        except ValueError as e:
            st.error(str(e))

st.divider()
