import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

st.set_page_config(page_title="Energy Consumption Predictor", page_icon="⚡")

st.title("⚡ Energy Consumption Predictor")
st.caption("Linear Regression — Sklearn vs From-Scratch (Gradient Descent)")

# --- Load models ---
@st.cache_resource
def load_models():
    sklearn_model = joblib.load("sklearn_model.pkl")
    with open("scratch_model.pkl", "rb") as f:
        scratch = pickle.load(f)
    return sklearn_model, scratch

sklearn_model, scratch = load_models()
scaler = scratch["scaler"]
num_cols = scratch["num_cols"]
cat_cols = scratch["cat_cols"]
columns = scratch["columns"]
theta = scratch["theta"]

# --- Sidebar inputs ---
st.sidebar.header("Input Features")

square_footage = st.sidebar.number_input("Square Footage", min_value=100, max_value=100000, value=10000)
occupants = st.sidebar.number_input("Number of Occupants", min_value=1, max_value=500, value=30)
appliances = st.sidebar.number_input("Appliances Used", min_value=0, max_value=200, value=20)
temperature = st.sidebar.number_input("Average Temperature", min_value=-20.0, max_value=50.0, value=22.0)
day_of_week = st.sidebar.selectbox("Day of Week", ["Weekday", "Weekend"])
building_type = st.sidebar.selectbox("Building Type", ["Residential", "Commercial", "Industrial"])

model_choice = st.sidebar.radio("Model", ["Sklearn (built-in)", "From Scratch (GD)", "Compare Both"])

# --- Build input row matching training format ---
def build_input():
    row = pd.DataFrame([{
        "Square Footage": square_footage,
        "Number of Occupants": occupants,
        "Appliances Used": appliances,
        "Average Temperature": temperature,
        "Day of Week": day_of_week,
        "Building Type": building_type
    }])

    # engineered features (must match training exactly)
    row["Occupants_per_sqft"] = row["Number of Occupants"] / row["Square Footage"]
    row["Appliances_per_person"] = row["Appliances Used"] / row["Number of Occupants"]

    row[num_cols] = scaler.transform(row[num_cols])
    row = pd.get_dummies(row, columns=cat_cols, drop_first=True)

    # align columns to match training set exactly
    row = row.reindex(columns=columns, fill_value=0)
    return row

def predict_scratch(X, theta):
    m = X.shape[0]
    X_b = np.c_[np.ones((m, 1)), X.values]
    return X_b.dot(theta).flatten()[0]

# --- Predict ---
if st.button("Predict Energy Consumption"):
    X_input = build_input()

    if model_choice == "Sklearn (built-in)":
        pred = sklearn_model.predict(X_input)[0]
        st.success(f"Predicted Energy Consumption: **{pred:.2f}**")

    elif model_choice == "From Scratch (GD)":
        pred = predict_scratch(X_input, theta)
        st.success(f"Predicted Energy Consumption: **{pred:.2f}**")

    else:
        pred_sklearn = sklearn_model.predict(X_input)[0]
        pred_scratch = predict_scratch(X_input, theta)
        col1, col2 = st.columns(2)
        col1.metric("Sklearn Prediction", f"{pred_sklearn:.2f}")
        col2.metric("Scratch (GD) Prediction", f"{pred_scratch:.2f}")
        st.caption(f"Difference: {abs(pred_sklearn - pred_scratch):.4f}")

# --- Loss curve ---
if "loss_history" in scratch:
    st.header("Gradient descent convergence")
    st.caption("MSE loss over training iterations — proves the from-scratch model actually learned, not just copied sklearn's output.")
    loss_df = pd.DataFrame({"Iteration": range(len(scratch["loss_history"])), "MSE Loss": scratch["loss_history"]})
    st.line_chart(loss_df.set_index("Iteration"))
    st.caption(f"Final loss: {scratch['loss_history'][-1]:.2f} after {len(scratch['loss_history'])} iterations")

# --- Feature importance / coefficients ---
if "coef_dict" in scratch:
    st.header("Feature importance")
    st.caption("Sklearn model coefficients — how much each feature moves predicted energy consumption.")
    coef_df = pd.DataFrame({
        "Feature": list(scratch["coef_dict"].keys()),
        "Coefficient": list(scratch["coef_dict"].values())
    }).sort_values("Coefficient")
    st.bar_chart(coef_df.set_index("Feature"))
    st.caption(f"Intercept (baseline value): {scratch.get('intercept', 0):.2f}")

    with st.expander("View learned equation"):
        eq = f"Energy Consumption = {scratch.get('intercept', 0):.2f}"
        for feat, c in scratch["coef_dict"].items():
            sign = "+" if c >= 0 else "-"
            eq += f" {sign} {abs(c):.3f} × ({feat})"
        st.code(eq, language=None)

with st.expander("Preprocessing steps"):
    st.markdown("""
    - Checked for missing values and duplicates
    - Engineered 3 features: Occupants_per_sqft, Appliances_per_person
    - Scaled numeric features (including engineered ones) with StandardScaler
    - One-hot encoded categorical features (Day of Week, Building Type) with drop_first to avoid multicollinearity
    - Aligned train/test columns after encoding to prevent mismatches
    """)
