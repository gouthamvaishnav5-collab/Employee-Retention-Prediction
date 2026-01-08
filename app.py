# app.py
import streamlit as st
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Employee Retention Prediction",
    layout="centered"
)

# -----------------------------
# Session State for Auth
# -----------------------------
if "users" not in st.session_state:
    st.session_state.users = {"admin": "admin123"}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None

# -----------------------------
# Login Page
# -----------------------------
def login_page():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("✅ Login successful")
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

    st.markdown("Don't have an account?")
    if st.button("Go to Signup"):
        st.session_state.page = "signup"
        st.rerun()

# -----------------------------
# Signup Page
# -----------------------------
def signup_page():
    st.title("📝 Signup")

    new_user = st.text_input("Create Username")
    new_pass = st.text_input("Create Password", type="password")
    confirm_pass = st.text_input("Confirm Password", type="password")

    if st.button("Signup"):
        if new_user in st.session_state.users:
            st.error("❌ Username already exists")
        elif new_pass != confirm_pass:
            st.error("❌ Passwords do not match")
        else:
            st.session_state.users[new_user] = new_pass
            st.success("✅ Account created successfully")
            st.session_state.page = "login"
            st.rerun()

    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()

# -----------------------------
# Prediction Page
# -----------------------------
def prediction_page():
    st.title("👨‍💼 Employee Retention Prediction")
    st.write("XGBoost-based Job Change Prediction App")

    st.sidebar.success(f"Logged in as {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()

    # Load model & encoders
    model = load("xgb_model.joblib")
    encoders = load("encoders.joblib")

    st.markdown("---")
    st.subheader("Enter Employee Details")

    EXCLUDED_FEATURES = ["enrollee_id"]
    input_data = {}

    # Build UI inputs (exclude enrollee_id)
    for col in model.feature_names_in_:
        if col in EXCLUDED_FEATURES:
            continue

        if col in encoders:
            input_data[col] = st.selectbox(col, list(encoders[col].classes_))
        else:
            input_data[col] = st.number_input(col)

    # Inject hidden feature so model doesn't break
    input_data["enrollee_id"] = 0

    # Create DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    for col, le in encoders.items():
        input_df[col] = le.transform(input_df[col])

    # IMPORTANT: reorder columns exactly as model expects
    input_df = input_df[model.feature_names_in_]

    if st.button("🔍 Predict Job Change"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("---")

        if prediction == 1:
            st.error("❌ Employee is **Likely to Change Job**")
        else:
            st.success("✅ Employee is **Likely to Stay**")

        st.write(f"📊 **Probability of Job Change:** `{probability:.2%}`")

        st.subheader("📈 Prediction Probability")
        fig, ax = plt.subplots()
        ax.bar(["Stay", "Change Job"], [1 - probability, probability])
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        st.pyplot(fig)

# -----------------------------
# Page Routing
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "login"

if not st.session_state.logged_in:
    if st.session_state.page == "login":
        login_page()
    else:
        signup_page()
else:
    prediction_page()
