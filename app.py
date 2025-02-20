import pickle
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
import base64

# Set page title and layout
st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🧑‍⚕")


# Apply Custom Styling for Buttons and Sidebar
custom_style = """
    <style>
    div.stButton > button:first-child {
        background-color: #4CAF50; /* Green for Prediction Button */
        color: white;
        border-radius: 12px;
        font-size: 18px;
        padding: 10px 24px;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #B0E0E6;
    }

    div.stButton > button:nth-child(2) {
        background-color: #B0E0E6; /* Blue for Accuracy Button */
        color: white;
        border-radius: 12px;
        font-size: 18px;
        padding: 10px 24px;
        transition: 0.3s;
    }
    div.stButton > button:nth-child(2):hover {
        background-color: #2196F3;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f0f8ff; /* Light blue background */
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar-text {
        font-size: 20px;
        font-weight: bold;
        color: #333;
    }
    .accuracy-text {
        font-size: 22px;
        font-weight: bold;
        color: #333; /* Orange color for accuracy */
    }
    </style>
"""
st.markdown(custom_style, unsafe_allow_html=True)

st.title('🧑‍⚕ Diabetes Prediction using ML')

# Load trained model safely
diabetes_model_path = r"diabetes_model.sav"

try:
    with open(diabetes_model_path, 'rb') as model_file:
        diabetes_model = pickle.load(model_file)
except FileNotFoundError:
    st.error("Error: Model file not found. Please check the file path.")
    diabetes_model = None

# Input fields
col1, col2, col3 = st.columns(3)

with col1:
    Pregnancies = st.text_input('Number of Pregnancies', value="0")
with col2:
    Glucose = st.text_input('Glucose Level', value="0")
with col3:
    BloodPressure = st.text_input('Blood Pressure Value', value="0")
with col1:
    SkinThickness = st.text_input('Skin Thickness Value', value="0")
with col2:
    Insulin = st.text_input('Insulin Level', value="0")
with col3:
    BMI = st.text_input('BMI Value', value="0")
with col1:
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', value="0")
with col2:
    Age = st.text_input('Age of the Person', value="0")

diab_diagnosis = ""

# Prediction Button (Green)
if st.button('Diabetes Test Result'):
    if diabetes_model is not None:
        try:
            user_input = [
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), float(Age)
            ]

            diab_prediction = diabetes_model.predict([user_input])

            if diab_prediction[0] == 1:
                diab_diagnosis = '⚠️ The person is **Diabetic**.'
                st.error(diab_diagnosis)
            else:
                diab_diagnosis = '✅ The person is **NOT Diabetic**.'
                st.success(diab_diagnosis)

        except ValueError:
            st.error("❌ Please enter valid numerical values for all fields.")
    else:
        st.error("Model is not loaded. Cannot make predictions.")

# Model Accuracy Button (Blue)
if st.button('Show Model Accuracy'):
    try:
        test_data = pd.read_csv(r"diabetes.csv")

        # Ensure 'Outcome' column exists
        if "Outcome" not in test_data.columns:
            st.error("Error: 'Outcome' column missing in dataset.")
        else:
            X_test = test_data.drop(columns=["Outcome"])
            Y_test = test_data["Outcome"]

            Y_pred = diabetes_model.predict(X_test)
            accuracy = accuracy_score(Y_test, Y_pred)

            st.sidebar.header("📊 Model Performance")
            st.sidebar.markdown('<p class="sidebar-text">Test Accuracy:</p>', unsafe_allow_html=True)
            st.sidebar.markdown(f'<p class="accuracy-text">{accuracy:.4f}</p>', unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("Error: Test dataset file not found. Check the file path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
