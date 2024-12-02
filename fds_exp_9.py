import streamlit as st
import numpy as np
import joblib

#Now you can load the model from the file
model = joblib.load('logistic_regression_model.pkl')

# Streamlit app
st.title("Cancer Detection App")
st.markdown("Predict if the cancer is **Benign (0)** or **Malignant (1)** based on clinical data.")
# Input fields
age = st.number_input("Age")
SMOKING = st.number_input("SMOKING")
COUGHING = st.number_input("COUGHING")
SHORTNESS_OF_BREATH = st.number_input("SHORTNESS OF BREATH")
SWALLOWING_DIFFICULTY = st.number_input("SWALLOWING DIFFICULTY")
CHEST_PAIN = st.number_input("CHEST PAIN")

# Predict button
if st.button("Predict"):
    # Indent the following lines within the 'if' block
    input_data = np.array([[age, SMOKING, COUGHING, SHORTNESS OF BREATH, SWALLOWING DIFFICULTY, CHEST PAIN]])
    prediction = model.predict(input_data)[0]
    st.write("Prediction: Malignant" if prediction == 1 else "Prediction: Benign")
