import streamlit as st
import numpy as np
import joblib

joblib.dump(model, 'logistic_regression_model.pkl')

model = joblib.load('logistic_regression_model.pkl')

st.title("Cancer Detection App")
st.markdown("Predict if the cancer is **Benign (0)** or **Malignant (1)** based on clinical data.")

age = st.number_input("Age")
tumor_size = st.number_input("Tumor Size")
biomarker = st.number_input("Biomarker Level")


if st.button("Predict"):
    # Indent the following lines within the 'if' block
    input_data = np.array([[age, tumor_size, biomarker]])
    prediction = model.predict(input_data)[0]
    st.write("Prediction: Malignant" if prediction == 1 else "Prediction: Benign")
