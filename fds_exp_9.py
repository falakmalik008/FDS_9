import streamlit as st
import numpy as np
import joblib

# Save the trained model to a file
joblib.dump(model, 'logistic_regression_model.pkl')

# Now you can load the model from the file
model = joblib.load('logistic_regression_model.pkl')

# Streamlit app
st.title("Cancer Detection App")
st.markdown("Predict if the cancer is **Benign (0)** or **Malignant (1)** based on clinical data.")
# Input fields
age = st.number_input("Age")
tumor_size = st.number_input("Tumor Size")
biomarker = st.number_input("Biomarker Level")

# Predict button
if st.button("Predict"):
    # Indent the following lines within the 'if' block
    input_data = np.array([[age, tumor_size, biomarker]])
    prediction = model.predict(input_data)[0]
    st.write("Prediction: Malignant" if prediction == 1 else "Prediction: Benign")
