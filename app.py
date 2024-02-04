import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load the pre-trained model and scaler
loaded_model = pickle.load(open('bank_churn_model.pkl', 'rb'))
loaded_scaler = pickle.load(open('scaler.pkl', 'rb'))

# Function to preprocess input data
def preprocess_input_data(input_data, scaler):
    input_df = pd.DataFrame([input_data], columns=feature_names)
    input_data_scaled = scaler.transform(input_df)
    return input_data_scaled

# Function to predict churn
def predict_churn(input_data):
    prediction = loaded_model.predict(input_data)
    return prediction

# Define feature names
feature_names = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
                 'Geography_Germany', 'Geography_Spain', 'Gender_Male', 'HasCrCard_1', 'IsActiveMember_1']

# Streamlit web app
st.title("Bank Churn Prediction App")

# Input form for user to provide data
st.sidebar.header("User Input")
credit_score = st.sidebar.slider("Credit Score", 300, 850, 500)
age = st.sidebar.slider("Age", 18, 100, 30)
tenure = st.sidebar.slider("Tenure", 0, 20, 5)
balance = st.sidebar.slider("Balance", 0, 250000, 50000)
num_of_products = st.sidebar.slider("Number of Products", 1, 4, 2)
estimated_salary = st.sidebar.slider("Estimated Salary", 0, 250000, 100000)
geography = st.sidebar.selectbox("Geography", ['France', 'Spain', 'Germany'])
gender = st.sidebar.radio("Gender", ['Male', 'Female'])
has_cr_card = st.sidebar.checkbox("Has Credit Card")
is_active_member = st.sidebar.checkbox("Is Active Member")

# Process input data
input_data = [credit_score, age, tenure, balance, num_of_products, estimated_salary]

if geography == 'France':
    input_data.extend([1, 0])
elif geography == 'Spain':
    input_data.extend([0, 1])
else:
    input_data.extend([0, 0])

if gender == 'Male':
    input_data.append(1)
else:
    input_data.append(0)

input_data.extend([1 if has_cr_card else 0, 1 if is_active_member else 0])

# Preprocess input data
input_data_processed = preprocess_input_data(input_data, loaded_scaler)

# Display processed input data
st.write("Processed Input Data:")
st.write(pd.DataFrame(input_data_processed, columns=feature_names))

# Make prediction
prediction = predict_churn(input_data_processed)

# Display result
st.subheader("Prediction:")
if prediction[0] == 0:
    st.success('Customer will not churn.')
else:
    st.error('Customer will churn.')