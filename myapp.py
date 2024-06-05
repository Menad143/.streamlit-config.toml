import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model, scaler, and columns
with open('logistic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
with open('columns.pkl', 'rb') as columns_file:
    X_columns = pickle.load(columns_file)

# Define a function for prediction
def predict_survival(input_data):
    input_data = pd.get_dummies(input_data, columns=['Pclass'], drop_first=True)
    missing_cols = set(X_columns) - set(input_data.columns)
    for c in missing_cols:
        input_data[c] = 0
    input_data = input_data[X_columns]
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction[0]

# Streamlit app
st.title("Titanic Survival Prediction")

# Input fields
Pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
Age = st.slider("Age", 0, 80, 30)
SibSp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
Parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
Fare = st.slider("Fare", 0, 500, 50)

# Create a DataFrame for input
input_data = pd.DataFrame({
    'Pclass': [Pclass],
    'Age': [Age],
    'SibSp': [SibSp],
    'Parch': [Parch],
    'Fare': [Fare]
})

# Predict button
if st.button("Predict"):
    result = predict_survival(input_data)
    if result == 1:
        st.success("The passenger is predicted to survive.")
    else:
        st.error("The passenger is predicted to not survive.")
