import numpy as np
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

model = load_model('Churn_Model.h5')

with open('geography_encoder.pkl', 'rb') as f:
    geography_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('gender_encoder.pkl', 'rb') as f:
    gender_encoder = pickle.load(f)

def preprocess_and_predict(input_data):
    credit_score, geography, gender, age, tenure, balance, num_products, has_cr_card, is_active_member, estimated_salary = input_data

    gender = gender_encoder.transform([gender])[0]  
    has_cr_card = 1 if has_cr_card == 'Yes' else 0
    is_active_member = 1 if is_active_member == 'Yes' else 0

    geo_encoded = geography_encoder.transform([[geography]]).toarray()

    numeric_features = np.array([[credit_score, age, tenure, balance, num_products, estimated_salary]])
    scaled_numeric = scaler.transform(numeric_features)

    final_input = np.concatenate([
        scaled_numeric,
        [[gender, has_cr_card, is_active_member]],
        geo_encoded
    ], axis=1)

    prediction = model.predict(final_input)[0][0]
    return prediction

st.title("Customer Churn Prediction Dashboard")
st.write("Enter customer details below to predict if they'll churn:")

credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])
gender = st.radio("Gender", ['Male', 'Female'])
age = st.slider("Age", min_value=18, max_value=100, value=35)
tenure = st.slider("Tenure (Years with Bank)", min_value=0, max_value=10, value=3)
balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.radio("Has Credit Card?", ['Yes', 'No'])
is_active_member = st.radio("Is Active Member?", ['Yes', 'No'])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0)

if st.button("Predict Churn"):
    input_data = [
        credit_score, geography, gender, age, tenure, balance, 
        num_products, has_cr_card, is_active_member, estimated_salary
    ]
    
    prediction_prob = preprocess_and_predict(input_data)
    
    # Modified logic based on 70% threshold
    if prediction_prob < 0.7:  # Below 70% probability means will churn
        st.error(f"🚨 Customer will CHURN (Churn Probability: {prediction_prob*100:.2f}%)")
        st.write("Recommended action: Immediate retention intervention needed!")
    else:  # 70% or above means will not churn
        st.success(f"✅ Customer will NOT churn (Retention Probability: {(1-prediction_prob)*100:.2f}%)")
        st.write("No immediate action required")