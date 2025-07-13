import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏡 House Price Prediction App")

# Use same feature names as in training
lot_area = st.number_input("Lot Area (in sqft):", min_value=500, max_value=20000, step=100)
bedrooms = st.number_input("Bedrooms Above Ground:", min_value=0, max_value=10, step=1)
bathrooms = st.number_input("Full Bathrooms:", min_value=0, max_value=5, step=1)
rooms = st.number_input("Total Rooms Above Ground:", min_value=1, max_value=15, step=1)

# Construct dataframe with correct feature names
input_df = pd.DataFrame([[lot_area, bedrooms, bathrooms, rooms]],
                        columns=['LotArea', 'BedroomAbvGr', 'FullBath', 'TotRmsAbvGrd'])

# Scale inputs
scaled_input = scaler.transform(input_df)

# Predict
if st.button("Predict Price"):
    prediction = model.predict(scaled_input)
    st.success(f"🏷 Estimated House Price: ₹{prediction[0]:,.2f}")
