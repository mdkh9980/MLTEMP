import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load model metrics and models
model_r2_scores = joblib.load('model_r2_scores.joblib')  # Load the R² scores
model_mae = joblib.load('model_mae_scores.joblib')       # Load MAE scores
model_rmse = joblib.load('model_rmse_scores.joblib')     # Load RMSE scores

# Load the best model based on the highest R² score
best_model_name = max(model_r2_scores, key=model_r2_scores.get)  # Find the best model based on R² score
best_model_r2_score = model_r2_scores[best_model_name]
best_model_mae = model_mae[best_model_name]
best_model_rmse = model_rmse[best_model_name]
best_model = joblib.load(f'{best_model_name}_model.joblib')

# Streamlit interface
st.title("Temperature Prediction App")
st.write("This app predicts temperature based on weather conditions.")

# User inputs
def user_input():
    humidity = st.number_input("Humidity", min_value=0.0, max_value=1.0, step=0.01)
    wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, step=0.1)
    wind_bearing = st.number_input("Wind Bearing (degrees)", min_value=0.0, max_value=360.0, step=1.0)
    visibility = st.number_input("Visibility (km)", min_value=0.0, step=0.1)
    pressure = st.number_input("Pressure (millibars)", min_value=800.0, step=0.1)
    summary = st.selectbox("Weather Summary", ['Partly Cloudy', 'Mostly Cloudy', 'Clear', 'Overcast', 'Rain'])
    precip_type = st.selectbox("Precipitation Type", ['rain', 'snow'])

    data = {
        'Humidity': humidity,
        'Wind Speed (km/h)': wind_speed,
        'Wind Bearing (degrees)': wind_bearing,
        'Visibility (km)': visibility,
        'Pressure (millibars)': pressure,
        'Summary': summary,
        'Precip Type': precip_type
    }
    return pd.DataFrame([data])

input_df = user_input()
st.write("Input Data:", input_df)

# Prediction and comparison
if st.button("Predict"):
    # Simulate a true target value (optional)
    true_value = st.number_input("True Temperature (optional, for accuracy metrics)", value=15.0, step=0.1)

    # Get predictions from the best model
    prediction = best_model.predict(input_df)[0]

    # Compute metrics for the best model
    mae = mean_absolute_error([true_value], [prediction])
    rmse = mean_squared_error([true_value], [prediction], squared=False)
    r2 = r2_score([true_value], [prediction])

    # Display the best model's metrics
    st.write(f"### 🏆 Best Performing Model: {best_model_name}")
    st.write(f"R²: {best_model_r2_score:.2f}, MAE: {best_model_mae:.2f}, RMSE: {best_model_rmse:.2f}")

    # Display the prediction
    st.write(f"### Predicted Temperature: {prediction:.2f}°C")

    # Plot R² score comparison
    st.write("### R² Score Comparison")
    fig_r2, ax_r2 = plt.subplots()
    ax_r2.bar(model_r2_scores.keys(), model_r2_scores.values(), color=['blue', 'green', 'orange', 'red', 'purple'])
    ax_r2.set_ylabel("R² Score")
    ax_r2.set_title("Model R² Score Comparison")
    st.pyplot(fig_r2)

    # Plot MAE comparison
    st.write("### MAE Comparison")
    fig_mae, ax_mae = plt.subplots()
    ax_mae.bar(model_mae.keys(), model_mae.values(), color=['blue', 'green', 'orange', 'red', 'purple'])
    ax_mae.set_ylabel("MAE")
    ax_mae.set_title("Model MAE Comparison")
    st.pyplot(fig_mae)

    # Plot RMSE comparison
    st.write("### RMSE Comparison")
    fig_rmse, ax_rmse = plt.subplots()
    ax_rmse.bar(model_rmse.keys(), model_rmse.values(), color=['blue', 'green', 'orange', 'red', 'purple'])
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_title("Model RMSE Comparison")
    st.pyplot(fig_rmse)