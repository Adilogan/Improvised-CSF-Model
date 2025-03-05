import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
import pickle
import time
import serial  # For Serial (UART) sensor
import requests  # For API-based sensor

# ----------------------- CONFIGURATION -----------------------
use_serial = False  # Set to True if using a Serial (UART) sensor
use_api = False  # Set to True if using an API-based sensor
SERIAL_PORT = "/dev/ttyUSB0"  # Change this for your device (Windows: "COM3")
BAUD_RATE = 9600
API_URL = "http://sensor-api.com/get_glucose"  # Replace with actual API
RETRAIN_THRESHOLD = 10  # Retrain after every 10 new readings
# ------------------------------------------------------------

# Step 1: Prepare the initial dataset
data = {
    "Glucose_Concentration": [50, 75, 100, 125, 150, 175, 200, 225, 250, 275],
    "Sensor_Value": [40, 40, 45, 72, 97, 125, 131, 179, 195, 218],
    "Lab_Value": [50, 75, 100, 125, 150, 175, 200, 225, 250, 275]  # True values
}

df = pd.DataFrame(data)

# Step 2: Train the initial model
X = df["Sensor_Value"].values.reshape(-1, 1)
y = df["Lab_Value"].values

model = LinearRegression()
model.fit(X, y)

# Step 3: Create interpolation function
interpolator = interp1d(df["Sensor_Value"], df["Lab_Value"], kind='linear', fill_value='extrapolate')

# Step 4: Save the trained model
with open("csf_glucose_model.pkl", "wb") as file:
    pickle.dump((model, interpolator, df), file)

# Load model to avoid repeated file I/O
with open("csf_glucose_model.pkl", "rb") as file:
    loaded_model, loaded_interpolator, loaded_df = pickle.load(file)

# Buffer to store new real-time data before retraining
real_time_data = []

# Function to predict corrected glucose value
def predict_glucose(sensor_value):
    corrected_value = float(loaded_interpolator(sensor_value))  # Use interpolation
    return round(corrected_value, 2)

# Function to retrain model with new data
def retrain_model():
    global loaded_model, loaded_interpolator, loaded_df, real_time_data

    if len(real_time_data) >= RETRAIN_THRESHOLD:
        new_df = pd.DataFrame(real_time_data, columns=["Sensor_Value", "Lab_Value"])
        
        # Merge old and new data
        updated_df = pd.concat([loaded_df, new_df]).drop_duplicates().sort_values("Sensor_Value")

        # Train new model
        X_new = updated_df["Sensor_Value"].values.reshape(-1, 1)
        y_new = updated_df["Lab_Value"].values

        new_model = LinearRegression()
        new_model.fit(X_new, y_new)

        # Update interpolation function
        new_interpolator = interp1d(updated_df["Sensor_Value"], updated_df["Lab_Value"], kind='linear', fill_value='extrapolate')

        # Save updated model
        with open("csf_glucose_model.pkl", "wb") as file:
            pickle.dump((new_model, new_interpolator, updated_df), file)

        # Load new model
        with open("csf_glucose_model.pkl", "rb") as file:
            loaded_model, loaded_interpolator, loaded_df = pickle.load(file)

        print("\nModel Retrained Successfully ✅\n")

        # Clear buffer
        real_time_data.clear()

# Function to read sensor data from Serial Port (UART)
def get_sensor_data_from_serial():
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            sensor_value = ser.readline().decode().strip()  # Read and decode sensor value
            return float(sensor_value)  # Convert to float
    except Exception as e:
        print("Error reading sensor (Serial):", e)
        return None  # Handle errors gracefully

# Function to read sensor data from API
def get_sensor_data_from_api():
    try:
        response = requests.get(API_URL)
        data = response.json()  # Assume JSON response
        return float(data['sensor_value'])  # Extract sensor reading
    except Exception as e:
        print("Error fetching data (API):", e)
        return None

# Function to fetch real-time sensor data (based on selected method)
def get_real_time_sensor_reading():
    if use_serial:
        return get_sensor_data_from_serial()
    elif use_api:
        return get_sensor_data_from_api()
    else:
        print("Error: No data source selected. Please set 'use_serial' or 'use_api' to True.")
        return None

# Real-time data processing loop
print("Reading real-time sensor data...\nPress Ctrl+C to stop.")

try:
    while True:
        sensor_value = get_real_time_sensor_reading()
        
        if sensor_value is not None:
            corrected_value = predict_glucose(sensor_value)

            print(f"Sensor Value: {sensor_value:.2f}, Corrected Glucose Value: {corrected_value}")

            # Store new real-time readings
            real_time_data.append([sensor_value, corrected_value])

            # Retrain model if needed
            retrain_model()

        time.sleep(10)  # Adjust delay as needed- it means it will fetch reading from the sensor after these seconds

except KeyboardInterrupt:
    print("\nReal-time data reading stopped.")
