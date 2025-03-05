# Improvised-CSF-Model
This study presents an adaptive model that calibrates CSF sensor values against laboratory data using a combination of machine learning and interpolation techniques. A linear regression model is trained on initial data, supplemented by real-time updates through continuous retraining.
**Overview**
This project calibrates CSF glucose sensor values using lab data with machine learning and interpolation techniques to improve accuracy.

**Features**
Linear regression to map sensor values to lab data.
SciPy’s interp1d for interpolation.
Real-time sensor data acquisition (Serial UART/API).
Adaptive learning with automatic retraining after 10 new readings.
Model persistence with pickle.

**Installation**

**Prerequisites****
Ensure Python (>=3.7) is installed. Install dependencies:
## pip install numpy pandas scikit-learn scipy requests pyserial

**Usage**
Clone the repository:
git clone https://github.com/Adilogan/csf-glucose-calibration.git
# cd csf-glucose-calibration

**Run the script:**
python CSF_lab_improvised.py

**Configure real-time sensor reading:**
use_serial = True  # Enable Serial (UART) sensor
use_api = True  # Enable API-based sensor

The script will fetch data, apply corrections, and retrain dynamically.

**File Structure**

CSF_lab_improvised.py - Main script.
csf_glucose_model.pkl - Saved trained model.
README.md - Documentation.

**Model Workflow**
Train linear regression on initial sensor and lab values.
Apply interpolation for refined predictions.
Fetch and correct real-time sensor readings.
Retrain model after collecting 10 new readings.

**Results**
This approach improves CSF glucose sensor accuracy and reduces lab test dependency.

**Contact**
For inquiries, email adityakita0623@gmail.com
