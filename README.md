🌬️ AirSage
AI-Powered Urban Air Quality Prediction & Pollution Source Mapping System
🧠 Overview

AirSage is an intelligent air-quality analytics system that fuses IoT sensor data, satellite weather inputs, and deep learning models to forecast AQI trends, detect major pollutants, and map urban pollution hotspots in real time.

This version uses a Hybrid CNN–LSTM model optimized with AIS (Artificial Immune System) + HSA (Harmony Search Algorithm) to capture both spatial and temporal dependencies in pollution data while tuning model parameters for the lowest RMSE.

Designed for:

Smart-city deployments

University campus monitoring

Environmental research

Real-time citizen alert dashboards

⚙️ Technical Stack
Layer	Technology
Language	Python 3.11
ML Frameworks	TensorFlow / Keras, Scikit-learn
Optimization Algorithms	AIS + HSA hybrid optimizer
Visualization	Matplotlib, Seaborn
Data Handling	Pandas, NumPy
Storage & Serialization	Joblib, YAML, JSON
Deployment Option	Streamlit / FastAPI (optional)
📂 Directory Structure
Air Quality Prediction and Pollution Source Mapping System/
│
├── archive/
│   └── delhi_aqi.csv                ← Input dataset
│
├── hybrid_AirSage_model.h5          ← Trained CNN-LSTM model
├── hybrid_AirSage_scalers.pkl       ← Saved MinMaxScaler objects
├── hybrid_AirSage_config.yaml       ← Model configuration + parameters
├── hybrid_AirSage_results.json      ← Training metrics summary
├── hybrid_AirSage_final_predictions.csv  ← Predicted AQI values
├── hybrid_AirSage_final_report.json ← Final result report
│
├── hybrid_AirSage_heatmap.png       ← Feature correlation heatmap
├── hybrid_AirSage_accuracy_graph.png← Training & validation loss graph
├── hybrid_AirSage_comparison_graph.png ← Actual vs predicted comparison
├── hybrid_AirSage_result_graph.png  ← RMSE & R² summary bar graph
├── hybrid_AirSage_prediction_graph.png ← AQI forecast trend
│
└── AirSage_Predict.py               ← Final prediction script (you ran)

🧩 How It Works
1️⃣ Data Acquisition

Load urban air quality data (delhi_aqi.csv)

Select relevant pollutants and weather parameters:
PM2.5, PM10, NO₂, SO₂, CO, O₃, Temp, Wind, Humidity

2️⃣ Data Preprocessing

Handle missing data

Normalize using MinMaxScaler

Reshape into time-step format for LSTM input

3️⃣ Hybrid Modeling (AIS + HSA Optimizer)
Algorithm	Purpose
AIS (Artificial Immune System)	Creates clones of best hyperparameter sets and mutates them to find better solutions
HSA (Harmony Search Algorithm)	Fine-tunes cloned solutions using pitch adjustments and harmony memory
CNN-LSTM	Learns spatial correlations + temporal evolution of AQI

Optimized hyperparameters include:

CNN filters

LSTM units

Dropout rate

4️⃣ Model Training

80/20 train-test split

Early stopping for stable convergence

Evaluation using RMSE and R² Score

5️⃣ Visualization
Graph	Description
hybrid_AirSage_heatmap.png	Correlation heatmap of pollutants
hybrid_AirSage_accuracy_graph.png	Training vs Validation Loss
hybrid_AirSage_comparison_graph.png	Actual vs Predicted AQI
hybrid_AirSage_result_graph.png	RMSE & R² bar graph
hybrid_AirSage_prediction_graph.png	AQI forecast trend line
6️⃣ Result Generation

The model automatically generates:

hybrid_AirSage_final_predictions.csv
→ Contains actual AQI, predicted AQI, and error column

hybrid_AirSage_final_report.json
→ Contains model summary, metrics, top pollutant, and sample predictions

📊 Sample Output Summary

Console Output Example

[INFO] Loading model and scalers...
[INFO] Generating predictions...
587/587 [==============================] - 1s 2ms/step
[RESULT] RMSE = 8.84, R² = 0.888
[INFO] Top pollutant contributor: CO (78.50%)
✅ Saved detailed predictions: hybrid_AirSage_final_predictions.csv
✅ Final report saved: hybrid_AirSage_final_report.json

🧩 AIRSAGE HYBRID PREDICTION SUMMARY
---------------------------------------------------
Dataset         : delhi_aqi.csv
Total Samples   : 4380
RMSE            : 8.84
R² Score        : 0.888
Top Pollutant   : CO (78.50%)


JSON Report Preview (hybrid_AirSage_final_report.json):

{
  "Model": "Hybrid AIS + HSA CNN-LSTM",
  "Dataset": "delhi_aqi.csv",
  "Features_Used": ["pm2.5", "pm10", "no2", "so2", "co", "o3", "temp", "wind", "humidity"],
  "Target": "aqi",
  "Results": {
    "RMSE": 8.84,
    "R2_Score": 0.888,
    "Top_Pollutant": "CO",
    "Contribution(%)": 78.5
  },
  "Sample_Predictions": [
    {"Actual_AQI": 102, "Predicted_AQI": 98.7},
    {"Actual_AQI": 114, "Predicted_AQI": 110.1}
  ]
}

🧮 Evaluation Metrics
Metric	Meaning	Ideal
RMSE	Root Mean Square Error – measures prediction deviation	Lower is better
R² Score	Coefficient of Determination – model fit quality	Closer to 1 is better
Top Pollutant	Major AQI influencer (by correlation)	Helps identify source hotspot
🧰 How to Run
🪜 Step 1: Install Dependencies
pip install tensorflow==2.16.1 scikit-learn pandas numpy seaborn matplotlib pyyaml joblib

🪜 Step 2: Train or Use Pretrained Model

You already have the trained hybrid model saved as:

hybrid_AirSage_model.h5
hybrid_AirSage_scalers.pkl

🪜 Step 3: Run Prediction Script
python AirSage_Predict.py

🪜 Step 4: Check Results

📁 hybrid_AirSage_final_predictions.csv → AQI results

📁 hybrid_AirSage_final_report.json → Metrics summary

🖼️ PNG files → Graphs and visualizations

🌍 Real-World Impact

✅ Predicts AQI variations 24 hours in advance
✅ Identifies dominant pollutants dynamically
✅ Helps municipalities plan interventions
✅ Enables citizens to take safety measures
✅ Integrates easily with IoT sensor streams via MQTT / Firebase

🔋 Future Enhancements

🛰️ Add NASA MODIS aerosol & satellite imagery

🤖 Introduce Reinforcement Learning for dynamic pollution control

📱 Deploy as mobile app using Streamlit / FastAPI backend

☀️ Integrate with SolarSense to correlate air quality ↔ solar efficiency

🧑‍💻 Author & Credits

Developed by: Sagnik Patra
Institution: NIAT / NIU, Noida
Mentorship: ChatGPT-5 AI Collaboration
Specialized Domain: AI, IoT, and Urban Analytics
