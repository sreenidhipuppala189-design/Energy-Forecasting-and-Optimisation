# realTime_controllerex.py
import time
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from stable_baselines3 import PPO
from sklearn.preprocessing import StandardScaler

#################################
# Load models
#################################
lstm_model = load_model("lstm_model.h5")
rl_agent = PPO.load("ppo_energy_policy")

#################################
# Config
#################################
PAST_HOURS = 240
features = ["load", "ghi", "wind"]
battery_soc = 100.0

scaler_X = StandardScaler()
scaler_y = StandardScaler()

#################################
# Fake real-time data source
#################################
def load_latest_data():
    return pd.read_csv("smart_meter.csv", parse_dates=["timestamp"])

#################################
# Forecast function
#################################
def predict_next_24h(df):
    data = df[features].tail(PAST_HOURS).values
    data = data.reshape(1, PAST_HOURS, 3)
    pred = lstm_model.predict(data)
    return pred.flatten()

#################################
# Real-time loop
#################################
print("Real-time controller started...")

while True:
    df = load_latest_data()

    forecast = predict_next_24h(df)

    obs = np.concatenate([forecast, [battery_soc]])
    action, _ = rl_agent.predict(obs, deterministic=True)

    power = action[0] * 10
    battery_soc = np.clip(battery_soc + power, 0, 200)

    print(f"Action={power:.2f} kW | Battery SOC={battery_soc:.2f}")

    time.sleep(3600)  # 1 hour

