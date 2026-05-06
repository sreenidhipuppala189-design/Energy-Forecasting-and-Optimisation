# streamlit_app.py
import streamlit as st
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime

# --------------------------------------------------
# PAGE CONFIG 
# --------------------------------------------------
st.set_page_config(page_title="Forecast + RL Energy Demo", layout="wide")
st.title("Forecast Tomorrow's Electricity + RL Optimization (Demo)")

# --------------------------------------------------
# REAL-TIME HOURLY UPDATE LOGIC 
# --------------------------------------------------
current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)

if "last_update_hour" not in st.session_state:
    st.session_state.last_update_hour = current_hour

# --------------------------------------------------
# 1) Synthetic data generator 
# --------------------------------------------------
@st.cache_data
def create_initial_data(days=120, seed=0):
    np.random.seed(seed)
    hours = days * 24
    t = np.arange(hours)

    daily = 2000 + 800 * np.sin(2 * np.pi * (t % 24) / 24 - 0.5)
    weekly = 150 * np.sin(2 * np.pi * (t % (24 * 7)) / (24 * 7))
    noise = 150 * np.random.randn(hours)

    ghi = np.maximum(0, 6 + 3 * np.sin(2 * np.pi * (t % 24) / 24 - 1))
    wind = np.maximum(0, 4 + 2 * np.sin(2 * np.pi * (t % 24) / 24 + 1))

    load = daily + weekly + noise - (ghi * 50) - (wind * 20)
    load = np.maximum(500, load)

    idx = pd.date_range(
        end=pd.Timestamp.now().floor("H") - pd.Timedelta(hours=1),
        periods=hours,
        freq="H"
    )

    df = pd.DataFrame(
        {"load": load, "ghi": ghi, "wind": wind},
        index=idx
    )
    df.index.name = "timestamp"
    return df


def append_new_hour(df):
    last_time = df.index[-1] + pd.Timedelta(hours=1)
    t = len(df)

    load = 2000 + 800 * np.sin(2 * np.pi * (t % 24) / 24 - 0.5) + np.random.randn() * 150
    ghi = max(0, 6 + 3 * np.sin(2 * np.pi * (t % 24) / 24 - 1))
    wind = max(0, 4 + 2 * np.sin(2 * np.pi * (t % 24) / 24 + 1))

    new_row = pd.DataFrame(
        {"load": [load], "ghi": [ghi], "wind": [wind]},
        index=[last_time]
    )
    return pd.concat([df, new_row])


# --------------------------------------------------
# DATA STATE 
# --------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = create_initial_data()

# Update data only when hour changes
if current_hour > st.session_state.last_update_hour:
    st.session_state.df = append_new_hour(st.session_state.df)
    st.session_state.last_update_hour = current_hour
    st.experimental_rerun()

df = st.session_state.df
df.to_csv("energy_data.csv")
# --------------------------------------------------
# SIDEBAR 
# --------------------------------------------------
st.sidebar.markdown("### Data")
st.sidebar.write("Synthetic hourly data (replace with your CSV). Rows:", df.shape[0])

st.sidebar.markdown("### Model settings")
past_days = st.sidebar.number_input("Past days (LSTM input window)", value=10, min_value=1, max_value=30)
forecast_horizon = st.sidebar.number_input("Forecast horizon (hours)", value=24, min_value=1, max_value=72)
retrain_every_run = st.sidebar.checkbox("Retrain LSTM now (uses last N days)", value=True)

past_hours = past_days * 24

# --------------------------------------------------
# PREPROCESSING 
# --------------------------------------------------
features = ["load", "ghi", "wind"]
scaler_X = StandardScaler()
scaler_y = StandardScaler()

def make_sequences(df, past_hours, horizon):
    X, y = [], []
    arr = df[features].values
    for i in range(past_hours, len(df) - horizon):
        X.append(arr[i - past_hours:i])
        y.append(arr[i:i + horizon, 0])
    return np.array(X), np.array(y)

X, y = make_sequences(df, past_hours, forecast_horizon)

ns, nt, nf = X.shape
X_scaled = scaler_X.fit_transform(X.reshape(-1, nf)).reshape(ns, nt, nf)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).reshape(y.shape)

# --------------------------------------------------
# LSTM MODEL 
# --------------------------------------------------
def build_lstm():
    model = Sequential([
        LSTM(128, input_shape=(past_hours, nf)),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(forecast_horizon)
    ])
    model.compile(optimizer="adam", loss="mae")
    return model

model = build_lstm()

if retrain_every_run:
    with st.spinner("Training LSTM..."):
        hist = model.fit(
            X_scaled, y_scaled,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            callbacks=[EarlyStopping(patience=3)],
            verbose=0
        )
    st.success("LSTM trained")

    fig, ax = plt.subplots(figsize=(5, 3))
    
    ax.plot(hist.history["loss"], label="train loss")
    ax.plot(hist.history["val_loss"], label="val loss")
    ax.set_xlabel("Epochs")     # X-axis
    ax.set_ylabel("Loss (MAE)")
    ax.legend()
    ax.set_title("Training Loss (MAE scaled)")
    st.pyplot(fig)

# --------------------------------------------------
# FORECAST 
# --------------------------------------------------
latest = df[features].iloc[-past_hours:].values
latest_scaled = scaler_X.transform(latest).reshape(1, past_hours, nf)
pred = model.predict(latest_scaled)
pred = scaler_y.inverse_transform(pred.reshape(-1, 1)).flatten()

pred_index = pd.date_range(
    start=df.index[-1] + pd.Timedelta(hours=1),
    periods=forecast_horizon,
    freq="H"
)

forecast_df = pd.DataFrame({"pred_load": pred}, index=pred_index)

st.subheader("Forecast for next 24 hours")
#st.line_chart(pd.concat([df["load"].tail(72), forecast_df["pred_load"]]))

fig, ax = plt.subplots(figsize=(8, 4))

# Combine actual + forecast
combined = pd.concat([df["load"].tail(72), forecast_df["pred_load"]])

# Plot
ax.plot(combined.index, combined.values)

# Axis labels (THIS is what you want)
ax.set_xlabel("Time (Hours)")
ax.set_ylabel("Load")

# Optional improvements
ax.set_title("Load Forecast (Actual + Predicted)")
ax.grid(True)

st.pyplot(fig)


# --------------------------------------------------
# RL ENV + PPO 
# --------------------------------------------------
class EnergyEnv(gym.Env):
    """Simple battery dispatch environment for RL optimization."""
    metadata = {"render.modes": []}

    def __init__(self, forecast, actual_future, battery_capacity_kwh=100.0, max_charge_kw=50.0, dt_hours=1.0):
        super().__init__()
        self.forecast = np.array(forecast, dtype=np.float32)
        self.actual = np.array(actual_future, dtype=np.float32)
        self.horizon = len(self.forecast)
        self.batt_cap = battery_capacity_kwh
        self.max_charge = max_charge_kw
        self.dt = dt_hours

        # Observation = [forecast values for horizon, SOC fraction]
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6, shape=(self.horizon + 1,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.t = 0
        self.soc = self.batt_cap / 2

    def _get_obs(self):
        """Pad remaining forecast so obs length stays constant."""
        remaining = self.forecast[self.t:]
        pad_len = self.horizon - len(remaining)
        padded_forecast = np.pad(remaining, (0, pad_len), mode="constant")
        obs = np.concatenate([padded_forecast, [self.soc / self.batt_cap]])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.soc = self.batt_cap / 2
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        act = float(action[0])
        power = np.clip(act * self.max_charge, -self.max_charge, self.max_charge)
        energy_delta = power * self.dt
        self.soc = np.clip(self.soc + energy_delta, 0, self.batt_cap)

        demand = self.actual[self.t]
        grid_draw = demand + max(0, power) - min(0, power)
        reward = -grid_draw / 1000.0  # minimize grid usage

        self.t += 1
        done = self.t >= self.horizon
        truncated = False
        obs = self._get_obs()
        info = {"grid_draw": grid_draw, "soc": self.soc, "power": power}
        return obs, reward, done, truncated, info

# Build environment using predicted forecast and pretend "actual" (demo uses pred as actual)
env = EnergyEnv(forecast=forecast_df['pred_load'].values, actual_future=forecast_df['pred_load'].values,
                battery_capacity_kwh=200.0, max_charge_kw=50.0)

st.subheader("RL optimization (demo)")
st.write("We run a short PPO training on the schedule (fast demo). In production, train properly or load saved model.")

# Wrap gym env for SB3
from stable_baselines3.common.env_util import DummyVecEnv
vec_env = DummyVecEnv([lambda: env])

# Train a tiny PPO for demo purposes
model_rl = PPO("MlpPolicy", vec_env, verbose=0)
with st.spinner("Training PPO agent (demo, quick)..."):
    model_rl.learn(total_timesteps=2000)
st.success("PPO trained (demo)")

# Run policy to get schedule
# Run policy to get schedule
obs, _ = env.reset()   # ✅ Unpack (obs, info)
actions, socs, grid_draws, powers = [], [], [], []
done = False
while not done:
    action, _ = model_rl.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)  # ✅ Unpack truncated per Gymnasium API
    actions.append(action[0])
    socs.append(info['soc'])
    grid_draws.append(info['grid_draw'])
    powers.append(info['power'])


# Prepare schedule df
schedule_idx = forecast_df.index[:len(actions)]
schedule = pd.DataFrame({'action': actions, 'power_kW': powers, 'soc_kWh': socs, 'grid_draw_kW': grid_draws}, index=schedule_idx)

st.write("RL schedule (first rows):")
st.dataframe(schedule.head())

# Plot forecast vs RL results
fig2, ax2 = plt.subplots(3,1, figsize=(9,8), sharex=True)
ax2[0].plot(forecast_df.index, forecast_df['pred_load'], label='Forecast load')
ax2[0].set_ylabel("kW")
ax2[0].legend()
ax2[1].plot(schedule.index, schedule['power_kW'], label='Battery power ( +charge / -discharge )')
ax2[1].set_ylabel("kW")
ax2[1].legend()
ax2[2].plot(schedule.index, schedule['grid_draw_kW'], label='Grid draw (kW)')
ax2[2].set_ylabel("kW")
ax2[2].set_xlabel("time stamp")
ax2[2].legend()
plt.tight_layout()
st.pyplot(fig2)

