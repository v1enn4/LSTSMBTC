# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# ------------------------------
# Konfigurasi Streamlit
# ------------------------------
st.set_page_config(page_title="BTC Price Prediction", layout="wide")
st.title("📊 Prediksi Harga Bitcoin (BTC) dengan LSTM dan Sentimen")

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("btc_final_dataset.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# ------------------------------
# Load Model
# ------------------------------
model = load_model("model/lstm_btc_model.h5")

# ------------------------------
# Preprocessing Data
# ------------------------------
features = ['Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_7', 'SMA_21', 'EMA_7', 'Daily_Return', 'RSI_14', 'Accurate Sentiments']

close_idx = features.index("Close")
lookback = 60

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

X_test = []
for i in range(lookback, len(scaled_data)):
    X_test.append(scaled_data[i - lookback:i])
X_test = np.array(X_test)

# ------------------------------
# Prediksi
# ------------------------------
y_pred = model.predict(X_test)

# Denormalisasi hanya untuk kolom 'Close'
dummy = np.zeros((len(y_pred), len(features)))
dummy[:, close_idx] = y_pred.flatten()
predicted_close = scaler.inverse_transform(dummy)[:, close_idx]

actual_close = df['Close'].values[lookback:]
dates = df['Date'].values[lookback:]

# ------------------------------
# Evaluasi Metrik
# ------------------------------
mae = mean_absolute_error(actual_close, predicted_close)
rmse = np.sqrt(mean_squared_error(actual_close, predicted_close))
r2 = r2_score(actual_close, predicted_close)

# ------------------------------
# Plot Harga Aktual vs Prediksi
# ------------------------------
st.subheader("📈 Harga BTC: Aktual vs Prediksi")
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(dates, actual_close, label="Actual Close")
ax.plot(dates, predicted_close, label="Predicted Close")
ax.set_xlabel("Tanggal")
ax.set_ylabel("Harga BTC (USD)")
ax.legend()
st.pyplot(fig)

# ------------------------------
# Tampilkan Metrik
# ------------------------------
st.subheader("📊 Evaluasi Model")
col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("R² Score", f"{r2:.4f}")
