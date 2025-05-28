import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# Configurações iniciais
st.set_page_config(page_title="Previsão de Ações com LSTM", layout="centered")
st.title("📈 Previsão de Preço de Ações com LSTM")
st.write("Faça upload de um arquivo .xlsx contendo uma coluna com os últimos preços (janela de 60 valores).")

# Upload de arquivo
uploaded_file = st.file_uploader("Envie seu arquivo Excel", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        if df.shape[0] < 60:
            st.warning("O arquivo precisa conter pelo menos 60 valores para a previsão.")
        else:
            prices = df.iloc[:, 0].values[-60:]  # usa os últimos 60 preços
            st.line_chart(prices, height=200)

            if st.button("🔮 Prever preço do próximo dia"):
                scaler = joblib.load("models/scaler.pkl")
                model = load_model("models/best_model.h5")

                sequence = scaler.transform(prices.reshape(-1, 1))
                model_input = np.reshape(sequence, (1, 60, 1))
                predicted = model.predict(model_input)
                predicted_price = scaler.inverse_transform(predicted)[0][0]

                st.success(f"📌 Preço previsto para o próximo dia: *${predicted_price:.2f}*")

                # Avaliação (usando o último valor como comparação)
                real_price = prices[-1]
                mae = mean_absolute_error([real_price], [predicted_price])
                rmse = math.sqrt(mean_squared_error([real_price], [predicted_price]))
                mape = np.mean(np.abs((real_price - predicted_price) / real_price)) * 100

                # Mostrar métricas
                st.subheader("📊 Indicadores de Eficiência")
                st.write(f"*MAE*: {mae:.2f}")
                st.write(f"*RMSE*: {rmse:.2f}")
                st.write(f"*MAPE*: {mape:.2f}%")

                # Gráfico real vs previsto
                st.subheader("📉 Comparação: Real vs. Previsto")
                fig, ax = plt.subplots()
                ax.plot([59], [real_price], 'bo', label='Real')
                ax.plot([60], [predicted_price], 'ro', label='Previsto')
                ax.legend()
                st.pyplot(fig)

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")