import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Configuración de la aplicación Streamlit
st.title("Predicción del Valor del VTI")
st.write("Gráfico que muestra los valores actuales y la predicción para los próximos 6 días")

# Cargar el modelo guardado con pickle
with open("/Users/eleinybellomanzo/Documents/proyecto ds/proyecto_DataScience_DMS/models/NeuralProphet_default.pkl", "rb") as f:
    m = pickle.load(f)

# Cargar los datos históricos para generar las predicciones
df_indicadores_D = pd.read_csv('/Users/eleinybellomanzo/Documents/proyecto ds/proyecto_DataScience_DMS/data/processed/total_data.csv')  # Ajusta la ruta según sea necesario
df_indicadores_D = df_indicadores_D.rename(columns={'date': 'ds', 'VTI_value': 'y'})

# Generar el DataFrame futuro con las columnas necesarias
future = m.make_future_dataframe(df_indicadores_D, periods=6, n_historic_predictions=True)

# Asegurarse de que solo las columnas necesarias estén presentes
columns_needed = ['ds', 'y', 'VTI_6_days_ago', 'pct_change_6d', 'TPM_var', 'TDES_var', 'IPC_var', 'BC_var']
future = future[columns_needed].copy()  # Seleccionar solo las columnas que el modelo necesita

# Realizar la predicción
forecast = m.predict(future)

# Dividir en datos actuales y predicciones
actual = forecast[forecast['ds'] < future['ds'].max() - pd.Timedelta(days=5)]
prediction = forecast[forecast['ds'] >= future['ds'].max() - pd.Timedelta(days=5)]

# Gráfico de predicción
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(actual['ds'], actual['yhat1'], label='Actual', color='blue')
ax.plot(prediction['ds'], prediction['yhat1'], label='Predicción (6 días)', color='orange', linestyle='--')
ax.set_xlabel('Fecha')
ax.set_ylabel('Valor del VTI')
ax.set_title('Predicción del VTI')
ax.legend()

# Mostrar el gráfico en Streamlit
st.pyplot(fig)

# Mostrar la tabla con los valores de predicción
st.write("Tabla de Valores de Predicción")
st.write(forecast.tail(6))  # Muestra solo los últimos 6 días de predicción

