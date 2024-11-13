import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from PIL import Image

# Cargar y mostrar el icono
icon_path = "../data/iconoDS.png"  # Ajusta la ruta si es necesario
icon = Image.open(icon_path)

# Dividir en dos columnas para colocar la imagen y el título
col1, col2 = st.columns([1, 5])
with col1:
    st.image(icon, width=100)  # Ajusta el tamaño si lo deseas
with col2:
    st.title("ValueTrak : Predicción del Valor del VTI")

# Descripción
st.write("Gráfico que muestra los valores actuales y la predicción para los próximos 6 días")

# Cargar el modelo guardado con pickle
with open("../models/NeuralProphet_default.pkl", "rb") as f:
    m = pickle.load(f)

# Cargar los datos históricos para generar las predicciones
df_indicadores_D = pd.read_csv('../data/processed/total_data.csv')  # Ajusta la ruta según sea necesario
df_indicadores_D = df_indicadores_D.rename(columns={'date': 'ds', 'VTI_value': 'y'})

# Generar el DataFrame futuro con las columnas necesarias
future = m.make_future_dataframe(df_indicadores_D, periods=6, n_historic_predictions=True)

# Asegurarse de que solo las columnas necesarias estén presentes
columns_needed = ['ds', 'y', 'VTI_6_days_ago', 'pct_change_6d', 'TPM_var', 'TDES_var', 'IPC_var', 'BC_var']
future = future[columns_needed].copy()  # Seleccionar solo las columnas que el modelo necesita

# Realizar la predicción
forecast = m.predict(future)

# Asegúrate de que las fechas estén en el formato correcto
df_indicadores_D['ds'] = pd.to_datetime(df_indicadores_D['ds'])
forecast['ds'] = pd.to_datetime(forecast['ds'])

# Divide el DataFrame en histórico y predicción para graficar
test_df = df_indicadores_D[int(0.8 * len(df_indicadores_D)):]  # Ajuste temporal del DataFrame de prueba
actual = forecast[forecast['ds'] < test_df['ds'].iloc[0]]
prediction = forecast[forecast['ds'] >= test_df['ds'].iloc[0]]

# Combina los datos históricos y predicciones en un solo DataFrame para una línea continua
combined_df = pd.concat([actual, prediction])

# Filtra los últimos 6 días de predicción
last_6_days_prediction = prediction.tail(6)

# Visualización en Streamlit
fig, ax = plt.subplots(figsize=(12, 6))
# Línea continua para el histórico y predicción completa
ax.plot(combined_df['ds'], combined_df['yhat1'], color='blue', label='Predicción Continuada')
ax.scatter(combined_df['ds'], combined_df['y'], color='black', marker='o', s=10, label='Datos Reales')
# Línea diferenciada para los últimos 6 días de predicción
ax.plot(last_6_days_prediction['ds'], last_6_days_prediction['yhat1'], color='red', label='Predicción a 6 días')

# Títulos y etiquetas
ax.set_title('Pronóstico Continuo del VTI con NeuralProphet')
ax.set_xlabel('Fecha')
ax.set_ylabel('Valor del VTI')
ax.legend()
ax.grid(True)

# Mostrar la gráfica en Streamlit
st.pyplot(fig)

# Mostrar la tabla de los últimos 6 días de predicción
st.write("### Últimos 6 Días de Predicción")
st.write(last_6_days_prediction[['ds', 'yhat1']].rename(columns={'ds': 'Fecha', 'yhat1': 'Predicción'}))
