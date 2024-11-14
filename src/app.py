import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from PIL import Image

# Cargar y mostrar el icono
icon_path = "iconoDS.png"  # Ajusta la ruta si es necesario
icon = Image.open(icon_path)

# Dividir en dos columnas para colocar la imagen y el título
col1, col2 = st.columns([1, 5])
with col1:
    st.image(icon, width=100)  # Ajusta el tamaño si lo deseas
with col2:
    st.title("ValueTrak : Predicción del Valor del VTI")

# Descripción
st.write("Gráfico que muestra los valores actuales y la predicción para los próximos 5 días")

# Cargar los datos históricos para generar las predicciones
df_indicadores_D = pd.read_csv('total_data.csv')  # Ajusta la ruta según sea necesario
df_indicadores_D = df_indicadores_D.rename(columns={'date': 'ds', 'VTI_value': 'y'})

# Separar en datos de entrenamiento y prueba
train_size = int(len(df_indicadores_D) * 0.8)  # 80% entrenamiento, 20% prueba
train_df = df_indicadores_D[:train_size]
test_df = df_indicadores_D[train_size:]

# Entrenar el modelo Prophet
m1 = Prophet()
m1.fit(train_df)

# Crear el DataFrame futuro y realizar la predicción
future1 = m1.make_future_dataframe(periods=6)
forecast1 = m1.predict(future1)

# Filtra los últimos 5 días de predicción
last_5_days_prediction = forecast1.tail(5)

# Visualización en Streamlit - Gráfico completo con datos históricos
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(forecast1['ds'], forecast1['yhat'], color='blue', label='Predicción Continuada')
ax.scatter(train_df['ds'], train_df['y'], color='black', marker='o', s=10, label='Datos Reales')
ax.plot(last_5_days_prediction['ds'], last_5_days_prediction['yhat'], color='red', label='Predicción a 5 días')
ax.set_title('Pronóstico Continuo del VTI con Prophet')
ax.set_xlabel('Fecha')
ax.set_ylabel('Valor del VTI')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Segunda gráfica enfocada en los últimos 5 días de predicción
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(last_5_days_prediction['ds'], last_5_days_prediction['yhat'], color='red', marker='o', label='Predicción a 5 días')
ax2.set_title('Zoom en la Predicción de los Últimos 5 Días del VTI')
ax2.set_xlabel('Fecha')
ax2.set_ylabel('Valor del VTI')
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# Mostrar la tabla de los últimos 5 días de predicción
st.write("### Últimos 5 Días de Predicción")
st.write(last_5_days_prediction[['ds', 'yhat']].rename(columns={'ds': 'Fecha', 'yhat': 'Predicción'}))
