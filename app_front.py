import streamlit as st
import requests

# Configuración de la página
st.set_page_config(page_title='Predicción de Diabetes', page_icon='🩺', layout='wide')

# Estilos CSS
primary = '#0b6fb6'
secondary = '#e9f4fb'
st.markdown(f"""
<style>
.main {{background-color: {secondary}; padding: 20px 40px;}}
.stApp > header {{background-color: white;}}
.card {{background: white; padding: 20px; border-radius: 10px;
       box-shadow: 0 2px 6px rgba(0,0,0,0.05); margin-bottom: 20px;}}
</style>
""", unsafe_allow_html=True)

st.title('🩺 Predicción de Diabetes')
st.markdown('Completa los datos del paciente y haz clic en **Calcular Riesgo**.')

# URL de la API
api_url = st.text_input('URL de la API', 'http://127.0.0.1:5000/predict')

# Crear columnas
col1, col2 = st.columns([1, 1.2])

with col1:
    # Formulario de datos
    with st.form('form_datos'):
        edad = st.number_input('Edad', min_value=0, max_value=120, value=45)
        colesterol_alto = st.selectbox('Colesterol alto (1=Sí, 0=No)', [1, 0], index=0)
        imc = st.number_input('IMC', min_value=10.0, max_value=60.0, value=28.4, step=0.1, format="%.1f")
        enfermedad_cardiaca_o_infarto = st.selectbox('Enfermedad cardíaca o infarto (1=Sí, 0=No)', [1, 0], index=0)
        salud_general = st.slider('Salud general (1=Excelente, 5=Muy mala)', min_value=1, max_value=5, value=3)
        salud_fisica = st.number_input('Días de mala salud física (0-30)', min_value=0, max_value=30, value=5)
        dificultad_para_caminar = st.selectbox('Dificultad para caminar (1=Sí, 0=No)', [1, 0], index=0)
        accidente_cerebrovascular = st.selectbox('Accidente cerebrovascular previo (1=Sí, 0=No)', [1, 0], index=0)
        hipertension = st.selectbox('Hipertensión (1=Sí, 0=No)', [1, 0], index=0)

        submitted = st.form_submit_button('Calcular Riesgo')

with col2:
    resultado_container = st.container()

# Envío de datos a la API
if submitted:
    data = {
        "edad": edad,
        "colesterol_alto": colesterol_alto,
        "imc": imc,
        "enfermedad_cardiaca_o_infarto": enfermedad_cardiaca_o_infarto,
        "salud_general": salud_general,
        "salud_fisica": salud_fisica,
        "dificultad_para_caminar": dificultad_para_caminar,
        "accidente_cerebrovascular": accidente_cerebrovascular,
        "hipertension": hipertension
    }

    try:
        with st.spinner('Conectando con la API...'):
            resp = requests.post(api_url, json=data, timeout=10)

        if resp.status_code == 200:
            r = resp.json()

            with resultado_container:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader(f'{r["icono"]} Nivel de riesgo: {r["nivel_riesgo"]}')
                st.metric(label='Probabilidad de diabetes', value=f"{r['probabilidad_diabetes'] * 100:.1f}%")
                st.write(f'**Mensaje:** {r["mensaje"]}')
                st.write(f'**Recomendación:** {r["recomendacion"]}')
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            with resultado_container:
                st.error(f'Error desde la API: {resp.status_code} - {resp.text}')

    except Exception as e:
        with resultado_container:
            st.error(f'No se pudo conectar con la API: {e}')
