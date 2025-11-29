import streamlit as st
import requests
import pandas as pd
import random 
import math 

st.set_page_config(
    page_title='Predicción de Diabetes',
    page_icon= '🧪',
    layout='wide',
)

STEEL_BLUE_PALETTE = {
    'primary': '#42A5F5',
    'secondary_bg': '#1E2B3E',
    'main_bg': '#0D131F',
    'text': '#E0E7FF',
    'text_secondary': '#9AB8D0',
    'button_hover': '#64B5F6',
    'warning_color': '#00BFFF',
    'error_color': '#E53935'
}
PALETTE = STEEL_BLUE_PALETTE

st.markdown(f"""
<style>
:root {{
    --primary-color: {PALETTE['primary']} !important;
    --background-color: {PALETTE['main_bg']} !important;
    --secondary-background-color: {PALETTE['secondary_bg']} !important;
    --text-color: {PALETTE['text']} !important;
    --font: 'Inter', sans-serif !important;
}}

.stApp {{
    background-color: var(--background-color) !important;
    color: var(--text-color) !important;
}}

*, h1, h2, h3, h4, h5, h6, label, p, .stMarkdown, .stText, [data-testid="stText"] {{
    color: var(--text-color) !important;
    -webkit-text-fill-color: var(--text-color) !important;
}}

div[data-testid="stText"] p, div[data-testid="stMarkdownContainer"] p {{
    color: {PALETTE['text_secondary']} !important;
}}

:focus-visible,
[data-testid*="stNumberInput"] input:focus-visible,
[data-testid*="stSelectbox"] div[role="button"]:focus-visible,
[data-testid*="stTextInput"] input:focus-visible,
[data-testid*="stButton"] button:focus-visible {{
    box-shadow: 0 0 0 0.2rem rgba(66, 165, 245, 0.5) !important;
    border-color: {PALETTE['primary']} !important;
    outline: none !important;
}}

div[data-baseweb="base-input"] > div,
div[data-baseweb="base-input"] {{
    background-color: var(--secondary-background-color) !important;
    border: 1px solid #3B4A66 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}}

div[data-baseweb="base-input"] input {{
    background-color: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
    -webkit-text-fill-color: var(--text-color) !important;
}}

div[data-testid="stSelectbox"] div[data-baseweb="select"] div[role="button"] {{
    background-color: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
    border: 1px solid #3B4A66 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}}

div[data-baseweb="popover"] div[role="listbox"] {{
    background-color: {PALETTE['secondary_bg']} !important;
    border: 1px solid #3B4A66 !important;
    border-radius: 8px;
}}
div[data-baseweb="popover"] div[role="option"] > div {{
    color: var(--text-color) !important;
}}
div[data-baseweb="popover"] div[role="option"]:hover {{
    background-color: #3B4A66 !important;
}}

div[data-testid="stSelectbox"] svg,
div[data-testid="stNumberInput-controls"] button,
div[data-testid="stNumberInput-controls"] button * {{
    fill: {PALETTE['text_secondary']} !important;
    color: {PALETTE['text_secondary']} !important;
}}

div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="application"] > div:nth-child(2) > div {{
    background-color: var(--primary-color) !important;
}}

div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {{
    background-color: var(--primary-color) !important;
    border: 3px solid #0D131F !important;
}}

div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"]:focus,
div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"]:active {{
    box-shadow: 0 0 0 4px rgba(66, 165, 245, 0.5) !important;
}}

[data-testid="stCheckbox"] input:checked + div:first-child {{
    background-color: var(--primary-color) !important;
    border-color: var(--primary-color) !important;
}}
[data-testid="stCheckbox"] input + div:first-child {{
    border-color: #3B4A66 !important;
}}
[data-testid="stCheckbox"] input:checked + div:first-child svg {{
    color: #0D131F !important;
    fill: #0D131F !important;
}}

[data-testid="stColumnProgress"] div[data-baseweb="progress-bar"] div:nth-child(2) {{
    background-color: var(--primary-color) !important;
}}

.stButton>button,
div[data-testid="stForm"] .stButton>button {{
    background-color: var(--primary-color);
    color: #0D131F !important;
    border: none;
    border-radius: 12px;
    padding: 10px 20px;
    font-weight: 700;
    box-shadow: 0 6px 15px rgba(66, 165, 245, 0.4);
    transition: all 0.3s ease-in-out;
}}

.stButton>button:hover {{
    background-color: var(--button_hover);
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(100, 181, 246, 0.5);
}}

.stButton:nth-last-child(1) button {{
    background-color: #3B4A66;
    color: var(--text-color) !important;
    box-shadow: none;
}}
.stButton:nth-last-child(1) button:hover {{
    background-color: #5A6D88;
}}

.card, div[data-testid="stForm"] {{
    background-color: var(--secondary-background-color);
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #3B4A66;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
    margin-bottom: 25px;
}}
.card {{
    border-left: 6px solid var(--primary-color);
}}

div[data-testid="stMetricValue"] {{
    color: var(--primary-color) !important;
    font-size: 2.2rem !important;
}}
div[data-testid="stMetricLabel"] p {{
    color: {PALETTE['text_secondary']} !important;
}}

[data-testid="stAlert"] [data-testid="stAlertContent"] {{
    background-color: {PALETTE['secondary_bg']} !important;
    color: var(--text-color) !important;
    border-radius: 10px;
}}

div[data-baseweb="notification"] {{
    background-color: transparent !important;
}}

div[data-baseweb="notification"][data-kind="positive"] {{
    border-left: 6px solid {PALETTE['primary']} !important;
}}
div[data-baseweb="notification"][data-kind="positive"] svg {{
    color: {PALETTE['primary']} !important;
    fill: {PALETTE['primary']} !important;
}}

div[data-baseweb="notification"][data-kind="info"] {{
    border-left: 6px solid {PALETTE['warning_color']} !important;
}}
div[data-baseweb="notification"][data-kind="info"] svg {{
    color: {PALETTE['warning_color']} !important;
    fill: {PALETTE['warning_color']} !important;
}}

div[data-baseweb="notification"][data-kind="warning"] {{
    border-left: 6px solid {PALETTE['warning_color']} !important;
}}
div[data-baseweb="notification"][data-kind="warning"] svg {{
    color: {PALETTE['warning_color']} !important;
    fill: {PALETTE['warning_color']} !important;
}}

div[data-baseweb="notification"][data-kind="negative"] {{
    border-left: 6px solid {PALETTE['error_color']} !important;
}}
div[data-baseweb="notification"][data-kind="negative"] svg {{
    color: {PALETTE['error_color']} !important;
    fill: {PALETTE['error_color']} !important;
}}

[data-testid="stAlert"] [data-testid="stAlertContent"] p {{
    color: var(--text-color) !important;
}}

</style>
""", unsafe_allow_html=True)


DEFAULT_VALUES = {
    'edad': 0,
    'colesterol_alto': 'No',
    'imc': 22.0,
    'peso_kg': 70.0,
    'altura_m': 1.75,
    'enfermedad_cardiaca_o_infarto': 'No',
    'salud_general': 3,
    'salud_fisica': 0,
    'dificultad_para_caminar': 'No',
    'accidente_cerebrovascular': 'No',
    'hipertension': 'No',
    'api_url': 'https://api-ligera-cb8c0af4c8d7.herokuapp.com/predict',
    'aceptacion_terminos': False
}


for key, default_value in DEFAULT_VALUES.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

if 'resultado_mostrado' not in st.session_state:
    st.session_state.resultado_mostrado = False

if 'historial_pacientes' not in st.session_state:
    st.session_state.historial_pacientes = []


def calculate_bmi():
    try:
        peso = st.session_state.peso_kg
        altura = st.session_state.altura_m

        if altura > 0:
            imc_calculado = peso / (altura ** 2)

            if imc_calculado < 10.0:
                 st.session_state.imc = 10.0
                 st.warning("IMC mínimo de 10.0 alcanzado para la visualización.", icon="⚠️")
            elif imc_calculado > 60.0:
                 st.session_state.imc = 60.0
                 st.warning("IMC máximo de 60.0 alcanzado para la visualización.", icon="⚠️")
            else:
                 st.session_state.imc = round(imc_calculado, 1)
        else:
            st.warning("La altura debe ser mayor que 0 metros para calcular el IMC.", icon="⚠️")
            st.session_state.imc = DEFAULT_VALUES['imc']

    except Exception:
        st.error("Error al calcular el IMC. Revise los valores ingresados.", icon="❌")
        st.session_state.imc = DEFAULT_VALUES['imc']

def reset_fields():
    for key, default_value in DEFAULT_VALUES.items():
        if key not in ['api_url', 'peso_kg', 'altura_m', 'aceptacion_terminos']:
            st.session_state[key] = default_value
    st.session_state.resultado_mostrado = False

def get_select_index(key):
    return 1 if st.session_state[key] == 'Sí' else 0


st.title('Cálculo de Riesgo de Diabetes por Condición de Salud 🧪')
st.markdown(f'<p style="color: {PALETTE["text_secondary"]};">Completa los datos del paciente para evaluar el riesgo de diabetes de forma rápida y sencilla. Los resultados aparecen en tiempo real a la derecha.</p>', unsafe_allow_html=True)

with st.expander("⚙️ Configuración de la API (Avanzado)", expanded=False):
    st.text_input(
        'URL del Servicio de Predicción (API)',
        value=st.session_state.api_url,
        key='api_url_input',
        on_change=lambda: st.session_state.update(api_url=st.session_state.api_url_input, resultado_mostrado=False)
    )
api_url = st.session_state.api_url_input

col1, col2 = st.columns([1, 1.2])

with col1:

    st.subheader("1️⃣ Aceptación de Términos")
    disclaimer_text = (
        "Este sistema es una herramienta de apoyo para la evaluación de riesgo de diabetes y **NO reemplaza el diagnóstico médico profesional**."
    )

    st.checkbox(
        disclaimer_text,
        key='aceptacion_terminos'
    )
    st.markdown('---')

    st.info(
        "**Cálculo de IMC Importante**: Si aún no dispone de su **Índice de Masa Corporal (IMC)**, le invitamos a calcularlo previamente. Este valor es **clave** para una evaluación de riesgo adecuada.",
        icon="ℹ️"
    )

    st.subheader("2️⃣ Calculadora de IMC (Auxiliar)")

    col_imc_1, col_imc_2, col_imc_3 = st.columns([1, 1, 1])

    with col_imc_1:
        peso_kg = st.number_input(
            'Peso (kg)', min_value=0.1, max_value=300.0, step=0.1, format="%.1f",
            key='peso_kg',
            value=st.session_state.peso_kg
        )

    with col_imc_2:
        altura_m = st.number_input(
            'Altura (m)', min_value=0.1, max_value=3.0, step=0.01, format="%.2f",
            key='altura_m',
            value=st.session_state.altura_m
        )

    with col_imc_3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button('Calcular IMC ⏎', on_click=calculate_bmi, use_container_width=True, key='btn_calc_imc')
        st.markdown("✅ IMC calculado y actualizado en el campo de la sección 3️⃣: 'IMC (Índice de Masa Corporal)'.")

    st.markdown("---")

    with st.form('form_datos', clear_on_submit=False):

        st.subheader("3️⃣ Datos de Evaluación de Riesgo")

        col_in1, col_in2 = st.columns(2)

        with col_in1:
            edad = st.number_input(
                '¿Cuál es tu edad? (años)', min_value=0, max_value=120, key='edad',
                value=st.session_state.edad,
                help="El riesgo de diabetes aumenta con la edad, especialmente después de los 45 años."
            )

            colesterol_alto = st.selectbox(
                '¿Te han diagnosticado con colesterol alto?',
                options=["No", "Sí"],
                index=get_select_index('colesterol_alto'),
                key='colesterol_alto',
                help="Los niveles altos de colesterol a menudo coexisten con la resistencia a la insulina."
            )

            enfermedad_cardiaca_o_infarto = st.selectbox(
                '¿Alguna vez has tenido una enfermedad cardíaca o un infarto?',
                options=["No", "Sí"],
                index=get_select_index('enfermedad_cardiaca_o_infarto'),
                key='enfermedad_cardiaca_o_infarto',
                help="La diabetes aumenta el riesgo de enfermedades cardiovasculares. Un historial previo indica un riesgo elevado."
            )

            hipertension = st.selectbox(
                '¿Te han diagnosticado con hipertensión?',
                options=["No", "Sí"],
                index=get_select_index('hipertension'),
                key='hipertension',
                help="La presión arterial alta y la diabetes a menudo ocurren juntas, dañando los vasos sanguíneos."
            )

        with col_in2:

            imc = st.number_input(
                'IMC (Índice de Masa Corporal)',
                min_value=10.0, max_value=60.0, step=0.1, format="%.1f",
                key='imc',
                value=st.session_state.imc,
                help="El IMC es un factor de riesgo primario para la diabetes. Puedes calcularlo en la sección auxiliar de arriba."
            )

            accidente_cerebrovascular = st.selectbox(
                '¿Alguna vez has tenido un accidente cerebrovascular?',
                options=["No", "Sí"],
                index=get_select_index('accidente_cerebrovascular'),
                key='accidente_cerebrovascular',
                help="Un historial de ACV es un indicador de problemas vasculares serios, correlacionado con el riesgo de diabetes."
            )

            salud_fisica = st.number_input(
                'Días de mala salud física (último mes) (0-30 días)', min_value=0, max_value=30,
                key='salud_fisica',
                value=st.session_state.salud_fisica,
                help="Refleja cómo las condiciones de salud física afectaron tus actividades diarias."
            )

            dificultad_para_caminar = st.selectbox(
                '¿Tienes dificultad seria para caminar?',
                options=["No", "Sí"],
                index=get_select_index('dificultad_para_caminar'),
                key='dificultad_para_caminar',
                help="La dificultad de movilidad puede estar relacionada con neuropatías o sedentarismo."
            )

        st.markdown('---')
        st.subheader("4️⃣ Percepción de Salud")

        salud_general = st.slider(
            '¿Cómo calificarías tu estado general de salud? (1=Excelente, 5=Muy mala)',
            min_value=1, max_value=5,
            key='salud_general',
            value=st.session_state.salud_general,
            help="Una percepción de salud 'mala' suele estar asociada con un mayor riesgo de enfermedades crónicas."
        )

        st.markdown('---')

        submitted = st.form_submit_button(
            'Calcular Riesgo',
            use_container_width=True,
            disabled=not st.session_state.aceptacion_terminos
        )

    st.button('Limpiar Campos', on_click=reset_fields, use_container_width=True, key='btn_limpiar')

with col2:
    resultado_container = st.container()
    historial_container = st.container()

if submitted:
    st.session_state.resultado_mostrado = True

    keys_to_send = [key for key in DEFAULT_VALUES.keys() if key not in ['api_url', 'peso_kg', 'altura_m', 'aceptacion_terminos']]
    raw_data = {key: st.session_state[key] for key in keys_to_send}

    data = {}
    for key, value in raw_data.items():
        if value == 'Sí':
            data[key] = 1
        elif value == 'No':
            data[key] = 0
        else:
            data[key] = value

    try:
        with st.spinner('Conectando con la API...'):
            max_retries = 3
            initial_delay = 1
            resp = None

            for attempt in range(max_retries):
                try:
                    resp = requests.post(api_url, json=data, timeout=10)
                    if resp.status_code == 200:
                        break
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(initial_delay * (2 ** attempt))
                    else:
                        raise e

        if resp.status_code == 200:
            r = resp.json()

            with resultado_container:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader(f'{r["icono"]} Nivel de riesgo: {r["nivel_riesgo"]}')
                st.metric(label='Probabilidad de diabetes', value=f"{r['probabilidad_diabetes'] * 100:.1f}%")
                st.markdown(f'<p style="color: {PALETTE["text"]}; font-weight: bold;">Mensaje:</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="color: {PALETTE["text_secondary"]}; margin-top: -10px;">{r["mensaje"]}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="color: {PALETTE["text"]}; font-weight: bold;">Recomendación:</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="color: {PALETTE["text_secondary"]}; margin-top: -10px;">{r["recomendacion"]}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            patient_id = f"P-{random.randint(100, 999)}"
            new_entry = {
                'ID Paciente': patient_id,
                'Probabilidad (%)': round(r['probabilidad_diabetes'] * 100, 1),
                'Riesgo': r['nivel_riesgo'],
                'IMC': data['imc'],
                'Hipertensión': 'Sí' if data['hipertension'] == 1 else 'No'
            }
            st.session_state.historial_pacientes.insert(0, new_entry)

        else:
            with resultado_container:
                st.error(f'Error desde la API: {resp.status_code} - {resp.text}', icon="❌")

    except Exception as e:
        with resultado_container:
            st.error(f'No se pudo conectar con la API después de varios intentos. Asegúrate de que la URL ({api_url}) es correcta y que el servicio está activo. Error: {e}', icon="❌")


with historial_container:
    if st.session_state.historial_pacientes:
        st.markdown('---')
        st.subheader('📈 Historial de Predicciones Recientes')

        df_historial = pd.DataFrame(st.session_state.historial_pacientes)

        df_display_for_chart = df_historial.head(10)
        df_table = df_display_for_chart.drop(columns=['ID Paciente'], errors='ignore')


        st.dataframe(
            df_table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Probabilidad (%)": st.column_config.ProgressColumn(
                    "Probabilidad (%)",
                    help="Probabilidad calculada por el modelo",
                    format="%f",
                    min_value=0,
                    max_value=100,
                    # Se eliminó el argumento 'color'
                ),
            }
        )

        #st.bar_chart(df_display_for_chart, x='ID Paciente', y='Probabilidad (%)', color=PALETTE['primary'])
        st.bar_chart(df_display_for_chart,y='Probabilidad (%)', color=PALETTE['primary'])

        st.button('Limpiar Historial', on_click=lambda: st.session_state.update(historial_pacientes=[]), use_container_width=True, key='btn_clear_history')