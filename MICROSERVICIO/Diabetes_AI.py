import pandas as pd

# Asegúrate de que tu modelo esté cargado antes o inclúyelo aquí
# from joblib import load
# final_model_RF = load("modelo_diabetes.pkl")

def predecir_diabetes(edad, colesterol_alto, imc, enfermedad_cardiaca, 
                     salud_general, salud_fisica, dificultad_caminar, 
                     accidente_cerebrovascular, hipertension, 
                     modelo=None, mostrar_detalle=True):
    """
    Predice el riesgo de diabetes con interpretación clínica.
    """

    if modelo is None:
        raise ValueError("⚠️ Debes cargar el modelo antes de usar esta función")

    datos_paciente = pd.DataFrame({
        'edad': [edad],
        'colesterol_alto': [colesterol_alto],
        'imc': [imc],
        'enfermedad_cardiaca_o_infarto': [enfermedad_cardiaca],
        'salud_general': [salud_general],
        'salud_fisica': [salud_fisica],
        'dificultad_para_caminar': [dificultad_caminar],
        'accidente_cerebrovascular': [accidente_cerebrovascular],
        'hipertension': [hipertension]
    })

    pred = modelo.predict(datos_paciente)[0]
    prob = modelo.predict_proba(datos_paciente)[0][1]

    if prob < 0.30:
        nivel = "BAJO"
        icono = "🟢"
        mensaje = "Los factores de riesgo evaluados dan baja probabilidad de tener diabetes Mellitus DM."
        recomendacion = "Mantener hábitos saludables."
    elif prob < 0.60:
        nivel = "MODERADO"
        icono = "🟡"
        mensaje = "Se recomienda ir al médico para la prevención de la diabetes."
        recomendacion = "Evalúe factores de riesgo y realice chequeo médico."
    else:
        nivel = "ALTO"
        icono = "🔴"
        mensaje = " Los factores de riesgo asociados a su salud lo predisponen con alta probabilidad a ser Diabético."
        recomendacion = "Realizar seguimiento periódico estricto con su médico y pruebas diagnósticas para Diabetes Mellitus DM (glucosa en ayunas, HbA1c, curva de tolerancia)."

    return {
        "prediccion": int(pred),
        "probabilidad_diabetes": float(prob),
        "nivel_riesgo": nivel,
        "icono": icono,
        "mensaje": mensaje,
        "recomendacion": recomendacion
    }

print("✅ Archivo Diabetes_AI.py generado correctamente")
