import pandas as pd
import os

try:
    import joblib
except Exception:
    joblib = None


def cargar_modelo(path="random_forest_diabetes_model.joblib"):
    """Carga un modelo si existe en la ruta indicada."""
    if joblib is None:
        raise ImportError("joblib no está instalado en el entorno")
    if os.path.exists(path):
        return joblib.load(path)
    # Intentar nombres alternativos
    alt = "modelo_diabetes_ajustado.pkl"
    if os.path.exists(alt):
        return joblib.load(alt)
    raise FileNotFoundError(f"No se encontró el archivo de modelo en '{path}' ni en '{alt}'")


def predecir_diabetes(edad, colesterol_alto, imc, enfermedad_cardiaca,
                     salud_general, salud_fisica, dificultad_caminar,
                     accidente_cerebrovascular, hipertension,
                     modelo=None, mostrar_detalle=True):
    """
    Predice el riesgo de diabetes con interpretación clínica.

    Si no se proporciona `modelo`, se intenta cargar el modelo por defecto
    usando `cargar_modelo()`.
    """

    if modelo is None:
        modelo = cargar_modelo()

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

    pred = int(modelo.predict(datos_paciente)[0])
    prob = float(modelo.predict_proba(datos_paciente)[0][1])

    if prob < 0.30:
        nivel = "BAJO"
        icono = "🟢"
        mensaje = "No presenta signos significativos que den indicios de diabetes."
        recomendacion = "Mantener hábitos saludables: dieta balanceada, ejercicio regular y chequeos médicos periódicos."
    elif prob < 0.60:
        nivel = "MODERADO"
        icono = "🟡"
        mensaje = "Se recomienda ir al médico para la prevención de la diabetes."
        recomendacion = "Consulte con su médico para evaluación de factores de riesgo y pruebas (glucosa, HbA1c)."
    else:
        nivel = "ALTO"
        icono = "🔴"
        mensaje = "Presenta altos síntomas de diabetes. Vaya al médico lo más pronto posible para exámenes y tratamiento."
        recomendacion = "URGENTE: Solicite evaluación médica inmediata y pruebas diagnósticas."

    return {
        "prediccion": pred,
        "probabilidad_diabetes": prob,
        "nivel_riesgo": nivel,
        "icono": icono,
        "mensaje": mensaje,
        "recomendacion": recomendacion
    }


if __name__ == "__main__":
    print("Módulo Diabetes_AI listo. Use cargar_modelo() o predecir_diabetes(..., modelo=mi_modelo)")
