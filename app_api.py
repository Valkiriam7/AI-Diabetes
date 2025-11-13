from flask import Flask, request, jsonify
import os
import joblib
from Diabetes_AI import predecir_diabetes

# Cargar el modelo (busca nombres comunes en el repo)
possible_names = [
    "random_forest_diabetes_model.joblib",
    "modelo_diabetes_ajustado.pkl",
    "random_forest.pkl",
    "modelo_diabetes.joblib"
]
model = None
for p in possible_names:
    if os.path.exists(p):
        model = joblib.load(p)
        print(f"Cargado modelo desde: {p}")
        break

if model is None:
    raise FileNotFoundError("No se encontró un archivo de modelo en el directorio del proyecto. Coloca 'random_forest_diabetes_model.joblib' o 'modelo_diabetes_ajustado.pkl'.")

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "API de Predicción de Diabetes funcionando correctamente"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        resultado = predecir_diabetes(
            edad=data['edad'],
            colesterol_alto=data['colesterol_alto'],
            imc=data['imc'],
            enfermedad_cardiaca=data['enfermedad_cardiaca_o_infarto'],
            salud_general=data['salud_general'],
            salud_fisica=data['salud_fisica'],
            dificultad_caminar=data['dificultad_para_caminar'],
            accidente_cerebrovascular=data['accidente_cerebrovascular'],
            hipertension=data['hipertension'],
            modelo=model,
            mostrar_detalle=False
        )

        return jsonify({
            "prediccion": int(resultado["prediccion"]),
            "probabilidad_diabetes": round(float(resultado["probabilidad_diabetes"]), 4),
            "nivel_riesgo": resultado["nivel_riesgo"],
            "icono": resultado["icono"],
            "mensaje": resultado["mensaje"],
            "recomendacion": resultado["recomendacion"]
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
