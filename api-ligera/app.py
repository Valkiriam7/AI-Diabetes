from flask import Flask, request, jsonify
import requests

INFERENCE_URL = "https://microservicio-47c7e2e294b1.herokuapp.com/inferir"

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "API Diabetes funcionando correctamente"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Fuerza la lectura del JSON
        data = request.get_json(force=True)

        # Validar campos
        required_fields = [
            "edad", "colesterol_alto", "imc", "enfermedad_cardiaca_o_infarto",
            "salud_general", "salud_fisica", "dificultad_para_caminar",
            "accidente_cerebrovascular", "hipertension"
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Falta el campo {field}"}), 400

        # Enviar la data al microservicio de inferencia
        response = requests.post(INFERENCE_URL, json=data)

        return jsonify(response.json()), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

