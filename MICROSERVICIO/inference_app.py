from flask import Flask, request, jsonify
import joblib
from Diabetes_AI import predecir_diabetes

model = joblib.load("modelo_diabetes_ajustado.pkl")

app = Flask(__name__)

@app.route("/inferir", methods=["POST"])
def inferir():
    try:
        data = request.get_json(force=True)
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
        resultado["probabilidad_diabetes"] = float(resultado["probabilidad_diabetes"])
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
