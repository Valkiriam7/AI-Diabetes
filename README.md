# 🩺 AI Diabetes - Sistema de Predicción de Diabetes

Proyecto de Machine Learning para predecir el riesgo de diabetes en pacientes basado en variables clínicas y demográficas.

## 📊 Descripción del Proyecto

Este proyecto utiliza técnicas de Machine Learning para predecir la probabilidad de que un paciente desarrolle diabetes tipo 2, basándose en 9 variables clave:

- **Edad** del paciente
- **Colesterol alto**
- **Índice de Masa Corporal (IMC)**
- **Enfermedad cardíaca o infarto previo**
- **Salud general autorreportada**
- **Días de mala salud física**
- **Dificultad para caminar**
- **Accidente cerebrovascular previo**
- **Hipertensión**

## 🎯 Resultados del Modelo

El modelo **Random Forest** seleccionado alcanzó las siguientes métricas en el conjunto de prueba:

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 83.46% |
| **Precision** | 83.33% |
| **Recall** | 84.08% |
| **F1-Score** | 83.71% |
| **Specificity** | 82.95% |

### Comparación de Modelos Evaluados

Se evaluaron 11 modelos diferentes de clasificación:

1. **Random Forest** - 83.46% ✅ (Modelo seleccionado)
2. Voting Ensemble - 83.36%
3. Extra Trees - 83.34%
4. XGBoost - 83.24%
5. Gradient Boosting - 83.06%
6. Histogram Gradient Boosting - 82.96%
7. Logistic Regression - 74.60%
8. K-Nearest Neighbors - 71.00%
9. Gaussian Naive Bayes - 69.19%
10. Decision Tree - 68.48%
11. Support Vector Machine (RBF) - 67.48%

## 🚀 Características Principales

- ✅ **Análisis Exploratorio de Datos (EDA)** completo
- ✅ **Preprocesamiento robusto** con PowerTransformer y StandardScaler
- ✅ **11 modelos evaluados** con métricas exhaustivas
- ✅ **Sistema de predicción interactivo** con 3 niveles de riesgo
- ✅ **Interpretación clínica** automática de resultados
- ✅ **Modelo entrenado y guardado** listo para producción

## 📁 Estructura del Proyecto

```
AI-Diabetes/
│
├── Diabretes_AI.ipynb                    # Notebook principal con todo el pipeline
├── diabetes_data.csv                      # Dataset original
├── random_forest_diabetes_model.joblib    # Modelo entrenado (Random Forest)
├── preprocessor_pipeline.joblib           # Pipeline de preprocesamiento
├── README.md                              # Este archivo
└── requirements.txt                       # Dependencias del proyecto
```

## 🔧 Instalación

### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar el repositorio**:
```bash
git clone https://github.com/tu-usuario/AI-Diabetes.git
cd AI-Diabetes
```

2. **Crear un entorno virtual (recomendado)**:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

4. **Abrir el notebook**:
```bash
jupyter notebook Diabretes_AI.ipynb
```

## 💻 Uso del Sistema de Predicción

### Opción 1: Uso del Notebook

1. Abrir `Diabretes_AI.ipynb`
2. Ejecutar todas las celdas hasta llegar a "Sistema de Predicción Interactivo"
3. Modificar los valores en la sección "Entrada Manual de Datos del Paciente"
4. Ejecutar la celda para obtener la predicción

### Opción 2: Uso Programático

```python
import pandas as pd
import joblib

# Cargar el modelo entrenado
modelo = joblib.load('random_forest_diabetes_model.joblib')

# Datos del paciente
datos_paciente = pd.DataFrame({
    'edad': [50],
    'colesterol_alto': [1],
    'imc': [28.5],
    'enfermedad_cardiaca_o_infarto': [0],
    'salud_general': [3],
    'salud_fisica': [5],
    'dificultad_para_caminar': [0],
    'accidente_cerebrovascular': [0],
    'hipertension': [1]
})

# Hacer predicción
probabilidad = modelo.predict_proba(datos_paciente)[0][1]
print(f"Probabilidad de diabetes: {probabilidad:.1%}")
```

## 🎨 Interpretación de Resultados

El sistema proporciona **3 niveles de riesgo** basados en la probabilidad predicha:

### 🟢 Riesgo BAJO (< 30%)
- ✅ "No presenta signos significativos que den indicios de diabetes"
- **Recomendación**: Mantener hábitos saludables y chequeos anuales

### 🟡 Riesgo MODERADO (30% - 60%)
- ⚠️ "Se recomienda ir al médico para la prevención de la diabetes"
- **Recomendación**: Evaluación médica, pruebas de glucosa, cambios en estilo de vida

### 🔴 Riesgo ALTO (> 60%)
- 🚨 "Presenta altos síntomas de diabetes. Vaya al médico lo más pronto posible"
- **Recomendación**: URGENTE - Consulta médica inmediata para diagnóstico y tratamiento

## 📈 Metodología

### 1. Preprocesamiento de Datos
- Transformación de edad codificada a edad real
- Renombrado de columnas a español
- Selección de 9 variables más relevantes
- PowerTransformer (Yeo-Johnson) para normalización
- StandardScaler para estandarización

### 2. División de Datos
- 80% entrenamiento / 20% prueba
- Estratificación por clase (diabetes)

### 3. Entrenamiento de Modelos
- Evaluación de 11 algoritmos diferentes
- Optimización de hiperparámetros para Random Forest
- Validación cruzada y análisis de métricas

### 4. Selección de Modelo
- Random Forest seleccionado por balance entre accuracy y generalización
- Configuración final:
  - n_estimators: 200
  - max_depth: 15
  - min_samples_split: 5
  - min_samples_leaf: 2

## 📊 Dataset

El dataset contiene aproximadamente **250,000 registros** de pacientes con las siguientes características:

- **18 variables originales** (9 seleccionadas para el modelo)
- **Variable objetivo**: Presencia o ausencia de diabetes (binaria)
- **Origen**: Encuesta de salud pública
- **Preprocesamiento**: Limpieza, transformación de edad, selección de features

## 🔬 Próximos Pasos

- [ ] Desarrollo de interfaz web con Streamlit/Flask
- [ ] Implementación de API REST para predicciones
- [ ] Despliegue en la nube (AWS/Azure/GCP)
- [ ] Análisis de importancia de features con SHAP
- [ ] Validación con datos de otras regiones geográficas
- [ ] Incorporación de más variables clínicas

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/NuevaCaracteristica`)
3. Commit tus cambios (`git commit -m 'Añadir nueva característica'`)
4. Push a la rama (`git push origin feature/NuevaCaracteristica`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## ⚠️ Disclaimer

Este sistema es una **herramienta de apoyo** para la evaluación de riesgo de diabetes y **NO reemplaza el diagnóstico médico profesional**. Los resultados deben ser interpretados por personal médico calificado. Siempre consulte con un profesional de la salud para diagnóstico y tratamiento.

## 👨‍💻 Autor

**Tu Nombre**
- GitHub: [@Valkiriam7](https://github.com/Valkiriam7)
- Email: jpablo.montoya1@udea.edu.co

## 🙏 Agradecimientos

- Dataset proporcionado por [fuente del dataset]
- Comunidad de scikit-learn y XGBoost
- Documentación y recursos de Machine Learning

---

⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub!
