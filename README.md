# 🩺 AI Diabetes - Sistema de Predicción de Diabetes

Proyecto de Machine Learning para predecir el riesgo de diabetes utilizando un enfoque robusto con datos balanceados y múltiples algoritmos de clasificación.

## 📊 Descripción del Proyecto

Este proyecto realiza un análisis riguroso utilizando un dataset **perfectamente balanceado (50% sanos / 50% diabéticos)** y analiza **18 variables clínicas y demográficas** para maximizar la capacidad predictiva del modelo.

### Variables Analizadas
El modelo considera factores como:
- **Demográficos:** Edad, Sexo, Educación, Ingresos.
- **Clínicos:** IMC, Colesterol Alto, Presión Alta, Enfermedades Cardíacas, Derrame.
- **Estilo de Vida:** Actividad Física, Consumo de Frutas/Verduras, Alcohol, Fumar.
- **Salud General:** Salud Mental, Salud Física, Dificultad para Caminar.

## 🎯 Resultados y Selección del Modelo

Tras evaluar 7 algoritmos diferentes, seleccionamos el **HistGradientBoostingClassifier** por su rendimiento superior y estabilidad.

| Modelo | Accuracy | F1-Score | AUC-ROC |
|--------|----------|----------|---------|
| **Hist Gradient Boosting** 🏆 | **75.06%** | **76.17%** | **0.8280** |
| Random Forest | 74.98% | 76.12% | 0.8260 |
| MLP Neural Network | 74.90% | 76.17% | 0.8262 |
| Logistic Regression | 74.54% | 75.00% | 0.8217 |
| AdaBoost | 74.33% | 74.87% | 0.8216 |
| Decision Tree | 73.39% | 74.61% | 0.8067 |
| Gaussian Naive Bayes | 72.14% | 72.03% | 0.7832 |

> **Nota sobre el Accuracy:** Un accuracy del 75% en un dataset balanceado (50/50) es un resultado mucho más robusto y valioso que un 83% en un dataset desbalanceado (donde el modelo podría simplemente predecir "sano" siempre).

## 📂 Estructura del Repositorio

El proyecto está organizado de la siguiente manera:

- **`final_analysis.py`**: Script maestro. Ejecuta todo el proceso: carga de datos, EDA, entrenamiento de 7 modelos, evaluación y generación de gráficas.
- **`diabetes_data.csv`**: Dataset utilizado.
- **`best_diabetes_model_final.pkl`**: El modelo entrenado listo para producción.
- **`scaler_final.pkl`**: Escalador para preprocesar nuevos datos.
- **`images_eda/`**: Gráficas del Análisis Exploratorio de Datos (Correlaciones, Distribuciones, Riesgo por edad).
- **`images_model_selection/`**: Gráficas de rendimiento de modelos (Curvas ROC, Matrices de Confusión, Comparativas).

## 🚀 Instrucciones de Uso

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar el análisis completo:**
   ```bash
   python final_analysis.py
   ```
   Esto generará las imágenes en las carpetas correspondientes y guardará el mejor modelo.

## 📈 Visualizaciones Destacadas

El proyecto incluye visualizaciones diseñadas para presentaciones educativas:
- **Matriz de Correlación Top 10:** Identifica las variables más influyentes.
- **Curvas ROC Explicadas:** Muestra visualmente la capacidad de diagnóstico.
- **Riesgo por Edad:** Probabilidad de diabetes desglosada por grupos etarios.

## 🔬 Próximos Pasos

- [x] Desarrollo de interfaz web con Streamlit (`app_front.py`)
- [x] Implementación de API REST para predicciones (`app_api.py`)
- [ ] Despliegue en la nube (en progreso - ver DEPLOY.md)
- [ ] Análisis de importancia de features con SHAP
- [ ] Validación con datos de otras regiones geográficas
- [ ] Incorporación de más variables clínicas

## ⚠️ Disclaimer

Este sistema es una **herramienta de apoyo** para la evaluación de riesgo de diabetes y **NO reemplaza el diagnóstico médico profesional**. Los resultados deben ser interpretados por personal médico calificado. Siempre consulte con un profesional de la salud para diagnóstico y tratamiento.

## 👨‍💻 Autores

- **Juan Pablo Montoya** - [@Valkiriam7](https://github.com/Valkiriam7)
- **Alejandra Lopera** - [@techia976-ai](https://github.com/techia976-ai)

### Agradecimiento Especial
A la **Dra. Gloria Marín**, cuya experiencia y guía fueron fundamentales para el entendimiento clínico del proyecto y el refinamiento de la propuesta.

---
*Proyecto realizado para el Bootcamp de IA - Noviembre 2025*
