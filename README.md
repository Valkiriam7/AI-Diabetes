# 🩺 AI Diabetes - Sistema de Predicción de Diabetes

Proyecto de Machine Learning para predecir el riesgo de diabetes utilizando un enfoque robusto con datos balanceados y múltiples algoritmos de clasificación.

## 📊 Descripción del Proyecto

Este proyecto ha sido refactorizado para ofrecer un análisis más riguroso y realista. A diferencia de versiones anteriores, utilizamos un dataset **perfectamente balanceado (50% sanos / 50% diabéticos)** y analizamos **18 variables clínicas y demográficas** (en lugar de un subconjunto limitado) para maximizar la capacidad predictiva del modelo.

### Variables Analizadas
El modelo considera factores como:
- **Demográficos:** Edad, Sexo, Educación, Ingresos.
- **Clínicos:** IMC, Colesterol Alto, Presión Alta, Enfermedades Cardíacas, Derrame.
- **Estilo de Vida:** Actividad Física, Consumo de Frutas/Verduras, Alcohol, Fumar.
- **Salud General:** Salud Mental, Salud Física, Dificultad para Caminar.

## 🎯 Resultados y Selección del Modelo

Tras evaluar 7 algoritmos diferentes, seleccionamos el **HistGradientBoostingClassifier** por su rendimiento superior y estabilidad.

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Accuracy** | **75.06%** | Exactitud global en datos balanceados (50/50). |
| **F1-Score** | **76.17%** | Balance óptimo entre precisión y sensibilidad. |
| **AUC-ROC** | **> 0.82** | Excelente capacidad de discriminación diagnóstica. |

> **Nota sobre el Accuracy:** Un accuracy del 75% en un dataset balanceado (50/50) es un resultado mucho más robusto y valioso que un 83% en un dataset desbalanceado (donde el modelo podría simplemente predecir "sano" siempre).

### Modelos Evaluados
1. **Hist Gradient Boosting** (Seleccionado 🏆)
2. Random Forest
3. Logistic Regression
4. AdaBoost
5. Decision Tree
6. Gaussian Naive Bayes
7. MLP Neural Network

## 📂 Estructura del Repositorio

El proyecto se ha limpiado y organizado para facilitar su comprensión:

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

---
*Proyecto realizado para el Bootcamp de IA - Noviembre 2025*
