import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, auc
import warnings

warnings.filterwarnings('ignore')

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

# Directorios de salida
EDA_DIR = 'images_eda'
MODEL_DIR = 'images_model_selection'

def load_and_preprocess_data(filepath):
    print("Cargando datos...")
    data = pd.read_csv(filepath)
    
    # Renombrar columnas al español
    data.columns = [
        'Edad', 'Sexo', 'Colesterol_Alto', 'Chequeo_Colesterol', 'IMC', 'Fumador',
        'Enfermedad_Corazon', 'Actividad_Fisica', 'Frutas', 'Verduras',
        'Alcohol_Excesivo', 'Salud_General', 'Salud_Mental', 'Salud_Fisica',
        'Dificultad_Caminar', 'Derrame', 'Presion_Alta', 'Diabetes'
    ]
    
    # Transformación de edad
    data['Edad'] = data['Edad'] * 5 + 13
    data['Edad'] = data['Edad'].replace({118: 80})
    
    return data

def perform_eda(data):
    print("Generando análisis exploratorio...")
    
    # 1. Distribución de la variable objetivo (Balance de datos)
    plt.figure(figsize=(8, 6))
    counts = data['Diabetes'].value_counts()
    plt.pie(counts, labels=['Sano (0)', 'Diabético (1)'], autopct='%1.1f%%', colors=['#66b3ff','#ff9999'], startangle=90)
    plt.title('Distribución de Clases: Datos Perfectamente Balanceados\n(Crucial para métricas reales)', fontsize=14)
    plt.savefig(os.path.join(EDA_DIR, '1_distribucion_clases.png'))
    plt.close()
    
    # 2. Matriz de Correlación
    plt.figure(figsize=(14, 12))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
    plt.title("Matriz de Correlación: ¿Qué variables se relacionan más?", fontsize=16)
    plt.savefig(os.path.join(EDA_DIR, '2_matriz_correlacion.png'))
    plt.close()

    # 2b. Matriz de Correlación (Top Variables)
    plt.figure(figsize=(10, 10))
    # Calcular correlaciones con Diabetes y seleccionar top 10
    correlations = data.corr()['Diabetes'].abs().sort_values(ascending=False)
    top_features = correlations.index[:10] 
    
    sns.heatmap(data[top_features].corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
    plt.title("Matriz de Correlación: Top 10 Variables más Influyentes", fontsize=16)
    
    # Nota explicativa al pie
    plt.figtext(0.5, 0.02, 
                "Nota: Se seleccionaron las 10 variables con mayor correlación absoluta con la Diabetes.\n"
                "Esto nos permite enfocar el análisis en los factores de riesgo más determinantes.", 
                wrap=True, horizontalalignment='center', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))
                
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(EDA_DIR, '2b_matriz_correlacion_top.png'))
    plt.close()
    
    # 3. Relación Salud General vs Diabetes
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Salud_General', hue='Diabetes', data=data, palette='viridis')
    plt.title('Salud General vs Diabetes\n(1=Excelente, 5=Mala)', fontsize=14)
    plt.xlabel('Autopercepción de Salud General')
    plt.ylabel('Cantidad de Personas')
    plt.legend(title='Diabetes', labels=['No', 'Sí'])
    plt.savefig(os.path.join(EDA_DIR, '3_salud_general_vs_diabetes.png'))
    plt.close()
    
    # 4. IMC vs Diabetes (KDE Plot - Densidad)
    data_plot = data.copy()
    data_plot['Condición'] = data_plot['Diabetes'].map({0: 'Sano', 1: 'Diabético'})
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=data_plot, x='IMC', hue='Condición', fill=True, palette={'Sano': '#66b3ff', 'Diabético': '#ff9999'}, alpha=0.6, linewidth=2)
    plt.title('Distribución de IMC: ¿Pesan más los diabéticos?', fontsize=16)
    plt.xlabel('Índice de Masa Corporal (IMC)')
    plt.ylabel('Densidad de Pacientes')
    plt.savefig(os.path.join(EDA_DIR, '4_imc_densidad.png'))
    plt.close()
    
    # 5. Edad vs Diabetes (Prevalencia por Grupo Etario)
    # Binning age
    data_plot['Grupo_Edad'] = pd.cut(data_plot['Edad'], bins=[0, 30, 40, 50, 60, 70, 80, 100], labels=['<30', '30-40', '40-50', '50-60', '60-70', '70-80', '80+'])
    
    # Calculate percentage of diabetics per group
    age_risk = data_plot.groupby('Grupo_Edad')['Diabetes'].mean() * 100
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=age_risk.index, y=age_risk.values, palette='Reds')
    plt.title('Riesgo por Edad: % de Diabéticos en cada Grupo', fontsize=16)
    plt.ylabel('Porcentaje con Diabetes (%)')
    plt.xlabel('Rango de Edad')
    
    # Add value labels
    for i, v in enumerate(age_risk.values):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
        
    plt.savefig(os.path.join(EDA_DIR, '5_riesgo_por_edad.png'))
    plt.close()

def plot_model_comparison_extended(results_df):
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Model', y='Accuracy', data=results_df, palette='viridis')
    
    plt.title('Comparación de Modelos (7 Algoritmos)', fontsize=18)
    plt.ylim(0.6, 0.85)
    plt.ylabel('Accuracy (Exactitud)', fontsize=12)
    plt.xlabel('Modelo', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Etiquetas de valor
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1%}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
        
    # Nota explicativa sobre el balanceo
    text_str = (
        "NOTA IMPORTANTE:\n"
        "Estos datos están BALANCEADOS (50% Sano / 50% Diabético).\n"
        "Un Accuracy de ~75% aquí es MUCHO MEJOR que un 83% en datos desbalanceados.\n"
        "En datos desbalanceados (ej. 85% sanos), un modelo 'tonto' que prediga siempre 'Sano'\n"
        "tendría 85% de accuracy pero 0% de utilidad."
    )
    plt.text(0.98, 0.95, text_str, transform=ax.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(os.path.join(MODEL_DIR, 'comparacion_modelos_explicada.png'))
    plt.close()

def plot_roc_curve_multi(models_dict, X_test, y_test):
    plt.figure(figsize=(10, 8))
    
    # Pre-calcular AUCs para identificar los mejores
    aucs = []
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            aucs.append((name, roc_auc, fpr, tpr))
    
    # Ordenar por AUC descendente
    aucs.sort(key=lambda x: x[1], reverse=True)
    
    # Seleccionar modelos para la leyenda: Top 1 + Logistic Regression + Random Forest (o Top 3)
    models_to_label = [aucs[0][0]] # El mejor
    if 'Logistic Regression' in models_dict: models_to_label.append('Logistic Regression')
    if 'Random Forest' in models_dict: models_to_label.append('Random Forest')
    
    # Asegurar 3 únicos
    models_to_label = list(set(models_to_label))
    if len(models_to_label) < 3:
        for name, _, _, _ in aucs:
            if name not in models_to_label:
                models_to_label.append(name)
                if len(models_to_label) >= 3: break
    
    for name, roc_auc, fpr, tpr in aucs:
        if name in models_to_label:
            label = f'{name} (AUC = {roc_auc:.2f})'
            linewidth = 3 if name == aucs[0][0] else 2
            alpha = 1.0
        else:
            label = None # Sin etiqueta en leyenda
            linewidth = 1
            alpha = 0.3 # Línea tenue
            
        plt.plot(fpr, tpr, label=label, linewidth=linewidth, alpha=alpha)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Azar (AUC = 0.5)', alpha=0.5)
    
    # Explanation text box
    text_str = (
        "EXPLICACIÓN:\n"
        "• Eje X (Falsos Positivos): Errores de 'Falsa Alarma'.\n"
        "• Eje Y (Verdaderos Positivos): Aciertos detectando la enfermedad.\n"
        "• El objetivo es llegar a la esquina superior izquierda.\n"
        "• AUC > 0.80 indica un modelo excelente.\n"
        "• Se destacan los 3 modelos principales para claridad."
    )
    plt.text(0.5, 0.5, text_str, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
    plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
    plt.title('Curvas ROC: Capacidad de Diagnóstico (Top 3 Modelos)', fontsize=16)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(MODEL_DIR, 'curvas_roc_explicada.png'))
    plt.close()

def main():
    # 1. Carga y EDA
    data = load_and_preprocess_data('diabetes_data.csv')
    perform_eda(data)
    
    # 2. Preparación
    X = data.drop('Diabetes', axis=1)
    y = data['Diabetes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Definición de 7 Modelos
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(max_iter=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Gaussian Naive Bayes': GaussianNB(),
        'MLP Neural Network': MLPClassifier(hidden_layer_sizes=(50,30), max_iter=300, random_state=42)
    }
    
    results = []
    trained_models = {}
    
    print("\nEntrenando 7 modelos...")
    for name, model in models.items():
        print(f"  -> Entrenando {name}...")
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            results.append({'Model': name, 'Accuracy': acc, 'F1-Score': f1})
            trained_models[name] = model
        except Exception as e:
            print(f"    Error en {name}: {e}")

    # 4. Resultados y Selección
    results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
    print("\nResultados Finales:")
    print(results_df)
    
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    
    # Guardar
    joblib.dump(best_model, 'best_diabetes_model_final.pkl')
    joblib.dump(scaler, 'scaler_final.pkl')
    print(f"\nMejor modelo guardado: {best_model_name}")
    
    # 5. Gráficas de Modelos
    print("Generando gráficas de modelos...")
    plot_model_comparison_extended(results_df)
    plot_roc_curve_multi(trained_models, X_test_scaled, y_test)
    
    # Matriz de confusión del mejor
    y_pred_best = best_model.predict(X_test_scaled)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred_best), annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Sano', 'Diabético'], yticklabels=['Sano', 'Diabético'])
    plt.title(f'Matriz de Confusión: {best_model_name}', fontsize=14)
    plt.ylabel('Real')
    plt.xlabel('Predicción')
    plt.savefig(os.path.join(MODEL_DIR, 'matriz_confusion_final.png'))
    plt.close()
    
    print("¡Análisis completo finalizado!")

if __name__ == "__main__":
    main()
