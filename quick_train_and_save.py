"""Entrenamiento rápido y guardado de modelo para AI Diabetes.
Genera:
 - random_forest_diabetes_model.joblib
 - modelo_diabetes_ajustado.pkl

Uso: python quick_train_and_save.py
"""
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SRC_CSV = "diabetes_data.csv"
MODEL_NAMES = ["random_forest_diabetes_model.joblib", "modelo_diabetes_ajustado.pkl"]

if not os.path.exists(SRC_CSV):
    raise SystemExit(f"No se encontró {SRC_CSV} en el directorio actual: {os.getcwd()}")

print(f"Cargando dataset desde {SRC_CSV}...")
df = pd.read_csv(SRC_CSV)
print(f"Registros: {len(df)} | Columnas: {len(df.columns)}")

# Columnas esperadas
expected = ['edad','colesterol_alto','imc','enfermedad_cardiaca_o_infarto',
            'salud_general','salud_fisica','dificultad_para_caminar',
            'accidente_cerebrovascular','hipertension','diabetes']

# Detectar columna objetivo si tiene otro nombre
if 'diabetes' not in df.columns:
    for c in df.columns:
        if 'diab' in c.lower():
            df = df.rename(columns={c: 'diabetes'})
            print(f"Renombrada columna objetivo '{c}' a 'diabetes'")
            break

if 'diabetes' not in df.columns:
    raise SystemExit('No se encontró la columna objetivo "diabetes" en el dataset.')

# Seleccionar features disponibles
# Intentar mapear encabezados en inglés a los nombres en español usados por el notebook
col_mapping = {
    'Age': 'edad',
    'HighChol': 'colesterol_alto',
    'BMI': 'imc',
    'HeartDiseaseorAttack': 'enfermedad_cardiaca_o_infarto',
    'GenHlth': 'salud_general',
    'PhysHlth': 'salud_fisica',
    'DiffWalk': 'dificultad_para_caminar',
    'Stroke': 'accidente_cerebrovascular',
    'HighBP': 'hipertension',
    'Diabetes': 'diabetes'
}

# Renombrar columnas si el dataset está en inglés
rename_map = {k: v for k, v in col_mapping.items() if k in df.columns}
if rename_map:
    df = df.rename(columns=rename_map)
    print('Se renombraron columnas según mapping:', rename_map)

features = [c for c in expected if c in df.columns and c != 'diabetes']
if len(features) < 3:
    raise SystemExit(f'No hay suficientes features detectadas. Encontradas: {features}')

print('Features usadas:', features)
X = df[features].fillna(0)
y = df['diabetes']

# Intentar cargar preprocessor si existe y transformar
preprocessor_path = 'preprocessor_pipeline.joblib'
if os.path.exists(preprocessor_path):
    try:
        pre = joblib.load(preprocessor_path)
        print('Preprocessor cargado desde', preprocessor_path)
        X_proc = pre.transform(X)
        # convertir sparse si aplica
        try:
            X_proc = X_proc.toarray()
        except Exception:
            pass
    except Exception as e:
        print('No se pudo usar el preprocessor cargado:', e)
        X_proc = X.values
else:
    print('No se encontró preprocessor_pipeline.joblib, usando features crudas')
    X_proc = X.values

# Tomar muestra para entrenamiento rápido
n = min(10000, X_proc.shape[0])
X_sample = X_proc[:n]
y_sample = y[:n]

print(f"Usando {n} filas para entrenamiento rápido")
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample)

clf = RandomForestClassifier(n_estimators=150, max_depth=14, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"Accuracy rápido en muestra: {acc:.4f}")

# Guardar modelos
for name in MODEL_NAMES:
    joblib.dump(clf, name)
    print(f"Modelo guardado: {name}")

print('Finalizado')
