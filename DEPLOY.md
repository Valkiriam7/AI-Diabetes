# 🚀 Guía de Despliegue - AI Diabetes

## Despliegue en Render (Recomendado - Más Fácil)

### Paso 1: Crear cuenta en Render
1. Ve a https://render.com y crea una cuenta (puedes usar GitHub)
2. Conecta tu cuenta de GitHub

### Paso 2: Crear Web Service
1. Desde el dashboard, click en **"New +"** → **"Web Service"**
2. Conecta tu repositorio: `Valkiriam7/AI-Diabetes`
3. Configura el servicio:
   - **Name**: `ai-diabetes-api` (o el nombre que prefieras)
   - **Region**: Selecciona la más cercana
   - **Branch**: `main`
   - **Root Directory**: (dejar en blanco)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app_api:app`
   - **Instance Type**: `Free` (o el plan que prefieras)

### Paso 3: Variables de Entorno (opcional)
Si necesitas configurar variables:
- Click en **"Environment"** en el panel izquierdo
- Añade: `PYTHON_VERSION` = `3.11.0` (opcional)

### Paso 4: Deploy
1. Click en **"Create Web Service"**
2. Render automáticamente:
   - Clonará tu repo
   - Instalará dependencias
   - Iniciará la aplicación
3. En 3-5 minutos tendrás tu URL pública: `https://ai-diabetes-api.onrender.com`

### Paso 5: Probar la API
```bash
curl -X POST https://tu-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "edad": 50,
    "colesterol_alto": 1,
    "imc": 30.5,
    "enfermedad_cardiaca_o_infarto": 0,
    "salud_general": 3,
    "salud_fisica": 5,
    "dificultad_para_caminar": 0,
    "accidente_cerebrovascular": 0,
    "hipertension": 1
  }'
```

---

## Despliegue en Heroku (Alternativa)

### Requisitos previos
1. Crear cuenta en https://heroku.com
2. Instalar Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli

### Pasos
```bash
# 1. Login en Heroku
heroku login

# 2. Crear aplicación
heroku create ai-diabetes-predictor

# 3. Desplegar
git push heroku main

# 4. Abrir aplicación
heroku open

# 5. Ver logs (si hay problemas)
heroku logs --tail
```

### Configuración adicional para Heroku
Si tienes problemas con el buildpack:
```bash
heroku buildpacks:set heroku/python
```

---

## Despliegue con Docker (Local o VPS)

### Build y ejecución local
```bash
# Build
docker build -t ai-diabetes .

# Run
docker run -p 5000:5000 ai-diabetes

# Acceder en: http://localhost:5000
```

### Deploy en servidor VPS (DigitalOcean, AWS, etc.)
```bash
# 1. SSH a tu servidor
ssh usuario@tu-servidor.com

# 2. Clonar repositorio
git clone https://github.com/Valkiriam7/AI-Diabetes.git
cd AI-Diabetes

# 3. Build y run con Docker
docker build -t ai-diabetes .
docker run -d -p 80:5000 --name ai-diabetes-api ai-diabetes

# 4. Opcional: usar docker-compose
# Crear docker-compose.yml y ejecutar:
docker-compose up -d
```

---

## Interfaz Streamlit (Front-end)

### Opción 1: Desplegar en Streamlit Cloud (Gratis)
1. Ve a https://streamlit.io/cloud
2. Conecta tu cuenta de GitHub
3. Click en **"New app"**
4. Selecciona:
   - Repository: `Valkiriam7/AI-Diabetes`
   - Branch: `main`
   - Main file path: `app_front.py`
5. Click en **"Deploy!"**

### Opción 2: Ejecutar localmente
```bash
streamlit run app_front.py
```
Asegúrate de actualizar la URL de la API en la interfaz para apuntar a tu API desplegada.

---

## Verificación Post-Despliegue

### Test de la API
```python
import requests

url = "https://tu-app.onrender.com/predict"
data = {
    "edad": 65,
    "colesterol_alto": 1,
    "imc": 35.2,
    "enfermedad_cardiaca_o_infarto": 1,
    "salud_general": 4,
    "salud_fisica": 15,
    "dificultad_para_caminar": 1,
    "accidente_cerebrovascular": 0,
    "hipertension": 1
}

response = requests.post(url, json=data)
print(response.json())
```

---

## Solución de Problemas Comunes

### Error: "Application Error" en Render/Heroku
1. Verifica los logs: En Render dashboard → "Logs" o `heroku logs --tail`
2. Revisa que `requirements.txt` tenga todas las dependencias
3. Confirma que `Procfile` esté correcto

### Error: "Module not found"
Añade el módulo faltante a `requirements.txt` y redeploy

### Error 500 en /predict
1. Verifica que el archivo de modelo (`random_forest_diabetes_model.joblib`) esté en el repo
2. Confirma que los nombres de las columnas en la petición coincidan

### La API funciona pero es lenta
- Considera usar un plan de pago (Free tier tiene cold starts)
- Optimiza el modelo o usa un modelo más ligero
- Implementa caché para predicciones frecuentes

---

## Monitoreo y Mantenimiento

### Render
- **Logs**: Dashboard → Logs
- **Métricas**: Dashboard → Metrics (CPU, memoria, requests)
- **Auto-deploy**: Por defecto, cada push a `main` redeploys automáticamente

### Heroku
```bash
# Ver logs en tiempo real
heroku logs --tail

# Ver estado de la app
heroku ps

# Reiniciar la aplicación
heroku restart
```

---

## Próximos Pasos Recomendados

1. **Dominio personalizado**: Configura un dominio custom en Render/Heroku
2. **HTTPS**: Ya incluido por defecto en Render y Heroku
3. **Rate limiting**: Implementa límites de requests para evitar abuso
4. **Autenticación**: Añade API keys si es necesario
5. **Monitoreo avanzado**: Integra Sentry o New Relic
6. **CI/CD**: Configura tests automáticos con GitHub Actions

---

## Recursos Adicionales

- [Documentación de Render](https://render.com/docs)
- [Documentación de Heroku](https://devcenter.heroku.com/)
- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-community-cloud)
- [Docker Documentation](https://docs.docker.com/)

¿Necesitas ayuda? Abre un issue en el repositorio: https://github.com/Valkiriam7/AI-Diabetes/issues
