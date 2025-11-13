Despliegue rápido del proyecto AI-Diabetes

Opciones soportadas:

1) Deploy en Render (recomendado para facilidad)
- Crear cuenta en https://render.com
- Nuevo servicio -> Web Service -> Conectar a GitHub -> seleccionar repo `AI-Diabetes`
- Branch: `main`
- Build command: `pip install -r requirements.txt`
- Start command: `python app_api.py`
- Añadir variables de entorno si fuese necesario

2) Deploy en Heroku
- Instalar Heroku CLI y loguearte
- Crear app: `heroku create nombre-app`
- Subir por Git:
  - `git push heroku main`
- Si usas Docker:
  - `heroku stack:set container`
  - `heroku container:push web`
  - `heroku container:release web`

3) Deploy con Docker local o VPS
- Build: `docker build -t ai-diabetes .`
- Run: `docker run -p 5000:5000 ai-diabetes`

Notas:
- Asegúrate de que los archivos de modelo (`random_forest_diabetes_model.joblib` o `modelo_diabetes_ajustado.pkl`) están en el repo o en un storage accesible.
- Para producción usa un WSGI server (gunicorn/uvicorn) en lugar del servidor de desarrollo Flask.

