# Imagen base
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de requirements y luego instalar
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar la aplicación
COPY . /app

# Exponer puerto
EXPOSE 5000

# Comando por defecto
CMD ["python", "app_api.py"]
