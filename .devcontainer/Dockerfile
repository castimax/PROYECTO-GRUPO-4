FROM python:3.8

# Evita que Python escriba archivos .pyc en el disco (opcional)
ENV PYTHONDONTWRITEBYTECODE 1

# Desactiva el búfer de salida de Python (opcional)
ENV PYTHONUNBUFFERED 1

WORKDIR /workspace

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
