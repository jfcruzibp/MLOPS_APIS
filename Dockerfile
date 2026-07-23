# Usar imagen oficial de Python 3.13 slim
FROM python:3.13-slim

# Establecer variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.1.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    PATH="/opt/poetry/bin:$PATH"

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Instalar Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias primero (mejor cache)
COPY pyproject.toml poetry.lock ./

# Instalar dependencias del proyecto
RUN poetry install --no-interaction --no-ansi

# Copiar el resto del código
COPY . .

# Crear directorios necesarios
RUN mkdir -p /app/data/raw /app/data/processed /app/artifacts

# Exponer puerto de FastAPI
EXPOSE 8000

# Comando para ejecutar la API con uvicorn
CMD ["uvicorn", "src.financial_api.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]