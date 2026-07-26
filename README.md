# MLOPS_APIS
### Proyecto final de MLOPS 
### Api de predicción basado en YAHOO Finance
Creacion de Apis con la estructura de MLOPS 


## LINK DESPLIEGUE EN LA NUBE DE GOOGLE CLOUD RUN 

#### https://mi-api-service-951371977481.us-central1.run.app/docs


## Presentado por 
    Julián Javier Gómez Reyes
    John Fernando Cruz Becerra
    Mauricio Alejandro Gaviria Alzate


### Universidad Santo Tomás
### Maestría en Ciencia de Datos

## Instalación
### Clonar el repositorio:
git clone https://github.com/jfcruzibp/MLOPS_APIS.git MLOPS_APIS
cd MLOPS_APIS

### Instalar dependencias:
poetry install

### Ejecutar
poetry run uvicorn src.financial_api.api:app --reload
poetry run pytest
docker build -t financial-api:local .
docker run --rm -p 8000:8000 financial-api:local

Abrir en navegador:
http://127.0.0.1:8000/docs