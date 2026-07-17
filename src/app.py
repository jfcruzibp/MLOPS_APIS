import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel, Field

# Aseguramos que Python encuentre módulos locales en la carpeta 'src'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="API de Clasificación de Open Rate - MLOPS",
    description="API para predecir si un correo será abierto usando modelos entrenados en MLflow.",
    version="1.4.0"
)

# Modelos permitidos en este caso de uso
VALID_MODELS = ["LogisticRegression", "RandomForest"]


# =====================================================================
# ESQUEMA DE ENTRADA EXPLICITO (Basado en tus datos reales)
# =====================================================================
class SessionInput(BaseModel):
    user_id: str = Field(..., description="ID del usuario", examples=["USR_00000001"])
    site: str = Field(..., description="Sección del sitio (home, content, product, search)", examples=["home"])
    campaign_type: str = Field(..., description="Tipo de campaña (promo, transactional, engagement, re-engagement)", examples=["promo"])
    device_os: str = Field(..., description="Sistema operativo (ios, android, web)", examples=["ios"])
    hour_of_day: int = Field(..., description="Hora del día (0-23)", examples=[20])
    day_of_week: int = Field(..., description="Día de la semana (0-6)", examples=[3])
    historical_open_rate: float = Field(..., description="Tasa de apertura histórica (0.0 - 1.0)", examples=[0.4793])
    historical_push_count: int = Field(..., description="Cantidad de notificaciones push históricas", examples=[11])
    days_since_last_open: int = Field(..., description="Días desde la última apertura de correo", examples=[37])
    segment: str = Field(..., description="Segmento de usuario (at_risk, active, new, churn)", examples=["at_risk"])


def get_latest_model_from_mlflow(model_name: str):
    """
    Busca en MLflow el último 'run' exitoso para el modelo solicitado
    y lo carga en memoria.
    """
    try:
        experiment = mlflow.get_experiment_by_name("open-rate-classification")
        if not experiment:
            raise HTTPException(
                status_code=404, 
                detail="El experimento 'open-rate-classification' no existe. ¿Ya ejecutaste el entrenamiento?"
            )
        
        # Buscar ejecuciones filtrando por el parámetro 'model_type'
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"params.model_type = '{model_name}'",
            order_by=["attributes.start_time DESC"],
            max_results=1
        )
        
        if runs.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No se encontró ningún modelo entrenado en MLflow para: {model_name}"
            )
        
        # Obtener el run_id más reciente y cargar el modelo
        run_id = runs.iloc[0]["run_id"]
        model_uri = f"runs:///{run_id}/model"
        
        print(f"-> Cargando {model_name} desde el run {run_id}...")
        return mlflow.sklearn.load_model(model_uri)
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500, 
            detail=f"Error al conectar con MLflow: {str(e)}"
        )


# =====================================================================
# SERVICIO 1: PREDICCIÓN POR REGISTRO INDIVIDUAL (POST)
# =====================================================================
@app.post(
    "/predict",
    summary="Predicción Individual por JSON",
    description=(
        "### 📝 ¿Cómo funciona?\n"
        "Este servicio toma un registro de sesión con la estructura exacta de tus datos en formato JSON, "
        "lo procesa con la codificación One-Hot dummy (1515 columnas molde de entrenamiento) "
        "y predice si el correo será abierto (`target_opened`).\n\n"
        "### 💻 Ejemplo de Ejecución (Curl):\n"
        "```bash\n"
        "curl -X 'POST' '[http://127.0.0.1:8000/predict?model_type=LogisticRegression](http://127.0.0.1:8000/predict?model_type=LogisticRegression)' \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        "  -d '{\n"
        "    \"user_id\": \"USR_00000001\",\n"
        "    \"site\": \"home\",\n"
        "    \"campaign_type\": \"promo\",\n"
        "    \"device_os\": \"ios\",\n"
        "    \"hour_of_day\": 20,\n"
        "    \"day_of_week\": 3,\n"
        "    \"historical_open_rate\": 0.4793,\n"
        "    \"historical_push_count\": 11,\n"
        "    \"days_since_last_open\": 37,\n"
        "    \"segment\": \"at_risk\"\n"
        "  }'\n"
        "```"
    )
)
def predict(
    input_data: SessionInput = Body(...),
    model_type: str = Query(
        ..., 
        description="Tipo de modelo a utilizar ('LogisticRegression' o 'RandomForest')"
    )
):
    if model_type not in VALID_MODELS:
        raise HTTPException(
            status_code=400, 
            detail=f"Modelo no soportado. Elige uno de estos: {VALID_MODELS}"
        )
    
    model = get_latest_model_from_mlflow(model_type)
    
    try:
        # Cargar la ruta de tu archivo CSV de datos para obtener el molde de columnas dummies
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        dataset_path = os.path.join(base_dir, "data", "raw", "session_data.csv")
        
        if not os.path.exists(dataset_path):
            raise HTTPException(
                status_code=500, 
                detail="No se encontró el archivo 'session_data.csv' para generar el molde."
            )
        
        df_train = pd.read_csv(dataset_path)
        df_train = df_train.dropna()
        
        target_col = 'target_opened'
        if target_col in df_train.columns:
            df_train = df_train.drop(target_col, axis=1)
            
        df_train_dummies = pd.get_dummies(df_train, drop_first=True)
        trained_columns = df_train_dummies.columns  
        
        # Convertir datos de entrada a DataFrame
        user_raw_df = pd.DataFrame([input_data.model_dump()])
        user_raw_df = user_raw_df.dropna()
        
        # Convertir a variables dummies y alinear con el molde
        user_dummies = pd.get_dummies(user_raw_df, drop_first=True)
        user_prepared = user_dummies.reindex(columns=trained_columns, fill_value=0)
        
        # Predicción
        prediction = model.predict(user_prepared)
        
        try:
            probabilities = model.predict_proba(user_prepared)[0]
            probability = float(probabilities[1])
        except AttributeError:
            probability = None

        return {
            "status": "success",
            "model_used": model_type,
            "prediction": int(prediction[0]),
            "probability_opened": probability
        }
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500, 
            detail=f"Error al procesar el dato o ejecutar la predicción: {str(e)}"
        )


# =====================================================================
# SERVICIO 2: PREDICCIÓN POR SECCIÓN DE LA WEB / SITE (GET)
# =====================================================================
@app.get(
    "/predict/by_site",
    summary="Predicción Promedio por Sección de la Web (Site)",
    description=(
        "### 📊 ¿Cómo funciona?\n"
        "Filtra los datos del archivo `session_data.csv` según el valor de la columna `site` (sección web). "
        "Calcula el promedio para las variables numéricas y la moda (valor más frecuente) para las categóricas "
        "de modo que se genera un perfil representativo de comportamiento de navegación de ese 'site', "
        "y se realiza la predicción.\n\n"
        "### 💻 Secciones Válidas:\n"
        "* `home` (Página de inicio)\n"
        "* `content` (Contenido / Blogs)\n"
        "* `product` (Fichas de Producto)\n"
        "* `search` (Buscador)\n\n"
        "### 💡 Ejemplo de Ejecución (URL):\n"
        "Puedes consultar de forma directa ingresando en tu navegador:\n"
        "`http://127.0.0.1:8000/predict/by_site?model_type=RandomForest&site_name=home`"
    )
)
def predict_by_site(
    site_name: str = Query(
        ..., 
        description="Sección de la web a evaluar. Opciones válidas: 'home', 'content', 'product', 'search'"
    ),
    model_type: str = Query(
        ..., 
        description="Tipo de modelo a utilizar ('LogisticRegression' o 'RandomForest')"
    )
):
    if model_type not in VALID_MODELS:
        raise HTTPException(
            status_code=400, 
            detail=f"Modelo no soportado. Elige uno de estos: {VALID_MODELS}"
        )
        
    model = get_latest_model_from_mlflow(model_type)
    
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        dataset_path = os.path.join(base_dir, "data", "raw", "session_data.csv")
        
        if not os.path.exists(dataset_path):
            raise HTTPException(
                status_code=500, 
                detail="No se encontró el archivo 'session_data.csv'."
            )
            
        df_original = pd.read_csv(dataset_path)
        df_original = df_original.dropna()
        
        location_col = 'site'
        
        if location_col not in df_original.columns:
            raise HTTPException(
                status_code=500,
                detail=f"No se encontró la columna '{location_col}' en el CSV."
            )
            
        # Filtrar registros correspondientes al site enviado
        df_filtered = df_original[df_original[location_col].astype(str).str.lower() == site_name.strip().lower()]
        
        if df_filtered.empty:
            available_sites = df_original[location_col].dropna().unique().tolist()
            raise HTTPException(
                status_code=404,
                detail=f"No se encontraron registros para la sección '{site_name}'. Valores disponibles en tu CSV: {available_sites}"
            )
            
        # Generar el registro promedio de ese 'site'
        average_record = {}
        for col in df_original.columns:
            if col == 'target_opened':
                continue
            if pd.api.types.is_numeric_dtype(df_original[col]):
                average_record[col] = float(df_filtered[col].mean())
            else:
                mode_val = df_filtered[col].mode()
                average_record[col] = str(mode_val.iloc[0]) if not mode_val.empty else str(df_original[col].mode().iloc[0])
        
        # Seteamos el site que se consultó
        average_record[location_col] = site_name
        
        # Obtener molde de características de entrenamiento
        df_train = df_original.drop('target_opened', axis=1, errors='ignore')
        df_train_dummies = pd.get_dummies(df_train, drop_first=True)
        trained_columns = df_train_dummies.columns
        
        # Procesar y alinear el registro promedio
        df_profile_raw = pd.DataFrame([average_record])
        df_profile_dummies = pd.get_dummies(df_profile_raw, drop_first=True)
        df_profile_prepared = df_profile_dummies.reindex(columns=trained_columns, fill_value=0)
        
        # Predicción
        prediction = model.predict(df_profile_prepared)
        
        try:
            probabilities = model.predict_proba(df_profile_prepared)[0]
            probability = float(probabilities[1])
        except AttributeError:
            probability = None
            
        profile_summary = {
            k: (round(v, 4) if isinstance(v, float) else v) 
            for k, v in average_record.items() 
            if not k.endswith('_id')
        }

        return {
            "status": "success",
            "site_evaluated": site_name,
            "model_used": model_type,
            "matching_records_found": len(df_filtered),
            "profile_analyzed": profile_summary,
            "prediction": int(prediction[0]),
            "probability_opened": probability
        }
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=f"Error interno al procesar la predicción por sección de la web: {str(e)}"
        )