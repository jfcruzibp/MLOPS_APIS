import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Extra

# Aseguramos que Python encuentre módulos locales en la carpeta 'src'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="API de Clasificación de Open Rate - MLOPS",
    description="API para predecir si un correo será abierto usando modelos entrenados en MLflow.",
    version="1.2.0"
)

# Modelos permitidos en este caso de uso
VALID_MODELS = ["LogisticRegression", "RandomForest"]

# Esquema Pydantic dinámico que acepta cualquier JSON con la estructura del CSV
class DynamicInput(BaseModel):
    class Config:
        extra = Extra.allow  # Permite cualquier columna del dataset original sin fallar


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


@app.post("/predict")
def predict(
    input_data: DynamicInput,
    model_type: str = Query(
        ..., 
        description="Tipo de modelo a utilizar ('LogisticRegression' o 'RandomForest')"
    )
):
    # 1. Validar el modelo solicitado
    if model_type not in VALID_MODELS:
        raise HTTPException(
            status_code=400, 
            detail=f"Modelo no soportado. Elige uno de estos: {VALID_MODELS}"
        )
    
    # 2. Cargar el modelo correspondiente desde MLflow
    model = get_latest_model_from_mlflow(model_type)
    
    try:
        # 3. Construir la ruta absoluta de 'session_data.csv' de forma segura en Windows
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        dataset_path = os.path.join(base_dir, "data", "raw", "session_data.csv")
        
        if not os.path.exists(dataset_path):
            print(f"DEBUG: Buscando archivo en -> {dataset_path}")
            raise HTTPException(
                status_code=500, 
                detail=f"No se encontró el archivo 'session_data.csv' en {dataset_path} para generar el molde."
            )
        
        # 4. Leer el dataset original para armar el molde exacto de 1515 columnas
        df_train = pd.read_csv(dataset_path)
        df_train = df_train.dropna()
        
        # Eliminar la variable objetivo 'target_opened' del molde de características (X)
        target_col = 'target_opened'
        if target_col in df_train.columns:
            df_train = df_train.drop(target_col, axis=1)
            
        df_train_dummies = pd.get_dummies(df_train, drop_first=True)
        trained_columns = df_train_dummies.columns  # Aquí están las columnas que espera el modelo
        
        # 5. Procesar la entrada del usuario recibida en el JSON
        user_raw_df = pd.DataFrame([input_data.model_dump()])
        user_raw_df = user_raw_df.dropna()
        
        # Convertir variables categóricas a dummies para la fila del usuario
        user_dummies = pd.get_dummies(user_raw_df, drop_first=True)
        
        # 6. Alineación de columnas (Reindexar)
        # Esto rellena automáticamente con 0 las columnas dummy que falten en el JSON
        user_prepared = user_dummies.reindex(columns=trained_columns, fill_value=0)
        
        # 7. Ejecutar la predicción
        prediction = model.predict(user_prepared)
        
        # Intentar obtener la probabilidad de la clase positiva (si el modelo lo soporta)
        try:
            probabilities = model.predict_proba(user_prepared)[0]
            probability = float(probabilities[1])
        except AttributeError:
            probability = None

        # 8. Devolver la respuesta en un JSON limpio
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