import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Body, Query
import pandas as pd

from src.financial_api.data import FinancialDataLoader
from src.financial_api.features import FeatureEngineer
from src.financial_api.predict import PredictorService
from src.financial_api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)


# =====================================================================
# CICLO DE VIDA DE LA API (Lifespan: Ejecución automática al arrancar)
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ejecuta la validación y actualización del dataset local y el entrenamiento

    del modelo automáticamente al encender el servidor.
    """
    try:
        # Se verifica/actualiza para los tickers base de entrenamiento
        loader = FinancialDataLoader(tickers=["AAPL", "MSFT", "NVDA"])
        loader.check_and_update_dataset()
    except Exception as e:
        print(f"⚠️ Advertencia durante auto-setup inicial: {e}")
    yield


# =====================================================================
# INICIALIZACIÓN DE FASTAPI Y SERVICIOS
# =====================================================================
app = FastAPI(
    title="Universidad Santo Tomas - MLOps",
    description="\n* API de Clasificación de Tendencia Financiera\n* API para predecir si el retorno del siguiente día será positivo o negativo utilizando yfinance. \n- Presentado por: \n* John fernando Cruz Becerra\n* Mauricio Alejandro Gaviria Alzate\n* Julián Javier Gómez",
    version="1.0.0",
    lifespan=lifespan,  # Registro del lifespan
)

predictor = PredictorService()


# =====================================================================
# ENDPOINTS
# =====================================================================
@app.get(
    "/health", response_model=HealthResponse, summary="Verificar salud de la API"
)
def health_check():
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    model_exists = os.path.exists(
        os.path.join(base_dir, "artifacts", "model.joblib")
    )
    dataset_exists = os.path.exists(
        os.path.join(
            base_dir, "data", "processed", "processed_financial_data.csv"
        )
    )

    status = "healthy" if (model_exists and dataset_exists) else "degraded"
    return HealthResponse(
        status=status,
        model_available=model_exists,
        dataset_available=dataset_exists,
    )


@app.get(
    "/market-data/{symbol}", summary="Devolver datos recientes del activo"
)
def get_market_data(symbol: str, limit: int = Query(default=5, ge=1, le=100)):

    """"Obtiene los últimos registros de datos financieros para un símbolo específico desde yfinance
    o desde el dataset local procesado, dependiendo de la disponibilidad y configuración.
    EJEMPLOS DE PARAMETROS:
    - AAPL
    - MSFT
    - NVDA 
    """

    try:
        loader = FinancialDataLoader(tickers=symbol)
        df_raw = loader.fetch_from_yfinance(periodo="1mo")
        df_features = FeatureEngineer.create_features_for_df(
            df_raw["Close"] if "Close" in df_raw else df_raw
        )

        recent_records = df_features.tail(limit).to_dict(orient="records")
        return {
            "symbol": symbol.upper(),
            "count": len(recent_records),
            "data": recent_records,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al obtener mercado: {str(e)}"
        )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Realizar predicción",
)
def predict(request: PredictionRequest = Body(...)):
    """Realiza la predicción de tendencia para un símbolo específico.
    Si el usuario solicita usar datos en caché y estos existen, se utilizarán.
    NO SE REQUIERE ENTRENAMIENTO MANUAL: el modelo se carga automáticamente desde los artefactos.
    NO REQUIERE PARAMETROS DE ENTRADA ADICIONALES: el endpoint maneja la obtención de datos y la creación de features internamente.
    """
    try:
        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        processed_path = os.path.join(
            base_dir,
            "data",
            "processed",
            "processed_financial_data.csv",
        )

        # Si el usuario solicita usar la caché local y el archivo existe
        if request.use_cached_data and os.path.exists(processed_path):
            df_processed = pd.read_csv(processed_path)
            symbol_df = df_processed[
                df_processed["Symbol"].str.upper() == request.symbol.upper()
            ]
        else:
            # Consulta dinámica a yfinance en tiempo real
            loader = FinancialDataLoader(tickers=request.symbol)
            df_raw = loader.fetch_from_yfinance(periodo="3mo")
            symbol_df = FeatureEngineer.create_features_for_df(
                df_raw["Close"] if "Close" in df_raw else df_raw
            )

        if symbol_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No hay registros para el símbolo '{request.symbol}'.",
            )

        result = predictor.predict_symbol(
            symbol=request.symbol, feature_vector=symbol_df
        )
        return PredictionResponse(**result)

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error en la predicción: {str(e)}"
        )


@app.get("/model/metadata", summary="Metadatos del modelo")
def get_model_metadata():
    try:
        predictor.load_artifacts()
        return predictor.metadata
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Error al cargar metadatos: {str(e)}"
        )