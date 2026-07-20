from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    symbol: str = Field(..., description="Símbolo del activo financiero", examples=["AAPL"])
    prediction_horizon: int = Field(default=1, description="Horizonte de predicción en días", examples=[1])
    use_cached_data: bool = Field(default=True, description="Usar dataset local sin internet", examples=[True])


class PredictionResponse(BaseModel):
    symbol: str
    prediction: str  # "up" o "down"
    probability_up: float
    model_version: str
    prediction_horizon: str


class HealthResponse(BaseModel):
    status: str
    model_available: bool
    dataset_available: bool