import json
import os
import joblib
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "model.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "artifacts", "model_metadata.json")


class PredictorService:
    """Clase encargada de cargar los artefactos del modelo y realizar inferencias."""

    def __init__(self):
        self.model = None
        self.metadata = None

    def load_artifacts(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
            raise FileNotFoundError("Artefactos de modelo o metadatos no encontrados en 'artifacts/'.")

        self.model = joblib.load(MODEL_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def predict_symbol(self, symbol: str, feature_vector: pd.DataFrame) -> dict:
        if self.model is None or self.metadata is None:
            self.load_artifacts()

        feature_cols = self.metadata.get("feature_names", ["Return_1D", "Volatility_5D", "SMA_10_Ratio"])
        X = feature_vector[feature_cols].tail(1)

        pred_class = self.model.predict(X)[0]
        prob_up = float(self.model.predict_proba(X)[0][1]) if hasattr(self.model, "predict_proba") else 1.0

        return {
            "symbol": symbol.upper(),
            "prediction": "up" if pred_class == 1 else "down",
            "probability_up": round(prob_up, 4),
            "model_version": self.metadata.get("model_version", "random_forest_v1"),
            "prediction_horizon": "next_day"
        }