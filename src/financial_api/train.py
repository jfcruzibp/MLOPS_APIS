import json
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "model_metadata.json")


class ModelTrainer:
    """Clase encarga de entrenar el modelo de clasificación y persistir los artefactos."""

    def __init__(self, model_type: str = "RandomForest"):
        self.model_type = model_type
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_cols = ["Return_1D", "Volatility_5D", "SMA_10_Ratio"]

    def train(self, df_processed: pd.DataFrame) -> dict:
        X = df_processed[self.feature_cols]
        y = df_processed["Target"]

        self.model.fit(X, y)
        preds = self.model.predict(X)
        acc = accuracy_score(y, preds)

        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)

        metadata = {
            "model_name": self.model_type,
            "model_version": "random_forest_v1",
            "symbols_used": list(df_processed["Symbol"].unique()),
            "primary_metric": {"name": "accuracy", "value": round(float(acc), 4)},
            "prediction_horizon": "next_day",
            "feature_names": self.feature_cols
        }

        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"-> Modelo guardado en: {MODEL_PATH}")
        print(f"-> Metadatos guardados en: {METADATA_PATH}")
        return metadata