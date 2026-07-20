from datetime import datetime
import os
import pandas as pd
import yfinance as yf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
RAW_CSV_PATH = os.path.join(DATA_RAW_DIR, "session_data.csv")


class FinancialDataLoader:
    """Clase encargada de la ingesta de datos desde Yahoo Finance y el manejo

    del respaldo local CSV en la carpeta data/.
    """

    def __init__(self, tickers: list[str] | str = None):
        if tickers is None:
            tickers = ["AAPL", "MSFT", "NVDA"]
        self.tickers = [tickers] if isinstance(tickers, str) else tickers

    def fetch_from_yfinance(
        self, periodo: str = "1y", intervalo: str = "1d"
    ) -> pd.DataFrame:
        """Descarga datos recientes directo desde yfinance."""
        df = yf.download(
            tickers=self.tickers,
            period=periodo,
            interval=intervalo,
            auto_adjust=True,
            progress=False,
        )
        return df

    def save_raw_csv(self, df: pd.DataFrame, filepath: str = RAW_CSV_PATH):
        """Guarda los datos descargados en la carpeta data/raw/."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath)
        print(f"-> Datos crudos guardados exitosamente en: {filepath}")

    def load_local_raw_csv(self, filepath: str = RAW_CSV_PATH) -> pd.DataFrame:
        """Carga el dataset crudo local reproducible para operar sin internet."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"No se encontró el dataset local en '{filepath}'."
            )
        return pd.read_csv(filepath, header=[0, 1], index_col=0, parse_dates=True)

    def check_and_update_dataset(self):
        """Verifica si el dataset procesado existe y si está actualizado a la fecha de hoy.

        Si no existe o está desactualizado, descarga e instala todo
        automáticamente.
        """
        # Importación diferida para evitar ciclos de importación circular
        from src.financial_api.features import FeatureEngineer
        from src.financial_api.train import ModelTrainer

        processed_path = os.path.join(
            BASE_DIR, "data", "processed", "processed_financial_data.csv"
        )
        needs_update = False
        today_str = datetime.now().strftime("%Y-%m-%d")

        if not os.path.exists(processed_path):
            print(
                "-> [AUTO-SETUP] No se encontró dataset local. Iniciando descarga..."
            )
            needs_update = True
        else:
            try:
                df = pd.read_csv(processed_path)
                if "Date" in df.columns:
                    last_date = str(df["Date"].max())[:10]
                    if last_date < today_str:
                        print(
                            f"-> [AUTO-SETUP] Dataset desactualizado (Última fecha: {last_date}). Actualizando a {today_str}..."
                        )
                        needs_update = True
            except Exception:
                needs_update = True

        if needs_update:
            # 1. Ingesta utilizando los tickers configurados en la instancia
            df_raw = self.fetch_from_yfinance(periodo="1y")
            self.save_raw_csv(df_raw)

            # 2. Generación y guardado de Features
            close_df = df_raw["Close"] if "Close" in df_raw else df_raw
            df_processed = FeatureEngineer.create_features_for_df(close_df)
            FeatureEngineer().save_processed_csv(df_processed)

            # 3. Entrenamiento automático del modelo
            trainer = ModelTrainer(model_type="RandomForest")
            trainer.train(df_processed)
            print("-> [AUTO-SETUP] ¡Dataset y modelo actualizados exitosamente!")