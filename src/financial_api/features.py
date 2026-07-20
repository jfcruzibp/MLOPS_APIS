import os
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PROCESSED_CSV_PATH = os.path.join(DATA_PROCESSED_DIR, "processed_financial_data.csv")


class FeatureEngineer:
    """Clase encargada de transformar datos financieros crudos en características (features)
    y definir la variable objetivo para modelos de clasificación.
    """

    @staticmethod
    def create_features_for_df(df_close) -> pd.DataFrame:
        """Genera features técnicas por activo garantizando compatibilidad de formatos."""
        processed_dfs = []

        # 1. Si es una Series (un solo ticker), la convertimos a DataFrame
        if isinstance(df_close, pd.Series):
            ticker_name = df_close.name if df_close.name else "ASSET"
            df_work = pd.DataFrame({ticker_name: df_close})
        elif isinstance(df_close, pd.DataFrame):
            df_work = df_close.copy()
        else:
            raise ValueError("El formato de datos ingresado no es válido.")

        # 2. Iterar sobre las columnas (tickers)
        for ticker in df_work.columns:
            series = df_work[ticker].dropna()

            if series.empty:
                continue

            # Crear DataFrame individual asegurando el índice
            df_single = pd.DataFrame({"Close": series}, index=series.index)
            df_single["Close"] = df_single["Close"].ffill().bfill()

            df_single["Symbol"] = str(ticker).upper()
            df_single["Return_1D"] = df_single["Close"].pct_change()
            df_single["Volatility_5D"] = df_single["Return_1D"].rolling(5).std()
            df_single["SMA_10_Ratio"] = (
                df_single["Close"] / df_single["Close"].rolling(10).mean()
            )

            # Target: 1 si el retorno del SIGUIENTE día es positivo, 0 si es negativo
            df_single["Target"] = (
                df_single["Return_1D"].shift(-1) > 0
            ).astype(int)

            # Limpiar filas iniciales con NaNs por los rolling
            df_single = df_single.dropna()
            processed_dfs.append(df_single)

        if not processed_dfs:
            raise ValueError(
                "No se pudieron generar características. Verifica que el ticker exista o tenga datos."
            )

        final_df = pd.concat(processed_dfs).reset_index()
        return final_df

    def save_processed_csv(self, df: pd.DataFrame, filepath: str = PROCESSED_CSV_PATH):
        """Guarda el DataFrame procesado en la carpeta data/processed/."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"-> Dataset procesado guardado exitosamente en: {filepath}")