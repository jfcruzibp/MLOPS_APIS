import argparse
import numpy as np
import pandas as pd
from config import ensure_parent, load_params

# Listas de categorías ajustadas al nuevo contexto
SEGMENTS = ["new", "active", "at_risk", "churn"]
DEVICE_OS = ["android", "ios", "web"]
SITES = ["home", "search", "product", "content"]
CAMPAIGN_TYPES = ["promo", "transactional", "engagement", "re-engagement"]


def build_mock_sessions(n_samples: int, random_state: int, noise_std: float) -> pd.DataFrame:
    """Generate a reproducible mock dataset for campaign open prediction (classification)."""
    rng = np.random.default_rng(random_state)

    # 1. Generación de las columnas base solicitadas
    data = pd.DataFrame(
        {
            "user_id": [f"USR_{str(i).zfill(8)}" for i in range(1, n_samples + 1)],
            "site": rng.choice(SITES, size=n_samples),
            "campaign_type": rng.choice(CAMPAIGN_TYPES, size=n_samples, p=[0.4, 0.2, 0.3, 0.1]),
            "device_os": rng.choice(DEVICE_OS, size=n_samples, p=[0.45, 0.35, 0.2]),
            "hour_of_day": rng.integers(0, 24, size=n_samples),
            "day_of_week": rng.integers(0, 7, size=n_samples),
            "historical_open_rate": rng.uniform(0.0, 1.0, size=n_samples).round(4),
            "historical_push_count": rng.poisson(12, size=n_samples),
            "days_since_last_open": rng.integers(0, 45, size=n_samples),
            "segment": rng.choice(SEGMENTS, size=n_samples, p=[0.25, 0.45, 0.2, 0.1]),
        }
    )

    # 2. Definición de efectos sobre la probabilidad de apertura (Log-odds)
    segment_effect = data["segment"].map(
        {"new": 0.0, "active": 1.2, "at_risk": -0.8, "churn": -2.0}
    )
    campaign_effect = data["campaign_type"].map(
        {"promo": 0.2, "transactional": 2.5, "engagement": 0.5, "re-engagement": -0.5}
    )
    device_effect = data["device_os"].map(
        {"android": 0.3, "ios": 0.5, "web": -0.4}
    )
    
    # Efectos temporales
    evening_effect = np.where(data["hour_of_day"].between(18, 22), 0.4, 0.0)
    weekend_effect = np.where(data["day_of_week"].isin([5, 6]), -0.3, 0.0)
    
    # Ruido aleatorio para simular comportamiento humano impreciso
    noise = rng.normal(0.0, noise_std, size=n_samples)

    # 3. Combinación lineal (Logit)
    # Intercepto base en -1.5 para controlar la tasa global de apertura
    logit = (
        -1.5 
        + 4.0 * data["historical_open_rate"]
        - 0.05 * data["days_since_last_open"]
        + 0.02 * data["historical_push_count"]
        + segment_effect
        + campaign_effect
        + device_effect
        + evening_effect
        + weekend_effect
        + noise
    )

    # 4. Transformación Sigmoide para obtener probabilidades
    probabilities = 1 / (1 + np.exp(-logit))

    # 5. Generación de la columna objetivo binaria (0 o 1)
    data["target_opened"] = rng.binomial(1, probabilities)

    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    data_params = params["data"]

    data = build_mock_sessions(
        n_samples=int(data_params["n_samples"]),
        random_state=int(data_params["random_state"]),
        noise_std=float(data_params["noise_std"]),
    )

    output_path = ensure_parent(data_params["raw_path"])
    data.to_csv(output_path, index=False)
    print(f"Dataset mock creado en {output_path} con {len(data)} filas.")


if __name__ == "__main__":
    main()