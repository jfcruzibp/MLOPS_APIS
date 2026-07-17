import pandas as pd
import numpy as np

class DataSetBuilder:
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples

    @staticmethod
    def build_dataset():   
        np.random.seed(42)
        n = 1000

        sites = ['site_a', 'site_b', 'site_c']
        campaign_types = ['promocional', 'transaccional', 'informativa']
        device_os = ['android', 'ios', 'web']
        segments = ['nuevo', 'activo', 'dormido', 'vip']

        data = {
            'user_id': np.arange(10000, 11000),
            'site': np.random.choice(sites, n),
            'campaign_type': np.random.choice(campaign_types, n),
            'device_os': np.random.choice(device_os, n),
            'hour_of_day': np.random.randint(0, 24, n),
            'day_of_week': np.random.randint(0, 7, n),
            'historical_open_rate': np.random.uniform(0, 1, n).round(2),
            'historical_push_count': np.random.randint(1, 50, n),
            'days_since_last_open': np.random.randint(0, 30, n),
            'segment': np.random.choice(segments, n)
        }

        prob_open = (
            data['historical_open_rate'] * 0.5 +
            (1 - data['days_since_last_open']/30) * 0.3 +
            (data['historical_push_count'] < 20) * 0.2
        )

        prob_open = np.clip(prob_open, 0, 1)

        data['target_opened'] = (
            np.random.rand(n) < prob_open
        ).astype(int)

        df = pd.DataFrame(data)
        
        print(df["target_opened"].value_counts())

        # Asegúrate de que la carpeta 'data' exista
        df.to_csv('data/raw/open_rate_dataset.csv', index=False)
        print("Dataset generado con éxito en 'data/open_rate_dataset.csv'")
