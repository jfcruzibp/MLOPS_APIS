import pandas as pd
from data.Contruir_DataSet import DataSetBuilder
from sklearn.model_selection import train_test_split

class DataPreparation:
    def __init__(self):
        pass # No necesitas super().__init__() a menos que heredes de otra clase

    @staticmethod
    def load_and_prepare_data(file_path):
        data = pd.read_csv(file_path)
        data = data.dropna()    
        data = pd.get_dummies(data, drop_first=True)
        
        # Ajustado al nombre real de tu variable objetivo
        target_col = 'target_opened' 
        X = data.drop(target_col, axis=1)
        y = data[target_col]
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    # 1. Construir el dataset
    DataSetBuilder.build_dataset()
    
    # 2. Preparar los datos
    prep = DataPreparation()
    X_train, X_test, y_train, y_test = prep.load_and_prepare_data('data/raw/open_rate_dataset.csv')
    
    print(f"Datos preparados: {X_train.shape[0]} registros de entrenamiento.")


    