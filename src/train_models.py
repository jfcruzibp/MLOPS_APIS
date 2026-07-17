import mlflow
import mlflow.sklearn
import sys
import os


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_preparation import DataPreparation  


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# from src.data_preparation import DataPreparation


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions)
    }


def train_and_log(model_name, model, X_train, X_test, y_train, y_test):

    with mlflow.start_run(run_name=model_name):

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        mlflow.log_param("model_type", model_name)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        return metrics


def main():

    X_train, X_test, y_train, y_test = (
        DataPreparation.load_and_prepare_data(
            "data/raw/session_data.csv"
        )
    )

    mlflow.set_experiment("open-rate-classification")

    results = {}

    models = {
        "LogisticRegression":
            LogisticRegression(max_iter=1000),

        "RandomForest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
    }

    for model_name, model in models.items():

        metrics = train_and_log(
            model_name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results[model_name] = metrics

    print("\n===== RESULTADOS =====\n")

    for model_name, metrics in results.items():

        print(model_name)

        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        print()

    best_model = max(
        results,
        key=lambda x: results[x]["f1"]
    )

    print(f"MEJOR MODELO: {best_model}")
    print(
        f"F1 = {results[best_model]['f1']:.4f}"
    )


if __name__ == "__main__":
    main()