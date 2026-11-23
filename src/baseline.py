from pathlib import Path
import pandas as pd
from sklearn.dummy import DummyClassifier
from src.constants import (
    ALL_FEATURES,
    ID_COL,
    RANDOM_STATE,
    TARGET_COL,
    TRAIN_PATH,
    VAL_PATH,
)

# Если общий модуль метрик уже есть в проекте - берём его
# Если ещё нет - временно считаем метрики напрямую через sklearn
try:
    from src.metrics import compute_metrics
except ImportError:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

    def compute_metrics(y_true, y_pred):
        return {
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }


MODEL_NAME = "baseline_majority"
PREDICTIONS_DIR = Path("results") / "predictions"
VALIDATION_DIR = Path("results") / "validation"


# Обучает модель, которая всегда предсказывает самый частый класс в train
def train_baseline(df_train, features=None, params=None, random_state=RANDOM_STATE):
    if features is None:
        features = ALL_FEATURES

    strategy = "most_frequent"
    if params is not None and "strategy" in params:
        strategy = params["strategy"]

    X_train = df_train[features]
    y_train = df_train[TARGET_COL]

    model = DummyClassifier(strategy=strategy, random_state=random_state)
    model.fit(X_train, y_train)
    return model


# Делает предсказания обученной базовой моделью
def predict_baseline(model, df, features=None):
    if features is None:
        features = ALL_FEATURES
    return model.predict(df[features])


# Сохраняет предсказания в results/predictions/baseline_majority_<выборка>.csv.
def save_predictions(df, y_pred, split_name):
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PREDICTIONS_DIR / f"{MODEL_NAME}_{split_name}.csv"

    pred_df = pd.DataFrame(
        {
            ID_COL: df[ID_COL].values,
            "y_true": df[TARGET_COL].values,
            "y_pred": y_pred,
            "model": MODEL_NAME,
        }
    )
    pred_df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


# Загружает train и val, обучает baseline, сохраняет предсказания и метрики
def run_baseline_validation():
    df_train = pd.read_csv(TRAIN_PATH, encoding="utf-8")
    df_val = pd.read_csv(VAL_PATH, encoding="utf-8")

    # Обучаем модель только на train
    model = train_baseline(df_train, features=ALL_FEATURES, random_state=RANDOM_STATE)

    # Считаем предсказания и метрики для train и val
    rows = []
    for split_name, df_split in [("train", df_train), ("val", df_val)]:
        y_true = df_split[TARGET_COL].values
        y_pred = predict_baseline(model, df_split, features=ALL_FEATURES)

        save_predictions(df_split, y_pred, split_name)
        metrics = compute_metrics(y_true, y_pred)

        rows.append(
            {
                "model": MODEL_NAME,
                "split": split_name,
                "balanced_accuracy": round(metrics["balanced_accuracy"], 4),
                "accuracy": round(metrics["accuracy"], 4),
                "confusion_matrix": str(metrics["confusion_matrix"]),
            }
        )

    # Сохраняем итоговую таблицу метрик
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(VALIDATION_DIR / "baseline_metrics.csv", index=False, encoding="utf-8")

    return metrics_df


if __name__ == "__main__":
    results = run_baseline_validation()
    print(results.to_string(index=False))