"""
Единый модуль констант проекта «ПассажирМаршрут».
Содержит общие названия колонок, списки признаков, допустимые значения категорий,
стандартные имена моделей и пути к основным файлам.
"""

from pathlib import Path

# Общие параметры воспроизводимости
RANDOM_STATE: int = 42  # Фиксированный seed для разбиения выборки и обучения моделей
GENERATOR_VERSION: str = "v2"  # Версия генератора данных для итоговых экспериментов

# Схема датасета и списки признаков
ID_COL: str = "passenger_id"  # Уникальный идентификатор пассажира
TARGET_COL: str = "arrived_in_time"  # Целевая переменная: 1 — успел вовремя, 0 — не успел

NUM_FEATURES: list[str] = ["age", "distance_m", "luggage_count"]  # Числовые признаки
CAT_FEATURES: list[str] = ["deck", "briefing"]  # Категориальные признаки
ALL_FEATURES: list[str] = NUM_FEATURES + CAT_FEATURES  # Полный набор признаков для моделей

ID_AGE: str = "age"
ID_DIST: str = "distance_m"
ID_LUGGAGE: str = "luggage_count"
ID_BRIEFING: str = "briefing"
ID_DECK: str = "deck"

ABLATION_DROP_FEATURES: list[str] = ["distance_m", "briefing"]  # Признаки, исключаемые при абляции
REDUCED_FEATURES: list[str] = [
    col for col in ALL_FEATURES if col not in ABLATION_DROP_FEATURES
]  # Сокращённый набор признаков для абляции

HIDDEN_COLS: list[str] = [
    "actual_speed",  # Истинная скорость движения (м/с)
    "start_delay",  # Задержка перед началом движения (сек)
    "travel_time",  # Чистое время прохождения маршрута (сек)
    "random_delay",  # Случайная задержка в пути (сек)
    "total_time",  # Итоговое время прибытия к месту сбора (сек)
]  # Скрытые переменные генератора (в модели не передаются)
ID_ACTUAL_SPEED: str = "actual_speed"
ID_START_DELAY: str = "start_delay"
ID_TRAVEL_TIME: str = "travel_time"
ID_RANDOM_DELAY: str = "random_delay"
ID_TOTAL_TIME: str = "total_time"
# Допустимые значения категорий (для проверки данных и кодирования)
DECK_VALUES: list[str] = ["lower", "middle", "upper"]  # Палуба: нижняя, средняя, верхняя
BRIEFING_VALUES: list[int] = [0, 1]  # Инструктаж: 0 — не пройден, 1 — пройден

# Стандартные имена моделей и формат файлов предсказаний
MODEL_BASELINE: str = "baseline_majority"  # Базовый классификатор (самый частый класс)
MODEL_LOGREG: str = "logreg"  # Логистическая регрессия
MODEL_TREE_D3: str = "tree_d3"  # Решающее дерево глубины 3
MODEL_TREE_D5: str = "tree_d5"  # Решающее дерево глубины 5
MODEL_TREE_DNONE: str = "tree_dnone"  # Решающее дерево без ограничения глубины

PREDICTION_COLS: list[str] = [ID_COL, "y_true", "y_pred", "model"]  # Колонки файлов с прогнозами

# Пути к основным файлам и папкам
ROOT_DIR: Path = Path(__file__).resolve().parents[1]  # Корень репозитория

GENERATOR_CONFIG_PATH: Path = ROOT_DIR / "config" / "generator.yaml"  # Конфиг генератора данных
TRIAL_DATA_PATH: Path = ROOT_DIR / "data" / "trial" / "mock_passengers.csv"  # Пробный набор для отладки
RAW_DATA_PATH: Path = ROOT_DIR / "data" / "raw" / f"passengers_{GENERATOR_VERSION}_seed{RANDOM_STATE}.csv"  # Основной датасет
HIDDEN_DATA_PATH: Path = ROOT_DIR / "data" / "raw" / f"passengers_{GENERATOR_VERSION}_seed{RANDOM_STATE}_hidden.csv"  # Скрытые данные

TRAIN_PATH: Path = ROOT_DIR / "data" / "splits" / "train.csv"  # Обучающая выборка (60%)
VAL_PATH: Path = ROOT_DIR / "data" / "splits" / "val.csv"  # Валидационная выборка (20%)
TEST_PATH: Path = ROOT_DIR / "data" / "splits" / "test.csv"  # Тестовая выборка (20%)

RESULTS_DIR: Path = ROOT_DIR / "results"  # Общая папка результатов
PREDICTIONS_DIR: Path = RESULTS_DIR / "predictions"  # Папка с прогнозами моделей
FIGURES_DIR: Path = RESULTS_DIR / "figures"  # Папка с графиками для отчётов