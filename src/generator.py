import numpy as np
import pandas as pd
import yaml

from src.constants import (
    RANDOM_STATE,
    GENERATOR_CONFIG_PATH,
    RAW_DATA_PATH,
    HIDDEN_DATA_PATH,
    ID_COL,
    TARGET_COL,
    ID_AGE,
    ID_DECK,
    ID_DIST,
    ID_BRIEFING,
    ID_LUGGAGE,
    DECK_VALUES,
    ID_ACTUAL_SPEED,
    ID_START_DELAY,
    ID_TRAVEL_TIME,
    ID_RANDOM_DELAY,
    ID_TOTAL_TIME,
)

with open(GENERATOR_CONFIG_PATH, mode="r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

rng = np.random.default_rng(seed=RANDOM_STATE)
n = cfg["dataset"]["n_samples"]


def create_age() -> np.ndarray:
    low = cfg["distributions"]["age"]["low"]
    high = cfg["distributions"]["age"]["high"]
    return rng.integers(low, high + 1, size=n).astype(float)


def create_missing_for_age(age: np.ndarray) -> None:
    mask = rng.random(n) < cfg["dataset"]["age_missing_rate"]
    age[mask] = np.nan


def create_deck() -> np.ndarray:
    deck_cfg = cfg["distributions"]["deck"]
    return rng.choice(deck_cfg["values"], p=deck_cfg["probs"], size=n)


def create_distance_m(deck: np.ndarray) -> np.ndarray:
    dist_cfg = cfg["distributions"]["distance_m"]
    distance = np.empty(n, dtype=float)

    for level in cfg["distributions"]["deck"]["values"]:
        mask = deck == level
        level_cfg = dist_cfg[level]
        distance[mask] = rng.normal(
            loc=level_cfg["mean"],
            scale=level_cfg["std"],
            size=mask.sum(),
        )
        distance[mask] = np.clip(
            distance[mask],
            level_cfg["low"],
            level_cfg["high"],
        )

    return np.round(distance, dist_cfg["decimals"])


def create_luggage_count() -> np.ndarray:
    luggage_cfg = cfg["distributions"]["luggage_count"]
    return rng.choice(luggage_cfg["values"], p=luggage_cfg["probs"], size=n)


def create_briefing() -> np.ndarray:
    briefing_cfg = cfg["distributions"]["briefing"]
    return rng.choice(briefing_cfg["values"], p=briefing_cfg["probs"], size=n)


def create_speed(
    age: np.ndarray,
    deck: np.ndarray,
    luggage_count: np.ndarray,
    briefing: np.ndarray,
) -> np.ndarray:
    speed_cfg = cfg["hidden_model"]["speed"]

    speed = np.full(n, speed_cfg["base_mps"], dtype=float)
    speed += speed_cfg["age_slope_per_year"] * (age - cfg["distributions"]["age"]["low"])

    for level in DECK_VALUES:
        speed[deck == level] += speed_cfg["deck_bonus_mps"][level]

    speed += luggage_count * speed_cfg["luggage_penalty_mps"]
    speed += briefing * speed_cfg["briefing_bonus_mps"]
    speed += rng.normal(0, speed_cfg["noise_std_mps"], size=n)

    return np.clip(speed, 0.1, None)


def create_start_delay_sec(
    age: np.ndarray,
    luggage_count: np.ndarray,
    briefing: np.ndarray,
) -> np.ndarray:
    delay_cfg = cfg["hidden_model"]["start_delay_sec"]

    start_delay = np.full(n, delay_cfg["base"], dtype=float)
    start_delay += delay_cfg["age_slope_per_year"] * (age - cfg["distributions"]["age"]["low"])
    start_delay += luggage_count * delay_cfg["luggage_penalty_per_bag"]
    start_delay += briefing * delay_cfg["briefing_bonus"]
    start_delay += rng.normal(0, delay_cfg["noise_std"], size=n)

    return np.clip(start_delay, 0.0, None)


def create_random_delay_sec() -> np.ndarray:
    delay_cfg = cfg["hidden_model"]["random_delay_sec"]
    rnd_delay = rng.normal(delay_cfg["mean"], delay_cfg["std"], size=n)
    return np.clip(rnd_delay, 0.0, None)


def create_target(
    limit: float,
    start_delay_sec: np.ndarray,
    travel_time_sec: np.ndarray,
    rnd_delay_sec: np.ndarray,
) -> np.ndarray:
    total_time_sec = start_delay_sec + travel_time_sec + rnd_delay_sec
    return (total_time_sec <= limit).astype(int)


def generate_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    passenger_id = np.arange(1, n + 1)

    age = create_age()
    deck = create_deck()
    distance_m = create_distance_m(deck)
    luggage_count = create_luggage_count()
    briefing = create_briefing()

    actual_speed = create_speed(age, deck, luggage_count, briefing)
    start_delay_sec = create_start_delay_sec(age, luggage_count, briefing)
    random_delay_sec = create_random_delay_sec()
    travel_time_sec = distance_m / actual_speed

    time_limit = cfg["hidden_model"]["time_limit_sec"]
    arrived_in_time = create_target(time_limit, start_delay_sec, travel_time_sec, random_delay_sec)

    hidden_df = pd.DataFrame(
        {
            ID_COL: passenger_id,
            ID_ACTUAL_SPEED: actual_speed,
            ID_START_DELAY: start_delay_sec,
            ID_TRAVEL_TIME: travel_time_sec,
            ID_RANDOM_DELAY: random_delay_sec,
            ID_TOTAL_TIME: start_delay_sec + travel_time_sec + random_delay_sec,
        }
    )

    create_missing_for_age(age)

    raw_df = pd.DataFrame(
        {
            ID_COL: passenger_id,
            ID_AGE: age,
            ID_DECK: deck,
            ID_DIST: distance_m,
            ID_LUGGAGE: luggage_count,
            ID_BRIEFING: briefing,
            TARGET_COL: arrived_in_time,
        }
    )

    return raw_df, hidden_df


def save_dataset() -> None:
    raw_df, hidden_df = generate_dataset()

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    HIDDEN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_df.to_csv(RAW_DATA_PATH, index=False)
    hidden_df.to_csv(HIDDEN_DATA_PATH, index=False)


if __name__ == "__main__":
    save_dataset()

