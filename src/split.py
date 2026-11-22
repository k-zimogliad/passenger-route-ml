import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from src.constants import RAW_DATA_PATH, RANDOM_STATE, TRAIN_PATH, TEST_PATH, VAL_PATH, GENERATOR_CONFIG_PATH, TARGET_COL

with open(GENERATOR_CONFIG_PATH, mode='r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

df = pd.read_csv(RAW_DATA_PATH)

def create_split():
    train_size = cfg['dataset']['train_size']
    val_size = cfg['dataset']['val_size']
    test_size = cfg['dataset']['test_size']
    train_df, temp_df = train_test_split(df, test_size=(1 - train_size), random_state=RANDOM_STATE, stratify=df[TARGET_COL])
    val_df, test_df = train_test_split(temp_df, test_size=(test_size / (test_size + val_size)), random_state=RANDOM_STATE, stratify=temp_df[TARGET_COL])
    return train_df, val_df, test_df

def save_dataset():
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    VAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tr_df, vl_df, te_df = create_split()
    tr_df.to_csv(TRAIN_PATH, index=False)
    vl_df.to_csv(VAL_PATH, index=False)
    te_df.to_csv(TEST_PATH, index=False)

if __name__ == "__main__":
    save_dataset()