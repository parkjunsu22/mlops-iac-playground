import os
import pandas as pd
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

def to_abs(path: str) -> str:
    """Make path absolute, anchored at repo root (original cwd) if relative."""
    if os.path.isabs(path):
        return path
    return os.path.join(get_original_cwd(), path)

@hydra.main(version_base=None, config_path="../conf", config_name="dataset")
def main(cfg: DictConfig):
    raw_path = to_abs(cfg.raw_path)
    train_path = to_abs(cfg.processed_path)

    # ensure output directory exists (Hydra cwd may be different)
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    df = pd.read_csv(raw_path)

    # label column handling
    label_col = cfg.get("label_col", None)
    if label_col is None:
        if "label" in df.columns:
            label_col = "label"
        elif "y" in df.columns:
            label_col = "y"
        else:
            raise ValueError(f"No label column found. columns={list(df.columns)}")

    if label_col != "label":
        df = df.rename(columns={label_col: "label"})

    train, test = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=df["label"],
    )

    test_path = os.path.join(os.path.dirname(train_path), "test.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")

if __name__ == "__main__":
    main()

