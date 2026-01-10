import os
import pandas as pd
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path


@hydra.main(version_base=None, config_path="../conf", config_name="dataset")
def main(cfg: DictConfig):
    raw_path = to_absolute_path(cfg.raw_path)
    train_path = to_absolute_path(cfg.processed_path)
    test_path = to_absolute_path(cfg.processed_path.replace("train", "test"))

    df = pd.read_csv(raw_path)

    # label column handling:
    # 1) prefer cfg.label_col if present
    # 2) else fallback: if "label" not in columns and "y" exists -> rename y -> label
    label_col = cfg.get("label_col", "label")
    if label_col in df.columns and label_col != "label":
        df = df.rename(columns={label_col: "label"})
    elif "label" not in df.columns and "y" in df.columns:
        df = df.rename(columns={"y": "label"})

    # ensure output dir exists
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    train, test = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=df["label"],
    )

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print("Saved:", train_path)
    print("Saved:", test_path)


if __name__ == "__main__":
    main()

