import pandas as pd
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../conf", config_name="dataset")
def main(cfg: DictConfig):
    df = pd.read_csv(cfg.raw_path)
    train, test = train_test_split(df, test_size=cfg.test_size, random_state=cfg.random_state, stratify=df["label"])
    train.to_csv(cfg.processed_path, index=False)
    test.to_csv(cfg.processed_path.replace("train","test"), index=False)
    print("Saved:", cfg.processed_path)

if __name__ == "__main__":
    main()
