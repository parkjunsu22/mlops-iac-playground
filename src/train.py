import os, pandas as pd, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import mlflow, hydra
from omegaconf import DictConfig

class MLP(pl.LightningModule):
    def __init__(self, in_dim, hidden_dim, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def make_loader(csv, batch, num_workers=0):
    df = pd.read_csv(csv)
    X = torch.tensor(df.drop(columns=["label"]).values, dtype=torch.float32)
    y = torch.tensor(df["label"].values, dtype=torch.long)
    ds = TensorDataset(X, y)
    pw = True if num_workers and num_workers > 0 else False
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=True,
        num_workers=num_workers,   # ← CPU 워커 수
        pin_memory=True,           # CPU→GPU 시 유리(지금은 CPU여도 보통 안전)
        persistent_workers=pw,     # 에폭 간 워커 재사용
        prefetch_factor=2          # 워커당 선로딩 배치 수
    ), X.shape[1]

@hydra.main(version_base=None, config_path="../conf", config_name="model")
def main(cfg: DictConfig):
    train_loader, in_dim = make_loader(
        "data/processed/train.csv",
        cfg.batch_size,
        num_workers=getattr(cfg, "num_workers", 0)  # ← Hydra에서 읽음
    )
    model = MLP(in_dim=in_dim, hidden_dim=cfg.hidden_dim, lr=cfg.lr)

    os.makedirs("mlruns", exist_ok=True)
    mlf = MLFlowLogger(experiment_name="baseline", tracking_uri="file:mlruns")
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        logger=mlf,
        enable_checkpointing=False,
        log_every_n_steps=10
    )
    trainer.fit(model, train_loader)

    # ---- ONNX export (동적 배치) ----
    model.eval()
    dummy = torch.randn(1, in_dim)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            "model.onnx",
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch"},
                "logits": {0: "batch"},
            },
            opset_version=17,
        )
    print("Exported model.onnx")

if __name__ == "__main__":
    main()
