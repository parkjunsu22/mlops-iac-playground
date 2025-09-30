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
    def forward(self, x): return self.net(x)
    def training_step(self, batch, _):
        x,y = batch
        logits = self(x); loss = self.loss(logits, y)
        self.log("train_loss", loss, prog_bar=True); return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def make_loader(csv, batch):
    df = pd.read_csv(csv)
    X = torch.tensor(df.drop(columns=["label"]).values, dtype=torch.float32)
    y = torch.tensor(df["label"].values, dtype=torch.long)
    return DataLoader(TensorDataset(X,y), batch_size=batch, shuffle=True), X.shape[1]

@hydra.main(version_base=None, config_path="../conf", config_name="model")
def main(cfg: DictConfig):
    train_loader, in_dim = make_loader("data/processed/train.csv", cfg.batch_size)
    model = MLP(in_dim=in_dim, hidden_dim=cfg.hidden_dim, lr=cfg.lr)

    os.makedirs("mlruns", exist_ok=True)
    mlf = MLFlowLogger(experiment_name="baseline", tracking_uri="file:mlruns")
    trainer = pl.Trainer(max_epochs=cfg.max_epochs, logger=mlf, enable_checkpointing=False, log_every_n_steps=10)
    trainer.fit(model, train_loader)

    dummy = torch.randn(1, in_dim)
    torch.onnx.export(model, dummy, "model.onnx", input_names=["input"], output_names=["logits"], opset_version=17)
    print("Exported model.onnx")

if __name__ == "__main__":
    main()
