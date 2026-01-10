import os, json
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import accuracy_score

def load_csv(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=["label"]).values.astype("float32")
    y = df["label"].values
    return X, y

if __name__ == "__main__":
    X, y = load_csv("data/processed/test.csv")

    sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
    logits = sess.run(["logits"], {"input": X})[0]
    pred = logits.argmax(axis=1)

    acc = accuracy_score(y, pred)
    print(f"accuracy={acc:.4f}")

    # (대안 B) DVC가 추적할 "작은 metrics 파일"
    os.makedirs("outputs", exist_ok=True)
    metrics = {"accuracy": float(acc), "n_samples": int(len(y))}
    with open("outputs/eval.json", "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

