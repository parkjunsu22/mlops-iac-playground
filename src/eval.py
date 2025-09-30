import pandas as pd, numpy as np, onnxruntime as ort
from sklearn.metrics import accuracy_score

def load_csv(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["label"]).values.astype("float32")
    y = df["label"].values
    return X, y

if __name__ == "__main__":
    X, y = load_csv("data/processed/test.csv")
    sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
    logits = sess.run(["logits"], {"input": X})[0]
    pred = logits.argmax(axis=1)
    print(f"accuracy={accuracy_score(y, pred):.4f}")
