from typing import List
import os, numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "model.onnx")

app = FastAPI(title="mlops-svc", version="0.1")

# 앱 기동 시 모델 로드 (프로세스 시작시 1회)
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

class Item(BaseModel):
    features: List[float]  # Python 3.8 호환 표기

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(item: Item):
    X = np.array(item.features, dtype=np.float32).reshape(1, -1)
    logits = sess.run(["logits"], {"input": X})[0]
    pred = int(np.argmax(logits, axis=1)[0])
    return {"pred": pred}
