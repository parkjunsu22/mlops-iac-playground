.PHONY: preprocess train eval pipeline mlflow
preprocess: ; python -m src.preprocess
train:      ; python -m src.train
eval:       ; python -m src.eval
pipeline:   ; dvc repro
mlflow:     ; mlflow ui --backend-store-uri file:mlruns --host 0.0.0.0 --port 5000
