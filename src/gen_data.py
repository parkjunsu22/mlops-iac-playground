import os
import csv
import random

OUT = "data/raw/sample.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

random.seed(42)

rows = 1000
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["x1", "x2", "label"])   # ✅ y -> label
    for _ in range(rows):
        x1 = random.random()
        x2 = random.random()
        label = 1 if (x1 + 0.5*x2) > 0.75 else 0
        w.writerow([f"{x1:.6f}", f"{x2:.6f}", label])

print(f"[DONE] wrote {OUT} ({rows} rows)")

