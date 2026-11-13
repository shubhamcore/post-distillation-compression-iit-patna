"""Shared frozen-split loading and metric utilities for traditional ML baselines.
Scalers are fitted on the training split only. Hyperparameters are selected on
the validation split only. Test splits are evaluated once, after selection.
"""
import json, numpy as np
from pathlib import Path
from sklearn.datasets import fetch_california_housing
ROOT = Path(__file__).resolve().parent
FEATURES = ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']

def load_tier(tier):
    data = fetch_california_housing()
    X, y = np.asarray(data.data, dtype=float), np.asarray(data.target, dtype=float)
    split = json.loads((ROOT / f"data/splits/{tier}_split.json").read_text())["indices"]
    test_key = "test" if tier == "tier1" else "test_north"
    return {"X": X, "y": y, "train": split["train"], "val": split["val_es"], "test": split[test_key],
            "test_key": test_key}

def metrics(y_true, y_pred, p):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    resid = y_true - y_pred
    mae = float(np.mean(np.abs(resid))); mse = float(np.mean(resid**2)); rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(resid**2)); ss_tot = float(np.sum((y_true - y_true.mean())**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot else float("nan")
    n = len(y_true)
    # p = number of predictors (features) for every baseline here (8); see workbook notes.
    adj = 1.0 - (1.0-r2)*(n-1)/(n-p-1) if n-p-1 > 0 else float("nan")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "Adjusted_R2": adj, "n": int(n), "p_predictors": int(p)}

def save_predictions(path, idx, y_true, y_pred):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["row_index","y_true","y_pred"])
        for i, a, b in zip(idx, y_true, y_pred): w.writerow([int(i), float(a), float(b)])
