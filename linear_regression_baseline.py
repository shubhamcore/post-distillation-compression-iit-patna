"""Linear Regression baseline on the frozen Tier-1/Tier-2 splits (validation has no tuning role)."""
import json, sys, joblib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from baseline_common import load_tier, metrics, save_predictions, ROOT
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
OUT = ROOT/"results/baselines"; OUT.mkdir(parents=True, exist_ok=True)
rows=[]
for tier in ("tier1","tier2"):
    d = load_tier(tier); X,y = d["X"], d["y"]
    scaler = StandardScaler().fit(X[d["train"]])
    model = LinearRegression().fit(scaler.transform(X[d["train"]]), y[d["train"]])
    for split_name, idx in (("validation", d["val"]), ("test", d["test"])):
        pred = model.predict(scaler.transform(X[idx])); m = metrics(y[idx], pred, p=X.shape[1])
        rows.append({"Model":"Linear Regression","Tier":tier,"Split":split_name,"Python_File":"linear_regression_baseline.py","Hyperparameters":"LinearRegression()","Seed":"n/a (deterministic)",**m})
        if split_name=="test": save_predictions(OUT/f"linear_regression_{tier}_test_predictions.csv", idx, y[idx], pred)
    joblib.dump({"scaler":scaler,"model":model}, OUT/f"linear_regression_{tier}.joblib")
(OUT/"linear_regression_metrics.json").write_text(json.dumps(rows,indent=2)); print(json.dumps(rows,indent=2))
