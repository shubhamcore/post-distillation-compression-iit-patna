"""SVR baseline (RBF). C/gamma selected on validation RMSE only; test evaluated once."""
import json, sys, joblib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from baseline_common import load_tier, metrics, save_predictions, ROOT
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
SEED="n/a (deterministic given data)"
CANDIDATES=[{"C":3.0,"gamma":"scale","epsilon":0.1},{"C":10.0,"gamma":"scale","epsilon":0.1},{"C":10.0,"gamma":0.1,"epsilon":0.05}]
OUT=ROOT/"results/baselines"; OUT.mkdir(parents=True, exist_ok=True)
rows=[]
for tier in ("tier1","tier2"):
    d=load_tier(tier); X,y=d["X"],d["y"]
    scaler=StandardScaler().fit(X[d["train"]])
    Xtr,Xval=scaler.transform(X[d["train"]]),scaler.transform(X[d["val"]])
    best=None
    for hp in CANDIDATES:
        m=SVR(**hp).fit(Xtr,y[d["train"]]); vm=metrics(y[d["val"]],m.predict(Xval),p=X.shape[1])
        rows.append({"Model":"Support Vector Regression","Tier":tier,"Split":"validation (selection)","Python_File":"support_vector_regression_baseline.py","Hyperparameters":json.dumps(hp),"Seed":SEED,**vm})
        if best is None or vm["RMSE"]<best[0]["RMSE"]: best=(vm,hp,m)
    vm,hp,model=best
    rows.append({"Model":"Support Vector Regression","Tier":tier,"Split":"validation (selected)","Python_File":"support_vector_regression_baseline.py","Hyperparameters":json.dumps(hp),"Seed":SEED,**vm})
    pred=model.predict(scaler.transform(X[d["test"]]))
    rows.append({"Model":"Support Vector Regression","Tier":tier,"Split":"test","Python_File":"support_vector_regression_baseline.py","Hyperparameters":json.dumps(hp),"Seed":SEED,**metrics(y[d["test"]],pred,p=X.shape[1])})
    save_predictions(OUT/f"support_vector_regression_{tier}_test_predictions.csv", d["test"], y[d["test"]], pred)
    joblib.dump({"scaler":scaler,"model":model}, OUT/f"support_vector_regression_{tier}.joblib")
(OUT/"support_vector_regression_metrics.json").write_text(json.dumps(rows,indent=2)); print(json.dumps(rows,indent=2))
