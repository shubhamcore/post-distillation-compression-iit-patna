"""Gaussian Process baseline - APPROXIMATE by disclosed subsampling.
Exact GP on 12k+ training rows is O(n^3) and infeasible here, so a fixed
seed-42 subsample of 2,000 training rows is used. Results are labelled
'GPR (subsampled, approximate)' everywhere and must not be read as full-data GPR.
Kernel hyperparameters are optimised on that subsample; no validation/test tuning."""
import json, sys, joblib, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from baseline_common import load_tier, metrics, save_predictions, ROOT
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
SEED=42; N_SUB=2000
OUT=ROOT/"results/baselines"; OUT.mkdir(parents=True, exist_ok=True)
rows=[]
for tier in ("tier1","tier2"):
    d=load_tier(tier); X,y=d["X"],d["y"]
    scaler=StandardScaler().fit(X[d["train"]])
    rng=np.random.default_rng(SEED); sub=rng.choice(np.array(d["train"]), size=min(N_SUB,len(d["train"])), replace=False)
    kernel=ConstantKernel(1.0,(1e-2,1e3))*RBF(length_scale=np.ones(X.shape[1]),length_scale_bounds=(1e-1,1e2))+WhiteKernel(0.05,(1e-4,1.0))
    model=GaussianProcessRegressor(kernel=kernel,normalize_y=True,random_state=SEED,n_restarts_optimizer=0)
    model.fit(scaler.transform(X[sub]), y[sub])
    hp={"kernel":str(model.kernel_),"subsample_n":int(len(sub)),"subsample_seed":SEED,"note":"approximate subsampled GPR, not full-data"}
    for split_name, idx in (("validation",d["val"]),("test",d["test"])):
        pred=model.predict(scaler.transform(X[idx])); m=metrics(y[idx],pred,p=X.shape[1])
        rows.append({"Model":"Gaussian Process Regression (subsampled, approximate)","Tier":tier,"Split":split_name,"Python_File":"gaussian_process_baseline.py","Hyperparameters":json.dumps(hp),"Seed":SEED,**m})
        if split_name=="test": save_predictions(OUT/f"gaussian_process_{tier}_test_predictions.csv", idx, y[idx], pred)
    joblib.dump({"scaler":scaler,"model":model,"subsample_indices":sub}, OUT/f"gaussian_process_{tier}.joblib")
(OUT/"gaussian_process_metrics.json").write_text(json.dumps(rows,indent=2)); print(json.dumps(rows,indent=2))
