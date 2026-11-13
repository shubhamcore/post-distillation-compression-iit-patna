"""Verify training + compressed checkpoints for a seed prefix (usage: python src/verify_seed.py 1)."""
import json, sys, glob, torch, torch.nn as nn, numpy as np
sys.path.insert(0,"src"); from models import make_mlp, count_params; from data_pipeline import load
S=sys.argv[1]; X,y,_=load(); split=json.load(open("data/splits/tier1_split.json"))["indices"]
summ=json.load(open(f"results/logs/seed{S}_training_summary.json")) if S!="0" else json.load(open("results/logs/repro_seed0_v2_summary.json"))
for name,f,arch,drop in [("teacher",f"models/seed{S}/teacher_seed{S}.pt",[8,256,256,128,1],0.1),("student_KD",f"models/seed{S}/student_kd_seed{S}.pt",[8,128,64,1],0.0),("student_noKD",f"models/seed{S}/student_nokd_seed{S}.pt",[8,128,64,1],0.0)]:
    if S=="0": continue
    ck=torch.load(f,weights_only=False); m=make_mlp(arch,dropout=drop); m.load_state_dict(ck["state_dict"]); m.eval()
    Xv=torch.tensor((X[split["val_es"]]-np.array(ck["scaler_mean"]))/np.array(ck["scaler_scale"]),dtype=torch.float32)
    with torch.no_grad(): p=m(Xv).numpy().ravel()
    rmse=float(np.sqrt(np.mean((p-y[split["val_es"]])**2))); print(f"seed{S} {name}: reloaded RMSE={rmse:.4f} logged={summ[name]['val_RMSE']} match={abs(rmse-summ[name]['val_RMSE'])<1e-3}")
for f in sorted(glob.glob(f"models/seed{S}_compressed/*.pt")):
    ck=torch.load(f,weights_only=False); ranks=ck["compressed_ranks"]; m=make_mlp(ck["arch"])
    for pos,name in [(0,"L1"),(2,"L2")]:
        lin=m[pos]; mm,nn_=lin.weight.shape; r=ranks[name]; m[pos]=nn.Sequential(nn.Linear(nn_,r,bias=False),nn.Linear(r,mm,bias=True))
    m.load_state_dict(ck["state_dict"]); m.eval()
    Xv=torch.tensor((X[split["val_es"]]-np.array(ck["scaler_mean"]))/np.array(ck["scaler_scale"]),dtype=torch.float32)
    with torch.no_grad(): p=m(Xv).numpy().ravel()
    rmse=float(np.sqrt(np.mean((p-y[split["val_es"]])**2))); print(f" seed{S} {f.split('/')[-1]}: params={count_params(m)} RMSE={rmse:.4f} logged={ck['val_rmse']} match={abs(rmse-ck['val_rmse'])<1e-3} finite={np.isfinite(p).all()}")
print(f"seed{S} VERIFICATION DONE (test not used)")
