"""Reload checkpoints in a fresh process, check prediction consistency vs in-training metrics."""
import json, sys, torch, numpy as np
sys.path.insert(0,"src")
from models import make_mlp
from data_pipeline import load
X,y,_=load(); split=json.load(open("data/splits/tier1_split.json"))["indices"]
summ=json.load(open("results/logs/day3_seed0_summary.json"))
for name,f,arch,drop in [("teacher","models/day3_seed0/teacher_seed0.pt",[8,256,256,128,1],0.1),
                          ("student_KD","models/day3_seed0/student_kd_seed0.pt",[8,128,64,1],0.0),
                          ("student_noKD","models/day3_seed0/student_nokd_seed0.pt",[8,128,64,1],0.0)]:
    ck=torch.load(f,weights_only=False); m=make_mlp(arch,dropout=drop); m.load_state_dict(ck["state_dict"]); m.eval()
    mean=np.array(ck["scaler_mean"]); scale=np.array(ck["scaler_scale"])
    Xv=torch.tensor((X[split["val_es"]]-mean)/scale,dtype=torch.float32); yv=y[split["val_es"]]
    with torch.no_grad(): p=m(Xv).numpy().ravel()
    rmse=float(np.sqrt(np.mean((p-yv)**2)))
    print(f"{name}: reloaded {f} | recomputed val_RMSE={rmse:.4f} vs logged {summ[name]['val_RMSE']} | match={abs(rmse-summ[name]['val_RMSE'])<1e-3} | params={sum(q.numel() for q in m.parameters())} | seed={ck['seed']}")
print("CHECKPOINT RELOAD VERIFICATION DONE (test set not used)")
