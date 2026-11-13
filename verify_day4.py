"""Fresh-process verification of Day-4 compressed checkpoints: rebuild structure from metadata, load, check dims/params/predictions."""
import json, sys, glob, torch, torch.nn as nn, numpy as np
sys.path.insert(0,"src"); from models import make_mlp, count_params; from data_pipeline import load
X,y,_=load(); split=json.load(open("data/splits/tier1_split.json"))["indices"]
res={}
for f in sorted(glob.glob("models/day4_seed0_compressed/*.pt")):
    ck=torch.load(f,weights_only=False); ranks=ck["compressed_ranks"]
    m=make_mlp(ck["arch"]); m.eval()
    # replace layers to match checkpoint structure
    for pos,name in [(0,"L1"),(2,"L2")]:
        lin=m[pos]; mm,nn_=lin.weight.shape; r=ranks[name]
        seq=nn.Sequential(nn.Linear(nn_,r,bias=False), nn.Linear(r,mm,bias=True)); m[pos]=seq
    m.load_state_dict(ck["state_dict"]); m.eval()
    mean,scale=np.array(ck["scaler_mean"]),np.array(ck["scaler_scale"])
    Xv=torch.tensor((X[split["val_es"]]-mean)/scale,dtype=torch.float32)
    with torch.no_grad(): p=m(Xv).numpy().ravel()
    assert np.isfinite(p).all() and len(p)==len(split["val_es"])
    rmse=float(np.sqrt(np.mean((p-y[split["val_es"]])**2)))
    ok=abs(rmse-ck["val_rmse"])<1e-3
    print(f"{f.split('/')[-1]}: ranks={ranks} params={count_params(m)} dims_ok reloaded_RMSE={rmse:.4f} logged={ck['val_rmse']} match={ok}")
    assert ok
print("ALL COMPRESSED CHECKPOINTS VERIFIED (test set not used)")
