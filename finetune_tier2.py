"""Tier-2 FT ablation (usage: python src/finetune_tier2.py 0). South train gradients, South val_es early stop. North never loaded. Originals unchanged."""
import json, sys, os, time, csv, glob
import numpy as np, torch, torch.nn as nn
sys.path.insert(0,"src"); from models import make_mlp, count_params; from data_pipeline import load
S=int(sys.argv[1]); torch.set_num_threads(1); LR=1e-4; MAXEP=50; PAT=10; BS=256
def get(name,ck):
    X,y,_=load(); sp=json.load(open("data/splits/tier2_split.json"))["indices"]; idx=sp[name]
    return torch.tensor((X[idx]-np.array(ck["scaler_mean"]))/np.array(ck["scaler_scale"]),dtype=torch.float32), torch.tensor(y[idx],dtype=torch.float32).view(-1,1)
def build(ck):
    m=make_mlp(ck["arch"])
    for pos,name in [(0,"L1"),(2,"L2")]:
        lin=m[pos]; mm,nn_=lin.weight.shape; r=ck["compressed_ranks"][name]; m[pos]=nn.Sequential(nn.Linear(nn_,r,bias=False),nn.Linear(r,mm,bias=True))
    m.load_state_dict(ck["state_dict"]); return m
def rmse(m,X,y):
    m.eval()
    with torch.no_grad(): return float(torch.sqrt(nn.functional.mse_loss(m(X),y)))
if __name__=="__main__":
    os.makedirs(f"models/tier2_seed{S}_finetuned",exist_ok=True); rows=[]
    for f in sorted(glob.glob(f"models/tier2_seed{S}_compressed/*.pt")):
        ck=torch.load(f,weights_only=False); model=build(ck); Xtr,ytr=get("train",ck); Xv,yv=get("val_es",ck)
        frozen=rmse(model,Xv,yv); params=count_params(model); torch.manual_seed(1000+S)
        opt=torch.optim.Adam(model.parameters(),lr=LR); best=frozen; best_state={k:v.clone() for k,v in model.state_dict().items()}; wait=0; hist=[]; t0=time.time()
        for ep in range(MAXEP):
            model.train(); perm=torch.randperm(len(Xtr)); tot,nb=0.0,0
            for i in range(0,len(Xtr),BS):
                idx=perm[i:i+BS]; loss=nn.functional.mse_loss(model(Xtr[idx]),ytr[idx]); opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.detach().item(); nb+=1
            vr=rmse(model,Xv,yv); hist.append({"epoch":ep+1,"train_loss":round(tot/nb,6),"val_rmse":round(vr,6)})
            if vr<best-1e-6: best,wait=vr,0; best_state={k:v.clone() for k,v in model.state_dict().items()}
            else:
                wait+=1
                if wait>=PAT: break
        wall=time.time()-t0; model.load_state_dict(best_state); tag=os.path.basename(f).replace(".pt","")
        with open(f"results/logs/tier2_seed{S}_ft_{tag}_curve.csv","w",newline="") as fh:
            w=csv.DictWriter(fh,fieldnames=["epoch","train_loss","val_rmse"]); w.writeheader(); w.writerows(hist)
        torch.save({"state_dict":model.state_dict(),"arch":ck["arch"],"compressed_ranks":ck["compressed_ranks"],"method":ck["method"],"budget_target":ck["budget_target"],"seed":S,"tier":2,"finetuned":True,"ft_settings":{"lr":LR,"max_epochs":MAXEP,"patience":PAT,"ft_seed":1000+S},"scaler_mean":ck["scaler_mean"],"scaler_scale":ck["scaler_scale"],"south_val_rmse":round(best,4)},f"models/tier2_seed{S}_finetuned/{tag}_ft.pt")
        rows.append({"seed":S,"tier":2,"method":ck["method"],"budget_target":f"{ck['budget_target']}x","rank_L1":ck["compressed_ranks"]["L1"],"rank_L2":ck["compressed_ranks"]["L2"],"total_params":params,"frozen_south_val_RMSE":round(frozen,4),"finetuned_south_val_RMSE":round(best,4),"ft_epochs":len(hist),"ft_wall_seconds":round(wall,1)})
        print(f"T2 s{S} {tag}: frozen={frozen:.4f} -> ft={best:.4f}",flush=True)
    with open(f"results/tables/tier2_seed{S}_finetuning_ablation.csv","w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
