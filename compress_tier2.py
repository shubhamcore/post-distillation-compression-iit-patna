"""Tier-2 compression (usage: python src/compress_tier2.py 0). Sensitivity on Tier-2 SOUTH sensitivity split only. North never loaded.
Same 3 allocators + budgets as Tier-1. Energy = GREEDY baseline (documented in doc 05)."""
import json, sys, os, csv, copy
import numpy as np, torch, torch.nn as nn
sys.path.insert(0,"src"); from models import make_mlp, count_params; from svd_compression import factorize, factorized_params; from data_pipeline import load
S=int(sys.argv[1]); torch.set_num_threads(1); STUDENT=f"models/tier2_seed{S}/student_kd_t2_seed{S}.pt"
BUDGETS={2:4736,4:2368,8:1184}; LAYERS=[("L1",0,128,8,list(range(1,8))),("L2",2,64,128,list(range(1,43)))]
def get(name,ck):
    X,y,_=load(); sp=json.load(open("data/splits/tier2_split.json"))["indices"]; idx=sp[name]
    return torch.tensor((X[idx]-np.array(ck["scaler_mean"]))/np.array(ck["scaler_scale"]),dtype=torch.float32), torch.tensor(y[idx],dtype=torch.float32).view(-1,1)
def mse(m,X,y):
    with torch.no_grad(): return float(nn.functional.mse_loss(m(X),y))
def replace(base,name,r):
    tmp=copy.deepcopy(base)
    for nm,pos,mm,nn_,_ in LAYERS:
        if nm==name:
            A,B=factorize(tmp[pos].weight.data,r); seq=nn.Sequential(nn.Linear(nn_,r,bias=False),nn.Linear(r,mm,bias=True))
            with torch.no_grad(): seq[0].weight.copy_(B); seq[1].weight.copy_(A); seq[1].bias.copy_(tmp[pos].bias.data)
            tmp[pos]=seq
    return tmp
def compress(base,ranks):
    m=copy.deepcopy(base)
    for name,pos,mm,nn_,_ in LAYERS:
        r=ranks[name]; A,B=factorize(m[pos].weight.data,r); seq=nn.Sequential(nn.Linear(nn_,r,bias=False),nn.Linear(r,mm,bias=True))
        with torch.no_grad(): seq[0].weight.copy_(B); seq[1].weight.copy_(A); seq[1].bias.copy_(m[pos].bias.data)
        m[pos]=seq
    return m
def total(ranks): return factorized_params(128,8,ranks["L1"])+factorized_params(64,128,ranks["L2"])+65
def dp(Sclip,B):
    states={65:(0.0,{})}
    for name,pos,m,n,rng in LAYERS:
        new={}
        for used,(cost,ranks) in states.items():
            for r in rng:
                nu=used+factorized_params(m,n,r)
                if nu<=B:
                    nc=cost+Sclip[name][r]; nr=dict(ranks); nr[name]=r
                    if nu not in new or nc<new[nu][0]: new[nu]=(nc,nr)
        states=new
    b=min(states.items(),key=lambda kv:(kv[1][0],kv[0])); return b[1][1]
if __name__=="__main__":
    os.makedirs(f"models/tier2_seed{S}_compressed",exist_ok=True)
    ck=torch.load(STUDENT,weights_only=False); base=make_mlp(ck["arch"]); base.load_state_dict(ck["state_dict"]); base.eval()
    Xs,ys=get("sensitivity",ck); Xv,yv=get("val_es",ck); bs,bv=mse(base,Xs,ys),mse(base,Xv,yv)
    print(f"T2 seed{S} base south_sensMSE={bs:.6f} south_val_RMSE={np.sqrt(bv):.4f}")
    S_raw={}; S_clip={}; rows=[]
    for name,pos,m,n,rng in LAYERS:
        S_raw[name]={}; S_clip[name]={}
        for r in rng:
            s=mse(replace(base,name,r),Xs,ys)-bs; S_raw[name][r]=s; S_clip[name][r]=max(0.0,s)
            rows.append({"seed":S,"tier":2,"layer":name,"rank":r,"S_raw":round(s,6),"S_clipped":round(max(0.0,s),6)})
    with open(f"results/tables/tier2_seed{S}_sensitivity_scores.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    def uniform(B):
        best=None
        for i in range(1,1001):
            ranks={"L1":min(7,max(1,int(round(i/1000*8)))),"L2":min(42,max(1,int(round(i/1000*64))))}
            if total(ranks)<=B: best=ranks
            else: break
        return best
    def energy_greedy(B):
        ranks={}
        for name,pos,m,n,rng in LAYERS:
            sv=torch.linalg.svdvals(base[pos].weight.data); cum=torch.cumsum(sv**2,0)/torch.sum(sv**2)
            ranks[name]=min(int(torch.searchsorted(cum,0.95).item())+1,max(rng))
        while total(ranks)>B:
            if ranks["L2"]>=ranks["L1"] and ranks["L2"]>1: ranks["L2"]-=1
            elif ranks["L1"]>1: ranks["L1"]-=1
            else: break
        return ranks
    results=[]
    for target,B in BUDGETS.items():
        for method,ranks in {"uniform":uniform(B),"energy_greedy":energy_greedy(B),"sensitivity":dp(S_clip,B)}.items():
            cm=compress(base,ranks); tp=count_params(cm); assert tp==total(ranks)
            sm,vm=mse(cm,Xs,ys),mse(cm,Xv,yv); joint=sm-bs; pred=sum(S_raw[k][ranks[k]] for k in ranks)
            tag=f"{method}_{target}x_r{ranks['L1']}-{ranks['L2']}"
            torch.save({"state_dict":cm.state_dict(),"arch":ck["arch"],"compressed_ranks":ranks,"method":method,"budget_target":target,"seed":S,"tier":2,"scaler_mean":ck["scaler_mean"],"scaler_scale":ck["scaler_scale"],"south_val_rmse":round(float(np.sqrt(vm)),4)},f"models/tier2_seed{S}_compressed/{tag}.pt")
            with torch.no_grad(): mae=float(torch.mean(torch.abs(cm(Xv)-yv)))
            results.append({"seed":S,"tier":2,"method":method,"budget_target":f"{target}x","budget_B":B,"rank_L1":ranks["L1"],"rank_L2":ranks["L2"],"total_params":tp,"compression_ratio_vs_student":round(9473/tp,3),"south_val_RMSE":round(float(np.sqrt(vm)),4),"south_val_MAE":round(mae,4),"joint_damage_south_sens":round(joint,6),"predicted_damage":round(pred,6),"additivity_discrepancy":round(joint-pred,6)})
            print(f" T2 s{S} {tag} params={tp} south_val_RMSE={np.sqrt(vm):.4f}")
    with open(f"results/tables/tier2_seed{S}_compression_results.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
    print("north untouched")
