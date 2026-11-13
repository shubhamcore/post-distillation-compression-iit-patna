"""Day 4 Phase 1: seed-0 compression of repro v2 KD student. Test set NOT loaded.
Allocators at complete-model budgets {2x:4736, 4x:2368, 8x:1184} (student total 9473):
- uniform: largest equal rank-ratio rho=r/min(m,n) (same rho both layers) feasible under B.
- energy: smallest r per layer with energy>=0.95; if over B, decrement largest-rank layer (tie: L2 first) until feasible.
- sensitivity: S_l(r)=MSE_sens(only layer l @r)-MSE_sens(orig); DP exact min sum max(0,S) s.t. total<=B.
Compressed layer: Linear(n->r,bias=False) then Linear(r->m,bias=True). Factors from svd_compression.factorize.
"""
import json, sys, os, csv, copy, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0,"src")
from models import make_mlp, count_params
from svd_compression import factorize, factorized_params
from data_pipeline import load
torch.set_num_threads(1)
STUDENT="models/repro_seed0_v2/student_kd_seed0_v2.pt"
BUDGETS={2:4736,4:2368,8:1184}; LAYERS=[("L1",0,128,8,range(1,8)),("L2",2,64,128,range(1,43))]

def get_split(name):
    X,y,_=load(); split=json.load(open("data/splits/tier1_split.json"))["indices"]
    ck=torch.load(STUDENT,weights_only=False); mean,scale=np.array(ck["scaler_mean"]),np.array(ck["scaler_scale"])
    idx=split[name]; return torch.tensor((X[idx]-mean)/scale,dtype=torch.float32), torch.tensor(y[idx],dtype=torch.float32).view(-1,1), ck

def load_student():
    ck=torch.load(STUDENT,weights_only=False); m=make_mlp(ck["arch"]); m.load_state_dict(ck["state_dict"]); m.eval(); return m,ck

def mse(model,X,y):
    with torch.no_grad(): return float(nn.functional.mse_loss(model(X),y))

def compress(model, ranks):
    """ranks: dict layer_name->r. Returns new model."""
    m=copy.deepcopy(model)
    for name,pos,mm,nn_,_ in LAYERS:
        r=ranks[name]; lin=m[pos]; A,B=factorize(lin.weight.data, r)  # A(m,r), B(r,n); y= A(Bx)+b
        seq=nn.Sequential(nn.Linear(nn_,r,bias=False), nn.Linear(r,mm,bias=True))
        with torch.no_grad():
            seq[0].weight.copy_(B); seq[1].weight.copy_(A); seq[1].bias.copy_(lin.bias.data)
        m[pos]=seq
    return m

def total_params(ranks): return factorized_params(128,8,ranks["L1"])+factorized_params(64,128,ranks["L2"])+65

def dp_allocate(Sclip, B):
    # exact DP over layers: state = params used -> (cost, ranks)
    states={65:(0.0,{})}
    for name,pos,m,n,rng in LAYERS:
        new={}
        for used,(cost,ranks) in states.items():
            for r in rng:
                p=used - (65 if False else 0)  # placeholder no-op
                nu = used + (factorized_params(m,n,r) if used!=65 or True else 0)
                # used starts at 65 only for first layer; handle: first iteration used==65 base
                nu = (used + factorized_params(m,n,r)) if not ranks else used + factorized_params(m,n,r)
                if nu<=B:
                    nc=cost+Sclip[name][r]; nr=dict(ranks); nr[name]=r
                    if nu not in new or nc<new[nu][0]: new[nu]=(nc,nr)
        states=new
    best=min(states.items(), key=lambda kv:(kv[1][0], kv[0]))
    return best[1][1], best[1][0], best[0]

if __name__=="__main__":
    os.makedirs("models/day4_seed0_compressed",exist_ok=True)
    base,ck=load_student(); print("student params",count_params(base))
    Xs,ys,_=get_split("sensitivity"); Xv,yv,_=get_split("val_es")
    base_sens=mse(base,Xs,ys); base_val=mse(base,Xv,yv)
    print(f"base sensitivity MSE={base_sens:.6f} val_es MSE={base_val:.6f} val_RMSE={np.sqrt(base_val):.4f}")
    # 1. Sensitivity table
    S_raw={}; S_clip={}
    rows=[]
    for name,pos,m,n,rng in LAYERS:
        S_raw[name]={}; S_clip[name]={}
        W=base[pos].weight.data; sv=torch.linalg.svdvals(W)
        for r in rng:
            cm=compress(base,{name:r} if name=="L1" else {name:r}) if False else None
            # compress ONLY this layer: build ranks dict with other layer full-size equivalent -> use direct single-layer replace
            tmp=copy.deepcopy(base); A,B=factorize(W,r)
            seq=nn.Sequential(nn.Linear(n,r,bias=False), nn.Linear(r,m,bias=True))
            with torch.no_grad(): seq[0].weight.copy_(B); seq[1].weight.copy_(A); seq[1].bias.copy_(base[pos].bias.data)
            tmp[pos]=seq; s=mse(tmp,Xs,ys)-base_sens
            S_raw[name][r]=s; S_clip[name][r]=max(0.0,s)
            rows.append({"layer":name,"rank":r,"params_factorized":factorized_params(m,n,r),"S_raw":round(s,6),"S_clipped":round(max(0.0,s),6),"energy":round(float(torch.sum(sv[:r]**2)/torch.sum(sv**2)),4)})
    with open("results/tables/day4_sensitivity_scores_seed0.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    neg=sum(1 for r in rows if r["S_raw"]<0); print(f"sensitivity evals={len(rows)} negative={neg} ({neg/len(rows):.1%})")
    json.dump({"base_sens_mse":base_sens,"base_val_mse":base_val,"S_raw":{k:{str(r):v for r,v in d.items()} for k,d in S_raw.items()}},open("results/logs/day4_sensitivity_raw_seed0.json","w"),indent=1)
    # 2. Allocators
    results=[]
    def uniform(B):
        best=None
        for rho_i in range(1,1001):
            rho=rho_i/1000; r1=max(1,int(round(rho*8))); r2=max(1,int(round(rho*64)))
            r1=min(r1,7); r2=min(r2,42)
            if total_params({"L1":r1,"L2":r2})<=B: best={"L1":r1,"L2":r2}
            else: break
        return best
    def energy_alloc(B):
        ranks={}
        for name,pos,m,n,rng in LAYERS:
            sv=torch.linalg.svdvals(base[pos].weight.data); cum=torch.cumsum(sv**2,0)/torch.sum(sv**2)
            ranks[name]=int(torch.searchsorted(cum,0.95).item())+1
        ranks["L1"]=min(ranks["L1"],7); ranks["L2"]=min(ranks["L2"],42)
        while total_params(ranks)>B:
            # decrement largest rank (tie -> L2)
            if ranks["L2"]>=ranks["L1"] and ranks["L2"]>1: ranks["L2"]-=1
            elif ranks["L1"]>1: ranks["L1"]-=1
            else: break
        return ranks
    alloc_log={}
    for target,B in BUDGETS.items():
        allocs={"uniform":uniform(B),"energy":energy_alloc(B)}
        dp_ranks,dp_cost,dp_total=dp_allocate(S_clip,B); allocs["sensitivity"]=dp_ranks
        alloc_log[str(target)]={"budget":B,"allocations":{k:v for k,v in allocs.items()},"dp_surrogate_cost":dp_cost,"dp_total_params":dp_total}
        print(f"Budget {target}x B={B}: {allocs} DPcost={dp_cost:.6f}")
        for method,ranks in allocs.items():
            cm=compress(base,ranks); tp=count_params(cm)
            assert tp==total_params(ranks), (tp,ranks)
            # dims check: forward shape
            with torch.no_grad(): out=cm(Xs[:5]); assert out.shape==(5,1)
            sens_m=mse(cm,Xs,ys); val_m=mse(cm,Xv,yv)
            joint_damage=sens_m-base_sens; pred_damage=sum(S_raw[k][ranks[k]] for k in ranks)
            add_disc=joint_damage-pred_damage
            tag=f"{method}_{target}x_r{ranks['L1']}-{ranks['L2']}"
            torch.save({"state_dict":cm.state_dict(),"arch":ck["arch"],"compressed_ranks":ranks,"method":method,"budget_target":target,"seed":0,
                        "scaler_mean":ck["scaler_mean"],"scaler_scale":ck["scaler_scale"],
                        "structure":"L1,L2 replaced by Linear(n->r,bias=False)+Linear(r->m,bias=True); output untouched",
                        "val_rmse":round(float(np.sqrt(val_m)),4)},"models/day4_seed0_compressed/"+tag+".pt")
            results.append({"method":method,"budget_target":f"{target}x","budget_B":B,"rank_L1":ranks["L1"],"rank_L2":ranks["L2"],
                "total_params":tp,"complete_ratio":round(9473/tp,3),"eligible_params":tp-65,"eligible_ratio":round(9408/(tp-65),3),
                "sens_MSE":round(sens_m,6),"val_RMSE":round(float(np.sqrt(val_m)),4),"val_MAE":round(float(torch.mean(torch.abs(cm(Xv)-yv))),4) if False else "",
                "joint_damage_sens":round(joint_damage,6),"predicted_damage_sum_rawS":round(pred_damage,6),"additivity_discrepancy":round(add_disc,6),"checkpoint":f"models/day4_seed0_compressed/{tag}.pt"})
            print(f"  {tag}: params={tp} val_RMSE={np.sqrt(val_m):.4f} joint_dmg={joint_damage:.5f} pred={pred_damage:.5f} add_disc={add_disc:.5f}")
    # fix MAE properly in second pass from checkpoints (kept simple: recompute)
    for r in results:
        ck2=torch.load(r["checkpoint"],weights_only=False)
        # rebuild via compress again deterministically from ranks
        cm=compress(base,{"L1":r["rank_L1"],"L2":r["rank_L2"]})
        with torch.no_grad(): r["val_MAE"]=round(float(torch.mean(torch.abs(cm(Xv)-yv))),4)
    with open("results/tables/day4_compression_results_seed0.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
    json.dump(alloc_log,open("results/logs/day4_dp_allocation_seed0.json","w"),indent=1)
    json.dump({"base_val_RMSE":round(float(np.sqrt(base_val)),4),"base_sens_MSE":round(base_sens,6),"negative_sensitivity_count":neg,"test_set_touched":False},open("results/logs/day4_base_seed0.json","w"),indent=1)
    print("DONE test untouched")
