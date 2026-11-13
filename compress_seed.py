"""Compress a seed's KD student (usage: python src/compress_seed.py 1 [student_path] [out_prefix]).
Sensitivity computed FRESH from THIS seed's student on frozen sensitivity subset. Test indices never used.
Energy allocator = GREEDY energy baseline: start at smallest r with energy>=0.95/layer; while over budget,
decrement the largest current rank (tie -> L2). NOT a globally optimal energy allocator.
Also evaluates ALL feasible rank pairs per budget on val_es for the Pareto background (analysis only;
allocator selection never uses val_es)."""
import json, sys, os, csv, copy
import numpy as np, torch, torch.nn as nn
sys.path.insert(0,"src")
from models import make_mlp, count_params
from svd_compression import factorize, factorized_params
from data_pipeline import load
torch.set_num_threads(1)
S=int(sys.argv[1]) if len(sys.argv)>1 else 0
STUDENT=sys.argv[2] if len(sys.argv)>2 else (f"models/seed{S}/student_kd_seed{S}.pt" if S!=0 else "models/repro_seed0_v2/student_kd_seed0_v2.pt")
PREFIX=sys.argv[3] if len(sys.argv)>3 else f"seed{S}"
BUDGETS={2:4736,4:2368,8:1184}; LAYERS=[("L1",0,128,8,list(range(1,8))),("L2",2,64,128,list(range(1,43)))]
def get(name,ck):
    X,y,_=load(); split=json.load(open("data/splits/tier1_split.json"))["indices"]; idx=split[name]
    return torch.tensor((X[idx]-np.array(ck["scaler_mean"]))/np.array(ck["scaler_scale"]),dtype=torch.float32), torch.tensor(y[idx],dtype=torch.float32).view(-1,1)
def load_student():
    ck=torch.load(STUDENT,weights_only=False); m=make_mlp(ck["arch"]); m.load_state_dict(ck["state_dict"]); m.eval(); return m,ck
def mse(m,X,y):
    with torch.no_grad(): return float(nn.functional.mse_loss(m(X),y))
def single(base,name,r):
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
    b=min(states.items(),key=lambda kv:(kv[1][0],kv[0])); return b[1][1],b[1][0],b[0]
if __name__=="__main__":
    os.makedirs(f"models/{PREFIX}_compressed",exist_ok=True)
    base,ck=load_student(); Xs,ys=get("sensitivity",ck); Xv,yv=get("val_es",ck)
    bs, bv = mse(base,Xs,ys), mse(base,Xv,yv); print(f"{PREFIX} base sensMSE={bs:.6f} valRMSE={np.sqrt(bv):.4f}")
    S_raw={}; S_clip={}; rows=[]
    for name,pos,m,n,rng in LAYERS:
        S_raw[name]={}; S_clip[name]={}; sv=torch.linalg.svdvals(base[pos].weight.data)
        for r in rng:
            s=mse(single(base,name,r),Xs,ys)-bs; S_raw[name][r]=s; S_clip[name][r]=max(0.0,s)
            rows.append({"seed":S,"layer":name,"rank":r,"params_factorized":factorized_params(m,n,r),"S_raw":round(s,6),"S_clipped":round(max(0.0,s),6)})
    with open(f"results/tables/{PREFIX}_sensitivity_scores.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    neg=sum(1 for r in rows if r["S_raw"]<0); print(f"{PREFIX} sensitivity evals=49 negative={neg}")
    def uniform(B):
        best=None
        for i in range(1,1001):
            rho=i/1000; ranks={"L1":min(7,max(1,int(round(rho*8)))),"L2":min(42,max(1,int(round(rho*64))))}
            if total(ranks)<=B: best=ranks
            else: break
        return best
    def energy_greedy(B):
        ranks={}
        for name,pos,m,n,rng in LAYERS:
            sv=torch.linalg.svdvals(base[pos].weight.data); cum=torch.cumsum(sv**2,0)/torch.sum(sv**2)
            ranks[name]=min(int(torch.searchsorted(cum,0.95).item())+1, max(rng))
        while total(ranks)>B:
            if ranks["L2"]>=ranks["L1"] and ranks["L2"]>1: ranks["L2"]-=1
            elif ranks["L1"]>1: ranks["L1"]-=1
            else: break
        return ranks
    # Pareto background: all rank pairs (params vs val RMSE), analysis only
    pareto=[]
    for r1 in range(1,8):
        for r2 in range(1,43):
            cm=compress(base,{"L1":r1,"L2":r2}); pareto.append({"seed":S,"rank_L1":r1,"rank_L2":r2,"total_params":total({"L1":r1,"L2":r2}),"val_RMSE":round(float(np.sqrt(mse(cm,Xv,yv))),4)})
    with open(f"results/tables/{PREFIX}_all_rank_pairs_val.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=pareto[0].keys()); w.writeheader(); w.writerows(pareto)
    results=[]; alloc_log={}
    for target,B in BUDGETS.items():
        allocs={"uniform":uniform(B),"energy_greedy":energy_greedy(B)}
        dr,dcost,dt=dp(S_clip,B); allocs["sensitivity"]=dr; alloc_log[str(target)]={"budget_B":B,"allocations":allocs,"dp_surrogate_cost":dcost}
        for method,ranks in allocs.items():
            cm=compress(base,ranks); tp=count_params(cm); assert tp==total(ranks)
            with torch.no_grad(): out=cm(Xs[:5]); assert out.shape==(5,1)
            sm, vm = mse(cm,Xs,ys), mse(cm,Xv,yv); joint=sm-bs; pred=sum(S_raw[k][ranks[k]] for k in ranks)
            tag=f"{method}_{target}x_r{ranks['L1']}-{ranks['L2']}"
            torch.save({"state_dict":cm.state_dict(),"arch":ck["arch"],"compressed_ranks":ranks,"method":method,"budget_target":target,"seed":S,"scaler_mean":ck["scaler_mean"],"scaler_scale":ck["scaler_scale"],"val_rmse":round(float(np.sqrt(vm)),4)},f"models/{PREFIX}_compressed/{tag}.pt")
            with torch.no_grad(): mae=float(torch.mean(torch.abs(cm(Xv)-yv)))
            results.append({"seed":S,"method":method,"budget_target":f"{target}x","budget_B":B,"budget_type":"same_max_B_for_all_methods","rank_L1":ranks["L1"],"rank_L2":ranks["L2"],"total_params":tp,"params_vs_budget_diff":B-tp,"complete_ratio":round(9473/tp,3),"eligible_ratio":round(9408/(tp-65),3),"sens_MSE":round(sm,6),"val_RMSE":round(float(np.sqrt(vm)),4),"val_MAE":round(mae,4),"joint_damage_sens":round(joint,6),"predicted_damage_sum_rawS":round(pred,6),"additivity_discrepancy":round(joint-pred,6),"checkpoint":f"models/{PREFIX}_compressed/{tag}.pt"})
            print(f" {PREFIX} {tag} params={tp}/{B} valRMSE={np.sqrt(vm):.4f} addDisc={joint-pred:.5f}")
    with open(f"results/tables/{PREFIX}_compression_results.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
    json.dump(alloc_log,open(f"results/logs/{PREFIX}_dp_allocation.json","w"),indent=1)
    json.dump({"seed":S,"base_val_RMSE":round(float(np.sqrt(bv)),4),"base_sens_MSE":round(bs,6),"negative_sensitivity":neg,"student_checkpoint":STUDENT,"test_set_touched":False,"dataset_note":"Complete dataset array loaded; test indices never used for sensitivity, selection, or validation evaluation."},open(f"results/logs/{PREFIX}_compression_base.json","w"),indent=1)
    print(f"{PREFIX} DONE")
