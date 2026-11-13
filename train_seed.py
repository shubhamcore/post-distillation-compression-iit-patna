"""Train Tier-1 for a given seed (usage: python src/train_seed.py 1). Same seed policy as repro v2.
Seed set BEFORE each init; KD/no-KD identical init (hash asserted); teacher.eval asserted; train loss logged.
Test split excluded. Outputs: models/seed{S}/, results/logs/seed{S}_*"""
import json, sys, os, time, hashlib, csv, resource
import numpy as np, torch, torch.nn as nn
sys.path.insert(0,"src")
from models import make_mlp, count_params
from data_pipeline import load
S=int(sys.argv[1]); CFG=json.load(open("configs/model_training_config.json")); torch.set_num_threads(1)
def set_seed(s): np.random.seed(s); torch.manual_seed(s)
def ph(m):
    h=hashlib.sha256()
    for p in m.parameters(): h.update(p.detach().numpy().tobytes())
    return h.hexdigest()
def get_data():
    X,y,_=load(); split=json.load(open("data/splits/tier1_split.json"))["indices"]
    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler().fit(X[split["train"]])
    def T(i): return torch.tensor(sc.transform(X[i]),dtype=torch.float32), torch.tensor(y[i],dtype=torch.float32).view(-1,1)
    return {k:T(v) for k,v in split.items() if k!="test"}, sc
def train_one(model,data,teacher=None,alpha=0.5,tag=""):
    if teacher is not None:
        teacher.eval(); assert teacher.training is False; print(f"{tag}: teacher.training=False asserted")
    opt=torch.optim.Adam(model.parameters(),lr=CFG["training"]["learning_rate"]); bs=CFG["training"]["batch_size"]
    Xtr,ytr=data["train"]; Xv,yv=data["val_es"]; best=float("inf"); best_state=None; wait=0; hist=[]; t0=time.time()
    for ep in range(CFG["training"]["max_epochs"]):
        if teacher is not None: assert teacher.training is False
        model.train(); perm=torch.randperm(len(Xtr)); tot,nb=0.0,0
        for i in range(0,len(Xtr),bs):
            idx=perm[i:i+bs]; xb,yb=Xtr[idx],ytr[idx]; pred=model(xb)
            if teacher is not None:
                with torch.no_grad(): tp=teacher(xb)
                loss=alpha*nn.functional.mse_loss(pred,tp)+(1-alpha)*nn.functional.mse_loss(pred,yb)
            else: loss=nn.functional.mse_loss(pred,yb)
            opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.detach().item(); nb+=1
        model.eval()
        with torch.no_grad(): vr=float(torch.sqrt(nn.functional.mse_loss(model(Xv),yv)))
        hist.append({"epoch":ep+1,"train_loss":round(tot/nb,6),"val_rmse":round(vr,6)})
        if vr<best-1e-6: best,wait=vr,0; best_state={k:v.clone() for k,v in model.state_dict().items()}
        else:
            wait+=1
            if wait>=CFG["training"]["early_stopping"]["patience"]: break
    wall=time.time()-t0
    if best_state: model.load_state_dict(best_state)
    print(f"{tag}: epochs={len(hist)} best_val_RMSE={best:.4f} wall={wall:.1f}s"); return model,hist,{"best_val_rmse":round(best,4),"epochs":len(hist),"wall_seconds":round(wall,1),"params":count_params(model)}
def metrics(model,data):
    model.eval(); Xv,yv=data["val_es"]
    with torch.no_grad():
        p=model(Xv); mse=float(nn.functional.mse_loss(p,yv))
        return {"val_RMSE":round(float(np.sqrt(mse)),4),"val_MAE":round(float(torch.mean(torch.abs(p-yv))),4),"val_R2":round(float(1-mse/float(torch.var(yv,unbiased=False))),4)}
if __name__=="__main__":
    os.makedirs(f"models/seed{S}",exist_ok=True); data,sc=get_data(); runs={}
    set_seed(S); teacher=make_mlp(CFG["teacher"]["architecture"],dropout=CFG["teacher"]["dropout"]); ih_t=ph(teacher)
    set_seed(S); teacher,h1,runs["teacher"]=train_one(teacher,data,tag=f"TEACHER_s{S}")
    teacher.eval(); torch.save({"state_dict":teacher.state_dict(),"arch":CFG["teacher"]["architecture"],"dropout":0.1,"seed":S,"scaler_mean":sc.mean_.tolist(),"scaler_scale":sc.scale_.tolist(),"metrics":runs["teacher"]},f"models/seed{S}/teacher_seed{S}.pt")
    set_seed(S); student=make_mlp(CFG["student"]["architecture"]); ih_kd=ph(student)
    set_seed(S); student,h2,runs["student_KD"]=train_one(student,data,teacher=teacher,alpha=CFG["distillation"]["alpha"],tag=f"KD_s{S}")
    torch.save({"state_dict":student.state_dict(),"arch":CFG["student"]["architecture"],"dropout":0.0,"seed":S,"alpha":0.5,"init_hash":ih_kd,"scaler_mean":sc.mean_.tolist(),"scaler_scale":sc.scale_.tolist(),"metrics":runs["student_KD"]},f"models/seed{S}/student_kd_seed{S}.pt")
    set_seed(S); ctrl=make_mlp(CFG["student"]["architecture"]); ih_nk=ph(ctrl)
    assert ih_kd==ih_nk; print(f"INIT IDENTICAL hash {ih_kd[:16]}")
    set_seed(S); ctrl,h3,runs["student_noKD"]=train_one(ctrl,data,teacher=None,tag=f"noKD_s{S}")
    torch.save({"state_dict":ctrl.state_dict(),"arch":CFG["student"]["architecture"],"dropout":0.0,"seed":S,"alpha":0.0,"init_hash":ih_nk,"scaler_mean":sc.mean_.tolist(),"scaler_scale":sc.scale_.tolist(),"metrics":runs["student_noKD"]},f"models/seed{S}/student_nokd_seed{S}.pt")
    for n,m in [("teacher",teacher),("student_KD",student),("student_noKD",ctrl)]: runs[n].update(metrics(m,data))
    for tag,h in [("teacher",h1),("student_kd",h2),("student_nokd",h3)]:
        with open(f"results/logs/seed{S}_{tag}_curves.csv","w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["epoch","train_loss","val_rmse"]); w.writeheader(); w.writerows(h)
    runs.update({"seed":S,"student_init_hash_kd":ih_kd,"student_init_hash_nokd":ih_nk,"init_identical":ih_kd==ih_nk,"test_set_touched":False,
                 "note":"Complete dataset array loaded by pipeline; test indices never sliced for training/sensitivity/selection/validation in this run."})
    json.dump(runs,open(f"results/logs/seed{S}_training_summary.json","w"),indent=1); print(json.dumps(runs,indent=1))
