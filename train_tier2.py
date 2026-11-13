"""Tier-2 training (usage: python src/train_tier2.py 0). SOUTH data only.
Frozen split data/splits/tier2_split.json: train/val_es/sensitivity from Latitude<=38; North (test_north) NEVER loaded here.
Scaler fit on South train only. Same arch/KD/seed policy as Tier-1 (seed before init, identical student init, teacher.eval asserted, train loss logged)."""
import json, sys, os, time, hashlib, csv
import numpy as np, torch, torch.nn as nn
sys.path.insert(0,"src"); from models import make_mlp, count_params; from data_pipeline import load
S=int(sys.argv[1]); CFG=json.load(open("configs/model_training_config.json")); torch.set_num_threads(1)
def set_seed(s): np.random.seed(s); torch.manual_seed(s)
def ph(m):
    h=hashlib.sha256()
    for p in m.parameters(): h.update(p.detach().numpy().tobytes())
    return h.hexdigest()
def get_data():
    X,y,_=load(); sp=json.load(open("data/splits/tier2_split.json"))["indices"]
    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler().fit(X[sp["train"]])  # South train only
    def T(k): return torch.tensor(sc.transform(X[sp[k]]),dtype=torch.float32), torch.tensor(y[sp[k]],dtype=torch.float32).view(-1,1)
    return {k:T(k) for k in ["train","val_es","sensitivity"]}, sc  # north + reserve excluded
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
    print(f"{tag}: epochs={len(hist)} best_south_val_RMSE={best:.4f} wall={wall:.1f}s")
    return model,hist,{"best_val_rmse":round(best,4),"epochs":len(hist),"wall_seconds":round(wall,1),"params":count_params(model)}
def metrics(m,data):
    m.eval(); Xv,yv=data["val_es"]
    with torch.no_grad():
        p=m(Xv); mse=float(nn.functional.mse_loss(p,yv))
        return {"south_val_RMSE":round(float(np.sqrt(mse)),4),"south_val_MAE":round(float(torch.mean(torch.abs(p-yv))),4),"south_val_R2":round(float(1-mse/float(torch.var(yv,unbiased=False))),4)}
if __name__=="__main__":
    os.makedirs(f"models/tier2_seed{S}",exist_ok=True); data,sc=get_data(); runs={}
    set_seed(S); teacher=make_mlp(CFG["teacher"]["architecture"],dropout=CFG["teacher"]["dropout"])
    set_seed(S); teacher,h1,runs["teacher"]=train_one(teacher,data,tag=f"T2_TEACHER_s{S}")
    teacher.eval(); torch.save({"state_dict":teacher.state_dict(),"arch":CFG["teacher"]["architecture"],"dropout":0.1,"seed":S,"tier":2,"scaler_mean":sc.mean_.tolist(),"scaler_scale":sc.scale_.tolist(),"metrics":runs["teacher"]},f"models/tier2_seed{S}/teacher_t2_seed{S}.pt")
    set_seed(S); student=make_mlp(CFG["student"]["architecture"]); ih=ph(student)
    set_seed(S); student,h2,runs["student_KD"]=train_one(student,data,teacher=teacher,alpha=CFG["distillation"]["alpha"],tag=f"T2_KD_s{S}")
    torch.save({"state_dict":student.state_dict(),"arch":CFG["student"]["architecture"],"dropout":0.0,"seed":S,"alpha":0.5,"tier":2,"init_hash":ih,"scaler_mean":sc.mean_.tolist(),"scaler_scale":sc.scale_.tolist(),"metrics":runs["student_KD"]},f"models/tier2_seed{S}/student_kd_t2_seed{S}.pt")
    set_seed(S); ctrl=make_mlp(CFG["student"]["architecture"]); ih2=ph(ctrl); assert ih==ih2
    set_seed(S); ctrl,h3,runs["student_noKD"]=train_one(ctrl,data,teacher=None,tag=f"T2_noKD_s{S}")
    torch.save({"state_dict":ctrl.state_dict(),"arch":CFG["student"]["architecture"],"dropout":0.0,"seed":S,"alpha":0.0,"tier":2,"init_hash":ih2,"scaler_mean":sc.mean_.tolist(),"scaler_scale":sc.scale_.tolist(),"metrics":runs["student_noKD"]},f"models/tier2_seed{S}/student_nokd_t2_seed{S}.pt")
    for n,m in [("teacher",teacher),("student_KD",student),("student_noKD",ctrl)]: runs[n].update(metrics(m,data))
    for tag,h in [("teacher",h1),("student_kd",h2),("student_nokd",h3)]:
        with open(f"results/logs/tier2_seed{S}_{tag}_curves.csv","w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["epoch","train_loss","val_rmse"]); w.writeheader(); w.writerows(h)
    runs.update({"seed":S,"tier":2,"student_init_hash":ih,"init_identical":ih==ih2,"north_test_touched":False,"region":"South lat<=38 only"})
    json.dump(runs,open(f"results/logs/tier2_seed{S}_training_summary.json","w"),indent=1); print(json.dumps(runs,indent=1))
