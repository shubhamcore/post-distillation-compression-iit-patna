"""Reproducible Tier-1 seed-0 run (v2). Fixes Day-3 audit findings:
SEED POLICY: set_seed(SEED) is called BEFORE every model initialization.
 - Teacher: seed -> init -> train (seed governs init, shuffling, dropout).
 - KD student: seed -> init (record init hash) -> seed again before training loop for shuffling.
 - no-KD control: SAME seed -> init (hash must equal KD student's init hash: identical starting weights) -> train.
TEACHER MODE: teacher.eval() explicitly + assert teacher.training is False before/during KD.
LOGS: per-epoch average training loss + validation RMSE.
Original Day-3 checkpoints (models/day3_seed0/) are NOT touched. Test set NOT loaded.
"""
import json, time, resource, sys, os, hashlib, csv
import numpy as np, torch, torch.nn as nn
sys.path.insert(0,"src")
from models import make_mlp, count_params
from data_pipeline import load
SEED=0; CFG=json.load(open("configs/model_training_config.json")); torch.set_num_threads(1)

def set_seed(s): np.random.seed(s); torch.manual_seed(s)
def param_hash(model):
    h=hashlib.sha256()
    for p in model.parameters(): h.update(p.detach().numpy().tobytes())
    return h.hexdigest()

def get_data():
    X,y,_=load(); split=json.load(open("data/splits/tier1_split.json"))["indices"]
    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler().fit(X[split["train"]])
    def T(idxs): return torch.tensor(sc.transform(X[idxs]),dtype=torch.float32), torch.tensor(y[idxs],dtype=torch.float32).view(-1,1)
    return {k:T(v) for k,v in split.items() if k!="test"}, sc

def train_one(model, data, teacher=None, alpha=0.5, tag=""):
    # NOTE: seed/init handled by caller (per seed policy). This function only trains.
    if teacher is not None:
        teacher.eval()
        assert teacher.training is False, "Teacher must be in eval mode during distillation"
        print(f"{tag}: teacher.training={teacher.training} (asserted False)")
    opt=torch.optim.Adam(model.parameters(), lr=CFG["training"]["learning_rate"])
    bs=CFG["training"]["batch_size"]; Xtr,ytr=data["train"]; Xv,yv=data["val_es"]
    best=float("inf"); best_state=None; wait=0; hist=[]; t0=time.time()
    for ep in range(CFG["training"]["max_epochs"]):
        if teacher is not None: assert teacher.training is False
        model.train(); perm=torch.randperm(len(Xtr)); tot, nb = 0.0, 0
        for i in range(0,len(Xtr),bs):
            idx=perm[i:i+bs]; xb,yb=Xtr[idx],ytr[idx]; pred=model(xb)
            if teacher is not None:
                with torch.no_grad(): tp=teacher(xb)
                loss=alpha*nn.functional.mse_loss(pred,tp)+(1-alpha)*nn.functional.mse_loss(pred,yb)
            else: loss=nn.functional.mse_loss(pred,yb)
            opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.detach().item(); nb+=1
        model.eval()
        with torch.no_grad(): vrmse=float(torch.sqrt(nn.functional.mse_loss(model(Xv),yv)))
        hist.append({"epoch":ep+1,"train_loss":round(tot/nb,6),"val_rmse":round(vrmse,6)}); print(f"  {tag} ep{ep+1} train_loss={tot/nb:.4f} val_RMSE={vrmse:.4f}", flush=True) if (ep+1)%50==0 else None
        if vrmse<best-1e-6: best,wait=vrmse,0; best_state={k:v.clone() for k,v in model.state_dict().items()}
        else:
            wait+=1
            if wait>=CFG["training"]["early_stopping"]["patience"]: break
    wall=time.time()-t0
    if best_state: model.load_state_dict(best_state)
    print(f"{tag}: epochs={len(hist)} best_val_RMSE={best:.4f} wall={wall:.1f}s params={count_params(model)}")
    return model, hist, {"best_val_rmse":round(best,4),"epochs":len(hist),"wall_seconds":round(wall,1),"params":count_params(model)}

def metrics(model, data):
    model.eval(); Xv,yv=data["val_es"]
    with torch.no_grad():
        p=model(Xv); mse=float(nn.functional.mse_loss(p,yv))
        return {"val_RMSE":round(float(np.sqrt(mse)),4),"val_MAE":round(float(torch.mean(torch.abs(p-yv))),4),"val_R2":round(float(1-mse/float(torch.var(yv,unbiased=False))),4)}

if __name__=="__main__":
    os.makedirs("models/repro_seed0_v2",exist_ok=True); data,sc=get_data()
    runs={}; log={"seed_policy":"set_seed(0) BEFORE each init; KD and no-KD students share identical init (hash-checked)"}
    # TEACHER: seed before init
    set_seed(SEED); teacher=make_mlp(CFG["teacher"]["architecture"], dropout=CFG["teacher"]["dropout"]); log["teacher_init_hash"]=param_hash(teacher)
    set_seed(SEED); teacher,hist_t,runs["teacher"]=train_one(teacher,data,tag="TEACHER")
    teacher.eval(); torch.save({"state_dict":teacher.state_dict(),"arch":CFG["teacher"]["architecture"],"dropout":0.1,"seed":SEED,"version":"repro_v2","scaler_mean":sc.mean_.tolist(),"scaler_scale":sc.scale_.tolist(),"metrics":runs["teacher"]},"models/repro_seed0_v2/teacher_seed0_v2.pt")
    # KD STUDENT: seed before init, hash, then train (teacher.eval asserted inside)
    set_seed(SEED); student=make_mlp(CFG["student"]["architecture"]); log["student_kd_init_hash"]=param_hash(student)
    set_seed(SEED); student,hist_s,runs["student_KD"]=train_one(student,data,teacher=teacher,alpha=CFG["distillation"]["alpha"],tag="STUDENT_KD")
    torch.save({"state_dict":student.state_dict(),"arch":CFG["student"]["architecture"],"dropout":0.0,"seed":SEED,"alpha":0.5,"version":"repro_v2","init_hash":log["student_kd_init_hash"],"scaler_mean":sc.mean_.tolist(),"scaler_scale":sc.scale_.tolist(),"metrics":runs["student_KD"]},"models/repro_seed0_v2/student_kd_seed0_v2.pt")
    # NO-KD CONTROL: same seed before init -> identical weights
    set_seed(SEED); ctrl=make_mlp(CFG["student"]["architecture"]); log["student_nokd_init_hash"]=param_hash(ctrl)
    assert log["student_kd_init_hash"]==log["student_nokd_init_hash"], "Student inits must be identical"
    print(f"INIT CHECK: KD init hash == noKD init hash: {log['student_kd_init_hash'][:16]}... IDENTICAL")
    set_seed(SEED); ctrl,hist_c,runs["student_noKD"]=train_one(ctrl,data,teacher=None,tag="STUDENT_noKD")
    torch.save({"state_dict":ctrl.state_dict(),"arch":CFG["student"]["architecture"],"dropout":0.0,"seed":SEED,"alpha":0.0,"version":"repro_v2","init_hash":log["student_nokd_init_hash"],"scaler_mean":sc.mean_.tolist(),"scaler_scale":sc.scale_.tolist(),"metrics":runs["student_noKD"]},"models/repro_seed0_v2/student_nokd_seed0_v2.pt")
    for name,m in [("teacher",teacher),("student_KD",student),("student_noKD",ctrl)]: runs[name].update(metrics(m,data))
    for tag,hist in [("teacher",hist_t),("student_kd",hist_s),("student_nokd",hist_c)]:
        with open(f"results/logs/repro_{tag}_curves_seed0_v2.csv","w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["epoch","train_loss","val_rmse"]); w.writeheader(); w.writerows(hist)
    runs.update(log); runs["teacher_eval_asserted"]=True; runs["test_set_touched"]=False
    runs["resource"]={"max_rss_MB":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1),"torch_threads":1,"device":"CPU (sandbox)"}
    json.dump(runs,open("results/logs/repro_seed0_v2_summary.json","w"),indent=1); print(json.dumps(runs,indent=1))
