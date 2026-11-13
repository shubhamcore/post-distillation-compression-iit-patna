"""Day 3: Tier-1 seed 0 ONLY. Teacher, KD student, no-KD control. Test set NOT touched."""
import json, time, resource, sys, os
import numpy as np, torch, torch.nn as nn
sys.path.insert(0,"src")
from models import make_mlp, count_params
from data_pipeline import load

SEED=0; CFG=json.load(open("configs/model_training_config.json"))
torch.set_num_threads(1)  # recorded; sandbox CPU
def set_seed(s): np.random.seed(s); torch.manual_seed(s)

def get_data():
    X,y,_=load(); split=json.load(open("data/splits/tier1_split.json"))["indices"]
    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler().fit(X[split["train"]])
    def T(idxs): return torch.tensor(sc.transform(X[idxs]),dtype=torch.float32), torch.tensor(y[idxs],dtype=torch.float32).view(-1,1)
    return {k:T(v) for k,v in split.items() if k!="test"}, sc, split  # test excluded deliberately

def train_one(model, data, teacher=None, alpha=0.5, tag=""):
    set_seed(SEED)
    opt=torch.optim.Adam(model.parameters(), lr=CFG["training"]["learning_rate"])
    bs=CFG["training"]["batch_size"]; Xtr,ytr=data["train"]; Xv,yv=data["val_es"]
    best=float("inf"); best_state=None; wait=0; hist=[]
    t0=time.time()
    for ep in range(CFG["training"]["max_epochs"]):
        model.train(); perm=torch.randperm(len(Xtr))
        for i in range(0,len(Xtr),bs):
            idx=perm[i:i+bs]; xb,yb=Xtr[idx],ytr[idx]
            pred=model(xb)
            if teacher is not None:
                with torch.no_grad(): tp=teacher(xb)
                loss=alpha*nn.functional.mse_loss(pred,tp)+(1-alpha)*nn.functional.mse_loss(pred,yb)
            else: loss=nn.functional.mse_loss(pred,yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vrmse=float(torch.sqrt(nn.functional.mse_loss(model(Xv),yv)))
        hist.append({"epoch":ep+1,"val_rmse":vrmse})
        if vrmse<best-1e-6: best,wait=vrmse,0; best_state={k:v.clone() for k,v in model.state_dict().items()}
        else:
            wait+=1
            if wait>=CFG["training"]["early_stopping"]["patience"]: break
    wall=time.time()-t0
    if best_state: model.load_state_dict(best_state)
    print(f"{tag}: epochs={len(hist)} best_val_RMSE={best:.4f} wall={wall:.1f}s params={count_params(model)}")
    return model, hist, {"best_val_rmse":best,"epochs":len(hist),"wall_seconds":round(wall,1),"params":count_params(model)}

def metrics(model, data):
    model.eval()
    with torch.no_grad():
        Xv,yv=data["val_es"]; p=model(Xv); mse=float(nn.functional.mse_loss(p,yv))
        return {"val_RMSE":round(float(np.sqrt(mse)),4),"val_MAE":round(float(torch.mean(torch.abs(p-yv))),4),
                "val_R2":round(float(1-mse/float(torch.var(yv,unbiased=False))),4)}

if __name__=="__main__":
    os.makedirs("models/day3_seed0",exist_ok=True); os.makedirs("results/logs",exist_ok=True)
    data,sc,split=get_data()
    print("data (test excluded):",{k:list(v[0].shape) for k,v in data.items()})
    runs={}
    teacher=make_mlp(CFG["teacher"]["architecture"], dropout=CFG["teacher"]["dropout"])
    teacher,hist_t,runs["teacher"]=train_one(teacher,data,tag="TEACHER")
    torch.save({"state_dict":teacher.state_dict(),"arch":CFG["teacher"]["architecture"],"dropout":0.1,"seed":SEED,"scaler_mean":sc.mean_.tolist(),"scaler_scale":sc.scale_.tolist(),"metrics":runs["teacher"]},"models/day3_seed0/teacher_seed0.pt")
    student=make_mlp(CFG["student"]["architecture"])
    student,hist_s,runs["student_KD"]=train_one(student,data,teacher=teacher,alpha=CFG["distillation"]["alpha"],tag="STUDENT_KD")
    torch.save({"state_dict":student.state_dict(),"arch":CFG["student"]["architecture"],"dropout":0.0,"seed":SEED,"alpha":0.5,"scaler_mean":sc.mean_.tolist(),"scaler_scale":sc.scale_.tolist(),"metrics":runs["student_KD"]},"models/day3_seed0/student_kd_seed0.pt")
    ctrl=make_mlp(CFG["student"]["architecture"])
    ctrl,hist_c,runs["student_noKD"]=train_one(ctrl,data,teacher=None,tag="STUDENT_noKD")
    torch.save({"state_dict":ctrl.state_dict(),"arch":CFG["student"]["architecture"],"dropout":0.0,"seed":SEED,"alpha":0.0,"scaler_mean":sc.mean_.tolist(),"scaler_scale":sc.scale_.tolist(),"metrics":runs["student_noKD"]},"models/day3_seed0/student_nokd_seed0.pt")
    for name,m in [("teacher",teacher),("student_KD",student),("student_noKD",ctrl)]: runs[name].update(metrics(m,data))
    import csv
    for tag,hist in [("teacher",hist_t),("student_kd",hist_s),("student_nokd",hist_c)]:
        with open(f"results/logs/day3_{tag}_val_curve_seed0.csv","w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["epoch","val_rmse"]); w.writeheader(); w.writerows(hist)
    runs["resource"]={"max_rss_MB":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1),"torch_threads":torch.get_num_threads(),"device":"CPU (sandbox)"}
    runs["test_set_touched"]=False
    json.dump(runs,open("results/logs/day3_seed0_summary.json","w"),indent=1)
    print(json.dumps(runs,indent=1))
