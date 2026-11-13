"""Fresh-process verification of repro v2: reload 3 checkpoints, recompute val RMSE, check init-hash equality + teacher eval assertion log."""
import json, sys, torch, numpy as np
sys.path.insert(0,"src"); from models import make_mlp; from data_pipeline import load
X,y,_=load(); split=json.load(open("data/splits/tier1_split.json"))["indices"]
summ=json.load(open("results/logs/repro_seed0_v2_summary.json"))
print("init_hash KD == noKD:", summ["student_kd_init_hash"]==summ["student_nokd_init_hash"], summ["student_kd_init_hash"][:16])
print("teacher_eval_asserted in summary:", summ["teacher_eval_asserted"])
for name,f,arch,drop in [("teacher","models/repro_seed0_v2/teacher_seed0_v2.pt",[8,256,256,128,1],0.1),
                          ("student_KD","models/repro_seed0_v2/student_kd_seed0_v2.pt",[8,128,64,1],0.0),
                          ("student_noKD","models/repro_seed0_v2/student_nokd_seed0_v2.pt",[8,128,64,1],0.0)]:
    ck=torch.load(f,weights_only=False); m=make_mlp(arch,dropout=drop); m.load_state_dict(ck["state_dict"]); m.eval()
    Xv=torch.tensor((X[split["val_es"]]-np.array(ck["scaler_mean"]))/np.array(ck["scaler_scale"]),dtype=torch.float32)
    with torch.no_grad(): p=m(Xv).numpy().ravel()
    rmse=float(np.sqrt(np.mean((p-y[split["val_es"]])**2)))
    print(f"{name}: {f} reloaded val_RMSE={rmse:.4f} logged={summ[name]['val_RMSE']} match={abs(rmse-summ[name]['val_RMSE'])<1e-3} version={ck.get('version')}")
print("REPRO V2 VERIFICATION DONE (test set not used)")
