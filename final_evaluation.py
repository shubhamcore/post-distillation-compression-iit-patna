"""FINAL TEST EVALUATION — RUN ONCE, ONLY AFTER APPROVAL.
Loads frozen checkpoints (teacher, KD student, no-KD control, 27 frozen compressed, 27 fine-tuned),
evaluates ONLY the frozen Tier-1 test split. Never updates weights, never selects ranks.
Usage:
  python src/final_evaluation.py --list          # list frozen checkpoints, NO test access
  python src/final_evaluation.py --run-once      # THE single final run (writes results/tables/final_test_results_raw.csv + log)
Guard: --run-once refuses to run if the output CSV already exists (prevents accidental second run).
Metrics: RMSE, MAE, R2, params, serialized size (bytes). Each row records checkpoint path + ranks/method/seed."""
import json, sys, os, glob, argparse
import numpy as np, torch, torch.nn as nn
sys.path.insert(0,"src"); from models import make_mlp, count_params; from data_pipeline import load

def checkpoint_list():
    items=[]
    items.append(("teacher","models/repro_seed0_v2/teacher_seed0_v2.pt",[8,256,256,128,1],0.1,None,"seed0"))
    for s in [1,2]: items.append(("teacher",f"models/seed{s}/teacher_seed{s}.pt",[8,256,256,128,1],0.1,None,f"seed{s}"))
    items.append(("student_KD","models/repro_seed0_v2/student_kd_seed0_v2.pt",[8,128,64,1],0.0,None,"seed0"))
    items.append(("student_noKD","models/repro_seed0_v2/student_nokd_seed0_v2.pt",[8,128,64,1],0.0,None,"seed0"))
    for s in [1,2]:
        items.append(("student_KD",f"models/seed{s}/student_kd_seed{s}.pt",[8,128,64,1],0.0,None,f"seed{s}"))
        items.append(("student_noKD",f"models/seed{s}/student_nokd_seed{s}.pt",[8,128,64,1],0.0,None,f"seed{s}"))
    for f in sorted(glob.glob("models/seed*_compressed/*.pt")): items.append(("compressed_frozen",f,None,0.0,None,""))
    for f in sorted(glob.glob("models/seed*_finetuned/*.pt")): items.append(("compressed_finetuned",f,None,0.0,None,""))
    return items

def build(path, kind):
    ck=torch.load(path,weights_only=False)
    arch=ck.get("arch",[8,128,64,1]); m=make_mlp(arch,dropout=ck.get("dropout",0.0))
    if "compressed_ranks" in ck:
        for pos,name in [(0,"L1"),(2,"L2")]:
            lin=m[pos]; mm,nn_=lin.weight.shape; r=ck["compressed_ranks"][name]
            m[pos]=nn.Sequential(nn.Linear(nn_,r,bias=False),nn.Linear(r,mm,bias=True))
    m.load_state_dict(ck["state_dict"]); m.eval(); return m,ck

EXPECTED_TIER1_SHA256="9a586abd92a948009ba18a78a5aa5cbc421370ddde11dbae1f3022a9b86eb490"
def preflight():
    """Verify split, all 63 checkpoints + scaler metadata, dataset/deps BEFORE any test access. Returns list of failures."""
    import hashlib
    fails=[]
    h=hashlib.sha256(open("data/splits/tier1_split.json","rb").read()).hexdigest()
    if h!=EXPECTED_TIER1_SHA256: fails.append(f"tier1_split.json sha256 mismatch: {h}")
    items=checkpoint_list()
    if len(items)!=63: fails.append(f"expected 63 checkpoints, found {len(items)}")
    # Frozen checkpoint hash manifest verification
    import csv as _csv, hashlib as _hl
    manifest={}
    try:
        with open("configs/frozen_checkpoint_sha256_manifest.csv") as fh:
            for row in _csv.DictReader(fh): manifest[row["checkpoint"]]=row["sha256"]
        if len(manifest)!=63: fails.append(f"hash manifest has {len(manifest)} entries, expected 63")
    except Exception as e: fails.append(f"hash manifest unreadable: {e}")
    # Canonical scaler from frozen Tier-1 training split (comparison rule: shape (8,), finite, scale>0, max-abs-diff <= 1e-10)
    canonical=None
    try:
        from sklearn.preprocessing import StandardScaler
        _X,_,_=load(); _split=json.load(open("data/splits/tier1_split.json"))["indices"]
        canonical=StandardScaler().fit(_X[_split["train"]])
    except Exception as e: fails.append(f"canonical scaler fit failed: {e}")
    for kind,path,*_ in items:
        if not os.path.exists(path): fails.append(f"missing checkpoint {path}"); continue
        try:
            actual=_hl.sha256(open(path,"rb").read()).hexdigest()
            if path in manifest and actual!=manifest[path]: fails.append(f"{path}: SHA256 MISMATCH vs frozen manifest")
            ck=torch.load(path,weights_only=False)
            for key in ["state_dict","scaler_mean","scaler_scale"]:
                if key not in ck: fails.append(f"{path}: missing {key}")
            sm=np.array(ck["scaler_mean"]); ss=np.array(ck["scaler_scale"])
            if sm.shape!=(8,) or ss.shape!=(8,) or not np.isfinite(sm).all() or not np.isfinite(ss).all() or (ss<=0).any():
                fails.append(f"{path}: scaler metadata inconsistent (shapes/finite/positive-scale)")
            elif canonical is not None:
                if float(np.max(np.abs(sm-canonical.mean_)))>1e-10 or float(np.max(np.abs(ss-canonical.scale_)))>1e-10:
                    fails.append(f"{path}: scaler differs from canonical Tier-1 train scaler (tol 1e-10)")
        except Exception as e: fails.append(f"{path}: unloadable {e}")
    try:
        X,y,_=load()
        if X.shape!=(20640,8): fails.append(f"dataset shape unexpected {X.shape}")
    except Exception as e: fails.append(f"dataset load failed {e}")
    print(f"PREFLIGHT: tier1 sha256 ok={h==EXPECTED_TIER1_SHA256}, checkpoints={len(items)}, failures={len(fails)}")
    for f in fails: print("  FAIL:",f)
    if not fails: print("PREFLIGHT PASSED — no test data accessed during preflight")
    return fails

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--list",action="store_true"); ap.add_argument("--run-once",action="store_true"); ap.add_argument("--preflight",action="store_true"); a=ap.parse_args()
    items=checkpoint_list()
    if a.preflight:
        fails=preflight(); sys.exit(1 if fails else 0)
    if a.list or not a.run_once:
        print(f"Frozen checkpoints to evaluate ({len(items)}):")
        for kind,path,*_ in items: print(f"  [{kind}] {path} exists={os.path.exists(path)} size={os.path.getsize(path) if os.path.exists(path) else 'MISSING'}")
        print("NO test data accessed. Run with --run-once only after approval.")
        sys.exit(0)
    out="results/tables/final_test_results_raw.csv"
    if os.path.exists(out): print(f"REFUSING: {out} already exists — final test evaluation must run only once."); sys.exit(1)
    # --- preflight gate: must pass BEFORE any test access ---
    if preflight(): print("PREFLIGHT FAILED — aborting without evaluating test set."); sys.exit(1)
    # --- the single final run below this line ---
    torch.set_num_threads(1)
    X,y,_=load(); split=json.load(open("data/splits/tier1_split.json"))["indices"]; idx=split["test"]
    rows=[]; log=[f"FINAL TEST EVALUATION (single run) checkpoints={len(items)} test_n={len(idx)}"]
    for kind,path,*_ in items:
        m,ck=build(path,kind)
        Xs=torch.tensor((X[idx]-np.array(ck["scaler_mean"]))/np.array(ck["scaler_scale"]),dtype=torch.float32)
        with torch.no_grad(): p=m(Xs).numpy().ravel()
        yt=y[idx]; rmse=float(np.sqrt(np.mean((p-yt)**2))); mae=float(np.mean(np.abs(p-yt))); r2=float(1-np.sum((p-yt)**2)/np.sum((yt-yt.mean())**2))
        seed = ck.get("seed","seed0_from_path"); method=ck.get("method",kind)
        rows.append({"kind":kind,"method":method,"seed":seed,"budget_target":ck.get("budget_target",""),
                     "rank_L1":ck.get("compressed_ranks",{}).get("L1",""),"rank_L2":ck.get("compressed_ranks",{}).get("L2",""),
                     "finetuned":ck.get("finetuned",False),"checkpoint":path,"checkpoint_bytes":os.path.getsize(path),
                     "params":count_params(m),"test_RMSE":rmse,"test_MAE":mae,"test_R2":r2})  # FULL precision in raw CSV; rounding only in final report
        log.append(f"{path}: RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f} params={count_params(m)}")
    import csv
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    open("results/logs/final_test_evaluation_log.txt","w").write("\n".join(log))
    print("\n".join(log)); print(f"Saved {out} — DO NOT RERUN.")
