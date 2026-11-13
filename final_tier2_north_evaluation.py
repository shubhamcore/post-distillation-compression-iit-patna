"""TIER-2 NORTH EVALUATION — RUN ONCE, ONLY AFTER INDEPENDENT APPROVAL.
Evaluates the 63 frozen Tier-2 models on North (test_north, Latitude>38) ONLY.
Never trains, scales (uses each checkpoint's stored South scaler), stops, selects, or tunes.
Usage: --list / --preflight (no North evaluation) | --run-once (the single run; refuses if outputs exist).
Preflight verifies: exact tier2 split SHA256, 63 checkpoint hashes vs configs/tier2_checkpoint_sha256_manifest.csv,
scaler consistency vs canonical South-train scaler (shape (8,), finite, positive, max-abs-diff <=1e-10),
architecture/param reconstruction for all 63, North/South disjointness, output absence. Any failure aborts BEFORE North evaluation.
Outputs: results/tables/tier2_north_test_results_raw.csv (FULL precision), results/logs/tier2_north_evaluation_log.txt.
"""
import json, sys, os, glob, argparse, hashlib, csv
import numpy as np, torch, torch.nn as nn
sys.path.insert(0,"src"); from models import make_mlp, count_params; from data_pipeline import load
EXPECTED_SPLIT_SHA="08be45d1d93995499182e2a00659978d719d24723241313ca38655d44040b67b"
OUT_CSV="results/tables/tier2_north_test_results_raw.csv"; OUT_SUM="results/tables/tier2_north_test_summary.csv"; OUT_LOG="results/logs/tier2_north_evaluation_log.txt"
LOCK_FILE="results/logs/tier2_north_evaluation.lock"

def checkpoint_list():
    items=[]
    for s in [0,1,2]:
        items.append(("teacher",f"models/tier2_seed{s}/teacher_t2_seed{s}.pt"))
        items.append(("student_KD",f"models/tier2_seed{s}/student_kd_t2_seed{s}.pt"))
        items.append(("student_noKD",f"models/tier2_seed{s}/student_nokd_t2_seed{s}.pt"))
    for f in sorted(glob.glob("models/tier2_seed*_compressed/*.pt")): items.append(("compressed_frozen",f))
    for f in sorted(glob.glob("models/tier2_seed*_finetuned/*.pt")): items.append(("compressed_finetuned",f))
    return items

def build(path):
    ck=torch.load(path,weights_only=False); m=make_mlp(ck["arch"],dropout=ck.get("dropout",0.0))
    if "compressed_ranks" in ck:
        for pos,name in [(0,"L1"),(2,"L2")]:
            lin=m[pos]; mm,nn_=lin.weight.shape; r=ck["compressed_ranks"][name]
            m[pos]=nn.Sequential(nn.Linear(nn_,r,bias=False),nn.Linear(r,mm,bias=True))
    m.load_state_dict(ck["state_dict"]); m.eval(); return m,ck

def preflight():
    fails=[]
    h=hashlib.sha256(open("data/splits/tier2_split.json","rb").read()).hexdigest()
    if h!=EXPECTED_SPLIT_SHA: fails.append("tier2_split.json SHA256 mismatch")
    sp=json.load(open("data/splits/tier2_split.json"))["indices"]
    sets={k:set(v) for k,v in sp.items()}; names=list(sets)
    if any(sets[a]&sets[b] for i,a in enumerate(names) for b in names[i+1:]): fails.append("split partitions not disjoint")
    if set().union(*sets.values())!=set(range(20640)): fails.append("split union does not cover exactly 20640")
    items=checkpoint_list()
    if len(items)!=63: fails.append(f"expected 63 models, found {len(items)}")
    manifest={}
    try:
        with open("configs/tier2_checkpoint_sha256_manifest.csv") as fh:
            for row in csv.DictReader(fh): manifest[row["checkpoint"]]=row["sha256"]
    except Exception as e: fails.append(f"hash manifest unreadable {e}")
    canonical=None
    try:
        from sklearn.preprocessing import StandardScaler
        X,_,_=load(); canonical=StandardScaler().fit(X[sp["train"]])  # South train indices only
    except Exception as e: fails.append(f"canonical South scaler failed {e}")

    # Geographic boundary + coverage enforcement (dataset feature col 6 = Latitude)
    try:
        _X,_,_ = load(); _lat=_X[:,6]; sp2=sp
        north=np.array(sp2["test_north"]); south=np.concatenate([np.array(sp2[k]) for k in ["train","val_es","sensitivity","south_holdout_reserve"]])
        if not bool((_lat[north]>38.0).all()): fails.append("geography: some test_north Latitude <= 38.0")
        if not bool((_lat[south]<=38.0).all()): fails.append("geography: some South Latitude > 38.0")
        if len(set(north.tolist()) & set(south.tolist()))>0: fails.append("geography: North/South overlap")
        print(f" geography check: north lat min={_lat[north].min():.2f} south lat max={_lat[south].max():.2f}")
    except Exception as e: fails.append(f"geography check failed: {e}")
    # Expected parameter counts
    EXPECTED_BASE={"teacher":101121,"student_KD":9473,"student_noKD":9473}
    for kind,path in items:
        if not os.path.exists(path): fails.append(f"missing {path}"); continue
        try:
            if hashlib.sha256(open(path,"rb").read()).hexdigest()!=manifest.get(path): fails.append(f"{path}: hash mismatch vs manifest")
            m,ck=build(path)  # reconstruct architecture
            actual_params=count_params(m)
            if "compressed_ranks" in ck:
                r1=ck["compressed_ranks"].get("L1"); r2=ck["compressed_ranks"].get("L2")
                expected=136*r1 + 192*r2 + 257  # P(r1,r2): factorised L1/L2 with biases + output layer (65) + retained biases, as frozen
                if actual_params!=expected: fails.append(f"{path}: param mismatch reconstructed={actual_params} expected P({r1},{r2})={expected}")
            elif kind in EXPECTED_BASE:
                if actual_params!=EXPECTED_BASE[kind]: fails.append(f"{path}: param mismatch reconstructed={actual_params} expected={EXPECTED_BASE[kind]}")
            sm=np.array(ck["scaler_mean"]); ss=np.array(ck["scaler_scale"])
            if sm.shape!=(8,) or ss.shape!=(8,) or not np.isfinite(sm).all() or not np.isfinite(ss).all() or (ss<=0).any(): fails.append(f"{path}: scaler shape/finite/positive fail")
            elif canonical is not None and (float(np.max(np.abs(sm-canonical.mean_)))>1e-10 or float(np.max(np.abs(ss-canonical.scale_)))>1e-10): fails.append(f"{path}: scaler differs from South canonical (tol 1e-10)")
            with torch.no_grad(): _=m(torch.zeros(2,8))  # forward-shape reconstruction check on dummy input, no real data
        except Exception as e: fails.append(f"{path}: {e}")
    for out in [OUT_CSV,OUT_SUM,OUT_LOG]:
        if os.path.exists(out): fails.append(f"output already exists: {out}")
    print(f"NORTH PREFLIGHT: split_sha_ok={h==EXPECTED_SPLIT_SHA} models={len(items)} failures={len(fails)}")
    for f in fails: print(" FAIL:",f)
    if not fails: print("NORTH PREFLIGHT PASSED — no North evaluation performed")
    return fails

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--list",action="store_true"); ap.add_argument("--preflight",action="store_true"); ap.add_argument("--run-once",action="store_true"); a=ap.parse_args()
    items=checkpoint_list()
    if a.list and not a.run_once:
        for kind,path in items: print(kind,path,os.path.exists(path))
        print(f"total {len(items)}; NO North evaluation."); sys.exit(0)
    if a.preflight and not a.run_once: sys.exit(1 if preflight() else 0)
    if not a.run_once: print("Use --list, --preflight, or (after approval) --run-once"); sys.exit(0)
    if preflight(): print("ABORT: preflight failed, North not evaluated"); sys.exit(1)
    # Atomic run-lock: refuse if present; NEVER auto-delete (survives crash/interruption)
    try:
        fd=os.open(LOCK_FILE, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        print(f"REFUSING: run-lock exists ({LOCK_FILE}) — North evaluation must not be rerun."); sys.exit(1)
    import datetime, platform
    RUN_TS=datetime.datetime.now(datetime.timezone.utc).isoformat()
    ENV=f"python {platform.python_version()} | torch {torch.__version__} | {platform.platform()} | CPU sandbox 1 thread"
    with os.fdopen(fd,"w") as lf: lf.write(f"north_run_started_utc={RUN_TS}\nenvironment={ENV}\nscript=src/final_tier2_north_evaluation.py --run-once\n")
    print(f"RUN-LOCK created: {LOCK_FILE} at {RUN_TS}")
    torch.set_num_threads(1)
    X,y,_=load(); sp=json.load(open("data/splits/tier2_split.json"))["indices"]; idx=sp["test_north"]
    rows=[]; log=[f"TIER2 NORTH EVALUATION single run n={len(idx)}"]
    for kind,path in items:
        m,ck=build(path)
        Xs=torch.tensor((X[idx]-np.array(ck["scaler_mean"]))/np.array(ck["scaler_scale"]),dtype=torch.float32)
        with torch.no_grad(): p=m(Xs).numpy().ravel()
        yt=y[idx]; rmse=float(np.sqrt(np.mean((p-yt)**2))); mae=float(np.mean(np.abs(p-yt))); r2=float(1-np.sum((p-yt)**2)/np.sum((yt-yt.mean())**2))
        rows.append({"kind":kind,"method":ck.get("method",kind),"seed":ck.get("seed",""),"budget_target":ck.get("budget_target",""),"checkpoint_sha256":hashlib.sha256(open(path,"rb").read()).hexdigest(),"evaluation_timestamp_utc":RUN_TS,"environment":ENV,
                     "rank_L1":ck.get("compressed_ranks",{}).get("L1",""),"rank_L2":ck.get("compressed_ranks",{}).get("L2",""),
                     "finetuned":ck.get("finetuned",False),"checkpoint":path,"checkpoint_bytes":os.path.getsize(path),"params":count_params(m),
                     "north_RMSE":rmse,"north_MAE":mae,"north_R2":r2})  # full precision
        log.append(f"{path}: RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
    with open(OUT_CSV,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    df={}
    import collections
    for r in rows: df.setdefault((r["kind"],r["method"],str(r["budget_target"])),[]).append(r["north_RMSE"])
    with open(OUT_SUM,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["kind","method","budget_target","n","north_RMSE_mean","north_RMSE_std"])
        for k,v in sorted(df.items()): w.writerow([*k,len(v),float(np.mean(v)),float(np.std(v,ddof=1)) if len(v)>1 else 0.0])
    open(OUT_LOG,"w").write("\n".join(log)); print("\n".join(log)); print(f"Saved {OUT_CSV} — DO NOT RERUN.")
