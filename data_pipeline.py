"""Dataset pipeline: load California Housing, freeze splits. NO training here.
Splits (Tier-1): train 60% / val_es 10% / sensitivity 10% / test 20% (disjoint).
Tier-2: North (lat>38) = test only. South split 70/10/10/10 (train/val_es/sens/south_holdout_unused_for_tier2_test).
Sensitivity subset is frozen BEFORE any sensitivity score exists and is never used for early stopping or test.
"""
import json, hashlib, numpy as np
from sklearn.datasets import fetch_california_housing

SPLIT_SEED = 42  # split-generation seed, independent of training seeds {0,1,2}

def load():
    d = fetch_california_housing(as_frame=True)
    return d.data.values.astype(np.float64), d.target.values.astype(np.float64), list(d.data.columns)

def _idx(rng, n, fracs):
    perm = rng.permutation(n); out=[]; start=0
    for i,f in enumerate(fracs):
        k = int(round(f*n)) if i < len(fracs)-1 else n-start
        out.append(sorted(perm[start:start+k].tolist())); start+=k
    return out

def make_splits(X):
    rng = np.random.default_rng(SPLIT_SEED); n=len(X)
    tr, ves, sens, te = _idx(rng, n, [0.60,0.10,0.10,0.20])
    tier1 = {"train":tr,"val_es":ves,"sensitivity":sens,"test":te}
    lat = X[:,6]; north = np.where(lat>38.0)[0]; south = np.where(lat<=38.0)[0]
    rng2 = np.random.default_rng(SPLIT_SEED)
    s_tr, s_ves, s_sens, s_hold = _idx(rng2, len(south), [0.70,0.10,0.10,0.10])
    tier2 = {"train": sorted(south[s_tr].tolist()), "val_es": sorted(south[s_ves].tolist()),
             "sensitivity": sorted(south[s_sens].tolist()), "south_holdout_reserve": sorted(south[s_hold].tolist()),
             "test_north": sorted(north.tolist())}
    return tier1, tier2

if __name__ == "__main__":
    X,y,cols = load(); t1,t2 = make_splits(X)
    for name,t in [("tier1",t1),("tier2",t2)]:
        # disjointness check
        allidx=[i for v in t.values() for i in v]
        assert len(allidx)==len(set(allidx)), f"{name} overlap!"
        print(name, {k:len(v) for k,v in t.items()}, "disjoint OK, total covered", len(allidx))
    meta={"split_seed":SPLIT_SEED,"features":cols,"n":len(X),
          "tier1":{k:len(v) for k,v in t1.items()},"tier2":{k:len(v) for k,v in t2.items()},
          "sensitivity_subset_rule":"Frozen in this file before any model/sensitivity computation; used ONLY for S_l(r); never for early stopping (val_es) or final evaluation (test).",
          "candidate_config_log":"See configs/candidate_configurations.json"}
    open("data/splits/tier1_split.json","w").write(json.dumps({"indices":t1,"meta":meta},indent=1))
    open("data/splits/tier2_split.json","w").write(json.dumps({"indices":t2,"meta":meta},indent=1))
    for f in ["tier1_split.json","tier2_split.json"]:
        h=hashlib.sha256(open(f"data/splits/{f}","rb").read()).hexdigest()
        print(f, "sha256", h)
    json.dump(meta, open("data/splits/split_meta.json","w"), indent=1)
