import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, json, sys
sys.path.insert(0,"src")
from data_pipeline import load
X,y,cols=load()
fig,ax=plt.subplots(1,3,figsize=(12,3.5))
ax[0].hist(y,bins=50); ax[0].set_xlabel("MedHouseVal ($100k)"); ax[0].set_title("Target (note spike at 5.0 cap)")
ax[1].scatter(X[:,7],X[:,6],c=y,s=1); ax[1].set_xlabel("Longitude"); ax[1].set_ylabel("Latitude"); ax[1].set_title("Geography coloured by price")
ax[2].boxplot([X[:,i] for i in range(8)],tick_labels=cols); ax[2].tick_params(labelrotation=45); ax[2].set_title("Feature scales (raw)")
plt.tight_layout(); plt.savefig("results/figures/eda_overview.png",dpi=120)
stats={c:{"min":float(X[i].min()) if False else float(X[:,j].min()),"max":float(X[:,j].max()),"mean":float(X[:,j].mean()),"std":float(X[:,j].std())} for j,c in enumerate(cols)}
stats["target"]={"min":float(y.min()),"max":float(y.max()),"frac_at_cap_5.0":float((y>=5.0).mean())}
json.dump(stats,open("results/tables/eda_stats.json","w"),indent=1)
print("EDA saved. frac at cap:",stats["target"]["frac_at_cap_5.0"],"| corr matrix max offdiag:", end=" ")
C=np.corrcoef(X,rowvar=False); print(round(float(np.abs(C-np.eye(8)).max()),3))
