import torch, torch.nn as nn
def make_mlp(sizes, dropout=0.0):
    layers=[]
    for i in range(len(sizes)-1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes)-2:
            layers.append(nn.ReLU())
            if dropout>0: layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)
def count_params(model): return sum(p.numel() for p in model.parameters())
