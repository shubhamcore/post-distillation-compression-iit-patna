# Post-Distillation Compression of Neural Networks

**Author:** Shubham  
**Roll number:** 2401CS65  
**Department:** Computer Science and Engineering, IIT Patna

This repository contains the source code and selected reproducibility files for **Post-Distillation Compression of Neural Networks Using Sensitivity-Aware Adaptive Low-Rank Decomposition**.

## Project overview

The project studies truncated-SVD compression of a knowledge-distilled neural-network student. It compares uniform, greedy energy-based, and sensitivity-aware dynamic-programming rank allocation under 2x, 4x, and 8x parameter budgets. Frozen and fine-tuned compressed models are evaluated using a random split and a South-to-North geographic hold-out.

This is an experimental evaluation, not a claim to invent a new SVD or knowledge-distillation algorithm.

## Repository contents

- Python source files for data preparation, EDA, teacher/student training, SVD compression, rank allocation, fine-tuning, evaluation, and traditional ML baselines.
- `requirements.txt` with the recorded project environment dependencies.
- `data/splits/tier1_split.json` and `data/splits/tier2_split.json` with frozen split indices.
- `data/splits/split_meta.json` with split metadata.
- `.gitignore` to exclude virtual environments, caches, and generated model artefacts.

Large checkpoints, full raw-result archives, and some run logs are distributed separately where required; this repository does not claim to contain every experiment artefact.

## Local setup

Recommended: use a clean virtual environment. The dependency pins reflect the recorded experiment environment.

**Windows PowerShell**
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS/Linux**
```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the exact pinned PyTorch CPU wheel is unavailable for your platform, consult the official PyTorch installation selector and use a compatible CPU build; document any environment change before attempting reproduction.

## Running the project

Run scripts from the repository root so that relative paths resolve correctly. Examples:

```bash
python eda.py
python data_pipeline.py
python linear_regression_baseline.py
python random_forest_baseline.py
python support_vector_regression_baseline.py
python gaussian_process_baseline.py
```

Training, compression, fine-tuning, and final evaluation scripts are also included. Check each script's arguments and prerequisites before running it; full neural-network reproduction may require the separately distributed checkpoint and result artefacts.

Use the supplied frozen split files. Do not regenerate them when reproducing the reported evaluation.

## Reproducibility and interpretation

- Reported neural-network summaries use three random seeds.
- The sensitivity-aware DP optimum is exact for its clipped additive sensitivity surrogate, not necessarily for the true joint validation loss.
- Sensitivity-aware allocation is not universally superior to greedy energy allocation.
- Fine-tuning recovers substantial degradation for several compressed configurations.
- Latency and energy consumption were not measured; no speed or energy claims are made.
- The California Housing dataset's formal licence status is disclosed in the project documentation.

## Citation

Shubham, *Post-Distillation Compression of Neural Networks Using Sensitivity-Aware Adaptive Low-Rank Decomposition*, IIT Patna, 2026.

## Repository status

The source files and frozen split definitions are committed on the `main` branch. This public repository is the code location for the project. Large checkpoints and any companion evidence archives should be obtained from the accompanying submission package when needed.
