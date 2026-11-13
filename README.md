# Post-Distillation Compression of Neural Networks

**Author:** Shubham  
**Roll number:** 2401CS65  
**Department:** Computer Science and Engineering, IIT Patna

This repository accompanies the project **Post-Distillation Compression of Neural Networks Using Sensitivity-Aware Adaptive Low-Rank Decomposition**.

## Project overview

The project studies low-rank SVD compression of a knowledge-distilled neural-network student. It compares uniform, greedy energy-based, and sensitivity-aware dynamic-programming rank allocation under 2x, 4x, and 8x parameter budgets. Both frozen compressed models and fine-tuned compressed models are evaluated using a random split and a South-to-North geographic hold-out.

This is an experimental evaluation, not a claim to invent a new SVD or knowledge-distillation algorithm.

## Source bundle

The prepared source-code archive is provided separately as **Shubham-Post-Distillation-Compression-Repository.zip**. The archive contains the Python scripts, dependency list, split files, and execution notes. The source files have **not yet been committed to this GitHub repository**; add them before submitting this repository as the public code link.

## Local setup

After extracting the source archive, open a terminal in its project folder and run:

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

Use the supplied frozen split files. Do not regenerate them when reproducing the reported evaluation.

## Reproducibility and interpretation

- The reported neural-network summaries use three random seeds.
- The sensitivity-aware DP optimum is exact for its clipped additive sensitivity surrogate, not necessarily for the true joint validation loss.
- Sensitivity-aware allocation is not universally superior to greedy energy allocation.
- Fine-tuning recovers substantial degradation for several compressed configurations.
- Latency and energy consumption were not measured; no speed or energy claims are made.
- The California Housing dataset's formal licence status is disclosed in the accompanying project documentation.

## Citation

Shubham, *Post-Distillation Compression of Neural Networks Using Sensitivity-Aware Adaptive Low-Rank Decomposition*, IIT Patna, 2026.

## Current repository status

The public repository and this README have been created. The source-code archive must still be pushed to the repository before it is complete.
