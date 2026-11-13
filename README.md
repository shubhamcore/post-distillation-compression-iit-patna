# Post-Distillation Compression of Neural Networks

**Author:** Shubham  
**Roll number:** 2401CS65  
**Department:** Computer Science and Engineering, IIT Patna

This repository accompanies the project **Post-Distillation Compression of Neural Networks Using Sensitivity-Aware Adaptive Low-Rank Decomposition**.

## Project overview

The project studies low-rank SVD compression of a knowledge-distilled neural-network student. It compares uniform, greedy energy-based, and sensitivity-aware dynamic-programming rank allocation under 2x, 4x, and 8x parameter budgets. Both frozen compressed models and fine-tuned compressed models are evaluated using a random split and a South-to-North geographic hold-out.

This is an experimental evaluation, not a claim to invent a new SVD or knowledge-distillation algorithm.

## Repository contents

The project source bundle is being added to this repository. It includes Python scripts for data preparation, model training, SVD compression, rank allocation, fine-tuning, evaluation, and baseline comparisons, along with frozen split definitions and execution notes.

## Environment

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Use the split files supplied with the repository. Do not regenerate them when reproducing the reported evaluation.

## Reproducibility and interpretation

- The reported neural-network summaries use three random seeds.
- The sensitivity-aware DP optimum is exact for its clipped additive sensitivity surrogate, not necessarily for the true joint validation loss.
- Sensitivity-aware allocation is not universally superior to greedy energy allocation.
- Fine-tuning recovers substantial degradation for several compressed configurations.
- Latency and energy consumption were not measured; no speed or energy claims are made.
- The California Housing dataset's formal licence status is disclosed in the accompanying project documentation.

## Citation

Please cite the project as:

Shubham, *Post-Distillation Compression of Neural Networks Using Sensitivity-Aware Adaptive Low-Rank Decomposition*, IIT Patna, 2026.

## Status

The repository README is in place. The source-code bundle and full file tree must be uploaded before this repository should be treated as a complete public code submission.
