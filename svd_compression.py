"""Truncated / factorized SVD for a Linear layer. Established: Eckart-Young."""
import torch

def factorize(W: torch.Tensor, r: int):
    """W (m x n) -> A (m x r), B (r x n) with A@B = best rank-r approx."""
    U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
    assert 1 <= r <= len(S)
    A = U[:, :r] * S[:r].sqrt()          # (m x r) * broadcast over columns
    B = S[:r].sqrt().unsqueeze(1) * Vt[:r, :]  # (r x n)
    return A, B

def reconstruction_error(W, A, B):
    return float(torch.linalg.matrix_norm(W - A @ B))

def theoretical_error(W, r):
    S = torch.linalg.svdvals(W.float())
    return float(torch.sqrt(torch.sum(S[r:]**2))) if r < len(S) else 0.0

def factorized_params(m, n, r, bias=True):
    return int(r*(m+n) + (m if bias else 0))
