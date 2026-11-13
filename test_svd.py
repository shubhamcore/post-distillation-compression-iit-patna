import torch, sys
sys.path.insert(0, "src")
from svd_compression import factorize, reconstruction_error, theoretical_error, factorized_params
torch.manual_seed(0)

def test_shapes_and_error():
    W = torch.randn(64, 128)
    for r in [1, 4, 16, 42]:
        A, B = factorize(W, r)
        assert A.shape == (64, r) and B.shape == (r, 128), (A.shape, B.shape)
        err, theo = reconstruction_error(W, A, B), theoretical_error(W, r)
        assert abs(err - theo) < 1e-3, (r, err, theo)
    print("PASS shapes + reconstruction error == sqrt(tail sigma^2) (Eckart-Young)")

def test_worked_example():
    W = torch.tensor([[4., 0.], [3., -5.]])
    S = torch.linalg.svdvals(W)
    assert abs(S[0].item() - 40**0.5) < 1e-4 and abs(S[1].item() - 10**0.5) < 1e-4
    A, B = factorize(W, 1)
    assert abs(reconstruction_error(W, A, B) - 10**0.5) < 1e-4
    print("PASS worked example sigma=(sqrt40, sqrt10), rank-1 error=sqrt10")

def test_full_rank_exact_and_params():
    W = torch.randn(128, 8)
    A, B = factorize(W, 8)
    assert reconstruction_error(W, A, B) < 1e-4
    assert factorized_params(128, 8, 7) == 1080 and factorized_params(128, 8, 8) == 1216
    assert factorized_params(64, 128, 42) == 8128
    print("PASS full-rank exact; param counts L1 r7=1080 r8=1216, L2 r42=8128")

def test_compressed_forward_differs():
    W = torch.randn(64, 128); x = torch.randn(128)
    A, B = factorize(W, 4)
    assert not torch.allclose(W @ x, A @ (B @ x), atol=1e-5)  # rank-4 must lose something on random W
    print("PASS rank-4 forward differs from dense on random matrix (information loss is real)")

if __name__ == "__main__":
    test_shapes_and_error(); test_worked_example(); test_full_rank_exact_and_params(); test_compressed_forward_differs()
    print("ALL SVD UNIT TESTS PASSED")
