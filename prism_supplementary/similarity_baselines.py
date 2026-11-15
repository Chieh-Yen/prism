"""Linear CKA and SVCCA between two feature matrices.

Used by ``train_forgetting.py`` when ``PRISM_EVAL_SIMILARITY=1``: at every
checkpoint, each PRISM evaluation record gets ``cka`` and ``svcca`` computed on
the same features the PRISM terms use (X = base-model features Z_T, Y =
fine-tuned features Z_P; rows are answer-token positions, columns are the
final-layer hidden dimensions).  The functions are the same as in
``scripts/rounds/similarity_scores.py`` (quantization cells), so both
kinds of cells use one estimator.  Both scores are similarities; rankings use
1 - score.

Exactly what is computed (inputs have shape (n, d)):

``linear_cka(X, Y)``
    Linear CKA in feature space (Kornblith et al., 2019).  Each column of X
    and of Y is centred over the n rows, then

        CKA = ||X^T Y||_F^2 / ( ||X^T X||_F * ||Y^T Y||_F ),

    with the denominator clamped at 1e-12.

``svcca(X, Y)``
    SVCCA (Raghu et al., 2017).  X and Y are centred column-wise.  For each
    matrix separately, the leading principal directions are kept: the
    smallest k whose cumulative explained variance reaches 99%, restricted
    to 2 <= k <= 256 (and to the number of available directions).  The two
    orthonormal bases Q_X (n, k_X) and Q_Y (n, k_Y) of the kept principal
    subspaces are compared by the singular values of Q_X^T Q_Y, which are the
    canonical correlations between the two subspaces; the score is their
    unweighted mean over all min(k_X, k_Y) of them.

    For n > 32768 the basis is taken from the eigendecomposition of X^T X
    instead of a thin SVD; both span the same subspace.
"""
import torch


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xty = (X.T @ Y)
    num = (xty ** 2).sum()
    den = torch.linalg.matrix_norm(X.T @ X) * torch.linalg.matrix_norm(Y.T @ Y)
    return (num / den.clamp(min=1e-12)).item()


def _pca_basis(X: torch.Tensor, ev_keep: float = 0.99, cap: int = 256):
    X = X - X.mean(0, keepdim=True)
    n, d = X.shape
    if n > 32768:
        C = X.T @ X
        evals, evecs = torch.linalg.eigh(C)
        evals = evals.flip(0).clamp(min=0)
        evecs = evecs.flip(1)
        ev = evals / evals.sum().clamp(min=1e-12)
        k = int(torch.searchsorted(ev.cumsum(0), torch.tensor(ev_keep, device=ev.device)).item()) + 1
        k = max(2, min(k, cap, d))
        S = evals[:k].sqrt().clamp(min=1e-12)
        return (X @ evecs[:, :k]) / S
    U, S, _ = torch.linalg.svd(X, full_matrices=False)
    ev = (S ** 2) / (S ** 2).sum().clamp(min=1e-12)
    k = int(torch.searchsorted(ev.cumsum(0), torch.tensor(ev_keep, device=S.device)).item()) + 1
    k = max(2, min(k, cap, S.shape[0]))
    return U[:, :k]


def svcca(X: torch.Tensor, Y: torch.Tensor) -> float:
    Qx, Qy = _pca_basis(X), _pca_basis(Y)
    rho = torch.linalg.svdvals(Qx.T @ Qy)
    return rho.mean().item()
