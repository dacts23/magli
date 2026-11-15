"""Penalized least-squares solver via SVD data augmentation."""

from typing import Tuple

import numpy as np


def solve_pirls(
    X: np.ndarray,
    y: np.ndarray,
    P: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve penalized least squares: (X^T X + P) beta = X^T y."""
    n, m = X.shape
    eps = np.finfo(float).eps
    # Eigendecompose symmetric P instead of Cholesky so huge lam cannot fail on tiny negative roundoff.
    P_sym = 0.5 * (P + P.T)
    eigvals, eigvecs = np.linalg.eigh(P_sym)
    E = np.sqrt(np.maximum(eigvals, 0.0) + np.sqrt(eps))[:, None] * eigvecs.T
    Q, R = np.linalg.qr(X)
    min_nm = min(m, n)
    U, d, Vt = np.linalg.svd(np.vstack([R, E]), full_matrices=False)
    U1 = U[:min_nm, :]
    B_solve = (Vt.T / d) @ U1.T @ Q.T
    return B_solve @ y, U1, B_solve
