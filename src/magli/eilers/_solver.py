"""Penalized least-squares solver via SVD data augmentation."""

from typing import Tuple

import numpy as np


def _pirls_factors(
    R: np.ndarray,
    P: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (U1, Vt_d @ U1.T, Vt_d @ Vt_d.T) for precomputed QR factor R and penalty P."""
    eps = np.finfo(float).eps
    # Eigendecompose symmetric P instead of Cholesky so huge lam cannot fail on tiny negative roundoff.
    P_sym = 0.5 * (P + P.T)
    eigvals, eigvecs = np.linalg.eigh(P_sym)
    E = np.sqrt(np.maximum(eigvals, 0.0) + np.sqrt(eps))[:, None] * eigvecs.T
    min_nm = R.shape[0]
    U, d, Vt = np.linalg.svd(np.vstack([R, E]), full_matrices=False)
    U1 = U[:min_nm, :]
    Vt_d = Vt.T / d
    return U1, Vt_d @ U1.T, Vt_d @ Vt_d.T


def solve_pirls(
    X: np.ndarray,
    y: np.ndarray,
    P: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve penalized least squares: (X^T X + P) beta = X^T y."""
    Q, R = np.linalg.qr(X)
    U1, B_q, _ = _pirls_factors(R, P)
    B_solve = B_q @ Q.T
    return B_q @ (Q.T @ y), U1, B_solve
