"""Wood (2013) approximate smooth-term significance test, a port of mgcv's ``testStat``."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import integrate, stats


def _psum_chisq(x: float, lam: np.ndarray, df: np.ndarray) -> float:
    """P(sum_j lam_j * chisq_{df_j} > x) via Imhof's inversion (mgcv uses Davies, same quantity)."""
    lam, df = np.asarray(lam, float), np.asarray(df, float)

    def integrand(u: float) -> float:
        theta = 0.5 * np.sum(df * np.arctan(lam * u)) - 0.5 * x * u
        inv_rho = np.exp(-0.25 * np.sum(df * np.log1p((lam * u) ** 2)))
        return np.sin(theta) * inv_rho / u if u > 0 else 0.5 * (np.sum(df * lam) - x)

    val, _ = integrate.quad(integrand, 0.0, np.inf, limit=1000, epsabs=1e-12, epsrel=1e-10)
    return float(np.clip(0.5 + val / np.pi, 0.0, 1.0))


def wood_test_stat(
    p: np.ndarray, X: np.ndarray, V: np.ndarray, rank: float, res_df: float = -1.0
) -> Tuple[float, float, float]:
    """Return (stat, pval, rank) for H0: smooth term == 0, following mgcv::testStat(type=0).

    ``X`` is the (weighted) design block of the term, ``V`` its Bayesian covariance block,
    ``rank`` the (fractional) reference df (edf1), ``res_df`` the residual df (<= 0 if the scale is known).
    """
    R = np.linalg.qr(X, mode="r")
    V = R @ V @ R.T
    V = 0.5 * (V + V.T)
    ev, U = np.linalg.eigh(V)
    ev, U = ev[::-1], U[:, ::-1]
    siv = np.where(U[0] == 0, 1.0, np.sign(U[0]))
    U = U * siv

    k = max(0, int(np.floor(rank)))
    nu = abs(rank - k)
    k1 = k + 1 if nu > 0 else k
    r_est = int(np.sum(ev > ev[0] * np.finfo(float).eps ** 0.9))
    if r_est < k1:
        k1 = k = r_est
        nu, rank = 0.0, float(r_est)

    vec = U[:, :k1] if k1 < U.shape[1] else U
    if nu > 0 and k > 0:
        if k > 1:
            vec[:, : k - 1] = vec[:, : k - 1] / np.sqrt(ev[: k - 1])
        b12 = np.sqrt(max(0.5 * nu * (1 - nu), 0.0))
        B = np.array([[1.0, b12], [b12, nu]])
        D = np.diag(ev[k - 1 : k1] ** -0.5)
        B = D @ B @ D
        w, Ev = np.linalg.eigh(B)
        rB = Ev @ np.diag(np.sqrt(np.maximum(w, 0.0))) @ Ev.T
        vec1 = vec.copy()
        vec1[:, k - 1 : k1] = vec[:, k - 1 : k1] @ (rB @ np.diag([-1.0, 1.0])).T
        vec[:, k - 1 : k1] = vec[:, k - 1 : k1] @ rB.T
    else:
        vec = vec / np.sqrt(ev[0] if k == 0 else ev[:k])
        vec1 = vec
        if k == 1:
            rank = 1.0

    Rp = R @ p
    d = float(np.sum((vec.T @ Rp) ** 2))
    d1 = float(np.sum((vec1.T @ Rp) ** 2))

    pval = 2.0
    rank1 = rank
    if nu > 0:
        if k1 == 1:
            rank1 = 1.0
            val = np.array([1.0])
        else:
            val = np.ones(k1)
            rp = nu + 1
            val[k - 1] = (rp + np.sqrt(rp * (2 - rp))) / 2
            val[k1 - 1] = rp - val[k - 1]
        if res_df <= 0:
            pval = 0.5 * (_psum_chisq(d, val, np.ones(k1)) + _psum_chisq(d1, val, np.ones(k1)))
        else:
            k0 = max(1, int(round(res_df)))
            lam_df = np.append(np.ones(len(val)), k0)
            pval = 0.5 * sum(
                _psum_chisq(0.0, np.append(val, -dd / k0), lam_df) for dd in (d, d1)
            )
    if pval > 1:
        if res_df <= 0:
            pval = 0.5 * (stats.chi2.sf(d, rank1) + stats.chi2.sf(d1, rank1))
        else:
            pval = 0.5 * (stats.f.sf(d / rank1, rank1, res_df) + stats.f.sf(d1 / rank1, rank1, res_df))
    return d, min(1.0, float(pval)), float(rank)
