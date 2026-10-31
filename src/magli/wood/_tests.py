"""Wood (2013) approximate smooth-term significance test, a port of mgcv's ``testStat``."""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
from scipy import integrate, stats


def _liu2(x: float, lam: np.ndarray, df: np.ndarray) -> float:
    """Liu-Tang-Zhang (2009) moment-matching fallback used by mgcv::psum.chisq."""
    lh = lam * df
    mu_q = float(np.sum(lh))
    lh = lh * lam
    c2 = float(np.sum(lh))
    lh = lh * lam
    c3 = float(np.sum(lh))
    if c2 <= 0.0:
        return 1.0 if x <= mu_q else 0.0
    s1, s2 = c3 / c2**1.5, float(np.sum(lh * lam)) / c2**2
    t = (x - mu_q) / np.sqrt(2.0 * c2)
    if s1**2 > s2:
        a = 1.0 / (s1 - np.sqrt(s1**2 - s2))
        delta = s1 * a**3 - a**2
        l = a**2 - 2.0 * delta
    else:
        if c3 == 0.0:
            return float(stats.norm.sf(t))
        a, delta, l = 1.0 / abs(s1), 0.0, c2**3 / c3**2
    q = (t if s1 >= 0 else -t) * np.sqrt(2.0) * a + l + delta
    sf = stats.ncx2.sf(q, l, delta) if delta > 0 else stats.chi2.sf(q, l)
    return float(sf if s1 >= 0 else 1.0 - sf)


def _psum_chisq(x: float, lam: np.ndarray, df: np.ndarray) -> float:
    """P(sum_j lam_j * chisq_{df_j} > x) via Imhof's inversion (mgcv uses Davies, same quantity)."""
    lam, df = np.asarray(lam, float), np.asarray(df, float)
    scale = float(np.sum(df * np.abs(lam)) + abs(x))
    if scale == 0.0:
        return 1.0
    lam, x = lam / scale, x / scale

    def integrand(u: float) -> float:
        theta = 0.5 * np.sum(df * np.arctan(lam * u)) - 0.5 * x * u
        inv_rho = np.exp(-0.25 * np.sum(df * np.log1p((lam * u) ** 2)))
        return np.sin(theta) * inv_rho / u if u > 0 else 0.5 * (np.sum(df * lam) - x)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        val, err = integrate.quad(integrand, 0.0, np.inf, limit=1000, epsabs=1e-12, epsrel=1e-10)
    if err > 1e-4:
        return _liu2(x, lam, df)
    return float(np.clip(0.5 + val / np.pi, 0.0, 1.0))


def _mroot_t(A: np.ndarray) -> np.ndarray:
    ev, U = np.linalg.eigh(0.5 * (A + A.T))
    if len(ev) == 0 or ev[-1] <= 0.0:
        return np.zeros((0, A.shape[0]))
    pos = ev > ev[-1] * np.finfo(float).eps
    return np.sqrt(ev[pos])[:, None] * U[:, pos].T


def wood_re_test(
    coef: np.ndarray,
    R: np.ndarray,
    S_total: np.ndarray,
    Vp: np.ndarray,
    Ve: np.ndarray,
    scale: float,
    start: int,
    end: int,
    res_df: float = -1.0,
    other_re_mask: np.ndarray | None = None,
) -> Tuple[float, float, float]:
    """Return (stat, pval, rank) for zero-null-space smooths (re, ts, cs, cc, cp), following mgcv::reTest."""
    p, q = len(coef), end - start
    if other_re_mask is None or not np.any(other_re_mask):
        cols = [j for j in range(p) if j < start or j >= end] + list(range(start, end))
        LRB = np.vstack([R, _mroot_t(S_total)])[:, cols]
        Rm = np.linalg.qr(LRB, mode="r")[-q:, -q:]
        Ve_m = 0.5 * (Ve[start:end, start:end] + Ve[start:end, start:end].T)
    else:
        keep = ~other_re_mask
        R1, R2 = R[:, keep], R[:, other_re_mask]
        S1 = S_total[np.ix_(keep, keep)]
        S2 = S_total[np.ix_(other_re_mask, other_re_mask)]
        ev2, U2 = np.linalg.eigh(0.5 * (S2 + S2.T))
        pos2 = ev2 > ev2[-1] * np.finfo(float).eps ** 0.8
        S2_half = np.where(pos2, 1.0 / np.sqrt(np.maximum(ev2, 1e-300)), 0.0)[:, None] * U2.T
        A = S2_half @ R2.T
        L = np.linalg.cholesky(np.eye(R.shape[0]) + A.T @ A).T
        tk = np.zeros(p, dtype=bool)
        tk[start:end] = True
        tk = tk[keep]
        cols = list(np.where(~tk)[0]) + list(np.where(tk)[0])
        LRB = np.vstack([L @ R1, _mroot_t(S1)])[:, cols]
        Rm = np.linalg.qr(LRB, mode="r")[-q:, -q:]
        LRV = L @ (R @ Vp[:, start:end])
        Ve_m = (LRV.T @ LRV) / scale

    d = Rm @ coef[start:end]
    stat = float(np.sum(d**2) / scale)
    M = (Rm @ Ve_m @ Rm.T) / scale
    ev = np.maximum(np.linalg.eigvalsh(0.5 * (M + M.T))[::-1], 0.0)
    rank = int(np.sum(ev > ev[0] * np.finfo(float).eps ** 0.8)) if len(ev) and ev[0] > 0 else q
    if res_df > 0:
        k0 = max(1, int(round(res_df)))
        pval = _psum_chisq(0.0, np.append(ev, -stat / k0), np.append(np.ones(len(ev)), k0))
    else:
        pval = _psum_chisq(stat, ev, np.ones(len(ev)))
    return stat, min(1.0, float(pval)), float(rank)


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
