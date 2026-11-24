"""Penalized least squares solvers and smoothing parameter optimization for Wood (2011) GAM backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import linalg, optimize

from .._base import _fit_summary_stats
from ._tests import wood_re_test, wood_test_stat

def _edf2_per_coef(
    hess, H_inv, coef, sp, free, bounds, subs, RtR, Vp, scale, df_reml, S_total
) -> np.ndarray:
    """mgcv's edf2: Vc = Vp + J V_rho J' + scale * Vb.corr, using the joint (rho, log phi) REML Hessian."""
    p, m = len(coef), len(free)
    S_full, b = [], np.zeros(m)
    J = np.zeros((p, m))
    for k, idx in enumerate(free):
        s0, s1 = bounds[idx]
        S = np.zeros((p, p))
        S[s0:s1, s0:s1] = sp[idx] * subs[idx]  # lambda_k * S_k
        S_full.append(S)
        J[:, k] = -H_inv @ (S @ coef)
        b[k] = -float(coef @ S @ coef) / (2.0 * scale)
    c = df_reml / 2.0
    H5 = np.block([[hess + np.outer(b, b) / c, b[:, None]], [b[None, :], np.array([[c]])]])
    d, U = np.linalg.eigh(H5)
    pos = d > 0
    V_rho = ((U[:m, pos] / d[pos]) @ U[:m, pos].T)
    Vr = (U[:m] / (np.where(pos, d, 0.0) + 0.1)) @ U[:m].T

    H_c = RtR + S_total
    try:
        Rc = np.linalg.cholesky(H_c).T
    except np.linalg.LinAlgError:
        max_d = float(np.max(np.diag(H_c))) if p > 0 else 0.0
        jitter = (max_d * 1e-12) if max_d > 0.0 else 1e-12
        Rc = np.linalg.cholesky(H_c + jitter * np.eye(p)).T
    Ri = np.linalg.inv(Rc)
    dRi = []
    for S in S_full:
        A = Ri.T @ S @ Ri
        dRc = (np.triu(A) - 0.5 * np.diag(np.diag(A))) @ Rc
        dRi.append(-Ri @ dRc @ Ri)
    Vb_corr = sum(Vr[i, j] * dRi[i] @ dRi[j].T for i in range(m) for j in range(m))
    Vc = Vp + J @ V_rho @ J.T + scale * Vb_corr
    return np.einsum("ij,ji->i", Vc, RtR) / scale


def _setup_penalties(
    S_list: List[np.ndarray],
    sp_fixed: Optional[List[Optional[float]]],
    RtR: np.ndarray,
    p: int,
) -> Tuple[
    List[float],
    List[int],
    List[Tuple[int, int]],
    List[np.ndarray],
    List[int],
    List[float],
    List[bool],
    List[Tuple[List[int], int, Dict[int, np.ndarray]]],
    int,
]:
    """Extract penalty block bounds, ranks, overlap groups, and null-space dimension M_p."""
    n_pens = len(S_list)
    sp_vals = [1.0] * n_pens
    free_indices = []
    if sp_fixed is not None and len(sp_fixed) == n_pens:
        for idx, val in enumerate(sp_fixed):
            if val is not None:
                sp_vals[idx] = float(val)
            else:
                free_indices.append(idx)
    else:
        free_indices = list(range(n_pens))

    pen_bounds: List[Tuple[int, int]] = []
    pen_sub: List[np.ndarray] = []
    rank_S_list: List[int] = []
    log_det_const: List[float] = []
    supports: List[np.ndarray] = []
    for idx, S in enumerate(S_list):
        nz = np.where(np.diag(S) > 0)[0]
        s0, s1 = (int(nz[0]), int(nz[-1]) + 1) if len(nz) > 0 else (0, 0)
        pen_bounds.append((s0, s1))
        sub = S[s0:s1, s0:s1]
        pen_sub.append(sub)
        supports.append(nz)
        if s1 > s0:
            ev = linalg.eigvalsh(sub, check_finite=False)
            pos = ev[ev > max(float(ev[-1]) * 1e-10, 1e-14)]
            rank_S_list.append(len(pos))
            log_det_const.append(float(np.sum(np.log(pos))) if len(pos) > 0 else 0.0)
        else:
            rank_S_list.append(0)
            log_det_const.append(0.0)
        if idx not in free_indices and sp_vals[idx] == 0.0 and s1 > s0:
            sp_vals[idx] = float(
                np.trace(RtR[s0:s1, s0:s1])
                / max(float(np.linalg.norm(sub, "fro")), 1e-300)
                * np.finfo(float).eps
                * 0.1
            )

    overlapping = [False] * n_pens
    for i in range(n_pens):
        for j in range(i + 1, n_pens):
            if (
                pen_bounds[i][0] < pen_bounds[j][1]
                and pen_bounds[j][0] < pen_bounds[i][1]
                and np.intersect1d(supports[i], supports[j]).size > 0
            ):
                overlapping[i] = overlapping[j] = True

    overlap_groups: List[Tuple[List[int], int, Dict[int, np.ndarray]]] = []
    visited = [False] * n_pens
    for i in range(n_pens):
        if overlapping[i] and not visited[i]:
            grp = [i]
            visited[i] = True
            changed = True
            while changed:
                changed = False
                for j in range(n_pens):
                    if overlapping[j] and not visited[j] and any(
                        np.intersect1d(supports[g], supports[j]).size > 0 for g in grp
                    ):
                        grp.append(j)
                        visited[j] = True
                        changed = True
            g_s0 = min(pen_bounds[g][0] for g in grp)
            g_s1 = max(pen_bounds[g][1] for g in grp)
            S_sum = np.zeros((g_s1 - g_s0, g_s1 - g_s0), dtype=float)
            for g_i in grp:
                s0, s1 = pen_bounds[g_i]
                S_sum[s0 - g_s0 : s1 - g_s0, s0 - g_s0 : s1 - g_s0] += pen_sub[g_i]
            ev0, U0 = linalg.eigh(S_sum, check_finite=False)
            U_pos = U0[:, ev0 > max(float(ev0[-1]) * 1e-10, 1e-14)]
            proj_subs = {
                g_i: U_pos[pen_bounds[g_i][0] - g_s0 : pen_bounds[g_i][1] - g_s0, :].T
                @ pen_sub[g_i]
                @ U_pos[pen_bounds[g_i][0] - g_s0 : pen_bounds[g_i][1] - g_s0, :]
                for g_i in grp
            }
            overlap_groups.append((grp, U_pos.shape[1], proj_subs))

    total_pen_rank = sum(
        rank_S_list[i] for i in range(n_pens) if not overlapping[i]
    ) + sum(r_dim for _, r_dim, _ in overlap_groups)
    M_p = max(p - total_pen_rank, 0)
    return sp_vals, free_indices, pen_bounds, pen_sub, rank_S_list, log_det_const, overlapping, overlap_groups, M_p


def _init_rho(
    RtR: np.ndarray,
    pen_bounds: List[Tuple[int, int]],
    pen_sub: List[np.ndarray],
    n_pens: int,
    free_indices: List[int],
    p: int,
) -> np.ndarray:
    """Compute initial log smoothing parameters matching mgcv's diagonal heuristic."""
    ldxx = np.diag(RtR)
    ldss = np.zeros(p)
    pen_mask = np.zeros(p, dtype=bool)
    def_sp = np.ones(n_pens)
    eps8 = np.finfo(float).eps ** 0.8
    for i, (s0, s1) in enumerate(pen_bounds):
        sub = pen_sub[i]
        abs_S = np.abs(sub)
        ma_s = float(np.max(abs_S))
        if ma_s <= 0.0:
            continue
        thresh = eps8 * ma_s
        ind = (
            (np.mean(abs_S, axis=1) > thresh)
            & (np.mean(abs_S, axis=0) > thresh)
            & (np.diag(abs_S) > thresh)
        )
        if np.any(ind):
            size_xx = float(np.mean(ldxx[s0:s1][ind]))
            size_s = float(np.mean(np.diag(sub)[ind]))
            if size_s > 0.0 and size_xx > 0.0:
                def_sp[i] = size_xx / size_s
                ldss[s0:s1] += def_sp[i] * np.diag(sub)
                pen_mask[s0:s1] |= ind
    ind_all = (ldss > 0.0) & pen_mask & (ldxx > 0.0)
    if np.any(ind_all):
        xx_a, ss_a = ldxx[ind_all], ldss[ind_all]
        while float(np.mean(xx_a / (xx_a + ss_a))) > 0.4:
            def_sp *= 10.0
            ss_a *= 10.0
        while float(np.mean(xx_a / (xx_a + ss_a))) < 0.4:
            def_sp /= 10.0
            ss_a /= 10.0
    return np.clip(np.log(def_sp[free_indices]), -15.0, 25.0)


def _compute_term_tests(
    coef: np.ndarray,
    X: np.ndarray,
    R: np.ndarray,
    S_final: np.ndarray,
    Vp: np.ndarray,
    Ve: np.ndarray,
    scale_val: float,
    coef_slices: List[Tuple[int, int]],
    pen_bounds: List[Tuple[int, int]],
    edf_per_coef: np.ndarray,
    edf1_per_coef: np.ndarray,
    df2: float,
    S_unit: Optional[np.ndarray] = None,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Compute Wood (2013) per-term EDF, F-statistics, p-values, and reference df."""
    p = len(coef)
    S_rank = S_final if S_unit is None else S_unit
    edf_per_term: List[float] = []
    f_stat_per_term: List[float] = []
    p_values_per_term: List[float] = []
    ref_df_per_term: List[float] = []
    is_re = [
        s1 > s0
        and sum(1 for b0, b1 in pen_bounds if b0 >= s0 and b1 <= s1 and b1 > b0) == 1
        and S_final[s0, s0] > 0
        and np.allclose(S_final[s0:s1, s0:s1], S_final[s0, s0] * np.eye(s1 - s0))
        for s0, s1 in coef_slices
    ]
    for ti, (start, end) in enumerate(coef_slices):
        q = end - start
        edf_per_term.append(float(np.sum(edf_per_coef[start:end])))
        S_blk = S_rank[start:end, start:end]
        ev_blk = linalg.eigvalsh(S_blk, check_finite=False) if q > 0 else np.zeros(0)
        pen_rank = int(np.sum(ev_blk > max(float(ev_blk[-1]) * 1e-10, 1e-14))) if q > 0 else 0
        try:
            if pen_rank == q and q > 0:
                other_re = np.zeros(p, dtype=bool)
                for rj, (rs0, rs1) in enumerate(coef_slices):
                    if rj != ti and is_re[rj]:
                        other_re[rs0:rs1] = True
                stat, pval, rank = wood_re_test(
                    coef, R, S_final, Vp, Ve, scale_val, start, end, df2, other_re
                )
            else:
                rank = float(q) if pen_rank == 0 else min(q, float(np.sum(edf1_per_coef[start:end])))
                if rank <= 0.0:
                    stat, pval, rank = 0.0, 1.0, float(q)
                else:
                    stat, pval, rank = wood_test_stat(
                        coef[start:end], X[:, start:end],
                        Vp[start:end, start:end], rank, df2,
                    )
            stat = stat / rank if rank > 0.0 else 0.0
        except np.linalg.LinAlgError:
            stat, pval, rank = 0.0, 1.0, float(q)
        f_stat_per_term.append(stat)
        p_values_per_term.append(pval)
        ref_df_per_term.append(rank)
    return edf_per_term, f_stat_per_term, p_values_per_term, ref_df_per_term


def fit_wood_engine(
    X: np.ndarray,
    y: np.ndarray,
    S_list: List[np.ndarray],
    sp_fixed: Optional[List[Optional[float]]] = None,
    method: str = "REML",
    gamma: float = 1.0,
    coef_slices: Optional[List[Tuple[int, int]]] = None,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Fit GAM using Wood (2011) REML or GCV criteria and compute full statistics."""
    n, p = X.shape
    method = method.upper()
    if method not in ("REML", "GCV.CP", "GCV"):
        raise ValueError(f"Unknown method '{method}'. Choose 'REML' or 'GCV.Cp'.")
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be finite and positive.")

    y_raw = y
    if weights is not None:
        n_obs = int(np.sum(weights > 0))
        log_w_sum = float(np.sum(np.log(weights[weights > 0])))
        w_sqrt = np.sqrt(weights)
        X_w = X * w_sqrt[:, None]
        y_w = y * w_sqrt
    else:
        n_obs = n
        log_w_sum = 0.0
        X_w = X
        y_w = y

    if n > p:
        R_aug = linalg.qr(
            np.column_stack([X_w, y_w]), mode="r", check_finite=False
        )[0]
        R = R_aug[:p, :p]
        qty = R_aug[:p, p]
        ss_orth = float(R_aug[p, p] ** 2)
    else:
        Q, R = np.linalg.qr(X_w)
        qty = Q.T @ y_w
        ss_orth = max(0.0, float(y_w @ y_w) - float(qty @ qty))
    RtR = R.T @ R
    Xty = R.T @ qty

    n_pens = len(S_list)
    (
        sp_vals,
        free_indices,
        pen_bounds,
        pen_sub,
        rank_S_list,
        log_det_const,
        overlapping,
        overlap_groups,
        M_p,
    ) = _setup_penalties(S_list, sp_fixed, RtR, p)

    eye_p = np.eye(p, dtype=float)

    def _eval_criterion(
        rho_vec: np.ndarray,
        compute_details: bool = False,
        compute_grad: bool = False,
    ) -> Any:
        current_sp = list(sp_vals)
        for i, idx in enumerate(free_indices):
            current_sp[idx] = float(np.exp(rho_vec[i]))

        S_total = np.zeros((p, p), dtype=float)
        for (s0, s1), sub, l_val in zip(pen_bounds, pen_sub, current_sp):
            if s1 > s0:
                S_total[s0:s1, s0:s1] += l_val * sub

        H = RtR + S_total

        try:
            c, lower = linalg.cho_factor(H, check_finite=False)
            coef = linalg.cho_solve((c, lower), Xty, check_finite=False)
            log_det_H = 2.0 * float(np.sum(np.log(np.diag(c))))
        except linalg.LinAlgError:
            max_d = float(np.max(np.diag(H))) if p > 0 else 0.0
            H_reg = H.copy()
            H_reg.flat[:: p + 1] += (max_d * 1e-12) if max_d > 0.0 else 1e-12
            try:
                c, lower = linalg.cho_factor(H_reg, check_finite=False)
                coef = linalg.cho_solve((c, lower), Xty, check_finite=False)
                log_det_H = 2.0 * float(np.sum(np.log(np.diag(c))))
            except linalg.LinAlgError:
                coef = np.linalg.lstsq(H, Xty, rcond=None)[0]
                ev_h = linalg.eigvalsh(0.5 * (H + H.T), check_finite=False)
                pos_h = (
                    ev_h[ev_h > max(float(ev_h[-1]) * 1e-12, np.finfo(float).tiny)]
                    if ev_h.size
                    else ev_h
                )
                log_det_H = float(np.sum(np.log(pos_h))) if pos_h.size else 0.0
                c, lower = None, False

        r_diff = qty - R @ coef
        rss = max(0.0, ss_orth + float(r_diff @ r_diff))
        Dp = rss + sum(
            l_val * float(coef[s0:s1] @ (sub @ coef[s0:s1]))
            for (s0, s1), sub, l_val in zip(pen_bounds, pen_sub, current_sp)
            if s1 > s0
        )

        H_inv = None
        if compute_details or compute_grad:
            if c is not None:
                try:
                    H_inv = linalg.cho_solve(
                        (c, lower), eye_p, check_finite=False
                    )
                except Exception:
                    H_inv = np.linalg.pinv(H)
            else:
                H_inv = np.linalg.pinv(H)

        edf = 0.0
        grad = np.zeros(len(free_indices), dtype=float) if compute_grad else None

        if method == "REML":
            df_reml = max(n_obs - M_p, 1)
            scale_est = max(Dp / df_reml, np.finfo(float).tiny)
            log_det_S = sum(
                rank_S_list[i] * np.log(current_sp[i]) + log_det_const[i]
                for i in range(n_pens)
                if not overlapping[i]
            )
            S_inv_proj: Dict[int, np.ndarray] = {}
            for grp, r_dim, proj_subs in overlap_groups:
                S_tilde = sum(current_sp[g_i] * proj_subs[g_i] for g_i in grp)
                d = np.sqrt(np.maximum(np.diag(S_tilde), 1e-300))
                S_bal = S_tilde / (d[:, None] * d[None, :])
                try:
                    c_s, low_s = linalg.cho_factor(S_bal, check_finite=False)
                    log_det_S += 2.0 * float(
                        np.sum(np.log(np.diag(c_s))) + np.sum(np.log(d))
                    )
                    if compute_grad:
                        S_bal_inv = linalg.cho_solve(
                            (c_s, low_s), np.eye(r_dim), check_finite=False
                        )
                        S_inv = S_bal_inv / (d[:, None] * d[None, :])
                        for g_i in grp:
                            S_inv_proj[g_i] = S_inv
                except Exception:
                    ev_s, u_s = linalg.eigh(S_tilde, check_finite=False)
                    pos_s = np.maximum(ev_s, 1e-300)
                    log_det_S += float(np.sum(np.log(pos_s)))
                    if compute_grad:
                        S_inv = (u_s / pos_s[None, :]) @ u_s.T
                        for g_i in grp:
                            S_inv_proj[g_i] = S_inv

            score = (
                Dp / (2.0 * scale_est)
                + 0.5 * df_reml * np.log(2.0 * np.pi * scale_est)
                + 0.5 * log_det_H
                - 0.5 * log_det_S
                - 0.5 * log_w_sum
            )
            if compute_grad and H_inv is not None and grad is not None:
                for k_i, idx in enumerate(free_indices):
                    s0, s1 = pen_bounds[idx]
                    sub = pen_sub[idx]
                    lam_i = current_sp[idx]
                    b_sub = coef[s0:s1]
                    dDp = lam_i * float(b_sub @ sub @ b_sub)
                    dlogH = lam_i * float(np.sum(H_inv[s0:s1, s0:s1] * sub))
                    if not overlapping[idx]:
                        dlogS = float(rank_S_list[idx])
                    else:
                        dlogS = lam_i * float(
                            np.sum(
                                S_inv_proj[idx]
                                * next(
                                    ps[idx]
                                    for grp, _, ps in overlap_groups
                                    if idx in grp
                                )
                            )
                        )
                    grad[k_i] = 0.5 * (dDp / scale_est + dlogH - dlogS)
        else:
            if H_inv is not None:
                edf = float(np.sum(H_inv * RtR))
            elif c is not None:
                try:
                    Z_Rt = linalg.solve_triangular(
                        c,
                        R.T,
                        lower=lower,
                        trans="N" if lower else "T",
                        check_finite=False,
                    )
                    edf = float(np.sum(Z_Rt**2))
                except Exception:
                    H_inv = linalg.cho_solve(
                        (c, lower), eye_p, check_finite=False
                    )
                    edf = float(np.sum(H_inv * RtR))
            else:
                H_inv = np.linalg.pinv(H)
                edf = float(np.sum(H_inv * RtR))

            denom = max(n_obs - gamma * edf, 1e-6)
            score = (n_obs * rss) / (denom**2)
            scale_est = rss / max(n_obs - edf, 1)
            if compute_grad and H_inv is not None and grad is not None:
                res_grad_vec = H_inv @ (RtR @ coef - Xty)
                H_RtR_H = (H_inv @ RtR) @ H_inv
                for k_i, idx in enumerate(free_indices):
                    s0, s1 = pen_bounds[idx]
                    sub = pen_sub[idx]
                    lam_i = current_sp[idx]
                    drss = (
                        -2.0
                        * lam_i
                        * float(res_grad_vec[s0:s1] @ (sub @ coef[s0:s1]))
                    )
                    dedf = -lam_i * float(np.sum(H_RtR_H[s0:s1, s0:s1] * sub))
                    grad[k_i] = (n_obs / (denom**2)) * drss + (
                        2.0 * n_obs * rss * gamma / (denom**3)
                    ) * dedf

        if compute_details and H_inv is not None and edf == 0.0:
            edf = float(np.sum(H_inv * RtR))

        if compute_grad:
            return score, grad
        return score, coef, H_inv, rss, scale_est, edf

    if len(free_indices) == 0:
        best_score, best_coef, best_H_inv, best_rss, _, best_edf = (
            _eval_criterion(np.zeros(0), compute_details=True)
        )
        best_scale = best_rss / max(n_obs - best_edf, 1)
        best_sp = sp_vals
    else:
        n_free = len(free_indices)
        rho_init = _init_rho(RtR, pen_bounds, pen_sub, n_pens, free_indices, p)
        opt = optimize.minimize(
            lambda r: _eval_criterion(r, compute_grad=True),
            rho_init,
            jac=True,
            method="L-BFGS-B",
            bounds=[(-15.0, 25.0)] * n_free,
            options={"ftol": 1e-14, "gtol": 1e-10},
        )
        if np.any(opt.x <= -14.9) or np.any(opt.x >= 20.0):
            candidates = [np.full(n_free, v) for v in (-8.0, -4.0, 0.0, 4.0, 8.0)]
            r_grid = min(candidates, key=lambda r: _eval_criterion(r)[0])
            opt2 = optimize.minimize(
                lambda r: _eval_criterion(r, compute_grad=True),
                r_grid,
                jac=True,
                method="L-BFGS-B",
                bounds=[(-15.0, 25.0)] * n_free,
                options={"ftol": 1e-14, "gtol": 1e-10},
            )
            if opt2.fun < opt.fun:
                opt = opt2
        best_score, best_coef, best_H_inv, best_rss, best_scale, best_edf = (
            _eval_criterion(opt.x, compute_details=True)
        )
        best_sp = list(sp_vals)
        for i, idx in enumerate(free_indices):
            best_sp[idx] = float(np.exp(opt.x[i]))

    scale_val = (
        best_rss / max(n_obs - best_edf, 1.0)
        if len(free_indices) < n_pens
        else best_scale
    )
    Vp = best_H_inv * scale_val
    H_inv_RtR = best_H_inv @ RtR
    Ve = (H_inv_RtR @ best_H_inv) * scale_val
    se = np.sqrt(np.maximum(np.diag(Vp), 0.0))

    edf_per_coef = np.diag(H_inv_RtR)
    total_edf = float(np.sum(edf_per_coef))

    F_mat = H_inv_RtR
    edf1_per_coef = 2.0 * edf_per_coef - np.einsum("ij,ji->i", F_mat, F_mat)
    S_final = np.zeros((p, p), dtype=float)
    S_unit = np.zeros((p, p), dtype=float)
    for (s0, s1), sub, l_val in zip(pen_bounds, pen_sub, best_sp):
        if s1 > s0:
            S_final[s0:s1, s0:s1] += l_val * sub
            if l_val > 0.0:
                s_norm = float(np.max(np.abs(sub)))
                if s_norm > 0.0:
                    S_unit[s0:s1, s0:s1] += sub / s_norm

    edf2_per_coef = None
    if method == "REML" and free_indices:
        # Wood et al. (2016) smoothing-parameter-uncertainty correction, as in mgcv's edf2.
        n_free = len(free_indices)
        eps = 1e-4
        hess = np.zeros((n_free, n_free))
        for k_i in range(n_free):
            e = np.zeros(n_free)
            e[k_i] = eps
            hess[:, k_i] = (
                _eval_criterion(opt.x + e, compute_grad=True)[1]
                - _eval_criterion(opt.x - e, compute_grad=True)[1]
            ) / (2 * eps)
        hess = 0.5 * (hess + hess.T)
        try:
            edf2_per_coef = _edf2_per_coef(
                hess, best_H_inv, best_coef, best_sp, free_indices, pen_bounds,
                pen_sub, RtR, Vp, scale_val, max(n_obs - M_p, 1), S_final,
            )
            if edf2_per_coef.sum() > edf1_per_coef.sum():
                edf2_per_coef = edf1_per_coef
        except Exception:
            edf2_per_coef = None
    if coef_slices is not None:
        edf_per_term, f_stat_per_term, p_values_per_term, ref_df_per_term = _compute_term_tests(
            best_coef, X_w, R, S_final, Vp, Ve, scale_val, coef_slices, pen_bounds,
            edf_per_coef, edf1_per_coef, max(n_obs - total_edf, 1.0), S_unit=S_unit,
        )
    else:
        edf_per_term, f_stat_per_term, p_values_per_term, ref_df_per_term = [], [], [], []
    null_deviance, dev_explained, r_sq_adj, log_w_sum, _ = _fit_summary_stats(
        y_raw, X @ best_coef, weights, best_rss, total_edf
    )

    gcv_score = (n_obs * best_rss) / (max(n_obs - gamma * total_edf, 1.0) ** 2)

    return {
        "coef": best_coef,
        "sp": best_sp,
        "scale": scale_val,
        "Vp": Vp,
        "Ve": Ve,
        "se": se,
        "edof": total_edf,
        "edof_per_coef": edf_per_coef,
        "edf_per_term": edf_per_term,
        "f_stat_per_term": f_stat_per_term,
        "ref_df_per_term": ref_df_per_term,
        "edf1_per_coef": edf1_per_coef,
        "edof_aic": float(np.sum(edf2_per_coef if edf2_per_coef is not None else edf_per_coef)),
        "p_values_per_term": p_values_per_term,
        "rss": best_rss,
        "deviance": best_rss,
        "null_deviance": null_deviance,
        "deviance_explained": dev_explained,
        "r_sq_adj": r_sq_adj,
        "GCV": gcv_score,
        "score": best_score,
        "method": method,
        "log_weights_sum": log_w_sum,
        "n_samples": n_obs,
        "H_inv": best_H_inv,
        "RtR": RtR,
    }
