"""Multi-term GAM with formula interface and REML/GCV optimization (Wood backend)."""

from __future__ import annotations

from itertools import product as cartesian_product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from .._base import _GAMBase, _validate_weights
from .._formula import parse_formula
from ._optimizer import fit_wood_engine
from ._terms import (
    _WoodFactorTerm,
    _WoodLinearTerm,
    _WoodSplineTerm,
    _WoodTensorTerm,
    _WoodTerm,
    instantiate_wood_terms,
)


class GAMCore(_GAMBase):
    r"""Multi-term Generalized Additive Model with formula interface (Wood backend).

    Fits multivariate additive models :math:`y = \beta_0 + \sum_j f_j(x_j) + \epsilon`
    using Simon Wood's continuous REML or GCV smoothing parameter optimization.

    Parameters
    ----------
    formula : str
        Additive formula expression (e.g. ``"s(0, k=10) + te(1, 2) + l(3)"``).
    method : {'REML', 'GCV'}, default='REML'
        Smoothing optimization criterion.
    gamma : float, default=1.0
        GCV penalty inflation multiplier.
    fit_intercept : bool, default=True
        Whether to fit an unpenalized global intercept.
    spline_order : int, default=3
        Default spline polynomial order.
    select : bool, default=False
        If True, applies extra null-space shrinkage penalties for feature selection.
    """

    _scale_is_var = True

    def __init__(
        self,
        formula: str,
        method: str = "REML",
        gamma: float = 1.0,
        fit_intercept: bool = True,
        spline_order: int = 3,
        select: bool = False,
    ) -> None:
        self.formula = formula
        self.method = method.upper()
        self.gamma = float(gamma)
        self.fit_intercept = bool(fit_intercept)
        self.spline_order = int(spline_order)
        self.select = bool(select)

        configs = parse_formula(formula)
        self._term_configs = configs
        self._terms: List[_WoodTerm] = instantiate_wood_terms(
            configs, select=self.select, fit_intercept=self.fit_intercept
        )

        self._coef_slices: List[Tuple[int, int]] = []
        self.coef_: Optional[np.ndarray] = None
        self.statistics_: Optional[Dict[str, Any]] = None
        self._design_matrix_: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return f"GAMCore(formula='{self.formula}', method='{self.method}') [{status}]"

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        weights: Optional[ArrayLike] = None,
    ) -> "GAMCore":
        r"""Fit multi-term GAM model using REML or GCV criteria.

        Parameters
        ----------
        x : array_like of shape (n_samples, n_features) or (n_samples,)
            Design covariate data.
        y : array_like of shape (n_samples,)
            Response values.
        weights : array_like of shape (n_samples,), optional
            Non-negative prior observation weights :math:`w_i`.

        Returns
        -------
        self : GAMCore
            Fitted model instance.
        """
        x_arr = np.asarray(x)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        for col_idx in range(x_arr.shape[1]):
            col = x_arr[:, col_idx]
            try:
                col_float = np.asarray(col, dtype=float)
            except (ValueError, TypeError):
                continue
            if not np.all(np.isfinite(col_float)):
                raise ValueError("Input data contains NaN or infinity.")
        y_arr = np.asarray(y, dtype=float).ravel()
        if not np.all(np.isfinite(y_arr)):
            raise ValueError("Input data contains NaN or infinity.")
        w_arr = _validate_weights(weights, len(y_arr))

        self._compile(x_arr)
        self._compute_coef_slices()
        X = self._build_model_matrix(x_arr)
        self._design_matrix_ = X
        p = X.shape[1]

        S_list: List[np.ndarray] = []
        sp_fixed: List[Optional[float]] = []
        pen_counts: List[int] = []

        for i, term in enumerate(self._terms):
            start, end = self._coef_slices[i]
            pens = term.build_penalties()
            pen_counts.append(len(pens))
            term_sps = term.sp if term.sp is not None else [None] * len(pens)
            for pen_idx, P_sub in enumerate(pens):
                P_full = np.zeros((p, p), dtype=float)
                P_full[start:end, start:end] = P_sub
                S_list.append(P_full)
                sp_fixed.append(
                    term_sps[pen_idx] if pen_idx < len(term_sps) else None
                )

        fit_res = fit_wood_engine(
            X=X,
            y=y_arr,
            S_list=S_list,
            sp_fixed=(
                sp_fixed if any(sp is not None for sp in sp_fixed) else None
            ),
            method=self.method,
            gamma=self.gamma,
            coef_slices=self._coef_slices,
            M_p=1 if self.fit_intercept else 0,
            weights=w_arr,
        )

        self.coef_ = fit_res["coef"]

        sp_idx = 0
        for term, n_pens in zip(self._terms, pen_counts):
            if n_pens > 0:
                term.sp = fit_res["sp"][sp_idx : sp_idx + n_pens]
                sp_idx += n_pens

        self.statistics_ = {
            "edof": fit_res["edof"],
            "edof_per_coef": fit_res["edof_per_coef"],
            "scale": fit_res["scale"],
            "cov": fit_res["Vp"],
            "Vp": fit_res["Vp"],
            "Ve": fit_res["Ve"],
            "se": fit_res["se"],
            "rss": fit_res["rss"],
            "deviance": fit_res["deviance"],
            "null_deviance": fit_res["null_deviance"],
            "deviance_explained": fit_res["deviance_explained"],
            "r_sq_adj": fit_res["r_sq_adj"],
            "n_samples": fit_res["n_samples"],
            "GCV": fit_res["GCV"],
            "score": fit_res["score"],
            "method": fit_res["method"],
            "sp": fit_res["sp"],
            "edf_per_term": fit_res["edf_per_term"],
            "f_stat_per_term": fit_res["f_stat_per_term"],
            "p_values_per_term": fit_res["p_values_per_term"],
            "ref_df_per_term": fit_res["ref_df_per_term"],
            "edof_aic": fit_res["edof_aic"],
            "log_weights_sum": fit_res["log_weights_sum"],
        }
        return self

    def gridsearch(
        self,
        x: ArrayLike,
        y: ArrayLike,
        n_splines_grids: Optional[Any] = None,
        gamma: float = 1.0,
        n_splines: Optional[Any] = None,
        search_terms: Optional[Any] = None,
        weights: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> "GAMCore":
        """Search over candidate basis dimensions k for smooth terms via GCV."""
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        w_arr = _validate_weights(weights, len(y_arr))

        if n_splines_grids is None:
            n_splines_grids = n_splines if n_splines is not None else kwargs.get("k")

        all_smooth = [
            i
            for i, t in enumerate(self._terms)
            if isinstance(t, (_WoodSplineTerm, _WoodTensorTerm))
        ]
        if not all_smooth:
            return self.fit(x_arr, y_arr, weights=w_arr)

        if search_terms is None:
            smooth_indices = all_smooth
        else:
            items = [search_terms] if isinstance(search_terms, (int, str)) else list(search_terms)
            allowed = set()
            for item in items:
                if isinstance(item, int):
                    allowed.add(item)
                elif item == "all":
                    allowed.update(all_smooth)
                elif item == "s":
                    allowed.update(i for i, t in enumerate(self._terms) if isinstance(t, _WoodSplineTerm))
                elif item in ("te", "ti"):
                    allowed.update(i for i, t in enumerate(self._terms) if isinstance(t, _WoodTensorTerm) and t.kind == item)
            smooth_indices = [i for i in all_smooth if i in allowed]
            if not smooth_indices:
                return self.fit(x_arr, y_arr, weights=w_arr)

        if n_splines_grids is None:
            grids = [
                np.array([4, 6, 8])
                if isinstance(self._terms[i], _WoodTensorTerm)
                else np.arange(6, 16, 2)
                for i in smooth_indices
            ]
        elif isinstance(n_splines_grids, list) and len(n_splines_grids) > 0 and isinstance(n_splines_grids[0], (list, tuple, np.ndarray)):
            grids = [np.atleast_1d(g) for g in n_splines_grids]
        else:
            grids = [np.atleast_1d(n_splines_grids)] * len(smooth_indices)

        best_gcv = np.inf
        best_model: Optional[GAMCore] = None

        for combo in cartesian_product(*grids):
            configs_copy = [dict(cfg) for cfg in self._term_configs]
            for idx_in_smooth, term_i in enumerate(smooth_indices):
                cfg = configs_copy[term_i]
                cfg["kwargs"] = {
                    **cfg.get("kwargs", {}),
                    "k": int(combo[idx_in_smooth]),
                }
            cand_model = GAMCore(
                formula=self.formula,
                method=self.method,
                gamma=gamma,
                fit_intercept=self.fit_intercept,
                spline_order=self.spline_order,
                select=self.select,
            )
            cand_model._terms = instantiate_wood_terms(
                configs_copy, select=self.select, fit_intercept=self.fit_intercept
            )
            cand_model._term_configs = configs_copy
            try:
                cand_model.fit(x_arr, y_arr, weights=w_arr)
                if cand_model.statistics_["GCV"] < best_gcv:
                    best_gcv = cand_model.statistics_["GCV"]
                    best_model = cand_model
            except Exception:
                continue

        if best_model is not None:
            self._term_configs = best_model._term_configs
            self._terms = best_model._terms
            self._coef_slices = best_model._coef_slices
            self.coef_ = best_model.coef_
            self.statistics_ = best_model.statistics_
            self._design_matrix_ = best_model._design_matrix_
            return self

        return self.fit(x_arr, y_arr, weights=w_arr)

    def plot_decomposition(
        self,
        x: ArrayLike,
        y: ArrayLike,
        n_grid: int = 200,
        figsize: Optional[Tuple[int, int]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple:
        """Plot partial dependence for all terms in the GAM."""
        return self._plot_gamcore_decomposition(
            x,
            y,
            n_grid=n_grid,
            extrap_frac=0.0,
            figsize=figsize,
            feature_names=feature_names,
        )
