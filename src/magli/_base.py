"""Shared base class, plotting, and tensor helpers for GAM backends."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def _validate_weights(
    weights: Optional[ArrayLike], n: int
) -> Optional[np.ndarray]:
    """Validate and return 1D observation weights array, or None."""
    if weights is None:
        return None
    w = np.asarray(weights, dtype=float).ravel()
    if len(w) != n:
        raise ValueError(
            f"weights length ({len(w)}) must match n_samples ({n})."
        )
    if not np.all(np.isfinite(w)) or np.any(w < 0) or np.all(w == 0):
        raise ValueError(
            "weights must be finite, non-negative, and not all zero."
        )
    return w


def _khatri_rao(marginals: List[np.ndarray]) -> np.ndarray:
    """Row-wise Kronecker (Khatri-Rao) product of marginal basis matrices."""
    n = marginals[0].shape[0]
    result = marginals[0]
    for B_next in marginals[1:]:
        result = (result[:, :, None] * B_next[:, None, :]).reshape(n, -1)
    return result


def _kron_sum(penalties: List[np.ndarray]) -> np.ndarray:
    """Kronecker sum: sum_i I_0 kron ... kron P_i kron ... kron I_{d-1}."""
    sizes = [p.shape[0] for p in penalties]
    total = int(np.prod(sizes))
    result = np.zeros((total, total))
    for i, P_i in enumerate(penalties):
        I_pre = np.eye(int(np.prod(sizes[:i])))
        I_post = np.eye(int(np.prod(sizes[i + 1 :])))
        result += np.kron(np.kron(I_pre, P_i), I_post)
    return result


class _GAMBase:
    r"""Shared lifecycle, prediction, inference, criteria, and diagnostic methods.

    Abstract base class inherited by both the Wood (:class:`magli.wood.GAMCore`)
    and Eilers-Marx (:class:`magli.eilers.GAMCore`) engines. Provides unified
    prediction, confidence/prediction interval computation, partial dependence
    evaluation, information criteria (AIC, BIC), log-likelihood, and visualization.

    Attributes
    ----------
    coef_ : ndarray of shape (n_coefficients,) or None
        Estimated regression coefficients across all model terms and intercept.
    statistics_ : dict or None
        Dictionary of fit criteria, residual statistics, and covariance matrices.
    formula : str
        Model formula string.
    fit_intercept : bool
        Whether an intercept term :math:`\beta_0` is fitted.
    is_fitted : bool
        Boolean flag indicating whether the model has been fitted to data.
    n_terms : int
        Number of additive terms in the model specification.
    """

    _scale_is_var: bool = False
    formula: str = ""
    fit_intercept: bool = True
    _terms: List[Any] = []
    _coef_slices: List[Tuple[int, int]] = []
    coef_: Optional[np.ndarray] = None
    statistics_: Optional[Dict[str, Any]] = None
    _design_matrix_: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        return self.coef_ is not None

    @property
    def n_terms(self) -> int:
        return len(self._terms)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() or gridsearch() first.")

    _check_is_fitted = _check_fitted

    def _compile(self, x: np.ndarray) -> None:
        self.n_features_in_ = x.shape[1] if x.ndim > 1 else 1
        for term in self._terms:
            term.compile(x)

    def _compute_coef_slices(self) -> None:
        slices = []
        offset = 0
        for t in self._terms:
            slices.append((offset, offset + t.n_coefs))
            offset += t.n_coefs
        self._coef_slices = slices

    def _build_model_matrix(self, x: np.ndarray) -> np.ndarray:
        blocks = [t.build_columns(x) for t in self._terms]
        if self.fit_intercept:
            blocks.append(np.ones((len(x), 1), dtype=float))
        return np.hstack(blocks).astype(float)

    def predict(
        self,
        x: ArrayLike,
        type: str = "response",
        se_fit: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        r"""Predict target values or individual additive term contributions for feature data.

        Evaluates the fitted model :math:`\hat{y} = \hat{\beta}_0 + \sum_j \hat{f}_j(x_j)`
        or returns a matrix of separate term evaluations :math:`\hat{f}_j(x_j)`.

        Parameters
        ----------
        x : array_like of shape (n_samples, n_features) or (n_samples,)
            Input evaluation points. If 1D, reshaped to `(n_samples, 1)`.
        type : {'response', 'terms'}, default='response'
            Prediction target:

            - 'response' : Returns the overall linear predictor :math:`\hat{y} = X\hat{\beta}`.
            - 'terms' : Returns a 2D matrix of shape `(n_samples, n_terms)` containing
              the centered contribution of each individual additive term.
        se_fit : bool, default=False
            Whether to calculate standard error estimates for each prediction based on
            the estimated Bayesian covariance matrix :math:`V_\beta = (X^T W X + S)^{-1} \hat{\sigma}^2`.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_terms)
            Fitted values. If `type='terms'`, returns an array where column `j` corresponds
            to term `j`.
        se : ndarray of shape (n_samples,) or (n_samples, n_terms), optional
            Standard error estimates corresponding to `y_pred`. Only returned if `se_fit=True`.

        Raises
        ------
        RuntimeError
            If the model has not been fitted prior to calling predict.
        ValueError
            If `type` is not `'response'` or `'terms'`.

        See Also
        --------
        partial_dependence : Evaluate contribution of a single term.
        confidence_intervals : Pointwise confidence intervals for the linear predictor.
        prediction_intervals : Pointwise prediction intervals for new observations.

        Notes
        -----
        Standard error calculation uses the Bayesian posterior covariance matrix:

        .. math::

            \operatorname{SE}(\hat{y}_i) = \sqrt{X_i V_\beta X_i^T}

        where :math:`X_i` is the row of the model matrix for sample :math:`i`. For individual
        terms (:math:`type='terms'`), only the sub-block of :math:`V_\beta` corresponding to
        that term's spline coefficients is used.

        Examples
        --------
        >>> import numpy as np
        >>> import magli as mg
        >>> x = np.linspace(0, 1, 50)
        >>> y = 2.0 * x**2 + np.random.default_rng(0).normal(0, 0.05, size=50)
        >>> model = mg.LinGAM(n_splines=6).fit(x, y)
        >>> preds, ses = model.predict(np.array([0.2, 0.8]), se_fit=True)
        >>> preds.shape
        (2,)
        >>> ses.shape
        (2,)
        """
        self._check_fitted()
        x_arr = np.asarray(x, dtype=float)
        n_feat = getattr(self, "n_features_in_", 1)
        if x_arr.ndim == 0:
            x_arr = x_arr.reshape(1, 1)
        elif x_arr.ndim == 1:
            if n_feat > 1 and len(x_arr) == n_feat:
                x_arr = x_arr.reshape(1, -1)
            else:
                x_arr = x_arr.reshape(-1, 1)

        cov = self.statistics_.get("cov")
        if type == "terms":
            n = len(x_arr)
            terms_fit = np.zeros((n, len(self._terms)), dtype=float)
            terms_se = (
                np.zeros((n, len(self._terms)), dtype=float) if se_fit else None
            )
            for j, term in enumerate(self._terms):
                B_j = term.build_columns(x_arr)
                start, end = self._coef_slices[j]
                terms_fit[:, j] = B_j @ self.coef_[start:end]
                if se_fit and cov is not None:
                    cov_j = cov[start:end, start:end]
                    terms_se[:, j] = np.sqrt(
                        np.maximum(np.sum((B_j @ cov_j) * B_j, axis=1), 0.0)
                    )
            return (terms_fit, terms_se) if se_fit else terms_fit

        if type != "response":
            raise ValueError(f"Unknown predict type '{type}'. Choose 'response' or 'terms'.")

        X = self._build_model_matrix(x_arr)
        lp = X @ self.coef_
        if se_fit:
            se = (
                np.sqrt(np.maximum(np.sum((X @ cov) * X, axis=1), 0.0))
                if cov is not None
                else np.zeros_like(lp)
            )
            return lp, se
        return lp

    def partial_dependence(
        self, term_idx: int, x: ArrayLike, se_fit: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        r"""Evaluate the centered partial contribution of a single model term.

        Computes :math:`\hat{f}_j(x_j) = B_j(x_j) \hat{\beta}_j` for the specified term
        index, isolating its isolated effect from the intercept and all other covariates.

        Parameters
        ----------
        term_idx : int
            Zero-based index of the term in the model formula (e.g. 0 for the first term).
        x : array_like of shape (n_samples, n_features) or (n_samples,)
            Evaluation data.
        se_fit : bool, default=False
            Whether to return standard error estimates for the term contribution.

        Returns
        -------
        contrib : ndarray of shape (n_samples,)
            Partial contribution of the specified term across evaluation points.
        se : ndarray of shape (n_samples,), optional
            Standard error of the term contribution. Only returned if `se_fit=True`.

        Raises
        ------
        IndexError
            If `term_idx` is out of bounds for the fitted model.

        See Also
        --------
        predict : Predict overall response or all terms simultaneously.
        plot_decomposition : Visualize all term partial dependencies in subplots.

        Examples
        --------
        >>> import numpy as np
        >>> import magli as mg
        >>> X = np.column_stack([np.linspace(0, 1, 50), np.linspace(-1, 1, 50)])
        >>> y = np.sin(X[:, 0]) + X[:, 1]
        >>> model = mg.GAMCore("s(0) + l(1)").fit(X, y)
        >>> pdep_s0 = model.partial_dependence(0, X[:3])
        >>> pdep_s0.shape
        (3,)
        """
        self._check_fitted()
        if term_idx < 0 or term_idx >= len(self._terms):
            raise IndexError(
                f"term_idx {term_idx} out of range (0..{len(self._terms) - 1})."
            )
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 0:
            x_arr = x_arr.reshape(1)
        if x_arr.ndim == 1:
            term = self._terms[term_idx]
            feat_idx = getattr(term, "feature", None)
            if feat_idx is not None and feat_idx > 0:
                mat = np.zeros((len(x_arr), feat_idx + 1), dtype=float)
                mat[:, feat_idx] = x_arr
                x_arr = mat
            else:
                x_arr = x_arr.reshape(-1, 1)
        B_term = self._terms[term_idx].build_columns(x_arr)
        start, end = self._coef_slices[term_idx]
        contrib = B_term @ self.coef_[start:end]
        if se_fit:
            cov = self.statistics_.get("cov")
            cov_sub = (
                cov[start:end, start:end]
                if cov is not None
                else np.zeros((end - start, end - start))
            )
            se = np.sqrt(
                np.maximum(np.sum((B_term @ cov_sub) * B_term, axis=1), 0.0)
            )
            return contrib, se
        return contrib

    def confidence_intervals(self, x: ArrayLike, width: float = 0.95) -> np.ndarray:
        r"""Compute pointwise confidence intervals for the true mean function :math:`E[y|x]`.

        Calculates upper and lower confidence bounds reflecting epistemic parameter
        uncertainty in the estimated spline coefficients.

        Parameters
        ----------
        x : array_like of shape (n_samples, n_features) or (n_samples,)
            Evaluation points.
        width : float, default=0.95
            Nominal coverage probability of the interval (e.g. 0.95 for a 95% interval).
            Must be strictly between 0 and 1.

        Returns
        -------
        intervals : ndarray of shape (n_samples, 2)
            Two-column array where column 0 contains the lower bound and column 1
            contains the upper bound: `[lower_bound, upper_bound]`.

        See Also
        --------
        prediction_intervals : Prediction intervals incorporating residual noise :math:`\sigma^2`.
        predict : Point predictions without uncertainty bounds.

        Notes
        -----
        The confidence interval is given by:

        .. math::

            \hat{y}(x) \pm z_{1-lpha/2} \sqrt{X(x) V_\beta X(x)^T}

        where :math:`V_\beta = (X^T W X + S)^{-1} \hat{\sigma}^2` is the Bayesian posterior
        covariance matrix of the regression coefficients and :math:`z_{1-lpha/2}` is the
        normal quantile (or Student's :math:`t` quantile for small-sample models).

        Examples
        --------
        >>> import numpy as np
        >>> import magli as mg
        >>> x = np.linspace(0, 1, 40)
        >>> y = np.exp(x) + np.random.default_rng(1).normal(0, 0.05, size=40)
        >>> model = mg.LinGAM(n_splines=6).fit(x, y)
        >>> ci = model.confidence_intervals(np.array([0.5]), width=0.95)
        >>> ci.shape
        (1, 2)
        >>> ci[0, 0] < ci[0, 1]
        True
        """
        return self._compute_intervals(x, width, prediction=False)

    def prediction_intervals(self, x: ArrayLike, width: float = 0.95) -> np.ndarray:
        r"""Compute pointwise prediction intervals for future observable responses :math:`y_{\text{new}}`.

        Unlike confidence intervals which quantify uncertainty in the expected mean :math:`E[y|x]`,
        prediction intervals also account for aleatoric observation noise :math:`\epsilon \sim \mathcal{N}(0, \sigma^2)`.

        Parameters
        ----------
        x : array_like of shape (n_samples, n_features) or (n_samples,)
            Evaluation points.
        width : float, default=0.95
            Nominal coverage probability of the interval (e.g. 0.95 for a 95% interval).

        Returns
        -------
        intervals : ndarray of shape (n_samples, 2)
            Two-column array where column 0 is the lower prediction limit and column 1
            is the upper prediction limit.

        See Also
        --------
        confidence_intervals : Confidence intervals for the mean response.

        Notes
        -----
        The total variance for a new observation combines parameter variance and residual variance:

        .. math::

            \operatorname{Var}(y_{\text{new}} | x) = X(x) V_\beta X(x)^T + \hat{\sigma}^2

        The prediction interval is:

        .. math::

            \hat{y}(x) \pm t_{n - \text{edf}, 1-lpha/2} \sqrt{\operatorname{Var}(y_{\text{new}} | x)}

        Examples
        --------
        >>> import numpy as np
        >>> import magli as mg
        >>> x = np.linspace(0, 1, 40)
        >>> y = np.sin(x) + np.random.default_rng(2).normal(0, 0.1, size=40)
        >>> model = mg.LinGAM(n_splines=6).fit(x, y)
        >>> pi = model.prediction_intervals(np.array([0.5]), width=0.95)
        >>> ci = model.confidence_intervals(np.array([0.5]), width=0.95)
        >>> (pi[0, 1] - pi[0, 0]) > (ci[0, 1] - ci[0, 0])
        True
        """
        return self._compute_intervals(x, width, prediction=True)

    def _compute_intervals(
        self, x: ArrayLike, width: float, prediction: bool
    ) -> np.ndarray:
        self._check_fitted()
        x_arr = np.asarray(x, dtype=float)
        n_feat = getattr(self, "n_features_in_", 1)
        if x_arr.ndim == 0:
            x_arr = x_arr.reshape(1, 1)
        elif x_arr.ndim == 1:
            if n_feat > 1 and len(x_arr) == n_feat:
                x_arr = x_arr.reshape(1, -1)
            else:
                x_arr = x_arr.reshape(-1, 1)
        X = self._build_model_matrix(x_arr)
        lp = X @ self.coef_

        cov = self.statistics_["cov"]
        var = np.sum((X @ cov) * X, axis=1)
        if prediction:
            s = self.statistics_["scale"]
            var = var + (s if self._scale_is_var else s**2)

        alpha = (1.0 - width) / 2.0
        df = max(self.statistics_["n_samples"] - self.statistics_["edof"], 1.0)
        # mgcv confidence bands use normal quantiles with the Bayesian covariance
        dist = stats.norm if (self._scale_is_var and not prediction) else stats.t(df=df)
        q_lo = dist.ppf(alpha)
        q_hi = dist.ppf(1.0 - alpha)

        se = np.sqrt(np.maximum(var, 1e-14 if self._scale_is_var else 0.0))
        return np.column_stack([lp + q_lo * se, lp + q_hi * se])

    def loglikelihood(self) -> float:
        r"""Compute the Gaussian log-likelihood of the fitted model.

        Evaluates the log-likelihood function under Gaussian errors:

        .. math::

            \ell(\beta, \sigma^2) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{\text{RSS}}{2\sigma^2} + \frac{1}{2}\sum_i \ln w_i

        where :math:`\text{RSS} = \sum_i w_i (y_i - \hat{y}_i)^2` is the weighted residual
        sum of squares and :math:`\sigma^2` is the estimated scale parameter.

        Returns
        -------
        loglik : float
            Gaussian log-likelihood value.

        See Also
        --------
        aic : Akaike Information Criterion.
        bic : Bayesian Information Criterion.
        """
        self._check_fitted()
        n = self.statistics_["n_samples"]
        rss = self.statistics_["rss"]
        s = self.statistics_["scale"]
        s2 = rss / n if self._scale_is_var else s**2  # mgcv uses the ML scale dev/n
        log_w = self.statistics_.get("log_weights_sum", 0.0)
        return float(
            -0.5 * n * np.log(2.0 * np.pi * s2) - 0.5 * rss / s2 + 0.5 * log_w
        )

    def _ll_df(self) -> float:
        """Degrees of freedom for AIC/BIC; mgcv uses edf2 (smoothing-parameter-uncertainty corrected) when available."""
        st = self.statistics_
        df = st.get("edof_aic", st["edof"]) + 1
        return float(min(df, len(self.coef_) + 1)) if self._scale_is_var else float(df)

    def aic(self) -> float:
        r"""Compute the Akaike Information Criterion (AIC).

        Calculates:

        .. math::

            \text{AIC} = -2\ell + 2 \cdot \text{EDF}

        where :math:`\ell` is the log-likelihood and :math:`\text{EDF}` includes the
        degrees of freedom of the fitted model (accounting for smoothing parameter
        uncertainty via ``edf2`` when available, matching R's ``mgcv``).

        Returns
        -------
        aic_score : float
            AIC score (lower values indicate a preferred model).
        """
        self._check_fitted()
        return float(-2.0 * self.loglikelihood() + 2.0 * self._ll_df())

    def bic(self) -> float:
        r"""Compute the Bayesian Information Criterion (BIC).

        Calculates:

        .. math::

            \text{BIC} = -2\ell + \ln(n) \cdot \text{EDF}

        penalizing model complexity more heavily than AIC for sample sizes :math:`n > 7`.

        Returns
        -------
        bic_score : float
            BIC score (lower values indicate a preferred model).
        """
        self._check_fitted()
        n = self.statistics_["n_samples"]
        return float(
            -2.0 * self.loglikelihood() + np.log(n) * self._ll_df()
        )

    def deviance_explained(self) -> float:
        r"""Compute the fraction of null deviance explained (pseudo :math:`R^2`).

        Calculates:

        .. math::

            R^2_{\text{dev}} = 1 - \frac{\text{Deviance}_{\text{model}}}{\text{Deviance}_{\text{null}}}

        Returns
        -------
        r2 : float
            Proportion of deviance explained in :math:`[0, 1]`.
        """
        self._check_fitted()
        null_dev = self.statistics_.get("null_deviance")
        if not null_dev:
            return 0.0
        return float(max(0.0, 1.0 - self.statistics_["rss"] / null_dev))

    def summary(self) -> str:
        r"""Generate a formatted regression table summarizing terms, significance, and fit criteria.

        Outputs parametric coefficient estimates (intercept, linear terms) with standard
        errors, t-statistics, and p-values, followed by smooth terms with effective degrees
        of freedom (EDF), rank, test statistics, and approximate p-values matching R's
        ``summary.gam`` Wood (2013).

        Returns
        -------
        summary_str : str
            Multi-line formatted summary table string.

        See Also
        --------
        deviance_explained : Pseudo R-squared metric.
        aic : Akaike Information Criterion.
        bic : Bayesian Information Criterion.

        References
        ----------
        - Wood (2013): Wood, S. N. (2013). "On p-values for smooth components of an extended
               generalized additive model." *Biometrika*, 100(1), 221-228.
        """
        self._check_fitted()
        n = self.statistics_["n_samples"]
        score_name = "REML" if getattr(self, "method", None) == "REML" else "GCV"
        score_val = self.statistics_.get("score", self.statistics_["GCV"])

        lines = [
            "",
            "Family: gaussian ",
            "Link function: identity ",
            "",
            f"Formula: {self.formula}",
            "",
        ]

        param_rows = []
        df_res = max(n - self.statistics_["edof"], 1.0)
        if self.fit_intercept:
            b0 = self.coef_[-1]
            se0 = self.statistics_["se"][-1]
            t0 = b0 / max(se0, 1e-12)
            p0 = float(2.0 * stats.t.sf(np.abs(t0), df=df_res))
            param_rows.append(("(Intercept)", b0, se0, t0, p0))

        for i, t in enumerate(self._terms):
            tname = type(t).__name__
            if "FactorTerm" in tname:
                levels = t._levels[1:] if t.coding == "dummy" else t._levels
                for j, level in enumerate(levels):
                    k = self._coef_slices[i][0] + j
                    b, se_b = self.coef_[k], self.statistics_["se"][k]
                    tb = b / max(se_b, 1e-12)
                    pb = float(2.0 * stats.t.sf(np.abs(tb), df=df_res))
                    lvl_str = f"{level:g}" if isinstance(level, (int, float, np.number)) else str(level)
                    param_rows.append((f"f({t.feature}){lvl_str}", b, se_b, tb, pb))
            if "LinearTerm" in tname:
                start, _ = self._coef_slices[i]
                b = self.coef_[start]
                se_b = self.statistics_["se"][start]
                tb = b / max(se_b, 1e-12)
                pb = float(2.0 * stats.t.sf(np.abs(tb), df=df_res))
                param_rows.append((f"l({t.feature})", b, se_b, tb, pb))

        if param_rows:
            lines.append("Parametric coefficients:")
            lines.append("            Estimate Std. Error t value Pr(>|t|)")
            for name, b, se_b, tb, pb in param_rows:
                lines.append(
                    f"{name:11s} {b:10.4f} {se_b:10.4f} {tb:7.3f} {pb:8.4g}"
                )
            lines.append("")

        smooth_rows = []
        for i, t in enumerate(self._terms):
            tname = type(t).__name__
            if "SplineTerm" in tname or "TensorTerm" in tname:
                by_str = f",by={t.by}" if getattr(t, "by", None) is not None else ""
                if "SplineTerm" in tname:
                    label = f"s({t.feature}{by_str})"
                else:
                    kind = getattr(t, "kind", "te")
                    label = f"{kind}({','.join(str(f) for f in t.features)}{by_str})"
                smooth_rows.append(
                    (
                        label,
                        self.statistics_["edf_per_term"][i],
                        self.statistics_["ref_df_per_term"][i],
                        self.statistics_["f_stat_per_term"][i],
                        self.statistics_["p_values_per_term"][i],
                    )
                )

        if smooth_rows:
            lines.append("Approximate significance of smooth terms:")
            lines.append("        edf Ref.df      F p-value")
            for label, edf, ref_df, f_stat, pval in smooth_rows:
                lines.append(
                    f"{label:7s} {edf:6.2f} {ref_df:6.3f} {f_stat:6.3f} {pval:8.4g}"
                )
            lines.append("")

        lines.extend(
            [
                f"R-sq.(adj) = {self.statistics_['r_sq_adj']:.4f}   Deviance explained = {self.statistics_['deviance_explained'] * 100:.1f}%",
                f"{score_name} = {score_val:10.4f}  Scale est. = {self.statistics_['scale']:.4f}   n = {n}",
                "",
            ]
        )
        return "\n".join(lines)

    def _diagnostics_title(self) -> str:
        return f"{type(self).__name__} Diagnostics: {self.formula}"

    def _overall_plot_title(self) -> str:
        s = self.statistics_
        if self._scale_is_var:
            return f"GAM Fit (Score={s['score']:.3f}, EDF={s['edof']:.1f}, n={s['n_samples']})"
        constrained = [
            f"s({t.feature})={t.constraint}"
            for t in self._terms
            if getattr(t, "constraint", None)
        ]
        c_info = f"  constraints: {', '.join(constrained)}" if constrained else ""
        return f"GAM Fit  (GCV={s['GCV']:.3f}, EDF={s['edof']:.1f}, n={s['n_samples']}){c_info}"

    def _plot_gamcore_decomposition(
        self,
        x: ArrayLike,
        y: ArrayLike,
        n_grid: int = 200,
        extrap_frac: float = 0.0,
        figsize: Optional[Tuple[int, int]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[Any, np.ndarray]:
        import matplotlib.pyplot as plt

        self._check_fitted()
        x_arr = np.asarray(x)
        y_arr = np.asarray(y, dtype=float).ravel()
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        def _fname(idx: int) -> str:
            if feature_names is not None and idx < len(feature_names):
                return feature_names[idx]
            return f"x[{idx}]"

        p_predictors = x_arr.shape[1]
        n_terms = len(self._terms)
        n_term_cols = min(3, n_terms)
        n_term_rows = int(np.ceil(n_terms / n_term_cols)) if n_terms > 0 else 0
        n_rows = 1 + n_term_rows
        n_cols = max(n_term_cols, 1)

        fig = plt.figure(figsize=figsize or (n_cols * 5, n_rows * 4))
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.35)
        ax_overall = fig.add_subplot(gs[0, :])

        medians_list = []
        feat_mins = []
        feat_maxs = []
        feat_grids = []
        for j in range(p_predictors):
            col = x_arr[:, j]
            try:
                col_f = np.asarray(col, dtype=float)
                c_min = float(np.min(col_f))
                c_max = float(np.max(col_f))
                c_med = float(np.median(col_f))
                c_rng = (c_max - c_min) or 1.0
                grid = np.linspace(
                    c_min - extrap_frac * c_rng,
                    c_max + extrap_frac * c_rng,
                    n_grid,
                )
            except (ValueError, TypeError):
                c_min, c_max, c_med = 0.0, 1.0, col[0]
                grid = np.linspace(0.0, 1.0, n_grid)
            feat_mins.append(c_min)
            feat_maxs.append(c_max)
            medians_list.append(c_med)
            feat_grids.append(grid)
        medians = np.array(medians_list, dtype=object)

        first_term = self._terms[0] if self._terms else None
        if first_term is not None:
            first_feat = getattr(first_term, "feature", getattr(first_term, "features", [0])[0])
            first_feat = first_feat if first_feat < p_predictors else 0
        else:
            first_feat = 0

        x_overall = np.tile(medians, (n_grid, 1))
        x_overall[:, first_feat] = feat_grids[first_feat]
        y_overall = self.predict(x_overall)
        sort_idx = np.argsort(x_overall[:, first_feat])
        ax_overall.plot(
            x_overall[sort_idx, first_feat], y_overall[sort_idx], "C0-", lw=2, label="GAM fit"
        )
        ax_overall.scatter(
            x_arr[:, first_feat], y_arr, alpha=0.4, color="gray", s=10, label="Data"
        )
        if extrap_frac > 0:
            ax_overall.axvspan(feat_grids[first_feat][0], feat_mins[first_feat], alpha=0.06, color="red")
            ax_overall.axvspan(
                feat_maxs[first_feat],
                feat_grids[first_feat][-1],
                alpha=0.06,
                color="red",
                label="Extrapolated",
            )
        ax_overall.set(
            xlabel=_fname(first_feat) if p_predictors > 1 else "x",
            ylabel="y",
            title=self._overall_plot_title(),
        )
        ax_overall.legend(fontsize=7, loc="best")

        for i, term in enumerate(self._terms):
            ax = fig.add_subplot(gs[1 + i // n_term_cols, i % n_term_cols])
            tname = type(term).__name__

            if "SplineTerm" in tname:
                x_sweep = np.tile(medians, (n_grid, 1))
                fv = feat_grids[term.feature]
                x_sweep[:, term.feature] = fv
                pdep = self.partial_dependence(i, x_sweep)
                cons = getattr(term, "constraint", None)
                label = f"s({term.feature})" + (f" [{cons}]" if cons else "")
                ax.plot(fv, pdep, "C1-", lw=2, label=label)
                if extrap_frac > 0:
                    ax.axvspan(fv[0], feat_mins[term.feature], alpha=0.06, color="red")
                    ax.axvspan(
                        feat_maxs[term.feature], fv[-1], alpha=0.06, color="red"
                    )
                ax.axhline(0, color="gray", ls="--", lw=0.5)
                if hasattr(term, "n_splines"):
                    title = (
                        f"s({term.feature})   K={term.n_splines}, lam={term.lam:.3f}"
                        + (f"  [{cons}]" if cons else "")
                    )
                else:
                    sp_val = term.sp[0] if term.sp else 0.0
                    title = f"s({term.feature})  k={term.k}, sp={sp_val:.3g}"
                ax.set(xlabel=_fname(term.feature), ylabel="Contribution", title=title)
                ax.legend(fontsize=7)

            elif "TensorTerm" in tname:
                if term.d == 2:
                    ng2d = min(n_grid, 60 if self._scale_is_var else 75)
                    f0_idx, f1_idx = term.features[0], term.features[1]
                    g0 = np.linspace(
                        feat_grids[f0_idx][0], feat_grids[f0_idx][-1], ng2d
                    )
                    g1 = np.linspace(
                        feat_grids[f1_idx][0], feat_grids[f1_idx][-1], ng2d
                    )
                    f0, f1 = np.meshgrid(g0, g1, indexing="ij")
                    x_2d = np.tile(medians, (ng2d * ng2d, 1))
                    x_2d[:, f0_idx] = f0.ravel()
                    x_2d[:, f1_idx] = f1.ravel()
                    pdep_2d = self.partial_dependence(i, x_2d).reshape(ng2d, ng2d)
                    im = ax.pcolormesh(f0, f1, pdep_2d, shading="auto", cmap="RdBu_r")
                    plt.colorbar(im, ax=ax, label="Contribution")
                    ax.set(xlabel=_fname(f0_idx), ylabel=_fname(f1_idx))
                else:
                    x_sweep = np.tile(medians, (n_grid, 1))
                    fv = feat_grids[term.features[0]]
                    x_sweep[:, term.features[0]] = fv
                    ax.plot(fv, self.partial_dependence(i, x_sweep), "C2-", lw=1.5)
                    ax.set_xlabel(f"{_fname(term.features[0])} (projection)")
                feats_str = ",".join(str(f) for f in term.features)
                kind = getattr(term, "kind", "te")
                if hasattr(term, "n_splines_per"):
                    ks_str = ",".join(str(k) for k in term.n_splines_per)
                    ax.set_title(
                        f"{kind}({feats_str})   K=({ks_str})   {term.n_coefs} coefs"
                    )
                else:
                    ax.set_title(f"{kind}({feats_str})")

            elif "FactorTerm" in tname:
                unique_lvls = np.unique(np.asarray(x_arr[:, term.feature]).ravel())
                contribs = []
                for lvl in unique_lvls:
                    x_lvl = medians.copy().reshape(1, -1)
                    x_lvl[0, term.feature] = lvl
                    contribs.append(self.partial_dependence(i, x_lvl)[0])
                colors = ["C2" if v >= 0 else "C3" for v in contribs]
                ax.bar(
                    range(len(contribs)),
                    contribs,
                    color=colors,
                    edgecolor="k",
                    lw=0.5,
                )
                ax.axhline(0, color="gray", ls="-", lw=0.5)
                ax.set_xticks(range(len(contribs)))
                ax.set_xticklabels(unique_lvls, fontsize=8)
                title = (
                    f"f({term.feature})"
                    if self._scale_is_var
                    else f"f({term.feature})   lam={term.lam:.3f}"
                )
                ax.set(ylabel="Contribution", title=title)

            elif "LinearTerm" in tname:
                x_sweep = np.tile(medians, (n_grid, 1))
                fv = feat_grids[term.feature]
                x_sweep[:, term.feature] = fv
                beta = self.coef_[self._coef_slices[i][0]]
                ax.plot(fv, self.partial_dependence(i, x_sweep), "C4-", lw=2)
                if extrap_frac > 0:
                    ax.axvspan(fv[0], feat_mins[term.feature], alpha=0.06, color="red")
                    ax.axvspan(
                        feat_maxs[term.feature], fv[-1], alpha=0.06, color="red"
                    )
                ax.axhline(0, color="gray", ls="--", lw=0.5)
                ax.set(
                    xlabel=_fname(term.feature),
                    ylabel="Contribution",
                    title=f"l({term.feature})   beta={beta:.4f}",
                )

        fig.suptitle(
            f"GAMCore Decomposition: {self.formula}",
            fontsize=12,
            fontweight="bold",
        )
        return fig, np.array(fig.axes)

    def _plot_lingam_decomposition(
        self,
        x: ArrayLike,
        y: ArrayLike,
        k_smooth: int,
        prefix: str,
        basis_title: str,
        fit_title: str,
        suptitle: str,
        knots: Optional[np.ndarray] = None,
        show_weighted: bool = False,
        n_grid: int = 300,
        figsize: Tuple[int, int] = (16, 5),
    ) -> Tuple[Any, np.ndarray]:
        import matplotlib.pyplot as plt

        self._check_fitted()
        x_arr = np.asarray(x, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()

        x_grid = np.linspace(np.min(x_arr), np.max(x_arr), n_grid)
        X_grid = self._build_model_matrix(x_grid.reshape(-1, 1))
        B_grid = X_grid[:, :k_smooth]
        y_pred = X_grid @ self.coef_

        cmap = plt.get_cmap(
            "tab20" if k_smooth <= 20 else "tab10" if k_smooth <= 10 else "hsv"
        )
        colors = [
            cmap(i % 20 if k_smooth <= 20 else i / k_smooth)
            for i in range(k_smooth)
        ]

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        ax1 = axes[0]
        for j in range(k_smooth):
            ax1.plot(
                x_grid,
                B_grid[:, j],
                color=colors[j],
                ls="-",
                alpha=0.8,
                label=f"{prefix}_{j}",
            )
        if knots is not None and show_weighted:
            for i_knot, knot in enumerate(knots):
                ax1.axvline(
                    knot,
                    color="gray",
                    ls="--",
                    lw=1,
                    label="Knots" if i_knot == 0 else None,
                )
            ax1.minorticks_on()
        ax1.set(xlabel="x", ylabel="B_j(x)", title=basis_title)
        if show_weighted or k_smooth <= 12:
            ax1.legend(fontsize=6 if show_weighted else 7, loc="upper right")

        ax2 = axes[1]
        vals = np.append(self.coef_[:k_smooth], self.coef_[-1])
        labels = [f"{prefix}_{j}" for j in range(k_smooth)] + ["intercept"]
        ax2.bar(range(len(vals)), vals, color=colors + ["dimgray"])
        ax2.set_xticks(range(len(vals)))
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax2.axhline(0, color="k")
        ax2.set(ylabel="Coefficient value", title="Estimated Coefficients")

        ax3 = axes[2]
        ax3.scatter(x_arr, y_arr, alpha=0.5, color="gray", s=12, label="Data")
        if show_weighted:
            for j in range(k_smooth):
                ax3.plot(
                    x_grid,
                    self.coef_[j] * B_grid[:, j] + self.coef_[-1],
                    color=colors[j],
                    lw=1,
                    ls="-",
                )
            ax3.plot(x_grid, y_pred, "k-", label="Fit (sum)")
            if knots is not None:
                for knot in knots:
                    ax3.axvline(knot, color="gray", ls="--", lw=1)
            ax3.minorticks_on()
            ax3.legend(fontsize=6, loc="upper left")
        else:
            ax3.plot(x_grid, y_pred, "k-", lw=2, label="GAM fit")
            ax3.legend(fontsize=7, loc="upper left")
        ax3.set(xlabel="x", ylabel="y", title=fit_title)

        fig.suptitle(suptitle)
        return fig, axes

    def plot_diagnostics(
        self,
        x: ArrayLike,
        y: ArrayLike,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Any, np.ndarray]:
        r"""Generate a four-panel diagnostic plot for model validation and residual inspection.

        Creates four diagnostic subplots:
        1. Observed vs. Fitted values with a 1:1 reference line and :math:`R^2`.
        2. Residuals vs. Fitted values checking for homoscedasticity and non-linear drift.
        3. Normal Q-Q plot assessing residual normality.
        4. Residual Distribution histogram with superimposed Gaussian kernel density.

        Parameters
        ----------
        x : array_like of shape (n_samples, n_features) or (n_samples,)
            Feature data corresponding to observations `y`.
        y : array_like of shape (n_samples,)
            True observed target values.
        figsize : tuple of (int, int), optional
            Figure dimension `(width, height)` in inches. Default is `(10, 9)`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure object.
        axes : ndarray of matplotlib.axes.Axes of shape (4,)
            Flat array containing the 4 subplot axis objects.

        See Also
        --------
        plot_decomposition : Visualize individual additive term contributions.
        """
        import matplotlib.pyplot as plt

        self._check_fitted()
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        y_fit = self.predict(x_arr)
        residuals = y_arr - y_fit
        r2 = self.deviance_explained()

        fig, axes = plt.subplots(2, 2, figsize=figsize or (10, 9))
        axes = axes.ravel()

        ax = axes[0]
        ax.scatter(y_fit, y_arr, alpha=0.5, edgecolors="k", s=20)
        lims = [min(y_arr.min(), y_fit.min()), max(y_arr.max(), y_fit.max())]
        ax.plot(lims, lims, "r--", lw=1.5, label="1:1")
        ax.set(
            xlabel="Fitted values",
            ylabel="Observed values",
            title=f"Observed vs Fitted ($R^2$={r2:.3f})",
        )
        ax.legend()

        ax = axes[1]
        ax.scatter(y_fit, residuals, alpha=0.5, edgecolors="k", s=20)
        ax.axhline(0, color="r", linestyle="--", lw=1.5)
        ax.set(xlabel="Fitted values", ylabel="Residuals", title="Residuals vs Fitted")

        ax = axes[2]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("Normal Q-Q")
        ax.get_lines()[0].set_markerfacecolor("steelblue")
        ax.get_lines()[0].set_markeredgecolor("k")
        ax.get_lines()[0].set_alpha(0.6)
        ax.get_lines()[1].set_color("r")

        ax = axes[3]
        ax.hist(
            residuals,
            bins="auto",
            density=True,
            color="steelblue",
            edgecolor="k",
            alpha=0.7,
        )
        sigma = np.std(residuals, ddof=1)
        x_norm = np.linspace(residuals.min(), residuals.max(), 200)
        ax.plot(
            x_norm,
            stats.norm.pdf(x_norm, 0.0, sigma),
            "r-",
            lw=2,
            label="Normal approx.",
        )
        ax.set(xlabel="Residuals", ylabel="Density", title="Residual Distribution")
        ax.legend()

        fig.suptitle(self._diagnostics_title(), fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        return fig, axes
