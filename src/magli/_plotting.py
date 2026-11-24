"""Plotting helpers for GAM decomposition and residual diagnostics."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def _import_pyplot() -> Any:
    """Import matplotlib.pyplot lazily with a helpful install hint."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install magli[plot]"
        ) from exc
    return plt


def _profile_predictors(
    x_arr: np.ndarray, n_grid: int, extrap_frac: float
) -> Tuple[np.ndarray, List[float], List[float], List[np.ndarray], List[bool]]:
    """Compute medians, bounds, grids, and numeric flags for each predictor column."""
    medians_list, feat_mins, feat_maxs, feat_grids, is_numeric = [], [], [], [], []
    for j in range(x_arr.shape[1]):
        col = x_arr[:, j]
        try:
            col_f = np.asarray(col, dtype=float)
            c_min, c_max, c_med = float(np.min(col_f)), float(np.max(col_f)), float(np.median(col_f))
            c_rng = (c_max - c_min) or 1.0
            grid = np.linspace(c_min - extrap_frac * c_rng, c_max + extrap_frac * c_rng, n_grid)
            is_numeric.append(True)
        except (ValueError, TypeError):
            c_min, c_max, c_med = 0.0, 1.0, col[0]
            grid = np.linspace(0.0, 1.0, n_grid)
            is_numeric.append(False)
        feat_mins.append(c_min)
        feat_maxs.append(c_max)
        medians_list.append(c_med)
        feat_grids.append(grid)
    return np.array(medians_list, dtype=object), feat_mins, feat_maxs, feat_grids, is_numeric


def _shade_extrap(
    ax: Any, grid: np.ndarray, fmin: float, fmax: float, label: Optional[str] = None
) -> None:
    ax.axvspan(grid[0], fmin, alpha=0.06, color="red")
    ax.axvspan(fmax, grid[-1], alpha=0.06, color="red", label=label)


def plot_gamcore_decomposition(
    model: Any,
    x: ArrayLike,
    y: ArrayLike,
    n_grid: int = 200,
    extrap_frac: float = 0.0,
    figsize: Optional[Tuple[int, int]] = None,
    feature_names: Optional[List[str]] = None,
) -> Tuple[Any, np.ndarray]:
    """Plot overall fit and per-term partial dependence panels for a fitted GAM."""
    plt = _import_pyplot()
    model._check_fitted()
    x_arr = np.asarray(x)
    y_arr = np.asarray(y, dtype=float).ravel()
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)

    def _fname(idx: int) -> str:
        return feature_names[idx] if feature_names is not None and idx < len(feature_names) else f"x[{idx}]"

    p_predictors = x_arr.shape[1]
    n_terms = len(model._terms)
    n_term_cols = min(3, n_terms)
    n_rows = 1 + (int(np.ceil(n_terms / n_term_cols)) if n_terms > 0 else 0)
    n_cols = max(n_term_cols, 1)

    fig = plt.figure(figsize=figsize or (n_cols * 5, n_rows * 4))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.35)
    medians, feat_mins, feat_maxs, feat_grids, is_numeric = _profile_predictors(
        x_arr, n_grid, extrap_frac
    )

    ax_overall = fig.add_subplot(gs[0, :])
    term_feats = [
        f for t in model._terms
        for f in (getattr(t, "features", None) or [getattr(t, "feature", 0)])
        if f < p_predictors
    ]
    first_feat = next(
        (f for f in term_feats if is_numeric[f]),
        next((j for j, num in enumerate(is_numeric) if num), term_feats[0] if term_feats else 0),
    )
    if is_numeric[first_feat]:
        grid = feat_grids[first_feat]
        x_overall = np.tile(medians, (len(grid), 1))
        x_overall[:, first_feat] = grid
        y_overall = model.predict(x_overall)
        sort_idx = np.argsort(x_overall[:, first_feat])
        ax_overall.plot(x_overall[sort_idx, first_feat], y_overall[sort_idx], "C0-", lw=2, label="GAM fit")
        ax_overall.scatter(np.asarray(x_arr[:, first_feat], dtype=float), y_arr, alpha=0.4, color="gray", s=10, label="Data")
        if extrap_frac > 0:
            _shade_extrap(ax_overall, grid, feat_mins[first_feat], feat_maxs[first_feat], label="Extrapolated")
    else:
        unique_lvls = np.unique(np.asarray(x_arr[:, first_feat]).ravel())
        lvl_pos = {lvl: idx for idx, lvl in enumerate(unique_lvls)}
        x_lvls = np.tile(medians, (len(unique_lvls), 1))
        x_lvls[:, first_feat] = unique_lvls
        ax_overall.scatter([lvl_pos[v] for v in np.asarray(x_arr[:, first_feat]).ravel()], y_arr, alpha=0.4, color="gray", s=10, label="Data")
        ax_overall.plot(range(len(unique_lvls)), model.predict(x_lvls), "C0o-", lw=2, label="GAM fit")
        ax_overall.set_xticks(range(len(unique_lvls)))
        ax_overall.set_xticklabels(unique_lvls, fontsize=8)
    ax_overall.set(xlabel=_fname(first_feat) if p_predictors > 1 else "x", ylabel="y", title=model._overall_plot_title())
    ax_overall.legend(fontsize=7, loc="best")

    for i, term in enumerate(model._terms):
        ax = fig.add_subplot(gs[1 + i // n_term_cols, i % n_term_cols])
        tname = type(term).__name__
        if "SplineTerm" in tname and getattr(term, "bs", None) != "re":
            fv = feat_grids[term.feature]
            x_sweep = np.tile(medians, (n_grid, 1))
            x_sweep[:, term.feature] = fv
            cons = getattr(term, "constraint", None)
            ax.plot(fv, model.partial_dependence(i, x_sweep), "C1-", lw=2, label=f"s({term.feature})" + (f" [{cons}]" if cons else ""))
            if extrap_frac > 0:
                _shade_extrap(ax, fv, feat_mins[term.feature], feat_maxs[term.feature])
            ax.axhline(0, color="gray", ls="--", lw=0.5)
            if hasattr(term, "n_splines"):
                title = f"s({term.feature})   K={term.n_splines}, lam={term.lam:.3f}" + (f"  [{cons}]" if cons else "")
            else:
                title = f"s({term.feature})  k={term.k}, sp={(term.sp[0] if term.sp else 0.0):.3g}"
            ax.set(xlabel=_fname(term.feature), ylabel="Contribution", title=title)
            ax.legend(fontsize=7)
        elif "TensorTerm" in tname:
            if term.d == 2:
                ng2d = min(n_grid, 60 if model._scale_is_var else 75)
                f0_idx, f1_idx = term.features[0], term.features[1]
                g0 = np.linspace(feat_grids[f0_idx][0], feat_grids[f0_idx][-1], ng2d)
                g1 = np.linspace(feat_grids[f1_idx][0], feat_grids[f1_idx][-1], ng2d)
                f0, f1 = np.meshgrid(g0, g1, indexing="ij")
                x_2d = np.tile(medians, (ng2d * ng2d, 1))
                x_2d[:, f0_idx], x_2d[:, f1_idx] = f0.ravel(), f1.ravel()
                im = ax.pcolormesh(f0, f1, model.partial_dependence(i, x_2d).reshape(ng2d, ng2d), shading="auto", cmap="RdBu_r")
                plt.colorbar(im, ax=ax, label="Contribution")
                ax.set(xlabel=_fname(f0_idx), ylabel=_fname(f1_idx))
            else:
                fv = feat_grids[term.features[0]]
                x_sweep = np.tile(medians, (n_grid, 1))
                x_sweep[:, term.features[0]] = fv
                ax.plot(fv, model.partial_dependence(i, x_sweep), "C2-", lw=1.5)
                ax.set_xlabel(f"{_fname(term.features[0])} (projection)")
            feats_str = ",".join(str(f) for f in term.features)
            kind = getattr(term, "kind", "te")
            title = (
                f"{kind}({feats_str})   K=({','.join(str(k) for k in term.n_splines_per)})   {term.n_coefs} coefs"
                if hasattr(term, "n_splines_per")
                else f"{kind}({feats_str})"
            )
            ax.set_title(title)
        elif "FactorTerm" in tname or getattr(term, "bs", None) == "re":
            unique_lvls = np.unique(np.asarray(x_arr[:, term.feature]).ravel())
            contribs = []
            for lvl in unique_lvls:
                x_lvl = medians.copy().reshape(1, -1)
                x_lvl[0, term.feature] = lvl
                contribs.append(model.partial_dependence(i, x_lvl)[0])
            ax.bar(range(len(contribs)), contribs, color=["C2" if v >= 0 else "C3" for v in contribs], edgecolor="k", lw=0.5)
            ax.axhline(0, color="gray", ls="-", lw=0.5)
            ax.set_xticks(range(len(contribs)))
            ax.set_xticklabels(unique_lvls, fontsize=8)
            if getattr(term, "bs", None) == "re":
                title = f"s({term.feature}, bs='re')"
            elif model._scale_is_var:
                title = f"f({term.feature})"
            else:
                title = f"f({term.feature})   lam={term.lam:.3f}"
            ax.set(ylabel="Contribution", title=title)
        elif "LinearTerm" in tname:
            fv = feat_grids[term.feature]
            x_sweep = np.tile(medians, (n_grid, 1))
            x_sweep[:, term.feature] = fv
            beta = model.coef_[model._coef_slices[i][0]]
            ax.plot(fv, model.partial_dependence(i, x_sweep), "C4-", lw=2)
            if extrap_frac > 0:
                _shade_extrap(ax, fv, feat_mins[term.feature], feat_maxs[term.feature])
            ax.axhline(0, color="gray", ls="--", lw=0.5)
            ax.set(xlabel=_fname(term.feature), ylabel="Contribution", title=f"l({term.feature})   beta={beta:.4f}")

    fig.suptitle(f"GAMCore Decomposition: {model.formula}", fontsize=12, fontweight="bold")
    return fig, np.array(fig.axes)


def plot_lingam_decomposition(
    model: Any,
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
    """Plot basis functions, coefficient bar chart, and overall fit for a single-term GAM."""
    plt = _import_pyplot()
    model._check_fitted()
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()

    x_grid = np.linspace(np.min(x_arr), np.max(x_arr), n_grid)
    X_grid = model._build_model_matrix(x_grid.reshape(-1, 1))
    B_grid = X_grid[:, :k_smooth]
    y_pred = X_grid @ model.coef_

    cmap = plt.get_cmap(
        "tab10" if k_smooth <= 10 else "tab20" if k_smooth <= 20 else "hsv"
    )
    colors = [
        cmap(i % 10 if k_smooth <= 10 else i % 20 if k_smooth <= 20 else i / k_smooth)
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
    vals = np.append(model.coef_[:k_smooth], model.coef_[-1])
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
                model.coef_[j] * B_grid[:, j] + model.coef_[-1],
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
    model: Any,
    x: ArrayLike,
    y: ArrayLike,
    figsize: Optional[Tuple[int, int]] = None,
) -> Tuple[Any, np.ndarray]:
    """Generate a four-panel residual diagnostic plot for a fitted GAM."""
    plt = _import_pyplot()
    model._check_fitted()
    x_arr = model._reshape_x(x)
    y_arr = np.asarray(y, dtype=float).ravel()

    y_fit = model.predict(x_arr)
    residuals = y_arr - y_fit
    r2 = model.deviance_explained()

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
    sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    if np.isfinite(sigma) and sigma > 0.0:
        x_norm = np.linspace(residuals.min(), residuals.max(), 200)
        ax.plot(
            x_norm,
            stats.norm.pdf(x_norm, 0.0, sigma),
            "r-",
            lw=2,
            label="Normal approx.",
        )
        ax.legend()
    ax.set(xlabel="Residuals", ylabel="Density", title="Residual Distribution")

    fig.suptitle(model._diagnostics_title(), fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig, axes
