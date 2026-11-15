"""API parity and usability tests across Eilers-Marx and Wood backends."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from magli import (
    GAM,
    GAMCore,
    LinGAM,
    EilersGAMCore,
    EilersLinGAM,
    WoodGAMCore,
    WoodLinGAM,
)
from magli.wood import GAMCore as DirectWoodGAMCore
from magli.wood import LinGAM as DirectWoodLinGAM, GAM as DirectWoodGAM
from magli.eilers import GAMCore as DirectEilersGAMCore
from magli.eilers import LinGAM as DirectEilersLinGAM, GAM as DirectEilersGAM


def test_import_and_factory():
    """Verify that imports and backend factories work identically."""
    # Default backend is wood
    assert isinstance(GAM(k=10), WoodLinGAM)
    assert isinstance(LinGAM(k=10), WoodLinGAM)
    assert isinstance(GAMCore("s(0)"), WoodGAMCore)

    # Explicit backend selection ('wood' and 'eilers' / 'eilers')
    m_em = GAM(n_splines=10, backend="eilers")
    assert isinstance(m_em, EilersLinGAM)

    m_wd = GAM(k=10, backend="wood")
    assert isinstance(m_wd, WoodLinGAM)

    m_lingam = LinGAM(n_splines=10, backend="eilers")
    assert isinstance(m_lingam, EilersLinGAM)

    # Direct import tests
    m_direct_wd = DirectWoodGAM(k=10)
    assert isinstance(m_direct_wd, WoodLinGAM)

    m_direct_em = DirectEilersGAM(n_splines=10)
    assert isinstance(m_direct_em, EilersLinGAM)

    g_em = GAMCore("s(0)", backend="eilers")
    assert isinstance(g_em, EilersGAMCore)

    g_wd = GAMCore("s(0)", backend="wood")
    assert isinstance(g_wd, WoodGAMCore)

    g_direct_wd = DirectWoodGAMCore("s(0)")
    assert isinstance(g_direct_wd, WoodGAMCore)

    with pytest.raises(ValueError, match="Unknown backend"):
        LinGAM(backend="invalid_backend")

    with pytest.raises(ValueError, match="Unknown backend"):
        GAMCore("s(0)", backend="invalid_backend")


def test_lingam_lifecycle_and_methods():
    """Verify that both backends provide identical lifecycle methods and attributes."""
    np.random.seed(42)
    x = np.linspace(0, 1, 60)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, 60)
    x_test = np.linspace(0.1, 0.9, 20)

    for backend in ("eilers", "wood"):
        m = LinGAM(n_splines=8, backend=backend)
        assert not m.is_fitted
        assert repr(m).startswith("LinGAM")

        with pytest.raises(RuntimeError):
            m.predict(x_test)

        # Fit
        m.fit(x, y)
        assert m.is_fitted
        assert m.coef_ is not None
        assert m.design_matrix_ is not None

        # Predict
        preds = m.predict(x_test)
        assert preds.shape == (20,)

        # Intervals
        ci = m.confidence_intervals(x_test, width=0.95)
        pi = m.prediction_intervals(x_test, width=0.95)
        assert ci.shape == (20, 2)
        assert pi.shape == (20, 2)
        assert np.all(ci[:, 0] <= ci[:, 1])
        assert np.all(pi[:, 0] <= pi[:, 1])
        # Prediction interval must be wider than confidence interval
        assert np.all((pi[:, 1] - pi[:, 0]) >= (ci[:, 1] - ci[:, 0]))

        # Common statistics
        stats = m.statistics_
        assert isinstance(stats, dict)
        for expected_key in (
            "edof",
            "edof_per_coef",
            "scale",
            "cov",
            "se",
            "rss",
            "n_samples",
            "GCV",
        ):
            assert expected_key in stats, f"Missing key '{expected_key}' in backend '{backend}'"

        # Information criteria and deviance
        ll = m.loglikelihood()
        aic = m.aic()
        bic = m.bic()
        dev_expl = m.deviance_explained()
        assert isinstance(ll, float)
        assert isinstance(aic, float)
        assert isinstance(bic, float)
        assert 0.0 <= dev_expl <= 1.0

        # Plotting
        fig, axes = m.plot_decomposition(x, y)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        fig, axes = m.plot_diagnostics(x, y)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


def test_gamcore_lifecycle_and_methods():
    """Verify that GAMCore formula models work identically across both backends."""
    np.random.seed(42)
    n = 80
    x0 = np.random.uniform(0, 1, n)
    x1 = np.random.uniform(0, 1, n)
    X = np.column_stack([x0, x1])
    y = np.sin(2 * np.pi * x0) + 1.8 * x1 + np.random.normal(0, 0.1, n)
    X_test = np.column_stack([np.linspace(0.1, 0.9, 15), np.linspace(0.2, 0.8, 15)])

    formula = "s(0, n_splines=8) + l(1)"

    for backend in ("eilers", "wood"):
        gam = GAMCore(formula, backend=backend)
        assert not gam.is_fitted
        assert gam.n_terms == 2

        with pytest.raises(RuntimeError):
            gam.predict(X_test)

        gam.fit(X, y)
        assert gam.is_fitted
        assert gam.coef_ is not None

        preds = gam.predict(X_test)
        assert preds.shape == (15,)

        pdep0 = gam.partial_dependence(0, X_test)
        pdep1 = gam.partial_dependence(1, X_test)
        assert pdep0.shape == (15,)
        assert pdep1.shape == (15,)

        ci = gam.confidence_intervals(X_test)
        pi = gam.prediction_intervals(X_test)
        assert ci.shape == (15, 2)
        assert pi.shape == (15, 2)
        assert np.all(ci[:, 0] <= ci[:, 1])
        assert np.all(pi[:, 0] <= pi[:, 1])

        # Plotting
        fig, axes = gam.plot_decomposition(X, y)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        fig, axes = gam.plot_diagnostics(X, y)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


def test_gridsearch_both_backends():
    """Verify that gridsearch works on both backends."""
    np.random.seed(42)
    x = np.linspace(0, 1, 50)
    y = np.cos(2 * np.pi * x) + np.random.normal(0, 0.1, 50)

    # Eilers-Marx grid search
    m_em = LinGAM(backend="eilers")
    m_em.gridsearch(x, y, n_splines=[6, 8, 10], lam=[0.01, 0.1, 1.0])
    assert m_em.is_fitted
    assert m_em.n_splines in [6, 8, 10]

    # Wood grid search
    m_wd = LinGAM(backend="wood")
    m_wd.gridsearch(x, y, n_splines=[6, 8, 10])
    assert m_wd.is_fitted
    assert m_wd.n_splines in [6, 8, 10]

    X2 = np.column_stack([x, np.random.uniform(0, 1, 50)])
    g_wd = GAMCore("s(0, bs='cr', k=10) + s(1, bs='cr', k=10)", backend="wood")
    g_wd.gridsearch(X2, y, n_splines=[6, 8], search_terms=[0])
    assert g_wd.is_fitted
    assert g_wd._terms[0].k in [6, 8]
    assert g_wd._terms[1].k == 10


def test_weights_both_backends():
    """Verify observation weights across fit() and gridsearch() on both backends."""
    np.random.seed(42)
    x = np.linspace(0, 10, 150)
    y_clean = np.sin(x)
    y = y_clean + np.random.normal(0, 0.15, len(x))
    outlier_idx = [20, 75, 120]
    y[20] += 8.0
    y[75] -= 8.0
    y[120] += 7.0

    w = np.ones_like(y)
    w[outlier_idx] = 1e-4

    for backend in ("eilers", "wood"):
        # Unit weights match unweighted fit
        m_unw = LinGAM(n_splines=14, backend=backend).fit(x, y)
        m_ones = LinGAM(n_splines=14, backend=backend).fit(
            x, y, weights=np.ones_like(y)
        )
        assert np.allclose(m_unw.coef_, m_ones.coef_)

        # Downweighting outliers recovers clean curve
        m_w = LinGAM(n_splines=14, backend=backend).fit(x, y, weights=w)
        rmse_unw = np.sqrt(np.mean((m_unw.predict(x) - y_clean) ** 2))
        rmse_w = np.sqrt(np.mean((m_w.predict(x) - y_clean) ** 2))
        assert rmse_w < rmse_unw * 0.6

        g_unw = GAMCore("s(0, k=14)", backend=backend).fit(x, y)
        g_w = GAMCore("s(0, k=14)", backend=backend).fit(x, y, weights=w)
        assert np.sqrt(np.mean((g_w.predict(x) - y_clean) ** 2)) < np.sqrt(
            np.mean((g_unw.predict(x) - y_clean) ** 2)
        ) * 0.6

        # Gridsearch with weights
        gs_w = LinGAM(backend=backend).gridsearch(
            x, y, n_splines=[10, 14], weights=w
        )
        assert np.sqrt(np.mean((gs_w.predict(x) - y_clean) ** 2)) < rmse_unw * 0.6

        # Invalid weights validation
        with pytest.raises(ValueError, match="weights length"):
            LinGAM(backend=backend).fit(x, y, weights=np.ones(5))
        with pytest.raises(ValueError, match="weights must be finite"):
            LinGAM(backend=backend).fit(x, y, weights=-np.ones_like(y))
        with pytest.raises(ValueError, match="weights must be finite"):
            LinGAM(backend=backend).fit(x, y, weights=np.zeros_like(y))


def test_audit_fixes_comprehensive():
    """Verify bug fixes for fit_intercept=False, aliases, scalar eval, 1D pdep, formula args, strings."""
    rng = np.random.default_rng(42)
    x = np.linspace(0.1, 1.0, 50)
    y = 2.0 * x

    # 1. fit_intercept=False on non-zero mean data
    for backend in ("wood", "eilers"):
        m = GAMCore("s(0, k=8)", fit_intercept=False, backend=backend).fit(x, y)
        assert m.is_fitted
        assert m.deviance_explained() > 0.99
        assert np.allclose(m.predict(x), y, atol=1e-2)

    # 2. Documented constraint aliases 'monotonic_inc' and 'monotonic_dec'
    y_noisy = np.exp(x) + rng.normal(0, 0.05, len(x))
    for c_name in ("monotonic_inc", "mono_inc"):
        m_inc = LinGAM(n_splines=10, constraint=c_name, backend="eilers").fit(x, y_noisy)
        preds = m_inc.predict(np.linspace(0.1, 1.0, 100))
        assert np.all(np.diff(preds) >= -1e-6)
    for c_name in ("monotonic_dec", "mono_dec"):
        m_dec = LinGAM(n_splines=10, constraint=c_name, backend="eilers").fit(x, -y_noisy)
        preds = m_dec.predict(np.linspace(0.1, 1.0, 100))
        assert np.all(np.diff(preds) <= 1e-6)

    # 3. Scalar inputs to predict and intervals
    m_single = LinGAM(k=8, backend="wood").fit(x, y)
    assert m_single.predict(0.5).shape == (1,)
    assert m_single.confidence_intervals(0.5).shape == (1, 2)
    assert m_single.prediction_intervals(0.5).shape == (1, 2)

    # 4. Multi-feature 1-row predict and 1D partial_dependence
    X_2d = np.column_stack([x, rng.uniform(0, 1, len(x))])
    y_2d = 2.0 * x + 3.0 * X_2d[:, 1]
    for backend in ("wood", "eilers"):
        m_2d = GAMCore("s(0, k=8) + s(1, k=8)", backend=backend).fit(X_2d, y_2d)
        assert m_2d.predict([0.5, 0.2]).shape == (1,)
        grid_1d = np.linspace(0, 1, 25)
        pdep0 = m_2d.partial_dependence(0, grid_1d)
        pdep1 = m_2d.partial_dependence(1, grid_1d)
        assert pdep0.shape == (25,)
        assert pdep1.shape == (25,)

    # 5. Formula parser rejects extra positional arguments
    with pytest.raises(ValueError, match="accepts only 1 positional"):
        GAMCore("s(0, 10)")
    with pytest.raises(ValueError, match="accepts only 1 positional"):
        GAMCore("l(0, 1)")

    # 6. Categorical string factor variables & NaN validation with string columns
    X_str = np.array([["ctrl", 0.1], ["treat", 0.5], ["ctrl", 0.2], ["treat", 0.6], ["treat", 0.7]], dtype=object)
    y_str = np.array([1.0, 3.0, 1.2, 2.9, 3.2])
    for backend in ("wood", "eilers"):
        m_cat = GAMCore("f(0) + s(1, k=4)", backend=backend).fit(X_str, y_str)
        assert m_cat.is_fitted
        summary_str = m_cat.summary()
        assert "f(0)" in summary_str

        # NaN validation in presence of string column
        X_nan = X_str.copy()
        X_nan[0, 1] = np.nan
        with pytest.raises(ValueError, match="contains NaN or infinity"):
            GAMCore("f(0) + s(1, k=4)", backend=backend).fit(X_nan, y_str)

    # 7. Thin-plate spline knot capping > 2000 points
    from magli.wood._smooths import build_tp_basis
    x_large = np.linspace(0, 1, 2500)
    X_tp, S_tp, info_tp = build_tp_basis(x_large, k=10)
    assert X_tp.shape == (2500, 10)
    assert len(info_tp["Xu"]) == 2000


def test_audit_regressions_fixed():
    """Verify fixes for all bugs and validation gaps surfaced by the repo audit."""
    from magli._formula import parse_formula

    rng = np.random.default_rng(7)

    # 1. String categorical f(0) works end-to-end across both backends
    X_str = np.array(
        [["ctrl", 0.1], ["treat", 0.5], ["ctrl", 0.2], ["treat", 0.6], ["treat", 0.7], ["ctrl", 0.3], ["ctrl", 0.4], ["treat", 0.8]],
        dtype=object,
    )
    y_str = np.array([1.0, 3.0, 1.2, 2.9, 3.2, 1.1, 1.3, 3.1])
    for backend in ("wood", "eilers"):
        m = GAMCore("f(0) + s(1, k=4)", backend=backend).fit(X_str, y_str)
        assert m.predict(X_str).shape == (len(y_str),)
        assert m.confidence_intervals(X_str).shape == (len(y_str), 2)
        assert m.prediction_intervals(X_str).shape == (len(y_str), 2)
        assert m.partial_dependence(0, X_str).shape == (len(y_str),)
        fig_d, _ = m.plot_decomposition(X_str, y_str)
        plt.close(fig_d)
        fig_g, _ = m.plot_diagnostics(X_str, y_str)
        plt.close(fig_g)
        m_gs = GAMCore("f(0) + s(1, k=4)", backend=backend)
        if backend == "wood":
            m_gs.gridsearch(X_str, y_str, n_splines=[4, 5])
        else:
            m_gs.gridsearch(X_str, y_str, lam=[0.1, 1.0])
        assert m_gs.is_fitted and m_gs.predict(X_str).shape == (len(y_str),)

    # 2. String group labels in s(0, bs='re') and bar-chart plot_decomposition
    m_re = GAMCore("s(0, bs='re') + s(1, k=4)", backend="wood").fit(X_str, y_str)
    assert m_re.predict(X_str).shape == (len(y_str),)
    fig_re, axes_re = m_re.plot_decomposition(X_str, y_str)
    assert len(axes_re[1].patches) == 2
    plt.close(fig_re)

    # 3. Re-fitting wood.GAMCore re-optimizes free smoothing parameters
    X_refit = rng.uniform(0, 1, (80, 1))
    y_noisy = np.sin(2 * np.pi * X_refit[:, 0]) + rng.normal(0, 0.5, 80)
    y_clean = np.sin(2 * np.pi * X_refit[:, 0]) + rng.normal(0, 0.01, 80)
    g_refit = GAMCore("s(0, k=10)", backend="wood").fit(X_refit, y_noisy)
    sp_noisy = g_refit.statistics_["sp"][0]
    g_refit.fit(X_refit, y_clean)
    assert g_refit.statistics_["sp"][0] < sp_noisy * 0.05

    # 4. te() defaults (k=5, bs='cr') and tuple k=(4, 6) in wood backend
    g_te_def = GAMCore("te(0, 1)", backend="wood")
    assert g_te_def._terms[0].k_per == [5, 5]
    assert g_te_def._terms[0].bs_per == ["cr", "cr"]
    g_te_tup = GAMCore("te(0, 1, k=(4, 6))", backend="wood")
    assert g_te_tup._terms[0].k_per == [4, 6]

    # 5. Invalid factor coding and wood one-hot intercept collinearity guard
    for backend in ("wood", "eilers"):
        with pytest.raises(ValueError, match="Unknown coding"):
            GAMCore("f(0, coding='invalid')", backend=backend)
    with pytest.raises(ValueError, match="collinear with the intercept"):
        GAMCore("f(0, coding='one-hot')", backend="wood", fit_intercept=True)
    m_oh = GAMCore("f(0, coding='one-hot')", backend="wood", fit_intercept=False).fit(X_str, y_str)
    assert m_oh.is_fitted and np.all(m_oh.statistics_["se"] < 1.0)

    # 6. k/sp constructor aliases and properties on eilers LinGAM
    m_e = GAM(k=8, sp=0.5, backend="eilers")
    assert isinstance(m_e, EilersLinGAM)
    assert m_e.k == 8 and m_e.sp == 0.5
    m_e.k = 9
    m_e.sp = 0.25
    assert m_e.n_splines == 9 and m_e.lam == 0.25

    # 7. Documented fitted properties edf_, scale_, intercept_
    x_1d = np.linspace(0, 1, 40)
    y_1d = np.sin(2 * np.pi * x_1d) + rng.normal(0, 0.1, 40)
    for backend in ("wood", "eilers"):
        m_lin = LinGAM(n_splines=8, backend=backend).fit(x_1d, y_1d)
        assert np.isclose(m_lin.edf_, m_lin.statistics_["edof"])
        assert np.isclose(m_lin.scale_, m_lin.statistics_["scale"])
        assert np.isclose(m_lin.intercept_, m_lin.coef_[-1])
        m_no_int = GAMCore("s(0, k=8)", fit_intercept=False, backend=backend).fit(x_1d, y_1d)
        assert m_no_int.intercept_ == 0.0

    # 8. gridsearch validates NaN/inf and x/y length mismatch on both backends
    x_bad = x_1d.copy()
    x_bad[3] = np.nan
    y_bad = y_1d.copy()
    y_bad[3] = np.inf
    for backend in ("wood", "eilers"):
        with pytest.raises(ValueError, match="NaN or infinity"):
            LinGAM(backend=backend).gridsearch(x_bad, y_1d)
        with pytest.raises(ValueError, match="NaN or infinity"):
            LinGAM(backend=backend).gridsearch(x_1d, y_bad)
        with pytest.raises(ValueError, match="must match y length"):
            LinGAM(backend=backend).gridsearch(x_1d[:10], y_1d)

    # 9. Negative sp / lam rejected in GAMCore formulas on both backends
    with pytest.raises(ValueError, match="non-negative"):
        GAMCore("s(0, sp=-1.0)", backend="wood")
    with pytest.raises(ValueError, match="non-negative"):
        GAMCore("s(0, lam=-1.0)", backend="eilers")

    # 10. Interval width validation
    m_iv = LinGAM(k=6, backend="wood").fit(x_1d, y_1d)
    for bad_w in (0.0, 1.0, 95.0, -0.5):
        with pytest.raises(ValueError, match="width"):
            m_iv.confidence_intervals(x_1d, width=bad_w)
        with pytest.raises(ValueError, match="width"):
            m_iv.prediction_intervals(x_1d, width=bad_w)

    # 11. partial_dependence on 1D single-row input for te() and s(..., by=...)
    X_2d = np.column_stack([x_1d, rng.uniform(0.2, 1.0, len(x_1d))])
    y_2d = np.sin(2 * np.pi * X_2d[:, 0]) * X_2d[:, 1]
    m_te_by = GAMCore("te(0, 1, k=4) + s(0, by=1, k=5)", backend="wood").fit(X_2d, y_2d)
    assert m_te_by.partial_dependence(0, [0.5, 0.4]).shape == (1,)
    assert m_te_by.partial_dependence(1, [0.5, 0.4]).shape == (1,)

    # 12. parse_formula rejects non-integer or negative feature indices
    for bad_formula in ("s(x0)", "s(-1)", "te(0, 'a')", "l(True)"):
        with pytest.raises(ValueError, match="non-negative integers"):
            parse_formula(bad_formula)

    # 13. Mutating n_splines on fitted LinGAM resets fitted state and rejects < 3
    for backend in ("wood", "eilers"):
        m_mut = LinGAM(n_splines=8, backend=backend).fit(x_1d, y_1d)
        assert m_mut.is_fitted
        with pytest.raises(ValueError):
            m_mut.n_splines = 2
        m_mut.n_splines = 10
        assert not m_mut.is_fitted

    # 14. wood.GAMCore.gridsearch rejects unknown kwargs and invalid search_terms
    g_err = GAMCore("s(0, k=6) + l(1)", backend="wood")
    with pytest.raises(ValueError, match="Unrecognised kwargs"):
        g_err.gridsearch(X_2d, y_2d, lam=[0.1, 1.0])
    with pytest.raises(IndexError, match="out of range"):
        g_err.gridsearch(X_2d, y_2d, search_terms=[5])
    with pytest.raises(ValueError, match="Unknown term type"):
        g_err.gridsearch(X_2d, y_2d, search_terms=["bad"])


