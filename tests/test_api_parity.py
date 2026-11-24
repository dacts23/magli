"""API parity and usability tests across Eilers-Marx and Wood backends."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import magli
from magli.eilers import GAM as EilersGAM, GAMCore as EilersGAMCore, LinGAM as EilersLinGAM
from magli.wood import GAM as WoodGAM, GAMCore as WoodGAMCore, LinGAM as WoodLinGAM


def test_import_and_factory():
    """Verify v2.0 removes top-level dispatchers and exposes magli.wood / magli.eilers."""
    assert not hasattr(magli, "GAM")
    assert not hasattr(magli, "LinGAM")
    assert not hasattr(magli, "GAMCore")
    assert magli.__version__ == "2.1.0"

    assert isinstance(WoodGAM(k=10), WoodLinGAM)
    assert isinstance(EilersGAM(n_splines=10), EilersLinGAM)
    assert isinstance(WoodGAMCore("s(0)"), WoodGAMCore)
    assert isinstance(EilersGAMCore("s(0)"), EilersGAMCore)


def test_lingam_lifecycle_and_methods():
    """Verify that both backends provide identical lifecycle methods and attributes."""
    np.random.seed(42)
    x = np.linspace(0, 1, 60)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, 60)
    x_test = np.linspace(0.1, 0.9, 20)

    for backend, LinCls in (("eilers", EilersLinGAM), ("wood", WoodLinGAM)):
        m = LinCls(n_splines=8)
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

    for backend, GamCls in (("eilers", EilersGAMCore), ("wood", WoodGAMCore)):
        gam = GamCls(formula)
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
    m_em = EilersLinGAM()
    m_em.gridsearch(x, y, n_splines=[6, 8, 10], lam=[0.01, 0.1, 1.0])
    assert m_em.is_fitted
    assert m_em.n_splines in [6, 8, 10]

    # Wood grid search
    m_wd = WoodLinGAM()
    m_wd.gridsearch(x, y, n_splines=[6, 8, 10])
    assert m_wd.is_fitted
    assert m_wd.n_splines in [6, 8, 10]

    X2 = np.column_stack([x, np.random.uniform(0, 1, 50)])
    g_wd = WoodGAMCore("s(0, bs='cr', k=10) + s(1, bs='cr', k=10)")
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

    for backend, LinCls, GamCls in (("eilers", EilersLinGAM, EilersGAMCore), ("wood", WoodLinGAM, WoodGAMCore)):
        # Unit weights match unweighted fit
        m_unw = LinCls(n_splines=14).fit(x, y)
        m_ones = LinCls(n_splines=14).fit(
            x, y, weights=np.ones_like(y)
        )
        assert np.allclose(m_unw.coef_, m_ones.coef_)

        # Downweighting outliers recovers clean curve
        m_w = LinCls(n_splines=14).fit(x, y, weights=w)
        rmse_unw = np.sqrt(np.mean((m_unw.predict(x) - y_clean) ** 2))
        rmse_w = np.sqrt(np.mean((m_w.predict(x) - y_clean) ** 2))
        assert rmse_w < rmse_unw * 0.6

        g_unw = GamCls("s(0, k=14)").fit(x, y)
        g_w = GamCls("s(0, k=14)").fit(x, y, weights=w)
        assert np.sqrt(np.mean((g_w.predict(x) - y_clean) ** 2)) < np.sqrt(
            np.mean((g_unw.predict(x) - y_clean) ** 2)
        ) * 0.6

        # Gridsearch with weights
        gs_w = LinCls().gridsearch(
            x, y, n_splines=[10, 14], weights=w
        )
        assert np.sqrt(np.mean((gs_w.predict(x) - y_clean) ** 2)) < rmse_unw * 0.6

        # Invalid weights validation
        with pytest.raises(ValueError, match="weights length"):
            LinCls().fit(x, y, weights=np.ones(5))
        with pytest.raises(ValueError, match="weights must be finite"):
            LinCls().fit(x, y, weights=-np.ones_like(y))
        with pytest.raises(ValueError, match="weights must be finite"):
            LinCls().fit(x, y, weights=np.zeros_like(y))


def test_audit_fixes_comprehensive():
    """Verify bug fixes for fit_intercept=False, aliases, scalar eval, 1D pdep, formula args, strings."""
    rng = np.random.default_rng(42)
    x = np.linspace(0.1, 1.0, 50)
    y = 2.0 * x

    # 1. fit_intercept=False on non-zero mean data
    for backend, LinCls, GamCls in (("wood", WoodLinGAM, WoodGAMCore), ("eilers", EilersLinGAM, EilersGAMCore)):
        m = GamCls("s(0, k=8)", fit_intercept=False).fit(x, y)
        assert m.is_fitted
        assert m.deviance_explained() > 0.99
        assert np.allclose(m.predict(x), y, atol=1e-2)

    # 2. Documented constraint aliases 'monotonic_inc' and 'monotonic_dec'
    y_noisy = np.exp(x) + rng.normal(0, 0.05, len(x))
    for c_name in ("monotonic_inc", "mono_inc"):
        m_inc = EilersLinGAM(n_splines=10, constraint=c_name).fit(x, y_noisy)
        preds = m_inc.predict(np.linspace(0.1, 1.0, 100))
        assert np.all(np.diff(preds) >= -1e-6)
    for c_name in ("monotonic_dec", "mono_dec"):
        m_dec = EilersLinGAM(n_splines=10, constraint=c_name).fit(x, -y_noisy)
        preds = m_dec.predict(np.linspace(0.1, 1.0, 100))
        assert np.all(np.diff(preds) <= 1e-6)

    # 3. Scalar inputs to predict and intervals
    m_single = WoodLinGAM(k=8).fit(x, y)
    assert m_single.predict(0.5).shape == (1,)
    assert m_single.confidence_intervals(0.5).shape == (1, 2)
    assert m_single.prediction_intervals(0.5).shape == (1, 2)

    # 4. Multi-feature 1-row predict and 1D partial_dependence
    X_2d = np.column_stack([x, rng.uniform(0, 1, len(x))])
    y_2d = 2.0 * x + 3.0 * X_2d[:, 1]
    for backend, LinCls, GamCls in (("wood", WoodLinGAM, WoodGAMCore), ("eilers", EilersLinGAM, EilersGAMCore)):
        m_2d = GamCls("s(0, k=8) + s(1, k=8)").fit(X_2d, y_2d)
        assert m_2d.predict([0.5, 0.2]).shape == (1,)
        grid_1d = np.linspace(0, 1, 25)
        pdep0 = m_2d.partial_dependence(0, grid_1d)
        pdep1 = m_2d.partial_dependence(1, grid_1d)
        assert pdep0.shape == (25,)
        assert pdep1.shape == (25,)

    # 5. Formula parser rejects extra positional arguments
    with pytest.raises(ValueError, match="accepts only 1 positional"):
        WoodGAMCore("s(0, 10)")
    with pytest.raises(ValueError, match="accepts only 1 positional"):
        WoodGAMCore("l(0, 1)")

    # 6. Categorical string factor variables & NaN validation with string columns
    X_str = np.array([["ctrl", 0.1], ["treat", 0.5], ["ctrl", 0.2], ["treat", 0.6], ["treat", 0.7]], dtype=object)
    y_str = np.array([1.0, 3.0, 1.2, 2.9, 3.2])
    for backend, LinCls, GamCls in (("wood", WoodLinGAM, WoodGAMCore), ("eilers", EilersLinGAM, EilersGAMCore)):
        m_cat = GamCls("f(0) + s(1, k=4)").fit(X_str, y_str)
        assert m_cat.is_fitted
        summary_str = m_cat.summary()
        assert "f(0)" in summary_str

        # NaN validation in presence of string column
        X_nan = X_str.copy()
        X_nan[0, 1] = np.nan
        with pytest.raises(ValueError, match="contains NaN or infinity"):
            GamCls("f(0) + s(1, k=4)").fit(X_nan, y_str)

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
    for backend, LinCls, GamCls in (("wood", WoodLinGAM, WoodGAMCore), ("eilers", EilersLinGAM, EilersGAMCore)):
        m = GamCls("f(0) + s(1, k=4)").fit(X_str, y_str)
        assert m.predict(X_str).shape == (len(y_str),)
        assert m.confidence_intervals(X_str).shape == (len(y_str), 2)
        assert m.prediction_intervals(X_str).shape == (len(y_str), 2)
        assert m.partial_dependence(0, X_str).shape == (len(y_str),)
        fig_d, _ = m.plot_decomposition(X_str, y_str)
        plt.close(fig_d)
        fig_g, _ = m.plot_diagnostics(X_str, y_str)
        plt.close(fig_g)
        m_gs = GamCls("f(0) + s(1, k=4)")
        if backend == "wood":
            m_gs.gridsearch(X_str, y_str, n_splines=[4, 5])
        else:
            m_gs.gridsearch(X_str, y_str, lam=[0.1, 1.0])
        assert m_gs.is_fitted and m_gs.predict(X_str).shape == (len(y_str),)

    # 2. String group labels in s(0, bs='re') and bar-chart plot_decomposition
    m_re = WoodGAMCore("s(0, bs='re') + s(1, k=4)").fit(X_str, y_str)
    assert m_re.predict(X_str).shape == (len(y_str),)
    fig_re, axes_re = m_re.plot_decomposition(X_str, y_str)
    assert len(axes_re[1].patches) == 2
    plt.close(fig_re)

    # 3. Re-fitting wood.GAMCore re-optimizes free smoothing parameters
    X_refit = rng.uniform(0, 1, (80, 1))
    y_noisy = np.sin(2 * np.pi * X_refit[:, 0]) + rng.normal(0, 0.5, 80)
    y_clean = np.sin(2 * np.pi * X_refit[:, 0]) + rng.normal(0, 0.01, 80)
    g_refit = WoodGAMCore("s(0, k=10)").fit(X_refit, y_noisy)
    sp_noisy = g_refit.statistics_["sp"][0]
    g_refit.fit(X_refit, y_clean)
    assert g_refit.statistics_["sp"][0] < sp_noisy * 0.05

    # 4. te() defaults (k=5, bs='cr') and tuple k=(4, 6) in wood backend
    g_te_def = WoodGAMCore("te(0, 1)")
    assert g_te_def._terms[0].k_per == [5, 5]
    assert g_te_def._terms[0].bs_per == ["cr", "cr"]
    g_te_tup = WoodGAMCore("te(0, 1, k=(4, 6))")
    assert g_te_tup._terms[0].k_per == [4, 6]

    # 5. Invalid factor coding and wood one-hot intercept collinearity guard
    for backend, LinCls, GamCls in (("wood", WoodLinGAM, WoodGAMCore), ("eilers", EilersLinGAM, EilersGAMCore)):
        with pytest.raises(ValueError, match="Unknown coding"):
            GamCls("f(0, coding='invalid')")
    with pytest.raises(ValueError, match="collinear with the intercept"):
        WoodGAMCore("f(0, coding='one-hot')", fit_intercept=True)
    m_oh = WoodGAMCore("f(0, coding='one-hot')", fit_intercept=False).fit(X_str, y_str)
    assert m_oh.is_fitted and np.all(m_oh.statistics_["se"] < 1.0)

    # 6. k/sp constructor aliases and properties on eilers LinGAM
    m_e = EilersGAM(k=8, sp=0.5)
    assert isinstance(m_e, EilersLinGAM)
    assert m_e.k == 8 and m_e.sp == 0.5
    m_e.k = 9
    m_e.sp = 0.25
    assert m_e.n_splines == 9 and m_e.lam == 0.25

    # 7. Documented fitted properties edf_, scale_, intercept_
    x_1d = np.linspace(0, 1, 40)
    y_1d = np.sin(2 * np.pi * x_1d) + rng.normal(0, 0.1, 40)
    for backend, LinCls, GamCls in (("wood", WoodLinGAM, WoodGAMCore), ("eilers", EilersLinGAM, EilersGAMCore)):
        m_lin = LinCls(n_splines=8).fit(x_1d, y_1d)
        assert np.isclose(m_lin.edf_, m_lin.statistics_["edof"])
        assert np.isclose(m_lin.scale_, m_lin.statistics_["scale"])
        assert np.isclose(m_lin.intercept_, m_lin.coef_[-1])
        m_no_int = GamCls("s(0, k=8)", fit_intercept=False).fit(x_1d, y_1d)
        assert m_no_int.intercept_ == 0.0

    # 8. gridsearch validates NaN/inf and x/y length mismatch on both backends
    x_bad = x_1d.copy()
    x_bad[3] = np.nan
    y_bad = y_1d.copy()
    y_bad[3] = np.inf
    for backend, LinCls, GamCls in (("wood", WoodLinGAM, WoodGAMCore), ("eilers", EilersLinGAM, EilersGAMCore)):
        with pytest.raises(ValueError, match="NaN or infinity"):
            LinCls().gridsearch(x_bad, y_1d)
        with pytest.raises(ValueError, match="NaN or infinity"):
            LinCls().gridsearch(x_1d, y_bad)
        with pytest.raises(ValueError, match="must match y length"):
            LinCls().gridsearch(x_1d[:10], y_1d)

    # 9. Negative sp / lam rejected in GAMCore formulas on both backends
    with pytest.raises(ValueError, match="non-negative"):
        WoodGAMCore("s(0, sp=-1.0)")
    with pytest.raises(ValueError, match="non-negative"):
        EilersGAMCore("s(0, lam=-1.0)")

    # 10. Interval width validation
    m_iv = WoodLinGAM(k=6).fit(x_1d, y_1d)
    for bad_w in (0.0, 1.0, 95.0, -0.5):
        with pytest.raises(ValueError, match="width"):
            m_iv.confidence_intervals(x_1d, width=bad_w)
        with pytest.raises(ValueError, match="width"):
            m_iv.prediction_intervals(x_1d, width=bad_w)

    # 11. partial_dependence on 1D single-row input for te() and s(..., by=...)
    X_2d = np.column_stack([x_1d, rng.uniform(0.2, 1.0, len(x_1d))])
    y_2d = np.sin(2 * np.pi * X_2d[:, 0]) * X_2d[:, 1]
    m_te_by = WoodGAMCore("te(0, 1, k=4) + s(0, by=1, k=5)").fit(X_2d, y_2d)
    assert m_te_by.partial_dependence(0, [0.5, 0.4]).shape == (1,)
    assert m_te_by.partial_dependence(1, [0.5, 0.4]).shape == (1,)

    # 12. parse_formula rejects non-integer or negative feature indices
    for bad_formula in ("s(x0)", "s(-1)", "te(0, 'a')", "l(True)"):
        with pytest.raises(ValueError, match="non-negative integers"):
            parse_formula(bad_formula)

    # 13. Mutating n_splines on fitted LinGAM resets fitted state and rejects < 3
    for backend, LinCls, GamCls in (("wood", WoodLinGAM, WoodGAMCore), ("eilers", EilersLinGAM, EilersGAMCore)):
        m_mut = LinCls(n_splines=8).fit(x_1d, y_1d)
        assert m_mut.is_fitted
        with pytest.raises(ValueError):
            m_mut.n_splines = 2
        m_mut.n_splines = 10
        assert not m_mut.is_fitted

    # 14. wood.GAMCore.gridsearch rejects unknown kwargs and invalid search_terms
    g_err = WoodGAMCore("s(0, k=6) + l(1)")
    with pytest.raises(ValueError, match="Unrecognised kwargs"):
        g_err.gridsearch(X_2d, y_2d, lam=[0.1, 1.0])
    with pytest.raises(IndexError, match="out of range"):
        g_err.gridsearch(X_2d, y_2d, search_terms=[5])
    with pytest.raises(ValueError, match="Unknown term type"):
        g_err.gridsearch(X_2d, y_2d, search_terms=["bad"])

    # 15. wood gridsearch surfaces errors when all candidates fail instead of silent fallback
    with pytest.raises(ValueError, match="All gridsearch candidates failed"):
        WoodLinGAM().gridsearch(x_1d, y_1d, n_splines=[2])
    with pytest.raises(ValueError, match="All gridsearch candidates failed"):
        WoodGAMCore("s(0)").gridsearch(x_1d[:, None], y_1d, n_splines=[2])

    # 16. plot_decomposition uses numeric feature on overall panel when leading term is categorical,
    #     and supports pure-categorical models without crashing
    m_cat_num = WoodGAMCore("f(0) + s(1, k=4)").fit(X_str, y_str)
    fig_cn, axes_cn = m_cat_num.plot_decomposition(X_str, y_str)
    x_line = axes_cn[0].lines[0].get_xdata()
    assert np.min(x_line) < 0.2 and np.max(x_line) > 0.7
    plt.close(fig_cn)

    m_cat_only = WoodGAMCore("f(0)").fit(X_str[:, :1], y_str)
    fig_co, axes_co = m_cat_only.plot_decomposition(X_str[:, :1], y_str)
    assert [t.get_text() for t in axes_co[0].get_xticklabels()] == ["ctrl", "treat"]
    plt.close(fig_co)


@pytest.mark.parametrize("GAMCls", [EilersGAMCore, WoodGAMCore])
def test_zero_residual_likelihood_and_diagnostics(GAMCls):
    """Exact zero residuals return finite likelihood/IC and plot diagnostics without warnings."""
    import warnings

    x = np.linspace(0, 1, 20)
    y = np.zeros(20)
    m = GAMCls("l(0)").fit(x, y)
    assert np.isfinite(m.loglikelihood())
    assert np.isfinite(m.aic())
    assert np.isfinite(m.bic())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        fig, _ = m.plot_diagnostics(x, y)
    plt.close(fig)
    assert not caught


def test_wood_gridsearch_preserves_gamma_and_validates_nested_grids():
    """Wood gridsearch preserves self.gamma when omitted and rejects mismatched grids."""
    rng = np.random.default_rng(7)
    X = rng.uniform(0, 1, (80, 2))
    y = np.sin(2 * np.pi * X[:, 0]) + 0.5 * X[:, 1] + rng.normal(0, 0.1, 80)

    m = WoodGAMCore("s(0) + s(1)", method="GCV", gamma=1.8).gridsearch(
        X, y, n_splines_grids=[[5, 7], [5, 7]]
    )
    assert m.gamma == pytest.approx(1.8)

    with pytest.raises(ValueError, match="Expected 2 grids"):
        WoodGAMCore("s(0) + s(1)").gridsearch(X, y, n_splines_grids=[[5, 6]])

    lg = WoodLinGAM(method="GCV", gamma=1.8).gridsearch(X[:, 0], y, n_splines=[6, 8])
    assert lg.gamma == pytest.approx(1.8)

    m_noint = WoodGAMCore("s(0) + s(1)", fit_intercept=False).gridsearch(
        X, y, n_splines_grids=[[5, 6], [5, 6]]
    )
    assert [t.absorb_cons for t in m_noint._terms] == [False, True]
    assert np.linalg.matrix_rank(m_noint._design_matrix_) == m_noint._design_matrix_.shape[1]


@pytest.mark.parametrize("GAMCls", [EilersGAMCore, WoodGAMCore])
def test_predict_wrong_1d_length_raises_value_error(GAMCls):
    """Passing a mismatched 1D vector to a multi-feature model raises ValueError."""
    rng = np.random.default_rng(11)
    X = rng.uniform(0, 1, (30, 2))
    y = X[:, 0] + X[:, 1]
    m = GAMCls("s(0, k=5) + s(1, k=5)").fit(X, y)
    with pytest.raises(ValueError):
        m.predict(np.array([0.1, 0.2, 0.3]))


@pytest.mark.parametrize("GAMCls", [EilersGAMCore, WoodGAMCore])
def test_deviance_explained_matches_stats_on_tiny_y(GAMCls):
    """deviance_explained() matches statistics_['deviance_explained'] even on tiny y ~ 1e-8."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 80)
    y = (x + rng.normal(0.0, 0.3, 80)) * 1e-8
    m = GAMCls("l(0)").fit(x, y)
    assert np.isclose(m.deviance_explained(), m.statistics_["deviance_explained"])


def test_zero_weights_effective_sample_count():
    """Zero weights exclude inactive observations from n_samples, scale, GCV, and loglikelihood."""
    rng = np.random.default_rng(42)
    x = np.linspace(0.0, 1.0, 100)
    y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.1, 100)
    w = np.r_[np.ones(50), np.zeros(50)]

    for LinCls, GamCls in ((EilersLinGAM, EilersGAMCore), (WoodLinGAM, WoodGAMCore)):
        m = LinCls(n_splines=8).fit(x, y, weights=w)
        assert m.statistics_["n_samples"] == 50
        assert np.isfinite(m.scale_)
        assert np.isfinite(m.statistics_["GCV"])
        assert np.isfinite(m.loglikelihood())

        g_sub = GamCls("l(0)").fit(x[:50], y[:50])
        g_zw = GamCls("l(0)").fit(x, y, weights=w)
        assert g_zw.statistics_["n_samples"] == 50
        assert np.isclose(g_zw.scale_, g_sub.scale_, rtol=1e-10)
        assert np.isclose(g_zw.loglikelihood(), g_sub.loglikelihood(), atol=1e-10)


def test_3d_predict_lam_reset_gamma_validation_and_docstrings():
    """Reject 3D predict inputs, reset is_fitted on lam mutation, validate gamma, and check _base docstrings."""
    import pathlib

    rng = np.random.default_rng(21)
    X = rng.uniform(0.0, 1.0, (40, 2))
    y = np.sin(2.0 * np.pi * X[:, 0]) + 0.5 * X[:, 1] + rng.normal(0.0, 0.1, 40)

    for LinCls, GamCls in ((EilersLinGAM, EilersGAMCore), (WoodLinGAM, WoodGAMCore)):
        gm = GamCls("s(0, k=5) + s(1, k=5)").fit(X, y)
        with pytest.raises(ValueError, match="1D or 2D"):
            gm.predict(np.ones((4, 2, 1)))

        lm = LinCls(n_splines=8).fit(X[:, 0], y)
        assert lm.is_fitted
        lm.lam = 5.0
        assert not lm.is_fitted

        for bad_gamma in (0.0, -1.5, np.nan):
            with pytest.raises(ValueError, match="gamma"):
                GamCls("s(0, k=5)").gridsearch(X[:, :1], y, gamma=bad_gamma)

    for bad_gamma in (0.0, -1.5, np.nan):
        with pytest.raises(ValueError, match="gamma"):
            WoodGAMCore("s(0, k=5)", gamma=bad_gamma)
        with pytest.raises(ValueError, match="gamma"):
            WoodLinGAM(k=6).gridsearch(X[:, 0], y, gamma=bad_gamma)

    base_bytes = pathlib.Path("src/magli/_base.py").read_bytes()
    assert b"\x07" not in base_bytes
    assert rb"\alpha" in base_bytes



