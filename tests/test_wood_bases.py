"""Tests for all Wood basis families: tp, ts, cr, cs, cc, ps, cp, bs, re, and te."""

import numpy as np
import pytest
from magli.wood import GAMCore, LinGAM



@pytest.mark.parametrize("basis", ["tp", "ts", "cr", "cs", "cc", "ps", "cp", "bs", "re"])
def test_all_univariate_bases(basis):
    """Test that every basis can fit, predict, and compute intervals without error."""
    np.random.seed(123)
    n = 60
    if basis == "re":
        # Categorical covariate for random effect
        x = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n)
        y = 2.0 * x + np.random.normal(0, 0.2, n)
        x_test = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    else:
        x = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, n)
        x_test = np.linspace(0.05, 0.95, 25)

    model = LinGAM(k=8, bs=basis, method="REML")
    model.fit(x, y)
    assert model.is_fitted
    assert model.coef_ is not None

    preds = model.predict(x_test)
    assert preds.shape == (len(x_test),)
    assert np.all(np.isfinite(preds))

    ci = model.confidence_intervals(x_test)
    pi = model.prediction_intervals(x_test)
    assert ci.shape == (len(x_test), 2)
    assert pi.shape == (len(x_test), 2)
    assert np.all(ci[:, 0] <= ci[:, 1])
    assert np.all(pi[:, 0] <= pi[:, 1])

    # Summary
    s = model.summary()
    assert isinstance(s, str)
    assert "Parametric coefficients" in s
    assert "Approximate significance of smooth terms" in s


def test_tensor_product_basis():
    """Test 2D tensor product te(0, 1) interaction term."""
    np.random.seed(42)
    n = 100
    x0 = np.random.uniform(0, 1, n)
    x1 = np.random.uniform(0, 1, n)
    X = np.column_stack([x0, x1])
    y = np.sin(np.pi * x0) * np.cos(np.pi * x1) + np.random.normal(0, 0.05, n)

    gam = GAMCore("te(0, 1, bs='cr', k=4)")
    gam.fit(X, y)
    assert gam.is_fitted

    preds = gam.predict(X[:10])
    assert preds.shape == (10,)
    assert np.all(np.isfinite(preds))

    ci = gam.confidence_intervals(X[:10])
    assert ci.shape == (10, 2)
    assert np.all(ci[:, 0] <= ci[:, 1])


def test_periodic_constraint():
    """Test periodic smooth term constraint='periodic' and cyclic P-spline 'cp'."""
    np.random.seed(42)
    n = 60
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + np.random.normal(0, 0.1, n)

    for m in (LinGAM(k=8, constraint="periodic"), LinGAM(k=8, bs="cp")):
        m.fit(x, y)
        assert m.is_fitted
        pred_0 = m.predict([0.0])[0]
        pred_2pi = m.predict([2 * np.pi])[0]
        assert np.isclose(pred_0, pred_2pi, atol=1e-6)


def test_ts_shrinks_null_space():
    from magli.wood._smooths import build_tp_basis

    x = np.linspace(0, 1, 50)
    _, [S_tp], _ = build_tp_basis(x, k=8, shrink=False)
    _, [S_ts], _ = build_tp_basis(x, k=8, shrink=True)
    ev_tp = np.linalg.eigvalsh(S_tp)
    ev_ts = np.linalg.eigvalsh(S_ts)
    assert np.allclose(ev_tp[:2], 0.0, atol=1e-12)
    assert np.allclose(ev_ts[:2], ev_tp[2] * 0.1, rtol=1e-6)
    assert np.allclose(ev_ts[2:], ev_tp[2:], rtol=1e-6)


def test_gcv_method():
    """Test GCV.Cp smoothing parameter estimation method."""
    np.random.seed(42)
    x = np.linspace(0, 1, 80)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, 80)

    m = LinGAM(k=10, bs="cr", method="GCV.Cp")
    m.fit(x, y)
    assert m.is_fitted
    assert m.statistics_["method"] == "GCV.CP"
    assert m.statistics_["score"] > 0


def test_multi_penalty_reml_and_gcv():
    """Verify multi-penalty REML and GCV optimization across disjoint and overlapping penalties."""
    np.random.seed(42)
    n = 80
    X = np.random.uniform(0, 1, (n, 3))
    y = np.sin(2 * np.pi * X[:, 0]) + np.cos(2 * np.pi * X[:, 1]) + np.random.normal(0, 0.08, n)

    for method in ("REML", "GCV.Cp"):
        gam = GAMCore("s(0, bs='cr', k=8) + te(1, 2, bs='cr', k=5)", method=method).fit(X, y)
        assert gam.is_fitted
        assert len(gam.statistics_["sp"]) == 3
        assert gam.deviance_explained() > 0.9
        # x2 is pure noise so its smoothing parameter should be much larger than x1's
        assert gam.statistics_["sp"][2] > gam.statistics_["sp"][1] * 10
        assert np.allclose(gam.predict(X), gam.predict(X + 0.0))


def test_wood_constraint_validation():
    """Wood honors only 'periodic' (via a cyclic basis); other constraints raise, as LinGAM does."""
    with pytest.raises(ValueError, match="Unsupported constraint 'mono_inc'"):
        GAMCore("s(0, constraint='mono_inc', k=8)")
    with pytest.raises(ValueError, match="Unsupported constraint 'nonsense'"):
        GAMCore("s(0, constraint='nonsense', k=8)")
    assert GAMCore("s(0, constraint='periodic', bs='cr', k=8)")._terms[0].bs == "cc"
    assert GAMCore("s(0, constraint='periodic', bs='cp', k=8)")._terms[0].bs == "cp"


def test_m_option_validated_and_coerced():
    """m raises for bases that ignore it; int and list forms match R mgcv rules on ps/cp/bs."""
    from magli.wood._smooths import build_smooth_basis

    x = np.linspace(0, 1, 50)
    with pytest.raises(ValueError, match="m must be 2"):
        build_smooth_basis(x, bs="tp", k=8, m=3)
    with pytest.raises(ValueError, match="m is not supported"):
        build_smooth_basis(x, bs="cr", k=8, m=2)
    Xa, [Sa], _ = build_smooth_basis(x, bs="ps", k=8, m=(2, 1))
    Xb, [Sb], _ = build_smooth_basis(x, bs="ps", k=8, m=[2, 1])
    Xp1, [Sp1], _ = build_smooth_basis(x, bs="ps", k=8, m=(1, 1))
    Xc1, [Sc1], _ = build_smooth_basis(x, bs="ps", k=8, m=1)
    assert np.allclose(Xa, Xb)
    assert np.allclose(Sa, Sb)
    assert np.allclose(Xp1, Xc1)
    assert np.allclose(Sp1, Sc1)
    Xbs_tup, [Sbs_tup], _ = build_smooth_basis(x, bs="bs", k=8, m=(2, 1))
    Xbs_int, [Sbs_int], _ = build_smooth_basis(x, bs="bs", k=8, m=2)
    assert np.allclose(Xbs_tup, Xbs_int)
    assert np.allclose(Sbs_tup, Sbs_int)
    with pytest.raises(ValueError, match="m must be non-negative"):
        build_smooth_basis(x, bs="ps", k=8, m=-1)
    with pytest.raises(ValueError, match="m must be a non-negative"):
        build_smooth_basis(x, bs="ps", k=8, m=[1, 2, 3])


def test_engine_drops_stale_aic_keys():
    """fit_wood_engine no longer returns old-definition aic/bic/loglik; GAMCore.aic() is the one AIC."""
    from magli.wood._optimizer import fit_wood_engine

    rng = np.random.default_rng(3)
    X = rng.uniform(0, 1, (60, 2))
    y = np.sin(3 * X[:, 0]) + rng.normal(0, 0.1, 60)
    gam = GAMCore("s(0, bs='cr', k=8)")
    gam._compile(X)
    gam._compute_coef_slices()
    Xm = gam._build_model_matrix(X)
    P = np.zeros((Xm.shape[1],) * 2)
    n_c = gam._terms[0].n_coefs
    P[:n_c, :n_c] = gam._terms[0].build_penalties()[0]
    res = fit_wood_engine(X=Xm, y=y, S_list=[P], coef_slices=gam._coef_slices)
    assert not {"aic", "bic", "loglik"} & res.keys()


def test_ti_pure_interaction():
    """ti() excludes main effects: (k-1)^2 coefs, full-rank alongside s(0)+s(1)."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (400, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + 2 * X[:, 0] * X[:, 1] + rng.normal(0, 0.1, 400)
    gam = GAMCore("s(0, bs='cr', k=8) + s(1, bs='cr', k=8) + ti(0, 1, bs='cr', k=5)").fit(X, y)
    ti = gam._terms[2]
    assert ti.n_coefs == 16
    D = gam._design_matrix_
    assert np.linalg.matrix_rank(D) == D.shape[1]  # no collinearity with s(0)+s(1)
    assert np.all(np.isfinite(gam.predict(X)))
    assert "ti(0,1)" in gam.summary()


@pytest.mark.parametrize("basis", ["ps", "bs", "cp"])
def test_nan_prediction_propagates_for_bspline_bases(basis):
    """NaN inputs produce NaN predictions instead of zero for ps/bs/cp bases."""
    x = np.linspace(0, 1, 40)
    y = np.sin(2 * np.pi * x) + 1.5
    m = GAMCore(f"s(0, bs='{basis}', k=8)").fit(x, y)
    preds = m.predict(np.array([0.2, np.nan, 0.8]))
    assert np.isfinite(preds[0])
    assert np.isnan(preds[1])
    assert np.isfinite(preds[2])


def test_tensor_mismatched_sequence_lengths_raise():
    """Tensor product validates sequence length of m and knots against d."""
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 1, (50, 3))
    y = rng.normal(size=50)
    with pytest.raises(ValueError, match="length"):
        GAMCore("te(0, 1, 2, bs='ps', m=(2, 1))").fit(X, y)


@pytest.mark.parametrize("basis", ["cc", "cp"])
def test_cyclic_bases_reject_small_k(basis):
    """Cyclic bases raise ValueError when k is too small instead of mutating k."""
    x = np.linspace(0, 1, 30)
    y = np.sin(2 * np.pi * x)
    with pytest.raises(ValueError):
        LinGAM(n_splines=3, bs=basis).fit(x, y)


def test_wood_scale_invariance_tiny_y_and_tiny_collinear_x():
    """Wood REML and linear solver remain invariant on tiny y/x and collinear large-scale inputs."""
    rng = np.random.default_rng(42)
    x = np.linspace(0.0, 1.0, 100)
    y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.1, 100)

    m_ref = LinGAM(k=10, bs="cr", method="REML").fit(x, y)
    m_sm = LinGAM(k=10, bs="cr", method="REML").fit(x, y * 1e-6)
    assert np.isclose(m_sm.lam, m_ref.lam, rtol=1e-4)
    assert np.isclose(m_sm.edf_, m_ref.edf_, atol=1e-4)
    assert np.max(np.abs(m_sm.predict(x) / 1e-6 - m_ref.predict(x))) < 1e-5

    y_lin_sm = (2.0 * x + 1.0 + rng.normal(0.0, 0.05, 100)) * 1e-8
    m_lsm = GAMCore("l(0)").fit(x, y_lin_sm)
    p_line = [ln for ln in m_lsm.summary().splitlines() if ln.strip().startswith("l(0)")][0]
    assert float(p_line.split()[-1]) < 1e-6

    x_tiny = x * 1e-7
    y_tiny = 2.0 * x_tiny + 1.0 + rng.normal(0.0, 1e-10, 100)
    assert np.isclose(GAMCore("l(0)").fit(x_tiny, y_tiny).coef_[0], 2.0, atol=1e-3)

    X_col = np.column_stack([x * 1e8, x * 1e8])
    m_col = GAMCore("l(0) + l(1)").fit(X_col, 2.0 * x)
    assert np.allclose(m_col.predict(X_col), 2.0 * x, atol=1e-6)


def test_wood_chunked_tp_matches_direct():
    """Chunked thin-plate spline evaluation (> 2048 rows) matches the full direct matrix product."""
    from magli.wood._smooths import _eta_1d, build_tp_basis, predict_smooth_basis

    rng = np.random.default_rng(5)
    x = rng.uniform(0.0, 1.0, 2500)
    X_tp, _, meta = build_tp_basis(x, k=8)
    direct = _eta_1d((x - meta["shift"])[:, None] - meta["Xu"][None, :]) @ meta["UkZ"]
    T_new = np.column_stack([np.ones(len(x)), x - meta["shift"]])
    expected = np.column_stack([direct, T_new]) / meta["col_rms"][None, :]
    assert np.allclose(X_tp, expected, atol=1e-12)
    assert np.allclose(predict_smooth_basis(x, meta), expected, atol=1e-12)


def test_wood_basis_k_m_and_knots_validation():
    """ps/cp/bs reject invalid k, m, wrong-length knots, and unsorted/duplicate knots."""
    x = np.linspace(0.0, 1.0, 40)
    y = np.sin(2.0 * np.pi * x)

    for formula in (
        "s(0, bs='ps', k=3)",
        "s(0, bs='cp', k=4, m=(4, 2))",
        "s(0, bs='cp', k=4, m=(2, 5))",
        "s(0, bs='bs', k=3)",
        "s(0, bs='ps', k=5, knots=[0.0, 0.2, 0.5, 0.8, 1.0])",
        "s(0, bs='bs', k=5, knots=[0.0, 0.2, 0.5, 0.8, 1.0])",
        "s(0, bs='cr', k=5, knots=[1.0, 0.5, 0.2, 0.8, 0.0])",
        "s(0, bs='cr', k=5, knots=[0.0, 0.0, 0.5, 0.8, 1.0])",
        "s(0, bs='cc', k=5, knots=[1.0, 0.5, 0.2, 0.8, 0.0])",
    ):
        with pytest.raises(ValueError):
            GAMCore(formula).fit(x, y)


def test_wood_formula_kwargs_onehot_order_and_re_cases():
    """Validate l()/f() kwargs, term-order-independent one-hot, uppercase RE, and re/gridsearch edge cases."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with pytest.raises(ValueError, match="Unrecognised kwargs"):
        GAMCore("l(0, sp=10.0)")
    with pytest.raises(ValueError, match="Unrecognised kwargs"):
        GAMCore("f(0, by=1)")

    X_str = np.array(
        [["a", 0.1], ["b", 0.2], ["a", 0.4], ["b", 0.6], ["a", 0.8], ["b", 0.9]],
        dtype=object,
    )
    y_str = np.array([1.0, 2.0, 1.5, 2.5, 1.8, 2.9])

    m_fs = GAMCore("f(0, coding='one-hot') + s(1, bs='cr', k=4)", fit_intercept=False).fit(X_str, y_str)
    m_sf = GAMCore("s(1, bs='cr', k=4) + f(0, coding='one-hot')", fit_intercept=False).fit(X_str, y_str)
    assert np.allclose(m_fs.predict(X_str), m_sf.predict(X_str), atol=1e-10)

    m_re_lc = GAMCore("s(0, bs='re') + s(1, bs='cr', k=4)", fit_intercept=False)
    m_re_uc = GAMCore("s(0, bs='RE') + s(1, bs='cr', k=4)", fit_intercept=False)
    assert [t.absorb_cons for t in m_re_lc._terms] == [t.absorb_cons for t in m_re_uc._terms]

    fig, _ = LinGAM(bs="re").fit(X_str[:, 0], y_str).plot_decomposition(X_str[:, 0], y_str)
    plt.close(fig)

    g_re = GAMCore("s(0, bs='re') + s(1, bs='cr')").gridsearch(X_str, y_str, n_splines_grids=[[4, 5]])
    assert g_re.is_fitted and g_re._terms[1].k in (4, 5)

    rng = np.random.default_rng(13)
    x = np.linspace(0.0, 1.0, 40)
    y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.1, 40)
    lg = LinGAM(bs="cr").gridsearch(x, y, n_splines=[6, 8], lam=[0.01, 1.0])
    assert lg.is_fitted and lg.lam in (0.01, 1.0)

    X2 = rng.uniform(0.0, 1.0, (50, 2))
    y2 = np.sin(2.0 * np.pi * X2[:, 0]) + 0.5 * X2[:, 1] + rng.normal(0.0, 0.1, 50)
    g_2d = GAMCore("s(0) + s(1)").gridsearch(X2, y2, n_splines_grids=np.array([[5, 6], [6, 7]]))
    g_mix = GAMCore("s(0) + s(1)").gridsearch(X2, y2, n_splines_grids=[5, [6, 7]])
    assert g_2d.is_fitted and g_mix.is_fitted and g_mix._terms[0].k == 5

