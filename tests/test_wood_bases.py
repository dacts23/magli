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
    """m raises for bases that ignore it; int and list forms act like the tuple form on ps/cp/bs."""
    from magli.wood._smooths import build_smooth_basis

    x = np.linspace(0, 1, 50)
    with pytest.raises(ValueError, match="m must be 2"):
        build_smooth_basis(x, bs="tp", k=8, m=3)
    with pytest.raises(ValueError, match="m is not supported"):
        build_smooth_basis(x, bs="cr", k=8, m=2)
    Xa, [Sa], _ = build_smooth_basis(x, bs="ps", k=8, m=(2, 1))
    Xb, [Sb], _ = build_smooth_basis(x, bs="ps", k=8, m=[2, 1])
    _, [Sc], _ = build_smooth_basis(x, bs="ps", k=8, m=1)
    assert np.allclose(Xa, Xb)
    assert np.allclose(Sa, Sb)
    assert np.allclose(Sa, Sc)
    with pytest.raises(ValueError, match="m must be"):
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
