"""Parity and adversarial edge-case tests covering previously unchecked combinations."""

import numpy as np
import pytest

from magli._formula import parse_formula
from magli.eilers import GAMCore as EilersGAM, LinGAM as EilersLin
from magli.wood import GAMCore as WoodGAM, LinGAM as WoodLin
from test_wood_vs_r import RSCRIPT_PATH, run_r_gam


def test_wood_oos_and_extrapolation_vs_r():
    """Verify out-of-sample predict(X_new, se_fit=True) against R mgcv across bases."""
    if not RSCRIPT_PATH:
        pytest.skip("Rscript not installed.")
    rng = np.random.default_rng(2026)
    n = 100
    x = np.linspace(0.0, 1.0, n)
    x_in = np.linspace(0.05, 0.95, 25)
    x_out = np.array([-0.25, -0.05, 1.05, 1.25])
    y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.15, n)
    f = lambda a: ",".join(f"{v:.15g}" for v in a)

    bases = ["tp", "ts", "cr", "cs", "cc", "ps", "cp", "bs"]
    r = run_r_gam(
        f"""
        suppressPackageStartupMessages({{library(mgcv); library(jsonlite)}})
        d1 <- data.frame(x=c({f(x)}), y=c({f(y)}))
        d_in <- data.frame(x=c({f(x_in)}))
        d_out <- data.frame(x=c({f(x_out)}))
        oos <- list()
        for (bs in c("tp","ts","cr","cs","cc","ps","cp","bs")) {{
          m <- gam(y ~ s(x, bs=bs, k=8), data=d1, method="REML")
          pin <- predict(m, newdata=d_in, se.fit=TRUE)
          pout <- predict(m, newdata=d_out, se.fit=TRUE)
          oos[[bs]] <- list(
            in_fit=as.numeric(pin$fit), in_se=as.numeric(pin$se.fit),
            out_fit=as.numeric(pout$fit), out_se=as.numeric(pout$se.fit)
          )
        }}
        cat(toJSON(oos, digits=14, auto_unbox=TRUE))
        """
    )

    for bs in bases:
        m = WoodLin(k=8, bs=bs, method="REML").fit(x, y)
        p_in, se_in = m.predict(x_in, se_fit=True)
        ro = r[bs]
        tol = 1e-4 if bs == "cs" else 1e-6
        assert np.max(np.abs(p_in - ro["in_fit"])) < tol
        assert np.max(np.abs(se_in - ro["in_se"])) < tol
        p_out, se_out = m.predict(x_out, se_fit=True)
        assert np.max(np.abs(p_out - ro["out_fit"])) < (1e-3 if bs == "cs" else 1e-5)
        assert np.max(np.abs(se_out - ro["out_se"])) < (1e-4 if bs == "cs" else 1e-5)


def test_wood_advanced_combinations_vs_r():
    """Verify weighted REML, one-hot no-intercept, gamma, m, mixed sp, 3D k=3, te(by), and te_aniso vs R."""
    if not RSCRIPT_PATH:
        pytest.skip("Rscript not installed.")
    rng = np.random.default_rng(2026)
    n = 100
    x = np.linspace(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.15, n)

    x0, x1, x2 = rng.uniform(0, 1, n), rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    z = rng.normal(size=n)
    g = rng.integers(0, 4, n)
    w = rng.uniform(0.2, 3.0, n)
    y_add = (
        np.sin(2.0 * np.pi * x0)
        + 0.5 * x1**2
        + 0.4 * z
        + np.array([0.0, 0.5, -0.3, 0.2])[g]
        + rng.normal(0.0, 0.2, n)
    )
    y_3d = np.sin(np.pi * x0) * np.cos(np.pi * x1) + 0.5 * x2**2 + rng.normal(0.0, 0.1, n)
    f = lambda a: ",".join(f"{v:.15g}" for v in a)

    r = run_r_gam(
        f"""
        suppressPackageStartupMessages({{library(mgcv); library(jsonlite)}})
        d1 <- data.frame(x=c({f(x)}), y=c({f(y)}))
        d2 <- data.frame(x0=c({f(x0)}), x1=c({f(x1)}), x2=c({f(x2)}), z=c({f(z)}),
                         g=factor(c({f(g)})), w=c({f(w)}),
                         y_add=c({f(y_add)}), y_3d=c({f(y_3d)}))

        mw <- gam(y_add ~ s(x0, bs="tp", k=8) + s(x1, bs="cr", k=8) + z, data=d2, weights=w, method="REML")
        smw <- summary(mw); prw <- predict(mw, se.fit=TRUE)

        m_nif <- gam(y_add ~ g + s(x1, bs="cr", k=8) - 1, data=d2, method="REML")
        m_gam <- gam(y ~ s(x, bs="cr", k=10), data=d1, method="GCV.Cp", gamma=1.4)
        m_ps21 <- gam(y ~ s(x, bs="ps", k=8, m=c(2, 1)), data=d1, method="REML")
        m_bs31 <- gam(y ~ s(x, bs="bs", k=8, m=c(3, 1)), data=d1, method="REML")
        m_mix <- gam(y_add ~ s(x0, bs="cr", k=8, sp=0.5) + s(x1, bs="cr", k=8), data=d2, method="REML")
        m_3d <- gam(y_3d ~ te(x0, x1, x2, bs="cr", k=3), data=d2, method="REML")
        m_te_by <- gam(y_3d ~ te(x0, x1, by=x2, bs="cr", k=4), data=d2, method="REML")
        m_aniso <- gam(y_3d ~ te(x0, x1, k=c(4, 6), bs=c("tp", "cr")), data=d2, method="REML")
        m_ti_sel <- gam(y_3d ~ s(x0, bs="cr", k=6) + s(x1, bs="cr", k=6) + ti(x0, x1, bs="cr", k=4), data=d2, method="REML", select=TRUE)
        pr_mix <- predict(m_mix, se.fit=TRUE)

        cat(toJSON(list(
          w_reml=list(fit=as.numeric(prw$fit), se=as.numeric(prw$se.fit),
                      edf=sum(mw$edf), scale=mw$sig2, rsq=smw$r.sq, dev=smw$dev.expl,
                      ll=as.numeric(logLik(mw)), aic=AIC(mw)),
          nif=list(fit=as.numeric(fitted(m_nif)), edf=sum(m_nif$edf), scale=m_nif$sig2),
          gam14=list(fit=as.numeric(fitted(m_gam)), edf=sum(m_gam$edf)),
          ps21=list(fit=as.numeric(fitted(m_ps21)), edf=sum(m_ps21$edf)),
          bs31=list(fit=as.numeric(fitted(m_bs31)), edf=sum(m_bs31$edf)),
          mix=list(fit=as.numeric(pr_mix$fit), se=as.numeric(pr_mix$se.fit),
                   edf=sum(m_mix$edf), scale=m_mix$sig2, aic=AIC(m_mix)),
          te3d=list(fit=as.numeric(fitted(m_3d)), edf=sum(m_3d$edf)),
          teby=list(fit=as.numeric(fitted(m_te_by)), edf=sum(m_te_by$edf)),
          aniso=list(fit=as.numeric(fitted(m_aniso)), edf=sum(m_aniso$edf)),
          ti_sel=list(fit=as.numeric(fitted(m_ti_sel)), edf=sum(m_ti_sel$edf))
        ), digits=14, auto_unbox=TRUE))
        """
    )

    X_w = np.column_stack([x0, x1, z])
    mw = WoodGAM("s(0, bs='tp', k=8) + s(1, bs='cr', k=8) + l(2)", method="REML").fit(X_w, y_add, weights=w)
    pw, sew = mw.predict(X_w, se_fit=True)
    rw = r["w_reml"]
    assert np.max(np.abs(pw - rw["fit"])) < 1e-4
    assert np.max(np.abs(sew - rw["se"])) < 1e-5
    assert np.isclose(mw.edf_, rw["edf"], atol=5e-4)
    assert np.isclose(mw.scale_, rw["scale"], rtol=1e-5)
    assert np.isclose(mw.statistics_["r_sq_adj"], rw["rsq"], atol=1e-5)
    assert np.isclose(mw.statistics_["deviance_explained"], rw["dev"], atol=1e-5)
    assert np.isclose(mw.loglikelihood(), rw["ll"], atol=1e-3)
    assert np.isclose(mw.aic(), rw["aic"], atol=1e-3)

    X_nif = np.column_stack([g.astype(float), x1])
    m_nif = WoodGAM("f(0, coding='one-hot') + s(1, bs='cr', k=8)", fit_intercept=False).fit(X_nif, y_add)
    assert np.max(np.abs(m_nif.predict(X_nif) - r["nif"]["fit"])) < 1e-5
    assert np.isclose(m_nif.edf_, r["nif"]["edf"], atol=1e-4)

    m_g14 = WoodLin(k=10, bs="cr", method="GCV.Cp", gamma=1.4).fit(x, y)
    assert np.max(np.abs(m_g14.predict(x) - r["gam14"]["fit"])) < 1e-4
    assert np.isclose(m_g14.edf_, r["gam14"]["edf"], atol=5e-4)

    m_ps21 = WoodGAM("s(0, bs='ps', k=8, m=(2, 1))").fit(x, y)
    m_bs31 = WoodGAM("s(0, bs='bs', k=8, m=(3, 1))").fit(x, y)
    assert np.max(np.abs(m_ps21.predict(x) - r["ps21"]["fit"])) < 1e-6
    assert np.max(np.abs(m_bs31.predict(x) - r["bs31"]["fit"])) < 1e-6

    m_mix = WoodGAM("s(0, bs='cr', k=8, sp=0.5) + s(1, bs='cr', k=8)").fit(X_w, y_add)
    assert np.max(np.abs(m_mix.predict(X_w) - r["mix"]["fit"])) < 1e-4
    assert np.isclose(m_mix.aic(), r["mix"]["aic"], atol=1e-2)

    from scipy import stats as _stats
    assert np.isclose(m_mix.scale_, r["mix"]["scale"], rtol=1e-4)
    pi_py = m_mix.prediction_intervals(X_w, width=0.95)
    t_crit = _stats.t.ppf(0.975, df=n - r["mix"]["edf"])
    half_r = t_crit * np.sqrt(np.array(r["mix"]["se"]) ** 2 + r["mix"]["scale"])
    pi_r = np.column_stack([np.array(r["mix"]["fit"]) - half_r, np.array(r["mix"]["fit"]) + half_r])
    assert np.max(np.abs(pi_py - pi_r)) < 1e-4

    X_3d = np.column_stack([x0, x1, x2])
    m_3d = WoodGAM("te(0, 1, 2, bs='cr', k=3)").fit(X_3d, y_3d)
    assert np.max(np.abs(m_3d.predict(X_3d) - r["te3d"]["fit"])) < 1e-4
    assert np.isclose(m_3d.edf_, r["te3d"]["edf"], atol=1e-3)

    m_te_by = WoodGAM("te(0, 1, by=2, bs='cr', k=4)").fit(X_3d, y_3d)
    assert np.max(np.abs(m_te_by.predict(X_3d) - r["teby"]["fit"])) < 1e-4
    assert np.isclose(m_te_by.edf_, r["teby"]["edf"], atol=5e-4)

    m_aniso = WoodGAM("te(0, 1, k=(4, 6), bs=('tp', 'cr'))").fit(X_3d, y_3d)
    assert np.max(np.abs(m_aniso.predict(X_3d) - r["aniso"]["fit"])) < 1e-3
    assert np.isclose(m_aniso.edf_, r["aniso"]["edf"], atol=1e-3)

    m_ti_sel = WoodGAM("s(0, bs='cr', k=6) + s(1, bs='cr', k=6) + ti(0, 1, bs='cr', k=4)", select=True).fit(X_3d, y_3d)
    assert np.max(np.abs(m_ti_sel.predict(X_3d) - r["ti_sel"]["fit"])) < 1e-4
    assert np.isclose(m_ti_sel.edf_, r["ti_sel"]["edf"], atol=5e-4)


def test_eilers_multiterm_weights_intervals_orders_vs_pygam():
    """Verify multi-term s+te+f+l, intervals, weights, spline_order, and fit_intercept=False vs pygam."""
    pygam = pytest.importorskip("pygam")
    rng = np.random.default_rng(2026)
    n = 120
    x0, x1, x2 = rng.uniform(0, 1, n), rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    g = rng.integers(0, 4, n).astype(float)
    z = rng.normal(size=n)
    w = rng.uniform(0.3, 2.5, n)
    y = (
        np.sin(2.0 * np.pi * x0)
        + np.sin(np.pi * x1) * np.cos(np.pi * x2)
        + np.array([0.0, 0.5, -0.3, 0.2])[g.astype(int)]
        + 0.4 * z
        + rng.normal(0.0, 0.15, n)
    )
    X = np.column_stack([x0, x1, x2, g, z])

    pg = pygam.LinearGAM(
        pygam.s(0, n_splines=8, lam=0.8)
        + pygam.te(1, 2, n_splines=[5, 6], lam=[0.5, 1.5])
        + pygam.f(3, coding="one-hot", lam=1.0)
        + pygam.l(4, lam=0.0),
        fit_intercept=True,
    ).fit(X, y)
    eg = EilersGAM(
        "s(0, n_splines=8, lam=0.8) + te(1, 2, n_splines=(5, 6), lam=(0.5, 1.5)) "
        "+ f(3, coding='one-hot', lam=1.0) + l(4, lam=0.0)",
        fit_intercept=True,
    ).fit(X, y)

    assert np.allclose(eg.predict(X), pg.predict(X), atol=1e-12)
    assert np.isclose(eg.edf_, pg.statistics_["edof"], atol=1e-12)
    assert np.isclose(eg.scale_, pg.statistics_["scale"], atol=1e-12)
    assert np.isclose(eg.statistics_["GCV"], pg.statistics_["GCV"], atol=1e-12)
    assert np.allclose(eg.confidence_intervals(X, width=0.95), pg.confidence_intervals(X, width=0.95), atol=1e-12)
    assert np.allclose(eg.prediction_intervals(X, width=0.95), pg.prediction_intervals(X, width=0.95), atol=1e-12)

    terms_fit, terms_se = eg.predict(X, type="terms", se_fit=True)
    assert np.allclose(terms_fit.sum(axis=1) + eg.intercept_, eg.predict(X), atol=1e-12)
    assert terms_se.shape == terms_fit.shape and np.all(terms_se > 0)

    pg_w = pygam.LinearGAM(pygam.s(0, n_splines=10, lam=0.6)).fit(x0, y, weights=w)
    eg_w = EilersLin(n_splines=10, lam=0.6).fit(x0, y, weights=w)
    assert np.allclose(eg_w.predict(x0), pg_w.predict(x0), atol=1e-6)
    assert np.isclose(eg_w.edf_, pg_w.statistics_["edof"], atol=1e-6)

    for order in (1, 2, 4):
        pg_o = pygam.LinearGAM(pygam.s(0, n_splines=10, lam=0.6, spline_order=order)).fit(x0, y)
        eg_o = EilersLin(n_splines=10, lam=0.6, spline_order=order).fit(x0, y)
        assert np.allclose(eg_o.predict(x0), pg_o.predict(x0), atol=1e-12)
        assert np.isclose(eg_o.edf_, pg_o.statistics_["edof"], atol=1e-12)

    pg_ni = pygam.LinearGAM(pygam.s(0, n_splines=10, lam=0.6), fit_intercept=False).fit(x0, y)
    eg_ni = EilersGAM("s(0, n_splines=10, lam=0.6)", fit_intercept=False).fit(x0, y)
    assert np.allclose(eg_ni.predict(x0), pg_ni.predict(x0), atol=1e-12)


def test_adversarial_and_edge_cases():
    """Verify formula, shape, dtype, k=3 cr, unfitted guards, and constrained gridsearch edge cases."""
    for bad_formula in (123, ["s(0)"], "", "   ", "s(0) +", "s(0, **{'k': 10})", "te(0)", "s(0, 1)"):
        with pytest.raises(ValueError):
            parse_formula(bad_formula)

    x_ok = np.linspace(0, 1, 20)
    y_ok = np.sin(x_ok)
    m_wood = WoodLin(k=3, bs="cr").fit(x_ok, y_ok)
    assert m_wood.predict(np.array([0.2, 0.8])).shape == (2,)

    for bad_x, bad_y in [
        (5.0, [1.0]),
        (np.ones((4, 2, 2)), np.ones(4)),
        (np.empty((0, 1)), np.empty(0)),
        (np.empty((4, 0)), np.ones(4)),
        (np.array([["a", 1.0], [None, 2.0]], dtype=object), np.ones(2)),
        (np.array([["a", 1.0], [np.nan, 2.0]], dtype=object), np.ones(2)),
    ]:
        with pytest.raises(ValueError):
            WoodLin(k=3).fit(bad_x, bad_y)

    with pytest.raises(ValueError):
        EilersLin(n_splines=6).fit(x_ok, y_ok, weights=np.full(20, np.nan))

    unfitted = WoodGAM("s(0)")
    for fn in (
        lambda: unfitted.edf_,
        lambda: unfitted.scale_,
        lambda: unfitted.intercept_,
        lambda: unfitted.confidence_intervals(x_ok),
        lambda: unfitted.prediction_intervals(x_ok),
        lambda: unfitted.partial_dependence(0, x_ok),
        lambda: unfitted.loglikelihood(),
        lambda: unfitted.aic(),
        lambda: unfitted.bic(),
        lambda: unfitted.summary(),
    ):
        with pytest.raises(RuntimeError):
            fn()

    fitted = EilersGAM("s(0, n_splines=8) + te(1, 2, n_splines=5)").fit(
        np.column_stack([x_ok, x_ok, x_ok**2]), y_ok
    )
    with pytest.raises(ValueError):
        fitted.predict(np.column_stack([x_ok, x_ok, x_ok**2]), type="bad")
    with pytest.raises(IndexError):
        fitted.partial_dependence(5, x_ok)
    with pytest.raises(ValueError):
        fitted.partial_dependence(1, x_ok)

    # Eilers constrained gridsearch + search_terms coverage
    X_gs = np.column_stack([x_ok, np.cos(x_ok)])
    y_gs = 2.0 * x_ok + 0.5 * X_gs[:, 1]
    gs = EilersGAM("s(0, n_splines=6, constraint='mono_inc') + s(1, n_splines=6)").gridsearch(
        X_gs, y_gs, lam=[0.1, 1.0], search_terms=[0]
    )
    assert np.min(np.diff(gs.partial_dependence(0, np.linspace(0, 1, 50)))) >= -1e-6

    # Mixed object-dtype input with f() + te() must yield float64 partial_dependence and plots
    X_obj = np.array(
        [["a", 0.1, 0.2], ["b", 0.4, 0.5], ["a", 0.7, 0.8], ["b", 0.9, 0.3], ["a", 0.2, 0.9], ["b", 0.6, 0.1]],
        dtype=object,
    )
    y_obj = np.array([1.0, 2.0, 3.0, 4.0, 2.5, 1.5])
    m_obj = EilersGAM("f(0) + te(1, 2, n_splines=4)").fit(X_obj, y_obj)
    assert m_obj.partial_dependence(1, X_obj).dtype == np.float64
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_obj, _ = m_obj.plot_decomposition(X_obj, y_obj)
    plt.close(fig_obj)


def test_wood_large_n_and_custom_knots_vs_r(tmp_path):
    """Verify n > 2000 knot capping and custom user-supplied knots= against R mgcv."""
    if not RSCRIPT_PATH:
        pytest.skip("Rscript not installed.")
    rng = np.random.default_rng(42)

    # 1. n > 2000 knot capping (tp uses deterministic _place_knots vs R's random 2000 subsample)
    n = 2500
    x_big = rng.uniform(0.0, 1.0, n)
    y_big = np.sin(2.0 * np.pi * x_big) + rng.normal(0.0, 0.2, n)
    csv_path = (tmp_path / "big.csv").as_posix()
    np.savetxt(csv_path, np.column_stack([x_big, y_big]), delimiter=",", header="x,y", comments="")
    r_big = run_r_gam(
        f"""
        suppressPackageStartupMessages({{library(mgcv); library(jsonlite)}})
        d <- read.csv("{csv_path}")
        out <- list()
        for (b in c("tp", "cr", "ps", "bs")) {{
            m <- gam(y ~ s(x, bs=b, k=10), data=d, method="REML")
            out[[b]] <- list(fit=as.numeric(fitted(m)), edf=sum(m$edf))
        }}
        cat(toJSON(out, digits=14))
        """
    )
    for bs, atol_fit, atol_edf in [("tp", 5e-4, 2e-4), ("cr", 1e-7, 1e-6), ("ps", 1e-7, 1e-6), ("bs", 1e-7, 1e-6)]:
        gm = WoodGAM(f"s(0, bs='{bs}', k=10)", method="REML").fit(x_big[:, None], y_big)
        assert np.allclose(gm.predict(x_big[:, None]), r_big[bs]["fit"], atol=atol_fit)
        assert np.isclose(gm.statistics_["edof"], r_big[bs]["edf"][0], atol=atol_edf)

    # 2. Custom knots= across cr, cc, ps, bs, cp
    rng_k = np.random.default_rng(7)
    x_k = rng_k.uniform(0.0, 1.0, 80)
    y_k = np.sin(2.0 * np.pi * x_k) + rng_k.normal(0.0, 0.15, 80)
    f = lambda a: ",".join(f"{v:.15g}" for v in a)
    k_cr = [round(float(v), 12) for v in np.linspace(0.05, 0.95, 7)]
    k_bounds = [-0.1, 1.1]
    for bs, kn in [("cr", k_cr), ("cc", k_cr), ("ps", k_bounds), ("bs", k_bounds), ("cp", k_bounds)]:
        gm = WoodGAM(f"s(0, bs='{bs}', k=7, knots={kn})", method="REML").fit(x_k[:, None], y_k)
        r_k = run_r_gam(
            f"""
            suppressPackageStartupMessages({{library(mgcv); library(jsonlite)}})
            d <- data.frame(x=c({f(x_k)}), y=c({f(y_k)}))
            m <- gam(y ~ s(x, bs="{bs}", k=7), data=d, knots=list(x=c({",".join(str(v) for v in kn)})), method="REML")
            cat(toJSON(list(fit=as.numeric(fitted(m)), edf=sum(m$edf)), digits=14))
            """
        )
        assert np.allclose(gm.predict(x_k[:, None]), r_k["fit"], atol=1e-6)
        assert np.isclose(gm.statistics_["edof"], r_k["edf"][0], atol=1e-5)


def test_wood_ti_all_bases_and_oos_vs_r():
    """Verify ti() noterp + constraint order across bases, 3D, select, GCV, by, and OOS."""
    if not RSCRIPT_PATH:
        pytest.skip("Rscript not installed.")
    rng = np.random.default_rng(7)
    n = 120
    x0 = rng.uniform(0.0, 1.0, n)
    x1 = rng.uniform(-1.0, 1.0, n)
    x2 = rng.uniform(0.0, 2.0, n)
    X = np.column_stack([x0, x1, x2])
    y = (
        np.sin(2.0 * np.pi * x0)
        + 0.8 * np.sin(np.pi * x0) * x2
        + 0.4 * x1**2
        + rng.normal(0.0, 0.2, n)
    )
    X_oos = np.column_stack(
        [rng.uniform(0.05, 0.95, 15), rng.uniform(-0.8, 0.8, 15), rng.uniform(0.1, 1.9, 15)]
    )
    f = lambda a: ",".join(f"{v:.15g}" for v in a)

    r = run_r_gam(
        f"""
        suppressPackageStartupMessages({{library(mgcv); library(jsonlite)}})
        d <- data.frame(x0=c({f(x0)}), x1=c({f(x1)}), x2=c({f(x2)}), y=c({f(y)}))
        d_oos <- data.frame(x0=c({f(X_oos[:, 0])}), x1=c({f(X_oos[:, 1])}), x2=c({f(X_oos[:, 2])}))
        pack <- function(m) {{
          pr <- predict(m, newdata=d_oos, se.fit=TRUE)
          list(fit=as.numeric(fitted(m)), edf=sum(m$edf), score=m$gcv.ubre[[1]],
               aic=AIC(m), oos=as.numeric(pr$fit), oos_se=as.numeric(pr$se.fit))
        }}
        res <- list(
          ti_bs=pack(gam(y ~ s(x0, bs="bs", k=6) + s(x2, bs="bs", k=6) + ti(x0, x2, bs="bs", k=5), data=d, method="REML")),
          ti_cr=pack(gam(y ~ s(x0, k=6) + s(x2, k=6) + ti(x0, x2, bs="cr", k=5), data=d, method="REML")),
          ti_cc=pack(gam(y ~ ti(x0, x2, bs="cc", k=5), data=d, method="REML")),
          ti_cp=pack(gam(y ~ ti(x0, x2, bs="cp", k=5), data=d, method="REML")),
          ti_tp=pack(gam(y ~ ti(x0, x2, bs="tp", k=5), data=d, method="REML")),
          ti_ps=pack(gam(y ~ ti(x0, x2, bs="ps", k=5), data=d, method="REML")),
          ti_aniso=pack(gam(y ~ s(x0, k=5) + s(x2, k=5) + ti(x0, x2, bs=c("tp","ps"), k=c(4,5)), data=d, method="REML")),
          ti_sel=pack(gam(y ~ ti(x0, x2, bs="tp", k=5), data=d, method="REML", select=TRUE)),
          ti_gcv=pack(gam(y ~ ti(x0, x2, bs="bs", k=5), data=d, method="GCV.Cp")),
          ti_by=pack(gam(y ~ ti(x0, x2, bs="bs", k=5, by=x1), data=d, method="REML")),
          ti_3d=pack(gam(y ~ ti(x0, x1, x2, bs="cr", k=4), data=d, method="REML"))
        )
        cat(toJSON(res, digits=14))
        """
    )

    specs = [
        ("ti_bs", "s(0, bs='bs', k=6) + s(2, bs='bs', k=6) + ti(0, 2, bs='bs', k=5)", "REML", False),
        ("ti_cr", "s(0, k=6) + s(2, k=6) + ti(0, 2, bs='cr', k=5)", "REML", False),
        ("ti_cc", "ti(0, 2, bs='cc', k=5)", "REML", False),
        ("ti_cp", "ti(0, 2, bs='cp', k=5)", "REML", False),
        ("ti_tp", "ti(0, 2, bs='tp', k=5)", "REML", False),
        ("ti_ps", "ti(0, 2, bs='ps', k=5)", "REML", False),
        ("ti_aniso", "s(0, k=5) + s(2, k=5) + ti(0, 2, bs=('tp', 'ps'), k=(4, 5))", "REML", False),
        ("ti_sel", "ti(0, 2, bs='tp', k=5)", "REML", True),
        ("ti_gcv", "ti(0, 2, bs='bs', k=5)", "GCV.Cp", False),
        ("ti_by", "ti(0, 2, bs='bs', k=5, by=1)", "REML", False),
        ("ti_3d", "ti(0, 1, 2, bs='cr', k=4)", "REML", False),
    ]
    for name, formula, method, select in specs:
        gm = WoodGAM(formula, method=method, select=select).fit(X, y)
        rv = r[name]
        oos, oos_se = gm.predict(X_oos, se_fit=True)
        assert np.allclose(gm.predict(X), rv["fit"], atol=3e-3), f"{name} fit mismatch"
        assert np.allclose(oos, rv["oos"], atol=3e-3), f"{name} oos mismatch"
        assert np.allclose(oos_se, rv["oos_se"], atol=2e-3), f"{name} oos_se mismatch"
        assert np.isclose(gm.statistics_["edof"], rv["edf"][0], atol=4e-3), f"{name} edf mismatch"
        assert np.isclose(gm.statistics_["score"], rv["score"][0], atol=1e-4), f"{name} score mismatch"


def test_wood_weighted_reml_score_and_gcv_boundary_vs_r():
    """Verify weighted REML score constant (-0.5*sum(log(w))) and GCV upper-bound escape."""
    if not RSCRIPT_PATH:
        pytest.skip("Rscript not installed.")
    rng = np.random.default_rng(42)
    n = 150
    X = np.column_stack([rng.uniform(0.0, 1.0, n), rng.integers(0, 3, n), rng.uniform(-1.0, 1.0, n)])
    y = (
        np.sin(2.0 * np.pi * X[:, 0])
        + 1.5 * np.sin(np.pi * X[:, 0]) * X[:, 2]
        + 0.5 * (X[:, 1] == 1)
        + rng.normal(0.0, 0.2, n)
    )
    w = rng.uniform(0.4, 2.0, n)
    f = lambda a: ",".join(f"{v:.15g}" for v in a)

    r = run_r_gam(
        f"""
        suppressPackageStartupMessages({{library(mgcv); library(jsonlite)}})
        d <- data.frame(x0=c({f(X[:, 0])}), x1=factor(c({f(X[:, 1])})), x2=c({f(X[:, 2])}), y=c({f(y)}), w=c({f(w)}))
        m_reml <- gam(y ~ s(x0, k=8) + x1 + x2, data=d, weights=w, method="REML")
        m_tgw <- gam(y ~ te(x0, x2, bs=c("tp","ps"), k=c(4,5)), data=d, weights=w, method="GCV.Cp")
        cat(toJSON(list(
          reml=list(fit=as.numeric(fitted(m_reml)), edf=sum(m_reml$edf), score=m_reml$gcv.ubre[[1]], aic=AIC(m_reml), bic=BIC(m_reml)),
          tgw=list(fit=as.numeric(fitted(m_tgw)), edf=sum(m_tgw$edf), score=m_tgw$gcv.ubre[[1]], aic=AIC(m_tgw), bic=BIC(m_tgw))
        ), digits=14))
        """
    )

    m_reml = WoodGAM("s(0, k=8) + f(1) + l(2)", method="REML").fit(X, y, weights=w)
    assert np.allclose(m_reml.predict(X), r["reml"]["fit"], atol=1e-5)
    assert np.isclose(m_reml.statistics_["score"], r["reml"]["score"][0], atol=1e-4)
    assert np.isclose(m_reml.aic(), r["reml"]["aic"][0], atol=1e-4)
    assert np.isclose(m_reml.bic(), r["reml"]["bic"][0], atol=1e-3)

    m_tgw = WoodGAM("te(0, 2, bs=('tp', 'ps'), k=(4, 5))", method="GCV.Cp").fit(X, y, weights=w)
    assert np.allclose(m_tgw.predict(X), r["tgw"]["fit"], atol=1e-4)
    assert np.isclose(m_tgw.statistics_["edof"], r["tgw"]["edf"][0], atol=1e-3)
    assert np.isclose(m_tgw.statistics_["score"], r["tgw"]["score"][0], atol=1e-6)
    assert np.isclose(m_tgw.aic(), r["tgw"]["aic"][0], atol=1e-3)
    assert np.isclose(m_tgw.bic(), r["tgw"]["bic"][0], atol=1e-3)


def test_wood_spline_order_te_advanced_and_terms_vs_r():
    """Verify GAMCore spline_order, 4D te, te(m=), te(knots=), y-scale invariance, terms SE, and BIC/summary."""
    if not RSCRIPT_PATH:
        pytest.skip("Rscript not installed.")
    rng = np.random.default_rng(11)
    n = 110
    X = rng.uniform(0.0, 1.0, (n, 4))
    y = np.sin(2.0 * np.pi * X[:, 0]) + 0.6 * X[:, 1] * X[:, 2] + 0.3 * X[:, 3] + rng.normal(0.0, 0.15, n)
    f = lambda a: ",".join(f"{v:.15g}" for v in a)

    # GAMCore spline_order=2 matches WoodLin and R m=c(1,2) / m=c(2,2)
    r = run_r_gam(
        f"""
        suppressPackageStartupMessages({{library(mgcv); library(jsonlite)}})
        d <- data.frame(x0=c({f(X[:, 0])}), x1=c({f(X[:, 1])}), x2=c({f(X[:, 2])}), x3=c({f(X[:, 3])}), y=c({f(y)}))
        m_ps2 <- gam(y ~ s(x0, bs="ps", k=8, m=c(1,2)), data=d, method="REML")
        m_bs2 <- gam(y ~ s(x0, bs="bs", k=8, m=c(2,2)), data=d, method="REML")
        m_4d <- gam(y ~ te(x0, x1, x2, x3, k=3), data=d, method="REML")
        m_tem <- gam(y ~ te(x0, x1, bs="ps", k=4, m=2), data=d, method="REML")
        m_sm21 <- gam(y ~ s(x0, bs="ps", k=6, m=c(2,1)), data=d, method="REML")
        m_tek <- gam(y ~ te(x0, x1, bs="cr", k=4), data=d, knots=list(x0=c(0.1,0.35,0.65,0.9)), method="REML")
        m_terms <- gam(y ~ s(x0, k=6) + te(x1, x2, k=4) + x3, data=d, method="REML")
        pr_t <- predict(m_terms, newdata=d[1:12,], type="terms", se.fit=TRUE)
        m_hi <- gam(I(y * 1e4) ~ s(x0, k=6), data=d, method="REML")
        cat(toJSON(list(
          ps2=list(fit=as.numeric(fitted(m_ps2)), edf=sum(m_ps2$edf), bic=BIC(m_ps2)),
          bs2=list(fit=as.numeric(fitted(m_bs2)), edf=sum(m_bs2$edf)),
          te4d=list(fit=as.numeric(fitted(m_4d)), edf=sum(m_4d$edf)),
          tem=list(fit=as.numeric(fitted(m_tem)), edf=sum(m_tem$edf)),
          sm21=list(fit=as.numeric(fitted(m_sm21)), edf=sum(m_sm21$edf)),
          tek=list(fit=as.numeric(fitted(m_tek)), edf=sum(m_tek$edf)),
          terms_fit=unname(as.matrix(pr_t$fit)),
          terms_se=unname(as.matrix(pr_t$se.fit)),
          hi=list(fit=as.numeric(fitted(m_hi)), edf=sum(m_hi$edf))
        ), digits=14))
        """
    )

    m_ps2 = WoodGAM("s(0, bs='ps', k=8)", method="REML", spline_order=2).fit(X, y)
    m_lin_ps2 = WoodLin(n_splines=8, bs="ps", spline_order=2).fit(X[:, 0], y)
    assert np.allclose(m_ps2.predict(X), r["ps2"]["fit"], atol=1e-5)
    assert np.allclose(m_ps2.predict(X), m_lin_ps2.predict(X[:, 0]), atol=1e-10)
    assert np.isclose(m_ps2.bic(), r["ps2"]["bic"][0], atol=1e-3)

    m_bs2 = WoodGAM("s(0, bs='bs', k=8)", method="REML", spline_order=2).fit(X, y)
    assert np.allclose(m_bs2.predict(X), r["bs2"]["fit"], atol=1e-5)
    assert np.isclose(m_bs2.statistics_["edof"], r["bs2"]["edf"][0], atol=1e-4)

    m_4d = WoodGAM("te(0, 1, 2, 3, k=3)", method="REML").fit(X, y)
    assert np.allclose(m_4d.predict(X), r["te4d"]["fit"], atol=2e-3)
    assert np.isclose(m_4d.statistics_["edof"], r["te4d"]["edf"][0], atol=3e-3)

    m_tem = WoodGAM("te(0, 1, bs='ps', k=4, m=2)", method="REML").fit(X, y)
    assert np.allclose(m_tem.predict(X), r["tem"]["fit"], atol=1e-5)
    assert np.isclose(m_tem.statistics_["edof"], r["tem"]["edf"][0], atol=1e-4)

    m_sm21 = WoodGAM("s(0, bs='ps', k=6, m=(2, 1))", method="REML").fit(X, y)
    assert np.allclose(m_sm21.predict(X), r["sm21"]["fit"], atol=1e-6)
    assert np.isclose(m_sm21.statistics_["edof"], r["sm21"]["edf"][0], atol=1e-5)

    # Scalar m on ps/cp/bs matches R mgcv's (m, m) and (m, max(0, m-1)) conventions
    r_m = run_r_gam(
        f"""
        suppressPackageStartupMessages({{library(mgcv); library(jsonlite)}})
        d <- data.frame(x0=c({f(X[:, 0])}), x1=c({f(X[:, 1])}), y=c({f(y)}))
        res <- list()
        for (b in c("ps", "cp", "bs")) {{
          for (mv in 0:3) {{
            m_fit <- gam(y ~ s(x0, bs=b, k=8, m=mv), data=d, method="REML")
            res[[paste0(b, "_", mv)]] <- list(fit=as.numeric(fitted(m_fit)), edf=sum(m_fit$edf))
          }}
        }}
        m_te_bs2 <- gam(y ~ te(x0, x1, bs="bs", k=5, m=2), data=d, method="REML")
        res$te_bs2 <- list(fit=as.numeric(fitted(m_te_bs2)), edf=sum(m_te_bs2$edf))
        cat(toJSON(res, digits=14))
        """
    )
    for b in ("ps", "cp", "bs"):
        for mv in range(4):
            gm_m = WoodGAM(f"s(0, bs='{b}', k=8, m={mv})", method="REML").fit(X, y)
            rv_m = r_m[f"{b}_{mv}"]
            assert np.allclose(gm_m.predict(X), rv_m["fit"], atol=1e-5), f"{b} m={mv} fit"
            assert np.isclose(gm_m.statistics_["edof"], rv_m["edf"][0], atol=1e-4), f"{b} m={mv} edf"
    gm_te_bs2 = WoodGAM("te(0, 1, bs='bs', k=5, m=2)", method="REML").fit(X, y)
    assert np.allclose(gm_te_bs2.predict(X), r_m["te_bs2"]["fit"], atol=1e-5)
    assert np.isclose(gm_te_bs2.statistics_["edof"], r_m["te_bs2"]["edf"][0], atol=1e-4)

    m_tek = WoodGAM("te(0, 1, bs='cr', k=4, knots=([0.1, 0.35, 0.65, 0.9], None))", method="REML").fit(X, y)
    assert np.allclose(m_tek.predict(X), r["tek"]["fit"], atol=1e-4)
    assert np.isclose(m_tek.statistics_["edof"], r["tek"]["edf"][0], atol=1e-4)

    m_terms = WoodGAM("s(0, k=6) + te(1, 2, k=4) + l(3)", method="REML").fit(X, y)
    tp, tse = m_terms.predict(X[:12], type="terms", se_fit=True)
    # R column order matches formula: s(x0), l(x3) / te(x1,x2) depending on smooth vs parametric order; verify each term matches
    r_tf = np.asarray(r["terms_fit"])
    r_ts = np.asarray(r["terms_se"])
    for j in range(3):
        errs = [np.max(np.abs(tp[:, j] - r_tf[:, k])) for k in range(3)]
        k_match = int(np.argmin(errs))
        assert errs[k_match] < 1e-6
        assert np.allclose(tse[:, j], r_ts[:, k_match], atol=1e-6)

    m_hi = WoodGAM("s(0, k=6)", method="REML").fit(X, y * 1e4)
    assert np.allclose(m_hi.predict(X), r["hi"]["fit"], rtol=1e-5, atol=1e-2)
    assert np.isclose(m_hi.statistics_["edof"], r["hi"]["edf"][0], atol=1e-4)

    # Summary edge cases: smooth-only, parametric-only, fit_intercept=False
    assert "Intercept" in WoodGAM("s(0, k=6)").fit(X, y).summary()
    assert "Intercept" not in WoodGAM("s(0, k=6)", fit_intercept=False).fit(X, y).summary()
    assert len(WoodGAM("l(0) + l(1)").fit(X, y).summary().splitlines()) > 5


def test_eilers_untested_combinations_vs_pygam():
    """Verify Eilers per-marginal te spline_order, 3D te, f(dummy), terms SE, string f(), and constant y."""
    pygam = pytest.importorskip("pygam")
    from scipy.stats import t as student_t

    rng = np.random.default_rng(123)
    n = 120
    X = np.column_stack([rng.uniform(0.0, 1.0, n), rng.integers(0, 3, n), rng.uniform(-1.0, 1.0, n)])
    y = np.sin(2.0 * np.pi * X[:, 0]) + 0.8 * (X[:, 1] == 1) + 0.5 * X[:, 2] + rng.normal(0.0, 0.15, n)

    # 1. Per-marginal te(..., spline_order=(2, 3))
    m_te = EilersGAM("te(0, 2, n_splines=(5, 6), spline_order=(2, 3), lam=(0.4, 0.9))").fit(X, y)
    p_te = pygam.LinearGAM(pygam.te(0, 2, n_splines=(5, 6), spline_order=(2, 3), lam=(0.4, 0.9))).fit(X, y)
    assert np.allclose(m_te.predict(X), p_te.predict(X), atol=1e-12)
    assert np.isclose(m_te.statistics_["edof"], p_te.statistics_["edof"], atol=1e-12)

    # 2. 3D te
    m_te3 = EilersGAM("te(0, 1, 2, n_splines=4, lam=0.8)").fit(X, y)
    p_te3 = pygam.LinearGAM(pygam.te(0, 1, 2, n_splines=4, lam=0.8)).fit(X, y)
    assert np.allclose(m_te3.predict(X), p_te3.predict(X), atol=1e-11)

    # 3. predict(type='terms', se_fit=True) vs pygam.partial_dependence
    m_s = EilersGAM("s(0, n_splines=10, lam=0.8) + s(2, n_splines=8, lam=1.2)").fit(X, y)
    p_s = pygam.LinearGAM(pygam.s(0, n_splines=10, lam=0.8) + pygam.s(2, n_splines=8, lam=1.2)).fit(X, y)
    tp, tse = m_s.predict(X[:20], type="terms", se_fit=True)
    tq = student_t.ppf(0.975, df=n - p_s.statistics_["edof"])
    for idx in range(2):
        pp, pci = p_s.partial_dependence(term=idx, X=X[:20], width=0.95)
        assert np.allclose(tp[:, idx], pp, atol=1e-12)
        assert np.allclose(tse[:, idx], (pci[:, 1] - pp) / tq, atol=1e-12)

    # 4. f(coding='dummy') + l(lam > 0) + constant y
    m_fl = EilersGAM("s(0, k=8, lam=0.5) + f(1, coding='dummy') + l(2, lam=0.3)").fit(X, y)
    assert m_fl.predict(X).shape == (n,)
    y_const = np.full(n, 3.5)
    m_c = EilersGAM("s(0, n_splines=8, lam=0.6)").fit(X, y_const)
    p_c = pygam.LinearGAM(pygam.s(0, n_splines=8, lam=0.6)).fit(X, y_const)
    assert np.allclose(m_c.predict(X), p_c.predict(X), atol=1e-12)

    # 5. String labels in object array + 1D partial_dependence on non-zero feature
    cats = np.array(["a", "b", "c"])[X[:, 1].astype(int)]
    X_str = np.column_stack([X[:, 0], cats, X[:, 2]]).astype(object)
    X_str[:, 0] = X_str[:, 0].astype(float)
    X_str[:, 2] = X_str[:, 2].astype(float)
    for cls in (EilersGAM, WoodGAM):
        ms = cls("s(0, k=8) + f(1) + l(2)").fit(X_str, y)
        p1, se1 = ms.partial_dependence(1, np.array(["a", "b", "c"]), se_fit=True)
        assert p1.shape == (3,) and se1.shape == (3,)


