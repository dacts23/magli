"""Gaussian summary/inference parity between magli.wood and R's mgcv (p-values, AIC, edf2, CI, predict terms, by, re)."""

import numpy as np
import pytest

from magli.wood import GAMCore
from test_wood_vs_r import RSCRIPT_PATH, run_r_gam

CASES = {
    # name: (magli formula, mgcv formula, method, weighted)
    "ti_factor_linear": (
        "s(0,bs='cr',k=8)+s(1,bs='cr',k=8)+ti(0,1,bs='cr',k=4)+f(2,coding='dummy')+l(3)",
        'y~s(x0,bs="cr",k=8)+s(x1,bs="cr",k=8)+ti(x0,x1,bs="cr",k=4)+g+z',
        "REML",
        False,
    ),
    "additive_weighted_gcv": (
        "s(0,bs='tp',k=8)+s(1,bs='cr',k=8)+l(3)",
        'y~s(x0,bs="tp",k=8)+s(x1,bs="cr",k=8)+z',
        "GCV.Cp",
        True,
    ),
    # weak signal so mgcv's edf2 is not capped at edf1
    "additive_reml_uncapped_edf2": (
        "s(0,bs='cr',k=10)+s(1,bs='cr',k=10)",
        'y2~s(x0,bs="cr",k=10)+s(x1,bs="cr",k=10)',
        "REML",
        False,
    ),
}


@pytest.fixture(scope="module")
def data_and_r():
    if not RSCRIPT_PATH:
        pytest.skip("Rscript not installed.")
    rng = np.random.default_rng(1)
    n = 200
    x0, x1, z = rng.uniform(0, 1, n), rng.uniform(0, 1, n), rng.normal(size=n)
    g = rng.integers(0, 4, n).astype(float)
    y = (
        np.sin(2 * np.pi * x0) + 0.5 * x1**2 + x0 * x1
        + np.array([0, 0.5, -0.3, 0.2])[g.astype(int)] + 0.3 * z + rng.normal(0, 0.2, n)
    )
    y2 = np.sin(3 * x0) + 0.6 * x1**2 + rng.normal(0, 0.7, n)
    w = rng.uniform(0.5, 2, n)
    f = lambda a: ",".join(f"{v:.15g}" for v in a)
    blocks = []
    for name, (_, rf, method, weighted) in CASES.items():
        wt = ", weights=w" if weighted else ""
        blocks.append(
            f'm<-gam({rf},data=d,method="{method}"{wt}); sm<-summary(m); pr<-predict(m,se.fit=TRUE)\n'
            f'out[["{name}"]]<-list(rsq=sm$r.sq,dev=sm$dev.expl,score=as.numeric(m$gcv.ubre),'
            f"aic=AIC(m),ll=as.numeric(logLik(m)),edf=as.numeric(sm$s.table[,1]),"
            f"refdf=as.numeric(sm$s.table[,2]),F=as.numeric(sm$s.table[,3]),p=as.numeric(sm$s.table[,4]),"
            f"ptab=sm$p.table[,c(1,2,4)],se=as.numeric(pr$se.fit),fit=as.numeric(pr$fit))"
        )
    script = (
        "suppressPackageStartupMessages({library(mgcv);library(jsonlite)})\n"
        f"d<-data.frame(x0=c({f(x0)}),x1=c({f(x1)}),g=factor(c({f(g)})),z=c({f(z)}),y=c({f(y)}),y2=c({f(y2)}),w=c({f(w)}))\n"
        "out<-list()\n" + "\n".join(blocks) + "\ncat(toJSON(out,digits=14,auto_unbox=TRUE))"
    )
    return np.column_stack([x0, x1, g, z]), {"y": y, "y2": y2}, w, run_r_gam(script)


@pytest.mark.parametrize("case", list(CASES))
def test_summary_inference_vs_r(data_and_r, case):
    X, ys, w, r = data_and_r
    formula, rform, method, weighted = CASES[case]
    y = ys["y2"] if rform.startswith("y2") else ys["y"]
    gam = GAMCore(formula, method=method).fit(X, y, weights=w if weighted else None)
    o, st = r[case], gam.statistics_
    n_s = len(o["edf"])
    smooth = [i for i, t in enumerate(gam._terms) if type(t).__name__ in ("_WoodSplineTerm", "_WoodTensorTerm")]

    assert np.max(np.abs(gam.predict(X) - np.array(o["fit"]))) < 5e-4  # REML optima agree to optimizer tolerance only
    assert np.isclose(st["r_sq_adj"], o["rsq"], atol=1e-5)
    assert np.isclose(st["deviance_explained"], o["dev"], atol=1e-6)
    assert np.isclose(gam.loglikelihood(), o["ll"], atol=1e-4)
    assert np.isclose(gam.aic(), o["aic"], atol=1e-3)
    assert np.allclose([st["edf_per_term"][i] for i in smooth], o["edf"], atol=1e-3)
    assert np.allclose([st["ref_df_per_term"][i] for i in smooth], o["refdf"], rtol=1e-3)
    assert np.allclose([st["f_stat_per_term"][i] for i in smooth], o["F"], rtol=2e-3)
    assert np.allclose([st["p_values_per_term"][i] for i in smooth], o["p"], rtol=5e-3, atol=1e-5)
    ci = gam.confidence_intervals(X)
    assert np.allclose((ci[:, 1] - ci[:, 0]) / 2 / 1.959963984540054, o["se"], rtol=1e-4)
    assert n_s == len(smooth)

    # Parametric table parity: intercept, factor levels, and linear term vs R's summary p.table.
    lines = gam.summary().splitlines()
    ptab_py = []
    for ln in lines[lines.index("Parametric coefficients:") + 2 :]:
        parts = ln.split()
        if not parts:
            break
        ptab_py.append((float(parts[1]), float(parts[2]), float(parts[4])))
    ptab_r = np.array(o["ptab"], dtype=float).reshape(len(ptab_py), 3)
    assert np.allclose(ptab_py, ptab_r, atol=2e-3)


def test_predict_terms_and_se_fit_vs_r():
    """Verify predict(type='terms', se_fit=True) against R mgcv."""
    if not RSCRIPT_PATH:
        pytest.skip("Rscript not installed.")
    rng = np.random.default_rng(42)
    n = 60
    x0, x1 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x0) + x1**2 + rng.normal(0, 0.1, n)
    f = lambda a: ",".join(f"{v:.15g}" for v in a)
    script = f"""
    suppressPackageStartupMessages({{library(mgcv);library(jsonlite)}})
    d <- data.frame(x0=c({f(x0)}), x1=c({f(x1)}), y=c({f(y)}))
    m <- gam(y ~ s(x0, bs="cr", k=6) + s(x1, bs="cr", k=6), data=d, method="REML")
    pr <- predict(m, type="terms", se.fit=TRUE)
    cat(toJSON(list(fit=as.matrix(pr$fit), se=as.matrix(pr$se.fit)), digits=14))
    """
    r = run_r_gam(script)
    X = np.column_stack([x0, x1])
    gm = GAMCore("s(0, bs='cr', k=6) + s(1, bs='cr', k=6)", method="REML").fit(X, y)

    terms_fit, terms_se = gm.predict(X, type="terms", se_fit=True)
    assert np.max(np.abs(terms_fit - np.array(r["fit"]))) < 1e-3
    assert np.max(np.abs(terms_se - np.array(r["se"]))) < 1e-4

    p0, s0 = gm.partial_dependence(0, X, se_fit=True)
    assert np.allclose(p0, terms_fit[:, 0])
    assert np.allclose(s0, terms_se[:, 0])


def test_varying_coefficient_by_vs_r():
    """Verify varying-coefficient smooth s(x, by=z) against R mgcv."""
    if not RSCRIPT_PATH:
        pytest.skip("Rscript not installed.")
    rng = np.random.default_rng(123)
    n = 80
    x0 = rng.uniform(0, 1, n)
    z = rng.uniform(0.2, 1.5, n)
    y = np.sin(2 * np.pi * x0) * z + rng.normal(0, 0.1, n)
    f = lambda a: ",".join(f"{v:.15g}" for v in a)
    script = f"""
    suppressPackageStartupMessages({{library(mgcv);library(jsonlite)}})
    d <- data.frame(x0=c({f(x0)}), z=c({f(z)}), y=c({f(y)}))
    m <- gam(y ~ s(x0, by=z, bs="cr", k=8), data=d, method="REML")
    sm <- summary(m)
    cat(toJSON(list(
      fit=as.numeric(fitted(m)),
      edf=as.numeric(sum(m$edf)),
      sp=as.numeric(m$sp),
      sig2=as.numeric(m$sig2),
      F=as.numeric(sm$s.table[,3]),
      p=as.numeric(sm$s.table[,4])
    ), digits=14))
    """
    r = run_r_gam(script)
    X = np.column_stack([x0, z])
    gm = GAMCore("s(0, by=1, bs='cr', k=8)", method="REML").fit(X, y)
    st = gm.statistics_

    assert np.max(np.abs(gm.predict(X) - np.array(r["fit"]))) < 5e-4
    assert np.isclose(st["edof"], r["edf"], atol=1e-3)
    assert np.isclose(st["scale"], r["sig2"], rtol=1e-3)
    assert np.allclose(st["sp"], r["sp"], rtol=5e-2)
    assert np.allclose(st["f_stat_per_term"], r["F"], rtol=5e-2)
    assert "s(0,by=1)" in gm.summary()


def test_random_effects_re_vs_r():
    """Verify random effect smooth s(g, bs='re') against R mgcv."""
    if not RSCRIPT_PATH:
        pytest.skip("Rscript not installed.")
    rng = np.random.default_rng(99)
    n = 90
    g = rng.integers(0, 5, n)
    x = rng.uniform(0, 1, n)
    eff = np.array([-0.6, 0.3, 0.9, -0.4, 0.0])
    y = np.sin(2 * np.pi * x) + eff[g] + rng.normal(0, 0.1, n)
    f = lambda a: ",".join(f"{v:.15g}" for v in a)
    script = f"""
    suppressPackageStartupMessages({{library(mgcv);library(jsonlite)}})
    d <- data.frame(x=c({f(x)}), g=factor(c({f(g)})), y=c({f(y)}))
    m <- gam(y ~ s(x, bs="cr", k=6) + s(g, bs="re"), data=d, method="REML")
    cat(toJSON(list(
      fit=as.numeric(fitted(m)),
      edf=as.numeric(sum(m$edf)),
      sp=as.numeric(m$sp),
      sig2=as.numeric(m$sig2)
    ), digits=14))
    """
    r = run_r_gam(script)
    X = np.column_stack([x, g.astype(float)])
    gm = GAMCore("s(0, bs='cr', k=6) + s(1, bs='re')", method="REML").fit(X, y)
    st = gm.statistics_

    assert np.max(np.abs(gm.predict(X) - np.array(r["fit"]))) < 1e-3
    assert np.isclose(st["edof"], r["edf"], atol=1e-3)
    assert np.isclose(st["scale"], r["sig2"], rtol=1e-3)
    assert np.allclose(st["sp"], r["sp"], rtol=5e-2)


def test_select_shrinkage_smooths():
    """Verify select=True enables shrinkage bases (ts/cs) penalizing null space."""
    rng = np.random.default_rng(7)
    n = 60
    X = rng.uniform(0, 1, (n, 2))
    # pure noise on feature 1
    y = np.sin(2 * np.pi * X[:, 0]) + rng.normal(0, 0.1, n)
    gm = GAMCore("s(0, bs='tp', k=8) + s(1, bs='cr', k=8)", select=True, method="REML").fit(X, y)
    # feature 1 should shrink towards 0 edf
    assert gm._terms[0].bs == "ts"
    assert gm._terms[1].bs == "cs"
    assert gm.statistics_["edf_per_term"][1] < 1.5
