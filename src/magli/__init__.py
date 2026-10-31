"""magli (Modelo Aditivo Generalizado Lineal): Pythonic Generalized Additive Models (GAMs) with Wood and Eilers backends."""

from . import eilers, wood
from .eilers import GAMCore as EilersGAMCore, LinGAM as EilersLinGAM
from .wood import GAMCore as WoodGAMCore, LinGAM as WoodLinGAM

__version__ = "1.1.2"

_BACKENDS = {
    "wood": (WoodLinGAM, WoodGAMCore),
    "eilers": (EilersLinGAM, EilersGAMCore),
    "pygam": (EilersLinGAM, EilersGAMCore),
}


def _dispatch(idx: int, backend: str, *args, **kwargs):
    key = str(backend).lower()
    if key not in _BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'wood' or 'eilers'.")
    return _BACKENDS[key][idx](*args, **kwargs)


class GAM:
    r"""Single-term Generalized Additive Model (GAM) dispatcher.

    Fits univariate smooth non-linear regression models of the form:

    .. math::

        y = \beta_0 + f(x) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)

    by minimizing a penalized sum of squared errors:

    .. math::

        \mathcal{S}(\beta) = \|y - X\beta\|^2 + \lambda \beta^T S \beta

    where :math:`X` is the spline design matrix, :math:`S` is the smoothing
    penalty matrix, and :math:`\lambda \ge 0` is the smoothing parameter balancing
    goodness-of-fit against curve wiggliness.

    Instantiates either :class:`magli.wood.LinGAM` (Simon Wood's penalized regression
    splines with continuous REML or GCV optimization) or :class:`magli.eilers.LinGAM`
    (Paul Eilers & Brian Marx's P-splines with discrete difference penalties and shape
    constraints).

    Parameters
    ----------
    n_splines : int, default=10
        Number of spline basis functions (also accepted as `k`). For the Wood backend,
        this represents the upper bound on basis dimension :math:`k`. For the Eilers
        backend, this defines the number of evenly spaced B-spline columns. Must be >= 3.
    lam : float, optional
        Fixed smoothing parameter :math:`\lambda` (also accepted as `sp`). If None,
        optimal :math:`\lambda` is estimated automatically during :meth:`fit` using
        REML (Wood) or grid search via :meth:`gridsearch` (Eilers).
    spline_order : int, default=3
        Degree of the spline polynomial. Order 3 corresponds to cubic splines.
    constraint : {None, 'periodic', 'monotonic_inc', 'monotonic_dec', 'convex', 'concave'}, default=None
        Shape constraint enforced on the smooth term:

        - None : Unconstrained smooth.
        - 'periodic' : Cyclic boundary condition ensuring :math:`f(x_{\min}) = f(x_{\max})`.
        - 'monotonic_inc' : Enforces :math:`f'(x) \ge 0` (Eilers backend).
        - 'monotonic_dec' : Enforces :math:`f'(x) \le 0` (Eilers backend).
        - 'convex' : Enforces :math:`f''(x) \ge 0` (Eilers backend).
        - 'concave' : Enforces :math:`f''(x) \le 0` (Eilers backend).
    k : int, optional
        Alternative alias for `n_splines` following R ``mgcv`` naming conventions.
    sp : float, optional
        Alternative alias for `lam` following R ``mgcv`` naming conventions.
    bs : {'tp', 'ts', 'cr', 'cs', 'cc', 'ps', 'cp', 'bs'}, default='tp'
        Basis type (Wood backend):

        - 'tp' : Thin plate regression splines (optimal low-rank isotropic basis).
        - 'ts' : Thin plate splines with shrinkage (can be penalized to zero).
        - 'cr' : Cubic regression splines with evenly spaced knots.
        - 'cs' : Cubic regression splines with shrinkage.
        - 'cc' : Cyclic cubic regression splines.
        - 'ps' : P-splines with B-spline basis and difference penalty.
        - 'cp' : Cyclic P-splines.
        - 'bs' : Standard B-splines with integrated derivative penalty.
    method : {'REML', 'GCV'}, default='REML'
        Criterion for automatic smoothing parameter estimation in the Wood backend:

        - 'REML' : Restricted Maximum Likelihood (recommended; avoids undersmoothing).
        - 'GCV' : Generalized Cross Validation.
    gamma : float, default=1.0
        Degrees-of-freedom multiplier inflating the model penalty in GCV to prevent
        overfitting. A value of 1.4 is often used in practice.
    backend : {'wood', 'eilers'}, default='wood'
        Engine backend:

        - 'wood' : Simon Wood's continuous REML/GCV engine (:class:`magli.wood.LinGAM`).
        - 'eilers' : Eilers & Marx P-spline engine (:class:`magli.eilers.LinGAM`).

    Attributes
    ----------
    coef_ : ndarray of shape (n_coefficients,)
        Estimated model coefficients including the intercept and spline weights.
    intercept_ : float
        Estimated global intercept :math:`\beta_0`.
    scale_ : float
        Estimated residual variance :math:`\hat{\sigma}^2` (Wood) or residual standard
        deviation :math:`\hat{\sigma}` (Eilers).
    edf_ : float
        Effective degrees of freedom of the fitted model, :math:`\operatorname{tr}(F)`.
    statistics_ : dict
        Diagnostic criteria dictionary containing `'reml'`, `'gcv'`, `'aic'`, `'bic'`,
        `'deviance'`, and `'edfs'`.
    is_fitted : bool
        True if the model has been fitted to data.

    See Also
    --------
    GAMCore : Multi-term Generalized Additive Model with formula interface.
    magli.wood.LinGAM : Direct Wood backend class.
    magli.eilers.LinGAM : Direct Eilers backend class.

    Notes
    -----
    Generalized Additive Models (Hastie & Tibshirani, 1990; Wood 2017) relax linear
    regression assumptions by replacing linear predictors with sums of smooth functions.

    Identifiability is resolved by imposing the sum-to-zero constraint:

    .. math::

        \sum_{i=1}^n f(x_i) = 0

    which guarantees a unique decomposition between the intercept :math:`\beta_0` and
    the centered smooth curve :math:`f(x)`.

    The effective degrees of freedom (EDF) measures model complexity:

    .. math::

        \operatorname{EDF} = \operatorname{tr}\left((X^T W X + S)^{-1} X^T W X\right)

    When :math:`\lambda \to 0`, EDF approaches the unpenalized basis dimension :math:`k`.
    When :math:`\lambda \to \infty`, EDF shrinks to the dimension of the penalty null space
    (typically 1, representing a straight line).

    References
    ----------
    - Wood, S. N. (2011). "Fast stable restricted maximum likelihood and marginal
           likelihood estimation of semiparametric generalized linear models."
           *Journal of the Royal Statistical Society: Series B*, 73(1), 3-36.
    - Eilers, P. H. C., & Marx, B. D. (1996). "Flexible smoothing with B-splines
           and penalties." *Statistical Science*, 11(2), 89-121.
    - Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*.
           2nd Edition, Chapman and Hall/CRC.

    Examples
    --------
    Fit a univariate smooth to noisy observations from a nonlinear function:

    >>> import numpy as np
    >>> import magli as mg
    >>> rng = np.random.default_rng(42)
    >>> x = np.linspace(0, 1, 100)
    >>> y = np.sin(2 * np.pi * x) + rng.normal(0, 0.1, size=100)
    >>> model = mg.LinGAM(n_splines=10, backend='wood')
    >>> _ = model.fit(x, y)
    >>> model.is_fitted
    True
    >>> np.round(model.edf_, 1)
    5.6
    >>> y_pred = model.predict(np.array([0.25, 0.75]))
    >>> np.allclose(y_pred, [1.0, -1.0], atol=0.25)
    True
    """

    def __new__(cls, *args, backend: str = "wood", **kwargs):
        return _dispatch(0, backend, *args, **kwargs)


LinGAM = GAM


class GAMCore:
    r"""Multi-term Generalized Additive Model (GAM) formula dispatcher.

    Fits multivariate additive regression models of the form:

    .. math::

        y = \beta_0 + \sum_{j=1}^p f_j(x_j) + \sum_{m} f_m(x_{m1}, x_{m2}) + \epsilon

    using a formula specification supporting univariate smooth terms (``s()``),
    tensor product smooths (``te()``), pure tensor interactions (``ti()``),
    linear terms (``l()``), and categorical factors (``f()``).

    Parameters
    ----------
    formula : str
        Formula string specifying the additive structure of the model.
        Available term constructors:

        - ``s(j, k=10, bs='tp', m=None, sp=None, by=None, constraint=None)`` : Univariate smooth.
        - ``te(j1, j2, k=10, bs='cr', m=None, sp=None, by=None)`` : Full tensor product smooth.
        - ``ti(j1, j2, k=10, bs='cr', m=None, sp=None, by=None)`` : Pure tensor interaction (ANOVA).
        - ``l(j)`` : Parametric linear covariate.
        - ``f(j)`` : Categorical factor covariate with reference level coding.
    method : {'REML', 'GCV'}, default='REML'
        Criterion for continuous smoothing parameter estimation in the Wood backend:

        - 'REML' : Restricted Maximum Likelihood (recommended; avoids undersmoothing).
        - 'GCV' : Generalized Cross Validation.
    gamma : float, default=1.0
        Degrees-of-freedom multiplier inflating GCV penalty to prevent undersmoothing.
    fit_intercept : bool, default=True
        Whether to include a global intercept :math:`\beta_0`.
    spline_order : int, default=3
        Default spline polynomial order (3 for cubic splines).
    select : bool, default=False
        If True, adds an extra shrinkage penalty to null spaces of smooth terms,
        enabling terms to be completely penalized to zero (model selection).
    backend : {'wood', 'eilers'}, default='wood'
        Engine backend:

        - 'wood' : Simon Wood's REML engine (:class:`magli.wood.GAMCore`).
        - 'eilers' : Eilers & Marx P-spline engine (:class:`magli.eilers.GAMCore`).

    Attributes
    ----------
    coef_ : ndarray of shape (n_coefficients,)
        Estimated model coefficients across all terms.
    intercept_ : float
        Estimated global intercept :math:`\beta_0`.
    scale_ : float
        Estimated residual variance (Wood) or standard deviation (Eilers).
    statistics_ : dict
        Model fit criteria including `'reml'`, `'gcv'`, `'aic'`, `'bic'`, and `'edfs'`.
    is_fitted : bool
        True if the model has been fitted to data.

    See Also
    --------
    LinGAM : Single-term GAM dispatcher.
    magli.wood.GAMCore : Direct Wood multi-term class.
    magli.eilers.GAMCore : Direct Eilers multi-term class.

    Notes
    -----
    Multi-term GAMs allow flexible non-parametric modeling of complex phenomena while
    preserving modular interpretability: each smooth term :math:`f_j(x_j)` can be
    plotted and inspected in isolation without confounding from other variables.

    For multidimensional interactions with differing physical units (e.g. spatial
    coordinates, time and temperature), tensor product smooths (``te()``) or pure
    interaction smooths (``ti()``) with marginal scale invariance are recommended
    over isotropic thin plate splines.

    References
    ----------
    - Wood, S. N. (2006). "Low-rank scale-invariant tensor product smooths for
           generalized additive mixed models." *Biometrics*, 62(4), 1025-1036.
    - Wood, S. N. (2011). "Fast stable restricted maximum likelihood and marginal
           likelihood estimation of semiparametric generalized linear models."
           *Journal of the Royal Statistical Society: Series B*, 73(1), 3-36.

    Examples
    --------
    Fit a two-term additive model:

    >>> import numpy as np
    >>> import magli as mg
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(0, 1, size=(100, 2))
    >>> y = np.sin(2 * np.pi * X[:, 0]) + (X[:, 1] ** 2) + rng.normal(0, 0.1, size=100)
    >>> model = mg.GAMCore("s(0) + s(1)", backend="wood")
    >>> _ = model.fit(X, y)
    >>> model.is_fitted
    True
    >>> preds = model.predict(X[:5])
    >>> len(preds)
    5
    """

    def __new__(cls, *args, backend: str = "wood", **kwargs):
        return _dispatch(1, backend, *args, **kwargs)


__all__ = [
    "GAM",
    "LinGAM",
    "GAMCore",
    "EilersLinGAM",
    "EilersGAMCore",
    "WoodLinGAM",
    "WoodGAMCore",
    "eilers",
    "wood",
    "__version__",
]
