API Reference
=============

Welcome to the **magli** API reference. This document provides in-depth, NumPy-style
technical specifications, mathematical formulations, parameter tables, and runnable
examples for all public classes, methods, formula terms, and backend engines.

.. currentmodule:: magli

.. contents:: Table of Contents
   :local:
   :depth: 2


Unified Dispatch API
--------------------

The top-level :class:`GAM` (aliased as :class:`LinGAM`) and :class:`GAMCore` classes provide
a unified entrypoint that dynamically dispatches to either the Wood (``backend='wood'``, default)
or Eilers-Marx (``backend='eilers'``) engine.

Backend-specific classes are also directly importable from :mod:`magli` as
``WoodLinGAM``, ``WoodGAMCore``, ``EilersLinGAM``, and ``EilersGAMCore``.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   LinGAM
   GAMCore


Backend Engines Compared
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Feature / Aspect
     - Wood Backend (``backend='wood'``)
     - Eilers-Marx Backend (``backend='eilers'``)
   * - **Primary Paradigm**
     - Penalized Regression Splines (Wood 2011, 2017)
     - P-splines with Difference Penalties (Eilers & Marx 1996)
   * - **Smoothing Parameter Estimation**
     - Continuous REML or GCV optimization via Newton-Raphson / L-BFGS-B
     - Discrete 2D grid search over :math:`(k, \lambda)` using cached Cholesky
   * - **Basis Families**
     - Thin plate (``tp``, ``ts``), cubic regression (``cr``, ``cs``), cyclic (``cc``), P-splines (``ps``, ``cp``), B-splines (``bs``), random effects (``re``)
     - Uniform B-splines of arbitrary order (default cubic)
   * - **Multidimensional Interactions**
     - Full tensor products (``te``), pure ANOVA interactions (``ti``)
     - Tensor products with Kronecker sum penalties
   * - **Shape Constraints**
     - Periodic boundary conditions (``constraint='periodic'`` / ``bs='cc'``)
     - Asymmetric penalty active-set: ``monotonic_inc``, ``monotonic_dec``, ``convex``, ``concave``, ``periodic``
   * - **Statistical Inference**
     - Approximate F/Chi-square tests, Wood (2013) p-values, ``edf2``
     - Standard errors, Bayesian covariance matrix, pseudo :math:`R^2`


Formula Grammar & Term Constructors
-----------------------------------

The formula parser :func:`magli._formula.parse_formula` allows specifying multi-term models
using concise expressions joined by ``+``:

.. code-block:: python

    import magli as mg
    model = mg.GAMCore("s(0, k=12, bs='cr') + te(1, 2) + l(3) + f(4)")

Term Constructors
^^^^^^^^^^^^^^^^^

.. rubric:: 1. Univariate Smooths: ``s(feature, ...)``

Fits an unconstrained or constrained smooth curve :math:`f(x_j)` using penalized splines:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Parameter
     - Type & Default
     - Description
   * - ``feature``
     - ``int``
     - 0-based column index of the input feature in :math:`X`.
   * - ``k`` / ``n_splines``
     - ``int``, default=10
     - Maximum basis dimension (number of spline functions). Must be :math:`\ge 3`.
   * - ``bs``
     - ``str``, default=``'tp'``
     - Basis family: ``'tp'`` (thin plate), ``'ts'`` (shrinkage TP), ``'cr'`` (cubic regression), ``'cs'`` (shrinkage cubic), ``'cc'`` (cyclic cubic), ``'ps'`` (P-spline), ``'cp'`` (cyclic P-spline), ``'bs'`` (B-spline), ``'re'`` (random effect).
   * - ``m``
     - ``tuple`` or ``int``, optional
     - Spline and derivative penalty order (e.g. ``m=(3, 2)`` for cubic spline with 2nd difference penalty).
   * - ``sp`` / ``lam``
     - ``float``, optional
     - Fixed smoothing parameter :math:`\lambda`. If omitted, estimated automatically.
   * - ``by``
     - ``int``, optional
     - Column index of a continuous covariate :math:`z` for varying-coefficient smooths :math:`z \cdot f(x_j)`.
   * - ``constraint``
     - ``str``, optional
     - Shape constraint: ``'periodic'``, ``'monotonic_inc'``, ``'monotonic_dec'``, ``'convex'``, ``'concave'``.

.. rubric:: 2. Tensor Product Smooths: ``te(feat1, feat2, ...)``

Constructs multidimensional interaction smooths :math:`f(x_1, x_2)` invariant to coordinate scaling:

.. math::

    B_{\text{tensor}} = B_1 \otimes B_2, \quad S_{\text{tensor}} = \lambda_1 (S_1 \otimes I_{k_2}) + \lambda_2 (I_{k_1} \otimes S_2)

.. rubric:: 3. Pure Tensor Interactions: ``ti(feat1, feat2, ...)``

Constructs pure interaction smooths with marginal main effects projected out, enabling classical functional ANOVA decompositions:

.. math::

    y = \beta_0 + f_1(x_1) + f_2(x_2) + f_{12}(x_1, x_2) + \epsilon

specified as ``formula="s(0) + s(1) + ti(0, 1)"``.

.. rubric:: 4. Parametric Linear Terms: ``l(feature)``

Includes a standard linear covariate :math:`\beta_j x_j` without a non-linear spline penalty.

.. rubric:: 5. Categorical Factors: ``f(feature, coding='dummy')``

Includes categorical variables with reference-level dummy coding.


Shared Model Methods & Lifecycle
--------------------------------

All models in ``magli`` inherit from :class:`magli._base._GAMBase`, providing a unified
method interface across both backends:

.. autoclass:: magli._base._GAMBase
   :members: predict, partial_dependence, confidence_intervals, prediction_intervals, summary, aic, bic, loglikelihood, deviance_explained, plot_decomposition, plot_diagnostics
   :noindex:


Engine Implementations
----------------------

Direct access to engine-specific classes:

Wood Engine (:mod:`magli.wood`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simon Wood's GAM methods (Wood 2011, 2017) with multiple basis types,
null-space constraint absorption, and continuous REML / GCV smoothing optimization.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   wood.LinGAM
   wood.GAMCore

Eilers-Marx Engine (:mod:`magli.eilers`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

B-spline additive models with discrete GCV grid search and soft shape constraints
following Eilers & Marx (1996).

.. autosummary::
   :toctree: generated/
   :nosignatures:

   eilers.LinGAM
   eilers.GAMCore
