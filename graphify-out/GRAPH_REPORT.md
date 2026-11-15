# Graph Report - magli  (2026-11-15)

## Corpus Check
- 47 files · ~61,688 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 10 file(s) not represented in the graph (top: (none) 4, .ipynb 3, .bat 2)

## Summary
- 621 nodes · 1262 edges · 44 communities (23 shown, 21 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 26 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `54fde7e9`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- fit_wood_engine
- GAMCore
- Base GAM Class & Model Diagnostics
- eilers/_gam.py
- GitHub Pages CI/CD Workflow
- Wood Multi-term GAMCore & REML
- Smoothing Parameter Tuning & Backend Comparison
- eilers/__init__.py
- eilers/_terms.py
- PyPI Release Workflow
- _smooths.py
- steps
- matplotlib_pyplot
- _SplineTerm
- numpy
- parse_formula
- test_api_parity.py
- build_smooth_basis
- b_spline_basis
- _FactorTerm
- Section-by-Section Guide
- Identifiability & Sum-to-Zero Constraints
- Wood Basis Functions Catalog
- Generalized Additive Models (GAMs)
- wood/__init__.py
- magli: Generalized Additive Models (GAMs) in Python
- build_tp_basis
- Quickstart Documentation
- 2D Tensor Interaction Surface Visualization
- magli Root Package Overview
- test_smoothcon_basis_and_penalty_vs_r
- .build_constraint_penalty
- 13_eilers_vs_wood_comparison.py
- 14_wood_basis_decomposition_plots.py
- pytest
- pytest
- _TensorTerm
- ndarray
- 15_eilers_basis_decomposition_plots.py
- 04_shape_constraints.py
- Unified Dispatch API (GAM / LinGAM / GAMCore)
- GAMCore Additive Term Decomposition Plot

## God Nodes (most connected - your core abstractions)
1. `_GAMBase` - 33 edges
2. `run_r_gam()` - 21 edges
3. `GAMCore` - 18 edges
4. `LinGAM` - 18 edges
5. `LinGAM` - 17 edges
6. `build_smooth_basis()` - 17 edges
7. `_Term` - 15 edges
8. `_SplineTerm` - 15 edges
9. `predict_smooth_basis()` - 15 edges
10. `_validate_xyw()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `Observation Weights for Outlier Handling` --conceptually_related_to--> `test_weights_both_backends()`  [INFERRED]
  docs/_static/images/observation_weights_outliers.png → tests/test_api_parity.py
- `Simon Wood Penalized Regression Splines` --implements--> `LinGAM`  [EXTRACTED]
  docs/math/wood.rst → src/magli/wood/_lingam.py
- `Eilers & Marx P-Splines` --implements--> `LinGAM`  [EXTRACTED]
  docs/math/eilers.rst → src/magli/eilers/_lingam.py
- `test_engine_drops_stale_aic_keys()` --calls--> `fit_wood_engine()`  [INFERRED]
  tests/test_wood_bases.py → src/magli/wood/_optimizer.py
- `test_audit_fixes_comprehensive()` --calls--> `build_tp_basis()`  [INFERRED]
  tests/test_api_parity.py → src/magli/wood/_smooths.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Eilers & Marx P-Spline System** — docs_math_eilers_psplines, docs_math_eilers_difference_penalty, docs_math_eilers_shape_constraints, docs_static_images_eilers_bspline_decomposition [INFERRED 0.85]
- **Wood Penalized Regression Spline System** — docs_math_wood_penalized_regression_splines, docs_math_wood_basis_zoo, docs_math_wood_reml, src_magli_wood_lingam_lingam [INFERRED 0.85]

## Communities (44 total, 21 thin omitted)

### Community 0 - "fit_wood_engine"
Cohesion: 0.15
Nodes (7): _edf2_per_coef(), fit_wood_engine(), _liu2(), _mroot_t(), _psum_chisq(), wood_re_test(), wood_test_stat()

### Community 1 - "GAMCore"
Cohesion: 0.14
Nodes (5): _assemble_penalty_from_bases(), GAMCore, _eval_k_combo(), _solve_constrained(), solve_pirls()

### Community 3 - "eilers/_gam.py"
Cohesion: 0.21
Nodes (5): _precompute_k(), _precompute_penalty_bases(), _set_lams_from_combo(), instantiate_terms(), _Term

### Community 4 - "GitHub Pages CI/CD Workflow"
Cohesion: 0.06
Nodes (32): runs-on, steps, concurrency, cancel-in-progress, group, environment, needs, runs-on (+24 more)

### Community 5 - "Wood Multi-term GAMCore & REML"
Cohesion: 0.16
Nodes (4): Smoothing Parameter Tuning & Grid Search, Restricted Maximum Likelihood (REML) Criterion, GCV Curve and Effective Degrees of Freedom (EDF), LinGAM

### Community 6 - "Smoothing Parameter Tuning & Backend Comparison"
Cohesion: 0.17
Nodes (7): Two Philosophies for GAM Smoothing, Eilers & Marx P-Splines, Simon Wood Penalized Regression Splines, LinGAM Diagnostics 4-Panel Plot, Eilers B-Spline Basis & P-Spline Decomposition, Eilers vs Wood Comparison Plot, LinGAM

### Community 8 - "eilers/_terms.py"
Cohesion: 0.24
Nodes (3): circular_second_diff_penalty(), ridge_penalty(), shape_constraint_penalty()

### Community 9 - "PyPI Release Workflow"
Cohesion: 0.11
Nodes (17): jobs, publish, name, on, release, workflow_dispatch, permissions, contents (+9 more)

### Community 10 - "_smooths.py"
Cohesion: 0.11
Nodes (13): build_cc_basis(), build_cp_basis(), build_cr_basis(), _c_spline_des(), _cwrap(), _eta_1d(), _eval_cc_basis(), _eval_cr_basis() (+5 more)

### Community 11 - "steps"
Cohesion: 0.10
Nodes (22): runs-on, steps, jobs, docs, test, python-version, name, on (+14 more)

### Community 14 - "numpy"
Cohesion: 0.10
Nodes (13): test_audit_fixes_comprehensive(), test_bspline_linear_extrapolation(), test_deviance_explained_accuracy(), test_directional_shape_constraints_enforced(), test_eilers_constrained_small_scale(), test_eilers_edof_per_coef_and_wood_summary_pvalues(), test_eilers_gridsearch_diagonal_and_design_space(), test_eilers_huge_lam_and_gridsearch_flat_and_tensor_orders() (+5 more)

### Community 15 - "parse_formula"
Cohesion: 0.24
Nodes (4): _eval_ast(), _flatten_plus(), parse_formula(), test_audit_regressions_fixed()

### Community 16 - "test_api_parity.py"
Cohesion: 0.12
Nodes (6): Observation Weights for Outlier Handling, test_gamcore_lifecycle_and_methods(), test_gridsearch_both_backends(), test_import_and_factory(), test_lingam_lifecycle_and_methods(), test_weights_both_backends()

### Community 17 - "build_smooth_basis"
Cohesion: 0.22
Nodes (5): build_bs_basis(), build_ps_basis(), build_smooth_basis(), _spline_design(), test_m_option_validated_and_coerced()

### Community 20 - "Section-by-Section Guide"
Cohesion: 0.09
Nodes (20): Example 10: Engineering Case Study README, How to Adapt This for Your Own Data, Key Takeaways, Quick Start, Section 10: Weighted vs Standard Comparison, Section 1: Data Summary, Section 2: Baseline Fit, Section 3: Grid Search (with new partial-grid flexibility) (+12 more)

### Community 23 - "Generalized Additive Models (GAMs)"
Cohesion: 0.50
Nodes (4): Generalized Additive Models (GAMs), Penalized Splines & Regularization, Regularization Penalty Lambda Effect, Why GAMs: The Interpretability-Flexibility Sweet Spot

### Community 24 - "wood/__init__.py"
Cohesion: 0.07
Nodes (10): _khatri_rao(), _validate_xyw(), GAMCore, absorb_sum_to_zero_constraint(), _append_select_null_penalty(), _default_spline_m_1d(), instantiate_wood_terms(), _WoodFactorTerm (+2 more)

### Community 25 - "magli: Generalized Additive Models (GAMs) in Python"
Cohesion: 0.14
Nodes (13): 1. Wood (2011) Engine (Automatic REML Smoothing), 2. Eilers & Marx (1996) Engine (B-Splines & Constraints), Documentation, Features, Formula Syntax, [→ https://dacts23.github.io/magli/](https://dacts23.github.io/magli/), Installation, Interactive Tutorials & Demo Scripts (+5 more)

### Community 26 - "build_tp_basis"
Cohesion: 0.29
Nodes (4): build_tp_basis(), _top_k_eig(), _wood_qt(), test_ts_shrinks_null_space()

### Community 31 - ".build_constraint_penalty"
Cohesion: 0.50
Nodes (3): Shape Constraints API, Asymmetric Difference Penalty for Shape Constraints, Shape Constraints Showcase

### Community 34 - "pytest"
Cohesion: 0.11
Nodes (8): test_all_univariate_bases(), test_engine_drops_stale_aic_keys(), test_gcv_method(), test_multi_penalty_reml_and_gcv(), test_periodic_constraint(), test_tensor_product_basis(), test_ti_pure_interaction(), test_wood_constraint_validation()

### Community 35 - "pytest"
Cohesion: 0.06
Nodes (21): test_adversarial_and_edge_cases(), test_eilers_multiterm_weights_intervals_orders_vs_pygam(), test_eilers_untested_combinations_vs_pygam(), test_wood_advanced_combinations_vs_r(), test_wood_large_n_and_custom_knots_vs_r(), test_wood_oos_and_extrapolation_vs_r(), test_wood_spline_order_te_advanced_and_terms_vs_r(), test_wood_ti_all_bases_and_oos_vs_r() (+13 more)

### Community 36 - "_TensorTerm"
Cohesion: 0.22
Nodes (3): _kron_sum(), second_diff_penalty(), _TensorTerm

## Knowledge Gaps
- **19 isolated node(s):** `magli`, `Discrete Difference Penalty Matrix`, `Identifiability Constraints (Sum-to-Zero)`, `Wood Basis Zoo (cr, cs, tp, ps, te, ti)`, `magli Quickstart Workflows` (+14 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 274 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **21 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_GAMBase` connect `Base GAM Class & Model Diagnostics` to `wood/__init__.py`, `GAMCore`, `eilers/_gam.py`?**
  _High betweenness centrality (0.135) - this node is a cross-community bridge._
- **What connects `magli`, `Discrete Difference Penalty Matrix`, `Identifiability Constraints (Sum-to-Zero)` to the rest of the system?**
  _19 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `fit_wood_engine` be split into smaller, more focused modules?**
  _Cohesion score 0.14761904761904762 - nodes in this community are weakly interconnected._
- **Why does `Scripts` connect `Section-by-Section Guide` to `Base GAM Class & Model Diagnostics`?**
  _High betweenness centrality (0.054) - this node is a cross-community bridge._
- **Should `GAMCore` be split into smaller, more focused modules?**
  _Cohesion score 0.14153846153846153 - nodes in this community are weakly interconnected._
- **Should `Base GAM Class & Model Diagnostics` be split into smaller, more focused modules?**
  _Cohesion score 0.0966183574879227 - nodes in this community are weakly interconnected._
- **Should `GitHub Pages CI/CD Workflow` be split into smaller, more focused modules?**
  _Cohesion score 0.06439393939393939 - nodes in this community are weakly interconnected._