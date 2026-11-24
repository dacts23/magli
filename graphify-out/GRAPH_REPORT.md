# Graph Report - magli  (2026-11-24)

## Corpus Check
- 48 files · ~64,972 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 10 file(s) not represented in the graph (top: (none) 4, .ipynb 3, .bat 2)

## Summary
- 692 nodes · 1426 edges · 40 communities (28 shown, 12 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 29 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `53588fbd`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- _tests.py
- eilers/_gam.py
- _GAMBase
- _Term
- GitHub Pages CI/CD Workflow
- Wood Multi-term GAMCore & REML
- LinGAM
- eilers/__init__.py
- plot_gamcore_decomposition
- PyPI Release Workflow
- build_cr_basis
- steps
- _optimizer.py
- numpy
- parse_formula
- test_api_parity.py
- _TensorTerm
- _smooths.py
- _penalty.py
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
- build_smooth_basis
- _SplineTerm
- _FactorTerm
- ndarray
- pytest
- pytest
- b_spline_basis
- .build_constraint_penalty
- Unified Dispatch API (GAM / LinGAM / GAMCore)
- GAMCore Additive Term Decomposition Plot

## God Nodes (most connected - your core abstractions)
1. `_GAMBase` - 34 edges
2. `run_r_gam()` - 21 edges
3. `LinGAM` - 18 edges
4. `LinGAM` - 18 edges
5. `GAMCore` - 17 edges
6. `build_smooth_basis()` - 17 edges
7. `predict_smooth_basis()` - 16 edges
8. `_Term` - 15 edges
9. `_SplineTerm` - 15 edges
10. `fit_wood_engine()` - 15 edges

## Surprising Connections (you probably didn't know these)
- `Observation Weights for Outlier Handling` --conceptually_related_to--> `test_weights_both_backends()`  [INFERRED]
  docs/_static/images/observation_weights_outliers.png → tests/test_api_parity.py
- `Eilers & Marx P-Splines` --implements--> `LinGAM`  [EXTRACTED]
  docs/math/eilers.rst → src/magli/eilers/_lingam.py
- `Simon Wood Penalized Regression Splines` --implements--> `LinGAM`  [EXTRACTED]
  docs/math/wood.rst → src/magli/wood/_lingam.py
- `test_engine_drops_stale_aic_keys()` --calls--> `fit_wood_engine()`  [INFERRED]
  tests/test_wood_bases.py → src/magli/wood/_optimizer.py
- `test_wood_chunked_tp_matches_direct()` --calls--> `predict_smooth_basis()`  [INFERRED]
  tests/test_wood_bases.py → src/magli/wood/_smooths.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Eilers & Marx P-Spline System** — docs_math_eilers_psplines, docs_math_eilers_difference_penalty, docs_math_eilers_shape_constraints, docs_static_images_eilers_bspline_decomposition [INFERRED 0.85]
- **Wood Penalized Regression Spline System** — docs_math_wood_penalized_regression_splines, docs_math_wood_basis_zoo, docs_math_wood_reml, src_magli_wood_lingam_lingam [INFERRED 0.85]

## Communities (40 total, 12 thin omitted)

### Community 0 - "_tests.py"
Cohesion: 0.24
Nodes (5): _liu2(), _mroot_t(), _psum_chisq(), wood_re_test(), wood_test_stat()

### Community 1 - "eilers/_gam.py"
Cohesion: 0.16
Nodes (3): _validate_weights(), _validate_xyw(), GAMCore

### Community 2 - "_GAMBase"
Cohesion: 0.07
Nodes (5): _GAMBase, _assemble_penalty_from_bases(), GAMCore, _eval_k_combo(), _solve_constrained()

### Community 3 - "_Term"
Cohesion: 0.22
Nodes (5): _precompute_k(), _precompute_penalty_bases(), _set_lams_from_combo(), instantiate_terms(), _Term

### Community 4 - "GitHub Pages CI/CD Workflow"
Cohesion: 0.06
Nodes (32): runs-on, steps, concurrency, cancel-in-progress, group, environment, needs, runs-on (+24 more)

### Community 5 - "Wood Multi-term GAMCore & REML"
Cohesion: 0.14
Nodes (7): Two Philosophies for GAM Smoothing, Eilers & Marx P-Splines, Simon Wood Penalized Regression Splines, Restricted Maximum Likelihood (REML) Criterion, Eilers B-Spline Basis & P-Spline Decomposition, Eilers vs Wood Comparison Plot, LinGAM

### Community 6 - "LinGAM"
Cohesion: 0.19
Nodes (4): Smoothing Parameter Tuning & Grid Search, LinGAM Diagnostics 4-Panel Plot, GCV Curve and Effective Degrees of Freedom (EDF), LinGAM

### Community 7 - "eilers/__init__.py"
Cohesion: 0.06
Nodes (8): plot_fit(), sigmoid(), _concrete_demo(), run_comparison(), wood_test_function(), wood_test_function(), _pirls_factors(), solve_pirls()

### Community 8 - "plot_gamcore_decomposition"
Cohesion: 0.21
Nodes (6): _import_pyplot(), plot_diagnostics(), plot_gamcore_decomposition(), plot_lingam_decomposition(), _profile_predictors(), _shade_extrap()

### Community 9 - "PyPI Release Workflow"
Cohesion: 0.11
Nodes (17): jobs, publish, name, on, release, workflow_dispatch, permissions, contents (+9 more)

### Community 10 - "build_cr_basis"
Cohesion: 0.33
Nodes (3): build_cr_basis(), _eval_cr_basis(), _get_cr_FS()

### Community 11 - "steps"
Cohesion: 0.10
Nodes (22): runs-on, steps, jobs, docs, test, python-version, name, on (+14 more)

### Community 12 - "_optimizer.py"
Cohesion: 0.18
Nodes (6): _fit_summary_stats(), _compute_term_tests(), _edf2_per_coef(), fit_wood_engine(), _init_rho(), _setup_penalties()

### Community 14 - "numpy"
Cohesion: 0.09
Nodes (14): test_bspline_linear_extrapolation(), test_deviance_explained_accuracy(), test_directional_shape_constraints_enforced(), test_eilers_constrained_and_high_lam_noise_pvalues(), test_eilers_constrained_small_scale(), test_eilers_edof_per_coef_and_wood_summary_pvalues(), test_eilers_gridsearch_diagonal_and_design_space(), test_eilers_gridsearch_partial_validation_and_parallel() (+6 more)

### Community 15 - "parse_formula"
Cohesion: 0.20
Nodes (4): _eval_ast(), _flatten_plus(), parse_formula(), test_audit_regressions_fixed()

### Community 16 - "test_api_parity.py"
Cohesion: 0.07
Nodes (12): Observation Weights for Outlier Handling, test_3d_predict_lam_reset_gamma_validation_and_docstrings(), test_deviance_explained_matches_stats_on_tiny_y(), test_gamcore_lifecycle_and_methods(), test_gridsearch_both_backends(), test_import_and_factory(), test_lingam_lifecycle_and_methods(), test_predict_wrong_1d_length_raises_value_error() (+4 more)

### Community 17 - "_TensorTerm"
Cohesion: 0.22
Nodes (3): _kron_sum(), second_diff_penalty(), _TensorTerm

### Community 18 - "_smooths.py"
Cohesion: 0.15
Nodes (11): build_bs_basis(), build_cc_basis(), build_cp_basis(), build_ps_basis(), _c_spline_des(), _cwrap(), _eta_1d_matmul(), _eval_cc_basis() (+3 more)

### Community 19 - "_penalty.py"
Cohesion: 0.24
Nodes (3): circular_second_diff_penalty(), ridge_penalty(), shape_constraint_penalty()

### Community 20 - "Section-by-Section Guide"
Cohesion: 0.09
Nodes (20): Example 10: Engineering Case Study README, How to Adapt This for Your Own Data, Key Takeaways, Quick Start, Section 10: Weighted vs Standard Comparison, Section 1: Data Summary, Section 2: Baseline Fit, Section 3: Grid Search (with new partial-grid flexibility) (+12 more)

### Community 23 - "Generalized Additive Models (GAMs)"
Cohesion: 0.50
Nodes (4): Generalized Additive Models (GAMs), Penalized Splines & Regularization, Regularization Penalty Lambda Effect, Why GAMs: The Interpretability-Flexibility Sweet Spot

### Community 24 - "wood/__init__.py"
Cohesion: 0.10
Nodes (12): _khatri_rao(), _kron_blocks(), absorb_sum_to_zero_constraint(), rescale_penalty(), _append_select_null_penalty(), _cached_columns(), _default_spline_m_1d(), instantiate_wood_terms() (+4 more)

### Community 25 - "magli: Generalized Additive Models (GAMs) in Python"
Cohesion: 0.14
Nodes (13): 1. Wood (2011) Engine (Automatic REML Smoothing), 2. Eilers & Marx (1996) Engine (B-Splines & Constraints), Documentation, Features, Formula Syntax, [→ https://dacts23.github.io/magli/](https://dacts23.github.io/magli/), Installation, Interactive Tutorials & Demo Scripts (+5 more)

### Community 26 - "build_tp_basis"
Cohesion: 0.14
Nodes (8): build_tp_basis(), _eta_1d(), _place_knots(), _top_k_eig(), _wood_qt(), test_audit_fixes_comprehensive(), test_ts_shrinks_null_space(), test_wood_chunked_tp_matches_direct()

### Community 30 - "build_smooth_basis"
Cohesion: 0.17
Nodes (5): build_smooth_basis(), _pair_m(), test_m_option_validated_and_coerced(), test_all_bases_reml_and_gcv_vs_r(), test_smoothcon_basis_and_penalty_vs_r()

### Community 34 - "pytest"
Cohesion: 0.07
Nodes (14): test_all_univariate_bases(), test_cyclic_bases_reject_small_k(), test_engine_drops_stale_aic_keys(), test_gcv_method(), test_multi_penalty_reml_and_gcv(), test_nan_prediction_propagates_for_bspline_bases(), test_periodic_constraint(), test_tensor_mismatched_sequence_lengths_raise() (+6 more)

### Community 35 - "pytest"
Cohesion: 0.06
Nodes (21): test_adversarial_and_edge_cases(), test_eilers_multiterm_weights_intervals_orders_vs_pygam(), test_eilers_untested_combinations_vs_pygam(), test_wood_advanced_combinations_vs_r(), test_wood_large_n_and_custom_knots_vs_r(), test_wood_oos_and_extrapolation_vs_r(), test_wood_spline_order_te_advanced_and_terms_vs_r(), test_wood_ti_all_bases_and_oos_vs_r() (+13 more)

### Community 37 - ".build_constraint_penalty"
Cohesion: 0.50
Nodes (3): Shape Constraints API, Asymmetric Difference Penalty for Shape Constraints, Shape Constraints Showcase

## Knowledge Gaps
- **19 isolated node(s):** `magli`, `Identifiability Constraints (Sum-to-Zero)`, `Wood Basis Zoo (cr, cs, tp, ps, te, ti)`, `magli Quickstart Workflows`, `Discrete Difference Penalty Matrix` (+14 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 305 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_GAMBase` connect `_GAMBase` to `eilers/_gam.py`, `.fit`, `LinGAM`?**
  _High betweenness centrality (0.128) - this node is a cross-community bridge._
- **What connects `magli`, `Identifiability Constraints (Sum-to-Zero)`, `Wood Basis Zoo (cr, cs, tp, ps, te, ti)` to the rest of the system?**
  _19 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `_GAMBase` be split into smaller, more focused modules?**
  _Cohesion score 0.06547619047619048 - nodes in this community are weakly interconnected._
- **Why does `Scripts` connect `Section-by-Section Guide` to `_GAMBase`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Should `GitHub Pages CI/CD Workflow` be split into smaller, more focused modules?**
  _Cohesion score 0.06439393939393939 - nodes in this community are weakly interconnected._
- **Should `Wood Multi-term GAMCore & REML` be split into smaller, more focused modules?**
  _Cohesion score 0.14461538461538462 - nodes in this community are weakly interconnected._
- **Should `eilers/__init__.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05782312925170068 - nodes in this community are weakly interconnected._