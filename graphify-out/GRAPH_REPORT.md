# Graph Report - magli  (2026-10-30)

## Corpus Check
- 46 files · ~57,371 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 10 file(s) not represented in the graph (top: (none) 4, .ipynb 3, .bat 2)

## Summary
- 593 nodes · 1183 edges · 41 communities (22 shown, 19 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 35 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d1b5f9bf`
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
- magli/__init__.py
- eilers/_terms.py
- PyPI Release Workflow
- Eilers Feature & Shape Constraint Tests
- steps
- run_r_gam
- _SplineTerm
- numpy
- parse_formula
- GAMCore
- eilers/__init__.py
- b_spline_basis
- 13_eilers_vs_wood_comparison.py
- Section-by-Section Guide
- Identifiability & Sum-to-Zero Constraints
- Wood Basis Functions Catalog
- GAM
- ndarray
- magli: Generalized Additive Models (GAMs) in Python
- 14_wood_basis_decomposition_plots.py
- Quickstart Documentation
- 2D Tensor Interaction Surface Visualization
- magli Root Package Overview
- 15_eilers_basis_decomposition_plots.py
- 04_shape_constraints.py
- GAMCore
- 10_engineering_case_study.py
- pytest
- test_wood_vs_r.py
- _TensorTerm
- _FactorTerm
- test_smoothcon_basis_and_penalty_vs_r
- test_all_univariate_bases
- r_mgcv_suite

## God Nodes (most connected - your core abstractions)
1. `_GAMBase` - 28 edges
2. `GAMCore` - 19 edges
3. `LinGAM` - 19 edges
4. `LinGAM` - 18 edges
5. `build_smooth_basis()` - 17 edges
6. `GAMCore` - 16 edges
7. `_Term` - 16 edges
8. `_SplineTerm` - 15 edges
9. `predict_smooth_basis()` - 14 edges
10. `run_r_gam()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `magli: Generalized Additive Models (GAMs) in Python` --references--> `GAM`  [INFERRED]
  README.md → src/magli/__init__.py
- `Observation Weights for Outlier Handling` --conceptually_related_to--> `test_weights_both_backends()`  [INFERRED]
  docs/_static/images/observation_weights_outliers.png → tests/test_api_parity.py
- `Unified Dispatch API (GAM / LinGAM / GAMCore)` --references--> `GAM`  [EXTRACTED]
  docs/api.rst → src/magli/__init__.py
- `Generalized Additive Models (GAMs)` --references--> `GAMCore`  [EXTRACTED]
  docs/math/foundations.rst → src/magli/__init__.py
- `GAMCore Additive Term Decomposition Plot` --references--> `GAMCore`  [EXTRACTED]
  docs/_static/images/gamcore_decomposition.png → src/magli/__init__.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **GAM Dynamic Dispatch & Parity System** — docs_api_unified_dispatch, src_magli_init_gam, src_magli_init_gamcore, src_magli_wood_lingam_lingam, src_magli_eilers_lingam_lingam [EXTRACTED 1.00]
- **Eilers & Marx P-Spline System** — docs_math_eilers_psplines, docs_math_eilers_difference_penalty, docs_math_eilers_shape_constraints, docs_static_images_eilers_bspline_decomposition [INFERRED 0.85]
- **Wood Penalized Regression Spline System** — docs_math_wood_penalized_regression_splines, docs_math_wood_basis_zoo, docs_math_wood_reml, src_magli_wood_lingam_lingam [INFERRED 0.85]

## Communities (41 total, 19 thin omitted)

### Community 0 - "fit_wood_engine"
Cohesion: 0.13
Nodes (8): _edf2_per_coef(), fit_wood_engine(), _liu2(), _mroot_t(), _psum_chisq(), wood_re_test(), wood_test_stat(), test_engine_drops_stale_aic_keys()

### Community 1 - "GAMCore"
Cohesion: 0.19
Nodes (4): _assemble_penalty_from_bases(), GAMCore, _eval_k_combo(), _solve_constrained()

### Community 3 - "eilers/_gam.py"
Cohesion: 0.21
Nodes (6): _build_matrix_from_terms(), _precompute_k(), _precompute_penalty_bases(), _set_lams_from_combo(), instantiate_terms(), _Term

### Community 4 - "GitHub Pages CI/CD Workflow"
Cohesion: 0.06
Nodes (32): runs-on, steps, concurrency, cancel-in-progress, group, environment, needs, runs-on (+24 more)

### Community 5 - "Wood Multi-term GAMCore & REML"
Cohesion: 0.14
Nodes (7): Two Philosophies for GAM Smoothing, Eilers & Marx P-Splines, Simon Wood Penalized Regression Splines, Restricted Maximum Likelihood (REML) Criterion, Eilers B-Spline Basis & P-Spline Decomposition, Eilers vs Wood Comparison Plot, LinGAM

### Community 6 - "Smoothing Parameter Tuning & Backend Comparison"
Cohesion: 0.19
Nodes (4): Smoothing Parameter Tuning & Grid Search, LinGAM Diagnostics 4-Panel Plot, GCV Curve and Effective Degrees of Freedom (EDF), LinGAM

### Community 8 - "eilers/_terms.py"
Cohesion: 0.21
Nodes (4): circular_second_diff_penalty(), first_diff_penalty(), ridge_penalty(), shape_constraint_penalty()

### Community 9 - "PyPI Release Workflow"
Cohesion: 0.11
Nodes (17): jobs, publish, name, on, release, workflow_dispatch, permissions, contents (+9 more)

### Community 10 - "Eilers Feature & Shape Constraint Tests"
Cohesion: 0.12
Nodes (7): test_bspline_linear_extrapolation(), test_directional_shape_constraints_enforced(), test_eilers_constrained_small_scale(), test_eilers_n_less_than_m_and_custom_gamma(), test_eilers_vs_pygam_parity(), test_periodic_bspline_partition_of_unity_and_continuity(), test_weighted_loglikelihood_includes_log_weights()

### Community 11 - "steps"
Cohesion: 0.10
Nodes (22): runs-on, steps, jobs, docs, test, python-version, name, on (+14 more)

### Community 12 - "run_r_gam"
Cohesion: 0.14
Nodes (7): data_and_r(), test_predict_terms_and_se_fit_vs_r(), test_random_effects_re_vs_r(), test_select_shrinkage_smooths(), test_summary_inference_vs_r(), test_varying_coefficient_by_vs_r(), run_r_gam()

### Community 13 - "_SplineTerm"
Cohesion: 0.18
Nodes (5): Shape Constraints API, Discrete Difference Penalty Matrix, Asymmetric Difference Penalty for Shape Constraints, Shape Constraints Showcase, _SplineTerm

### Community 14 - "numpy"
Cohesion: 0.06
Nodes (27): _khatri_rao(), GAMCore, absorb_sum_to_zero_constraint(), build_bs_basis(), build_cc_basis(), build_cp_basis(), build_cr_basis(), build_ps_basis() (+19 more)

### Community 15 - "parse_formula"
Cohesion: 0.31
Nodes (3): _eval_ast(), _flatten_plus(), parse_formula()

### Community 16 - "GAMCore"
Cohesion: 0.17
Nodes (5): Observation Weights for Outlier Handling, test_audit_fixes_comprehensive(), test_gamcore_lifecycle_and_methods(), test_lingam_lifecycle_and_methods(), test_weights_both_backends()

### Community 20 - "Section-by-Section Guide"
Cohesion: 0.09
Nodes (20): Example 10: Engineering Case Study README, How to Adapt This for Your Own Data, Key Takeaways, Quick Start, Section 10: Weighted vs Standard Comparison, Section 1: Data Summary, Section 2: Baseline Fit, Section 3: Grid Search (with new partial-grid flexibility) (+12 more)

### Community 23 - "GAM"
Cohesion: 0.18
Nodes (8): Unified Dispatch API (GAM / LinGAM / GAMCore), Generalized Additive Models (GAMs), Penalized Splines & Regularization, Regularization Penalty Lambda Effect, Why GAMs: The Interpretability-Flexibility Sweet Spot, _dispatch(), GAM, test_import_and_factory()

### Community 25 - "magli: Generalized Additive Models (GAMs) in Python"
Cohesion: 0.14
Nodes (13): 1. Wood (2011) Engine (Automatic REML Smoothing), 2. Eilers & Marx (1996) Engine (B-Splines & Constraints), Documentation, Features, Formula Syntax, [→ https://dacts23.github.io/magli/](https://dacts23.github.io/magli/), Installation, Interactive Tutorials & Demo Scripts (+5 more)

### Community 32 - "GAMCore"
Cohesion: 0.15
Nodes (7): GAMCore Additive Term Decomposition Plot, GAMCore, test_gridsearch_both_backends(), test_deviance_explained_accuracy(), test_eilers_edof_per_coef_and_wood_summary_pvalues(), test_eilers_gridsearch_diagonal_and_design_space(), test_nan_input_validation()

### Community 34 - "pytest"
Cohesion: 0.13
Nodes (7): test_gcv_method(), test_m_option_validated_and_coerced(), test_multi_penalty_reml_and_gcv(), test_periodic_constraint(), test_tensor_product_basis(), test_ti_pure_interaction(), test_wood_constraint_validation()

### Community 35 - "test_wood_vs_r.py"
Cohesion: 0.13
Nodes (4): test_fixed_sp_vs_r(), test_multiterm_reml_vs_r(), test_tensor_interaction_ti_vs_r(), test_tensor_product_vs_r()

### Community 36 - "_TensorTerm"
Cohesion: 0.22
Nodes (3): _kron_sum(), second_diff_penalty(), _TensorTerm

## Knowledge Gaps
- **18 isolated node(s):** `magli`, `Discrete Difference Penalty Matrix`, `Identifiability Constraints (Sum-to-Zero)`, `Wood Basis Zoo (cr, cs, tp, ps, te, ti)`, `magli Quickstart Workflows` (+13 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 260 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **19 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_GAMBase` connect `Base GAM Class & Model Diagnostics` to `GAMCore`, `eilers/_gam.py`, `numpy`, `Smoothing Parameter Tuning & Backend Comparison`?**
  _High betweenness centrality (0.137) - this node is a cross-community bridge._
- **What connects `magli`, `Discrete Difference Penalty Matrix`, `Identifiability Constraints (Sum-to-Zero)` to the rest of the system?**
  _18 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `fit_wood_engine` be split into smaller, more focused modules?**
  _Cohesion score 0.13043478260869565 - nodes in this community are weakly interconnected._
- **Why does `Scripts` connect `Section-by-Section Guide` to `Base GAM Class & Model Diagnostics`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Should `Base GAM Class & Model Diagnostics` be split into smaller, more focused modules?**
  _Cohesion score 0.0824829931972789 - nodes in this community are weakly interconnected._
- **Should `GitHub Pages CI/CD Workflow` be split into smaller, more focused modules?**
  _Cohesion score 0.06439393939393939 - nodes in this community are weakly interconnected._
- **Should `Wood Multi-term GAMCore & REML` be split into smaller, more focused modules?**
  _Cohesion score 0.14153846153846153 - nodes in this community are weakly interconnected._