# Graph Report - magli  (2026-10-30)

## Corpus Check
- 45 files · ~56,345 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 10 file(s) not represented in the graph (top: (none) 4, .ipynb 3, .bat 2)

## Summary
- 568 nodes · 1150 edges · 27 communities (18 shown, 9 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 33 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `816e01ec`
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
- Core Tutorial Examples
- eilers/_terms.py
- PyPI Release Workflow
- Eilers Feature & Shape Constraint Tests
- Wood Basis Families Tests
- pytest
- _SplineTerm
- numpy
- parse_formula
- _TensorTerm
- eilers/__init__.py
- b_spline_basis
- Section-by-Section Guide
- Identifiability & Sum-to-Zero Constraints
- Wood Basis Functions Catalog
- ndarray
- magli: Generalized Additive Models (GAMs) in Python
- Quickstart Documentation
- 2D Tensor Interaction Surface Visualization
- magli Root Package Overview

## God Nodes (most connected - your core abstractions)
1. `_GAMBase` - 28 edges
2. `GAMCore` - 19 edges
3. `LinGAM` - 19 edges
4. `LinGAM` - 18 edges
5. `build_smooth_basis()` - 17 edges
6. `_Term` - 16 edges
7. `GAMCore` - 15 edges
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
- `Generalized Additive Models (GAMs)` --references--> `GAM`  [EXTRACTED]
  docs/math/foundations.rst → src/magli/__init__.py
- `GAMCore Additive Term Decomposition Plot` --references--> `GAMCore`  [EXTRACTED]
  docs/_static/images/gamcore_decomposition.png → src/magli/__init__.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **GAM Dynamic Dispatch & Parity System** — docs_api_unified_dispatch, src_magli_init_gam, src_magli_init_gamcore, src_magli_wood_lingam_lingam, src_magli_eilers_lingam_lingam [EXTRACTED 1.00]
- **Eilers & Marx P-Spline System** — docs_math_eilers_psplines, docs_math_eilers_difference_penalty, docs_math_eilers_shape_constraints, docs_static_images_eilers_bspline_decomposition [INFERRED 0.85]
- **Wood Penalized Regression Spline System** — docs_math_wood_penalized_regression_splines, docs_math_wood_basis_zoo, docs_math_wood_reml, src_magli_wood_lingam_lingam [INFERRED 0.85]

## Communities (27 total, 9 thin omitted)

### Community 0 - "fit_wood_engine"
Cohesion: 0.14
Nodes (7): _edf2_per_coef(), fit_wood_engine(), _liu2(), _mroot_t(), _psum_chisq(), wood_re_test(), wood_test_stat()

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
Cohesion: 0.17
Nodes (4): Smoothing Parameter Tuning & Grid Search, Restricted Maximum Likelihood (REML) Criterion, GCV Curve and Effective Degrees of Freedom (EDF), LinGAM

### Community 6 - "Smoothing Parameter Tuning & Backend Comparison"
Cohesion: 0.17
Nodes (7): Two Philosophies for GAM Smoothing, Eilers & Marx P-Splines, Simon Wood Penalized Regression Splines, LinGAM Diagnostics 4-Panel Plot, Eilers B-Spline Basis & P-Spline Decomposition, Eilers vs Wood Comparison Plot, LinGAM

### Community 7 - "Core Tutorial Examples"
Cohesion: 0.06
Nodes (9): Unified Dispatch API (GAM / LinGAM / GAMCore), plot_fit(), sigmoid(), _concrete_demo(), run_comparison(), wood_test_function(), wood_test_function(), _dispatch() (+1 more)

### Community 8 - "eilers/_terms.py"
Cohesion: 0.21
Nodes (4): circular_second_diff_penalty(), first_diff_penalty(), ridge_penalty(), shape_constraint_penalty()

### Community 9 - "PyPI Release Workflow"
Cohesion: 0.11
Nodes (17): jobs, publish, name, on, release, workflow_dispatch, permissions, contents (+9 more)

### Community 10 - "Eilers Feature & Shape Constraint Tests"
Cohesion: 0.06
Nodes (18): Generalized Additive Models (GAMs), Penalized Splines & Regularization, GAMCore Additive Term Decomposition Plot, Regularization Penalty Lambda Effect, Why GAMs: The Interpretability-Flexibility Sweet Spot, GAMCore, test_gridsearch_both_backends(), test_bspline_linear_extrapolation() (+10 more)

### Community 11 - "Wood Basis Families Tests"
Cohesion: 0.10
Nodes (9): test_all_univariate_bases(), test_engine_drops_stale_aic_keys(), test_gcv_method(), test_m_option_validated_and_coerced(), test_multi_penalty_reml_and_gcv(), test_periodic_constraint(), test_tensor_product_basis(), test_ti_pure_interaction() (+1 more)

### Community 12 - "pytest"
Cohesion: 0.05
Nodes (19): Observation Weights for Outlier Handling, test_gamcore_lifecycle_and_methods(), test_import_and_factory(), test_lingam_lifecycle_and_methods(), test_weights_both_backends(), data_and_r(), test_predict_terms_and_se_fit_vs_r(), test_random_effects_re_vs_r() (+11 more)

### Community 13 - "_SplineTerm"
Cohesion: 0.18
Nodes (5): Shape Constraints API, Discrete Difference Penalty Matrix, Asymmetric Difference Penalty for Shape Constraints, Shape Constraints Showcase, _SplineTerm

### Community 14 - "numpy"
Cohesion: 0.06
Nodes (27): _khatri_rao(), GAMCore, absorb_sum_to_zero_constraint(), build_bs_basis(), build_cc_basis(), build_cp_basis(), build_cr_basis(), build_ps_basis() (+19 more)

### Community 15 - "parse_formula"
Cohesion: 0.31
Nodes (3): _eval_ast(), _flatten_plus(), parse_formula()

### Community 16 - "_TensorTerm"
Cohesion: 0.22
Nodes (3): _kron_sum(), second_diff_penalty(), _TensorTerm

### Community 20 - "Section-by-Section Guide"
Cohesion: 0.09
Nodes (20): Example 10: Engineering Case Study README, How to Adapt This for Your Own Data, Key Takeaways, Quick Start, Section 10: Weighted vs Standard Comparison, Section 1: Data Summary, Section 2: Baseline Fit, Section 3: Grid Search (with new partial-grid flexibility) (+12 more)

### Community 25 - "magli: Generalized Additive Models (GAMs) in Python"
Cohesion: 0.14
Nodes (13): 1. Wood (2011) Engine (Automatic REML Smoothing), 2. Eilers & Marx (1996) Engine (B-Splines & Constraints), Documentation, Features, Formula Syntax, [→ https://dacts23.github.io/magli/](https://dacts23.github.io/magli/), Installation, Interactive Tutorials & Demo Scripts (+5 more)

## Knowledge Gaps
- **18 isolated node(s):** `magli`, `Identifiability Constraints (Sum-to-Zero)`, `Wood Basis Zoo (cr, cs, tp, ps, te, ti)`, `Discrete Difference Penalty Matrix`, `magli Quickstart Workflows` (+13 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 251 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_GAMBase` connect `Base GAM Class & Model Diagnostics` to `GAMCore`, `eilers/_gam.py`, `numpy`?**
  _High betweenness centrality (0.149) - this node is a cross-community bridge._
- **What connects `magli`, `Identifiability Constraints (Sum-to-Zero)`, `Wood Basis Zoo (cr, cs, tp, ps, te, ti)` to the rest of the system?**
  _18 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `fit_wood_engine` be split into smaller, more focused modules?**
  _Cohesion score 0.14130434782608695 - nodes in this community are weakly interconnected._
- **Why does `Scripts` connect `Section-by-Section Guide` to `Base GAM Class & Model Diagnostics`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Should `Base GAM Class & Model Diagnostics` be split into smaller, more focused modules?**
  _Cohesion score 0.08081632653061224 - nodes in this community are weakly interconnected._
- **Should `GitHub Pages CI/CD Workflow` be split into smaller, more focused modules?**
  _Cohesion score 0.06439393939393939 - nodes in this community are weakly interconnected._
- **Should `Core Tutorial Examples` be split into smaller, more focused modules?**
  _Cohesion score 0.060129509713228495 - nodes in this community are weakly interconnected._