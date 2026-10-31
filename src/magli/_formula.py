"""Formula string parser for Generalized Additive Model (GAM) terms."""

import ast
from typing import Any, Dict, List


def _eval_ast(node: ast.AST) -> Any:
    return node.id if isinstance(node, ast.Name) else ast.literal_eval(node)


def _flatten_plus(node: ast.AST) -> List[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _flatten_plus(node.left) + _flatten_plus(node.right)
    return [node]


def parse_formula(formula: str) -> List[Dict[str, Any]]:
    r"""Parse an additive model formula string into structured term specifications.

    Parses formula expressions composed of additive terms joined by ``+``,
    supporting univariate smooths (``s()``), tensor product smooths (``te()``),
    pure tensor interactions (``ti()``), categorical factor terms (``f()``), and
    linear terms (``l()``).

    Parameters
    ----------
    formula : str
        Formula string describing the additive model structure.
        Terms are separated by ``+``:

        - ``s(j, ...)`` : Univariate smooth of feature index `j`.
          Supports options: `k` (basis dimension), `bs` (basis family: 'tp', 'cr',
          'cs', 'cc', 'ps', 'cp', 'bs', 're'), `sp` / `lam` (fixed smoothing parameter),
          `m` (penalty order tuple), `by` (varying coefficient feature index),
          `constraint` ('periodic', 'monotonic_inc', 'monotonic_dec', 'convex', 'concave').
        - ``te(j1, j2, ...)`` : Full tensor product smooth of 2 or more features.
          Constructed via Kronecker products of marginal bases with multiple penalties.
        - ``ti(j1, j2, ...)`` : Pure tensor product interaction with marginal main
          effects removed, enabling functional ANOVA decomposition (e.g. ``s(0) + s(1) + ti(0, 1)``).
        - ``l(j)`` : Strictly linear parametric term :math:`\beta_j x_j`.
        - ``f(j, coding='dummy')`` : Categorical factor term with reference coding.

    Returns
    -------
    term_configs : list of dict
        List of parsed term dictionaries. Each dictionary contains:
        - ``'type'`` : str in ``{'s', 'te', 'ti', 'f', 'l'}``.
        - ``'features'`` : list of int indicating the 0-based column indices.
        - ``'kwargs'`` : dict of optional keyword arguments passed to the term.

    Raises
    ------
    ValueError
        If the formula is empty, has invalid Python syntax, specifies an unknown
        term type, or provides fewer feature arguments than required.

    See Also
    --------
    magli.GAMCore : Model class taking formula strings.

    Examples
    --------
    >>> from magli._formula import parse_formula
    >>> terms = parse_formula("s(0, k=12, bs='cr') + te(1, 2, k=8) + l(3)")
    >>> len(terms)
    3
    >>> terms[0]['type']
    's'
    >>> terms[0]['features']
    [0]
    >>> terms[0]['kwargs']
    {'k': 12, 'bs': 'cr'}
    >>> terms[1]['type']
    'te'
    >>> terms[1]['features']
    [1, 2]
    """
    if not formula or not formula.strip():
        raise ValueError("Formula string is empty.")
    try:
        tree = ast.parse(formula.strip(), mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"Cannot parse formula '{formula}'.") from exc

    result: List[Dict[str, Any]] = []
    for node in _flatten_plus(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            raise ValueError(f"Cannot parse term in '{formula}'.")
        func_name = node.func.id.lower()
        if func_name not in ("s", "te", "ti", "f", "l"):
            raise ValueError(
                f"Unknown term type '{func_name}'. Expected s, te, ti, f, or l."
            )
        positional = [_eval_ast(a) for a in node.args]
        kwargs = {kw.arg: _eval_ast(kw.value) for kw in node.keywords if kw.arg}
        min_feats = 2 if func_name in ("te", "ti") else 1
        if len(positional) < min_feats:
            raise ValueError(
                f"'{func_name}()' requires at least {min_feats} feature index(es)."
            )
        result.append(
            {
                "type": func_name,
                "features": positional[:1] if min_feats == 1 else list(positional),
                "kwargs": kwargs,
            }
        )
    return result
