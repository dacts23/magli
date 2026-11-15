"""Term classes for Wood GAM backend: spline, tensor product, factor, and linear terms."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .._base import _khatri_rao
from ..eilers._terms import _FactorTerm, _LinearTerm, _Term as _WoodTerm
from ._smooths import (
    absorb_sum_to_zero_constraint,
    build_smooth_basis,
    predict_smooth_basis,
    rescale_penalty,
)


class _WoodSplineTerm(_WoodTerm):
    """Univariate smooth term s(x, bs='tp', k=10)."""

    def __init__(
        self,
        feature: int,
        k: int = 10,
        bs: str = "tp",
        sp: Optional[float] = None,
        m: Optional[Union[int, Tuple[int, int]]] = None,
        knots: Optional[np.ndarray] = None,
        constraint: Optional[str] = None,
        by: Optional[int] = None,
        select: bool = False,
        absorb_cons: bool = True,
    ) -> None:
        self.feature = int(feature)
        self.k = int(k)
        if self.k < 3:
            raise ValueError("n_splines (k) must be at least 3.")
        self.bs = bs.lower()
        self.select = bool(select)
        self.absorb_cons = bool(absorb_cons)
        if constraint is not None and constraint != "periodic":
            raise ValueError(
                f"Unsupported constraint '{constraint}' for Wood backend. "
                "Wood backend natively supports 'periodic' (bs='cc'). For soft shape "
                "constraints (monotonicity, convexity), use backend='eilers'."
            )
        if constraint == "periodic" and self.bs not in ("cc", "cp"):
            self.bs = "cc"
        if sp is not None and float(sp) < 0:
            raise ValueError("Smoothing parameter 'lam' / 'sp' must be non-negative.")
        self.sp = [float(sp)] if sp is not None else None
        self._user_sp: Optional[List[Optional[float]]] = (
            list(self.sp) if self.sp is not None else None
        )
        self.m = m
        self.knots = knots
        if by is not None and (
            not isinstance(by, (int, np.integer))
            or isinstance(by, bool)
            or int(by) < 0
        ):
            raise ValueError("by must be a non-negative integer feature index.")
        self.by = int(by) if by is not None else None
        self._Z: Optional[np.ndarray] = None
        self._penalties: List[np.ndarray] = []
        self._meta: Dict[str, Any] = {}
        self._cached_cols: Optional[np.ndarray] = None
        self._cached_X: Optional[np.ndarray] = None
        self.n_coefs = 0

    def _extract_cols(self, x: np.ndarray) -> np.ndarray:
        cols = [self.feature] + ([self.by] if self.by is not None else [])
        return (
            np.asarray(x[:, cols])
            if self.bs == "re"
            else np.asarray(x[:, cols], dtype=float)
        )

    def compile(self, x: np.ndarray) -> None:
        col = (
            np.asarray(x[:, self.feature]).ravel()
            if self.bs == "re"
            else np.asarray(x[:, self.feature], dtype=float).ravel()
        )
        X_raw, S_list, self._meta = build_smooth_basis(
            col, bs=self.bs, k=self.k, knots=self.knots, m=self.m
        )
        S_scaled, _ = rescale_penalty(X_raw, S_list)
        if self.by is not None:
            by_col = np.asarray(x[:, self.by], dtype=float).ravel()
            X_raw = X_raw * by_col[:, None]

        if self.by is not None or self.bs == "re" or not self.absorb_cons:
            X_abs = X_raw
            self._penalties = S_scaled
            self._Z = np.eye(X_raw.shape[1])
        else:
            X_abs, self._penalties, self._Z = absorb_sum_to_zero_constraint(
                X_raw, S_scaled
            )
        if self.select and len(self._penalties) == 1:
            es_val, es_vec = np.linalg.eigh(self._penalties[0])
            pos = es_val > es_val[-1] * 1e-10
            if np.any(pos) and not np.all(pos):
                es_val[~pos] = float(np.min(es_val[pos])) * 0.1
                S_sh = es_vec @ (es_val[:, None] * es_vec.T)
                self._penalties = [0.5 * (S_sh + S_sh.T)]
        self.n_coefs = X_abs.shape[1]
        self._cached_cols = self._extract_cols(x)
        self._cached_X = X_abs

    def build_columns(self, x: np.ndarray) -> np.ndarray:
        col = (
            np.asarray(x[:, self.feature]).ravel()
            if self.bs == "re"
            else np.asarray(x[:, self.feature], dtype=float).ravel()
        )
        if self._Z is None:
            self.compile(x)
        curr_cols = self._extract_cols(x)
        if (
            self._cached_cols is not None
            and curr_cols.shape == self._cached_cols.shape
            and np.array_equal(curr_cols, self._cached_cols)
        ):
            return self._cached_X  # type: ignore[return-value]
        B = predict_smooth_basis(col, self._meta) @ self._Z
        if self.by is not None:
            by_col = np.asarray(x[:, self.by], dtype=float).ravel()
            B = B * by_col[:, None]
        return B

    def build_penalties(self) -> List[np.ndarray]:
        return [p.copy() for p in self._penalties]


class _WoodTensorTerm(_WoodTerm):
    """Tensor product smooth te(x1, x2, bs='cr', k=5), or ti() pure interaction."""

    def __init__(
        self,
        features: List[int],
        k: Union[int, List[int]] = 5,
        bs: Union[str, List[str]] = "cr",
        sp: Optional[Union[float, List[float]]] = None,
        m: Optional[Union[int, List[Any]]] = None,
        knots: Optional[Union[np.ndarray, List[Any]]] = None,
        interaction: bool = False,
        by: Optional[int] = None,
        absorb_cons: bool = True,
    ) -> None:
        self.interaction = interaction
        self.absorb_cons = bool(absorb_cons)
        self.kind = "ti" if interaction else "te"
        self.features = [int(f) for f in features]
        self.d = len(self.features)
        if self.d < 2:
            raise ValueError(
                "Tensor product term requires at least 2 features."
            )

        if isinstance(k, (int, np.integer)):
            self.k_per = [int(k)] * self.d
        else:
            self.k_per = [int(v) for v in k]
            if len(self.k_per) != self.d:
                raise ValueError(
                    f"k length ({len(self.k_per)}) must match number of features ({self.d})."
                )
        if any(v < 3 for v in self.k_per):
            raise ValueError("n_splines (k) must be at least 3.")

        if isinstance(bs, str):
            self.bs_per = [bs.lower()] * self.d
        else:
            self.bs_per = [str(v).lower() for v in bs]
            if len(self.bs_per) != self.d:
                raise ValueError(
                    f"bs length ({len(self.bs_per)}) must match number of features ({self.d})."
                )

        if sp is None:
            self.sp = None
        elif isinstance(sp, (int, float, np.number)):
            self.sp = [float(sp)] * self.d
        else:
            self.sp = [float(v) if v is not None else None for v in sp]
            if len(self.sp) != self.d:
                raise ValueError(
                    f"sp length ({len(self.sp)}) must match number of features ({self.d})."
                )
        if self.sp is not None and any(v is not None and v < 0 for v in self.sp):
            raise ValueError("Smoothing parameter 'lam' / 'sp' must be non-negative.")
        self._user_sp: Optional[List[Optional[float]]] = (
            list(self.sp) if self.sp is not None else None
        )
        if by is not None and (
            not isinstance(by, (int, np.integer))
            or isinstance(by, bool)
            or int(by) < 0
        ):
            raise ValueError("by must be a non-negative integer feature index.")
        self.by = int(by) if by is not None else None
        if m is None:
            self.m_per = [None] * self.d
        elif isinstance(m, (list, tuple)):
            self.m_per = list(m)
        else:
            self.m_per = [m] * self.d

        if knots is None:
            self.knots_per = [None] * self.d
        elif isinstance(knots, (list, tuple)):
            self.knots_per = list(knots)
        else:
            self.knots_per = [knots] * self.d

        self._Z: Optional[np.ndarray] = None
        self._penalties: List[np.ndarray] = []
        self._marginal_meta: List[Dict] = []
        self._marginal_Z: List[np.ndarray] = []
        self._cached_cols: Optional[np.ndarray] = None
        self._cached_X: Optional[np.ndarray] = None
        self.n_coefs = 0

    def compile(self, x: np.ndarray) -> None:
        marg_X, marg_S, self._marginal_meta, self._marginal_Z = [], [], [], []
        cols = np.asarray(x[:, self.features], dtype=float)
        for i in range(self.d):
            col = cols[:, i].ravel()
            Xi, Si, meta_i = build_smooth_basis(
                col,
                bs=self.bs_per[i],
                k=self.k_per[i],
                m=self.m_per[i],
                knots=self.knots_per[i],
            )
            ev_max = np.linalg.eigvalsh(Si[0])[-1]
            if ev_max > 0:
                Si[0] = Si[0] / ev_max
            if self.interaction:  # mgcv ti(): centre each margin so main effects are excluded
                Xi, Si, Zi = absorb_sum_to_zero_constraint(Xi, Si)
                self._marginal_Z.append(Zi)
            marg_X.append(Xi)
            marg_S.append(Si[0])
            self._marginal_meta.append(meta_i)

        X_te = _khatri_rao(marg_X)
        sizes = [m.shape[1] for m in marg_X]
        TS_list = []
        for i in range(self.d):
            I_pre = np.eye(int(np.prod(sizes[:i])))
            I_post = np.eye(int(np.prod(sizes[i + 1 :])))
            P_i = np.kron(np.kron(I_pre, marg_S[i]), I_post)
            TS_list.append(0.5 * (P_i + P_i.T))

        S_scaled, _ = rescale_penalty(X_te, TS_list)
        if self.by is not None:
            by_col = np.asarray(x[:, self.by], dtype=float).ravel()
            X_te = X_te * by_col[:, None]

        if self.interaction or self.by is not None or not self.absorb_cons:
            X_abs, self._penalties = X_te, S_scaled
            self._Z = np.eye(X_te.shape[1])
        else:
            X_abs, self._penalties, self._Z = absorb_sum_to_zero_constraint(
                X_te, S_scaled
            )
        self.n_coefs = X_abs.shape[1]
        self._cached_cols = np.asarray(x[:, self.features + ([self.by] if self.by is not None else [])], dtype=float)
        self._cached_X = X_abs

    def build_columns(self, x: np.ndarray) -> np.ndarray:
        if self._Z is None:
            self.compile(x)
        curr_cols = np.asarray(x[:, self.features + ([self.by] if self.by is not None else [])], dtype=float)
        if (
            self._cached_cols is not None
            and curr_cols.shape == self._cached_cols.shape
            and np.array_equal(curr_cols, self._cached_cols)
        ):
            return self._cached_X  # type: ignore[return-value]
        cols = np.asarray(x[:, self.features], dtype=float)
        marg_X = [
            predict_smooth_basis(cols[:, i].ravel(), self._marginal_meta[i])
            for i in range(self.d)
        ]
        if self.interaction:
            marg_X = [B @ Z for B, Z in zip(marg_X, self._marginal_Z)]
        res = _khatri_rao(marg_X) @ self._Z
        if self.by is not None:
            by_col = np.asarray(x[:, self.by], dtype=float).ravel()
            res = res * by_col[:, None]
        return res

    def build_penalties(self) -> List[np.ndarray]:
        return [p.copy() for p in self._penalties]


class _WoodFactorTerm(_FactorTerm):
    """Categorical factor term f(feature)."""

    def __init__(self, feature: int, coding: str = "dummy") -> None:
        super().__init__(feature, lam=0.0, coding=coding)


_WoodLinearTerm = _LinearTerm


def instantiate_wood_terms(
    configs: List[Dict[str, Any]], select: bool = False, fit_intercept: bool = True
) -> List[_WoodTerm]:
    """Instantiate Wood term objects from parsed formula config dicts."""
    terms: List[_WoodTerm] = []
    has_dropped_cons = fit_intercept
    for cfg in configs:
        kind = cfg["type"]
        feats = cfg["features"]
        kw = dict(cfg["kwargs"])

        default_k = 5 if kind in ("te", "ti") else 10
        default_bs = "cr" if kind in ("te", "ti") else "tp"
        k_val = kw.pop("k", kw.pop("n_splines", default_k))
        sp_val = kw.pop("sp", kw.pop("lam", None))
        bs_val = kw.pop("bs", default_bs)
        by_val = kw.pop("by", None)

        if select:
            if isinstance(bs_val, str):
                bs_val = "ts" if bs_val == "tp" else ("cs" if bs_val == "cr" else bs_val)
            elif isinstance(bs_val, (list, tuple)):
                bs_val = ["ts" if b == "tp" else ("cs" if b == "cr" else b) for b in bs_val]

        if kind == "s":
            absorb = True
            if not has_dropped_cons and by_val is None and bs_val != "re":
                absorb = False
                has_dropped_cons = True
            terms.append(
                _WoodSplineTerm(
                    feature=feats[0],
                    k=k_val,
                    bs=bs_val,
                    sp=sp_val,
                    m=kw.pop("m", None),
                    knots=kw.pop("knots", None),
                    constraint=kw.pop("constraint", None),
                    by=by_val,
                    select=select,
                    absorb_cons=absorb,
                )
            )
        elif kind in ("te", "ti"):
            absorb = True
            if not has_dropped_cons and kind == "te" and by_val is None:
                absorb = False
                has_dropped_cons = True
            terms.append(
                _WoodTensorTerm(
                    features=feats,
                    k=k_val,
                    bs=bs_val,
                    sp=sp_val,
                    m=kw.pop("m", None),
                    knots=kw.pop("knots", None),
                    interaction=kind == "ti",
                    by=by_val,
                    absorb_cons=absorb,
                )
            )
        elif kind == "f":
            coding = kw.pop("coding", "dummy")
            if coding == "one-hot":
                if has_dropped_cons:
                    raise ValueError(
                        "f(..., coding='one-hot') in the Wood backend is collinear with the intercept; "
                        "use coding='dummy', fit_intercept=False, or s(..., bs='re')."
                    )
                has_dropped_cons = True
            terms.append(
                _WoodFactorTerm(
                    feature=feats[0], coding=coding
                )
            )
        elif kind == "l":
            terms.append(_WoodLinearTerm(feature=feats[0]))
        else:
            raise ValueError(f"Unknown term type '{kind}' in formula.")
        if kw:
            raise ValueError(f"Unrecognised kwargs for {kind}(): {list(kw)}")
    return terms
