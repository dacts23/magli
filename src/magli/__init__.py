"""magli (Modelo Aditivo Generalizado Lineal): Pythonic Generalized Additive Models (GAMs).

Choose a backend explicitly via :mod:`magli.wood` (Simon Wood's penalized
regression splines with REML/GCV optimization) or :mod:`magli.eilers`
(Paul Eilers & Brian Marx's P-splines with discrete difference penalties
and shape constraints).
"""

from . import eilers, wood

__version__ = "2.1.0"

__all__ = [
    "eilers",
    "wood",
    "__version__",
]
