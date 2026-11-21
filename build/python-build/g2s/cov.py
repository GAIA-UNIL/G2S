"""Covariance models for the Gaussian geostatistics algorithms of G2S.

This module builds the ``-cov`` text argument (and its ``-cvar`` bindings) of
the Gaussian algorithms (sgs, kriging, covEval, ...) from Python objects, and
parses such a text back into objects. It never re-implements the covariance
functions: evaluating a model (``model(h)``, ``model.variogram(h)``) runs the
``covEval`` algorithm on a G2S server, so there is a single implementation of
every model, in C++ (include/geostat/covarianceModel.hpp).

Example::

    from g2s import g2s, cov
    model = 0.1*cov.nugget() + 0.9*cov.spherical(range=[40, 20], angle=30)
    str(model)   # '0.1*nugget + 0.9*spherical(range=[40, 20], angle=30)'
    sim = g2s('-a', 'sgs', '-di', di, *model.args(), '-n', 50)

    # parameters may vary in space (a numpy array on the simulation grid) and
    # may be expressions of named variables
    nug = cov.var('nug', 0.1)
    L = cov.var('L', range_map)                      # array on the -di grid
    model = nug*cov.nugget() + (1 - nug)*cov.spherical(range=[L, L/2], angle=theta_map)
    model.args()
    # ['-cov', 'nug*nugget + (1 - nug)*spherical(range=[L, L/2], angle=_a1)',
    #  '-cvar', 'nug', '0.1', '-cvar', 'L', range_map, '-cvar', '_a1', theta_map]

Structures
    ``nugget()``, ``spherical``, ``exponential``, ``gaussian``, ``cubic``,
    ``matern`` (``nu=``), ``stable`` (``alpha=``) and ``map(table)``. They are
    combined with ``+`` and scaled with ``*`` (the factor is the partial sill);
    a structure without a factor has partial sill 1. Structures cannot be
    subtracted.

Range (``range=``) is the practical range, in grid cells
    spherical and cubic reach 0 at the range; exponential, gaussian, stable and
    matern have dropped to exp(-3) = 4.98% of their sill there.
    ``scale=`` gives the raw scale parameter instead (not both):
    exponential range = 3*scale, gaussian range = sqrt(3)*scale, stable range =
    3**(1/alpha)*scale, matern range = u95(nu)*scale (u95(0.5) = 3), spherical
    and cubic range = scale. The matern is the plain (h/scale)**nu K_nu(h/scale)
    form (not the sqrt(2 nu) variant).

Axes and angles
    Lags, ``range=[r0, r1, r2]`` and ``angle`` are given per axis in the
    server's axis order: component 0 is the fastest-varying axis, which is the
    LAST index of a numpy array (the columns of a 2D array), component 1 the
    one before it (the rows of a 2D array), component 2 the first index of a
    3D array. So for ``a.shape == (ny, nx)``, ``range=[rx, ry]`` is ``rx``
    along ``a[i, :]`` and ``ry`` along ``a[:, j]``.
    2D: ``angle=a`` (degrees) turns the direction of ``range[0]`` from axis 0
    towards axis 1; with axis 0 to the right and axis 1 upwards a > 0 is
    counter-clockwise, so on an image shown with row 0 at the top (imshow) it
    appears clockwise. 3D: ``angle=[a, b, c]`` (``angle=a`` means [a, 0, 0]):
    the direction of range[0] is (cos a cos b, sin a cos b, sin b), i.e. a is
    its azimuth in the (0, 1) plane, b its elevation towards axis 2, and c
    turns the two other principal axes about it (axis 1 towards axis 2).

Parameters
    A parameter (partial sill, range, scale, angle, nu, alpha) is
      * a number;
      * a numpy array: a spatially varying parameter, one value per cell of the
        simulation grid (same shape as ``-di``); it becomes a variable with an
        automatic name (``_a1``, ``_a2``, ...);
      * a named variable ``cov.var('L', value)`` (value: number or array),
        reusable in several places;
      * an expression of those: ``+ - * /``, unary minus, ``cov.min``,
        ``cov.max``, ``cov.sqrt``, ``cov.exp``, ``cov.log``, ``cov.abs``.
    ``range``, ``scale`` and ``angle`` take a list for one value per axis:
    ``range=[40, L]``. A numpy array is always ONE spatially varying value,
    never a list of per-axis values.

Precision
    Numbers are written with the shortest text that reads back to the same
    double, both in the model text and in the ``-cvar`` values (which are
    passed as strings: the compiled interface would print a float with only 6
    decimals). Negative constants are passed as one ``name=value`` token,
    because the interface reads a string starting with '-' as a flag.
"""

import builtins as _builtins
import math as _math
import re as _re

import numpy as _np

# cov.map, cov.min, cov.max and cov.abs shadow builtins: use them qualified,
# they are deliberately left out of "from g2s.cov import *".
__all__ = [
    "CovarianceError", "ServerError", "Model", "ModelSet", "Expr", "Var",
    "var", "parse", "regions",
    "nugget", "spherical", "exponential", "gaussian", "cubic", "matern", "stable",
    "sqrt", "exp", "log",
    "format_number", "is_reserved_word", "is_valid_variable_name",
]


class CovarianceError(ValueError):
    """Invalid covariance model, parameter or variable (the C++ messages are mirrored)."""


# ------------------------------------------------------------------ grammar constants

STRUCTURES = ("nugget", "spherical", "exponential", "gaussian", "cubic", "matern", "stable", "map")
_STRUCTURE_LIST = "nugget, spherical, exponential, gaussian, cubic, matern, stable or map"
_KEYWORDS = ("range", "scale", "angle", "nu", "alpha", "h")
_FUNCTIONS = ("min", "max", "sqrt", "exp", "log", "abs")
MAX_DIMS = 4
_IDENTIFIER = _re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def is_reserved_word(name):
    """True for structure names, parameter keywords, function names and 'h' (any case)."""
    lower = name.lower()
    return lower in STRUCTURES or lower in _KEYWORDS or lower in _FUNCTIONS


def is_valid_variable_name(name):
    """A valid variable name: [A-Za-z_][A-Za-z0-9_]* and not a reserved word."""
    return isinstance(name, str) and bool(_IDENTIFIER.match(name)) and not is_reserved_word(name)


def _check_variable_name(name):
    if not isinstance(name, str):
        raise CovarianceError("a variable name must be a string, got %r" % (name,))
    if is_valid_variable_name(name):
        return
    if is_reserved_word(name):
        raise CovarianceError("'%s' is a reserved word (structure, keyword or function name) and cannot be a variable" % name)
    raise CovarianceError("'%s' is not a valid variable name (letters, digits and '_', not starting with a digit)" % name)


def format_number(value):
    """Shortest text of a double, identical to the C++ formatNumber.

    "0" for zero; integers with |v| < 1e15 without decimals; otherwise the
    shortest '%.{p}g' (p = 1..17) that reads back to the same double.
    """
    value = float(value)
    if value == 0.0:
        return "0"
    if _math.isnan(value):
        return "nan"
    if _math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if _builtins.abs(value) < 1e15 and value == _math.floor(value):
        return "%.0f" % value
    for precision in range(1, 18):
        text = "%.*g" % (precision, value)
        if float(text) == value:
            return text
    return text


# ------------------------------------------------------------------ expressions

_UNARY = ("neg", "sqrt", "exp", "log", "abs")
_BINARY = ("add", "sub", "mul", "div", "min", "max")


def _apply(op, a, b=0.0):
    """IEEE arithmetic like the C++ (no Python exceptions on 1/0, log(0), ...)."""
    if op == "neg":
        return -a
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "div":
        if b == 0.0:
            if a == 0.0 or _math.isnan(a):
                return _math.nan
            return _math.copysign(_math.inf, a) * _math.copysign(1.0, b)
        return a / b
    if op == "min":
        return b if b < a else a      # std::min(a, b)
    if op == "max":
        return b if a < b else a      # std::max(a, b)
    if op == "sqrt":
        return _math.sqrt(a) if a >= 0 else _math.nan
    if op == "exp":
        try:
            return _math.exp(a)
        except OverflowError:
            return _math.inf
    if op == "log":
        if a == 0.0:
            return -_math.inf
        return _math.log(a) if a > 0 else _math.nan
    if op == "abs":
        return _builtins.abs(a)
    raise AssertionError(op)


def _is_number(x):
    return isinstance(x, (int, float, _np.integer, _np.floating)) and not isinstance(x, (bool, _np.bool_))


class Expr:
    """A parameter expression: a constant, a variable or an operation on them.

    Built with the arithmetic operators and ``cov.min``/``max``/``sqrt``/
    ``exp``/``log``/``abs``. Constant sub-expressions are folded when they are
    built, exactly like the C++ parser does, so ``(1 - 0.1)`` is the constant
    0.9. ``str(expr)`` is the canonical text (automatic variable names show as
    '?' until the expression is part of a model).
    """

    __slots__ = ("op", "value", "left", "right")
    __array_ufunc__ = None  # numpy must defer to our reflected operators

    def __init__(self, op, value=None, left=None, right=None):
        self.op = op
        self.value = value
        self.left = left
        self.right = right

    # -- construction
    @staticmethod
    def _make(op, left, right=None):
        if op in _UNARY:
            if left.op == "const":
                return Expr._fold(op, left.value, None)
        elif left.op == "const" and right.op == "const":
            return Expr._fold(op, left.value, right.value)
        return Expr(op, left=left, right=right)

    @staticmethod
    def _fold(op, a, b):
        value = _apply(op, a, b if b is not None else 0.0)
        if not _math.isfinite(value):
            raise CovarianceError("constant sub-expression %s is not finite" % _operation_text(op, a, b))
        return Expr("const", value=float(value))

    def is_constant(self):
        return self.op == "const"

    # -- operators
    def __add__(self, other):
        o = _as_operand(other)
        return NotImplemented if o is None else Expr._make("add", self, o)

    def __radd__(self, other):
        o = _as_operand(other)
        return NotImplemented if o is None else Expr._make("add", o, self)

    def __sub__(self, other):
        o = _as_operand(other)
        return NotImplemented if o is None else Expr._make("sub", self, o)

    def __rsub__(self, other):
        o = _as_operand(other)
        return NotImplemented if o is None else Expr._make("sub", o, self)

    def __mul__(self, other):
        o = _as_operand(other)
        return NotImplemented if o is None else Expr._make("mul", self, o)

    def __rmul__(self, other):
        o = _as_operand(other)
        return NotImplemented if o is None else Expr._make("mul", o, self)

    def __truediv__(self, other):
        o = _as_operand(other)
        return NotImplemented if o is None else Expr._make("div", self, o)

    def __rtruediv__(self, other):
        o = _as_operand(other)
        return NotImplemented if o is None else Expr._make("div", o, self)

    def __neg__(self):
        return Expr._make("neg", self)

    def __pos__(self):
        return self

    def __bool__(self):
        raise TypeError("a covariance parameter expression has no truth value "
                        "(use cov.min/cov.max instead of comparisons, min() or max())")

    # -- inspection
    def variables(self):
        """The Var leaves of this expression, in order of first appearance."""
        out = []
        _collect_vars(self, out)
        return out

    def evaluate(self, values=None):
        """Value of the expression; ``values`` maps Var objects or names to numbers.

        Returns None when a variable has no scalar value (e.g. it is an array).
        """
        return _evaluate(self, values or {})

    def to_string(self, names=None):
        return _print(self, names or {})

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return "cov.Expr(%r)" % self.to_string()


class Var(Expr):
    """A named model variable, bound with ``-cvar name value``.

    Create it with ``cov.var(name, value)``. The value is a number (a
    constant) or a numpy array on the simulation grid (a spatially varying
    parameter, or the covariance map of ``cov.map``); an array follows the
    axis convention of the module docstring. The same Var may be used in
    several places and in several models. ``name=None`` gives an automatic
    name (``_a1``, ...) when the model is converted to text.
    """

    __slots__ = ("name",)

    def __init__(self, name, value=None):
        if name is not None:
            _check_variable_name(name)
        Expr.__init__(self, "var", value=_check_value(name, value))
        self.name = name

    def __repr__(self):
        if self.value is None:
            return "cov.var(%r)" % (self.name,)
        if isinstance(self.value, _np.ndarray):
            return "cov.var(%r, <array %s>)" % (self.name, "x".join(str(d) for d in self.value.shape))
        return "cov.var(%r, %r)" % (self.name, self.value)


def _check_value(name, value):
    label = "variable '%s'" % name if name else "parameter"
    if value is None or isinstance(value, str):
        return value
    if _is_number(value):
        value = float(value)
        if not _math.isfinite(value):
            raise CovarianceError("%s: the value is not finite" % label)
        return value
    if isinstance(value, (Expr, Model)):
        raise CovarianceError("%s: the value must be a number or a numpy array, not %s" % (label, type(value).__name__))
    array = _np.asarray(value)
    if array.dtype == object or array.dtype.kind not in "biuf":
        raise CovarianceError("%s: the value must be a number or a numeric array" % label)
    if array.ndim == 0:
        return _check_value(name, array.item())
    if array.size == 0:
        raise CovarianceError("%s: the array is empty" % label)
    return array


def var(name, value=None):
    """A named variable (see :class:`Var`): ``L = cov.var('L', 40)``, ``cov.var('theta', angle_map)``."""
    return Var(name, value)


def _as_operand(x):
    """Converts a number or array to an Expr; None when x is not a parameter."""
    if isinstance(x, Expr):
        return x
    if _is_number(x):
        v = float(x)
        if not _math.isfinite(v):
            raise CovarianceError("a parameter is not finite (%r)" % (x,))
        return Expr("const", value=v)
    if isinstance(x, _np.ndarray):
        if x.ndim == 0:
            return _as_operand(x.item())
        return Var(None, x)
    return None


def _as_expr(x, what):
    e = _as_operand(x)
    if e is None:
        if isinstance(x, (list, tuple)):
            raise CovarianceError("%s takes a single value, got a list" % what)
        if isinstance(x, str):
            raise CovarianceError("%s: got the string %r; use a number, a numpy array, cov.var(name, value) "
                                  "or cov.parse(text)" % (what, x))
        raise CovarianceError("%s must be a number, a numpy array, a cov.var or an expression, not %s"
                              % (what, type(x).__name__))
    return e


def _as_vector(x, what):
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise CovarianceError("empty vector for %s" % what)
        return tuple(_as_expr(e, what) for e in x)
    return (_as_expr(x, what),)


def _operation_text(op, a, b):
    fa = format_number(a)
    fb = format_number(b) if b is not None else ""
    return {"neg": "-" + fa, "add": fa + " + " + fb, "sub": fa + " - " + fb, "mul": fa + "*" + fb,
            "div": fa + "/" + fb, "min": "min(%s, %s)" % (fa, fb), "max": "max(%s, %s)" % (fa, fb),
            "sqrt": "sqrt(%s)" % fa, "exp": "exp(%s)" % fa, "log": "log(%s)" % fa, "abs": "abs(%s)" % fa}[op]


def _precedence(e):
    if e.op in ("add", "sub"):
        return 1
    if e.op in ("mul", "div"):
        return 2
    if e.op == "neg":
        return 3
    if e.op == "const":
        return 3 if e.value < 0 else 4
    return 4


def _var_name(v, names):
    n = names.get(id(v))
    if n is not None:
        return n
    return v.name if v.name is not None else "?"


def _print(e, names):
    op = e.op
    if op == "const":
        return format_number(e.value)
    if op == "var":
        return _var_name(e, names)
    if op == "neg":
        inner = _print(e.left, names)
        return "-" + ("(" + inner + ")" if _precedence(e.left) < 3 else inner)
    if op in ("add", "sub", "mul", "div"):
        own = _precedence(e)
        left = _print(e.left, names)
        right = _print(e.right, names)
        if _precedence(e.left) < own:
            left = "(" + left + ")"
        if _precedence(e.right) <= own:
            right = "(" + right + ")"
        return left + {"add": " + ", "sub": " - ", "mul": "*", "div": "/"}[op] + right
    if op in ("min", "max"):
        return "%s(%s, %s)" % (op, _print(e.left, names), _print(e.right, names))
    return "%s(%s)" % (op, _print(e.left, names))


def _collect_vars(e, out):
    if e is None:
        return
    if e.op == "var":
        if not _builtins.any(v is e for v in out):
            out.append(e)
        return
    _collect_vars(e.left, out)
    _collect_vars(e.right, out)


def _evaluate(e, values):
    if e.op == "const":
        return e.value
    if e.op == "var":
        v = values.get(id(e), values.get(e.name) if e.name is not None else None) if values else None
        if v is None:
            v = e.value
        return float(v) if _is_number(v) else None
    a = _evaluate(e.left, values)
    if a is None:
        return None
    b = 0.0
    if e.right is not None:
        b = _evaluate(e.right, values)
        if b is None:
            return None
    return _apply(e.op, a, b)


def _function(op, *args):
    exprs = [_as_expr(a, op + "()") for a in args]
    return Expr._make(op, *exprs)


def min(a, b):
    """min(a, b) of parameter expressions."""
    return _function("min", a, b)


def max(a, b):
    """max(a, b) of parameter expressions."""
    return _function("max", a, b)


def sqrt(x):
    """Square root of a parameter expression."""
    return _function("sqrt", x)


def exp(x):
    """Exponential of a parameter expression."""
    return _function("exp", x)


def log(x):
    """Natural logarithm of a parameter expression."""
    return _function("log", x)


def abs(x):
    """Absolute value of a parameter expression."""
    return _function("abs", x)


# ------------------------------------------------------------------ structures and models

class _Structure:
    __slots__ = ("type", "range", "use_scale", "angle", "shape", "table")

    def __init__(self, type, range=(), use_scale=False, angle=(), shape=None, table=None):
        self.type = type
        self.range = tuple(range)
        self.use_scale = use_scale
        self.angle = tuple(angle)
        self.shape = shape
        self.table = table

    def expressions(self):
        out = list(self.range) + list(self.angle)
        if self.shape is not None:
            out.append(self.shape)
        return out


def _check_values(type, sill, range, angle, shape):
    """Mirror of ModelSpec::checkValues (C++); returns '' when valid."""
    if sill is not None:
        if not _math.isfinite(sill):
            return "the partial sill is not finite (%s)" % format_number(sill)
        if sill < 0:
            return "the partial sill must be >= 0, got " + format_number(sill)
    for r in range:
        if r is not None and (not _math.isfinite(r) or not r > 0):
            return "range/scale must be > 0, got " + format_number(r)
    for a in angle:
        if a is not None and not _math.isfinite(a):
            return "angle is not finite (%s)" % format_number(a)
    if shape is not None:
        if type == "matern" and not (0 < shape <= 100):
            return "nu must be in (0, 100], got " + format_number(shape)
        if type == "stable" and not (0 < shape <= 2):
            return "alpha must be in (0, 2], got " + format_number(shape)
    return ""


class Model:
    """A covariance model: a sum of structures, each with its partial sill.

    Build it with the structure functions (``cov.spherical(...)``), ``+`` and
    ``*``, or with ``cov.parse(text)``.

    * ``str(model)`` is the canonical text, identical to what the C++ prints.
    * ``model.args()`` is the list to splice into ``g2s(...)``.
    * ``model(h)`` / ``model.variogram(h)`` evaluate it on a G2S server.
    """

    __slots__ = ("_terms",)
    __array_ufunc__ = None  # numpy must defer to our reflected operators

    def __init__(self, terms):
        self._terms = tuple(terms)  # ((sill Expr, _Structure), ...)

    # -- algebra
    def __add__(self, other):
        if isinstance(other, Model):
            return Model(self._terms + other._terms)
        if _is_number(other) and other == 0:
            return self  # sum([...])
        if isinstance(other, (Expr, _np.ndarray)) or _is_number(other):
            raise CovarianceError("a number or parameter cannot be added to a covariance model; "
                                  "a constant term is a nugget: c*cov.nugget()")
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        raise CovarianceError("a structure cannot be subtracted; parenthesise a difference in a partial sill: "
                              "(a - b)*cov.spherical(...)")

    __rsub__ = __sub__

    def _scaled(self, factor, divide=False):
        if isinstance(factor, Model):
            raise CovarianceError("two covariance models cannot be multiplied (only a model and a partial sill)")
        f = _as_expr(factor, "the partial sill")
        terms = []
        for sill, structure in self._terms:
            if divide:
                new = Expr._make("div", sill, f)
            elif sill.op == "const" and sill.value == 1.0:
                new = f
            elif f.op == "const" and f.value == 1.0:
                new = sill
            else:
                new = Expr._make("mul", f, sill)
            _check_sill(new, structure.type)
            terms.append((new, structure))
        return Model(terms)

    def __mul__(self, other):
        return self._scaled(other)

    def __rmul__(self, other):
        return self._scaled(other)

    def __truediv__(self, other):
        return self._scaled(other, divide=True)

    def __neg__(self):
        raise CovarianceError("the partial sill must be >= 0; a structure cannot be negated")

    def __bool__(self):
        return True

    # -- text
    def __str__(self):
        return _Compiled([self]).texts[0]

    def __repr__(self):
        return "cov.parse(%r)" % str(self)

    def __len__(self):
        return len(self._terms)

    def __eq__(self, other):
        return isinstance(other, Model) and str(self) == str(other) and _same_bindings(self, other)

    __hash__ = None

    @property
    def structures(self):
        """The structure names, in order (e.g. ['nugget', 'spherical'])."""
        return [s.type for _, s in self._terms]

    @property
    def variables(self):
        """{name: value} of every variable of the model, in order of first appearance."""
        return dict(_Compiled([self]).bindings)

    def args(self):
        """Arguments for ``g2s(...)``: ['-cov', text, '-cvar', name, value, ...].

        Constants are passed as round-trip strings (negative ones as a single
        'name=value' token), arrays as numpy arrays (uploaded by the
        interface). Raises CovarianceError when a variable has no value or a
        constant parameter is out of range.
        """
        return _Compiled([self]).args()

    # -- evaluation through the server
    def __call__(self, h, pos=None, **g2s_options):
        """Covariance C(h), computed by the ``covEval`` algorithm on a running G2S server.

        This needs a G2S server (``g2s server``, or ``./g2s_server`` in
        build/c++-build) reachable with the usual ``sa=``/``p=`` options, which
        are forwarded to ``g2s(...)`` with any other keyword (e.g. ``TO=5000``, a timeout in ms);
        ``verbose=True`` shows the job progress. Raises ServerError when no
        server answers or covEval reports an error.

        h: lags in grid cells. A scalar or a 1-D array is a list of 1D lags; an
           N x ndim array is one lag per row, column 0 being the shift along the
           LAST numpy axis (the columns), column 1 along the axis before it.
        pos: where a spatially varying model is resolved, in the same component
           order: one node (length ndim) for every lag, or one per lag (N x ndim).
           Integer cell coordinates; column 0 indexes the last numpy axis.
        Returns a 1-D array of N values.
        """
        return _cov_eval([self], None, h, pos, False, g2s_options)

    def variogram(self, h, pos=None, **g2s_options):
        """Variogram gamma(h) = C(0) - C(h), computed by ``covEval`` on a G2S server.

        Same arguments as ``model(h)``; needs a running G2S server.
        """
        return _cov_eval([self], None, h, pos, True, g2s_options)


def _same_bindings(a, b):
    ba, bb = _Compiled([a]).bindings, _Compiled([b]).bindings
    if [n for n, _ in ba] != [n for n, _ in bb]:
        return False
    for (_, va), (_, vb) in zip(ba, bb):
        if not _values_equal(va, vb):
            return False
    return True


def _check_sill(sill, type):
    if sill.op == "const":
        problem = _check_values(type, sill.value, (), (), None)
        if problem:
            raise CovarianceError("%s: %s" % (type, problem))


def _structure(type, range=None, scale=None, angle=None, shape=None):
    if range is not None and scale is not None:
        raise CovarianceError("range/scale given twice in %s" % type)
    if range is None and scale is None:
        raise CovarianceError("%s needs range= or scale=" % type)
    use_scale = scale is not None
    keyword = "scale" if use_scale else "range"
    r = _as_vector(scale if use_scale else range, keyword + " of " + type)
    if len(r) > MAX_DIMS:
        raise CovarianceError("%s has more than %d values" % (keyword, MAX_DIMS))
    a = ()
    if angle is not None:
        a = _as_vector(angle, "angle of " + type)
        if len(a) not in (1, 3):
            raise CovarianceError("angle takes one value (2D) or three values (3D)")
    s = None
    if type in ("matern", "stable"):
        name = "nu" if type == "matern" else "alpha"
        if shape is None:
            raise CovarianceError("%s needs %s=" % (type, name))
        s = _as_expr(shape, name)
    structure = _Structure(type, r, use_scale, a, s)
    problem = _check_values(type, None,
                            [e.value for e in r if e.op == "const"],
                            [e.value for e in a if e.op == "const"],
                            s.value if s is not None and s.op == "const" else None)
    if problem:
        raise CovarianceError("%s: %s" % (type, problem))
    return Model([(Expr("const", value=1.0), structure)])


def nugget():
    """Nugget: the partial sill at lag exactly 0, 0 elsewhere."""
    return Model([(Expr("const", value=1.0), _Structure("nugget"))])


def spherical(range=None, angle=None, scale=None):
    """Spherical: rho = 1 - 1.5 t + 0.5 t^3 for t < 1, 0 beyond; t = lag/range; scale = range.

    range: practical range (cells), a value or [r0, r1, (r2)] per axis (axis 0
    = last numpy index); angle: degrees (2D), or [a, b, c] (3D). See the module
    docstring for the axis and angle conventions.
    """
    return _structure("spherical", range, scale, angle)


def exponential(range=None, angle=None, scale=None):
    """Exponential: rho = exp(-3 t) = exp(-h/scale); practical range = 3*scale.

    See :func:`spherical` for range and angle.
    """
    return _structure("exponential", range, scale, angle)


def gaussian(range=None, angle=None, scale=None):
    """Gaussian: rho = exp(-3 t^2) = exp(-(h/scale)^2); practical range = sqrt(3)*scale.

    See :func:`spherical` for range and angle.
    """
    return _structure("gaussian", range, scale, angle)


def cubic(range=None, angle=None, scale=None):
    """Cubic (GSLIB): rho = 1 - 7t^2 + 35/4 t^3 - 7/2 t^5 + 3/4 t^7 for t < 1, 0 beyond; scale = range.

    See :func:`spherical` for range and angle.
    """
    return _structure("cubic", range, scale, angle)


def matern(range=None, nu=None, angle=None, scale=None):
    """Matern with smoothness nu (0 < nu <= 100): rho = 2^(1-nu)/Gamma(nu) (h/scale)^nu K_nu(h/scale).

    Practical range = u95(nu)*scale (u95(0.5) = 3, u95(1.5) = 4.7490,
    u95(2.5) = 5.9245); nu = 0.5 is the exponential with the same range.
    See :func:`spherical` for range and angle.
    """
    return _structure("matern", range, scale, angle, nu)


def stable(range=None, alpha=None, angle=None, scale=None):
    """Stable with exponent alpha (0 < alpha <= 2): rho = exp(-3 t^alpha) = exp(-(h/scale)^alpha).

    Practical range = 3^(1/alpha)*scale; alpha = 1 (2) is the exponential
    (gaussian) with the same range. See :func:`spherical` for range and angle.
    """
    return _structure("stable", range, scale, angle, alpha)


def map(table):
    """Covariance given as a map: C(h) = sill * M[centre + round(h)], 0 outside M.

    table: a numpy array M (same size as the simulation grid; its centre, lag
    0, is at shape//2 on each axis and must be > 0), a ``cov.var(name, M)``,
    or a name (string) to bind later. The partial sill is the factor times
    M[centre].
    """
    if isinstance(table, str):
        table = Var(table)
    elif isinstance(table, Var):
        pass
    elif isinstance(table, _np.ndarray) or (isinstance(table, (list, tuple)) and len(table) > 0):
        table = Var(None, _np.asarray(table))
    else:
        raise CovarianceError("map takes one table name: map(name), a cov.var or a numpy array")
    if isinstance(table.value, float):
        raise CovarianceError("map(%s) needs a covariance map image, not a number" % (table.name or "?"))
    return Model([(Expr("const", value=1.0), _Structure("map", table=table))])


# ------------------------------------------------------------------ compilation (names, text, args)

def _values_equal(a, b):
    if a is b:
        return True
    if isinstance(a, _np.ndarray) or isinstance(b, _np.ndarray):
        return isinstance(a, _np.ndarray) and isinstance(b, _np.ndarray) and a.shape == b.shape and _np.array_equal(a, b)
    return a == b


class _Compiled:
    """Names every variable of one or several models and prints their texts."""

    def __init__(self, models):
        self.names = {}        # id(Var) -> name
        self.bindings = []     # [(name, value)] in order of first appearance
        self._keep = []        # keeps the Var objects alive while their ids are used
        by_name = {}           # name -> (Var, is_map)
        auto = []              # [(Var, is_map)] without a name, in order
        order = []             # ('named', name) or ('auto', index)
        for model in models:
            for sill, structure in model._terms:
                exprs = [(sill, False)] + [(e, False) for e in structure.expressions()]
                if structure.table is not None:
                    exprs.append((structure.table, True))
                for e, is_map in exprs:
                    for v in ([e] if is_map else e.variables()):
                        self._keep.append(v)
                        if v.name is None:
                            for i, (other, other_map) in enumerate(auto):
                                if other is v or (other.value is v.value and v.value is not None):
                                    if other_map != is_map:
                                        raise CovarianceError("the same array is used both as a map table and as a parameter")
                                    self.names[id(v)] = i
                                    break
                            else:
                                self.names[id(v)] = len(auto)
                                order.append(("auto", len(auto)))
                                auto.append((v, is_map))
                            continue
                        known = by_name.get(v.name)
                        if known is None:
                            by_name[v.name] = (v, is_map)
                            order.append(("named", v.name))
                        else:
                            other, other_map = known
                            if other_map != is_map:
                                raise CovarianceError("'%s' is used both as a map table and as a parameter variable" % v.name)
                            if other is not v and not _values_equal(other.value, v.value):
                                raise CovarianceError("variable '%s' is defined twice with different values" % v.name)
                        self.names[id(v)] = v.name
        # automatic names avoid every user name (case-sensitive, like the C++)
        auto_names = []
        k = 1
        for _ in auto:
            while "_a%d" % k in by_name:
                k += 1
            auto_names.append("_a%d" % k)
            k += 1
        for key, value in list(self.names.items()):
            if isinstance(value, int):
                self.names[key] = auto_names[value]
        for kind, ref in order:
            if kind == "named":
                v = by_name[ref][0]
                self.bindings.append((ref, v.value))
            else:
                self.bindings.append((auto_names[ref], auto[ref][0].value))
        self.models = models
        self.texts = [self._text(m) for m in models]

    def _text(self, model):
        parts = []
        for sill, structure in model._terms:
            s = _print(sill, self.names)
            if sill.op in ("add", "sub"):
                s = "(" + s + ")"
            t = structure.type
            if t == "nugget":
                parts.append(s + "*nugget")
                continue
            if t == "map":
                parts.append("%s*map(%s)" % (s, self.names[id(structure.table)]))
                continue
            text = "%s*%s(%s=%s" % (s, t, "scale" if structure.use_scale else "range", self._vector(structure.range))
            if structure.angle:
                text += ", angle=" + self._vector(structure.angle)
            if t == "matern":
                text += ", nu=" + _print(structure.shape, self.names)
            if t == "stable":
                text += ", alpha=" + _print(structure.shape, self.names)
            parts.append(text + ")")
        return " + ".join(parts)

    def _vector(self, values):
        if len(values) == 1:
            return _print(values[0], self.names)
        return "[" + ", ".join(_print(v, self.names) for v in values) + "]"

    def validate(self):
        """Client-side checks of what can be checked cheaply (unbound names, scalar values)."""
        scalars = {}
        for name, value in self.bindings:
            if value is None:
                raise CovarianceError("variable '%s' has no value; use cov.var('%s', value) or cov.parse(text, %s=value)"
                                      % (name, name, name))
            if isinstance(value, float):
                scalars[name] = value
        lookup = {key: scalars[name] for key, name in self.names.items() if name in scalars}
        for model in self.models:
            for sill, structure in model._terms:
                if structure.type == "map":
                    if isinstance(structure.table.value, float):
                        raise CovarianceError("map(%s) needs a covariance map image, not a number" % self.names[id(structure.table)])
                    continue
                problem = _check_values(structure.type, _evaluate(sill, lookup),
                                        [_evaluate(e, lookup) for e in structure.range],
                                        [_evaluate(e, lookup) for e in structure.angle],
                                        _evaluate(structure.shape, lookup) if structure.shape is not None else None)
                if problem:
                    raise CovarianceError("%s: %s" % (structure.type, problem))

    def cvar_args(self):
        out = []
        for name, value in self.bindings:
            if isinstance(value, float):
                text = format_number(value)
                if text.startswith("-"):
                    out += ["-cvar", name + "=" + text]   # '-30' alone would be read as a flag
                else:
                    out += ["-cvar", name, text]
            else:
                out += ["-cvar", name, value]
        return out

    def cov_texts(self):
        # a text starting with '-' would be read as a flag by the interface; a
        # leading space is ignored by the C++ tokenizer
        return [(" " + t) if t.startswith("-") else t for t in self.texts]

    def args(self):
        self.validate()
        return ["-cov"] + self.cov_texts() + self.cvar_args()


# ------------------------------------------------------------------ several models (regions)

class ModelSet:
    """Several models, one per discrete region, selected per cell by an index map.

    ``index`` is an integer array on the simulation grid (same shape as -di):
    0 selects ``models[0]``, 1 ``models[1]``, ... (``-ii``). Variables may be
    shared between models (same cov.var, or same name and value).
    """

    __slots__ = ("models", "index")

    def __init__(self, models, index):
        models = list(models)
        if not models:
            raise CovarianceError("no covariance model given")
        for m in models:
            if not isinstance(m, Model):
                raise CovarianceError("regions() takes cov models, got %s" % type(m).__name__)
        index = _np.asarray(index)
        if index.dtype.kind not in "biuf" or index.size == 0:
            raise CovarianceError("the model index must be a non-empty numeric array")
        if not _np.all(_np.isfinite(index)) or _np.any(index != _np.floor(index)) or index.min() < 0:
            raise CovarianceError("the model index must hold model indices (0, 1, ...)")
        if index.max() >= len(models):
            raise CovarianceError("model index %d but only %d model(s) given (the index is 0-based)"
                                  % (int(index.max()), len(models)))
        self.models = models
        self.index = index

    def texts(self):
        return _Compiled(self.models).texts

    def __str__(self):
        return "\n".join(self.texts())

    def args(self):
        """['-cov', text0, text1, ..., '-cvar', ..., '-ii', index]."""
        return _Compiled(self.models).args() + ["-ii", self.index.astype(_np.float32)]

    def __call__(self, h, pos, **g2s_options):
        """C(h) of the model selected at ``pos`` (see Model.__call__); needs a running G2S server."""
        return _cov_eval(self.models, self.index, h, pos, False, g2s_options)

    def variogram(self, h, pos, **g2s_options):
        """gamma(h) of the model selected at ``pos``; needs a running G2S server."""
        return _cov_eval(self.models, self.index, h, pos, True, g2s_options)


def regions(models, index):
    """Several models for discrete regions: ``g2s(..., *cov.regions([m0, m1], index).args())``."""
    return ModelSet(models, index)


# ------------------------------------------------------------------ evaluation through covEval

class ServerError(RuntimeError):
    """covEval could not run (no reachable G2S server, or the server reported an error)."""


def _cov_eval(models, index, h, pos, variogram, g2s_options):
    compiled = _Compiled(models)
    args = compiled.args()
    if index is not None:
        args += ["-ii", index.astype(_np.float32)]
    lags = _np.asarray(h, dtype=float)
    if lags.ndim == 0:
        lags = lags.reshape(1)
    if lags.ndim > 2:
        raise CovarianceError("h must be a scalar, a vector of 1D lags or an N x ndim array")
    ndim = 1 if lags.ndim == 1 else lags.shape[1]
    count = lags.shape[0]
    call = ["-a", "covEval"] + args + ["-lag", _np.ascontiguousarray(lags, dtype=_np.float32)]
    if pos is not None:
        p = _np.asarray(pos, dtype=float)
        if ndim == 1:
            p = p.reshape(-1, 1)
        elif p.ndim == 1:
            p = p.reshape(1, -1)
        if p.ndim != 2 or p.shape[1] != ndim or p.shape[0] not in (1, count):
            raise CovarianceError("pos must have %d component(s): one node for every lag, or one per lag" % ndim)
        call += ["-pos", _np.ascontiguousarray(p, dtype=_np.float32)]
    if variogram:
        call.append("-vario")
    try:
        from .g2s import run
    except ImportError as error:
        raise ServerError("model(h) evaluates the covariance with the covEval algorithm of a G2S server, "
                          "but the compiled g2s interface is not available: %s" % error) from error
    options = dict(g2s_options)
    if not options.pop("verbose", False):
        call.append("-silent")
    try:
        result = run(*call, **options)
    except Exception as error:
        message = str(error)
        if "off-line" in message or "timeout" in message.lower():
            raise ServerError("model(h) evaluates the covariance with the covEval algorithm of a running G2S server, "
                              "and none answered (start one with 'g2s server', or pass sa='host', p=port): %s"
                              % message) from error
        raise ServerError("covEval (on the G2S server) failed: %s" % message) from error
    key, position = ("variogram", 1) if variogram else ("covariance", 0)
    if isinstance(result, dict):
        if key not in result:
            raise ServerError("covEval returned no %s: %r" % (key, result.get("error", result)))
        value = result[key]
    elif isinstance(result, (tuple, list)):
        value = result[position]
    else:
        value = result
    return _np.asarray(value, dtype=float).reshape(count)


# ------------------------------------------------------------------ parser (port of ModelSpec::parse)

class _Token:
    __slots__ = ("kind", "text", "value", "position")

    def __init__(self, kind, text="", value=0.0, position=0):
        self.kind = kind          # 'number', 'name', 'symbol', 'end'
        self.text = text
        self.value = value
        self.position = position  # 0-based character offset


def _isdigit(c):
    return "0" <= c <= "9"


def _isalpha(c):
    return ("a" <= c <= "z") or ("A" <= c <= "Z")


def _tokenize(text):
    tokens = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in " \t\n\v\f\r":
            i += 1
            continue
        if _isdigit(c) or (c == "." and i + 1 < n and _isdigit(text[i + 1])):
            j = i
            while j < n and _isdigit(text[j]):
                j += 1
            if j < n and text[j] == ".":
                j += 1
                while j < n and _isdigit(text[j]):
                    j += 1
            if j < n and text[j] in "eE":
                k = j + 1
                if k < n and text[k] in "+-":
                    k += 1
                if k >= n or not _isdigit(text[k]):
                    raise CovarianceError("invalid number '%s' at column %d" % (text[i:k], i + 1))
                while k < n and _isdigit(text[k]):
                    k += 1
                j = k
            if j < n and (_isalpha(text[j]) or text[j] in "_."):
                raise CovarianceError("invalid number '%s' at column %d" % (text[i:j + 1], i + 1))
            literal = text[i:j]
            value = float(literal)
            if not _math.isfinite(value):
                raise CovarianceError("number '%s' is out of range" % literal)
            tokens.append(_Token("number", literal, value, i))
            i = j
        elif _isalpha(c) or c == "_":
            j = i
            while j < n and (_isalpha(text[j]) or _isdigit(text[j]) or text[j] == "_"):
                j += 1
            tokens.append(_Token("name", text[i:j], 0.0, i))
            i = j
        elif c in "+-*/()[],=":
            tokens.append(_Token("symbol", c, 0.0, i))
            i += 1
        else:
            raise CovarianceError("unexpected character '%s' at column %d" % (c, i + 1))
    tokens.append(_Token("end", "", 0.0, n))
    return tokens


class _ExpressionParser:
    """Port of ExpressionParser: tokens[position, end), shared name table."""

    def __init__(self, tokens, position, end, variables):
        self.tokens = tokens
        self.position = position
        self.end = end
        self.variables = variables  # name -> Var

    def peek(self):
        if self.position < self.end:
            return self.tokens[self.position]
        return self.tokens[self.end] if self.end < len(self.tokens) else _Token("end")

    def is_symbol(self, symbol):
        return self.position < self.end and self.peek().kind == "symbol" and self.peek().text == symbol

    def fail(self, message):
        raise CovarianceError("%s (column %d)" % (message, self.peek().position + 1))

    def describe(self):
        if self.position >= self.end or self.peek().kind == "end":
            return "end of the expression"
        return "'%s'" % self.peek().text

    def expect(self, symbol):
        if not self.is_symbol(symbol):
            self.fail("expected '%s' but found %s" % (symbol, self.describe()))
        self.position += 1

    def parse(self):
        return self.parse_sum()

    def parse_sum(self):
        left = self.parse_product()
        while self.is_symbol("+") or self.is_symbol("-"):
            op = "add" if self.peek().text == "+" else "sub"
            self.position += 1
            left = Expr._make(op, left, self.parse_product())
        return left

    def parse_product(self):
        left = self.parse_unary()
        while self.is_symbol("*") or self.is_symbol("/"):
            op = "mul" if self.peek().text == "*" else "div"
            self.position += 1
            left = Expr._make(op, left, self.parse_unary())
        return left

    def parse_unary(self):
        if self.is_symbol("-"):
            self.position += 1
            return Expr._make("neg", self.parse_unary())
        return self.parse_primary()

    def parse_primary(self):
        if self.position >= self.end:
            self.fail("expected a value but found " + self.describe())
        token = self.peek()
        if token.kind == "number":
            self.position += 1
            return Expr("const", value=token.value)
        if self.is_symbol("("):
            self.position += 1
            inner = self.parse_sum()
            self.expect(")")
            return inner
        if token.kind == "name":
            lower = token.text.lower()
            if lower in _FUNCTIONS:
                self.position += 1
                self.expect("(")
                first = self.parse_sum()
                if lower in ("min", "max"):
                    self.expect(",")
                    second = self.parse_sum()
                    result = Expr._make(lower, first, second)
                else:
                    result = Expr._make(lower, first)
                self.expect(")")
                return result
            if lower == "h":
                self.fail("the lag 'h' cannot be used in a parameter: parameters are constant or vary with the node, never with the lag")
            if is_reserved_word(token.text):
                self.fail("'%s' is a reserved word and cannot be used as a variable" % token.text)
            self.position += 1
            v = self.variables.get(token.text)
            if v is None:
                v = self.variables[token.text] = Var(token.text)
            return v
        self.fail("expected a value but found " + self.describe())


def parse(text, **bindings):
    """Parses a model text (the ``-cov`` grammar) into a Model.

    ``bindings`` give the variable values: ``cov.parse('nug*nugget + (1 - nug)*
    exponential(range=L)', nug=0.1, L=range_map)``. Variables without a value
    can be bound later by building the model again; ``model.args()`` refuses
    unbound variables. Errors mirror the C++ messages (with the column).
    """
    if not isinstance(text, str):
        raise CovarianceError("a covariance model text must be a string")
    context = 'covariance model "%s": ' % text
    try:
        model, variables = _parse(text, context)
    except CovarianceError as error:
        message = str(error)
        raise CovarianceError(message if message.startswith(context) else context + message) from None
    for name, value in bindings.items():
        v = variables.get(name)
        if v is None:
            raise CovarianceError("variable '%s' is bound but the model does not use it" % name)
        v.value = _check_value(name, value)
        if v in _map_tables(model) and isinstance(v.value, float):
            raise CovarianceError("map(%s) needs a covariance map image, not a number" % name)
    return model


def _map_tables(model):
    return [s.table for _, s in model._terms if s.table is not None]


def _parse(text, context):
    tokens = _tokenize(text)
    end = len(tokens) - 1
    if end == 0:
        raise CovarianceError("empty model")

    def fail(index, message):
        return CovarianceError("%s (column %d)" % (message, tokens[index].position + 1))

    def is_symbol(i, symbol):
        return i < end and tokens[i].kind == "symbol" and tokens[i].text == symbol

    match = [0] * len(tokens)
    stack = []
    for i in range(end):
        if is_symbol(i, "(") or is_symbol(i, "["):
            stack.append(i)
        elif is_symbol(i, ")") or is_symbol(i, "]"):
            if not stack:
                raise fail(i, "unbalanced '%s'" % tokens[i].text)
            open_ = stack[-1]
            if (tokens[open_].text == "(") != (tokens[i].text == ")"):
                raise fail(i, "'%s' closed by '%s'" % (tokens[open_].text, tokens[i].text))
            match[open_] = i
            stack.pop()
    if stack:
        raise fail(stack[-1], "unclosed '%s'" % tokens[stack[-1]].text)

    variables = {}   # name -> Var (scalar variables and map tables share the name space)

    def parse_expression(start, stop, what):
        if start >= stop:
            raise fail(_builtins.min(start, end), "missing value for " + what)
        parser = _ExpressionParser(tokens, start, stop, variables)
        expression = parser.parse()
        if parser.position != stop:
            raise fail(parser.position, "unexpected '%s' in %s" % (tokens[parser.position].text, what))
        return expression

    def split_commas(start, stop):
        parts = []
        first = start
        i = start
        while i < stop:
            if is_symbol(i, "(") or is_symbol(i, "["):
                i = match[i] + 1
                continue
            if is_symbol(i, ","):
                parts.append((first, i))
                first = i + 1
            i += 1
        parts.append((first, stop))
        return parts

    terms = []
    first = 0
    i = 0
    while i < end:
        if is_symbol(i, "(") or is_symbol(i, "["):
            i = match[i] + 1
            continue
        if is_symbol(i, "+"):
            terms.append((first, i))
            first = i + 1
        i += 1
    terms.append((first, end))

    result = []
    tables = []
    for start, stop in terms:
        if start >= stop:
            raise fail(_builtins.min(start, end), "empty term (a '+' without a structure)")
        structure_token = stop
        i = start
        while i < stop:
            if is_symbol(i, "(") or is_symbol(i, "["):
                for j in range(i + 1, match[i]):
                    if (tokens[j].kind == "name" and tokens[j].text.lower() in STRUCTURES
                            and not (j + 1 < end and is_symbol(j + 1, "="))):
                        raise fail(j, "structure '%s' inside brackets; write sill*%s(...)" % (tokens[j].text, tokens[j].text.lower()))
                i = match[i] + 1
                continue
            if tokens[i].kind == "name" and tokens[i].text.lower() in STRUCTURES:
                if structure_token != stop:
                    raise fail(i, "two structures in one term; separate them with '+'")
                structure_token = i
            i += 1
        if structure_token == stop:
            i = start
            while i + 1 < stop:
                if is_symbol(i, "(") or is_symbol(i, "["):
                    i = match[i] + 1
                    continue
                if tokens[i].kind == "name" and is_symbol(i + 1, "(") and tokens[i].text.lower() not in _FUNCTIONS:
                    raise fail(i, "unknown structure '%s' (%s)" % (tokens[i].text, _STRUCTURE_LIST))
                i += 1
            term_text = "".join(tokens[k].text for k in range(start, stop))
            raise fail(start, "term '%s' has no structure (%s)" % (term_text, _STRUCTURE_LIST))
        type = tokens[structure_token].text.lower()

        # partial sill
        if structure_token == start:
            sill = Expr("const", value=1.0)
        else:
            if not is_symbol(structure_token - 1, "*"):
                raise fail(structure_token, "expected '*' between the partial sill and '%s'" % tokens[structure_token].text)
            sill_end = structure_token - 1
            i = start
            while i < sill_end:
                if is_symbol(i, "(") or is_symbol(i, "["):
                    i = match[i] + 1
                    continue
                if is_symbol(i, "-") and i > start:
                    previous = tokens[i - 1]
                    binary = previous.kind in ("number", "name") or (previous.kind == "symbol" and previous.text in ")]")
                    if binary:
                        raise fail(i, "a structure cannot be subtracted; parenthesise a difference in a partial sill: (a - b)*" + type)
                i += 1
            sill = parse_expression(start, sill_end, "the partial sill of " + type)

        # arguments
        arguments_from = arguments_to = structure_token + 1
        if structure_token + 1 < stop:
            parenthesis = is_symbol(structure_token + 1, "(")
            if not parenthesis or match[structure_token + 1] + 1 != stop:
                after = match[structure_token + 1] + 1 if parenthesis else structure_token + 1
                if is_symbol(after, "*") or is_symbol(after, "/"):
                    raise fail(after, "the partial sill must come before the structure: sill*%s(...)" % type)
                raise fail(after, "unexpected '%s' after %s" % (tokens[after].text, type))
            arguments_from = structure_token + 2
            arguments_to = stop - 1
        has_arguments = arguments_from < arguments_to

        if type == "nugget":
            if has_arguments:
                raise fail(arguments_from, "nugget takes no parameter")
            structure = _Structure("nugget")
        elif type == "map":
            if not has_arguments or arguments_to != arguments_from + 1 or tokens[arguments_from].kind != "name":
                raise fail(arguments_from if has_arguments else structure_token, "map takes one table name: map(name)")
            table = tokens[arguments_from].text
            if not is_valid_variable_name(table):
                raise fail(arguments_from, "'%s' is a reserved word and cannot name a map" % table)
            v = variables.get(table)
            if v is None:
                v = variables[table] = Var(table)
            tables.append(v)
            structure = _Structure("map", table=v)
        else:
            if not has_arguments:
                raise fail(structure_token, type + " needs range= or scale=")
            seen = {"range": False, "angle": False, "shape": False}
            rng, use_scale, angle, shape = (), False, (), None
            for a, b in split_commas(arguments_from, arguments_to):
                if a >= b:
                    raise fail(_builtins.min(a, end), "empty argument in %s(...)" % type)
                if tokens[a].kind != "name" or not is_symbol(a + 1, "="):
                    raise fail(a, "expected keyword=value in %s(...)" % type)
                keyword = tokens[a].text.lower()
                is_range = keyword in ("range", "scale")
                is_shape = (keyword == "nu" and type == "matern") or (keyword == "alpha" and type == "stable")
                if not is_range and not is_shape and keyword != "angle":
                    allowed = "range, scale, angle"
                    if type == "matern":
                        allowed += ", nu"
                    if type == "stable":
                        allowed += ", alpha"
                    raise fail(a, "unknown keyword '%s' for %s (allowed: %s)" % (tokens[a].text, type, allowed))
                slot = "range" if is_range else "shape" if is_shape else "angle"
                if seen[slot]:
                    raise fail(a, ("range/scale given twice in " + type) if is_range else "'%s' given twice in %s" % (keyword, type))
                seen[slot] = True
                value_from = a + 2
                values = []
                if is_symbol(value_from, "[") and match[value_from] + 1 == b:
                    if is_shape:
                        raise fail(value_from, keyword + " takes a single value")
                    if match[value_from] == value_from + 1:
                        raise fail(value_from, "empty vector for " + keyword)
                    for e0, e1 in split_commas(value_from + 1, match[value_from]):
                        values.append(parse_expression(e0, e1, "%s of %s" % (keyword, type)))
                else:
                    values.append(parse_expression(value_from, b, "%s of %s" % (keyword, type)))
                if is_range:
                    if len(values) > MAX_DIMS:
                        raise fail(value_from, "%s has more than %d values" % (keyword, MAX_DIMS))
                    use_scale = keyword == "scale"
                    rng = tuple(values)
                elif is_shape:
                    shape = values[0]
                else:
                    if len(values) not in (1, 3):
                        raise fail(value_from, "angle takes one value (2D) or three values (3D)")
                    angle = tuple(values)
            if not seen["range"]:
                raise fail(structure_token, type + " needs range= or scale=")
            if type == "matern" and not seen["shape"]:
                raise fail(structure_token, "matern needs nu=")
            if type == "stable" and not seen["shape"]:
                raise fail(structure_token, "stable needs alpha=")
            structure = _Structure(type, rng, use_scale, angle, shape)

        problem = _check_values(type, sill.value if sill.op == "const" else 0.0,
                                [e.value for e in structure.range if e.op == "const"],
                                [e.value for e in structure.angle if e.op == "const"],
                                structure.shape.value if structure.shape is not None and structure.shape.op == "const" else 1.0)
        if problem:
            raise fail(structure_token, "%s: %s" % (type, problem))
        result.append((sill, structure))

    used = []
    for sill, structure in result:
        for e in [sill] + structure.expressions():
            _collect_vars(e, used)
    for t in tables:
        if _builtins.any(t is u for u in used):
            raise CovarianceError("'%s' is used both as a map table and as a parameter variable" % t.name)
    return Model(result), variables
