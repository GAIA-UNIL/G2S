---
title: Covariance models
author:
  - Mathieu Gravey
date: 2026-11-21
toc: true
redirect_from:
  - /cov
  - /covEval
---

# Covariance models

The Gaussian algorithms of G2S, [sgs]({{ site.baseurl }}/algorithms/SGS.html), [kriging]({{ site.baseurl }}/algorithms/Kriging.html), [fftma and spectral]({{ site.baseurl }}/algorithms/FFTMA.html), [tb]({{ site.baseurl }}/algorithms/TurningBands.html) and `covEval`, all read the covariance model the same way: as a text given with `-cov`, for example

```
0.1*nugget + 0.6*spherical(range=[40,20], angle=30) + 0.3*exponential(range=100)
```

This page describes that text, how its parameters can be bound to numbers or to maps, the axis and angle conventions, the Python and MATLAB helpers that write the text for you, and `covEval`, which evaluates a model. To choose a simulation method, see [Choosing a Gaussian simulation method]({{ site.baseurl }}/algorithms/GaussianSimulation.html).

## Grammar

A model is a sum of nested structures. The factor in front of a structure is its partial sill; a structure without a factor has a partial sill of 1.

```
model     := term ('+' term)*
term      := [sill '*'] structure
structure := nugget | nugget() | map(NAME) | TYPE(KEY=value, KEY=value, ...)
TYPE      := spherical | exponential | gaussian | cubic | matern | stable
value     := expression | '[' expression (',' expression)* ']'
```

- Structure names, keywords and functions are case-insensitive. Variable names are case-sensitive.
- The keywords of a structure can be given in any order, each at most once.
- Structures cannot be subtracted. A partial sill that is a sum or a difference must be in parentheses: `(1-nug)*spherical(range=40)`, not `1-nug*spherical(range=40)`.
- The partial sill comes before the structure: `0.5*spherical(range=10)`, not `spherical(range=10)*0.5`.

### Canonical form

Every model has a canonical text: terms joined by `+` with a space on each side, every partial sill written (`1*` when absent), lower-case names, keywords in the order `range` (or `scale`), `angle`, `nu` (or `alpha`), constant sub-expressions evaluated. For example `0.3 * Exponential ( Range = 2*(3+4) )` becomes `0.3*exponential(range=14)`. The canonical text of each model is written to the job log (`cov_0`, `cov_1`, ...), and it is what `str(model)` returns in Python and `char(model)` in MATLAB. Reading the canonical text back gives the same model.

### Structures

Let *t* be the lag measured in practical ranges (*t* = 1 on the practical-range ellipse, see [Anisotropy and angles](#anisotropy-and-angles)), and ρ(*t*) the correlation, so that the structure contributes *C*(*h*) = sill × ρ(*t*).

| Structure | ρ(*t*) | `range` in terms of `scale` | Extra parameter |
| --- | --- | --- | --- |
| `nugget` | 1 at a lag of exactly 0, 0 elsewhere | (none) | none |
| `spherical` | 1 − 1.5 *t* + 0.5 *t*³ for *t* < 1, 0 beyond | range = scale | |
| `cubic` | 1 − 7 *t*² + 35/4 *t*³ − 7/2 *t*⁵ + 3/4 *t*⁷ for *t* < 1, 0 beyond (GSLIB cubic) | range = scale | |
| `exponential` | exp(−3 *t*) = exp(−*h*/scale) | range = 3 scale | |
| `gaussian` | exp(−3 *t*²) = exp(−(*h*/scale)²) | range = √3 scale | |
| `stable` | exp(−3 *t*^α) = exp(−(*h*/scale)^α) | range = 3^(1/α) scale | `alpha` in (0, 2] |
| `matern` | 2^(1−ν)/Γ(ν) (*h*/scale)^ν *K*<sub>ν</sub>(*h*/scale) | range = u₉₅(ν) scale | `nu` in (0, 100] |
| `map(NAME)` | read from a table, see [Covariance maps](#covariance-maps) | (none) | none |

**Practical range.** `range` is the practical range of GSLIB: spherical and cubic reach 0 there; exponential, gaussian, stable and matern have dropped to exp(−3) = 4.98% of their sill. For the Matérn model, u₉₅(ν) is the distance at which the correlation of scale 1 equals exp(−3): u₉₅(0.5) = 3, u₉₅(1.5) = 4.7490, u₉₅(2.5) = 5.9245. With the same range, `matern` with `nu=0.5` is exactly `exponential`, `stable` with `alpha=1` is `exponential` and `stable` with `alpha=2` is `gaussian`. The Matérn model is the plain (*h*/scale)^ν *K*<sub>ν</sub>(*h*/scale) form, not the variant with √(2ν) *h*/scale.

### Parameters

| Keyword | Value | Constraint |
| --- | --- | --- |
| partial sill | the factor in front of the structure | ≥ 0 |
| `range=r` or `range=[r0, r1, ...]` | practical range, in cells; one value (isotropic) or one per axis of the grid (geometric anisotropy) | > 0 |
| `scale=s` or `scale=[s0, s1, ...]` | the scale parameter instead of the practical range, same shape as `range` | > 0; not together with `range` |
| `angle=a` or `angle=[a, b, c]` | rotation in degrees: one angle in 2D, one or three in 3D | finite |
| `nu=v` | Matérn smoothness | 0 < ν ≤ 100 |
| `alpha=v` | stable exponent | 0 < α ≤ 2 |

`range` or `scale` is required by every structure except `nugget` and `map`; `nu` is required by `matern` and `alpha` by `stable`. `nugget` and `map` take no parameter. Models are defined for 1 to 4 axes; `angle` only for 2D and 3D grids.

### Expressions

Any value (partial sill, range, scale, angle, nu, alpha) can be an expression made of

- numbers: `12`, `0.5`, `.5`, `1e-3`;
- variable names: letters, digits and `_`, not starting with a digit, bound with `-cvar` (see [Variables](#variables-and-spatially-varying-parameters));
- `+ - * /`, parentheses and unary minus;
- the functions `min(a, b)`, `max(a, b)`, `sqrt(x)`, `exp(x)`, `log(x)` (natural logarithm) and `abs(x)`.

For example: `nug*nugget + (1-nug)*spherical(range=[L, L/2], angle=theta)`.

Structure names, keywords (`range`, `scale`, `angle`, `nu`, `alpha`), function names and `h` are reserved and cannot be variable names. The lag is never available to an expression: every parameter is a constant at each node, so every structure stays a valid covariance.

Many examples of valid and invalid texts, with their canonical form or the error they produce, are in [test/geostat/cov_cases.txt](https://github.com/GAIA-UNIL/G2S/blob/master/test/geostat/cov_cases.txt).

## Anisotropy and angles

Lags are counted in grid cells, one component per array axis. **Component 0 is the last index of the array** (the columns of a 2D array), component 1 the index before it (the rows of a 2D array), component 2 the first index of a 3D array. `range=[r0, r1, r2]`, `angle`, and the lags given to `covEval` use this order. This holds both for numpy arrays and for MATLAB arrays sent through the G2S interfaces.

In 2D, `angle=a` turns the direction of `range[0]` from axis 0 towards axis 1 by *a* degrees. With axis 0 to the right and axis 1 upwards, *a* > 0 is counter-clockwise; on an image displayed with row 0 at the top (`imshow`, `imagesc`), it appears clockwise.

In 3D, `angle=[a, b, c]` (`angle=a` means `[a, 0, 0]`) is the rotation *R* = *R*<sub>01</sub>(*a*) *R*<sub>02</sub>(*b*) *R*<sub>12</sub>(*c*), where *R*<sub>ij</sub>(θ) turns axis *i* towards axis *j* by θ. The direction of `range[0]` is (cos *a* cos *b*, sin *a* cos *b*, sin *b*): *a* is its azimuth in the plane of axes 0 and 1, measured from axis 0 towards axis 1, *b* its elevation towards axis 2, and *c* turns the two other principal axes about it (axis 1 towards axis 2). The *k*-th column of *R* is the direction of `range[k]`, and *t* = ‖diag(1/range) *R*<sup>T</sup> *h*‖.

### Worked example

`spherical(range=[40, 10], angle=30)` on a 2D grid:

- **numpy**, array of shape `(ny, nx)`, displayed with `imshow`: the long range (40 cells) runs along a direction that starts from the column axis (`a[i, :]`, to the right) and turns 30° towards increasing row index, so the structures point down and to the right on the image. A lag of 40 cells in that direction is the lag `[40 cos 30°, 40 sin 30°] = [34.6, 20]`, that is 34.6 columns and 20 rows, and the covariance is 0 there. Half of it, `[17.3, 10]`, gives ρ(0.5) = 0.3125. Perpendicular to it the range is 10 cells.
- **MATLAB**, matrix of size `ny × nx`, displayed with `imagesc`: identical. `range(1) = 40` is along the columns (the second index, `A(i, :)`), the major axis turns 30° towards increasing row index, so the structures point down and to the right, and a lag `[dx dy]` is a shift of `dx` columns and `dy` rows.

In 3D, for a numpy array of shape `(nz, ny, nx)` or a MATLAB array of size `nz × ny × nx`, `range=[rx, ry, rz]` gives `rx` along the last index, `ry` along the middle one and `rz` along the first one.

## Variables and spatially varying parameters

Variables used in a model are bound with `-cvar name value`, once per variable:

```
-cov "nug*nugget + (1-nug)*spherical(range=[L, L/2], angle=theta)" -cvar nug 0.1 -cvar L 40 -cvar theta 30
```

- `value` is a number (a constant) or an image: an image on the `-di` grid (same size, one value per cell) makes the parameter vary in space. An image whose values are all equal is treated as a constant.
- `-cvar name=value` (one token) is accepted too. A negative constant can be written `-cvar theta=-30`, or put directly in the model text.
- Every variable must be bound, and every bound variable must be used by a model.

Through the interfaces, an image is a numpy array or a MATLAB matrix of the size of `-di`, given as the value: `'-cvar', 'L', rangeMap`. The helpers below do this for you.

### Validity of spatially varying parameters

The parameters are resolved at a node: every variable takes its value at that node, and the structure is then an ordinary stationary covariance. The kriging system of a node (in `sgs` and in local `kriging`) is built entirely with the covariance resolved at that node; the parameters at its neighbours are not used. This is a **local stationarity** approximation: each system is valid, but the set of local models does not define the covariance of one random field, so the simulated field has the local model only approximately, and abrupt changes of the parameters produce visible transitions. Let the parameters vary slowly compared with the ranges.

Spatially varying parameters (and `-ii` below) are supported by `sgs`, `kriging` (local method) and `covEval`. `fftma`, `spectral` and `tb` simulate stationary models only and refuse them.

## Regions: several models selected by an index (`-ii`)

Give several models (repeat `-cov`, or give several texts after one `-cov`) and an index image with `-ii`: an integer per cell of the `-di` grid, 0 selecting the first model, 1 the second, and so on. Without `-ii`, model 0 is used everywhere. Variables can be shared between models.

As for spatially varying parameters, a node uses the model of its own region for its whole kriging system, including neighbours in another region.

## Covariance maps

`map(NAME)` reads the covariance from a table bound with `-cvar NAME image`:

- the table can have any size, with one axis per lag component; lag 0 is at the centre, at index `size/2` (integer division, 0-based) on each axis: `shape//2` in numpy, `floor(size/2)+1` in MATLAB's 1-based indexing;
- *C*(*h*) = coefficient × *M*[centre + round(*h*)], and 0 for lags outside the table;
- the partial sill is the coefficient times the value at the centre;
- a map can be combined with other structures: `map(C) + 0.1*nugget`.

The table must be a valid (positive definite) covariance. An experimental covariance map cut out of an image usually is not, and kriging systems built from it can be unstable: in our tests `sgs` with such a map produced values far outside the range of the data. A table sampled from a valid covariance, wide enough for the covariance to vanish at its edge, works (see the example below).

`map` is supported by `sgs`, `kriging`, `fftma`, `spectral` and `covEval`. `tb` refuses it (there is no turning-bands line process for a tabulated covariance).

## Python helper

`from g2s import cov` builds model texts and their `-cvar` bindings from Python objects, so that you do not write the text by hand:

```python
from g2s import g2s, cov
model = 0.1*cov.nugget() + 0.9*cov.spherical(range=[40, 20], angle=30)
str(model)    # '0.1*nugget + 0.9*spherical(range=[40, 20], angle=30)'
model.args()  # ['-cov', '0.1*nugget + 0.9*spherical(range=[40, 20], angle=30)']
result = g2s('-a', 'sgs', '-di', di, '-dt', [0], *model.args(), '-n', 50)
```

- Structures: `cov.nugget()`, `cov.spherical`, `cov.exponential`, `cov.gaussian`, `cov.cubic` (keywords `range`, `scale`, `angle`), `cov.matern` (`nu`), `cov.stable` (`alpha`) and `cov.map(table)`. Combine them with `+` and scale them with `*` or `/`; subtracting a structure raises an error, as in the grammar.
- A parameter is a number, a numpy array on the `-di` grid (a spatially varying parameter, which gets an automatic name `_a1`, `_a2`, ...), a named variable `cov.var('L', value)` that can be reused, or an expression of those with `+ - * /`, unary minus, `cov.min`, `cov.max`, `cov.sqrt`, `cov.exp`, `cov.log`, `cov.abs`. `range`, `scale` and `angle` take a list for one value per axis: `range=[40, L]`. A numpy array is always one spatially varying value, never a list of per-axis values.
- `model.args()` returns the list to splice into `g2s(...)`: `-cov` and its text, then a `-cvar` pair for each variable. Constants are passed as text with full precision.
- `cov.parse(text, name=value, ...)` reads a text back into a model, with optional bindings. Errors are raised as `cov.CovarianceError`, with the same messages as the server.
- `cov.regions([m0, m1], index)` groups models for `-ii`; its `args()` ends with `'-ii', index`.
- `model(h)` and `model.variogram(h)` evaluate the model by running `covEval` on a G2S server (the server address options `sa=`, `p=` are passed through). `h` is a scalar, a vector of 1D lags, or an `N × ndim` array with column 0 along the last numpy axis. For a spatially varying model or a region set, `pos=[column, row]` (0-based, same component order) gives the node where the model is resolved. When no server answers, `cov.ServerError` is raised.

## MATLAB helper

`g2sCov` (in `build/matlab-build`, next to the `g2s` MEX file) does the same in MATLAB:

```matlab
model = 0.1*g2sCov.nugget() + 0.9*g2sCov.spherical('range', [40 20], 'angle', 30);
char(model)     % '0.1*nugget + 0.9*spherical(range=[40, 20], angle=30)'
args = model.args();
result = g2s('-a', 'sgs', '-di', di, '-dt', [0], args{:}, '-n', 50);
```

- Structures are static methods with name/value pairs: `g2sCov.spherical('range', r, 'angle', a)`, `g2sCov.matern('range', r, 'nu', 1.5)`, `g2sCov.map(table)`.
- A parameter is a number, a matrix on the `-di` grid (spatially varying), `g2sCov.var('L', value)`, or an expression with `+ - * /` (also `.*` and `./`), unary minus, `min`, `max`, `sqrt`, `exp`, `log`, `abs`. `range`, `scale` and `angle` take one value per axis as a short numeric vector (`[40 20]`), `[L, L/2]`, or a cell `{L, 20}`. A numeric array that is not such a short vector is one spatially varying value; wrap a small map in `g2sCov.var` to avoid ambiguity.
- `g2sCov.parse(text, 'name', value, ...)`, `g2sCov.regions({m0, m1}, index)`, `model.args()`, `model.variables()`.
- `model.eval(h)` and `model.variogram(h)` run `covEval` on a server; options such as `'-sa', host, '-p', port` are passed through, `'pos', [column row]` gives the node for a spatially varying model, and `'ndim', 2` marks a single 2D lag (`model.eval([10 0], 'ndim', 2)`).

## covEval: evaluate a model

`covEval` (also `CovEval`, `coveval`) evaluates one or more models at given lags, so that a model can be plotted or checked with exactly the implementation used by the simulations.

Usage: `result = g2s('-a', 'covEval', flag1, value1, ...)`. Outputs: `result["covariance"]` / `result.covariance` (one value per lag) and, with `-vario`, `result["variogram"]` / `result.variogram`.

| Flag | Description | Default | Mandatory |
| --- | --- | --- | --- |
| `-cov` | Covariance model(s). | | &#x2714; |
| `-cvar` | `name value`: binds a model variable (number, image on the grid, or `map(name)` table of any size); repeat as needed. | | |
| `-ii` | Model index image (0-based) when several models are given. | model 0 | |
| `-lag` | `N × ndim` array of lags, one per row; column 0 is the shift along the last array index. A vector is `N` lags in 1D. | | &#x2714; |
| `-pos` | `1 × ndim` (the same node for every lag) or `N × ndim` (one node per lag): integer cell coordinates, in the same component order as `-lag`, where a spatially varying model is resolved. Required when the model varies in space. | | |
| `-di` | Grid. Optional when a `-cvar` or `-ii` image gives it. | | |
| `-vario` | Also output the variogram *C*(0) − *C*(*h*). | off | |

## Examples

These examples need a running G2S server. They are in `example/python/cov/` and `example/matlab/cov/`.

### Building and evaluating a model
{% include include_code.md examplePath="cov/covariance_model" %}

### Spatially varying range and angle
{% include include_code.md examplePath="cov/spatially_varying" %}

### Regions with different models
{% include include_code.md examplePath="cov/regions" %}

### Covariance map
{% include include_code.md examplePath="cov/covariance_map" %}
