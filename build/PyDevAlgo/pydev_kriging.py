"""Shared helpers for the pyDev sequential Gaussian simulation examples.

pyDev calls ``sample()`` once per simulated node, possibly from several threads
at the same time when G2S embeds a free-threaded Python (3.14t). Everything
here is therefore stateless during ``sample()`` or writes only to the row of
the current path index, so no lock is needed.

``sample()`` receives:
  coordinates  int32 array (k, ndim): neighbour offsets, NumPy axis order;
               row 0 is the node itself (offset 0)
  values       float64 array (k, nvar): neighbour values, NaN when unknown
  randUniform  float in [0, 1): the node's random number
  indexInThePath, threadId, fullStationary, variableOfInterest, idTI4Sampling
and returns the simulated values (scalar or one per variable), or a
``(values, index)`` tuple where ``index`` is stored in the index map output.
"""

import hashlib
import os
import tempfile
from statistics import NormalDist

import numpy

_STANDARD_NORMAL = NormalDist()
_EPSILON = 1e-12


def gaussian_quantile(uniform):
    """Standard normal quantile, safe at uniform == 0."""
    return _STANDARD_NORMAL.inv_cdf(min(max(uniform, _EPSILON), 1.0 - _EPSILON))


def informed(coordinates, values):
    """Offsets and values of the informed neighbours of a univariate node."""
    column = values[:, 0]
    mask = ~numpy.isnan(column)
    return coordinates[mask].astype(numpy.float64), column[mask]


def ordinary_kriging(lags, covariance):
    """Ordinary-kriging weights and standard deviation.

    lags: (k, ndim) offsets of the informed neighbours from the node.
    covariance(h): covariance for an array of lag vectors h (..., ndim).
    """
    count = lags.shape[0]
    system = numpy.ones((count + 1, count + 1))
    system[:count, :count] = covariance(lags[:, None, :] - lags[None, :, :])
    system[count, count] = 0.0
    right = numpy.ones(count + 1)
    right[:count] = covariance(lags)
    solution = numpy.linalg.lstsq(system, right, rcond=None)[0]
    sill = float(covariance(numpy.zeros((1, lags.shape[1])))[0])
    variance = sill - float(right @ solution)
    return solution[:count], numpy.sqrt(max(variance, 0.0))


class OrdinaryKrigingSGS:
    """Sequential Gaussian simulation: ordinary kriging + Gaussian noise."""

    def covariance(self, lags, ti_index):
        raise NotImplementedError

    def sill(self, ti_index):
        return float(self.covariance(numpy.zeros((1, self.ndim)), ti_index)[0])

    def __init__(self, TIs, kernel, di, dt, path):
        self.ndim = di.ndim if len(dt) == 1 else di.ndim - 1
        if len(dt) != 1:
            raise ValueError(f"{type(self).__name__} simulates a single continuous variable, got {len(dt)} variables")

    def build(self, n):
        self.n = int(n[0])

    def predict(self, coordinates, values, ti_index):
        lags, data = informed(coordinates, values)
        covariance = lambda h: self.covariance(h, ti_index)
        if data.size == 0:
            return numpy.zeros(0), 0.0, numpy.sqrt(self.sill(ti_index)), data
        weights, deviation = ordinary_kriging(lags, covariance)
        return weights, float(weights @ data), deviation, data

    def sample(self, coordinates, values, randUniform, indexInThePath, threadId,
               fullStationary, variableOfInterest, idTI4Sampling):
        _, mean, deviation, data = self.predict(coordinates, values, idTI4Sampling)
        return mean + gaussian_quantile(randUniform) * deviation, int(data.size)


class KrigingWeightCache:
    """Kriging weights stored per path index, re-used when the path is fixed.

    The geometry of each node's neighbourhood only depends on the path, the
    hard-data pattern, the kernel and n, so with a constant path the weights
    can be computed once and replayed for every new realisation. Each row is
    written by the thread simulating that path index, so concurrent writes
    never overlap. The file is published (atomic rename) in finalize() only
    after a complete run.

    Row layout: [neighbour count, deviation, tag, weight_0 .. weight_{n-1}].
    """

    def __init__(self, path, key_parts):
        if path is None:
            raise ValueError("this algorithm caches kriging weights per path index and needs an explicit path (-sp)")
        self.rows = int(path.size)
        digest = hashlib.blake2b(digest_size=16)
        digest.update(numpy.ascontiguousarray(path).tobytes())
        for part in key_parts:
            digest.update(part if isinstance(part, bytes) else repr(part).encode())
        self.digest = digest.hexdigest()
        folder = os.environ.get("G2S_PYDEV_CACHE") or os.path.join(tempfile.gettempdir(), "g2s_pydev_cache")
        os.makedirs(folder, exist_ok=True)
        self.folder = folder
        self.table = None
        self.temporary = None

    def open(self, n):
        self.width = int(n) + 3
        self.final = os.path.join(self.folder, f"kriging_{self.digest}_n{n}.npy")
        if os.path.isfile(self.final):
            self.table = numpy.load(self.final, mmap_mode="r")
            self.replay = True
        else:
            self.temporary = f"{self.final}.{os.getpid()}.tmp"
            self.table = numpy.lib.format.open_memmap(self.temporary, mode="w+", dtype=numpy.float64,
                                                      shape=(self.rows, self.width))
            self.table[:, 0] = -1.0
            self.replay = False

    def lookup(self, index, count, tag):
        row = self.table[index]
        if int(row[0]) != count or row[2] != tag:
            return None  # geometry differs from the cached run: recompute
        return row[3:3 + count], float(row[1])

    def store(self, index, weights, deviation, tag):
        if self.replay or weights.size > self.width - 3:
            return
        row = self.table[index]
        row[3:3 + weights.size] = weights
        row[1] = deviation
        row[2] = tag
        row[0] = weights.size

    def publish(self):
        if not self.replay and self.temporary:
            self.table.flush()
            self.table = None
            os.replace(self.temporary, self.final)
            self.temporary = None

    def discard(self):
        if self.temporary and os.path.exists(self.temporary):
            self.table = None
            os.remove(self.temporary)
            self.temporary = None


class CachedOrdinaryKrigingSGS(OrdinaryKrigingSGS):
    """OrdinaryKrigingSGS that caches its weights per path index (needs -sp)."""

    def cache_key(self, TIs, kernel, di):
        return [numpy.isnan(di).tobytes(), kernel.shape, type(self).__name__]

    def __init__(self, TIs, kernel, di, dt, path):
        super().__init__(TIs, kernel, di, dt, path)
        self.cache = KrigingWeightCache(path, self.cache_key(TIs, kernel, di))

    def build(self, n):
        super().build(n)
        self.cache.open(self.n)

    def sample(self, coordinates, values, randUniform, indexInThePath, threadId,
               fullStationary, variableOfInterest, idTI4Sampling):
        noise = gaussian_quantile(randUniform)
        lags, data = informed(coordinates, values)
        cached = self.cache.lookup(indexInThePath, data.size, idTI4Sampling)
        if cached is not None:
            weights, deviation = cached
            return float(weights @ data) + noise * deviation, 2
        weights, mean, deviation, data = self.predict(coordinates, values, idTI4Sampling)
        self.cache.store(indexInThePath, weights, deviation, idTI4Sampling)
        return mean + noise * deviation, 1

    def finalize(self):
        self.cache.publish()

    def __del__(self):
        cache = getattr(self, "cache", None)
        if cache is not None:
            cache.discard()
