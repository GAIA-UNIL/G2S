"""pyDev example: sequential Gaussian simulation with a constant path (-sp).

    g2s('-a', 'pyDev', '-pa', 'sgsCP', '-di', destination, '-dt', [0], '-n', 50, '-sp', path)

Same model as ``sgs``, but the kriging weights of every path index are cached
(see pydev_kriging.KrigingWeightCache). The first run with a given path stores
them; later runs with the same path, hard-data pattern, kernel and n only
replay them, which is much faster. Index map: 1 = weights computed, 2 = replayed.
The cache folder defaults to <tmp>/g2s_pydev_cache (override with G2S_PYDEV_CACHE).
"""

import numpy

from pydev_kriging import CachedOrdinaryKrigingSGS


class sgsCP(CachedOrdinaryKrigingSGS):
    correlation_length = 100.0

    def cache_key(self, TIs, kernel, di):
        return super().cache_key(TIs, kernel, di) + [self.correlation_length]

    def covariance(self, lags, ti_index):
        return numpy.exp(-numpy.linalg.norm(lags, axis=-1) / self.correlation_length)
