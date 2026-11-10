"""pyDev example: sequential Gaussian simulation with covariance maps given as -ti.

    g2s('-a', 'pyDev', '-pa', 'sgsTiCP', '-di', destination, '-dt', [0], '-n', 50,
        '-sp', path, '-ti', covariance_map_1, covariance_map_2, ..., '-ii', map_index)

Each -ti is a covariance map: its centre is lag 0 and C(h) is read at
centre + h (0 outside the map), so anisotropic covariances work as given.
With several maps, -ii selects the map used at each node (non-stationary
simulation); without -ii the first map is used. Kriging weights are cached per
path index as in ``sgsCP``.
"""

import hashlib

import numpy

from pydev_kriging import CachedOrdinaryKrigingSGS


class sgsTiCP(CachedOrdinaryKrigingSGS):
    def __init__(self, TIs, kernel, di, dt, path):
        if not TIs:
            raise ValueError("sgsTiCP needs at least one covariance map as -ti")
        self.maps = [numpy.asarray(ti, dtype=numpy.float64) for ti in TIs]
        if any(m.ndim != self.maps[0].ndim for m in self.maps):
            raise ValueError("all covariance maps must have the same number of dimensions")
        self.centers = [numpy.array(m.shape) // 2 for m in self.maps]
        super().__init__(TIs, kernel, di, dt, path)

    def cache_key(self, TIs, kernel, di):
        digest = hashlib.blake2b(digest_size=16)
        for m in self.maps:
            digest.update(repr(m.shape).encode())
            digest.update(m.tobytes())
        return super().cache_key(TIs, kernel, di) + [digest.digest()]

    def covariance(self, lags, ti_index):
        index = ti_index if 0 <= ti_index < len(self.maps) else 0
        cmap, center = self.maps[index], self.centers[index]
        position = numpy.rint(lags).astype(numpy.int64) + center
        inside = numpy.all((position >= 0) & (position < cmap.shape), axis=-1)
        result = numpy.zeros(position.shape[:-1])
        result[inside] = cmap[tuple(position[inside].T)]
        return result
