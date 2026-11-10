"""pyDev example: sequential Gaussian simulation with an exponential covariance.

    g2s('-a', 'pyDev', '-pa', 'sgs', '-di', destination, '-dt', [0], '-n', 50)

Covariance C(h) = exp(-|h| / 100) (unit sill). The index map output stores the
number of informed neighbours used for each node.
"""

import numpy

from pydev_kriging import OrdinaryKrigingSGS


class sgs(OrdinaryKrigingSGS):
    correlation_length = 100.0

    def covariance(self, lags, ti_index):
        return numpy.exp(-numpy.linalg.norm(lags, axis=-1) / self.correlation_length)
