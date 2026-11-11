"""pyDev example: sequential Gaussian simulation with an exponential covariance.

    g2s('-a', 'pyDev', '-pa', 'sgs', '-di', destination, '-dt', [0], '-n', 50, '-j', 8)

Covariance C(h) = exp(-|h| / 100) (unit sill). The index map output stores the
number of informed neighbours used for each node.

The class defines both sample() (one node) and sample_batch() (one linear solve
for a batch of nodes, the pattern to follow for ML models). pyDev uses
sample_batch() by default with batches of -j nodes (-bs to change); -bs 1
switches to sample(). Both give the same simulation up to rounding.
"""

import numpy

from pydev_kriging import OrdinaryKrigingSGS


class sgs(OrdinaryKrigingSGS):
    correlation_length = 100.0

    def covariance(self, lags, ti_index):
        return numpy.exp(-numpy.linalg.norm(lags, axis=-1) / self.correlation_length)

    def sample_batch(self, coordinates, values, randUniform, indexInThePath, idTI4Sampling):
        return self.sample_batch_kriging(coordinates, values, randUniform, indexInThePath, idTI4Sampling)
