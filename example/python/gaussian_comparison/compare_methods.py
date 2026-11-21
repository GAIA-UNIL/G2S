# Compares the Gaussian simulation algorithms of G2S (sgs, fftma, spectral, tb)
# on the same model: computing time, and how well the simulated covariance
# matches the model (experimental covariance averaged over many realisations).
# The numbers quoted on the "Choosing a Gaussian simulation method" page were
# produced with this script. Needs a running G2S server (G2S_PORT selects
# another port than the default 8128). Each time is the median of three runs;
# the load average of the machine is printed before and after.
import os
import numpy as np
from g2s import g2s, cov

model = cov.spherical(range=30)
methods = ['sgs', 'fftma', 'spectral', 'tb']
port = int(os.environ.get('G2S_PORT', 8128))
server = ['-sa', 'localhost', '-p', port]
repeats = 3


def simulate(method, di, *extra):
    result = g2s('-a', method, '-di', di, '-dt', [0], *model.args(), '-s', 1, '-j', 8, '-silent', *server, *extra)
    return result['simulation'], result['time']  # time: computing time reported by the job, in s


def timed(method, di, *extra):
    return float(np.median([simulate(method, di, *extra)[1] for _ in range(repeats)]))


print('load average (1, 5, 15 min): %.2f %.2f %.2f' % os.getloadavg())

# ---- computing time: 16 realisations of a 1000 x 1000 grid, without and with 2000 hard data
grid = np.full((1000, 1000), np.nan)
reference, _ = simulate('fftma', grid)
rng = np.random.default_rng(0)
data = grid.copy()
cells = rng.choice(data.size, 2000, replace=False)
data.flat[cells] = reference.flat[cells]
print('computing time for 16 realisations of 1000 x 1000 (s)')
for method in methods:
    unconditional = timed(method, grid, '-nr', 16)
    conditional = timed(method, data, '-nr', 16)
    print('  %-9s unconditional %6.2f   2000 hard data %6.2f   (load %.1f)' % (method, unconditional, conditional, os.getloadavg()[0]))

# ---- time against the number of realisations (1000 x 1000, unconditional)
print('computing time against the number of realisations, 1000 x 1000 (s)')
for method in methods:
    times = [timed(method, grid, '-nr', count) for count in (1, 16, 64)]
    print('  %-9s 1: %6.2f   16: %6.2f   64: %6.2f   (load %.1f)' % (method, *times, os.getloadavg()[0]))

# ---- covariance: 500 realisations of a 128 x 128 grid, lags along the columns.
# The mean is known (0), so the covariance is the mean product of values.
lags = np.array([0, 5, 10, 20, 30])
expected = model(np.column_stack([lags, np.zeros_like(lags)]), sa='localhost', p=port)
print('covariance at lags', lags)
print('  %-9s %s' % ('model', np.round(expected, 3)))
small = np.full((128, 128), np.nan)
for method in methods:
    fields, _ = simulate(method, small, '-nr', 500)
    per_realisation = np.array([np.mean(fields[:, :128 - h, :] * fields[:, h:, :], axis=(0, 1)) for h in lags])
    mean = per_realisation.mean(axis=1)
    error = per_realisation.std(axis=1) / np.sqrt(per_realisation.shape[1])
    print('  %-9s %s  (+/- %s)' % (method, np.round(mean, 3), np.round(2 * error, 3)))

print('load average (1, 5, 15 min): %.2f %.2f %.2f' % os.getloadavg())
