# Plan once, replay later: the kriging weights of one path are written to a
# file (-wo), then replayed for more realisations (-wi) without solving any
# kriging system. The file is written and read by the server, so the path is
# a path on the server machine.
import os
import tempfile
import numpy as np
from g2s import g2s, cov

model = cov.spherical(range=[50, 25], angle=45)
grid = np.full((300, 300), np.nan)
plan = os.path.join(tempfile.gettempdir(), 'sgs_plan.bin')

first = g2s('-a', 'sgs', '-di', grid, '-dt', [0], *model.args(),
            '-n', 30, '-nr', 8, '-s', 11, '-wo', plan, '-j', 0.5)
print('plan: %s ms, replay: %s ms' % (first['plan_ms'], first['replay_ms']))

# Same inputs, same seed: realisations 8 to 39 of the same path
more = g2s('-a', 'sgs', '-di', grid, '-dt', [0], *model.args(),
           '-n', 30, '-nr', 32, '-ri', 8, '-s', 11, '-wi', plan, '-j', 0.5)
print('replay only: %s s for 32 realisations' % more['time'])

# Replaying realisations 0 to 7 gives back the first run exactly
again = g2s('-a', 'sgs', '-di', grid, '-dt', [0], *model.args(),
            '-n', 30, '-nr', 8, '-s', 11, '-wi', plan, '-j', 0.5)
print('identical:', np.array_equal(again['simulation'], first['simulation']))
