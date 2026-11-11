# pyDev: prototype a sequential simulation in Python, G2S handles the path,
# the neighbour search and the threads. Needs the pyDev algorithm on the server
# (build it with `make pydev` in build/). See build/PyDevAlgo/sgs.py.
import numpy
from g2s import g2s
import matplotlib.pyplot as plt

destination = numpy.full((200, 200), numpy.nan)

result = g2s('-a', 'pyDev',
             '-pa', 'sgs',         # build/PyDevAlgo/sgs.py, class sgs
             '-di', destination,
             '-dt', [0],           # one continuous variable
             '-n', 50,             # informed neighbours per node
             '-j', 8)              # sgs defines sample_batch(): batches of 8 nodes per Python call (-bs to change)
simulation = result["simulation"]

plt.imshow(simulation)
plt.title('pyDev SGS, unconditional')
plt.colorbar()
plt.show()
