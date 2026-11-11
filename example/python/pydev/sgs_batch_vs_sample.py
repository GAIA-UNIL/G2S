# pyDev: the same SGS run per node (sample()) and in batch mode (sample_batch()).
# For a given seed both give the same simulation; batch mode makes one Python
# call per batch instead of one per node. See build/PyDevAlgo/sgs.py.
# For a cheap model like this one, per-node calls running in parallel (with a
# free-threaded Python build) can be as fast or faster; batching pays off when
# each call is expensive, e.g. a neural network on a GPU (see ml_batch_categorical.py).
import numpy
from g2s import g2s
import matplotlib.pyplot as plt

destination = numpy.full((200, 200), numpy.nan)
common = ('-a', 'pyDev', '-pa', 'sgs', '-di', destination, '-dt', [0], '-n', 50, '-s', 42)

per_node = g2s(*common, '-j', 8, '-bs', 1)   # -bs 1: sample(), 8 threads
batched = g2s(*common, '-bs', 32)            # sample_batch(), up to 32 nodes per call

for name, result in (('per node', per_node), ('batch', batched)):
    line = f"{name:>8}: {float(result['duration_ms']) / 1000:.2f} s, {result['python_calls']} Python calls"
    if 'batch_average_size' in result:
        line += f", {float(result['batch_average_size']):.1f} nodes per batch on average"
    print(line)
print("largest difference between the two simulations:",
      numpy.abs(per_node['simulation'] - batched['simulation']).max())

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, result, name in zip(axes, (per_node, batched), ('sample() per node', 'sample_batch()')):
    ax.imshow(result['simulation'])
    ax.set_title(name)
    ax.axis('off')
plt.show()
