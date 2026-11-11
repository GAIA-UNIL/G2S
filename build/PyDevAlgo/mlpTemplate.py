"""pyDev example: a small neural network trained on the training image, used in batch mode.

    g2s('-a', 'pyDev', '-pa', 'mlpTemplate', '-ti', ti, '-di', destination,
        '-dt', [1], '-n', 40, '-bs', 64)

A template for machine-learning models (NumPy only, so it runs anywhere; the
same structure works with PyTorch or JAX):

* build() trains a multilayer perceptron on the (categorical, 2D) training
  image: inputs are the n closest informed cells inside a square template
  (one-hot category + presence mask), output is the probability of each
  category at the centre. Training uses random masks that mimic the sparse
  neighbourhoods seen during simulation. The trained weights are cached, so
  later realisations with the same TI and settings skip training.
* sample_batch() turns a whole batch of neighbourhoods into one input matrix,
  runs one forward pass, and draws each node's category from its predicted
  probabilities with the node's own random number.

The index map output stores the number of neighbours that fell inside the
template.
"""

import hashlib
import os
import tempfile

import numpy


class mlpTemplate:
    radius = 7              # template: (2 * radius + 1)^2 - 1 cells around the node
    hidden = 128
    training_samples = 40000
    epochs = 15
    batch = 512
    learning_rate = 2e-3
    seed = 0

    def __init__(self, TIs, kernel, di, dt, path):
        if len(TIs) != 1 or len(dt) != 1 or dt[0] != 1:
            raise ValueError("mlpTemplate expects one categorical variable (-dt [1]) and one 2D training image")
        self.ti = numpy.asarray(TIs[0], dtype=numpy.float64)
        if self.ti.ndim != 2:
            raise ValueError("mlpTemplate expects a 2D training image")
        self.classes = numpy.unique(self.ti[~numpy.isnan(self.ti)])
        side = 2 * self.radius + 1
        offsets = numpy.array([(dy, dx) for dy in range(-self.radius, self.radius + 1)
                               for dx in range(-self.radius, self.radius + 1) if dy or dx])
        order = numpy.argsort((offsets ** 2).sum(axis=1), kind="stable")
        self.offsets = offsets[order]                      # template cells, closest first
        self.lookup = numpy.full((side, side), -1, dtype=numpy.int64)
        self.lookup[self.offsets[:, 0] + self.radius, self.offsets[:, 1] + self.radius] = numpy.arange(len(self.offsets))

    # ---- features: (B, template cells) category indices, -1 = absent
    def features(self, cells):
        count = len(self.classes)
        present = cells >= 0
        x = numpy.zeros(cells.shape + (count + 1,))
        rows, columns = numpy.nonzero(present)
        x[rows, columns, cells[present]] = 1.0
        x[..., count] = present
        return x.reshape(len(cells), -1)

    def forward(self, x):
        hidden = numpy.maximum(x @ self.w1 + self.b1, 0.0)
        logits = hidden @ self.w2 + self.b2
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = numpy.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return hidden, probabilities

    def build(self, n):
        self.n = int(n[0])
        digest = hashlib.blake2b(self.ti.tobytes(), digest_size=16)
        digest.update(repr((self.ti.shape, self.radius, self.hidden, self.training_samples, self.epochs,
                            self.learning_rate, self.seed, self.n)).encode())
        folder = os.environ.get("G2S_PYDEV_CACHE") or os.path.join(tempfile.gettempdir(), "g2s_pydev_cache")
        os.makedirs(folder, exist_ok=True)
        cache = os.path.join(folder, f"mlp_{digest.hexdigest()}.npz")
        if os.path.isfile(cache):
            with numpy.load(cache) as weights:
                self.w1, self.b1, self.w2, self.b2 = (weights[k] for k in ("w1", "b1", "w2", "b2"))
            return
        self.train()
        temporary = f"{cache}.{os.getpid()}.tmp.npz"
        numpy.savez(temporary, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
        os.replace(temporary, cache)

    def training_set(self, rng):
        r = self.radius
        height, width = self.ti.shape
        centres = numpy.stack([rng.integers(r, height - r, self.training_samples),
                               rng.integers(r, width - r, self.training_samples)], axis=1)
        values = self.ti[centres[:, None, 0] + self.offsets[None, :, 0], centres[:, None, 1] + self.offsets[None, :, 1]]
        cells = numpy.searchsorted(self.classes, values)
        # neighbourhood density from almost empty to full, then keep the n closest, as in pyDev
        density = rng.random((self.training_samples, 1)) ** 2
        present = rng.random(cells.shape) < density
        present &= numpy.cumsum(present, axis=1) <= self.n
        cells = numpy.where(present, cells, -1)
        target = numpy.searchsorted(self.classes, self.ti[centres[:, 0], centres[:, 1]])
        return self.features(cells), target

    def train(self):
        rng = numpy.random.default_rng(self.seed)
        x, target = self.training_set(rng)
        inputs, outputs = x.shape[1], len(self.classes)
        self.w1 = rng.normal(0, numpy.sqrt(2.0 / inputs), (inputs, self.hidden))
        self.b1 = numpy.zeros(self.hidden)
        self.w2 = rng.normal(0, numpy.sqrt(1.0 / self.hidden), (self.hidden, outputs))
        self.b2 = numpy.zeros(outputs)
        parameters = [self.w1, self.b1, self.w2, self.b2]
        first = [numpy.zeros_like(p) for p in parameters]
        second = [numpy.zeros_like(p) for p in parameters]
        step = 0
        for _ in range(self.epochs):
            permutation = rng.permutation(len(x))
            for start in range(0, len(x), self.batch):
                index = permutation[start:start + self.batch]
                hidden, probabilities = self.forward(x[index])
                error = probabilities
                error[numpy.arange(len(index)), target[index]] -= 1.0
                error /= len(index)
                grad_w2 = hidden.T @ error
                grad_b2 = error.sum(axis=0)
                back = (error @ self.w2.T) * (hidden > 0)
                grad_w1 = x[index].T @ back
                grad_b1 = back.sum(axis=0)
                step += 1
                for p, g, m, v in zip(parameters, (grad_w1, grad_b1, grad_w2, grad_b2), first, second):
                    m *= 0.9
                    m += 0.1 * g
                    v *= 0.999
                    v += 0.001 * g * g
                    p -= self.learning_rate * (m / (1 - 0.9 ** step)) / (numpy.sqrt(v / (1 - 0.999 ** step)) + 1e-8)

    def sample_batch(self, coordinates, values, randUniform, indexInThePath, idTI4Sampling):
        r = self.radius
        offsets = coordinates.astype(numpy.int64)
        data = values[:, :, 0]
        inside = (numpy.abs(offsets) <= r).all(axis=2) & ~numpy.isnan(data) & (offsets != 0).any(axis=2)
        known = numpy.isin(data, self.classes)
        inside &= known
        cells = numpy.full((len(data), len(self.offsets)), -1, dtype=numpy.int64)
        rows, columns = numpy.nonzero(inside)
        template = self.lookup[offsets[rows, columns, 0] + r, offsets[rows, columns, 1] + r]
        cells[rows, template] = numpy.searchsorted(self.classes, data[rows, columns])
        _, probabilities = self.forward(self.features(cells))
        cumulative = numpy.cumsum(probabilities, axis=1)
        choice = (cumulative < randUniform[:, None]).sum(axis=1)
        choice = numpy.minimum(choice, len(self.classes) - 1)
        return self.classes[choice], inside.sum(axis=1)
