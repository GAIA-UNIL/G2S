"""Tests of g2s.cov, the covariance-model builder of the Gaussian algorithms.

Run with ``python build/python-build/tests/test_cov.py`` (or pytest).

* Without a server: every line of test/geostat/cov_cases.txt through
  cov.parse, builder texts and -cvar lists, round trips, validation and the
  float-precision pitfall. These work from a source checkout even when the
  compiled extension is not installed (cov.py is then loaded directly).
* With build/c++-build/covEval built (or G2S_COVEVAL=/path/to/covEval): the
  canonical text and the argument list are checked against the real C++ by
  running covEval standalone (it reads and writes /tmp/G2S/data).
* With a reachable G2S server (G2S_TEST_SERVER, default localhost, and
  G2S_TEST_PORT, default 8128): model(h) and model.variogram(h) through the
  covEval algorithm. Skipped when no server answers.
"""

import gzip
import importlib.util
import math
import os
import random
import struct
import subprocess
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CASES = os.path.join(REPO, "test", "geostat", "cov_cases.txt")
COVEVAL = os.environ.get("G2S_COVEVAL", os.path.join(REPO, "build", "c++-build", "covEval"))
SERVER = os.environ.get("G2S_TEST_SERVER", "localhost")
PORT = int(os.environ.get("G2S_TEST_PORT", "8128"))

try:
    from g2s import cov
    HAVE_INTERFACE = True
except Exception:  # compiled extension or pyzmq missing: test the pure-Python part
    spec = importlib.util.spec_from_file_location("g2s_cov", os.path.join(HERE, "..", "g2s", "cov.py"))
    cov = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cov)
    HAVE_INTERFACE = False


def read_cases():
    cases = []
    with open(CASES, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            source, expected = line.split(" ==> ", 1)
            cases.append((number, source.strip(), expected.strip()))
    return cases


class CasesFile(unittest.TestCase):
    @unittest.skipUnless(os.path.isfile(CASES), "test/geostat/cov_cases.txt not found")
    def test_every_case(self):
        cases = read_cases()
        self.assertGreater(len(cases), 100)
        for number, source, expected in cases:
            with self.subTest(line=number, model=source):
                if expected.startswith("!error"):
                    fragment = expected[len("!error"):].strip()
                    with self.assertRaises(cov.CovarianceError) as caught:
                        cov.parse(source)
                    self.assertIn(fragment, str(caught.exception))
                    self.assertTrue(str(caught.exception).startswith('covariance model "'))
                else:
                    self.assertEqual(str(cov.parse(source)), expected)
                    self.assertEqual(str(cov.parse(expected)), expected)


class Numbers(unittest.TestCase):
    def test_format_number(self):
        for value, text in [(0.0, "0"), (-0.0, "0"), (1.0, "1"), (-3.0, "-3"), (100000.0, "100000"), (0.1, "0.1"),
                            (1e-05, "1e-05"), (1.5e16, "1.5e+16"), (1 / 3, "0.3333333333333333"),
                            (0.1 + 0.2, "0.30000000000000004"), (123456789012345678.0, "1.2345678901234568e+17"),
                            (999999999999999.0, "999999999999999"), (1e15, "1e+15")]:
            self.assertEqual(cov.format_number(value), text)

    def test_round_trip_random(self):
        rnd = random.Random(3)
        for _ in range(2000):
            value = rnd.choice([rnd.random(), rnd.uniform(-1e6, 1e6), 10 ** rnd.uniform(-300, 300),
                                float(rnd.randint(-10 ** 16, 10 ** 16))])
            self.assertEqual(float(cov.format_number(value)), value)


class Builders(unittest.TestCase):
    def test_specification_examples(self):
        m = 0.1 * cov.nugget() + 0.6 * cov.spherical(range=[40, 20], angle=30) + 0.3 * cov.exponential(range=100)
        self.assertEqual(str(m), "0.1*nugget + 0.6*spherical(range=[40, 20], angle=30) + 0.3*exponential(range=100)")
        self.assertEqual(m.args(), ["-cov", str(m)])

        range_map = np.full((30, 40), 25.0)
        angle_map = np.zeros((30, 40))
        nug, L, theta = cov.var("nug", 0.1), cov.var("L", range_map), cov.var("theta", angle_map)
        m = nug * cov.nugget() + (1 - nug) * cov.spherical(range=[L, L / 2], angle=theta)
        self.assertEqual(str(m), "nug*nugget + (1 - nug)*spherical(range=[L, L/2], angle=theta)")
        args = m.args()
        self.assertEqual(args[:2], ["-cov", str(m)])
        self.assertEqual(args[2:5], ["-cvar", "nug", "0.1"])
        self.assertEqual(args[5:7], ["-cvar", "L"])
        self.assertIs(args[7], range_map)
        self.assertEqual(args[8:10], ["-cvar", "theta"])
        self.assertIs(args[10], angle_map)
        self.assertEqual(len(args), 11)

    def test_structures(self):
        cases = [
            (cov.nugget(), "1*nugget"),
            (cov.spherical(range=10), "1*spherical(range=10)"),
            (cov.exponential(scale=5), "1*exponential(scale=5)"),
            (cov.gaussian(range=[12.5]), "1*gaussian(range=12.5)"),
            (cov.cubic(range=[10, 20, 30], angle=[10, 20, 30]), "1*cubic(range=[10, 20, 30], angle=[10, 20, 30])"),
            (cov.matern(nu=0.5, angle=45, range=[30, 10]), "1*matern(range=[30, 10], angle=45, nu=0.5)"),
            (cov.matern(scale=[3, 1], nu=2.5), "1*matern(scale=[3, 1], nu=2.5)"),
            (cov.stable(alpha=2, scale=4), "1*stable(scale=4, alpha=2)"),
            (0.7 * cov.map("C"), "0.7*map(C)"),
            (cov.spherical(range=10, angle=-30), "1*spherical(range=10, angle=-30)"),
            (2 * cov.nugget(), "2*nugget"),
            (cov.nugget() * 2, "2*nugget"),
            (cov.nugget() / 4, "0.25*nugget"),
            (0.5 * (cov.nugget() + 2 * cov.spherical(range=1)), "0.5*nugget + 1*spherical(range=1)"),
            (sum([0.5 * cov.nugget(), 0.5 * cov.gaussian(range=3)]), "0.5*nugget + 0.5*gaussian(range=3)"),
            (np.float64(0.25) * cov.nugget(), "0.25*nugget"),
            (np.float32(0.5) * cov.nugget(), "0.5*nugget"),
            (cov.spherical(range=np.int64(7)), "1*spherical(range=7)"),
            ((1 - 0.1) * cov.nugget(), "0.9*nugget"),
            (cov.exponential(range=1 / 3), "1*exponential(range=0.3333333333333333)"),
        ]
        for model, text in cases:
            with self.subTest(text=text):
                self.assertEqual(str(model), text)
                self.assertEqual(str(cov.parse(text)), text)

    def test_expressions(self):
        a, b, c, d, L = (cov.var(n, 1.0) for n in ("a", "b", "c", "d", "L"))
        nug, R, nu = cov.var("nug", 0.2), cov.var("R", 3.0), cov.var("nu_map", 1.5)
        cases = [
            (a * b * cov.spherical(range=L), "a*b*spherical(range=L)"),
            (a / 2 * cov.nugget(), "a/2*nugget"),
            ((a + b) / (c * d) * cov.nugget(), "(a + b)/(c*d)*nugget"),
            (cov.spherical(range=a - b), "1*spherical(range=a - b)"),
            (cov.spherical(range=a - (b - c)), "1*spherical(range=a - (b - c))"),
            (cov.spherical(range=(a - b) - c), "1*spherical(range=a - b - c)"),
            (cov.spherical(range=a * (b * c)), "1*spherical(range=a*(b*c))"),
            (cov.spherical(range=a / (b / c)), "1*spherical(range=a/(b/c))"),
            (cov.spherical(range=cov.min(L, 50), angle=-a), "1*spherical(range=min(L, 50), angle=-a)"),
            (cov.exponential(range=cov.sqrt(L) * 2 + cov.max(a, b)), "1*exponential(range=sqrt(L)*2 + max(a, b))"),
            (cov.exponential(range=cov.exp(cov.log(L))), "1*exponential(range=exp(log(L)))"),
            (cov.gaussian(range=2 * -L), "1*gaussian(range=2*-L)"),
            (cov.gaussian(range=-(L - 2 * L)), "1*gaussian(range=-(L - 2*L))"),
            (cov.exponential(range=cov.abs(L)), "1*exponential(range=abs(L))"),
            ((1 - nug) * cov.matern(range=R, nu=nu), "(1 - nug)*matern(range=R, nu=nu_map)"),
            (cov.exponential(range=L + (-2)), "1*exponential(range=L + -2)"),
            (cov.exponential(range=2 - (-L)), "1*exponential(range=2 - -L)"),
            (cov.exponential(range=cov.sqrt(16) + cov.max(1, 2) - cov.min(3, 4) + cov.abs(-5)), "1*exponential(range=8)"),
            (a * cov.nugget() * b, "b*a*nugget"),
        ]
        for model, text in cases:
            with self.subTest(text=text):
                self.assertEqual(str(model), text)
                self.assertEqual(str(cov.parse(text)), text)

    def test_automatic_names(self):
        sill, rng = np.full((4, 5), 0.5), np.full((4, 5), 10.0)
        m = sill * cov.nugget() + cov.exponential(range=[rng, rng / 2]) + cov.gaussian(range=rng)
        self.assertEqual(str(m), "_a1*nugget + 1*exponential(range=[_a2, _a3]) + 1*gaussian(range=_a2)")
        args = m.args()
        self.assertEqual([args[i] for i in (2, 3, 5, 6, 8, 9)], ["-cvar", "_a1", "-cvar", "_a2", "-cvar", "_a3"])
        self.assertIs(args[4], sill)
        self.assertIs(args[7], rng)
        # user names are never shadowed by automatic ones
        m = cov.var("_a1", 2.0) * cov.nugget() + cov.exponential(range=rng)
        self.assertEqual(str(m), "_a1*nugget + 1*exponential(range=_a2)")

    def test_shared_variables_and_regions(self):
        L = cov.var("L", 30.0)
        m0 = 0.2 * cov.nugget() + 0.8 * cov.spherical(range=L)
        m1 = cov.exponential(range=L / 2, angle=cov.var("theta", -15))
        index = np.zeros((3, 4))
        index[1:, :] = 1
        s = cov.regions([m0, m1], index)
        args = s.args()
        self.assertEqual(args[:3], ["-cov", "0.2*nugget + 0.8*spherical(range=L)", "1*exponential(range=L/2, angle=theta)"])
        self.assertEqual(args[3:-2], ["-cvar", "L", "30", "-cvar", "theta=-15"])
        self.assertEqual(args[-2], "-ii")
        np.testing.assert_array_equal(args[-1], index)
        with self.assertRaises(cov.CovarianceError):
            cov.regions([m0, m1], index + 1)
        with self.assertRaises(cov.CovarianceError):
            cov.regions([m0], index - 0.5)
        with self.assertRaises(cov.CovarianceError):
            cov.regions([m0, cov.exponential(range=cov.var("L", 31.0))], index).args()

    def test_parse_with_bindings(self):
        rm = np.full((5, 6), 12.0)
        m = cov.parse("nug*nugget + (1-nug)*exponential(range=L)", nug=0.25, L=rm)
        self.assertEqual(str(m), "nug*nugget + (1 - nug)*exponential(range=L)")
        args = m.args()
        self.assertEqual(args[2:5], ["-cvar", "nug", "0.25"])
        self.assertIs(args[7], rm)
        self.assertEqual(list(m.variables), ["nug", "L"])
        unbound = cov.parse("nug*nugget")
        with self.assertRaisesRegex(cov.CovarianceError, "has no value"):
            unbound.args()
        with self.assertRaisesRegex(cov.CovarianceError, "does not use it"):
            cov.parse("nug*nugget", other=1)
        # an unbound parsed variable takes the value of a bound one with the same name
        joined = cov.parse("exponential(range=L)") + cov.gaussian(range=cov.var("L", 4))
        self.assertEqual(joined.args()[-3:], ["-cvar", "L", "4"])
        # a parsed model is a normal model
        m2 = 0.5 * cov.parse("exponential(range=L)", L=3) + cov.nugget()
        self.assertEqual(str(m2), "0.5*exponential(range=L) + 1*nugget")

    def test_round_trip_builders(self):
        rnd = random.Random(7)
        names = [cov.var(n, 1.0) for n in ("a", "b", "L", "nug")]

        def expr(depth=0):
            r = rnd.random()
            if depth > 3 or r < 0.35:
                return rnd.choice(names + [0.5, 2, 3.25, 1e-7, 7])
            if r < 0.45:
                inner = expr(depth + 1)
                return -inner if isinstance(inner, cov.Expr) else -cov.var("a", 1.0)
            if r < 0.55:
                return rnd.choice([cov.min, cov.max])(expr(depth + 1), expr(depth + 1))
            if r < 0.6:
                return rnd.choice([cov.sqrt, cov.abs])(rnd.choice(names))
            x, y = expr(depth + 1), expr(depth + 1)
            op = rnd.choice("+-*/")
            if not isinstance(x, cov.Expr) and not isinstance(y, cov.Expr):
                x = rnd.choice(names)
            return {"+": lambda: x + y, "-": lambda: x - y, "*": lambda: x * y, "/": lambda: x / y}[op]()

        count = 0
        for _ in range(500):
            try:
                e = expr()
                m = cov.exponential(range=e) + cov.var("s", 1.0) * cov.nugget()
            except cov.CovarianceError:
                continue
            text = str(m)
            self.assertEqual(str(cov.parse(text)), text)
            count += 1
        self.assertGreater(count, 300)


class Validation(unittest.TestCase):
    def assertError(self, fragment, function, *args, **kwargs):
        with self.assertRaises(cov.CovarianceError) as caught:
            function(*args, **kwargs)
        self.assertIn(fragment, str(caught.exception))

    def test_names(self):
        self.assertError("reserved word", cov.var, "range", 1)
        self.assertError("reserved word", cov.var, "Spherical", 1)
        self.assertError("reserved word", cov.var, "H", 1)
        self.assertError("reserved word", cov.var, "sqrt", 1)
        self.assertError("not a valid variable name", cov.var, "2L", 1)
        self.assertError("not a valid variable name", cov.var, "L-1", 1)
        self.assertError("not a valid variable name", cov.var, "", 1)
        self.assertTrue(cov.is_valid_variable_name("_x1"))
        self.assertFalse(cov.is_valid_variable_name("map"))

    def test_parameters(self):
        self.assertError("needs range= or scale=", cov.spherical)
        self.assertError("given twice", cov.spherical, range=1, scale=2)
        self.assertError("range/scale must be > 0", cov.spherical, range=-10)
        self.assertError("range/scale must be > 0", cov.spherical, range=[10, 0])
        self.assertError("more than 4 values", cov.spherical, range=[1, 2, 3, 4, 5])
        self.assertError("angle takes one value (2D) or three values (3D)", cov.spherical, range=1, angle=[1, 2])
        self.assertError("empty vector", cov.spherical, range=[])
        self.assertError("matern needs nu=", cov.matern, range=10)
        self.assertError("nu must be in (0, 100]", cov.matern, range=10, nu=0)
        self.assertError("nu must be in (0, 100]", cov.matern, range=10, nu=101)
        self.assertError("stable needs alpha=", cov.stable, range=10)
        self.assertError("alpha must be in (0, 2]", cov.stable, range=10, alpha=2.5)
        self.assertError("takes a single value", cov.matern, range=10, nu=[1, 2])
        self.assertError("not finite", cov.spherical, range=float("inf"))
        self.assertError("not finite", lambda: cov.spherical(range=cov.sqrt(-1)))
        self.assertError("not finite", lambda: cov.spherical(range=cov.var("L", 1) * 0 + 1 / cov.Expr("const", value=0.0)))
        self.assertError("string", cov.spherical, range="L")
        with self.assertRaises(TypeError):
            cov.spherical(range=1, foo=2)  # unknown keyword

    def test_sills_and_algebra(self):
        self.assertError("partial sill must be >= 0", lambda: -0.5 * cov.nugget())
        self.assertError("cannot be subtracted", lambda: cov.nugget() - cov.spherical(range=1))
        self.assertError("cannot be multiplied", lambda: cov.nugget() * cov.spherical(range=1))
        self.assertError("cannot be added", lambda: cov.nugget() + 1)
        self.assertError("cannot be negated", lambda: -cov.nugget())
        with self.assertRaises(TypeError):
            bool(cov.var("L", 1))  # no truth value: min()/max() builtins cannot silently misbehave

    def test_variables(self):
        a1, a2 = cov.var("L", 10), cov.var("L", 20)
        self.assertError("defined twice with different values",
                         lambda: (cov.exponential(range=a1) + cov.gaussian(range=a2)).args())
        same = cov.exponential(range=cov.var("L", 10)) + cov.gaussian(range=cov.var("L", 10.0))
        self.assertEqual(same.args(), ["-cov", str(same), "-cvar", "L", "10"])
        self.assertError("map table and as a parameter",
                         lambda: str(cov.var("C", 1) * cov.map(cov.var("C", np.ones((3, 3))))))
        self.assertError("not a number", cov.map, cov.var("C", 1.0))
        self.assertError("has no value", lambda: cov.exponential(range=cov.var("L")).args())
        # constants bound to variables are checked when the arguments are built
        self.assertError("nu must be in (0, 100]", lambda: cov.matern(range=10, nu=cov.var("nu_v", 200)).args())
        self.assertError("partial sill must be >= 0", lambda: ((cov.var("s", 0.5) - 1) * cov.nugget()).args())
        self.assertError("range/scale must be > 0", lambda: cov.exponential(range=cov.var("L", 4) - 5).args())
        # ... but spatially varying ones are left to the server
        cov.exponential(range=cov.var("L", np.full((2, 2), -1.0))).args()
        self.assertError("value must be a number", cov.var, "L", cov.var("M", 1))
        self.assertError("numeric", cov.var, "L", [object()])


class Precision(unittest.TestCase):
    def test_constants_keep_full_precision(self):
        values = [0.1, 1.2345678901234567e-7, 123456.78901234567, 2.0 / 3.0, 1e-12, 6.02214076e23]
        for v in values:
            m = cov.var("s", v) * cov.nugget()
            args = m.args()
            self.assertEqual(args[2:4], ["-cvar", "s"])
            self.assertIsInstance(args[4], str)  # the interface would print a float with 6 decimals
            self.assertEqual(float(args[4]), v)
            literal = v * cov.nugget()
            self.assertEqual(float(str(literal).split("*")[0]), v)
        # what the interface would have sent for a float: std::to_string
        self.assertNotEqual(float("%f" % 1.2345678901234567e-7), 1.2345678901234567e-7)

    def test_negative_and_numpy_constants(self):
        m = cov.spherical(range=10, angle=cov.var("theta", -30.5))
        self.assertEqual(m.args(), ["-cov", "1*spherical(range=10, angle=theta)", "-cvar", "theta=-30.5"])
        m = cov.var("s", np.float32(0.1)) * cov.nugget()
        self.assertEqual(m.args()[4], cov.format_number(float(np.float32(0.1))))
        m = cov.var("s", np.array(0.3)) * cov.nugget()  # 0-d array: a constant
        self.assertEqual(m.args()[4], "0.3")


# ------------------------------------------------------------------ C++ cross-check (covEval executable)

DATA_DIR = "/tmp/G2S/data"


def write_bgrid(name, array):
    """Writes a DataImage file (uncompressed .bgrid) that covEval can read by name."""
    a = np.ascontiguousarray(array, dtype=np.float32)
    dims = list(reversed(a.shape))  # DataImage axis 0 = last numpy index
    header = struct.pack("<I", len(dims)) + struct.pack("<%dI" % len(dims), *dims) + struct.pack("<II", 1, 0) + struct.pack("<I", 0)
    payload = header + a.tobytes()
    raw = struct.pack("<Q", 8 + len(payload)) + payload
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, name + ".bgrid"), "wb") as f:
        f.write(raw)


def read_bgrid(name):
    path = os.path.join(DATA_DIR, name + ".bgrid.gz")
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = struct.unpack_from("<I", raw, 8)[0]
    dims = struct.unpack_from("<%dI" % ndim, raw, 12)
    offset = 12 + 4 * ndim
    ntypes = struct.unpack_from("<I", raw, offset)[0]
    offset += 4 + 4 * ntypes + 4
    count = int(np.prod(dims))
    return np.frombuffer(raw, dtype="<f4", count=count, offset=offset).astype(float)


class CovEvalExecutable(unittest.TestCase):
    """Runs covEval standalone with model.args(): the C++ parses our text and prints its canonical form."""

    def setUp(self):
        if not os.access(COVEVAL, os.X_OK):
            self.skipTest("covEval not built (%s)" % COVEVAL)
        self.prefix = "test_cov_py_%d" % os.getpid()
        self.files = []

    def tearDown(self):
        for name in self.files:
            for suffix in (".bgrid", ".bgrid.gz"):
                try:
                    os.remove(os.path.join(DATA_DIR, name + suffix))
                except OSError:
                    pass

    def run_coveval(self, args, lags, pos=None, job=0):
        argv = [COVEVAL]
        for k, a in enumerate(args):
            if isinstance(a, np.ndarray):
                name = "%s_arg%d" % (self.prefix, k)
                write_bgrid(name, a)
                self.files.append(name)
                a = name
            argv.append(a)
        lag_name = self.prefix + "_lag"
        write_bgrid(lag_name, lags)
        self.files.append(lag_name)
        argv += ["-lag", lag_name]
        if pos is not None:
            write_bgrid(self.prefix + "_pos", pos)
            self.files.append(self.prefix + "_pos")
            argv += ["-pos", self.prefix + "_pos"]
        job_id = 900000 + (os.getpid() % 10000) * 10 + job
        argv += ["-id", str(job_id), "-o", self.prefix + "_out"]
        self.files += ["im_1_%d" % job_id, self.prefix + "_out"]
        done = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        log = done.stdout + done.stderr
        canonical = {}
        for line in log.splitlines():
            if line.startswith("[PARAM] cov_") and "=" in line and not line.startswith("[PARAM] cov_stationary"):
                key, value = line[len("[PARAM] "):].split("=", 1)
                canonical[int(key[4:])] = value
        values = read_bgrid("im_1_%d" % job_id) if done.returncode == 0 else None
        return done.returncode, canonical, values, log

    def test_canonical_text_matches_cpp(self):
        L = cov.var("L", 30.0)
        models = [
            0.1 * cov.nugget() + 0.6 * cov.spherical(range=[40, 20], angle=30) + 0.3 * cov.exponential(range=100),
            cov.var("nug", 0.1) * cov.nugget() + (1 - cov.var("nug", 0.1)) * cov.spherical(range=[L, L / 2], angle=cov.var("theta", -12.5)),
            cov.matern(range=cov.min(L, 25) * 2, nu=cov.var("nu_v", 1.5), angle=-cov.var("a", 3)),
            cov.stable(scale=cov.sqrt(L) + 0.1, alpha=1.5) + cov.exponential(range=1 / 3),
            (cov.var("s", 0.5) / 3) * cov.gaussian(range=[L, 2 - (-L)]),
        ]
        for k, m in enumerate(models):
            with self.subTest(model=str(m)):
                code, canonical, values, log = self.run_coveval(m.args(), np.array([[0, 0], [3, 4]], dtype=float), job=k)
                self.assertEqual(code, 0, log)
                self.assertEqual(canonical.get(0), str(m), log)

    def test_values_and_precision(self):
        # a tiny constant passed through -cvar must arrive intact (6 decimals would give 0)
        s = 1.2345678e-7
        m = cov.var("s", s) * cov.nugget() + cov.exponential(range=cov.var("L", 7.0))
        code, canonical, values, log = self.run_coveval(m.args(), np.array([0.0, 7.0]), job=7)
        self.assertEqual(code, 0, log)
        self.assertAlmostEqual(values[0] / (s + 1.0), 1.0, places=6)
        self.assertAlmostEqual(values[1], math.exp(-3.0), places=6)
        code, _, values, log = self.run_coveval((s * cov.nugget()).args(), np.array([0.0]), job=8)
        self.assertEqual(code, 0, log)
        self.assertAlmostEqual(values[0] / s, 1.0, places=6)

    def test_spatially_varying_and_regions(self):
        grid = np.arange(12, dtype=float).reshape(3, 4) + 5  # 3 rows x 4 columns
        m = cov.exponential(range=grid)
        # pos (column 2, row 1) -> grid[1, 2] = 11
        code, _, values, log = self.run_coveval(m.args(), np.array([[0, 0], [10, 0]]), pos=np.array([[2, 1]]), job=5)
        self.assertEqual(code, 0, log)
        self.assertAlmostEqual(values[1], math.exp(-3.0 * 10 / 11), places=6)
        index = np.zeros((3, 4))
        index[:, 2:] = 1
        s = cov.regions([cov.exponential(range=10), 2 * cov.gaussian(range=10)], index)
        code, canonical, values, log = self.run_coveval(s.args(), np.array([[0, 0], [5, 0]]), pos=np.array([[3, 0]]), job=6)
        self.assertEqual(code, 0, log)
        self.assertEqual([canonical.get(0), canonical.get(1)], s.texts())
        self.assertAlmostEqual(values[0], 2.0, places=6)
        self.assertAlmostEqual(values[1], 2 * math.exp(-3.0 * 0.25), places=6)


# ------------------------------------------------------------------ server (covEval algorithm)

_server_state = None


def server_available():
    global _server_state
    if _server_state is None:
        if not HAVE_INTERFACE:
            _server_state = "the compiled g2s interface is not installed"
        else:
            try:
                cov.nugget()(0.0, sa=SERVER, p=PORT, TO=5000)
                _server_state = ""
            except Exception as error:  # noqa: BLE001
                _server_state = "no G2S server at %s:%d (%s)" % (SERVER, PORT, error)
    return _server_state


class Server(unittest.TestCase):
    def setUp(self):
        reason = server_available()
        if reason:
            self.skipTest(reason)

    def test_covariance_and_variogram(self):
        m = 0.1 * cov.nugget() + 0.9 * cov.exponential(range=30)
        h = np.array([0.0, 10.0, 30.0, 100.0])
        c = m(h, sa=SERVER, p=PORT)
        g = m.variogram(h, sa=SERVER, p=PORT)
        expected = np.array([1.0] + [0.9 * math.exp(-3 * x / 30) for x in h[1:]])
        np.testing.assert_allclose(c, expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(g, 1.0 - expected, rtol=1e-6, atol=1e-7)
        self.assertEqual(c.shape, (4,))
        self.assertAlmostEqual(float(m(10, sa=SERVER, p=PORT)[0]), expected[1], places=6)

    def test_anisotropy_axis_convention(self):
        m = cov.spherical(range=[40, 20])  # 40 along columns (last numpy axis), 20 along rows
        c = m(np.array([[20, 0], [0, 20], [0, 10]]), sa=SERVER, p=PORT)
        sph = lambda t: 1 - 1.5 * t + 0.5 * t ** 3 if t < 1 else 0.0
        np.testing.assert_allclose(c, [sph(0.5), 0.0, sph(0.5)], atol=1e-6)

    def test_spatially_varying_with_pos(self):
        grid = np.arange(12, dtype=float).reshape(3, 4) + 5
        m = cov.exponential(range=cov.var("L", grid))
        c = m(np.array([[0, 0], [10, 0]]), pos=[2, 1], sa=SERVER, p=PORT)
        np.testing.assert_allclose(c, [1.0, math.exp(-3.0 * 10 / 11)], rtol=1e-6)
        with self.assertRaises(cov.ServerError):
            m(np.array([[0, 0]]), sa=SERVER, p=PORT)  # no pos for a varying model

    def test_precision_through_the_interface(self):
        s = 1.2345678e-7
        m = cov.var("s", s) * cov.nugget()
        self.assertAlmostEqual(float(m(0.0, sa=SERVER, p=PORT)[0]) / s, 1.0, places=6)
        m = cov.spherical(range=10, angle=cov.var("theta", -90)) * 1.0
        c = m(np.array([[0, 5]]), sa=SERVER, p=PORT)  # rotated by -90: range 10 along axis 1
        np.testing.assert_allclose(c, [1 - 1.5 * 0.5 + 0.5 * 0.125], atol=1e-6)

    def test_server_prints_the_same_canonical_text(self):
        from g2s import g2s
        L = cov.var("L", 30.0)
        m = cov.var("nug", 0.1) * cov.nugget() + (1 - cov.var("nug", 0.1)) * cov.matern(range=[L, L / 2], angle=-20, nu=2.5)
        result = g2s("-a", "covEval", *m.args(), "-lag", np.zeros((1, 2), dtype=np.float32), "-silent", sa=SERVER, p=PORT)
        self.assertEqual(result["covariance_model_0"], str(m))


if __name__ == "__main__":
    unittest.main(verbosity=2)
