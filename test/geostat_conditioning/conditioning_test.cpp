// Tests of the conditioning of Gaussian simulations to hard data by kriging of
// the residuals (include/geostat/conditioning.hpp), of its use in the fftma /
// spectral and tb executables, and of the flags they share with sgs and kriging.
//
//   make -C test/geostat_conditioning              # build and run
//   make -C test/geostat_conditioning timings      # 1000 x 1000, 1000 / 10000 data, R = 1 / 16
//   GEOSTAT_CONDITIONING_VERBOSE=1 make -C test/geostat_conditioning
//
// Statistical tolerances. With R realisations, at a cell of kriging estimate e
// and kriging variance v, the sample mean has standard error sqrt(v / R) and
// the sample variance (Gaussian values) sqrt(2 / (R - 1)) v; every cell is
// checked at 4.5 standard errors plus 0.02 sill (simulation error of the
// unconditional methods: FFT clipping, finite number of turning-bands lines,
// float rounding), so a correct method fails a cell with probability below
// 1e-5 and a grid of 3000 cells below 3%. Averages over regions (far from the
// data) use 4.5 standard errors estimated from the spread of the
// per-realisation values plus 0.01 sill. Exactness (hard data, agreement of
// methods, determinism) is checked bitwise or to float rounding.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "DataImage.hpp"
#include "geostat/conditioning.hpp"
#include "geostat/covarianceModel.hpp"
#include "geostat/krigingGlobal.hpp"
#include "geostat/spectralEngine.hpp"
#include "geostat/turningBands.hpp"
#include "../geostat_common/testData.hpp"

using namespace g2s::geostat;
using geostat_test::testId;
using geostat_test::testIdText;

namespace {

std::string N(const std::string& stem) { return geostat_test::testName(stem); }

int failures = 0;
int checks = 0;
std::string currentSection;
bool verbose = false;

void report(bool ok, const std::string& what, const char* file, int line) {
	++checks;
	if (ok) return;
	++failures;
	fprintf(stderr, "FAIL [%s] %s:%d: %s\n", currentSection.c_str(), file, line, what.c_str());
}
#define CHECK(condition) report(bool(condition), #condition, __FILE__, __LINE__)
#define CHECK_MSG(condition, message) report(bool(condition), std::string(#condition) + " -- " + (message), __FILE__, __LINE__)

void section(const std::string& name) { currentSection = name; printf("-- %s\n", name.c_str()); fflush(stdout); }

ResolvedModel stationary(const std::string& text, unsigned ndim) {
	return CovarianceModel(ModelSpec::parse(text), Bindings(), ndim).stationaryModel();
}

size_t cellCount(const std::vector<unsigned>& dims) {
	size_t n = 1;
	for (unsigned d : dims) n *= d;
	return n;
}

// Deterministic pseudo-random numbers (splitmix64).
struct Rng {
	uint64_t state;
	explicit Rng(uint64_t seed) : state(seed) {}
	uint64_t next() {
		uint64_t z = (state += 0x9e3779b97f4a7c15ull);
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
		z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
		return z ^ (z >> 31);
	}
	double uniform() { return double(next() >> 11) * (1.0 / 9007199254740992.0); }
	double normal() {
		const double u = std::max(uniform(), 1e-300), v = uniform();
		return std::sqrt(-2.0 * std::log(u)) * std::cos(6.283185307179586 * v);
	}
};

// `count` distinct cells of a grid of `cells` cells, drawn with `seed`, kept
// when keep(cell) is true.
template <class Keep>
std::vector<size_t> randomCells(size_t cells, size_t count, uint64_t seed, Keep keep) {
	Rng rng(seed);
	std::vector<char> used(cells, 0);
	std::vector<size_t> out;
	while (out.size() < count) {
		const size_t cell = size_t(rng.uniform() * double(cells)) % cells;
		if (used[cell] || !keep(cell)) continue;
		used[cell] = 1;
		out.push_back(cell);
	}
	return out;
}
std::vector<size_t> randomCells(size_t cells, size_t count, uint64_t seed) {
	return randomCells(cells, count, seed, [](size_t) { return true; });
}

// ============================================================ images and executables

bool writeImage(const std::string& name, const std::vector<unsigned>& dims, const std::vector<float>& values) {
	std::vector<unsigned> d = dims;
	g2s::DataImage image(unsigned(d.size()), d.data(), 1);
	for (size_t i = 0; i < values.size(); ++i) image._data[i] = values[i];
	image.write(name);
	return !g2s::DataImage::createFromFile(name).isEmpty();
}

// A -di image: NaN everywhere but at `cells`, which get `values`.
std::vector<float> dataImage(const std::vector<unsigned>& dims, const std::vector<size_t>& cells, const std::vector<double>& values) {
	std::vector<float> image(cellCount(dims), std::nanf(""));
	for (size_t i = 0; i < cells.size(); ++i) image[cells[i]] = float(values[i]);
	return image;
}

std::string binDir;
std::string exe(const std::string& name) { return binDir + "/" + name; }

bool exists(const std::string& path) {
	FILE* f = fopen(path.c_str(), "r");
	if (!f) return false;
	fclose(f);
	return true;
}

int runQuiet(const std::string& command) { return std::system((command + " >/dev/null 2>&1").c_str()); }

// Runs `name` with `args`, job id `id`; returns the exit status.
int run(const std::string& name, const std::string& args, unsigned id) {
	return runQuiet(exe(name) + " " + args + " -id " + std::to_string(id) + " -o " + N("cond_out"));
}

g2s::DataImage load(int output, unsigned id) { return g2s::DataImage::createFromFile("im_" + std::to_string(output) + "_" + std::to_string(id)); }

bool sameData(g2s::DataImage& a, g2s::DataImage& b) {
	return !a.isEmpty() && !b.isEmpty() && a._dims == b._dims && a._nbVariable == b._nbVariable &&
	       std::memcmp(a._data, b._data, sizeof(float) * a.dataSize()) == 0;
}

std::string readFile(const std::string& path) {
	std::string text;
	if (FILE* file = fopen(path.c_str(), "r")) {
		char buffer[4096];
		size_t n;
		while ((n = fread(buffer, 1, sizeof(buffer), file)) > 0) text.append(buffer, n);
		fclose(file);
	}
	return text;
}

// ============================================================ statistics

struct Estimate { double mean = 0.0, se = 0.0; };

Estimate summarise(const std::vector<double>& values) {
	Estimate e;
	for (double v : values) e.mean += v;
	e.mean /= double(values.size());
	double ss = 0.0;
	for (double v : values) ss += (v - e.mean) * (v - e.mean);
	e.se = std::sqrt(ss / double(values.size() - 1) / double(values.size()));
	return e;
}

bool statOk(const Estimate& e, double expected, double sill, std::string& text) {
	char buffer[256];
	const double tolerance = 4.5 * e.se + 0.01 * sill;
	snprintf(buffer, sizeof(buffer), "estimate %.4f (se %.4f) vs %.4f, tolerance %.4f", e.mean, e.se, expected, tolerance);
	text = buffer;
	if (verbose) printf("      %s\n", buffer);
	return std::fabs(e.mean - expected) <= tolerance && e.se <= 0.05 * sill;
}

// Per-cell mean and variance of R realisations (DataImage with R variables).
void cellMoments(const g2s::DataImage& image, std::vector<double>& mean, std::vector<double>& variance) {
	const unsigned R = image._nbVariable;
	const size_t n = cellCount(image._dims);
	mean.assign(n, 0.0);
	variance.assign(n, 0.0);
	for (size_t i = 0; i < n; ++i) {
		double s = 0.0, ss = 0.0;
		for (unsigned r = 0; r < R; ++r) s += image._data[i * R + r];
		const double m = s / R;
		for (unsigned r = 0; r < R; ++r) { const double d = image._data[i * R + r] - m; ss += d * d; }
		mean[i] = m;
		variance[i] = ss / (R - 1);
	}
}

// Conditional mean and variance of R realisations against a kriging estimate
// and variance (see the tolerances at the top). Returns the number of cells
// outside the tolerance and prints the worst standardised deviations.
void checkAgainstKriging(const std::string& label, g2s::DataImage& sims, const std::vector<double>& estimate, const std::vector<double>& kvariance,
                         const std::vector<size_t>& dataCells, double sill, bool checkVariance = true) {
	const unsigned R = sims._nbVariable;
	std::vector<double> mean, variance;
	cellMoments(sims, mean, variance);
	std::vector<char> isData(mean.size(), 0);
	for (size_t c : dataCells) isData[c] = 1;
	size_t badMean = 0, badVariance = 0;
	double worstMean = 0.0, worstVariance = 0.0, ratioSum = 0.0;
	size_t ratioCount = 0;
	for (size_t i = 0; i < mean.size(); ++i) {
		if (isData[i]) continue;
		const double v = std::max(kvariance[i], 0.0);
		const double se = std::sqrt(v / R);
		const double dm = std::fabs(mean[i] - estimate[i]);
		worstMean = std::max(worstMean, dm / (se + 1e-12));
		if (dm > 4.5 * se + 0.02 * sill) ++badMean;
		const double sv = std::sqrt(2.0 / (R - 1)) * v;
		const double dv = std::fabs(variance[i] - v);
		worstVariance = std::max(worstVariance, dv / (sv + 1e-12));
		if (dv > 4.5 * sv + 0.02 * sill) ++badVariance;
		if (v > 0.2 * sill) { ratioSum += variance[i] / v; ++ratioCount; }
	}
	const double ratio = ratioCount ? ratioSum / double(ratioCount) : 1.0;
	printf("   %-40s R %u: worst |mean - estimate| %.2f se, worst |variance - kriging variance| %.2f se, mean variance ratio %.3f\n",
	       label.c_str(), R, worstMean, worstVariance, ratio);
	CHECK_MSG(badMean == 0, label + ": " + std::to_string(badMean) + " cell(s) whose mean differs from the kriging estimate");
	if (!checkVariance) return;
	CHECK_MSG(badVariance == 0, label + ": " + std::to_string(badVariance) + " cell(s) whose variance differs from the kriging variance");
	CHECK_MSG(std::fabs(ratio - 1.0) < 0.06, label + ": mean variance ratio " + std::to_string(ratio));
}

// ============================================================ engine

std::vector<float> whiteFields(size_t cells, size_t R, uint64_t seed) {
	Rng rng(seed);
	std::vector<float> f(cells * R);
	for (float& v : f) v = float(rng.normal());
	return f;
}

double maxDifference(const std::vector<float>& a, const std::vector<float>& b) {
	double worst = 0.0;
	for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(double(a[i]) - double(b[i])));
	return worst;
}

void testEngine() {
	section("engine: exact data, direct = local with every datum, determinism, geometries");
	const std::vector<unsigned> dims = { 50, 36 };
	const size_t cells = cellCount(dims), R = 5;
	const std::vector<size_t> dataCells = randomCells(cells, 40, 3);
	std::vector<double> values(dataCells.size());
	Rng rng(9);
	for (double& v : values) v = 2.0 * rng.normal() + 1.0;
	const std::vector<float> unconditional = whiteFields(cells, R, 17);
	for (const char* text : { "1*spherical(range=[25,12], angle=30)", "0.1*nugget + 0.9*exponential(range=15)", "1*gaussian(range=10)" }) {
		const ResolvedModel model = stationary(text, 2);
		for (KrigingType type : { KrigingType::Simple, KrigingType::Ordinary }) {
			for (bool periodic : { false, true }) {
				const std::string label = std::string(text) + (type == KrigingType::Simple ? " sk" : " ok") + (periodic ? " periodic" : "");
				// Realisation-major layout {1, cells}.
				std::vector<std::vector<float> > results;
				for (ConditioningMethod method : { ConditioningMethod::Direct, ConditioningMethod::Local }) {
					ConditioningOptions options;
					options.type = type;
					options.method = method;
					options.maxNeighbors = 100; // every datum
					options.nbThreads = 4;
					Conditioner conditioner;
					std::string error;
					CHECK_MSG(conditioner.prepare(model, ConditioningGeometry::regularGrid(dims, periodic), dataCells.data(), values.data(), dataCells.size(), options, error), error);
					std::vector<float> f = unconditional;
					conditioner.condition({ f.data(), 1, cells }, R);
					bool exact = true;
					for (size_t r = 0; r < R; ++r)
						for (size_t i = 0; i < dataCells.size(); ++i) exact &= f[r * cells + dataCells[i]] == float(values[i]);
					CHECK_MSG(exact, label + ": hard data");
					// Thread count and batch: one realisation alone, one thread.
					ConditioningOptions single = options;
					single.nbThreads = 1;
					Conditioner one;
					CHECK(one.prepare(model, ConditioningGeometry::regularGrid(dims, periodic), dataCells.data(), values.data(), dataCells.size(), single, error));
					std::vector<float> third(unconditional.begin() + 2 * cells, unconditional.begin() + 3 * cells);
					one.condition({ third.data(), 1, 0 }, 1);
					CHECK_MSG(std::equal(third.begin(), third.end(), f.begin() + 2 * cells), label + " " + conditioningMethodName(method) + ": realisation 2 alone, 1 thread");
					results.push_back(f);
				}
				const double difference = maxDifference(results[0], results[1]);
				if (verbose) printf("      %s: direct vs local %.2e\n", label.c_str(), difference);
				CHECK_MSG(difference < 2e-4, label + ": direct and local (all data) differ by " + std::to_string(difference));
			}
		}
		// Points geometry (no table): same as the grid.
		{
			std::vector<double> coordinates(cells * 2);
			for (size_t c = 0; c < cells; ++c) { coordinates[2 * c] = double(c % dims[0]); coordinates[2 * c + 1] = double(c / dims[0]); }
			ConditioningOptions options;
			options.method = ConditioningMethod::Direct;
			options.nbThreads = 3;
			Conditioner grid, points;
			std::string error;
			CHECK(grid.prepare(model, ConditioningGeometry::regularGrid(dims, false), dataCells.data(), values.data(), dataCells.size(), options, error));
			CHECK(points.prepare(model, ConditioningGeometry::points(coordinates.data(), cells, 2), dataCells.data(), values.data(), dataCells.size(), options, error));
			CHECK(grid.report().covarianceTable && !points.report().covarianceTable);
			std::vector<float> a = unconditional, b = unconditional;
			grid.condition({ a.data(), R, 1 }, R); // interleaved layout
			points.condition({ b.data(), R, 1 }, R);
			CHECK_MSG(maxDifference(a, b) < 1e-5, std::string(text) + ": grid and points geometries");
		}
	}
	// Local with fewer neighbours than data: the weights are shared by every
	// realisation, so conditioning R realisations at once = one at a time.
	{
		const ResolvedModel model = stationary("1*spherical(range=20)", 2);
		ConditioningOptions options;
		options.method = ConditioningMethod::Local;
		options.maxNeighbors = 8;
		options.maxPerSector = 3;
		options.nbThreads = 6;
		Conditioner all;
		std::string error;
		CHECK(all.prepare(model, ConditioningGeometry::regularGrid(dims, true), dataCells.data(), values.data(), dataCells.size(), options, error));
		std::vector<float> f = unconditional;
		all.condition({ f.data(), R, 1 }, R);
		bool same = true;
		for (size_t r = 0; r < R; ++r) {
			std::vector<float> g(cells);
			for (size_t c = 0; c < cells; ++c) g[c] = unconditional[c * R + r];
			all.condition({ g.data(), 1, 0 }, 1);
			for (size_t c = 0; c < cells; ++c) same &= g[c] == f[c * R + r];
		}
		CHECK_MSG(same, "local, periodic, sectors: batch = one realisation at a time");
		CHECK(all.report().local.solved > 0);
	}
	// No data: inactive, the fields are left as they are.
	{
		Conditioner none;
		std::string error;
		CHECK(none.prepare(stationary("1*spherical(range=20)", 2), ConditioningGeometry::regularGrid(dims, false), nullptr, nullptr, 0, ConditioningOptions(), error));
		CHECK(!none.active());
		std::vector<float> f = unconditional;
		none.condition({ f.data(), R, 1 }, R);
		CHECK(f == unconditional);
	}
	// Errors.
	{
		Conditioner c;
		std::string error;
		const size_t twice[2] = { 3, 3 };
		const double v[2] = { 1, 2 };
		CHECK(!c.prepare(stationary("1*spherical(range=20)", 2), ConditioningGeometry::regularGrid(dims, false), twice, v, 2, ConditioningOptions(), error));
		CHECK(error.find("two data") != std::string::npos);
		ConditioningOptions small;
		small.method = ConditioningMethod::Direct;
		small.maxMatrixBytes = 100;
		CHECK(!c.prepare(stationary("1*spherical(range=20)", 2), ConditioningGeometry::regularGrid(dims, false), dataCells.data(), values.data(), dataCells.size(), small, error));
		small.method = ConditioningMethod::Auto;
		CHECK(c.prepare(stationary("1*spherical(range=20)", 2), ConditioningGeometry::regularGrid(dims, false), dataCells.data(), values.data(), dataCells.size(), small, error));
		CHECK(c.report().method == ConditioningMethod::Local);
	}
	// The periodic covariance of the kriging is the circulant covariance the
	// FFT engine samples for -csim (even sizes: Nyquist lags, rotated anisotropy),
	// at every offset and its periodic shifts.
	for (const std::vector<unsigned>& grid : { std::vector<unsigned>{ 12, 10 }, std::vector<unsigned>{ 9, 8, 6 } }) {
		const unsigned nd = unsigned(grid.size());
		const ResolvedModel model = stationary(nd == 2 ? "0.2*nugget + 0.8*exponential(range=[9,4], angle=35)" : "1*gaussian(range=[7,4,3], angle=[20,30,40])", nd);
		spectral::GridLayout layout(grid, grid);
		std::vector<double> sampled(layout.realCount);
		spectral::sampleCovariance(model, layout, sampled.data(), 2);
		double period[3] = { 0, 0, 0 };
		for (unsigned d = 0; d < nd; ++d) period[d] = grid[d];
		double worst = 0.0;
		for (size_t cell = 0; cell < cellCount(grid); ++cell) {
			size_t rest = cell, row = 0, stride = 1;
			double lag[3] = { 0, 0, 0 };
			for (unsigned d = 0; d < nd; ++d) { lag[d] = double(rest % grid[d]); rest /= grid[d]; }
			for (unsigned d = nd; d-- > 1;) row = row * grid[d] + size_t(lag[d]);
			(void)stride;
			const double expected = sampled[row * layout.rowPitch + size_t(lag[0])];
			for (int shift = -2; shift <= 1; ++shift) {
				double shifted[3];
				for (unsigned d = 0; d < nd; ++d) shifted[d] = lag[d] + shift * period[d];
				worst = std::max(worst, std::fabs(periodicCovariance(model, shifted, period, nd) - expected));
				for (unsigned d = 0; d < nd; ++d) shifted[d] = -shifted[d];
				worst = std::max(worst, std::fabs(periodicCovariance(model, shifted, period, nd) - expected));
			}
		}
		CHECK_MSG(worst < 1e-12, std::to_string(nd) + "D: periodicCovariance vs sampleCovariance " + std::to_string(worst));
	}

	std::string reason;
	CHECK(chooseConditioningMethod(10000, size_t(2) << 30, reason) == ConditioningMethod::Direct);
	CHECK(chooseConditioningMethod(10001, size_t(2) << 30, reason) == ConditioningMethod::Local);
}

// ============================================================ executables

const char* kSimulators[] = { "fftma", "spectral", "tb" };

void testHardData() {
	section("executables: hard data reproduced exactly (direct and local, sk and ok)");
	const std::vector<unsigned> dims = { 70, 45 };
	const std::vector<size_t> cells = randomCells(cellCount(dims), 60, 21);
	std::vector<double> values(cells.size());
	Rng rng(5);
	for (double& v : values) v = 3.0 * rng.normal() - 2.0;
	CHECK(writeImage(N("cond_hard"), dims, dataImage(dims, cells, values)));
	unsigned id = testId(100);
	for (const char* name : kSimulators) {
		for (const char* options : { "-km direct", "-km local -n 12", "-km local -n 12 -os 3 -ks 41 -kt ok", "-kt ok", "-km direct -m 4" }) {
			const std::string args = std::string("-di ") + N("cond_hard") + " -cov '0.2*nugget + 0.8*spherical(range=[30,15], angle=20)' -nr 3 -s 4 -j 4 " + options;
			CHECK_MSG(run(name, args, id) == 0, std::string(name) + " " + options);
			g2s::DataImage out = load(1, id);
			bool exact = !out.isEmpty() && out._nbVariable == 3;
			for (size_t i = 0; exact && i < cells.size(); ++i)
				for (unsigned r = 0; r < 3; ++r) exact &= out._data[cells[i] * 3 + r] == float(values[i]);
			CHECK_MSG(exact, std::string(name) + " " + options + ": hard data");
			++id;
		}
	}
	// -csim (fftma / spectral): periodic distances, data honoured too.
	for (const char* options : { "-csim -km direct", "-csim -km local -n 10" }) {
		CHECK(run("fftma", "-di " + N("cond_hard") + " -cov 'spherical(range=[30,15], angle=20)' -nr 2 -s 4 " + options, id) == 0);
		g2s::DataImage out = load(1, id);
		bool exact = !out.isEmpty();
		for (size_t i = 0; exact && i < cells.size(); ++i) exact &= out._data[cells[i] * 2] == float(values[i]) && out._data[cells[i] * 2 + 1] == float(values[i]);
		CHECK_MSG(exact, std::string("fftma ") + options + ": hard data");
		++id;
	}
}

// Kriging estimate and variance of the kriging executable on the -di image `di`.
bool krigingReference(const std::string& di, const std::string& model, const std::string& options, unsigned id, std::vector<double>& estimate, std::vector<double>& variance) {
	if (run("kriging", "-di " + di + " -cov '" + model + "' -var " + options, id) != 0) return false;
	g2s::DataImage e = load(1, id), v = load(2, id);
	if (e.isEmpty() || v.isEmpty()) return false;
	estimate.assign(e._data, e._data + e.dataSize());
	variance.assign(v._data, v._data + v.dataSize());
	return true;
}

void testConditionalMoments() {
	section("executables: conditional mean and variance = kriging estimate and variance");
	const std::vector<unsigned> dims = { 64, 48 };
	const std::vector<size_t> cells = randomCells(cellCount(dims), 30, 31);
	std::vector<double> values(cells.size());
	Rng rng(8);
	for (double& v : values) v = 1.0 + 1.2 * rng.normal();
	CHECK(writeImage(N("cond_moments"), dims, dataImage(dims, cells, values)));
	const std::string model = "1*spherical(range=[34,20], angle=30)";
	const unsigned R = 400;
	std::vector<double> directEstimate, directVariance, localEstimate, localVariance, okEstimate, okVariance;
	CHECK(krigingReference(N("cond_moments"), model, "-kt sk -m 1 -method direct", testId(200), directEstimate, directVariance));
	CHECK(krigingReference(N("cond_moments"), model, "-kt sk -m 1 -method local -n 8", testId(201), localEstimate, localVariance));
	CHECK(krigingReference(N("cond_moments"), model, "-kt ok -method direct", testId(202), okEstimate, okVariance));
	if (directEstimate.empty() || localEstimate.empty() || okEstimate.empty()) return;
	unsigned id = testId(210);
	for (const char* name : kSimulators) {
		// Direct: the SK estimate and variance.
		CHECK(run(name, "-di " + N("cond_moments") + " -cov '" + model + "' -m 1 -nr " + std::to_string(R) + " -s 7 -j 8 -km direct", id) == 0);
		{ g2s::DataImage sims = load(1, id); if (!sims.isEmpty()) checkAgainstKriging(std::string(name) + " sk direct", sims, directEstimate, directVariance, cells, 1.0); }
		++id;
		// Local: with fixed neighbours the weights solve the local system, so
		// the conditional variance is the local SK variance (c0 - w.k).
		CHECK(run(name, "-di " + N("cond_moments") + " -cov '" + model + "' -m 1 -nr " + std::to_string(R) + " -s 8 -j 8 -km local -n 8", id) == 0);
		{ g2s::DataImage sims = load(1, id); if (!sims.isEmpty()) checkAgainstKriging(std::string(name) + " sk local -n 8", sims, localEstimate, localVariance, cells, 1.0); }
		++id;
	}
	// OK, direct: mean = OK estimate, variance = OK variance (the mean of the
	// simulation, -m 5 here, plays no role).
	CHECK(run("fftma", "-di " + N("cond_moments") + " -cov '" + model + "' -m 5 -nr " + std::to_string(R) + " -s 9 -j 8 -km direct -kt ok", id) == 0);
	{ g2s::DataImage sims = load(1, id); if (!sims.isEmpty()) checkAgainstKriging("fftma ok direct", sims, okEstimate, okVariance, cells, 1.0); }
}

void testFarFromData() {
	section("executables: far from the data, the unconditional mean, variance and covariance");
	const std::vector<unsigned> dims = { 160, 60 };
	const size_t cells = cellCount(dims);
	// Data in x < 20 only; the model reaches 25 cells: the region x >= 60 is independent of them.
	const std::vector<size_t> dataCells = randomCells(cells, 25, 41, [&](size_t c) { return c % dims[0] < 20; });
	std::vector<double> values(dataCells.size(), 4.0); // far from the mean on purpose
	CHECK(writeImage(N("cond_far"), dims, dataImage(dims, dataCells, values)));
	const std::string model = "1.5*spherical(range=25)";
	const ResolvedModel resolved = stationary(model, 2);
	const double mean = -1.0, sill = 1.5;
	const unsigned R = 120;
	unsigned id = testId(300);
	for (const char* name : kSimulators) {
		for (const char* options : { "-km direct", "-km local -n 20" }) {
			CHECK(run(name, "-di " + N("cond_far") + " -cov '" + model + "' -m -1 -nr " + std::to_string(R) + " -s 12 -j 8 " + options, id) == 0);
			g2s::DataImage sims = load(1, id);
			++id;
			if (sims.isEmpty()) continue;
			std::vector<double> means(R), variances(R), covX(R), covY(R), nearMeans(R);
			for (unsigned r = 0; r < R; ++r) {
				double s = 0, ss = 0, cx = 0, cy = 0, near = 0;
				size_t n = 0, nx = 0, ny = 0, nn = 0;
				for (unsigned y = 0; y < dims[1]; ++y)
					for (unsigned x = 0; x < dims[0]; ++x) {
						const double v = sims._data[(y * dims[0] + x) * R + r] - mean;
						if (x < 15) { near += v; ++nn; }
						if (x < 60) continue;
						s += v; ss += v * v; ++n;
						if (x + 6 < dims[0]) { cx += v * (sims._data[(y * dims[0] + x + 6) * R + r] - mean); ++nx; }
						if (y + 6 < dims[1]) { cy += v * (sims._data[((y + 6) * dims[0] + x) * R + r] - mean); ++ny; }
					}
				means[r] = s / n; variances[r] = ss / n; covX[r] = cx / nx; covY[r] = cy / ny; nearMeans[r] = near / nn;
			}
			const double h[2] = { 6, 0 };
			const std::string label = std::string(name) + " " + options;
			std::string text;
			CHECK_MSG(statOk(summarise(means), 0.0, sill, text), label + " mean: " + text);
			CHECK_MSG(statOk(summarise(variances), sill, sill, text), label + " variance: " + text);
			CHECK_MSG(statOk(summarise(covX), resolved.covariance(h), sill, text), label + " C(6,0): " + text);
			CHECK_MSG(statOk(summarise(covY), resolved.covariance(h), sill, text), label + " C(0,6): " + text);
			// Near the data the field is pulled towards them (sanity of the test itself).
			CHECK_MSG(summarise(nearMeans).mean > 2.0, label + ": the data region is not conditioned");
		}
	}
}

void testMethodsAgree() {
	section("executables: direct = local when the neighbourhood holds every datum");
	const std::vector<unsigned> dims = { 60, 50 };
	const std::vector<size_t> cells = randomCells(cellCount(dims), 35, 51);
	std::vector<double> values(cells.size());
	Rng rng(2);
	for (double& v : values) v = rng.normal();
	CHECK(writeImage(N("cond_agree"), dims, dataImage(dims, cells, values)));
	unsigned id = testId(400);
	const std::string model = "0.1*nugget + 0.9*exponential(range=[30,15], angle=-20)";
	for (const char* name : kSimulators) {
		for (const char* extra : { "", "-kt ok", "-csim" }) {
			if (std::string(extra) == "-csim" && std::string(name) == "tb") continue;
			const std::string common = "-di " + N("cond_agree") + " -cov '" + model + "' -nr 3 -s 3 -j 4 " + extra;
			CHECK(run(name, common + " -km direct", id) == 0);
			CHECK(run(name, common + " -km local -n 40", id + 1) == 0);
			g2s::DataImage a = load(1, id), b = load(1, id + 1);
			id += 2;
			double worst = a.isEmpty() || b.isEmpty() ? 1e9 : 0.0;
			for (size_t i = 0; worst < 1e9 && i < a.dataSize(); ++i) worst = std::max(worst, std::fabs(double(a._data[i]) - double(b._data[i])));
			if (verbose) printf("      %s %s: %.2e\n", name, extra, worst);
			CHECK_MSG(worst < 2e-4, std::string(name) + " " + extra + ": direct vs local " + std::to_string(worst));
		}
	}
}

void testDeterminism() {
	section("executables: realisation r of a batch = -ri r alone; any thread count");
	const std::vector<unsigned> dims = { 55, 40 };
	const std::vector<size_t> cells = randomCells(cellCount(dims), 25, 61);
	std::vector<double> values(cells.size());
	Rng rng(6);
	for (double& v : values) v = rng.normal();
	CHECK(writeImage(N("cond_det"), dims, dataImage(dims, cells, values)));
	CHECK(writeImage(N("cond_det_nan"), dims, std::vector<float>(cellCount(dims), std::nanf(""))));
	unsigned id = testId(500);
	for (const char* name : kSimulators) {
		for (const char* options : { "-km direct", "-km local -n 10", "-km local -n 10 -kt ok", "" }) {
			const std::string di = std::string(options).empty() ? N("cond_det_nan") : N("cond_det");
			const std::string common = "-di " + di + " -cov '0.1*nugget + 0.9*spherical(range=[25,15], angle=40)' -s 21 " + options;
			CHECK(run(name, common + " -nr 4 -j 1", id) == 0);
			CHECK(run(name, common + " -nr 4 -j 7", id + 1) == 0);
			CHECK(run(name, common + " -nr 1 -ri 2 -j 3", id + 2) == 0);
			CHECK(run(name, common + " -nr 2 -ri 1 -j 0.5", id + 3) == 0);
			g2s::DataImage batch = load(1, id), threads = load(1, id + 1), single = load(1, id + 2), pair = load(1, id + 3);
			id += 4;
			const std::string label = std::string(name) + " " + (std::string(options).empty() ? "unconditional" : options);
			CHECK_MSG(sameData(batch, threads), label + ": -j 1 vs -j 7");
			bool same = !batch.isEmpty() && !single.isEmpty() && !pair.isEmpty() && single._nbVariable == 1 && pair._nbVariable == 2;
			for (size_t c = 0; same && c < cellCount(dims); ++c)
				same = single._data[c] == batch._data[c * 4 + 2] && pair._data[c * 2] == batch._data[c * 4 + 1] && pair._data[c * 2 + 1] == batch._data[c * 4 + 2];
			CHECK_MSG(same, label + ": -ri");
		}
	}
}

// All-NaN -di: the unconditional simulation, the engine's values bit for bit.
void testUnconditional() {
	section("executables: an all-NaN -di gives the unconditional simulation");
	const std::vector<unsigned> dims = { 48, 40 };
	const size_t cells = cellCount(dims);
	CHECK(writeImage(N("cond_nan"), dims, std::vector<float>(cells, std::nanf(""))));
	const std::string model = "0.1*nugget + 0.9*spherical(range=[30,12], angle=20)";
	const ResolvedModel resolved = stationary(model, 2);
	const unsigned R = 3;
	for (spectral::Method method : { spectral::Method::FftMovingAverage, spectral::Method::Spectral }) {
		const unsigned id = testId(method == spectral::Method::Spectral ? 601 : 600);
		CHECK(run(spectral::methodName(method), "-di " + N("cond_nan") + " -cov '" + model + "' -nr 3 -s 99 -m 0.5 -j 4", id) == 0);
		g2s::DataImage out = load(1, id);
		std::vector<double> range = resolved.practicalRange();
		for (double& r : range) r = std::ceil(r);
		spectral::SpectralSimulator simulator(resolved, dims, spectral::extendedGrid(dims, range, false), method, 2);
		std::vector<float> reference(cells * R);
		simulator.simulate(99, 0, R, 0.5, { reference.data(), R, 1 });
		CHECK_MSG(!out.isEmpty() && std::equal(reference.begin(), reference.end(), out._data), std::string(spectral::methodName(method)) + ": engine values bit for bit");
	}
	{
		const unsigned id = testId(602);
		CHECK(run("tb", "-di " + N("cond_nan") + " -cov '" + model + "' -nr 3 -ri 5 -s 99 -m 0.5 -j 4", id) == 0);
		g2s::DataImage out = load(1, id);
		std::vector<double> points(cells * 2);
		for (size_t c = 0; c < cells; ++c) { points[2 * c] = double(c % dims[0]); points[2 * c + 1] = double(c / dims[0]); }
		TurningBandsSettings settings;
		settings.seed = 99;
		settings.mean = 0.5;
		std::vector<double> values(R * cells);
		simulateTurningBands(resolved, settings, points.data(), cells, 5, R, values.data());
		double worst = out.isEmpty() ? 1e9 : 0.0;
		for (unsigned r = 0; worst < 1e9 && r < R; ++r)
			for (size_t c = 0; c < cells; ++c) worst = std::max(worst, std::fabs(double(out._data[c * R + r]) - values[r * cells + c]));
		CHECK_MSG(worst <= 1e-5, "tb -ri 5: engine values, max difference " + std::to_string(worst));
	}
}

// -csim: the conditioning uses the periodic distances of the simulated field.
void testPeriodic() {
	section("executables: -csim conditions with periodic distances");
	const std::vector<unsigned> dims = { 64, 40 };
	const size_t cells = cellCount(dims);
	// Data along the left edge only: on the torus they also inform the right edge.
	const std::vector<size_t> dataCells = randomCells(cells, 16, 71, [&](size_t c) { return c % dims[0] < 4; });
	std::vector<double> values(dataCells.size());
	Rng rng(4);
	for (double& v : values) v = 2.0 + 0.5 * rng.normal();
	CHECK(writeImage(N("cond_periodic"), dims, dataImage(dims, dataCells, values)));
	const std::string model = "1*spherical(range=24)";
	const ResolvedModel resolved = stationary(model, 2);
	// Reference: periodic SK (mean 0) with the engine.
	std::vector<double> coordinates(dataCells.size() * 2);
	for (size_t i = 0; i < dataCells.size(); ++i) { coordinates[2 * i] = double(dataCells[i] % dims[0]); coordinates[2 * i + 1] = double(dataCells[i] / dims[0]); }
	GlobalKriging periodic, flat;
	GlobalKriging::Options options;
	options.type = KrigingType::Simple;
	std::string error;
	CHECK(flat.factorize(resolved, coordinates.data(), dataCells.size(), 2, options, error));
	options.period = { double(dims[0]), double(dims[1]) };
	CHECK(periodic.factorize(resolved, coordinates.data(), dataCells.size(), 2, options, error));
	std::vector<size_t> all(cells);
	for (size_t c = 0; c < cells; ++c) all[c] = c;
	GlobalKrigingDual dual;
	std::vector<double> estimate(cells), variance(cells), flatEstimate(cells);
	periodic.solve(values.data(), 0.0, dual);
	periodic.estimate(CellCrossCovariance(periodic, dims, all.data()), cells, dual, estimate.data(), cells);
	periodic.variance(CellCrossCovariance(periodic, dims, all.data()), cells, variance.data());
	flat.solve(values.data(), 0.0, dual);
	flat.estimate(CellCrossCovariance(flat, dims, all.data()), cells, dual, flatEstimate.data(), cells);
	// Same through the covariance table.
	{
		GridCovarianceTable table;
		CHECK(table.build(resolved, dims, 2, size_t(1) << 30, true));
		periodic.solve(values.data(), 0.0, dual);
		std::vector<double> viaTable(cells);
		estimateGridFields(table, dataCells.data(), dual, viaTable.data(), cells, 3);
		double worst = 0.0;
		for (size_t c = 0; c < cells; ++c) worst = std::max(worst, std::fabs(viaTable[c] - estimate[c]));
		CHECK_MSG(worst < 1e-9, "periodic table vs evaluated model: " + std::to_string(worst));
	}
	double rightEdge = 0.0, rightEdgeFlat = 0.0;
	for (unsigned y = 0; y < dims[1]; ++y) { rightEdge += estimate[y * dims[0] + dims[0] - 1]; rightEdgeFlat += flatEstimate[y * dims[0] + dims[0] - 1]; }
	printf("   right edge: periodic SK estimate %.3f, non-periodic %.3f\n", rightEdge / dims[1], rightEdgeFlat / dims[1]);
	CHECK(rightEdge / dims[1] > rightEdgeFlat / dims[1] + 0.5); // the test can tell them apart
	const unsigned R = 400;
	unsigned id = testId(700);
	for (const char* name : { "fftma", "spectral" }) {
		for (const char* options : { "-km direct", "-km local -n 30" }) {
			CHECK(run(name, "-di " + N("cond_periodic") + " -cov '" + model + "' -csim -nr " + std::to_string(R) + " -s 5 -j 8 " + options, id) == 0);
			g2s::DataImage sims = load(1, id);
			++id;
			// Local with 30 of 16 data: every datum, so the same moments as direct.
			if (!sims.isEmpty()) checkAgainstKriging(std::string(name) + " -csim " + options, sims, estimate, variance, dataCells, 1.0);
		}
	}
	// tb refuses -csim.
	CHECK(run("tb", "-di " + N("cond_periodic") + " -cov '" + model + "' -csim", id) != 0);
}

// Flags shared by sgs, kriging, fftma and tb.
void testFlags() {
	section("flags: -oi, -n default, -ri, -j fraction, help texts, errors");
	const std::vector<unsigned> dims = { 40, 30 };
	const std::vector<size_t> cells = randomCells(cellCount(dims), 70, 81);
	std::vector<double> values(cells.size());
	Rng rng(1);
	for (double& v : values) v = rng.normal();
	CHECK(writeImage(N("cond_flags"), dims, dataImage(dims, cells, values)));
	const std::string common = "-di " + N("cond_flags") + " -cov 'spherical(range=12)' -s 5";
	// Second outputs: -oi (default id_<id>) and im_2_<id>, as qs, pyDev and sgs.
	CHECK(run("kriging", common + " -var -oi " + N("cond_flags_var"), testId(800)) == 0);
	{ g2s::DataImage a = load(2, testId(800)), b = g2s::DataImage::createFromFile(N("cond_flags_var")); CHECK(sameData(a, b)); }
	CHECK(run("kriging", common + " -var", testId(801)) == 0);
	{ g2s::DataImage a = load(2, testId(801)), b = g2s::DataImage::createFromFile("id_" + testIdText(801)); CHECK(sameData(a, b)); }
	CHECK(run("fftma", common + " -on -oi " + N("cond_flags_noise"), testId(802)) == 0);
	{ g2s::DataImage a = load(2, testId(802)), b = g2s::DataImage::createFromFile(N("cond_flags_noise")); CHECK(sameData(a, b)); }
	CHECK(run("fftma", common + " -on", testId(803)) == 0);
	{ g2s::DataImage a = load(2, testId(803)), b = g2s::DataImage::createFromFile("id_" + testIdText(803)); CHECK(sameData(a, b)); }
	// The conditioning keeps the unconditional noise of -on.
	CHECK(writeImage(N("cond_flags_nan"), dims, std::vector<float>(cellCount(dims), std::nanf(""))));
	CHECK(run("fftma", "-di " + N("cond_flags_nan") + " -cov 'spherical(range=12)' -s 5 -on", testId(804)) == 0);
	{ g2s::DataImage a = load(2, testId(803)), b = load(2, testId(804)); CHECK(sameData(a, b)); }
	// -n defaults to 50 (sgs, kriging, conditioning).
	for (const char* name : { "sgs", "kriging", "fftma", "tb" }) {
		const std::string extra = std::string(name) == "kriging" ? " -method local" : std::string(name) == "sgs" ? "" : " -km local";
		CHECK(run(name, common + extra, testId(810)) == 0);
		CHECK(run(name, common + extra + " -n 50", testId(811)) == 0);
		CHECK(run(name, common + extra + " -n 49", testId(812)) == 0);
		g2s::DataImage a = load(1, testId(810)), b = load(1, testId(811)), c = load(1, testId(812));
		CHECK_MSG(sameData(a, b) && !sameData(a, c), std::string(name) + ": -n defaults to 50");
	}
	// Help texts: -j fraction everywhere, -ri for the simulations.
	for (const char* name : { "sgs", "kriging", "fftma", "tb" }) {
		const std::string log = N("cond_help.txt");
		std::system((exe(name) + " -h > " + log + " 2>&1").c_str());
		const std::string text = readFile(log);
		CHECK_MSG(text.find("-j") != std::string::npos && text.find("fraction = share of the cores") != std::string::npos, std::string(name) + " -h: -j fraction");
		if (std::string(name) != "kriging") CHECK_MSG(text.find("-ri") != std::string::npos, std::string(name) + " -h: -ri");
		if (std::string(name) == "fftma" || std::string(name) == "tb")
			CHECK_MSG(text.find("-km") != std::string::npos && text.find("-kt") != std::string::npos && text.find("default 50") != std::string::npos, std::string(name) + " -h: conditioning");
	}
	// Errors, with the same wording everywhere.
	auto refused = [&](const std::string& name, const std::string& arguments, const std::string& fragment) {
		const std::string log = N("cond_log.txt");
		const bool failed = std::system((exe(name) + " " + arguments + " -o " + N("cond_out") + " > " + log + " 2>&1").c_str()) != 0;
		const bool explained = readFile(log).find(fragment) != std::string::npos;
		report(failed && explained, name + " " + arguments + " should fail with '" + fragment + "'", __FILE__, __LINE__);
	};
	for (const char* name : { "fftma", "tb" }) {
		refused(name, common + " -km nearest -id " + testIdText(820), "-km expects direct, local or auto");
		refused(name, common + " -kt uk -id " + testIdText(821), "-kt expects sk or ok");
		refused(name, common + " -n 0 -id " + testIdText(822), "-n expects a positive integer");
		refused(name, common + " -km direct -maxmem 0.0000001 -id " + testIdText(823), "use the moving neighbourhood");
		refused(name, common + " -ri -3 -id " + testIdText(824), "-ri expects an integer >= 0");
		refused(name, "-cov 'spherical(range=12)' -id " + testIdText(825), "-di is mandatory");
	}
	refused("kriging", "-cov 'spherical(range=12)' -id " + testIdText(826), "-di is mandatory");
	refused("sgs", "-cov 'spherical(range=12)' -id " + testIdText(827), "-di is mandatory");
	refused("kriging", common + " -kt uk -id " + testIdText(828), "-kt expects sk or ok");
	refused("sgs", common + " -kt uk -id " + testIdText(829), "-kt expects sk or ok");
	// Several variables with data: refused (one variable holds the hard data).
	{
		std::vector<unsigned> d = dims;
		g2s::DataImage two(2, d.data(), 2);
		for (size_t i = 0; i < two.dataSize(); ++i) two._data[i] = i % 97 == 0 ? 1.f : std::nanf("");
		two.write(N("cond_two"));
		refused("fftma", "-di " + N("cond_two") + " -cov 'spherical(range=12)' -id " + testIdText(830), "single variable");
	}
}

// ============================================================ timings

void timings(unsigned threads) {
	const std::vector<unsigned> dims = { 1000, 1000 };
	const size_t cells = cellCount(dims);
	const std::string model = "1*spherical(range=50)";
	CHECK(writeImage(N("cond_t_nan"), dims, std::vector<float>(cells, std::nanf(""))));
	for (size_t count : { size_t(1000), size_t(10000) }) {
		const std::vector<size_t> dataCells = randomCells(cells, count, 1234 + count);
		std::vector<double> values(count);
		Rng rng(count);
		for (double& v : values) v = rng.normal();
		CHECK(writeImage(N("cond_t_" + std::to_string(count)), dims, dataImage(dims, dataCells, values)));
	}
	printf("%-8s %-6s %-8s %3s  %9s  %9s  %9s  %s\n", "method", "data", "kriging", "R", "total s", "uncond s", "cond s", "(conditioning: prepare + condition, from the report)");
	unsigned id = testId(900);
	for (const char* name : kSimulators) {
		for (unsigned R : { 1u, 16u }) {
			const std::string common = " -cov '" + model + "' -nr " + std::to_string(R) + " -s 1 -j " + std::to_string(threads);
			auto timed = [&](const std::string& args, std::string& log) {
				log = N("cond_t_log.txt");
				const auto t0 = std::chrono::steady_clock::now();
				const int status = std::system((exe(name) + " " + args + " -id " + std::to_string(id++) + " -o " + N("cond_out") + " > " + log + " 2>&1").c_str());
				const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
				CHECK(status == 0);
				log = readFile(log);
				return seconds;
			};
			std::string log;
			const double unconditional = timed("-di " + N("cond_t_nan") + common, log);
			for (size_t count : { size_t(1000), size_t(10000) }) {
				for (const char* method : { "direct", "local" }) {
					const double total = timed("-di " + N("cond_t_" + std::to_string(count)) + common + " -km " + method, log);
					double factorisation = 0.0, conditioning = 0.0;
					size_t at = log.find("factorisation of the");
					if (at != std::string::npos) sscanf(log.c_str() + log.find(':', at) + 1, "%lf", &factorisation);
					at = log.find("conditioning (");
					if (at != std::string::npos) sscanf(log.c_str() + log.find("):", at) + 2, "%lf", &conditioning);
					printf("%-8s %-6zu %-8s %3u  %9.2f  %9.2f  %9.2f  (factorisation %.2f s)\n", name, count, method, R, total, unconditional,
					       (factorisation + conditioning) / 1000.0, factorisation / 1000.0);
					fflush(stdout);
				}
			}
		}
	}
}

} // namespace

int main(int argc, char** argv) {
	binDir = argc > 1 ? argv[1] : "../../build/c++-build";
	verbose = getenv("GEOSTAT_CONDITIONING_VERBOSE") != nullptr;
	if (argc > 2 && std::string(argv[2]) == "timings") {
		timings(argc > 3 ? unsigned(atoi(argv[3])) : 8u);
		return failures == 0 ? 0 : 1;
	}
	const auto begin = std::chrono::steady_clock::now();
	testEngine();
	bool built = true;
	for (const char* name : { "fftma", "spectral", "tb", "kriging", "sgs" }) built &= exists(exe(name));
	if (built) {
		testHardData();
		testUnconditional();
		testDeterminism();
		testMethodsAgree();
		testConditionalMoments();
		testFarFromData();
		testPeriodic();
		testFlags();
	} else {
		printf("-- executables skipped: fftma, spectral, tb, kriging or sgs not built in %s\n", binDir.c_str());
	}
	const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
	printf("%d checks, %d failures (%.1f s)\n", checks, failures, seconds);
	return failures == 0 ? 0 : 1;
}
