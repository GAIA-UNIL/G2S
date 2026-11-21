// Tests of the turning-bands engine (include/geostat/turningBands.hpp) and of
// the tb executable.
//
//   make -C test/geostat_tb             # build and run (tb tested when build/c++-build/tb exists)
//   make -C test/geostat_tb TB=/path/to/tb
//
// Statistical checks compare moments estimated over many realisations with
// the model. Every realisation evaluates the field at a few far-apart origins
// and at origin + lag; the estimate of C(h) is the mean over realisations of
// the per-realisation average of (Z(o) - m)(Z(o + h) - m), and its standard
// error is computed from the spread of those per-realisation averages (so it
// is honest whatever the dependence between origins). A check passes when
// |estimate - C(h)| <= 5 standard errors (about 1e-6 false-alarm probability
// per check; the seeds are fixed, so the outcome is reproducible).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include "DataImage.hpp"
#include "geostat/covarianceModel.hpp"
#include "geostat/turningBands.hpp"
#include "../geostat_common/testData.hpp"

using namespace g2s::geostat;
using geostat_test::testId;
using geostat_test::testIdText;

namespace {

// Quick mode (CI): every structure type is still checked, with the same
// tolerances, but in fewer dimension and anisotropy combinations.
bool quick = false;

// Image names unique to this run (removed at exit with the outputs of testId()).
std::string N(const std::string& stem) { return geostat_test::testName(stem); }

int failures = 0;
int checks = 0;
std::string currentSection;

void report(bool ok, const std::string& what, const char* file, int line) {
	++checks;
	if (ok) return;
	++failures;
	fprintf(stderr, "FAIL [%s] %s:%d: %s\n", currentSection.c_str(), file, line, what.c_str());
}

#define CHECK(condition) report(bool(condition), #condition, __FILE__, __LINE__)
#define CHECK_MSG(condition, message) report(bool(condition), std::string(#condition) + " -- " + (message), __FILE__, __LINE__)

template <class F>
void checkThrows(F f, const std::string& fragment, const char* file, int line) {
	try {
		f();
		report(false, "expected an error containing '" + fragment + "', got none", file, line);
	} catch (const Error& e) {
		const std::string message = e.what();
		report(message.find(fragment) != std::string::npos, "error '" + message + "' should contain '" + fragment + "'", file, line);
	}
}
#define CHECK_THROWS(expression, fragment) checkThrows([&]() { expression; }, fragment, __FILE__, __LINE__)

void section(const std::string& name) { currentSection = name; printf("-- %s\n", name.c_str()); }

ResolvedModel stationary(const std::string& text, unsigned ndim) {
	return CovarianceModel(ModelSpec::parse(text), Bindings(), ndim).stationaryModel();
}

unsigned hardwareThreads() { return std::max(1u, std::thread::hardware_concurrency()); }

// Runs f(r) for r < count on all cores (order irrelevant to the caller).
template <class F>
void parallelFor(size_t count, F f) {
	detail::parallelBlocks(count, hardwareThreads(), f);
}

// ============================================================ helpers

void testCos() {
	section("fast cosine");
	double worst = 0.0;
	for (int i = -2000000; i <= 2000000; ++i) {
		const double x = i * 1.2345e-3 + (i % 7) * 1e-2;
		worst = std::max(worst, std::fabs(turningBandsCos(x) - std::cos(x)));
	}
	for (double x : { 1e4, 1e6, 12345.678, -98765.4321 }) worst = std::max(worst, std::fabs(turningBandsCos(x) - std::cos(x)) * 1e-3);
	printf("   max |error| = %.2e\n", worst);
	CHECK(worst < 1e-9);
}

void testLineProfiles() {
	section("dilution profiles reproduce the 3D line covariances");
	// C1(r) = int g(s) g(s + r) ds over |s| < 1/2, compared with d/dr [r C3(r)].
	auto check = [](bool cubic) {
		auto g = [&](double s) {
			if (s < -0.5 || s >= 0.5) return 0.0;
			return cubic ? std::sqrt(52.5) * s * (1 - 4 * s * s) : 2 * std::sqrt(3.0) * s;
		};
		double worst = 0.0;
		for (double r = 0.0; r <= 1.2; r += 0.05) {
			const int n = 200000;
			double sum = 0.0;
			for (int i = 0; i < n; ++i) { const double s = -0.5 + (i + 0.5) / n; sum += g(s) * g(s + r); }
			sum /= n;
			const double expected = r >= 1 ? 0.0 : cubic ? 1 - 21 * r * r + 35 * r * r * r - 21 * std::pow(r, 5) + 6 * std::pow(r, 7) : 1 - 3 * r + 2 * r * r * r;
			worst = std::max(worst, std::fabs(sum - expected));
		}
		return worst;
	};
	CHECK(check(false) < 1e-4);
	CHECK(check(true) < 1e-4);
}

// ============================================================ moments over realisations

struct LagSet {
	std::vector<std::vector<double> > lags; // lags[0] is the zero lag
};

// Lags along the principal axes of the first non-nugget structure (fractions
// of its ranges), along a diagonal, and zero.
LagSet makeLags(const ResolvedModel& model) {
	const unsigned ndim = model.ndim;
	const ResolvedStructure* s = nullptr;
	for (const ResolvedStructure& st : model.structures) if (st.type != StructureType::Nugget) { s = &st; break; }
	LagSet set;
	set.lags.push_back(std::vector<double>(ndim, 0.0));
	for (unsigned k = 0; k < ndim; ++k) {
		for (double f : { 0.1, 0.3, 0.6, 1.0 }) {
			std::vector<double> lag(ndim);
			for (unsigned d = 0; d < ndim; ++d) lag[d] = s->rotation[d * ndim + k] * s->range[k] * f;
			set.lags.push_back(lag);
		}
	}
	for (double f : { 0.2, 0.5 }) {
		std::vector<double> lag(ndim);
		for (unsigned d = 0; d < ndim; ++d) lag[d] = s->range[0] * f * (d == 1 ? -1.0 : 1.0) / std::sqrt(double(ndim));
		set.lags.push_back(lag);
	}
	return set;
}

struct MomentResult {
	double maxZ = 0.0;      // largest |estimate - model| / standard error
	double maxError = 0.0;  // largest |estimate - model|
	double meanZ = 0.0;
	double kurtosis = 0.0;
};

// Covariance at the lags, mean and kurtosis, over `realisations` realisations
// with `origins` origins each.
MomentResult moments(const ResolvedModel& model, const LagSet& set, double mean, unsigned realisations, unsigned origins, unsigned lines, uint64_t seed,
                     bool verbose) {
	const unsigned ndim = model.ndim;
	double reach = 0.0;
	for (const ResolvedStructure& s : model.structures) for (unsigned k = 0; k < ndim; ++k) reach = std::max(reach, s.range[k]);
	double maxLag = 0.0;
	for (const std::vector<double>& lag : set.lags) for (double v : lag) maxLag = std::max(maxLag, std::fabs(v));
	const double spacing = 4.0 * reach + 2.0 * maxLag; // origins far apart (correlation < exp(-12))
	const size_t lagCount = set.lags.size();
	std::vector<double> points;
	for (unsigned o = 0; o < origins; ++o) {
		for (size_t h = 0; h < lagCount; ++h) {
			for (unsigned d = 0; d < ndim; ++d) points.push_back((d == o % ndim ? spacing * (1 + o / ndim) : 0.37 * d) + set.lags[h][d]);
		}
	}
	const size_t count = points.size() / ndim;
	std::vector<double> perRealisation(size_t(realisations) * (lagCount + 3), 0.0); // lagCount products, mean, m2, m4
	TurningBandsSettings settings;
	settings.lineCount = lines;
	settings.seed = seed;
	settings.mean = mean;
	parallelFor(realisations, [&](size_t r) {
		std::vector<double> values(count);
		const TurningBandsRealisation realisation(model, settings, uint32_t(r));
		realisation.evaluate(points.data(), count, nullptr, 0, values.data());
		double* out = &perRealisation[r * (lagCount + 3)];
		for (unsigned o = 0; o < origins; ++o) {
			const double z0 = values[o * lagCount] - mean;
			for (size_t h = 0; h < lagCount; ++h) out[h] += z0 * (values[o * lagCount + h] - mean) / origins;
			out[lagCount] += z0 / origins;
			out[lagCount + 1] += z0 * z0 / origins;
			out[lagCount + 2] += z0 * z0 * z0 * z0 / origins;
		}
	});
	MomentResult result;
	auto estimate = [&](size_t column, double& average, double& standardError) {
		double s1 = 0, s2 = 0;
		for (unsigned r = 0; r < realisations; ++r) { const double v = perRealisation[r * (lagCount + 3) + column]; s1 += v; s2 += v * v; }
		average = s1 / realisations;
		standardError = std::sqrt(std::max(0.0, s2 / realisations - average * average) / (realisations - 1));
	};
	for (size_t h = 0; h < lagCount; ++h) {
		double average, se;
		estimate(h, average, se);
		const double expected = model.covariance(set.lags[h].data());
		const double error = std::fabs(average - expected);
		result.maxError = std::max(result.maxError, error);
		result.maxZ = std::max(result.maxZ, error / std::max(se, 1e-12));
		if (verbose) printf("      lag %zu: C = %.4f, estimate %.4f +- %.4f\n", h, expected, average, se);
	}
	double m1, se1, m2, se2, m4, se4;
	estimate(lagCount, m1, se1);
	estimate(lagCount + 1, m2, se2);
	estimate(lagCount + 2, m4, se4);
	result.meanZ = std::fabs(m1) / std::max(se1, 1e-12);
	result.kurtosis = m4 / (m2 * m2);
	return result;
}

void checkModel(const std::string& text, unsigned ndim, unsigned realisations = 3000, unsigned origins = 8) {
	const ResolvedModel model = stationary(text, ndim);
	const LagSet set = makeLags(model);
	const MomentResult result = moments(model, set, 2.5, realisations, origins, 1000, 20261121, false);
	printf("   %uD %-75s max|err| %.4f  max z %.2f  mean z %.2f  kurtosis %.3f\n", ndim, text.c_str(), result.maxError, result.maxZ, result.meanZ, result.kurtosis);
	CHECK_MSG(result.maxZ <= 5.0, text + " (" + std::to_string(ndim) + "D): covariance off by " + std::to_string(result.maxZ) + " standard errors");
	CHECK_MSG(result.meanZ <= 5.0, text + ": mean off by " + std::to_string(result.meanZ) + " standard errors");
	CHECK_MSG(std::fabs(result.kurtosis - 3.0) <= 0.15, text + ": kurtosis " + std::to_string(result.kurtosis));
}

void testCovariances() {
	section("covariance, mean and variance over realisations (5 standard errors)");
	const char* isotropic[] = {
		"spherical(range=20)", "exponential(range=20)", "gaussian(range=20)", "cubic(range=20)",
		"matern(range=20, nu=0.3)", "matern(range=20, nu=1.5)", "matern(range=20, nu=4)",
		"stable(range=20, alpha=0.5)", "stable(range=20, alpha=1.5)", "stable(range=20, alpha=2)",
	};
	const char* anisotropic2[] = {
		"spherical(range=[30,10], angle=30)", "exponential(range=[30,10], angle=-50)", "gaussian(range=[30,10], angle=75)",
		"cubic(range=[30,10], angle=30)", "matern(range=[30,10], angle=30, nu=1.5)", "stable(range=[30,10], angle=30, alpha=1.2)",
	};
	const char* anisotropic3[] = {
		"spherical(range=[30,15,8], angle=[30,20,10])", "exponential(range=[30,15,8], angle=[30,20,10])",
		"gaussian(range=[30,15,8], angle=[30,20,10])", "cubic(range=[30,15,8], angle=[30,20,10])",
		"matern(range=[30,15,8], angle=[30,20,10], nu=0.8)", "stable(range=[30,15,8], angle=[30,20,10], alpha=0.7)",
	};
	size_t k = 0;
	for (unsigned ndim : { 1u, 2u, 3u })
		for (const char* text : isotropic) if (!quick || k++ % 3 == ndim - 1) checkModel(text, ndim);
	k = 0;
	for (const char* text : anisotropic2) if (!quick || k++ % 2 == 0) checkModel(text, 2);
	k = 0;
	for (const char* text : anisotropic3) if (!quick || k++ % 2 == 1) checkModel(text, 3);
	section("nested model: nugget + two structures");
	checkModel("0.2*nugget + 0.5*spherical(range=[30,12], angle=25) + 0.3*exponential(range=50)", 2);
	checkModel("0.1*nugget + 0.6*cubic(range=[25,12,6], angle=[10,0,40]) + 0.3*gaussian(range=40)", 3);
	checkModel("0.15*nugget + 2*matern(range=[20,35], angle=-20, nu=2.5) + 0.5*stable(range=60, alpha=1)", 2);
}

void testGaussianity() {
	section("marginal close to Gaussian at the default number of lines (|kurtosis - 3| <= 0.08, 2000 x 64 values)");
	size_t k = 0;
	for (unsigned ndim : { 2u, 3u }) {
		for (const char* text : { "spherical(range=20)", "cubic(range=20)", "exponential(range=20)", "gaussian(range=20)", "matern(range=20, nu=1.5)", "stable(range=20, alpha=1)" }) {
			if (quick && k++ % 2 != ndim - 2) continue;
			const ResolvedModel model = stationary(text, ndim);
			LagSet set;
			set.lags.push_back(std::vector<double>(ndim, 0.0));
			const MomentResult result = moments(model, set, 0.0, 2000, 64, TurningBandsSettings().lineCount, 99, false);
			printf("   %uD %-28s kurtosis %.4f\n", ndim, text, result.kurtosis);
			CHECK_MSG(std::fabs(result.kurtosis - 3.0) <= 0.08, std::string(text) + ": kurtosis " + std::to_string(result.kurtosis));
		}
	}
}

// ============================================================ banding

// Directional variograms of one realisation on a grid, at lags of equal length.
double isotropyDeviation(const std::string& text, unsigned ndim, unsigned lines, unsigned realisations) {
	const ResolvedModel model = stationary(text, ndim);
	const unsigned side = ndim == 2 ? 400 : 72;
	std::vector<unsigned> dims(ndim, side);
	size_t count = 1;
	for (unsigned d : dims) count *= d;
	std::vector<double> points(count * ndim);
	for (size_t i = 0; i < count; ++i) {
		size_t rest = i;
		for (unsigned d = 0; d < ndim; ++d) { points[i * ndim + d] = double(rest % dims[d]); rest /= dims[d]; }
	}
	// Integer lags of equal length 5 (and 10) in different directions.
	std::vector<std::vector<int> > lags;
	if (ndim == 2) lags = { { 5, 0 }, { 0, 5 }, { 3, 4 }, { 4, -3 }, { 10, 0 }, { 0, 10 }, { 6, 8 }, { 8, -6 } };
	else lags = { { 5, 0, 0 }, { 0, 5, 0 }, { 0, 0, 5 }, { 3, 4, 0 }, { 0, 3, 4 }, { 4, 0, -3 }, { 10, 0, 0 }, { 0, 10, 0 }, { 0, 0, 10 }, { 6, 8, 0 }, { 0, 6, 8 }, { 8, 0, -6 } };
	const size_t perLength = lags.size() / 2;
	TurningBandsSettings settings;
	settings.lineCount = lines;
	settings.seed = 4242;
	settings.threads = hardwareThreads();
	std::vector<double> values(count);
	double deviation = 0.0;
	for (unsigned r = 0; r < realisations; ++r) {
		simulateTurningBands(model, settings, points.data(), count, r, 1, values.data());
		for (size_t group = 0; group < 2; ++group) {
			double lo = 1e300, hi = 0.0;
			for (size_t j = group * perLength; j < (group + 1) * perLength; ++j) {
				const std::vector<int>& lag = lags[j];
				double sum = 0.0;
				size_t pairs = 0;
				for (size_t i = 0; i < count; ++i) {
					size_t rest = i, other = 0, stride = 1;
					bool inside = true;
					for (unsigned d = 0; d < ndim; ++d) {
						const long c = long(rest % dims[d]) + lag[d];
						rest /= dims[d];
						if (c < 0 || c >= long(dims[d])) { inside = false; break; }
						other += size_t(c) * stride;
						stride *= dims[d];
					}
					if (!inside) continue;
					const double diff = values[i] - values[other];
					sum += diff * diff;
					++pairs;
				}
				const double gamma = 0.5 * sum / double(pairs);
				lo = std::min(lo, gamma);
				hi = std::max(hi, gamma);
			}
			deviation += (hi - lo) / (0.5 * (hi + lo));
		}
	}
	return deviation / (2.0 * realisations);
}

void testBanding() {
	section("no banding: isotropy of directional variograms within one realisation");
	// Mean relative spread (max - min) / mean of the variograms at equal lag
	// length in 4 (2D) or 6 (3D) directions, averaged over realisations and two
	// lag lengths. Part of the spread is the ergodic fluctuation of any finite
	// field; the reference is the same statistic with 8 times more lines
	// (close to an exact Gaussian field). At the default -nb the spread must
	// stay within 1.5 x reference + 0.02 (and below 0.1); with 4 lines
	// (banding) it is several times larger, which shows the statistic detects
	// banding.
	const unsigned lines = TurningBandsSettings().lineCount;
	for (unsigned ndim : { 2u, 3u }) {
		// 2D: 400 x 400 grid, range 25; 3D: 72^3 grid, range 12 (lags 5 and 10).
		const std::vector<std::string> texts = ndim == 2 ? std::vector<std::string>{ "spherical(range=25)", "exponential(range=25)", "gaussian(range=25)" }
		                                                 : std::vector<std::string>{ "spherical(range=12)", "exponential(range=12)", "gaussian(range=12)" };
		for (const std::string& text : texts) {
			if (quick && text.compare(0, 9, "spherical") != 0) continue;
			const unsigned realisations = ndim == 2 ? 6 : 4;
			const double good = isotropyDeviation(text, ndim, lines, realisations);
			const double reference = isotropyDeviation(text, ndim, 8 * lines, realisations);
			const double bad = isotropyDeviation(text, ndim, 4, realisations);
			printf("   %uD %-24s spread %.3f at -nb %u, %.3f at -nb %u (reference), %.3f at -nb 4\n", ndim, text.c_str(), good, lines, reference, 8 * lines, bad);
			CHECK_MSG(good <= 1.5 * reference + 0.02 && good <= 0.1, text + ": directional spread " + std::to_string(good));
			CHECK_MSG(bad > 3 * good, text + ": the statistic does not detect banding with 4 lines");
		}
	}
}

// ============================================================ determinism

std::vector<double> gridPoints(const std::vector<unsigned>& dims) {
	size_t count = 1;
	for (unsigned d : dims) count *= d;
	const unsigned ndim = unsigned(dims.size());
	std::vector<double> points(count * ndim);
	for (size_t i = 0; i < count; ++i) {
		size_t rest = i;
		for (unsigned d = 0; d < ndim; ++d) { points[i * ndim + d] = double(rest % dims[d]); rest /= dims[d]; }
	}
	return points;
}

void testDeterminism() {
	section("same seed, any thread count; batch = single realisation");
	for (const char* text : { "0.1*nugget + 0.5*spherical(range=[30,12], angle=25) + 0.4*matern(range=40, nu=0.7)",
	                          "0.3*cubic(range=[20,10,5], angle=[10,20,30]) + 0.7*stable(range=25, alpha=1.3)" }) {
		const ResolvedModel model = stationary(text, std::string(text).find("[20,10,5]") != std::string::npos ? 3 : 2);
		const std::vector<unsigned> dims = model.ndim == 2 ? std::vector<unsigned>{ 97, 61 } : std::vector<unsigned>{ 23, 19, 17 };
		const std::vector<double> points = gridPoints(dims);
		const size_t count = points.size() / model.ndim;
		TurningBandsSettings settings;
		settings.seed = 77;
		settings.mean = -1.5;
		std::vector<double> reference(4 * count), other(4 * count), single(count);
		settings.threads = 1;
		simulateTurningBands(model, settings, points.data(), count, 0, 4, reference.data());
		for (unsigned threads : { 2u, 3u, 8u, 13u }) {
			settings.threads = threads;
			simulateTurningBands(model, settings, points.data(), count, 0, 4, other.data());
			CHECK_MSG(other == reference, std::string(text) + ": differs with " + std::to_string(threads) + " threads");
		}
		for (unsigned r = 0; r < 4; ++r) {
			simulateTurningBands(model, settings, points.data(), count, r, 1, single.data());
			CHECK_MSG(std::equal(single.begin(), single.end(), reference.begin() + size_t(r) * count), std::string(text) + ": realisation " + std::to_string(r) + " differs from the batch");
		}
		bool distinct = false;
		for (size_t i = 0; i < count; ++i) distinct |= reference[i] != reference[count + i];
		CHECK(distinct);
		settings.seed = 78;
		simulateTurningBands(model, settings, points.data(), count, 0, 1, single.data());
		CHECK(!std::equal(single.begin(), single.end(), reference.begin()));
	}
}

void testScattered() {
	section("scattered points = grid at coinciding coordinates");
	for (unsigned ndim : { 1u, 2u, 3u }) {
		const std::string text = "0.2*nugget + 0.4*spherical(range=17) + 0.4*cubic(range=9) + 0.5*exponential(range=12) + 0.3*gaussian(range=8)";
		const ResolvedModel model = stationary(text, ndim);
		const std::vector<unsigned> dims = ndim == 1 ? std::vector<unsigned>{ 300 } : ndim == 2 ? std::vector<unsigned>{ 50, 40 } : std::vector<unsigned>{ 20, 16, 12 };
		const std::vector<double> grid = gridPoints(dims);
		const size_t count = grid.size() / ndim;
		std::vector<uint64_t> gridIds(count);
		for (size_t i = 0; i < count; ++i) gridIds[i] = i;
		TurningBandsSettings settings;
		settings.seed = 5;
		std::vector<double> gridValues(2 * count);
		simulateTurningBands(model, settings, grid.data(), count, 0, 2, gridValues.data(), gridIds.data());
		// Every 7th grid node, in shuffled order, mixed with off-grid points far outside the grid's box.
		std::vector<double> scattered;
		std::vector<uint64_t> ids, gridIndex;
		for (size_t i = 0; i < count; i += 7) {
			const size_t j = (i * 2654435761u) % count;
			for (unsigned d = 0; d < ndim; ++d) scattered.push_back(grid[j * ndim + d]);
			ids.push_back(j);
			gridIndex.push_back(j);
			for (unsigned d = 0; d < ndim; ++d) scattered.push_back(grid[j * ndim + d] * 3.3 - 500.25);
			ids.push_back(count + j);
			gridIndex.push_back(uint64_t(-1));
		}
		const size_t n = ids.size();
		std::vector<double> values(2 * n);
		settings.threads = 4;
		simulateTurningBands(model, settings, scattered.data(), n, 0, 2, values.data(), ids.data());
		double worst = 0.0;
		for (unsigned r = 0; r < 2; ++r)
			for (size_t i = 0; i < n; ++i)
				if (gridIndex[i] != uint64_t(-1)) worst = std::max(worst, std::fabs(values[r * n + i] - gridValues[r * count + gridIndex[i]]));
		printf("   %uD max |difference| %.2e\n", ndim, worst);
		CHECK(worst <= 1e-12);
	}
}

void testRefusals() {
	section("unsupported structures and non-stationary models are refused");
	{
		Bindings bindings;
		bindings.setTable("C", std::vector<float>(9, 1.0f), { 3, 3 });
		const CovarianceModel model(ModelSpec::parse("0.5*exponential(range=10) + 1*map(C)"), bindings, 2);
		CHECK_THROWS(checkTurningBandsModel(model.stationaryModel()), "structure 2 (map) is not supported");
		TurningBandsSettings settings;
		CHECK_THROWS(TurningBandsRealisation(model.stationaryModel(), settings, 0), "map");
	}
	{
		Bindings bindings;
		bindings.setField("L", std::vector<float>{ 5, 6, 7, 8 });
		const CovarianceModelSet set = buildCovarianceModelSet({ "exponential(range=L)" }, bindings, 2);
		CHECK_THROWS(turningBandsModel(set), "stationary models only");
	}
	{
		const CovarianceModelSet set = buildCovarianceModelSet({ "exponential(range=10)", "gaussian(range=5)" }, Bindings(), 2, { 0, 1, 1, 0 });
		CHECK_THROWS(turningBandsModel(set), "stationary models only");
	}
	{
		const ResolvedModel model = stationary("exponential(range=10)", 4);
		CHECK_THROWS(checkTurningBandsModel(model), "1D, 2D or 3D");
	}
	{
		const CovarianceModelSet set = buildCovarianceModelSet({ "0.1*nugget + exponential(range=10)" }, Bindings(), 2);
		CHECK(&turningBandsModel(set) == &set.stationaryModel());
	}
}

// ============================================================ executable

bool writeImage(const std::string& name, const std::vector<unsigned>& dims, float value) {
	std::vector<unsigned> d = dims;
	g2s::DataImage image(unsigned(d.size()), d.data(), 1);
	for (size_t i = 0; i < image.dataSize(); ++i) image._data[i] = value;
	image.write(name);
	return !g2s::DataImage::createFromFile(name).isEmpty();
}

void testExecutable(const std::string& binary) {
	section("tb executable");
	FILE* probe = fopen(binary.c_str(), "r");
	if (!probe) { printf("   skipped: %s not built\n", binary.c_str()); return; }
	fclose(probe);
	const std::vector<unsigned> dims = { 60, 45 };
	CHECK(writeImage(N("tb_test_grid"), dims, std::nanf("")));
	const std::string model = "0.1*nugget + 0.6*spherical(range=[30,12], angle=25) + 0.3*matern(range=20, nu=1.5)";
	auto run = [&](const std::string& options, int id) {
		const std::string command = binary + " -di " + N("tb_test_grid") + " -cov '" + model + "' " + options + " -id " + std::to_string(id) + " -o " + N("tb_test_out") + " 2>/dev/null >/dev/null";
		return std::system(command.c_str());
	};
	CHECK(run("-nr 3 -s 11 -m -2.5 -j 1", testId(700)) == 0);
	CHECK(run("-nr 3 -s 11 -m -2.5 -j 6", testId(701)) == 0);
	g2s::DataImage a = g2s::DataImage::createFromFile("im_1_" + testIdText(700));
	g2s::DataImage b = g2s::DataImage::createFromFile("im_1_" + testIdText(701));
	CHECK(!a.isEmpty() && !b.isEmpty());
	if (!a.isEmpty() && !b.isEmpty()) {
		CHECK(a._dims == dims && a._nbVariable == 3);
		CHECK(std::equal(a._data, a._data + a.dataSize(), b._data));
		// Same values as the engine on the grid coordinates (cell ids key the nugget).
		const ResolvedModel resolved = stationary(model, 2);
		const std::vector<double> points = gridPoints(dims);
		const size_t count = points.size() / 2;
		TurningBandsSettings settings;
		settings.seed = 11;
		settings.mean = -2.5;
		std::vector<double> values(3 * count);
		simulateTurningBands(resolved, settings, points.data(), count, 0, 3, values.data());
		double worst = 0.0, sum = 0.0;
		for (unsigned r = 0; r < 3; ++r)
			for (size_t i = 0; i < count; ++i) {
				worst = std::max(worst, std::fabs(double(a._data[i * 3 + r]) - values[r * count + i]));
				sum += a._data[i * 3 + r];
			}
		printf("   executable vs engine: max |difference| %.2e, mean %.3f\n", worst, sum / (3.0 * count));
		CHECK(worst <= 1e-5);
	}
	CHECK(run("-nb 300", testId(702)) == 0);
	// Refusals: non-zero exit and an explicit message (the report goes to stderr).
	auto refused = [&](const std::string& arguments, const std::string& fragment) {
		const std::string log = N("tb_test_log.txt");
		const std::string command = binary + " " + arguments + " -o " + N("tb_test_out") + " > " + log + " 2>&1";
		const bool failed = std::system(command.c_str()) != 0;
		std::string text;
		if (FILE* file = fopen(log.c_str(), "r")) {
			char buffer[4096];
			size_t n;
			while ((n = fread(buffer, 1, sizeof(buffer), file)) > 0) text.append(buffer, n);
			fclose(file);
		}
		const bool explained = text.find(fragment) != std::string::npos;
		report(failed && explained, "tb " + arguments + " should fail with '" + fragment + "'", __FILE__, __LINE__);
	};
	CHECK(writeImage(N("tb_test_table"), dims, 1.0f)); // map tables have the size of the grid
	refused("-di " + N("tb_test_grid") + " -cov '0.5*exponential(range=10) + 1*map(C)' -cvar C " + N("tb_test_table") + " -id " + testIdText(703), "structure 2 (map) is not supported");
	g2s::DataImage ranges(2, std::vector<unsigned>{ 60, 45 }.data(), 1);
	for (size_t i = 0; i < ranges.dataSize(); ++i) ranges._data[i] = float(5 + i % 7);
	ranges.write(N("tb_test_range"));
	refused("-di " + N("tb_test_grid") + " -cov 'exponential(range=L)' -cvar L " + N("tb_test_range") + " -id " + testIdText(704), "stationary models only");
	refused("-di " + N("tb_test_grid") + " -cov 'exponential(range=10)' -csim -id " + testIdText(705), "-csim");
	refused("-di " + N("tb_test_grid") + " -cov 'exponential(range=10)' -nb 0 -id " + testIdText(706), "-nb expects");
	CHECK(writeImage(N("tb_test_4d"), { 3, 3, 3, 3 }, std::nanf("")));
	refused("-di " + N("tb_test_4d") + " -cov 'exponential(range=10)' -id " + testIdText(708), "1D, 2D or 3D");
	// A constant -cvar and a positive mean.
	const std::string command = binary + " -di " + N("tb_test_grid") + " -cov 'exponential(range=L)' -cvar L 12 -m 3 -nr 2 -s 3 -id " + testIdText(707) + " -o " + N("tb_test_out") + " 2>/dev/null >/dev/null";
	CHECK(std::system(command.c_str()) == 0);
	g2s::DataImage c = g2s::DataImage::createFromFile("im_1_" + testIdText(707));
	CHECK(!c.isEmpty() && c._nbVariable == 2);
	if (!c.isEmpty()) {
		double sum = 0.0;
		for (size_t i = 0; i < c.dataSize(); ++i) sum += c._data[i];
		CHECK(std::fabs(sum / double(c.dataSize()) - 3.0) < 1.0);
	}
}

void timing(const std::string& binary) {
	section("timings of the executable at the default -nb (all cores)");
	FILE* probe = fopen(binary.c_str(), "r");
	if (!probe) { printf("   skipped\n"); return; }
	fclose(probe);
	struct Case { std::vector<unsigned> dims; const char* model; };
	const Case cases[] = {
		{ { 1000, 1000 }, "exponential(range=50)" }, { { 1000, 1000 }, "spherical(range=50)" },
		{ { 1000, 1000 }, "0.1*nugget + 0.6*spherical(range=[60,20], angle=30) + 0.3*matern(range=80, nu=1.5)" },
		{ { 200, 200, 200 }, "exponential(range=30)" }, { { 200, 200, 200 }, "spherical(range=30)" },
		{ { 200, 200, 200 }, "0.1*nugget + 0.6*spherical(range=[40,20,10], angle=[30,0,0]) + 0.3*matern(range=50, nu=1.5)" },
	};
	unsigned id = testId(800);
	for (const Case& c : cases) {
		const std::string grid = N("tb_test_timing_") + std::to_string(c.dims.size()) + "d";
		CHECK(writeImage(grid, c.dims, std::nanf("")));
		const std::string command = binary + " -di " + grid + " -cov '" + c.model + "' -s 1 -j " + std::to_string(hardwareThreads()) + " -id " + std::to_string(id++) + " -o " + N("tb_test_out") + " 2>/dev/null >/dev/null";
		const auto begin = std::chrono::steady_clock::now();
		CHECK(std::system(command.c_str()) == 0);
		const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
		std::string size;
		for (size_t d = 0; d < c.dims.size(); ++d) size += (d ? " x " : "") + std::to_string(c.dims[d]);
		printf("   %-15s %-90s %.2f s (wall, including I/O)\n", size.c_str(), c.model, seconds);
	}
}

} // namespace

int main(int argc, char** argv) {
	const std::string binary = argc > 1 ? argv[1] : "../../build/c++-build/tb";
	const bool withTimings = argc > 2 && std::string(argv[2]) == "timings";
	quick = argc > 2 && std::string(argv[2]) == "quick";
	if (quick) printf("quick mode: a subset of the statistical cases\n");
	const auto begin = std::chrono::steady_clock::now();
	testCos();
	testLineProfiles();
	testRefusals();
	testDeterminism();
	testScattered();
	testCovariances();
	testGaussianity();
	testBanding();
	testExecutable(binary);
	if (withTimings) timing(binary);
	printf("%d checks, %d failures (%.1f s)\n", checks, failures, std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
	return failures == 0 ? 0 : 1;
}
