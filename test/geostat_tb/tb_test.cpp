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
#include <array>
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
	section("fast cosine (float, phase reduced in double)");
	double worst = 0.0;
	for (int i = -2000000; i <= 2000000; ++i) {
		const double y = i * 1.2345e-4 + (i % 7) * 1e-3;
		worst = std::max(worst, std::fabs(double(turningBandsCosTurns(y)) - std::cos(kTwoPi * (y - std::nearbyint(y)))));
	}
	// Large phases: the error does not grow with |y| (the reduction is exact in double).
	for (double y : { 1e4 + 0.123, 1e6 + 0.37, 12345.678, -98765.4321, 3.0e9 + 0.25, -7.0e11 + 0.4 }) {
		const long double reduced = (long double)y - std::nearbyint((long double)y);
		worst = std::max(worst, std::fabs(double(turningBandsCosTurns(y)) - double(std::cos(2.0L * 3.14159265358979323846264338327950288L * reduced))));
	}
	printf("   max |error| = %.2e (instruction set %s, %u lanes)\n", worst, simd::kInstructionSet, simd::kLanes);
	CHECK(worst < 2.5e-7);
}

// sumLines against a long-double reference, for hand-made lines with high
// frequencies and coordinates far from the origin (the float part of the
// kernel works on the reduced phase only).
void testKernelAccuracy() {
	section("kernel accuracy against a long-double reference (large coordinates, high frequencies)");
	for (LineProcess process : { LineProcess::Spectral, LineProcess::DilutionSpherical, LineProcess::DilutionCubic }) {
		for (unsigned ndim : { 1u, 2u, 3u }) {
			for (double offset : { 0.0, 1e3, 1e6 }) {
				const size_t L = 1000, n = 777;
				std::vector<double> direction(ndim * L), shift(L);
				std::vector<uint32_t> key(2 * L);
				for (size_t l = 0; l < L; ++l) {
					for (unsigned d = 0; d < ndim; ++d) {
						const double u = g2s::geostat::uniform(3, 0, l * 3 + d, 9) - 0.5;
						// spectral: up to 5 turns per unit; dilution: up to 1/2 segment per unit
						direction[d * L + l] = process == LineProcess::Spectral ? 10.0 * u : u;
					}
					shift[l] = g2s::geostat::uniform(3, 0, l, 10);
					const std::array<uint32_t, 4> bits = g2s::geostat::randomBlock(3, 0, l, 11);
					key[l] = bits[0] | 1u;
					key[L + l] = bits[1];
				}
				LineSet lines;
				lines.process = process;
				lines.ndim = ndim;
				lines.count = L;
				lines.amplitude = std::sqrt((process == LineProcess::Spectral ? 2.0 : 1.0) / double(L));
				lines.direction = direction.data();
				lines.shift = shift.data();
				lines.signKey = key.data();
				std::vector<double> points(n * ndim), out(n, 0.0);
				for (size_t i = 0; i < n; ++i)
					for (unsigned d = 0; d < ndim; ++d) points[i * ndim + d] = offset + 97.0 * g2s::geostat::uniform(4, 0, i * 3 + d, 12);
				sumLines(lines, points.data(), n, out.data());
				double worst = 0.0;
				for (size_t i = 0; i < n; ++i) {
					long double sum = 0.0L;
					for (size_t l = 0; l < L; ++l) {
						long double t = shift[l];
						for (unsigned d = 0; d < ndim; ++d) t += (long double)direction[d * L + l] * (long double)points[i * ndim + d];
						if (process == LineProcess::Spectral) {
							sum += std::cos(2.0L * 3.14159265358979323846264338327950288L * (t - std::nearbyint(t)));
						} else {
							const long double k = std::floor(t), s = t - k - 0.5L;
							const long double g = process == LineProcess::DilutionCubic ? std::sqrt(52.5L) * s * (1 - 4 * s * s) : 2 * std::sqrt(3.0L) * s;
							sum += dilutionSign(key[l], key[L + l], int64_t(k)) * g;
						}
					}
					worst = std::max(worst, std::fabs(out[i] - double(lines.amplitude * sum)));
				}
				printf("   %-22s %uD offset %-7g max |error| %.2e\n", lineProcessName(process), ndim, offset, worst);
				CHECK_MSG(worst < 1.5e-6, std::string(lineProcessName(process)) + ": kernel error " + std::to_string(worst));
			}
		}
	}
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
	const RegularGrid grid = RegularGrid::cells(ndim, dims.data()); // the grid path, as tb
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
		simulateTurningBandsGrid(model, settings, grid, r, 1, values.data());
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

void testGridDeterminism() {
	section("grid path: same seed, any thread count; batch = single realisation (bit for bit)");
	// Grids with several column tiles and rows per block that do not divide the
	// rows, all three line processes, steep dilution lines (range 3) included.
	for (const char* text : { "0.1*nugget + 0.5*spherical(range=[30,12], angle=25) + 0.4*matern(range=40, nu=0.7)",
	                          "0.3*cubic(range=[20,10,5], angle=[10,20,30]) + 0.7*stable(range=25, alpha=1.3) + 0.2*spherical(range=3)",
	                          "0.5*cubic(range=3) + 0.5*exponential(range=7)" }) {
		const std::string t(text);
		const unsigned ndim = t.find("[20,10,5]") != std::string::npos ? 3 : t.find("[30,12]") != std::string::npos ? 2 : 1;
		const ResolvedModel model = stationary(text, ndim);
		const std::vector<unsigned> dims = ndim == 1 ? std::vector<unsigned>{ 3000 } : ndim == 2 ? std::vector<unsigned>{ 1100, 37 } : std::vector<unsigned>{ 530, 9, 7 };
		const RegularGrid grid = RegularGrid::cells(ndim, dims.data());
		const size_t count = grid.nodeCount();
		TurningBandsSettings settings;
		settings.seed = 77;
		settings.mean = -1.5;
		std::vector<double> reference(5 * count), other(5 * count), single(count);
		settings.threads = 1;
		simulateTurningBandsGrid(model, settings, grid, 0, 5, reference.data());
		for (unsigned threads : { 2u, 3u, 8u, 13u }) {
			settings.threads = threads;
			simulateTurningBandsGrid(model, settings, grid, 0, 5, other.data());
			CHECK_MSG(other == reference, t + ": grid path differs with " + std::to_string(threads) + " threads");
		}
		for (unsigned r = 0; r < 5; ++r) {
			simulateTurningBandsGrid(model, settings, grid, r, 1, single.data());
			CHECK_MSG(std::equal(single.begin(), single.end(), reference.begin() + size_t(r) * count), t + ": grid realisation " + std::to_string(r) + " differs from the batch");
		}
		// Every block computed alone, with a fresh scratch, gives the same values.
		TurningBandsRealisation realisation(model, settings, 2);
		realisation.attachGrid(grid);
		GridScratch scratch;
		std::vector<double> values;
		bool same = true;
		for (const GridBlock& block : planGridBlocks(grid)) {
			values.assign(block.count(), 0.0);
			realisation.evaluateGrid(block, values.data(), scratch);
			for (size_t row = block.rowBegin; row < block.rowEnd; ++row)
				for (size_t c = block.columnBegin; c < block.columnEnd; ++c)
					same &= values[(row - block.rowBegin) * block.width() + (c - block.columnBegin)] == reference[2 * count + row * grid.size[0] + c];
		}
		CHECK_MSG(same, t + ": a block evaluated alone differs");
	}
}

// The grid path and the meshless path compute the same field in different
// ways (per-axis tables and running sums versus per-point evaluation): equal
// up to a documented tolerance, |difference| <= 3e-6 sqrt(total sill)
// (measured: 1.5e-6 at most on these cases).
void testGridVersusMeshless() {
	section("grid path = meshless path at the nodes, within 3e-6 sqrt(sill)");
	struct Case { const char* model; unsigned ndim; std::vector<unsigned> dims; double origin, spacing; };
	const Case cases[] = {
		{ "0.4*spherical(range=17) + 0.4*cubic(range=9) + 0.5*exponential(range=12) + 0.3*gaussian(range=8)", 1, { 1300 }, 0.0, 1.0 },
		{ "0.4*spherical(range=17) + 0.4*cubic(range=9) + 0.5*exponential(range=12) + 0.3*gaussian(range=8)", 2, { 1100, 23 }, 0.0, 1.0 },
		{ "0.4*spherical(range=17) + 0.4*cubic(range=9) + 0.5*exponential(range=12) + 0.3*gaussian(range=8)", 3, { 70, 16, 12 }, 0.0, 1.0 },
		// steep dilution lines (direct evaluation), high frequencies, far origin, non-unit spacing
		{ "0.5*spherical(range=[3,2], angle=10) + 0.5*cubic(range=2.5) + 0.5*matern(range=2, nu=0.4) + 0.5*stable(range=4, alpha=0.6)", 2, { 600, 41 }, 12345.5, 0.7 },
		{ "0.6*spherical(range=[40,25,12], angle=[30,20,10]) + 0.4*cubic(range=[30,15,8], angle=[0,30,0]) + 1*exponential(range=[40,20,10], angle=[10,0,0])", 3, { 530, 7, 5 }, -2.0e5, 1.3 },
	};
	for (const Case& c : cases) {
		const ResolvedModel model = stationary(c.model, c.ndim);
		double sill = 0.0;
		for (const ResolvedStructure& s : model.structures) sill += s.sill;
		RegularGrid grid = RegularGrid::cells(c.ndim, c.dims.data());
		for (unsigned d = 0; d < c.ndim; ++d) { grid.origin[d] = c.origin * (d + 1); grid.spacing[d] = c.spacing; }
		const size_t count = grid.nodeCount();
		std::vector<double> points(count * c.ndim);
		for (size_t id = 0; id < count; ++id) {
			size_t rest = id;
			for (unsigned d = 0; d < c.ndim; ++d) { points[id * c.ndim + d] = grid.origin[d] + grid.spacing[d] * double(rest % grid.size[d]); rest /= grid.size[d]; }
		}
		TurningBandsSettings settings;
		settings.seed = 31;
		settings.threads = hardwareThreads();
		std::vector<double> onGrid(2 * count), meshless(2 * count);
		simulateTurningBandsGrid(model, settings, grid, 0, 2, onGrid.data());
		simulateTurningBands(model, settings, points.data(), count, 0, 2, meshless.data());
		double worst = 0.0, sum2 = 0.0;
		for (size_t i = 0; i < 2 * count; ++i) { worst = std::max(worst, std::fabs(onGrid[i] - meshless[i])); sum2 += meshless[i] * meshless[i]; }
		printf("   %uD %-70.70s max |difference| %.2e (field sd %.2f)\n", c.ndim, c.model, worst, std::sqrt(sum2 / (2 * count)));
		CHECK_MSG(worst <= 3e-6 * std::sqrt(sill), std::string(c.model) + ": grid and meshless differ by " + std::to_string(worst));
	}
}

void testScattered() {
	section("meshless path: scattered points = the same points in a grid order, exactly");
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
		std::vector<double> values(3 * count), meshless(3 * count);
		simulateTurningBandsGrid(resolved, settings, RegularGrid::cells(2, dims.data()), 0, 3, values.data());
		simulateTurningBands(resolved, settings, points.data(), count, 0, 3, meshless.data());
		bool exact = true;
		double worst = 0.0, sum = 0.0;
		for (unsigned r = 0; r < 3; ++r)
			for (size_t i = 0; i < count; ++i) {
				exact &= a._data[i * 3 + r] == float(values[r * count + i]);
				worst = std::max(worst, std::fabs(double(a._data[i * 3 + r]) - meshless[r * count + i]));
				sum += a._data[i * 3 + r];
			}
		printf("   executable = engine grid path: %s; vs meshless path: max |difference| %.2e, mean %.3f\n", exact ? "exact" : "DIFFERENT", worst, sum / (3.0 * count));
		CHECK(exact);
		CHECK(worst <= 1e-5);
	}
	CHECK(run("-nb 300", testId(702)) == 0);
	// 64-bit seeds, read as sgs and fftma read them: 2^32 + 11 is not 11, and
	// "11.000000" (a floating-point 11 sent by an interface) is 11.
	CHECK(run("-nr 3 -s 4294967307 -m -2.5 -j 2", testId(709)) == 0);
	CHECK(run("-nr 3 -s 11.000000 -m -2.5 -j 2", testId(710)) == 0);
	{
		g2s::DataImage wide = g2s::DataImage::createFromFile("im_1_" + testIdText(709));
		g2s::DataImage decimal = g2s::DataImage::createFromFile("im_1_" + testIdText(710));
		CHECK(!wide.isEmpty() && !a.isEmpty() && wide.dataSize() == a.dataSize() && !std::equal(a._data, a._data + a.dataSize(), wide._data));
		CHECK(!decimal.isEmpty() && !a.isEmpty() && decimal.dataSize() == a.dataSize() && std::equal(a._data, a._data + a.dataSize(), decimal._data));
	}
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
	refused("-di " + N("tb_test_grid") + " -cov 'exponential(range=10)' -s 1.5 -id " + testIdText(711), "-s expects an integer seed");
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
	testKernelAccuracy();
	testLineProfiles();
	testRefusals();
	testDeterminism();
	testGridDeterminism();
	testScattered();
	testGridVersusMeshless();
	testCovariances();
	testGaussianity();
	testBanding();
	testExecutable(binary);
	if (withTimings) timing(binary);
	printf("%d checks, %d failures (%.1f s)\n", checks, failures, std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
	return failures == 0 ? 0 : 1;
}
