// Tests of the FFT simulation engine (include/geostat/spectralEngine.hpp) and
// of the fftma / spectral executable.
//
//   make -C test/geostat_fft                 # build and run
//   make -C test/geostat_fft bench           # timings (1000x1000 and 200^3, R = 1 and 16)
//   GEOSTAT_FFT_VERBOSE=1 make -C test/geostat_fft   # print every statistical estimate
//
// Statistical checks compare averages over R realisations with the model; the
// tolerance of each is 4.5 standard errors (estimated from the spread of the
// per-realisation values) plus 0.005 sill for float rounding and clipping. A
// check also fails when its standard error exceeds 0.05 sill (a test that
// cannot detect a 10% error is not a test).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "DataImage.hpp"
#include "geostat/covarianceModel.hpp"
#include "geostat/spectralEngine.hpp"
#include "../geostat_common/testData.hpp"

using namespace g2s::geostat;
using namespace g2s::geostat::spectral;
using geostat_test::testId;
using geostat_test::testIdText;

namespace {

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

void section(const std::string& name) { currentSection = name; printf("-- %s\n", name.c_str()); fflush(stdout); }

std::string joinDims(const std::vector<unsigned>& dims) {
	std::string text;
	for (unsigned d : dims) text += (text.empty() ? "" : "x") + std::to_string(d);
	return text;
}

ResolvedModel stationary(const std::string& text, unsigned ndim) {
	return CovarianceModel(ModelSpec::parse(text), Bindings(), ndim).stationaryModel();
}

std::vector<unsigned> defaultExtended(const ResolvedModel& model, const std::vector<unsigned>& grid, bool periodic) {
	std::vector<double> range = model.practicalRange();
	for (double& r : range) r = std::ceil(r);
	return extendedGrid(grid, range, periodic);
}

// R realisations, realisation-major (field r at out[r * cells]).
std::vector<float> simulateMany(const ResolvedModel& model, const std::vector<unsigned>& grid, const std::vector<unsigned>& extended,
                                Method method, unsigned R, uint64_t seed, unsigned threads, double mean = 0.0, SpectrumReport* spectrum = nullptr) {
	size_t cells = 1;
	for (unsigned g : grid) cells *= g;
	std::vector<float> out(cells * R);
	SpectralSimulator simulator(model, grid, extended, method, threads);
	if (spectrum) *spectrum = simulator.report();
	simulator.simulate(seed, 0, R, mean, { out.data(), 1, cells });
	return out;
}

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

// Experimental covariance at integer lag h (known mean) of one field.
double experimentalCovariance(const float* field, const std::vector<unsigned>& grid, const int* lag, double mean) {
	const unsigned ndim = unsigned(grid.size());
	size_t cells = 1;
	for (unsigned g : grid) cells *= g;
	double sum = 0.0;
	size_t count = 0;
	std::vector<long> x(ndim);
	for (size_t i = 0; i < cells; ++i) {
		size_t r = i, j = 0, stride = 1;
		bool inside = true;
		for (unsigned d = 0; d < ndim; ++d) {
			x[d] = long(r % grid[d]);
			r /= grid[d];
			const long y = x[d] + lag[d];
			if (y < 0 || y >= long(grid[d])) { inside = false; break; }
			j += size_t(y) * stride;
			stride *= grid[d];
		}
		if (!inside) continue;
		sum += (double(field[i]) - mean) * (double(field[j]) - mean);
		++count;
	}
	return sum / double(count);
}

bool statOk(const Estimate& e, double expected, double sill, std::string& text) {
	char buffer[256];
	const double tolerance = 4.5 * e.se + 0.005 * sill;
	snprintf(buffer, sizeof(buffer), "estimate %.4f (se %.4f) vs model %.4f, tolerance %.4f", e.mean, e.se, expected, tolerance);
	text = buffer;
	if (getenv("GEOSTAT_FFT_VERBOSE")) printf("      %s\n", buffer);
	return std::fabs(e.mean - expected) <= tolerance && e.se <= 0.05 * sill;
}

// Mean, variance and covariance at `lags` over R realisations against the model.
void checkStatistics(const std::string& modelText, const std::vector<unsigned>& grid, const std::vector<std::vector<int> >& lags,
                     Method method, unsigned R, double mean) {
	const unsigned ndim = unsigned(grid.size());
	const ResolvedModel model = stationary(modelText, ndim);
	SpectrumReport spectrum;
	const std::vector<unsigned> extended = defaultExtended(model, grid, false);
	const std::vector<float> fields = simulateMany(model, grid, extended, method, R, 12345, 8, mean, &spectrum);
	size_t cells = 1;
	for (unsigned g : grid) cells *= g;
	const double sill = model.sill();
	printf("   %s %s on %s (extended", methodName(method), modelText.c_str(), joinDims(grid).c_str());
	for (unsigned e : extended) printf(" %u", e);
	printf("), clipped %.2e\n", spectrum.clippedFraction);
	std::string text;
	// Mean.
	std::vector<double> means(R);
	for (unsigned r = 0; r < R; ++r) {
		double s = 0.0;
		for (size_t i = 0; i < cells; ++i) s += fields[r * cells + i];
		means[r] = s / double(cells);
	}
	Estimate e = summarise(means);
	CHECK_MSG(statOk(e, mean, sill, text), modelText + " mean: " + text);
	// Variance and covariance (lag 0 first).
	std::vector<std::vector<int> > all = lags;
	all.insert(all.begin(), std::vector<int>(ndim, 0));
	for (const std::vector<int>& lag : all) {
		std::vector<double> values(R);
		for (unsigned r = 0; r < R; ++r) values[r] = experimentalCovariance(fields.data() + r * cells, grid, lag.data(), mean);
		e = summarise(values);
		std::vector<double> h(lag.begin(), lag.end());
		const double expected = model.covariance(h.data());
		std::string lagText;
		for (int v : lag) lagText += (lagText.empty() ? "" : ",") + std::to_string(v);
		CHECK_MSG(statOk(e, expected, sill, text), std::string(methodName(method)) + " " + modelText + " lag (" + lagText + "): " + text);
	}
}

void testStatistics() {
	section("statistics: mean, variance, covariance against the model");
	const std::vector<unsigned> grid2 = { 96, 80 };
	const std::vector<std::vector<int> > lags2 = { { 4, 0 }, { 0, 4 }, { 10, 0 }, { 0, 10 }, { 7, 7 }, { 7, -7 }, { 15, 5 }, { -5, 15 }, { 25, 0 } };
	const char* models2[] = {
		"1*spherical(range=30)",
		"2*exponential(range=24)",
		"1*gaussian(range=20)",
		"0.2*nugget + 0.8*spherical(range=[36,12], angle=30)",
		"1.5*exponential(range=[30,10], angle=-50)",
		"1*gaussian(range=[28,9], angle=120)",
	};
	for (const char* text : models2) {
		checkStatistics(text, grid2, lags2, Method::FftMovingAverage, 300, 0.0);
		checkStatistics(text, grid2, lags2, Method::Spectral, 300, 1.5);
	}
	// Short ranges on a large grid: standard errors near 0.2% of the sill, so
	// the normalisation (variance = sill) is checked to about 1.5%.
	checkStatistics("1*exponential(range=4)", { 256, 256 }, { { 1, 0 }, { 2, 2 } }, Method::FftMovingAverage, 40, 0.0);
	checkStatistics("1*exponential(range=4)", { 256, 256 }, { { 1, 0 }, { 2, 2 } }, Method::Spectral, 40, 0.0);
	checkStatistics("3*spherical(range=[6,3,4], angle=45)", { 64, 64, 64 }, { { 2, 0, 0 }, { 1, -1, 1 } }, Method::Spectral, 20, 0.0);
	const std::vector<unsigned> grid3 = { 30, 26, 22 };
	const std::vector<std::vector<int> > lags3 = { { 3, 0, 0 }, { 0, 3, 0 }, { 0, 0, 3 }, { 4, -4, 0 }, { 3, 0, 4 }, { 0, -3, 3 }, { 8, 2, 1 } };
	checkStatistics("1*exponential(range=[16,10,6], angle=[30,15,10])", grid3, lags3, Method::FftMovingAverage, 150, 0.0);
	checkStatistics("1*spherical(range=[14,9,7], angle=[-20,10,40])", grid3, lags3, Method::Spectral, 150, 0.0);
	checkStatistics("1*gaussian(range=10)", grid3, lags3, Method::Spectral, 150, -1.0);
	// 1D.
	checkStatistics("1*exponential(range=40)", { 600 }, { { 5 }, { 15 }, { 40 }, { 80 } }, Method::FftMovingAverage, 300, 0.0);
	checkStatistics("1*gaussian(range=30)", { 600 }, { { 5 }, { 15 }, { 40 } }, Method::Spectral, 300, 0.0);
}

void testPeriodicity() {
	section("edges: -csim wraps, the default buffer does not");
	const std::vector<unsigned> grid = { 64, 48 };
	const ResolvedModel model = stationary("1*exponential(range=20)", 2);
	const unsigned R = 400;
	const size_t cells = 64 * 48;
	std::string text;
	for (Method method : { Method::FftMovingAverage, Method::Spectral }) {
		const std::vector<float> periodic = simulateMany(model, grid, grid, method, R, 7, 4);
		const std::vector<float> buffered = simulateMany(model, grid, defaultExtended(model, grid, false), method, R, 7, 4);
		// Opposite edges along axis 0 (lag 63 = -1 on the torus) and axis 1 (lag 47).
		for (const std::vector<int>& lag : { std::vector<int>{ 63, 0 }, std::vector<int>{ 0, 47 } }) {
			std::vector<double> p(R), b(R);
			for (unsigned r = 0; r < R; ++r) {
				p[r] = experimentalCovariance(periodic.data() + r * cells, grid, lag.data(), 0.0);
				b[r] = experimentalCovariance(buffered.data() + r * cells, grid, lag.data(), 0.0);
			}
			const double wrapped[2] = { lag[0] ? 1.0 : 0.0, lag[1] ? 1.0 : 0.0 };
			const double straight[2] = { double(lag[0]), double(lag[1]) };
			const std::string which = std::string(methodName(method)) + " lag (" + std::to_string(lag[0]) + "," + std::to_string(lag[1]) + ")";
			CHECK_MSG(statOk(summarise(p), model.covariance(wrapped), 1.0, text), which + " with -csim should be C(1): " + text);
			CHECK_MSG(statOk(summarise(b), model.covariance(straight), 1.0, text), which + " with the buffer should be C(h): " + text);
		}
	}
}

void testNoise() {
	section("fftma noise: given noise is deterministic, returned unchanged, local effect");
	const std::vector<unsigned> grid = { 120, 100 };
	const ResolvedModel model = stationary("1*gaussian(range=20)", 2);
	const std::vector<unsigned> extended = defaultExtended(model, grid, false);
	const GridLayout L(grid, extended);
	const size_t cells = 120 * 100, R = 2;
	SpectralSimulator simulator(model, grid, extended, Method::FftMovingAverage, 4);
	std::vector<float> first(cells * R), second(cells * R), noise(L.cells * R), noiseBack(L.cells * R);
	simulator.simulate(99, 0, R, 0.0, { first.data(), 1, cells }, ConstFieldView(), { noise.data(), 1, L.cells });
	simulator.simulate(1234, 0, R, 0.0, { second.data(), 1, cells }, { noise.data(), 1, L.cells }, { noiseBack.data(), 1, L.cells });
	CHECK(first == second);       // the seed is irrelevant when the noise is given
	CHECK(noise == noiseBack);    // the noise comes back unchanged
	double sum = 0.0, sum2 = 0.0;
	for (float v : noise) { sum += v; sum2 += double(v) * v; }
	const double n = double(noise.size());
	// 5 standard errors: 1/sqrt(n) for the mean, sqrt(2/n) for the variance.
	CHECK_MSG(std::fabs(sum / n) < 5.0 / std::sqrt(n) && std::fabs(sum2 / n - 1.0) < 5.0 * std::sqrt(2.0 / n),
	          "white noise mean " + formatNumber(sum / n) + " variance " + formatNumber(sum2 / n));

	// Perturb one noise cell (60, 50): the field changes as the kernel g centred there.
	std::vector<float> perturbed(noise.begin(), noise.begin() + long(L.cells));
	const size_t centre = 50 * extended[0] + 60;
	perturbed[centre] += 1.0f;
	std::vector<float> base(cells), moved(cells);
	simulator.simulate(0, 0, 1, 0.0, { base.data(), 1, cells }, { noise.data(), 1, L.cells });
	simulator.simulate(0, 0, 1, 0.0, { moved.data(), 1, cells }, { perturbed.data(), 1, L.cells });
	// Gaussian model: g(h) = sqrt(c(0)) (12/(pi a^2))^(1/2) exp(-6 h^2/a^2) in 2D, a = 20.
	const double a = 20.0, pi = 3.14159265358979323846;
	const double peak = std::sqrt(12.0 / (pi * a * a));
	double worstFar = 0.0, worstShape = 0.0;
	for (size_t y = 0; y < 100; ++y) {
		for (size_t x = 0; x < 120; ++x) {
			const double diff = double(moved[y * 120 + x]) - double(base[y * 120 + x]);
			const double h2 = double((long(x) - 60) * (long(x) - 60) + (long(y) - 50) * (long(y) - 50));
			const double expected = peak * std::exp(-6.0 * h2 / (a * a));
			worstShape = std::max(worstShape, std::fabs(diff - expected));
			if (h2 > 1.5 * 1.5 * a * a) worstFar = std::max(worstFar, std::fabs(diff));
		}
	}
	CHECK_MSG(worstShape < 0.01 * peak, "perturbation = kernel, worst error " + formatNumber(worstShape) + " (peak " + formatNumber(peak) + ")");
	CHECK_MSG(worstFar < 1e-4, "beyond 1.5 ranges the field does not change: " + formatNumber(worstFar));

	// Spherical 3D: g is supported in a ball of diameter = range (the spherical
	// covariance is the self-convolution of a ball indicator).
	{
		const std::vector<unsigned> grid3 = { 40, 40, 40 };
		const ResolvedModel sph = stationary("1*spherical(range=16)", 3);
		const std::vector<unsigned> ext3 = defaultExtended(sph, grid3, false);
		const GridLayout L3(grid3, ext3);
		SpectralSimulator sim3(sph, grid3, ext3, Method::FftMovingAverage, 4);
		const size_t cells3 = 40 * 40 * 40;
		std::vector<float> n3(L3.cells), f0(cells3), f1(cells3);
		sim3.simulate(5, 0, 1, 0.0, { f0.data(), 1, cells3 }, ConstFieldView(), { n3.data(), 1, L3.cells });
		n3[(20 * ext3[1] + 20) * ext3[0] + 20] += 1.0f;
		sim3.simulate(5, 0, 1, 0.0, { f1.data(), 1, cells3 }, { n3.data(), 1, L3.cells });
		double inside = 0.0, outside = 0.0;
		for (size_t i = 0; i < cells3; ++i) {
			const long x = long(i % 40) - 20, y = long(i / 40 % 40) - 20, z = long(i / 1600) - 20;
			const double h = std::sqrt(double(x * x + y * y + z * z));
			const double diff = std::fabs(double(f1[i]) - double(f0[i]));
			if (h <= 8.0) inside = std::max(inside, diff);
			else if (h > 10.0) outside = std::max(outside, diff);
		}
		CHECK_MSG(inside > 0.05 && outside < 0.02 * inside, "spherical 3D kernel: max change inside radius 8 " + formatNumber(inside) + ", beyond 10 " + formatNumber(outside));
	}
}

// One determinism case: extended grid (odd, prime and non-power-of-two sizes
// included) and output grid.
struct DeterminismCase { std::string model; std::vector<unsigned> grid, extended; };

std::vector<DeterminismCase> determinismCases() {
	return {
		{ "0.1*nugget + 0.9*exponential(range=40)", { 600 }, { 640 } },
		{ "1*spherical(range=30)", { 900 }, { 1009 } },                              // prime
		{ "1*gaussian(range=80)", { 3000 }, { 3150 } },
		{ "0.1*nugget + 0.9*spherical(range=[40,15], angle=25)", { 150, 90 }, { 192, 108 } },
		{ "1*exponential(range=[20,8], angle=35)", { 60, 40 }, { 63, 45 } },          // odd
		{ "1*gaussian(range=[12,30], angle=-20)", { 101, 97 }, { 101, 97 } },          // primes (-csim)
		{ "1*spherical(range=[10,7,5], angle=[30,15,10])", { 30, 26, 22 }, { 45, 40, 30 } },
		{ "1*exponential(range=[6,9,4])", { 17, 13, 11 }, { 17, 13, 11 } },           // primes (-csim)
		{ "1*gaussian(range=[8,6,7])", { 40, 1, 21 }, { 48, 1, 25 } },                // axis of size 1
	};
}

void testDeterminism() {
	section("determinism: threads 1..8, workers 1..R, single runs, batches, -ri (1D, 2D, 3D)");
	const unsigned R = 5;
	for (const DeterminismCase& c : determinismCases()) {
		const ResolvedModel model = stationary(c.model, unsigned(c.grid.size()));
		size_t cells = 1;
		for (unsigned g : c.grid) cells *= g;
		// The amplitude (double-precision spectrum) does not depend on the threads.
		std::vector<float> amplitude;
		for (unsigned threads = 1; threads <= 8; ++threads) {
			SpectralSimulator simulator(model, c.grid, c.extended, Method::Spectral, threads);
			if (threads == 1) amplitude = simulator.amplitude();
			CHECK_MSG(simulator.amplitude() == amplitude, "amplitude on " + joinDims(c.extended) + " differs with " + std::to_string(threads) + " threads");
		}
		for (Method method : { Method::FftMovingAverage, Method::Spectral }) {
			const std::string what = std::string(methodName(method)) + " on " + joinDims(c.extended) + " ";
			// Reference: one thread, one realisation at a time.
			const std::vector<float> reference = simulateMany(model, c.grid, c.extended, method, R, 42, 1);
			int mismatches = 0;
			std::string first;
			auto expect = [&](bool same, const std::string& configuration) {
				if (same) return;
				if (mismatches++ == 0) first = configuration;
			};
			for (unsigned threads = 1; threads <= 8; ++threads) {
				SpectralSimulator simulator(model, c.grid, c.extended, method, threads);
				// Every worker count: workers = 1 is one realisation at a time with
				// all the threads in its transforms, workers = R one thread each.
				for (unsigned workers = 0; workers <= R; ++workers) {
					std::vector<float> out(cells * R);
					simulator.realisationWorkers = workers;
					simulator.simulate(42, 0, R, 0.0, { out.data(), 1, cells });
					expect(out == reference, std::to_string(threads) + " threads, " + (workers ? std::to_string(workers) : std::string("auto")) + " workers");
				}
				// Each realisation alone (-nr 1 -ri r) and a batch starting at r = 2.
				simulator.realisationWorkers = 0;
				for (unsigned r = 0; r < R; ++r) {
					std::vector<float> single(cells);
					simulator.simulate(42, r, 1, 0.0, { single.data(), 1, cells });
					expect(std::equal(single.begin(), single.end(), reference.begin() + long(r * cells)),
					       "realisation " + std::to_string(r) + " alone with " + std::to_string(threads) + " threads");
				}
				std::vector<float> tail(cells * (R - 2));
				simulator.simulate(42, 2, R - 2, 0.0, { tail.data(), 1, cells });
				expect(std::equal(tail.begin(), tail.end(), reference.begin() + long(2 * cells)), "batch from realisation 2 with " + std::to_string(threads) + " threads");
			}
			CHECK_MSG(mismatches == 0, what + "differs from 1 thread in " + std::to_string(mismatches) + " configurations, first: " + first);
		}
	}
	// Interleaved output (DataImage layout) holds the same values; different seeds differ.
	const DeterminismCase c = determinismCases()[3];
	const ResolvedModel model = stationary(c.model, 2);
	const size_t cells = 150 * 90;
	for (Method method : { Method::FftMovingAverage, Method::Spectral }) {
		const std::vector<float> reference = simulateMany(model, c.grid, c.extended, method, R, 42, 1);
		SpectralSimulator simulator(model, c.grid, c.extended, method, 4);
		std::vector<float> interleaved(cells * R);
		simulator.simulate(42, 0, R, 0.0, { interleaved.data(), R, 1 });
		bool same = true;
		for (size_t i = 0; i < cells; ++i)
			for (unsigned r = 0; r < R; ++r) same &= interleaved[i * R + r] == reference[r * cells + i];
		CHECK(same);
		CHECK(simulateMany(model, c.grid, c.extended, method, 1, 43, 1) != std::vector<float>(reference.begin(), reference.begin() + long(cells)));
	}
}

// The separable transforms against FFTW's multi-dimensional plans (accuracy)
// and against themselves over 1 to 8 threads (bitwise).
template <class Real, class Plan, class MakePlan, class Execute, class Destroy>
void checkSeparable(const std::vector<unsigned>& extended, MakePlan makePlan, Execute execute, Destroy destroy, double tolerance) {
	const GridLayout L(extended, extended);
	const std::vector<int> n = L.fftwDims();
	AlignedBuffer<Real> input(L.realCount), buffer(L.realCount), direct(L.realCount), first(L.realCount);
	uint64_t state = 12345;
	for (size_t row = 0; row < L.rows; ++row)
		for (size_t k = 0; k < L.rowPitch; ++k) {
			state = state * 6364136223846793005ull + 1442695040888963407ull;
			input.data()[row * L.rowPitch + k] = k < L.extended[0] ? Real(double(state >> 11) / 9007199254740992.0 - 0.5) : Real(0);
		}
	const std::string what = std::string(sizeof(Real) == 4 ? "float" : "double") + " " + joinDims(extended);
	const SeparableFft<Real> fft(L, true, true, buffer.data());
	// Forward.
	std::memcpy(direct.data(), input.data(), sizeof(Real) * L.realCount);
	Plan plan = makePlan(int(L.ndim), n.data(), direct.data(), true);
	execute(plan);
	destroy(plan);
	double worst = 0.0, scale = 0.0;
	for (unsigned threads = 1; threads <= 8; ++threads) {
		std::memcpy(buffer.data(), input.data(), sizeof(Real) * L.realCount);
		fft.forward(buffer.data(), threads);
		if (threads == 1) {
			std::memcpy(first.data(), buffer.data(), sizeof(Real) * L.realCount);
			for (size_t i = 0; i < 2 * L.halfCount; ++i) {
				worst = std::max(worst, std::fabs(double(buffer.data()[i]) - double(direct.data()[i])));
				scale = std::max(scale, std::fabs(double(direct.data()[i])));
			}
		}
		CHECK_MSG(std::memcmp(buffer.data(), first.data(), sizeof(Real) * 2 * L.halfCount) == 0, what + " forward differs with " + std::to_string(threads) + " threads");
	}
	CHECK_MSG(worst <= tolerance * scale, what + " forward: largest difference with FFTW " + formatNumber(worst) + " (scale " + formatNumber(scale) + ")");
	// Backward (from the forward result).
	std::memcpy(direct.data(), first.data(), sizeof(Real) * L.realCount);
	plan = makePlan(int(L.ndim), n.data(), direct.data(), false);
	execute(plan);
	destroy(plan);
	AlignedBuffer<Real> backwardFirst(L.realCount);
	worst = 0.0; scale = 0.0;
	for (unsigned threads = 1; threads <= 8; ++threads) {
		std::memcpy(buffer.data(), first.data(), sizeof(Real) * L.realCount);
		fft.backward(buffer.data(), threads);
		bool same = true;
		for (size_t row = 0; row < L.rows; ++row)
			for (size_t k = 0; k < L.extended[0]; ++k) {
				const size_t i = row * L.rowPitch + k;
				if (threads == 1) {
					backwardFirst.data()[i] = buffer.data()[i];
					worst = std::max(worst, std::fabs(double(buffer.data()[i]) - double(direct.data()[i])));
					scale = std::max(scale, std::fabs(double(direct.data()[i])));
				}
				same &= std::memcmp(&buffer.data()[i], &backwardFirst.data()[i], sizeof(Real)) == 0;
			}
		CHECK_MSG(same, what + " backward differs with " + std::to_string(threads) + " threads");
	}
	CHECK_MSG(worst <= tolerance * scale, what + " backward: largest difference with FFTW " + formatNumber(worst) + " (scale " + formatNumber(scale) + ")");
}

void testSeparableFft() {
	section("separable transforms: = FFTW multi-dimensional plans, bitwise over 1..8 threads");
	const std::vector<std::vector<unsigned> > sizes = { { 1 }, { 2 }, { 640 }, { 1009 }, { 192, 108 }, { 63, 45 }, { 64, 1 }, { 1, 50 },
	                                                    { 101, 97 }, { 45, 40, 30 }, { 17, 13, 11 }, { 48, 1, 25 }, { 32, 36, 40 } };
	for (const std::vector<unsigned>& extended : sizes) {
		checkSeparable<float, fftwf_plan>(extended,
			[](int rank, const int* n, float* data, bool forward) {
				fftwf_plan_with_nthreads(1);
				return forward ? fftwf_plan_dft_r2c(rank, n, data, reinterpret_cast<fftwf_complex*>(data), FFTW_ESTIMATE)
				               : fftwf_plan_dft_c2r(rank, n, reinterpret_cast<fftwf_complex*>(data), data, FFTW_ESTIMATE);
			},
			[](fftwf_plan p) { fftwf_execute(p); }, [](fftwf_plan p) { fftwf_destroy_plan(p); }, 2e-5);
		checkSeparable<double, fftw_plan>(extended,
			[](int rank, const int* n, double* data, bool forward) {
				fftw_plan_with_nthreads(1);
				return forward ? fftw_plan_dft_r2c(rank, n, data, reinterpret_cast<fftw_complex*>(data), FFTW_ESTIMATE)
				               : fftw_plan_dft_c2r(rank, n, reinterpret_cast<fftw_complex*>(data), data, FFTW_ESTIMATE);
			},
			[](fftw_plan p) { fftw_execute(p); }, [](fftw_plan p) { fftw_destroy_plan(p); }, 1e-13);
	}
}

void testClipping() {
	section("clipped energy");
	SpectrumReport spectrum;
	auto clipped = [&](const std::string& text, const std::vector<unsigned>& grid, const std::vector<unsigned>& extended) {
		const ResolvedModel model = stationary(text, unsigned(grid.size()));
		SpectralSimulator simulator(model, grid, extended, Method::Spectral, 4);
		spectrum = simulator.report();
		printf("   %-45s extended %-10s clipped %.3e, sill %.6f, simulated variance %.6f\n", text.c_str(),
		       joinDims(extended).c_str(), spectrum.clippedFraction, spectrum.sill, spectrum.effectiveSill);
		CHECK_MSG(std::fabs(spectrum.sill - model.sill()) < 1e-9 * model.sill() + 1e-12 || spectrum.clippedFraction > 0,
		          "sum of the spectrum / M = " + formatNumber(spectrum.sill) + " vs c(0) = " + formatNumber(model.sill()));
		return spectrum.clippedFraction;
	};
	// Exactly zero: a pure nugget has a flat spectrum.
	CHECK(clipped("1*nugget", { 64, 64 }, { 64, 64 }) == 0.0);
	// Convex decreasing covariances in 1D (Polya type) have a nonnegative
	// circulant embedding: only rounding can be negative.
	CHECK(clipped("1*exponential(range=50)", { 200 }, { 250 }) < 1e-12);
	CHECK(clipped("1*spherical(range=80)", { 300 }, { 384 }) < 1e-12);
	CHECK(clipped("1*exponential(range=50)", { 200 }, { 200 }) < 1e-12);
	// Spherical with a buffer of its range: nonnegative in 2D too.
	CHECK(clipped("1*spherical(range=20)", { 64, 64 }, { 84, 84 }) < 1e-12);
	// A gaussian with a long range and a small buffer is the classic hard case.
	const double hard = clipped("1*gaussian(range=60)", { 64, 64 }, { 70, 70 });
	CHECK(hard > 1e-3);
	CHECK(spectrum.effectiveSill > 1.0);
	// The default buffer cures it for a moderate range.
	const ResolvedModel g = stationary("1*gaussian(range=20)", 2);
	CHECK(clipped("1*gaussian(range=20)", { 64, 64 }, defaultExtended(g, { 64, 64 }, false)) < 1e-6);
}

void testSizes() {
	section("grid sizes");
	CHECK(isFftFriendly(1) && isFftFriendly(2) && isFftFriendly(210) && isFftFriendly(1029) && !isFftFriendly(11) && !isFftFriendly(1031));
	CHECK(nextFftFriendly(1031) == 1050 && nextFftFriendly(97) == 98 && nextFftFriendly(1) == 1);
	CHECK((extendedGrid({ 100, 1 }, { 30, 30 }, false) == std::vector<unsigned>{ 135, 1 }));
	CHECK((extendedGrid({ 101, 37 }, { 0, 0 }, true) == std::vector<unsigned>{ 101, 37 }));
	CHECK((extendedGrid({ 100, 100 }, { 0, 0 }, false) == std::vector<unsigned>{ 100, 100 }));
	// Odd extended sizes, slices of a 3D model and the Nyquist symmetrisation.
	const ResolvedModel model = stationary("1*exponential(range=[20,8], angle=35)", 2);
	for (const std::vector<unsigned>& ext : { std::vector<unsigned>{ 63, 45 }, std::vector<unsigned>{ 64, 45 }, std::vector<unsigned>{ 63, 46 } }) {
		const std::vector<unsigned> grid = { 40, 30 };
		const std::vector<float> a = simulateMany(model, grid, ext, Method::FftMovingAverage, 200, 3, 4);
		const std::vector<float> b = simulateMany(model, grid, ext, Method::Spectral, 200, 3, 4);
		for (const std::vector<float>* f : { &a, &b }) {
			std::vector<double> v(200), c(200);
			const int lag0[2] = { 0, 0 }, lag1[2] = { 3, -3 };
			for (unsigned r = 0; r < 200; ++r) {
				v[r] = experimentalCovariance(f->data() + r * 1200, grid, lag0, 0.0);
				c[r] = experimentalCovariance(f->data() + r * 1200, grid, lag1, 0.0);
			}
			SpectralSimulator probe(model, grid, ext, Method::Spectral, 1);
			// The circulant covariance (clipping aside) is c(0) = 1 and C(3,-3) inside the window.
			std::string text;
			const double h[2] = { 3, -3 };
			CHECK_MSG(statOk(summarise(v), probe.report().effectiveSill, 1.0, text), "variance on " + joinDims(ext) + ": " + text);
			CHECK_MSG(statOk(summarise(c), model.covariance(h), 1.0, text), "C(3,-3) on " + joinDims(ext) + ": " + text);
		}
	}
}

// ============================================================ executable

bool writeImage(const std::string& name, const std::vector<unsigned>& dims, const std::vector<float>& values, unsigned variables = 1) {
	std::vector<unsigned> d = dims;
	g2s::DataImage image(unsigned(d.size()), d.data(), variables);
	for (size_t i = 0; i < values.size(); ++i) image._data[i] = values[i];
	image.write(name);
	return !g2s::DataImage::createFromFile(name).isEmpty();
}

bool sameData(g2s::DataImage& a, g2s::DataImage& b) {
	return !a.isEmpty() && !b.isEmpty() && a._dims == b._dims && a._nbVariable == b._nbVariable &&
	       std::memcmp(a._data, b._data, sizeof(float) * a.dataSize()) == 0;
}

void testExecutable(const std::string& binary) {
	section("fftma / spectral executable");
	FILE* probe = fopen(binary.c_str(), "r");
	if (!probe) { printf("   skipped: %s not built\n", binary.c_str()); return; }
	fclose(probe);
	const std::string spectralBinary = binary.substr(0, binary.find_last_of('/') + 1) + "spectral";
	std::vector<float> nan(70 * 50, std::nanf(""));
	CHECK(writeImage(N("geostat_fft_di"), { 70, 50 }, nan));
	const std::string model = "'0.1*nugget + 0.9*spherical(range=[30,12], angle=20)'";
	auto run = [&](const std::string& exe, const std::string& args, int id) {
		const std::string command = exe + " -di " + N("geostat_fft_di") + " -cov " + model + " " + args + " -id " + std::to_string(id) + " -o " + N("geostat_fft_out") + " 2>/dev/null";
		return std::system(command.c_str());
	};
	auto load = [](int output, int id) { return g2s::DataImage::createFromFile("im_" + std::to_string(output) + "_" + std::to_string(id)); };

	// R realisations as R variables; thread count and -ri invariance (-fr: its former name, an alias).
	CHECK(run(binary, "-nr 4 -s 11 -j 1 -on", testId(500)) == 0);
	g2s::DataImage batch = load(1, testId(500)), batchNoise = load(2, testId(500));
	CHECK(!batch.isEmpty() && batch._dims == (std::vector<unsigned>{ 70, 50 }) && batch._nbVariable == 4);
	CHECK(!batchNoise.isEmpty() && batchNoise._nbVariable == 4 && batchNoise._dims[0] >= 70 + 29 && batchNoise._dims[1] >= 50 + 16);
	CHECK(run(binary, "-nr 4 -s 11 -j 8", testId(501)) == 0);
	{ g2s::DataImage other = load(1, testId(501)); CHECK(sameData(batch, other)); }
	CHECK(run(binary, "-nr 1 -s 11 -ri 2 -j 3", testId(502)) == 0);
	g2s::DataImage third = load(1, testId(502));
	bool same = !third.isEmpty() && !batch.isEmpty();
	for (size_t i = 0; same && i < 70 * 50; ++i) same = third._data[i] == batch._data[i * 4 + 2];
	CHECK(same);
	CHECK(run(binary, "-nr 1 -s 11 -fr 2 -j 3", testId(517)) == 0);
	{ g2s::DataImage alias = load(1, testId(517)); CHECK(sameData(third, alias)); }
	// -j 1..8 with -nr 3 (batch) and -nr 1 -ri 1 (one realisation, all threads in its
	// transforms), on sizes where threaded FFTW plans round differently (-csim: the
	// grid is the extended grid).
	{
		int id = 520;
		const std::vector<std::vector<unsigned> > grids = { { 640 }, { 192, 108 }, { 17, 13, 11 } };
		for (const std::vector<unsigned>& grid : grids) {
			size_t cells = 1;
			for (unsigned g : grid) cells *= g;
			const std::string name = N("geostat_fft_dj" + std::to_string(grid.size()));
			CHECK(writeImage(name, grid, std::vector<float>(cells, std::nanf(""))));
			const std::string cov = grid.size() == 1 ? "'0.1*nugget + 0.9*exponential(range=60)'" : "'1*spherical(range=" + std::to_string(grid[0] / 3) + ")'";
			for (const std::string method : { "fftma", "spectral" }) {
				auto runJ = [&](const std::string& args, int runId) {
					const std::string command = binary + " -di " + name + " -cov " + cov + " -csim -method " + method + " -s 3 " + args +
					                            " -id " + std::to_string(runId) + " -o " + N("geostat_fft_out") + " 2>/dev/null";
					return std::system(command.c_str());
				};
				const int reference = testId(id++);
				CHECK(runJ("-nr 3 -j 1", reference) == 0);
				g2s::DataImage batchRef = load(1, reference);
				int mismatches = 0;
				for (unsigned j = 2; j <= 8; ++j) {
					const int batchId = testId(id++), singleId = testId(id++);
					const bool ran = runJ("-nr 3 -j " + std::to_string(j), batchId) == 0 && runJ("-nr 1 -ri 1 -j " + std::to_string(j), singleId) == 0;
					g2s::DataImage batchJ = load(1, batchId), singleJ = load(1, singleId);
					bool same = ran && sameData(batchRef, batchJ) && !singleJ.isEmpty() && singleJ.dataSize() == cells;
					for (size_t i = 0; same && i < cells; ++i) same = std::memcmp(&singleJ._data[i], &batchRef._data[i * 3 + 1], sizeof(float)) == 0;
					mismatches += same ? 0 : 1;
				}
				CHECK_MSG(mismatches == 0, method + " on " + joinDims(grid) + ": -j 2..8 differs from -j 1 for " + std::to_string(mismatches) + " thread counts");
			}
		}
	}
	// Given noise: same result, noise returned unchanged; -nr taken from -ni.
	CHECK(run(binary, "-ni im_2_" + testIdText(500) + " -s 999 -on -j 4", testId(503)) == 0);
	{ g2s::DataImage other = load(1, testId(503)); CHECK(sameData(batch, other)); }
	{ g2s::DataImage other = load(2, testId(503)); CHECK(sameData(batchNoise, other)); }
	// The same noise as one variable with the realisations as an extra, innermost
	// dimension (how the interfaces upload an (ny, nx, R) array without -dt).
	if (!batchNoise.isEmpty()) {
		std::vector<unsigned> stacked = batchNoise._dims;
		stacked.insert(stacked.begin(), batchNoise._nbVariable);
		CHECK(writeImage(N("geostat_fft_stacked_noise"), stacked, std::vector<float>(batchNoise._data, batchNoise._data + batchNoise.dataSize())));
		CHECK(run(binary, "-ni " + N("geostat_fft_stacked_noise") + " -s 999 -j 4", testId(518)) == 0);
		{ g2s::DataImage other = load(1, testId(518)); CHECK(sameData(batch, other)); }
		CHECK(run(binary, "-nr 3 -ni " + N("geostat_fft_stacked_noise"), testId(519)) != 0); // 4 realisations in -ni
	}
	// Mean (also negative), spectral entry name and -method.
	CHECK(run(spectralBinary, "-nr 2 -s 5 -m -2.5", testId(504)) == 0);
	CHECK(run(binary, "-method spectral -nr 2 -s 5 -m -2.5", testId(505)) == 0);
	g2s::DataImage spectral = load(1, testId(504));
	{ g2s::DataImage other = load(1, testId(505)); CHECK(sameData(spectral, other)); }
	CHECK(!sameData(spectral, batch));
	if (!spectral.isEmpty()) {
		double sum = 0.0;
		for (size_t i = 0; i < spectral.dataSize(); ++i) sum += spectral._data[i];
		CHECK(std::fabs(sum / double(spectral.dataSize()) + 2.5) < 0.6);
	}
	// -csim keeps the grid; -buf per axis.
	CHECK(run(binary, "-csim -s 1 -on", testId(506)) == 0);
	CHECK(load(2, testId(506))._dims == (std::vector<unsigned>{ 70, 50 }));
	CHECK(run(binary, "-buf 5 0 -s 1 -on", testId(507)) == 0);
	CHECK(load(2, testId(507))._dims == (std::vector<unsigned>{ 75, 50 }));
	// 1D and 3D grids.
	CHECK(writeImage(N("geostat_fft_di1"), { 500 }, std::vector<float>(500, std::nanf(""))));
	CHECK(std::system((binary + " -di " + N("geostat_fft_di1") + " -cov 'exponential(range=30)' -nr 3 -s 1 -id " + testIdText(508) + " -o " + N("geostat_fft_out") + " 2>/dev/null").c_str()) == 0);
	CHECK(load(1, testId(508))._nbVariable == 3);
	CHECK(writeImage(N("geostat_fft_di3"), { 20, 18, 16 }, std::vector<float>(20 * 18 * 16, std::nanf(""))));
	CHECK(std::system((spectralBinary + " -di " + N("geostat_fft_di3") + " -cov 'gaussian(range=[8,6,4], angle=[10,20,30])' -s 1 -id " + testIdText(509) + " -o " + N("geostat_fft_out") + " 2>/dev/null").c_str()) == 0);
	CHECK(load(1, testId(509))._dims == (std::vector<unsigned>{ 20, 18, 16 }));
	// A stationary map(...) model works.
	std::vector<float> table(21 * 21, 0.f);
	for (int y = -10; y <= 10; ++y)
		for (int x = -10; x <= 10; ++x) table[size_t((y + 10) * 21 + x + 10)] = float(std::max(0.0, 1.0 - std::sqrt(double(x * x + y * y)) / 10.0));
	CHECK(writeImage(N("geostat_fft_map"), { 21, 21 }, table));
	CHECK(std::system((binary + " -di " + N("geostat_fft_di") + " -cov 'map(M)' -cvar M " + N("geostat_fft_map") + " -s 1 -id " + testIdText(510) + " -o " + N("geostat_fft_out") + " 2>/dev/null").c_str()) == 0);
	// Several maps of different sizes, one larger than the grid along axis 0.
	std::vector<float> wide(101 * 11, 0.f);
	for (int y = -5; y <= 5; ++y)
		for (int x = -50; x <= 50; ++x) wide[size_t((y + 5) * 101 + x + 50)] = float(std::max(0.0, 1.0 - std::fabs(x / 50.0) - std::fabs(y / 5.0)));
	CHECK(writeImage(N("geostat_fft_map2"), { 101, 11 }, wide));
	CHECK(std::system((binary + " -di " + N("geostat_fft_di") + " -cov 'map(M) + 0.5*map(K)' -cvar M " + N("geostat_fft_map") + " -cvar K " + N("geostat_fft_map2") + " -s 1 -id " + testIdText(516) + " -o " + N("geostat_fft_out") + " 2>/dev/null").c_str()) == 0);
	{
		g2s::DataImage one = load(1, testId(510)), two = load(1, testId(516));
		CHECK(two._dims == (std::vector<unsigned>{ 70, 50 }) && !sameData(one, two));
	}
	// Errors: non-stationary model, noise with spectral, wrong noise size, bad method.
	std::vector<float> ranges(70 * 50);
	for (size_t i = 0; i < ranges.size(); ++i) ranges[i] = float(10 + i % 7);
	CHECK(writeImage(N("geostat_fft_range"), { 70, 50 }, ranges));
	CHECK(std::system((binary + " -di " + N("geostat_fft_di") + " -cov 'exponential(range=L)' -cvar L " + N("geostat_fft_range") + " -id " + testIdText(511) + " -o " + N("geostat_fft_out") + " 2>/dev/null").c_str()) != 0);
	CHECK(run(spectralBinary, "-on", testId(512)) != 0);
	CHECK(writeImage(N("geostat_fft_small"), { 60, 50 }, std::vector<float>(3000, 0.f)));
	CHECK(run(binary, "-ni " + N("geostat_fft_small"), testId(513)) != 0);
	CHECK(run(binary, "-method kriging", testId(514)) != 0);
	CHECK(run(binary, "-nr 2 -ni im_2_" + testIdText(500), testId(515)) != 0); // 4 noise variables
}

// ============================================================ timings

void bench(unsigned threads) {
	const ResolvedModel model2 = stationary("1*spherical(range=50)", 2);
	const ResolvedModel model3 = stationary("1*spherical(range=50)", 3);
	struct Case { const ResolvedModel* model; std::vector<unsigned> grid; };
	const Case cases[] = { { &model2, { 1000, 1000 } }, { &model3, { 200, 200, 200 } } };
	for (const Case& c : cases) {
		const std::vector<unsigned> extended = defaultExtended(*c.model, c.grid, false);
		size_t cells = 1;
		for (unsigned g : c.grid) cells *= g;
		for (Method method : { Method::FftMovingAverage, Method::Spectral }) {
			const auto t0 = std::chrono::steady_clock::now();
			SpectralSimulator simulator(*c.model, c.grid, extended, method, threads);
			const double spectrumTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
			for (unsigned R : { 1u, 16u }) {
				std::vector<float> out(cells * R);
				for (unsigned workers : { 0u, 1u }) {
					if (R == 1 && workers == 1) continue;
					simulator.realisationWorkers = workers;
					const auto t1 = std::chrono::steady_clock::now();
					if (getenv("BENCH_CONTIGUOUS")) simulator.simulate(1, 0, R, 0.0, { out.data(), 1, cells });
					else simulator.simulate(1, 0, R, 0.0, { out.data(), R, 1 });
					const double time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
					printf("%-8s %-12s -> %-12s threads %u R %2u workers %-4s: spectrum %6.3f s, realisations %7.3f s (%.3f s each)\n",
					       methodName(method), joinDims(c.grid).c_str(), joinDims(extended).c_str(), threads, R,
					       workers == 0 ? "auto" : "1", spectrumTime, time, time / R);
					fflush(stdout);
				}
			}
		}
	}
}

} // namespace

int main(int argc, char** argv) {
	if (argc > 1 && std::string(argv[1]) == "bench") {
		bench(argc > 2 ? unsigned(atoi(argv[2])) : 8u);
		return 0;
	}
	const std::string binary = argc > 1 ? argv[1] : "../../build/c++-build/fftma";
	const auto begin = std::chrono::steady_clock::now();
	testSizes();
	testClipping();
	testSeparableFft();
	testDeterminism();
	testNoise();
	testPeriodicity();
	testStatistics();
	testExecutable(binary);
	const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
	printf("%d checks, %d failures (%.1f s)\n", checks, failures, seconds);
	return failures == 0 ? 0 : 1;
}
