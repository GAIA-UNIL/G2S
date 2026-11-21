// Tests of the GPU back ends (include/geostat/gpu): device results against
// the CPU engines on the same seed, within the tolerances documented in
// gpu/deviceEngines.hpp, statistics of the device realisations, and the
// executables with -W_GPU against the same runs on the CPU.
//
// Without an OpenCL device (or a build without OpenCL) the suite prints
// "skipped: no OpenCL device" and succeeds. The CUDA back end is not
// exercised here (it needs an Nvidia GPU; see src/geostatCuda.cu).
//
//   gpu_test [executable directory] [timings]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "DataImage.hpp"
#include "geostat/covarianceModel.hpp"
#include "geostat/gpu/deviceEngines.hpp"
#include "geostat/gpu/openclDevice.hpp"
#include "geostat/krigingGlobal.hpp"
#include "geostat/spectralEngine.hpp"
#include "geostat/turningBands.hpp"
#include "../geostat_common/testData.hpp"

using namespace g2s::geostat;

namespace {

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

void section(const std::string& name) { currentSection = name; printf("-- %s\n", name.c_str()); }

ResolvedModel stationary(const std::string& text, unsigned ndim) {
	return CovarianceModel(ModelSpec::parse(text), Bindings(), ndim).stationaryModel();
}

unsigned threads() { return std::max(1u, std::min(8u, std::thread::hardware_concurrency())); }

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

// ------------------------------------------------------------ turning bands

std::vector<float> tbCpu(const ResolvedModel& model, const TurningBandsSettings& settings, const RegularGrid& grid, unsigned first, unsigned count) {
	std::vector<double> values(grid.nodeCount() * count);
	TurningBandsSettings s = settings;
	s.threads = threads();
	simulateTurningBandsGrid(model, s, grid, first, count, values.data());
	return std::vector<float>(values.begin(), values.end());
}

bool tbDevice(gpu::Device& device, const ResolvedModel& model, const TurningBandsSettings& settings, const RegularGrid& grid, unsigned first, unsigned count,
              std::vector<float>& out, std::string& error) {
	std::string reason;
	std::unique_ptr<gpu::TurningBandsKernel> kernel(device.turningBands(reason));
	if (!kernel) { error = reason; return false; }
	out.assign(grid.nodeCount() * count, 0.0f);
	WorkerPool pool(threads());
	const size_t nodes = grid.nodeCount(), columns = grid.size[0];
	return gpu::simulateTurningBandsGridDevice(model, settings, grid, first, count, *kernel, pool,
		[&](unsigned r, size_t rowBegin, size_t rowEnd, const float* values) {
			std::copy(values, values + (rowEnd - rowBegin) * columns, out.begin() + size_t(r) * nodes + rowBegin * columns);
		},
		[](unsigned) {}, error);
}

struct Difference { double max = 0.0; size_t large = 0; size_t count = 0; };

Difference compare(const std::vector<float>& a, const std::vector<float>& b, double tolerance) {
	Difference d;
	d.count = a.size();
	for (size_t i = 0; i < a.size(); ++i) {
		const double e = std::fabs(double(a[i]) - double(b[i]));
		if (e > tolerance) ++d.large;
		else d.max = std::max(d.max, e);
	}
	return d;
}

void testTurningBands(gpu::Device& device) {
	section("turning bands: device against the CPU grid path (same seed)");
	struct Case { const char* model; std::vector<unsigned> dims; unsigned lines; bool dilution; };
	const std::vector<Case> cases = {
		{ "1*exponential(range=20)", { 300, 200 }, 1000, false },
		{ "1*gaussian(range=[30,10], angle=30)", { 200, 150 }, 500, false },
		{ "0.2*nugget + 0.8*matern(range=15, nu=1.5)", { 257, 130 }, 700, false },
		{ "1*stable(range=10, alpha=1.5)", { 64, 48, 20 }, 300, false },
		{ "1*exponential(range=[30,20,8], angle=[20,10,5])", { 70, 45, 33 }, 400, false },
		{ "1*spherical(range=25)", { 300, 200 }, 1000, true },
		{ "1*cubic(range=[40,15], angle=-20)", { 200, 150 }, 500, true },
		{ "0.5*spherical(range=12) + 0.5*exponential(range=30)", { 50, 40, 30 }, 400, true },
		{ "1*cubic(range=[20,10,6])", { 64, 48, 20 }, 300, true },
		{ "1*spherical(range=50)", { 1500 }, 600, true },
		{ "1*exponential(range=50)", { 1500 }, 600, false },
		{ "1*spherical(range=40)", { 600, 3 }, 800, true },  // more than one column tile (512)
	};
	for (const Case& c : cases) {
		const unsigned ndim = unsigned(c.dims.size());
		const ResolvedModel model = stationary(c.model, ndim);
		TurningBandsSettings settings;
		settings.lineCount = c.lines;
		settings.seed = 1234;
		settings.mean = 1.5;
		const RegularGrid grid = RegularGrid::cells(ndim, c.dims.data());
		const std::vector<float> cpu = tbCpu(model, settings, grid, 3, 2);
		std::vector<float> gpuValues;
		std::string error;
		const bool ok = tbDevice(device, model, settings, grid, 3, 2, gpuValues, error);
		CHECK_MSG(ok, error);
		if (!ok) continue;
		const double tolerance = 2e-5 * std::sqrt(model.sill());
		const Difference d = compare(cpu, gpuValues, tolerance);
		printf("   %-52s max |gpu - cpu| %.2e (tolerance %.0e), beyond: %zu of %zu\n", c.model, d.max, tolerance, d.large, d.count);
		// Spectral lines: no exception. Dilution: segment boundaries within rounding (see deviceEngines.hpp).
		if (!c.dilution) CHECK(d.large == 0);
		else CHECK(double(d.large) <= 1e-5 * double(d.count) + 1);
		CHECK(d.max <= tolerance);
	}

	section("turning bands: device results reproducible and independent of the chunking");
	{
		const unsigned dims[3] = { 90, 70, 40 };
		const ResolvedModel model = stationary("0.1*nugget + 0.9*exponential(range=[20,12,6], angle=[30,0,0])", 3);
		TurningBandsSettings settings;
		settings.lineCount = 333;
		settings.seed = 99;
		const RegularGrid grid = RegularGrid::cells(3, dims);
		std::vector<float> a, b, c;
		std::string error;
		CHECK(tbDevice(device, model, settings, grid, 0, 3, a, error));
		CHECK(tbDevice(device, model, settings, grid, 0, 3, b, error));
		CHECK(a == b);
		// Realisation 2 alone (-ri 2) equals realisation 2 of the batch.
		CHECK(tbDevice(device, model, settings, grid, 2, 1, c, error));
		CHECK(std::equal(c.begin(), c.end(), a.begin() + 2 * grid.nodeCount()));
	}
}

// ------------------------------------------------------------ FFT-MA / spectral

namespace sp = spectral;

struct FftRun {
	std::vector<float> field, noise;
	double seconds = 0.0;
	gpu::DeviceTimes times;
};

bool fftRun(gpu::Device* device, const ResolvedModel& model, const std::vector<unsigned>& grid, const std::vector<unsigned>& extended, sp::Method method,
            unsigned first, unsigned count, bool withNoise, FftRun& run, std::string& error, const std::vector<float>* noiseIn = nullptr) {
	sp::SpectralSimulator simulator(model, grid, extended, method, threads());
	size_t cells = 1, extendedCells = 1;
	for (unsigned d : grid) cells *= d;
	for (unsigned d : extended) extendedCells *= d;
	run.field.assign(cells * count, 0.0f);
	run.noise.assign(withNoise ? extendedCells * count : 0, 0.0f);
	const sp::FieldView out{ run.field.data(), count, 1 };
	const sp::FieldView noiseOut = withNoise ? sp::FieldView{ run.noise.data(), count, 1 } : sp::FieldView();
	const sp::ConstFieldView in = noiseIn ? sp::ConstFieldView{ noiseIn->data(), count, 1 } : sp::ConstFieldView();
	const Clock::time_point t = Clock::now();
	if (!device) {
		simulator.simulate(7, first, count, 0.25, out, in, noiseOut);
		run.seconds = since(t);
		return true;
	}
	std::string reason;
	std::unique_ptr<gpu::SpectralKernel> kernel(device->spectral(reason));
	if (!kernel) { error = reason; return false; }
	gpu::SpectralLayout layout;
	gpu::toDeviceLayout(simulator.layout(), layout);
	if (!kernel->prepare(layout, simulator.amplitude().data(), method == sp::Method::FftMovingAverage, error)) return false;
	const bool ok = gpu::simulateSpectralDevice(simulator, *kernel, 7, first, count, 0.25, out, in, noiseOut, threads(), error);
	run.seconds = since(t);
	run.times = kernel->times();
	return ok;
}

void testSpectral(gpu::Device& device) {
	section("FFT-MA and spectral: device against the CPU (same seed)");
	struct Case { const char* model; std::vector<unsigned> grid, extended; };
	const std::vector<Case> cases = {
		{ "1*exponential(range=20)", { 300, 200 }, { 360, 250 } },
		{ "1*gaussian(range=[30,10], angle=30)", { 125, 98 }, { 189, 147 } },   // odd extended sizes
		{ "0.1*nugget + 0.9*spherical(range=15)", { 60, 50, 40 }, { 84, 70, 56 } },
		{ "1*matern(range=[20,10,5], nu=0.5)", { 45, 33, 21 }, { 75, 45, 35 } },
		{ "1*cubic(range=40)", { 1000 }, { 1125 } },
	};
	for (const Case& c : cases) {
		const ResolvedModel model = stationary(c.model, unsigned(c.grid.size()));
		const double tolerance = 1e-4 * std::sqrt(model.sill());
		for (sp::Method method : { sp::Method::FftMovingAverage, sp::Method::Spectral }) {
			const bool fftma = method == sp::Method::FftMovingAverage;
			FftRun cpu, gpuRun;
			std::string error;
			fftRun(nullptr, model, c.grid, c.extended, method, 2, 3, fftma, cpu, error);
			const bool ok = fftRun(&device, model, c.grid, c.extended, method, 2, 3, fftma, gpuRun, error);
			CHECK_MSG(ok, error);
			if (!ok) continue;
			const Difference d = compare(cpu.field, gpuRun.field, tolerance);
			size_t sameNoise = 0;
			for (size_t i = 0; i < cpu.noise.size(); ++i) sameNoise += cpu.noise[i] == gpuRun.noise[i];
			printf("   %-8s %-38s max |gpu - cpu| %.2e (tolerance %.0e)%s\n", sp::methodName(method), c.model, d.max, tolerance,
			       fftma ? (sameNoise == cpu.noise.size() ? ", white noise bit-identical" : (", white noise identical at " + std::to_string(sameNoise) + " of " + std::to_string(cpu.noise.size()))).c_str() : "");
			CHECK(d.large == 0);
			if (fftma) CHECK(sameNoise == cpu.noise.size());
		}
	}

	section("FFT-MA: given white noise (-ni) on the device");
	{
		const ResolvedModel model = stationary("1*exponential(range=[12,6], angle=10)", 2);
		const std::vector<unsigned> grid = { 80, 60 }, extended = { 105, 84 };
		FftRun reference, cpu, gpuRun;
		std::string error;
		fftRun(nullptr, model, grid, extended, sp::Method::FftMovingAverage, 0, 2, true, reference, error);
		for (float& v : reference.noise) v = -v; // any noise: the negated one gives the negated field
		fftRun(nullptr, model, grid, extended, sp::Method::FftMovingAverage, 0, 2, false, cpu, error, &reference.noise);
		CHECK_MSG(fftRun(&device, model, grid, extended, sp::Method::FftMovingAverage, 0, 2, false, gpuRun, error, &reference.noise), error);
		CHECK(compare(cpu.field, gpuRun.field, 1e-4).large == 0);
	}

	section("FFT: device results reproducible");
	{
		const ResolvedModel model = stationary("1*spherical(range=[15,9,4])", 3);
		FftRun a, b;
		std::string error;
		CHECK(fftRun(&device, model, { 50, 40, 30 }, { 70, 56, 40 }, sp::Method::Spectral, 0, 2, false, a, error));
		CHECK(fftRun(&device, model, { 50, 40, 30 }, { 70, 56, 40 }, sp::Method::Spectral, 0, 2, false, b, error));
		CHECK(a.field == b.field);
	}
}

void timeSpectral(gpu::Device& device) {
	section("FFT-MA / spectral timings (device path with its host work, CPU 8 threads)");
	struct Case { std::vector<unsigned> grid; unsigned R; };
	const std::vector<Case> cases = { { { 4000, 4000 }, 1 }, { { 4000, 4000 }, 16 }, { { 200, 200, 200 }, 1 }, { { 200, 200, 200 }, 16 },
	                                  { { 500, 500, 500 }, 1 }, { { 500, 500, 500 }, 4 } };
	for (const Case& c : cases) {
		const ResolvedModel model = stationary("1*exponential(range=20)", unsigned(c.grid.size()));
		const std::vector<unsigned> extended = sp::extendedGrid(c.grid, std::vector<double>(c.grid.size(), 20.0), false);
		for (sp::Method method : { sp::Method::FftMovingAverage, sp::Method::Spectral }) {
			FftRun cpu, gpuRun;
			std::string error;
			fftRun(nullptr, model, c.grid, extended, method, 0, c.R, false, cpu, error);
			const bool ok = fftRun(&device, model, c.grid, extended, method, 0, c.R, false, gpuRun, error);
			CHECK_MSG(ok, error);
			std::string size;
			for (size_t d = 0; d < c.grid.size(); ++d) size += (d ? "x" : "") + std::to_string(c.grid[d]);
			printf("   %-8s %-12s R=%-2u CPU %7.3f s  device %7.3f s (upload %.3f, kernels+FFT %.3f, download %.3f)  speed-up %.2f\n", sp::methodName(method), size.c_str(), c.R,
			       cpu.seconds, gpuRun.seconds, gpuRun.times.upload, gpuRun.times.compute, gpuRun.times.download, cpu.seconds / gpuRun.seconds);
		}
	}
}

// ------------------------------------------------------------ kriging variance

struct KrigingCase {
	std::vector<unsigned> dims;
	std::vector<size_t> dataCells, targetCells;
	std::vector<double> coordinates;
	GlobalKriging kriging;
	GridCovarianceTable table;
	std::vector<double> cpu;
	double cpuSeconds = 0.0;
};

// n data at random cells (clustered: in a few small clusters), targets every
// `step`-th cell; CPU variance as the reference.
bool krigingCase(const char* modelText, const std::vector<unsigned>& dims, size_t n, bool clustered, KrigingType type, size_t step, KrigingCase& c) {
	c.dims = dims;
	const ResolvedModel model = stationary(modelText, unsigned(dims.size()));
	size_t cells = 1;
	for (unsigned d : dims) cells *= d;
	uint64_t state = 12345;
	auto next = [&]() { state = state * 6364136223846793005ull + 1442695040888963407ull; return state >> 33; };
	std::vector<char> used(cells, 0);
	while (c.dataCells.size() < n) {
		size_t cell = size_t(next() % cells);
		if (clustered) {
			const size_t centre = size_t((next() % 5) * (cells / 5) + cells / 10);
			long cx = long(centre % dims[0]) + long(next() % 25) - 12, cy = long(centre / dims[0] % dims[1]) + long(next() % 25) - 12;
			cx = std::max(0L, std::min(long(dims[0]) - 1, cx));
			cy = std::max(0L, std::min(long(dims[1]) - 1, cy));
			cell = size_t(cx) + dims[0] * size_t(cy);
		}
		if (used[cell]) continue;
		used[cell] = 1;
		c.dataCells.push_back(cell);
	}
	for (size_t cell : c.dataCells) {
		long coordinate[kMaxDims];
		cellCoordinates(cell, dims, coordinate);
		for (size_t d = 0; d < dims.size(); ++d) c.coordinates.push_back(double(coordinate[d]));
	}
	for (size_t cell = 0; cell < cells; cell += step) c.targetCells.push_back(cell);
	GlobalKriging::Options options;
	options.type = type;
	options.nbThreads = threads();
	std::string error;
	if (!c.kriging.factorize(model, c.coordinates.data(), n, unsigned(dims.size()), options, error)) return false;
	c.table.build(model, dims, threads());
	c.cpu.assign(c.targetCells.size(), 0.0);
	const Clock::time_point t = Clock::now();
	c.kriging.variance(GridCrossCovariance(c.table, c.dataCells.data(), n, c.targetCells.data()), c.targetCells.size(), c.cpu.data());
	c.cpuSeconds = since(t);
	return true;
}

double maxDifference(const std::vector<double>& a, const std::vector<double>& b) {
	double worst = 0.0;
	for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(a[i] - b[i]));
	return worst;
}

void testKriging(gpu::Device& device) {
	section("kriging variance kernels");
	std::string reason;
	std::unique_ptr<gpu::KrigingKernel> doubles(device.kriging(reason));
	if (device.info().fp64) CHECK_MSG(doubles != nullptr, reason);
	else {
		CHECK(doubles == nullptr);
		printf("   double-precision kriging refused on this device, as it must be: %s\n", reason.c_str());
	}
	std::unique_ptr<gpu::KrigingKernel> floats(gpu::openclKrigingFloatForTests(&device, reason));
	struct Case { const char* model; size_t n; bool clustered; KrigingType type; };
	const std::vector<Case> cases = {
		{ "1*exponential(range=30)", 700, false, KrigingType::Simple },
		{ "1*exponential(range=30)", 700, false, KrigingType::Ordinary },
		{ "0.05*nugget + 0.95*spherical(range=40)", 1100, false, KrigingType::Ordinary },
		{ "1*gaussian(range=40)", 700, true, KrigingType::Ordinary },
	};
	for (const Case& k : cases) {
		KrigingCase c;
		CHECK(krigingCase(k.model, { 200, 150 }, k.n, k.clustered, k.type, 7, c));
		std::string error;
		std::vector<double> device64(c.cpu.size()), device32(c.cpu.size());
		auto run = [&](gpu::KrigingKernel& kernel, std::vector<double>& out) {
			if (kernel.doublePrecision())
				return gpu::globalKrigingVarianceDevice(c.kriging, kernel, GridCrossCovariance(c.table, c.dataCells.data(), k.n, c.targetCells.data()), &c.table,
				                                        c.dataCells.data(), c.targetCells.data(), c.targetCells.size(), out.data(), error);
			// Float test mode: the same sequence as globalKrigingVarianceDevice (which refuses float).
			const double c0 = c.kriging.model().sill();
			if (!kernel.prepare(c.kriging.factor(), k.n, c.kriging.forwardOnes(), error)) return false;
			std::vector<uint64_t> base(k.n);
			for (size_t i = 0; i < k.n; ++i) base[i] = c.table.base(c.dataCells[i]);
			if (!kernel.setTable(c.table.values(), c.table.bytes() / sizeof(double), base.data(), error)) return false;
			const size_t chunk = kernel.maxTargets();
			std::vector<double> sums(chunk), dots(chunk), columns(k.n * chunk);
			for (size_t t0 = 0; t0 < c.targetCells.size(); t0 += chunk) {
				const size_t m = std::min(chunk, c.targetCells.size() - t0);
				// Half of the chunks with host-filled columns, to test both inputs.
				std::vector<uint64_t> offsets(m);
				const bool gather = (t0 / chunk) % 2 == 0;
				for (size_t t = 0; t < m; ++t) offsets[t] = c.table.offset(c.targetCells[t0 + t]);
				if (!gather) {
					const GridCrossCovariance cross(c.table, c.dataCells.data(), k.n, c.targetCells.data());
					std::vector<double> kt(k.n);
					for (size_t t = 0; t < m; ++t) { cross(t0 + t, kt.data()); for (size_t i = 0; i < k.n; ++i) columns[i * m + t] = kt[i]; }
				}
				if (!kernel.varianceTerms(gather ? offsets.data() : nullptr, gather ? nullptr : columns.data(), m, sums.data(), dots.data(), error)) return false;
				for (size_t t = 0; t < m; ++t) {
					double v = c0 - sums[t];
					if (k.type == KrigingType::Ordinary) v += (1.0 - dots[t]) * (1.0 - dots[t]) / c.kriging.onesQuadraticForm();
					out[t0 + t] = std::max(0.0, v);
				}
			}
			return true;
		};
		if (doubles) {
			CHECK_MSG(run(*doubles, device64), error);
			const double d = maxDifference(c.cpu, device64);
			printf("   %-40s n=%-5zu %s %s double: max |gpu - cpu| %.2e\n", k.model, k.n, k.clustered ? "clustered" : "scattered", k.type == KrigingType::Simple ? "SK" : "OK", d);
			CHECK(d <= 1e-9 * c.kriging.model().sill() + 1e-6 * 1.0);
		}
		if (floats) {
			// Logic of the kernel code (the same kernels as the double ones): well-conditioned cases
			// agree with the CPU to float accuracy; the ill-conditioned one shows why float is refused.
			CHECK_MSG(run(*floats, device32), error);
			const double d = maxDifference(c.cpu, device32);
			printf("   %-40s n=%-5zu %s %s float test mode: max |gpu - cpu| %.2e (sill %.2f)\n", k.model, k.n, k.clustered ? "clustered" : "scattered",
			       k.type == KrigingType::Simple ? "SK" : "OK", d, c.kriging.model().sill());
			if (!k.clustered) CHECK(d < 2e-3 * c.kriging.model().sill());
		}
	}
}

void timeKriging(gpu::Device& device) {
	section("kriging variance timings (CPU 8 threads; device in float test mode on devices without fp64: NOT a result path)");
	std::string reason;
	std::unique_ptr<gpu::KrigingKernel> kernel(device.kriging(reason));
	if (!kernel) kernel.reset(gpu::openclKrigingFloatForTests(&device, reason));
	if (!kernel) return;
	for (size_t n : { size_t(1000), size_t(4000) }) {
		KrigingCase c;
		CHECK(krigingCase("1*exponential(range=60)", { 1000, 1000 }, n, false, KrigingType::Ordinary, 1, c));
		std::vector<double> out(c.cpu.size());
		std::string error;
		const Clock::time_point t = Clock::now();
		const bool ok = kernel->prepare(c.kriging.factor(), n, c.kriging.forwardOnes(), error);
		std::vector<uint64_t> base(n);
		for (size_t i = 0; i < n; ++i) base[i] = c.table.base(c.dataCells[i]);
		CHECK(ok && kernel->setTable(c.table.values(), c.table.bytes() / sizeof(double), base.data(), error));
		const size_t chunk = kernel->maxTargets();
		std::vector<double> sums(chunk), dots(chunk);
		std::vector<uint64_t> offsets(chunk);
		for (size_t t0 = 0; t0 < c.targetCells.size(); t0 += chunk) {
			const size_t m = std::min(chunk, c.targetCells.size() - t0);
			for (size_t i = 0; i < m; ++i) offsets[i] = c.table.offset(c.targetCells[t0 + i]);
			CHECK(kernel->varianceTerms(offsets.data(), nullptr, m, sums.data(), dots.data(), error));
		}
		const double gpuSeconds = since(t);
		printf("   n=%-5zu 1000x1000 targets: CPU %.2f s, device %s %.2f s (kernels %.2f s)\n", n, c.cpuSeconds, kernel->doublePrecision() ? "double" : "float", gpuSeconds, kernel->times().compute);
	}
}

void timeTurningBands(gpu::Device& device) {
	section("turning bands timings (device path with its host work, CPU grid path, 8 threads)");
	struct Case { const char* model; std::vector<unsigned> dims; unsigned R; };
	const std::vector<Case> cases = {
		{ "1*exponential(range=20)", { 1000, 1000 }, 1 }, { "1*exponential(range=20)", { 1000, 1000 }, 16 },
		{ "1*spherical(range=20)", { 1000, 1000 }, 1 }, { "1*spherical(range=20)", { 1000, 1000 }, 16 },
		{ "1*exponential(range=20)", { 200, 200, 200 }, 1 }, { "1*exponential(range=20)", { 200, 200, 200 }, 16 },
		{ "1*spherical(range=20)", { 200, 200, 200 }, 1 }, { "1*spherical(range=20)", { 200, 200, 200 }, 16 },
	};
	for (const Case& c : cases) {
		const unsigned ndim = unsigned(c.dims.size());
		const ResolvedModel model = stationary(c.model, ndim);
		TurningBandsSettings settings;
		settings.seed = 5;
		settings.threads = threads();
		const RegularGrid grid = RegularGrid::cells(ndim, c.dims.data());
		const size_t nodes = grid.nodeCount();
		std::vector<float> output(nodes * c.R);
		WorkerPool pool(threads());
		Clock::time_point t = Clock::now();
		simulateTurningBandsGrid(model, settings, grid, 0, c.R, pool,
			[&](unsigned r, const GridBlock& block, const double* values) {
				for (size_t row = block.rowBegin; row < block.rowEnd; ++row) {
					const double* source = values + (row - block.rowBegin) * block.width();
					float* target = output.data() + (row * grid.size[0] + block.columnBegin) * c.R + r;
					for (size_t i = 0; i < block.width(); ++i) target[i * c.R] = float(source[i]);
				}
			}, [](unsigned) {});
		const double cpu = since(t);
		std::string reason;
		std::unique_ptr<gpu::TurningBandsKernel> kernel(device.turningBands(reason));
		t = Clock::now();
		std::string error;
		const bool ok = gpu::simulateTurningBandsGridDevice(model, settings, grid, 0, c.R, *kernel, pool,
			[&](unsigned r, size_t rowBegin, size_t rowEnd, const float* values) {
				const size_t count = (rowEnd - rowBegin) * grid.size[0];
				float* target = output.data() + rowBegin * grid.size[0] * c.R + r;
				for (size_t i = 0; i < count; ++i) target[i * c.R] = values[i];
			}, [](unsigned) {}, error);
		const double gpuTime = since(t);
		CHECK_MSG(ok, error);
		const gpu::DeviceTimes times = kernel->times();
		printf("   %-26s %-14s R=%-2u CPU %7.3f s  device %7.3f s (upload %.3f, kernels %.3f, download %.3f)  speed-up %.2f\n", c.model,
		       (std::to_string(c.dims[0]) + "x" + std::to_string(c.dims[1]) + (ndim > 2 ? "x" + std::to_string(c.dims[2]) : "")).c_str(), c.R, cpu, gpuTime,
		       times.upload, times.compute, times.download, cpu / gpuTime);
	}
}

} // namespace

int main(int argc, char** argv) {
	const std::string executables = argc > 1 ? argv[1] : "";
	(void)executables;
	std::string message;
	if (!gpu::openclAvailable() || gpu::openclDeviceNames().empty()) {
		printf("skipped: no OpenCL device (%s)\n", gpu::openclAvailable() ? "no GPU found" : "built without OpenCL");
		return 0;
	}
	std::unique_ptr<gpu::Device> device(gpu::openOpenClDevice(-1, message));
	if (!device) {
		fprintf(stderr, "FAIL: the OpenCL device does not open: %s\n", message.c_str());
		return 1;
	}
	printf("device: %s %s, %llu MiB, fp64 %s, unified memory %s\n", device->info().api.c_str(), device->info().name.c_str(),
	       (unsigned long long)(device->info().globalMemory >> 20), device->info().fp64 ? "yes" : "no", device->info().unifiedMemory ? "yes" : "no");
	const Clock::time_point start = Clock::now();
	const bool timings = argc > 2 && std::string(argv[2]) == "timings";
	testTurningBands(*device);
	testSpectral(*device);
	testKriging(*device);
	if (timings) {
		timeKriging(*device);
		timeTurningBands(*device);
		timeSpectral(*device);
	}
	printf("%d checks, %d failures (%.1f s)\n", checks, failures, since(start));
	return failures == 0 ? 0 : 1;
}
