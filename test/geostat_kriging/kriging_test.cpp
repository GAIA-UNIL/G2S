// Tests of the kriging engines: dense kernels (denseSolver.hpp), global dual
// kriging (krigingGlobal.hpp), local kriging (krigingLocal.hpp), the k-d tree
// search (kdTreeSearch.hpp) and, when built, the kriging executable.
//
//   make -C test/geostat_kriging
//   KRIGING=/path/to/kriging make -C test/geostat_kriging

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <sys/stat.h>

#include "DataImage.hpp"
#include "geostat/covarianceModel.hpp"
#include "geostat/denseSolver.hpp"
#include "geostat/kdTreeSearch.hpp"
#include "geostat/kriging.hpp"
#include "geostat/krigingGlobal.hpp"
#include "geostat/krigingLocal.hpp"
#include "geostat/neighborSearch.hpp"
#include "../geostat_common/testData.hpp"

using namespace g2s::geostat;
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
#define CHECK_NEAR(a, b, tolerance) do { const double va_ = (a), vb_ = (b); \
	report(std::fabs(va_ - vb_) <= (tolerance), std::string(#a " ~ " #b ": ") + formatNumber(va_) + " vs " + formatNumber(vb_), __FILE__, __LINE__); } while (0)

void section(const std::string& name) { currentSection = name; printf("-- %s\n", name.c_str()); }

ResolvedModel stationary(const std::string& text, unsigned ndim) {
	return CovarianceModel(ModelSpec::parse(text), Bindings(), ndim).stationaryModel();
}

CovarianceModelSet stationarySet(const std::string& text, unsigned ndim) {
	return buildCovarianceModelSet({ text }, Bindings(), ndim);
}

double maxAbsDifference(const std::vector<double>& a, const std::vector<double>& b) {
	double worst = 0.0;
	for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(a[i] - b[i]));
	return worst;
}

// A 2D grid with `count` distinct random data cells.
struct GridCase {
	std::vector<unsigned> dims;
	std::vector<size_t> dataCells, targetCells;
	std::vector<double> values;       // per cell (NaN at targets)
	std::vector<double> dataValues;   // per datum
	std::vector<double> dataCoordinates, targetCoordinates;
	std::vector<int64_t> ranks;

	GridCase(std::vector<unsigned> d, size_t count, unsigned seed, double offset = 0.0) : dims(d) {
		size_t cells = 1;
		for (unsigned v : dims) cells *= v;
		std::mt19937 rng(seed);
		std::vector<size_t> all(cells);
		for (size_t i = 0; i < cells; ++i) all[i] = i;
		std::shuffle(all.begin(), all.end(), rng);
		std::vector<bool> isData(cells, false);
		for (size_t i = 0; i < count; ++i) isData[all[i]] = true;
		std::normal_distribution<double> normal(0.0, 1.0);
		values.assign(cells, std::nan(""));
		ranks.assign(cells, kRankNever);
		for (size_t cell = 0; cell < cells; ++cell) {
			long c[kMaxDims];
			cellCoordinates(cell, dims, c);
			if (isData[cell]) {
				dataCells.push_back(cell);
				values[cell] = offset + normal(rng);
				dataValues.push_back(values[cell]);
				ranks[cell] = kRankKnown;
				for (size_t k = 0; k < dims.size(); ++k) dataCoordinates.push_back(double(c[k]));
			} else {
				targetCells.push_back(cell);
				for (size_t k = 0; k < dims.size(); ++k) targetCoordinates.push_back(double(c[k]));
			}
		}
	}
	unsigned ndim() const { return unsigned(dims.size()); }
};

// Brute force: the full SK / OK system with every datum at every target.
void bruteForce(KrigingType type, const ResolvedModel& model, const GridCase& g, double mean, std::vector<double>& estimate, std::vector<double>& variance) {
	const size_t n = g.dataCells.size(), nd = g.ndim();
	estimate.assign(g.targetCells.size(), 0.0);
	variance.assign(g.targetCells.size(), 0.0);
	std::vector<double> lags(n * nd);
	KrigingWorkspace workspace;
	KrigingSolution solution;
	for (size_t t = 0; t < g.targetCells.size(); ++t) {
		for (size_t i = 0; i < n; ++i)
			for (size_t d = 0; d < nd; ++d) lags[i * nd + d] = g.dataCoordinates[i * nd + d] - g.targetCoordinates[t * nd + d];
		krige(type, model, lags.data(), n, workspace, solution);
		estimate[t] = krigingEstimate(solution, g.dataValues.data(), mean);
		variance[t] = solution.variance;
	}
}

// ============================================================ dense kernels

void testDense() {
	section("dense: blocked Cholesky and forward substitution");
	std::mt19937 rng(7);
	std::uniform_real_distribution<double> uniform(-1.0, 1.0);
	for (size_t n : { size_t(5), size_t(130), size_t(300), size_t(517) }) {
		// SPD matrix: exponential covariance of random 2D points.
		const ResolvedModel model = stationary("exponential(range=0.7)", 2);
		std::vector<double> x(2 * n);
		for (double& v : x) v = uniform(rng);
		std::vector<double> a(n * n), b(n * n);
		for (size_t i = 0; i < n; ++i)
			for (size_t j = 0; j < n; ++j) {
				const double lag[2] = { x[2 * i] - x[2 * j], x[2 * i + 1] - x[2 * j + 1] };
				a[i * n + j] = model.covariance(lag);
			}
		b = a;
		CHECK(choleskyFactor(a.data(), n, n) == 0);
		std::vector<double> c = b;
		CHECK(blockedCholesky(c.data(), n, n, 0.0, 4) == 0);
		double worst = 0.0;
		for (size_t i = 0; i < n; ++i)
			for (size_t j = 0; j <= i; ++j) worst = std::max(worst, std::fabs(a[i * n + j] - c[i * n + j]));
		CHECK_MSG(worst < 1e-10, "n=" + std::to_string(n) + " max |L - L_ref| = " + formatNumber(worst));
		std::vector<double> c1 = b;
		blockedCholesky(c1.data(), n, n, 0.0, 1);
		bool identical = true;
		for (size_t i = 0; i < n; ++i)
			for (size_t j = 0; j <= i; ++j) identical &= c1[i * n + j] == c[i * n + j];
		CHECK_MSG(identical, "blocked Cholesky must not depend on the thread count (n=" + std::to_string(n) + ")");
		// Forward substitution with 13 right-hand sides against the vector solver.
		const size_t nrhs = 13;
		std::vector<double> rhs(n * nrhs), single(n);
		for (double& v : rhs) v = uniform(rng);
		std::vector<double> y = rhs;
		blockedForwardSubstitution(c.data(), n, n, y.data(), nrhs, nrhs);
		double worstSolve = 0.0;
		for (size_t r = 0; r < nrhs; ++r) {
			for (size_t i = 0; i < n; ++i) single[i] = rhs[i * nrhs + r];
			for (size_t i = 0; i < n; ++i) single[i] = (single[i] - dotProduct(a.data() + i * n, single.data(), i)) / a[i * n + i];
			for (size_t i = 0; i < n; ++i) worstSolve = std::max(worstSolve, std::fabs(single[i] - y[i * nrhs + r]) / (1.0 + std::fabs(single[i])));
		}
		CHECK_MSG(worstSolve < 1e-9, "forward substitution n=" + std::to_string(n) + ": " + formatNumber(worstSolve));
	}
	// Non-positive-definite input is reported.
	std::vector<double> bad = { 1, 2, 2, 1 };
	CHECK(blockedCholesky(bad.data(), 2, 2, 0.0, 2) == 2);
}

// ============================================================ global engine

void testGlobal() {
	section("global: exactness, constant field, SK far from data");
	{
		GridCase g({ 40, 30 }, 60, 1);
		const ResolvedModel model = stationary("0.1*nugget + 0.9*spherical(range=[15,8], angle=20)", 2);
		for (KrigingType type : { KrigingType::Simple, KrigingType::Ordinary }) {
			GlobalKriging kriging;
			GlobalKriging::Options options;
			options.type = type;
			options.nbThreads = 3;
			std::string error;
			CHECK_MSG(kriging.factorize(model, g.dataCoordinates.data(), g.dataCells.size(), 2, options, error), error);
			GlobalKrigingDual dual;
			kriging.solve(g.dataValues.data(), 0.3, dual);
			std::vector<double> atData(g.dataCells.size()), varianceAtData(g.dataCells.size());
			kriging.estimate(PointCrossCovariance(kriging, g.dataCoordinates.data()), g.dataCells.size(), dual, atData.data(), g.dataCells.size());
			kriging.variance(PointCrossCovariance(kriging, g.dataCoordinates.data()), g.dataCells.size(), varianceAtData.data());
			CHECK_MSG(maxAbsDifference(atData, g.dataValues) < 1e-9, "exactness " + formatNumber(maxAbsDifference(atData, g.dataValues)));
			double worstVariance = 0.0;
			for (double v : varianceAtData) worstVariance = std::max(worstVariance, v);
			CHECK_MSG(worstVariance < 1e-9, "variance at data " + formatNumber(worstVariance));
		}
		// OK of a constant field is that constant everywhere (weights sum to 1).
		GlobalKriging kriging;
		GlobalKriging::Options options;
		std::string error;
		CHECK(kriging.factorize(model, g.dataCoordinates.data(), g.dataCells.size(), 2, options, error));
		std::vector<double> constant(g.dataCells.size(), 4.25), estimate(g.targetCells.size());
		GlobalKrigingDual dual;
		kriging.solve(constant.data(), 0.0, dual);
		kriging.estimate(PointCrossCovariance(kriging, g.targetCoordinates.data()), g.targetCells.size(), dual, estimate.data(), g.targetCells.size());
		double worst = 0.0;
		for (double v : estimate) worst = std::max(worst, std::fabs(v - 4.25));
		CHECK_MSG(worst < 1e-9, "OK of a constant " + formatNumber(worst));
		CHECK_NEAR(dual.constant[0], 4.25, 1e-9);
	}
	{
		// SK far from the data: the mean, with the sill as variance.
		const ResolvedModel model = stationary("2*spherical(range=5)", 1);
		const std::vector<double> x = { 0, 1, 3 }, z = { 1, -2, 0.5 };
		GlobalKriging kriging;
		GlobalKriging::Options options;
		options.type = KrigingType::Simple;
		std::string error;
		CHECK(kriging.factorize(model, x.data(), 3, 1, options, error));
		GlobalKrigingDual dual;
		kriging.solve(z.data(), 7.0, dual);
		const std::vector<double> far = { 50, -40 };
		std::vector<double> estimate(2), variance(2);
		kriging.estimate(PointCrossCovariance(kriging, far.data()), 2, dual, estimate.data(), 2);
		kriging.variance(PointCrossCovariance(kriging, far.data()), 2, variance.data());
		CHECK_NEAR(estimate[0], 7.0, 1e-12);
		CHECK_NEAR(estimate[1], 7.0, 1e-12);
		CHECK_NEAR(variance[0], 2.0, 1e-12);
		CHECK_NEAR(variance[1], 2.0, 1e-12);
	}

	section("global: direct against a brute-force full solve per target");
	{
		GridCase g({ 25, 20 }, 150, 2, 1.5);
		for (const char* text : { "0.2*nugget + 0.8*exponential(range=[12,6], angle=30)", "gaussian(range=9) + 0.01*nugget", "matern(range=10, nu=1.5)" }) {
			const ResolvedModel model = stationary(text, 2);
			for (KrigingType type : { KrigingType::Simple, KrigingType::Ordinary }) {
				const double mean = 1.5;
				std::vector<double> referenceEstimate, referenceVariance;
				bruteForce(type, model, g, mean, referenceEstimate, referenceVariance);
				GlobalKriging kriging;
				GlobalKriging::Options options;
				options.type = type;
				options.nbThreads = 4;
				std::string error;
				CHECK_MSG(kriging.factorize(model, g.dataCoordinates.data(), g.dataCells.size(), 2, options, error), error);
				GlobalKrigingDual dual;
				kriging.solve(g.dataValues.data(), mean, dual);
				std::vector<double> estimate(g.targetCells.size()), variance(g.targetCells.size()), field(g.values.size());
				kriging.estimate(PointCrossCovariance(kriging, g.targetCoordinates.data()), g.targetCells.size(), dual, estimate.data(), g.targetCells.size());
				kriging.variance(PointCrossCovariance(kriging, g.targetCoordinates.data()), g.targetCells.size(), variance.data());
				const std::string label = std::string(text) + (type == KrigingType::Simple ? " SK" : " OK");
				const double jitterScale = kriging.jitter() > 0 ? 1e3 : 1.0; // the brute force jitters per target
				CHECK_MSG(maxAbsDifference(estimate, referenceEstimate) < 1e-8 * jitterScale, label + " estimate " + formatNumber(maxAbsDifference(estimate, referenceEstimate)));
				CHECK_MSG(maxAbsDifference(variance, referenceVariance) < 1e-8 * jitterScale, label + " variance " + formatNumber(maxAbsDifference(variance, referenceVariance)));
				// Grid fast path (table + scatter, table lookup for the variance) gives the same.
				GridCovarianceTable table;
				CHECK(table.build(model, g.dims, 3));
				estimateGridFields(table, g.dataCells.data(), dual, field.data(), field.size(), 3);
				std::vector<double> fromField(g.targetCells.size()), tableVariance(g.targetCells.size());
				for (size_t t = 0; t < g.targetCells.size(); ++t) fromField[t] = field[g.targetCells[t]];
				CHECK_MSG(maxAbsDifference(fromField, estimate) < 1e-10, label + " grid fields " + formatNumber(maxAbsDifference(fromField, estimate)));
				kriging.variance(GridCrossCovariance(table, g.dataCells.data(), g.dataCells.size(), g.targetCells.data()), g.targetCells.size(), tableVariance.data());
				CHECK_MSG(maxAbsDifference(tableVariance, variance) < 1e-10, label + " table variance");
				std::vector<double> cellEstimate(g.targetCells.size());
				kriging.estimate(CellCrossCovariance(kriging, g.dims, g.targetCells.data()), g.targetCells.size(), dual, cellEstimate.data(), g.targetCells.size());
				CHECK(maxAbsDifference(cellEstimate, estimate) < 1e-12);
			}
		}
	}

	section("global: several right-hand sides = separate solves");
	{
		GridCase g({ 30, 30 }, 300, 3);
		const ResolvedModel model = stationary("0.05*nugget + 0.95*spherical(range=12)", 2);
		for (KrigingType type : { KrigingType::Simple, KrigingType::Ordinary }) {
			GlobalKriging kriging;
			GlobalKriging::Options options;
			options.type = type;
			options.nbThreads = 4;
			std::string error;
			CHECK(kriging.factorize(model, g.dataCoordinates.data(), g.dataCells.size(), 2, options, error));
			const size_t n = g.dataCells.size(), R = 37, T = g.targetCells.size();
			std::mt19937 rng(11);
			std::normal_distribution<double> normal(0.0, 1.0);
			std::vector<double> values(R * n), means(R);
			for (double& v : values) v = normal(rng);
			for (double& m : means) m = normal(rng);
			GlobalKrigingDual many;
			kriging.solve(values.data(), R, n, means.data(), many);
			std::vector<double> outMany(R * T), fields(R * g.values.size());
			kriging.estimate(PointCrossCovariance(kriging, g.targetCoordinates.data()), T, many, outMany.data(), T);
			GridCovarianceTable table;
			CHECK(table.build(model, g.dims, 4));
			estimateGridFields(table, g.dataCells.data(), many, fields.data(), g.values.size(), 4);
			double worst = 0.0, worstField = 0.0;
			for (size_t r = 0; r < R; ++r) {
				GlobalKrigingDual one;
				kriging.solve(values.data() + r * n, means[r], one);
				std::vector<double> outOne(T);
				kriging.estimate(PointCrossCovariance(kriging, g.targetCoordinates.data()), T, one, outOne.data(), T);
				for (size_t t = 0; t < T; ++t) {
					worst = std::max(worst, std::fabs(outOne[t] - outMany[r * T + t]));
					worstField = std::max(worstField, std::fabs(outOne[t] - fields[r * g.values.size() + g.targetCells[t]]));
				}
				for (size_t i = 0; i < n; ++i) worstField = std::max(worstField, std::fabs(values[r * n + i] - fields[r * g.values.size() + g.dataCells[i]]));
			}
			CHECK_MSG(worst < 1e-10, "many vs one " + formatNumber(worst));
			CHECK_MSG(worstField < 1e-9, "grid fields of many rhs (targets and data) " + formatNumber(worstField));
		}
	}

	section("global: threads, memory guard, duplicates");
	{
		GridCase g({ 60, 50 }, 700, 4);
		const ResolvedModel model = stationary("exponential(range=20)", 2);
		std::vector<std::vector<double> > estimates, variances;
		for (unsigned threads : { 1u, 3u, 8u }) {
			GlobalKriging kriging;
			GlobalKriging::Options options;
			options.nbThreads = threads;
			std::string error;
			CHECK(kriging.factorize(model, g.dataCoordinates.data(), g.dataCells.size(), 2, options, error));
			GlobalKrigingDual dual;
			kriging.solve(g.dataValues.data(), 0.0, dual);
			std::vector<double> estimate(g.targetCells.size()), variance(g.targetCells.size()), field(g.values.size());
			GridCovarianceTable table;
			table.build(model, g.dims, threads);
			estimateGridFields(table, g.dataCells.data(), dual, field.data(), field.size(), threads);
			for (size_t t = 0; t < g.targetCells.size(); ++t) estimate[t] = field[g.targetCells[t]];
			kriging.variance(GridCrossCovariance(table, g.dataCells.data(), g.dataCells.size(), g.targetCells.data()), g.targetCells.size(), variance.data());
			estimates.push_back(estimate);
			variances.push_back(variance);
		}
		CHECK_MSG(estimates[0] == estimates[1] && estimates[0] == estimates[2], "estimates must be identical for 1, 3 and 8 threads");
		CHECK_MSG(variances[0] == variances[1] && variances[0] == variances[2], "variances must be identical for 1, 3 and 8 threads");

		GlobalKriging kriging;
		GlobalKriging::Options options;
		options.maxMatrixBytes = 1000 * 1000; // 1 MB: 700 data need 3.9 MB
		std::string error;
		CHECK(!kriging.factorize(model, g.dataCoordinates.data(), g.dataCells.size(), 2, options, error));
		CHECK_MSG(error.find("above the limit") != std::string::npos, error);
		// Duplicate data: jittered, still usable.
		const std::vector<double> x = { 0, 0, 5, 5, 5, 5 };
		const std::vector<double> z = { 1, 2, 2 };
		GlobalKriging duplicate;
		CHECK(duplicate.factorize(model, x.data(), 3, 2, GlobalKriging::Options(), error));
		CHECK(duplicate.jitter() > 0);
	}
}

// ============================================================ search

void testKdTree() {
	section("k-d tree search = grid search");
	GridCase g({ 41, 37 }, 120, 5);
	const unsigned nd = 2;
	auto position = [&](size_t cell, double* x) { long c[kMaxDims]; cellCoordinates(cell, g.dims, c); for (unsigned d = 0; d < nd; ++d) x[d] = double(c[d]); };
	struct Setting { std::vector<unsigned> kernel; unsigned perSector; size_t count; bool mask; };
	const std::vector<Setting> settings = {
		{ { 83, 75 }, 0, 20, false }, { { 21, 15 }, 0, 12, false }, { { 31, 31 }, 3, 16, false }, { { 83, 75 }, 4, 10, false }, { { 25, 25 }, 0, 15, true }
	};
	for (const Setting& s : settings) {
		std::vector<float> weights;
		size_t kernelCells = 1;
		for (unsigned d : s.kernel) kernelCells *= d;
		if (s.mask) {
			// Disc-shaped kernel: weight 0 outside radius 10.
			weights.assign(kernelCells, 1.f);
			for (size_t k = 0; k < kernelCells; ++k) {
				const double dx = double(k % s.kernel[0]) - s.kernel[0] / 2, dy = double(k / s.kernel[0]) - s.kernel[1] / 2;
				if (dx * dx + dy * dy > 100) weights[k] = 0.f;
			}
		}
		const std::vector<std::vector<int> > offsets = kernelOffsets(s.kernel, s.mask ? weights.data() : nullptr, 1);
		GridSearchOptions gridOptions;
		gridOptions.maxPerSector = s.perSector;
		GridNeighborSearch grid(g.dims, offsets, g.ranks.data(), gridOptions);
		KdTreeSearchOptions treeOptions;
		treeOptions.maxPerSector = s.perSector;
		for (unsigned d : s.kernel) treeOptions.halfExtent.push_back(double(d / 2));
		const std::vector<unsigned> kernel = s.kernel;
		const std::vector<float> w = weights;
		const bool mask = s.mask;
		treeOptions.accept = [kernel, w, mask](const double* lag) {
			size_t flat = 0, stride = 1;
			for (size_t d = 0; d < kernel.size(); ++d) {
				const long p = long(kernel[d] / 2) + std::lround(lag[d]);
				if (p < 0 || p >= long(kernel[d])) return false;
				flat += size_t(p) * stride;
				stride *= kernel[d];
			}
			return !mask || w[flat] != 0.f;
		};
		KdTreeNeighborSearch tree(nd, g.values.size(), position, g.ranks.data(), 0, treeOptions);
		CHECK(tree.indexedCount() == g.dataCells.size());
		size_t mismatches = 0;
		NeighborSet a, b;
		for (size_t cell = 0; cell < g.values.size(); ++cell) {
			grid.find(cell, 0, s.count, a);
			tree.find(cell, 0, s.count, b);
			if (a.ids != b.ids || a.lags != b.lags) ++mismatches;
		}
		CHECK_MSG(mismatches == 0, std::to_string(mismatches) + " cells differ (kernel " + std::to_string(s.kernel[0]) + ", per sector " + std::to_string(s.perSector) + ")");
	}
	// Unrestricted: the nearest data whatever the distance.
	KdTreeNeighborSearch tree(nd, g.values.size(), position, g.ranks.data(), 0);
	NeighborSet all;
	tree.find(g.targetCells[0], 0, 1000, all);
	CHECK(all.size() == g.dataCells.size());
	bool sorted = true;
	for (size_t i = 1; i < all.size(); ++i) {
		const double* p = all.lag(i - 1);
		const double* q = all.lag(i);
		sorted &= p[0] * p[0] + p[1] * p[1] <= q[0] * q[0] + q[1] * q[1];
	}
	CHECK(sorted);
}

// ============================================================ local engine

void testLocal() {
	section("local: neighbourhood with every datum = direct");
	GridCase g({ 30, 25 }, 80, 6, -0.7);
	const std::string text = "0.1*nugget + 0.9*spherical(range=[20,10], angle=-25)";
	const CovarianceModelSet models = stationarySet(text, 2);
	const ResolvedModel& model = models.stationaryModel();
	auto position = [&](size_t cell, double* x) { long c[kMaxDims]; cellCoordinates(cell, g.dims, c); x[0] = double(c[0]); x[1] = double(c[1]); };
	KdTreeNeighborSearch search(2, g.values.size(), position, g.ranks.data(), 0);
	for (KrigingType type : { KrigingType::Simple, KrigingType::Ordinary }) {
		const double mean = -0.7;
		LocalKrigingOptions options;
		options.type = type;
		options.maxNeighbors = 1000;
		options.nbThreads = 4;
		std::vector<double> estimate(g.targetCells.size()), variance(g.targetCells.size());
		localKriging(search, models, g.targetCells.data(), g.targetCells.size(), g.values.data(), [mean](size_t) { return mean; }, options, estimate.data(), variance.data());
		GlobalKriging kriging;
		GlobalKriging::Options globalOptions;
		globalOptions.type = type;
		std::string error;
		CHECK(kriging.factorize(model, g.dataCoordinates.data(), g.dataCells.size(), 2, globalOptions, error));
		GlobalKrigingDual dual;
		kriging.solve(g.dataValues.data(), mean, dual);
		std::vector<double> direct(g.targetCells.size()), directVariance(g.targetCells.size());
		kriging.estimate(PointCrossCovariance(kriging, g.targetCoordinates.data()), g.targetCells.size(), dual, direct.data(), g.targetCells.size());
		kriging.variance(PointCrossCovariance(kriging, g.targetCoordinates.data()), g.targetCells.size(), directVariance.data());
		CHECK_MSG(maxAbsDifference(estimate, direct) < 1e-8, "estimate " + formatNumber(maxAbsDifference(estimate, direct)));
		CHECK_MSG(maxAbsDifference(variance, directVariance) < 1e-8, "variance " + formatNumber(maxAbsDifference(variance, directVariance)));
	}

	section("local: constant field, SK mean far from data, no neighbour");
	{
		std::vector<double> constant = g.values;
		for (size_t cell : g.dataCells) constant[cell] = 3.5;
		LocalKrigingOptions options;
		options.maxNeighbors = 12;
		std::vector<double> estimate(g.targetCells.size()), variance(g.targetCells.size());
		localKriging(search, models, g.targetCells.data(), g.targetCells.size(), constant.data(), [](size_t) { return 0.0; }, options, estimate.data(), variance.data());
		double worst = 0.0;
		for (double v : estimate) worst = std::max(worst, std::fabs(v - 3.5));
		CHECK_MSG(worst < 1e-9, "OK of a constant " + formatNumber(worst));
		// SK with a short range: targets far from every datum get the mean map value and the sill.
		const CovarianceModelSet shortRange = stationarySet("1.5*spherical(range=2)", 2);
		options.type = KrigingType::Simple;
		auto meanMap = [](size_t cell) { return 0.01 * double(cell); };
		localKriging(search, shortRange, g.targetCells.data(), g.targetCells.size(), g.values.data(), meanMap, options, estimate.data(), variance.data());
		size_t farTargets = 0, wrong = 0;
		for (size_t t = 0; t < g.targetCells.size(); ++t) {
			NeighborSet nearest;
			search.find(g.targetCells[t], 0, 1, nearest);
			const double* lag = nearest.lag(0);
			if (lag[0] * lag[0] + lag[1] * lag[1] < 4.0) continue;
			++farTargets;
			if (std::fabs(estimate[t] - meanMap(g.targetCells[t])) > 1e-12 || std::fabs(variance[t] - 1.5) > 1e-12) ++wrong;
		}
		CHECK(farTargets > 0);
		CHECK_MSG(wrong == 0, std::to_string(wrong) + " far targets differ from (mean, sill)");
		// A window without data: mean and sill, counted.
		KdTreeSearchOptions windowOptions;
		windowOptions.halfExtent = { 0.5, 0.5 };
		KdTreeNeighborSearch tiny(2, g.values.size(), position, g.ranks.data(), 0, windowOptions);
		options.type = KrigingType::Ordinary;
		const LocalKrigingStats stats = localKriging(tiny, models, g.targetCells.data(), g.targetCells.size(), g.values.data(), [](size_t) { return 9.0; }, options, estimate.data(), variance.data());
		CHECK(stats.noNeighbor == g.targetCells.size());
		CHECK(estimate[0] == 9.0 && std::fabs(variance[0] - model.sill()) < 1e-15);
	}

	section("local: threads and sharing");
	{
		// Regular sampling (every 5th cell): many targets share a lag pattern.
		std::vector<unsigned> dims = { 60, 45 };
		size_t cells = 60 * 45;
		std::vector<double> values(cells, std::nan(""));
		std::vector<int64_t> ranks(cells, kRankNever);
		std::vector<size_t> targets;
		std::mt19937 rng(9);
		std::normal_distribution<double> normal(0.0, 1.0);
		for (size_t cell = 0; cell < cells; ++cell) {
			if ((cell % 60) % 5 == 0 && (cell / 60) % 5 == 0) { values[cell] = normal(rng); ranks[cell] = kRankKnown; }
			else targets.push_back(cell);
		}
		const std::vector<std::vector<int> > offsets = kernelOffsets({ 31, 31 });
		GridNeighborSearch grid(dims, offsets, ranks.data());
		const CovarianceModelSet set = stationarySet("exponential(range=15)", 2);
		std::vector<std::vector<double> > estimates, variances;
		std::vector<LocalKrigingStats> stats;
		for (unsigned threads : { 1u, 2u, 7u }) {
			for (size_t capacity : { size_t(8192), size_t(0) }) {
				LocalKrigingOptions options;
				options.maxNeighbors = 16;
				options.nbThreads = threads;
				options.cacheCapacity = capacity;
				std::vector<double> estimate(targets.size()), variance(targets.size());
				stats.push_back(localKriging(grid, set, targets.data(), targets.size(), values.data(), [](size_t) { return 0.0; }, options, estimate.data(), variance.data()));
				estimates.push_back(estimate);
				variances.push_back(variance);
			}
		}
		bool identical = true;
		for (size_t i = 1; i < estimates.size(); ++i) identical &= estimates[i] == estimates[0] && variances[i] == variances[0];
		CHECK_MSG(identical, "local kriging must give identical results for 1, 2, 7 threads, with and without sharing");
		CHECK(stats[0].shared > 0 && stats[0].solved + stats[0].shared == targets.size());
		CHECK(stats[1].shared == 0 && stats[1].solved == targets.size());
		printf("   sharing ratio on a regular 5x5 sampling: %.3f (%zu systems for %zu targets)\n", stats[0].sharingRatio(), stats[0].solved, targets.size());
	}

	section("local: spatially varying range (non-stationary)");
	{
		GridCase h({ 35, 30 }, 90, 8);
		std::vector<float> ranges(h.values.size());
		for (size_t cell = 0; cell < ranges.size(); ++cell) ranges[cell] = float(5.0 + 0.6 * double(cell % 35));
		Bindings bindings;
		bindings.setField("L", ranges);
		const CovarianceModelSet set = buildCovarianceModelSet({ "0.05*nugget + 0.95*exponential(range=L)" }, bindings, 2);
		CHECK(!set.isStationary());
		auto pos = [&](size_t cell, double* x) { long c[kMaxDims]; cellCoordinates(cell, h.dims, c); x[0] = double(c[0]); x[1] = double(c[1]); };
		KdTreeNeighborSearch tree(2, h.values.size(), pos, h.ranks.data(), 0);
		LocalKrigingOptions options;
		options.maxNeighbors = 10;
		options.nbThreads = 3;
		std::vector<double> estimate(h.targetCells.size()), variance(h.targetCells.size());
		localKriging(tree, set, h.targetCells.data(), h.targetCells.size(), h.values.data(), [](size_t) { return 0.0; }, options, estimate.data(), variance.data());
		// Reference: krige with the model resolved at each target.
		double worst = 0.0;
		NeighborSet neighbors;
		KrigingWorkspace workspace;
		KrigingSolution solution;
		ResolvedModel scratch;
		for (size_t t = 0; t < h.targetCells.size(); ++t) {
			tree.find(h.targetCells[t], 0, 10, neighbors);
			krige(KrigingType::Ordinary, set.at(h.targetCells[t], scratch), neighbors.lags.data(), neighbors.size(), workspace, solution);
			double z = 0.0;
			for (size_t i = 0; i < neighbors.size(); ++i) z += solution.weights[i] * h.values[neighbors.ids[i]];
			worst = std::max(worst, std::fabs(z - estimate[t]) + std::fabs(solution.variance - variance[t]));
		}
		CHECK_MSG(worst < 1e-12, "non-stationary local vs per-target krige " + formatNumber(worst));
	}
}

// ============================================================ executable

bool writeImage(const std::string& name, const std::vector<unsigned>& dims, const std::vector<float>& values) {
	std::vector<unsigned> d = dims;
	g2s::DataImage image(unsigned(d.size()), d.data(), 1);
	for (size_t i = 0; i < values.size(); ++i) image._data[i] = values[i];
	image.write(name);
	return !g2s::DataImage::createFromFile(name).isEmpty();
}

void testExecutable(const std::string& binary) {
	section("kriging executable");
	FILE* probe = fopen(binary.c_str(), "r");
	if (!probe) { printf("   skipped: %s not built\n", binary.c_str()); return; }
	fclose(probe);
	GridCase g({ 50, 40 }, 120, 10, 2.0);
	std::vector<float> di(g.values.begin(), g.values.end()), ranges(g.values.size());
	for (size_t cell = 0; cell < ranges.size(); ++cell) ranges[cell] = float(6.0 + 0.3 * double(cell % 50));
	CHECK(writeImage(N("kriging_test_di"), g.dims, di));
	CHECK(writeImage(N("kriging_test_range"), g.dims, ranges));
	const std::string model = "0.1*nugget + 0.9*spherical(range=[18,9], angle=15)";
	auto run = [&](const std::string& options, int id) {
		const std::string command = binary + " -di " + N("kriging_test_di") + " -cov '" + model + "' " + options + " -id " + std::to_string(id) + " -o " + N("kriging_test_out") + " 2>/dev/null";
		return std::system(command.c_str());
	};
	auto load = [](const std::string& name) { return g2s::DataImage::createFromFile(name); };
	auto targetsOf = [&](const g2s::DataImage& image) {
		std::vector<double> out;
		for (size_t cell : g.targetCells) out.push_back(image._data[cell]);
		return out;
	};

	// direct vs local with every datum, OK, with variance; data cells keep their value, variance 0 there.
	CHECK(run("-method direct -var -j 4", testId(100)) == 0);
	CHECK(run("-method local -n 500 -var -j 3", testId(101)) == 0);
	g2s::DataImage directEstimate = load("im_1_" + testIdText(100)), directVariance = load("im_2_" + testIdText(100));
	g2s::DataImage localEstimate = load("im_1_" + testIdText(101)), localVariance = load("im_2_" + testIdText(101));
	CHECK(!directEstimate.isEmpty() && !directVariance.isEmpty() && !localEstimate.isEmpty() && !localVariance.isEmpty());
	if (!directEstimate.isEmpty() && !localEstimate.isEmpty() && !directVariance.isEmpty() && !localVariance.isEmpty()) {
		CHECK(directEstimate._dims == g.dims);
		CHECK_MSG(maxAbsDifference(targetsOf(directEstimate), targetsOf(localEstimate)) < 1e-5, "direct vs local estimate");
		CHECK_MSG(maxAbsDifference(targetsOf(directVariance), targetsOf(localVariance)) < 1e-5, "direct vs local variance");
		bool kept = true;
		for (size_t cell : g.dataCells) kept &= directEstimate._data[cell] == float(g.values[cell]) && directVariance._data[cell] == 0.f;
		CHECK(kept);
		// Against the engine-level brute force (float output).
		std::vector<double> reference, referenceVariance;
		bruteForce(KrigingType::Ordinary, stationary(model, 2), g, 0.0, reference, referenceVariance);
		CHECK(maxAbsDifference(targetsOf(directEstimate), reference) < 1e-5);
		CHECK(maxAbsDifference(targetsOf(directVariance), referenceVariance) < 1e-5);
	}
	// SK with a negative mean and auto (direct: stationary, few data); no -var: no output 2.
	CHECK(run("-kt sk -m -1.25", testId(102)) == 0);
	g2s::DataImage sk = load("im_1_" + testIdText(102));
	CHECK(!sk.isEmpty());
	if (!sk.isEmpty()) {
		std::vector<double> reference, referenceVariance;
		bruteForce(KrigingType::Simple, stationary(model, 2), g, -1.25, reference, referenceVariance);
		CHECK_MSG(maxAbsDifference(targetsOf(sk), reference) < 1e-5, "SK -m -1.25: " + formatNumber(maxAbsDifference(targetsOf(sk), reference)));
	}
	CHECK(load("im_2_" + testIdText(102)).isEmpty());
	// Non-stationary: local runs, direct is refused.
	const std::string nonStationary = binary + " -di " + N("kriging_test_di") + " -cov 'exponential(range=L)' -cvar L " + N("kriging_test_range");
	CHECK(std::system((nonStationary + " -method local -n 20 -var -id " + testIdText(103) + " -o " + N("kriging_test_out") + " 2>/dev/null").c_str()) == 0);
	CHECK(!load("im_2_" + testIdText(103)).isEmpty());
	CHECK(std::system((nonStationary + " -method direct -id " + testIdText(104) + " -o " + N("kriging_test_out") + " 2>/dev/null").c_str()) != 0);
	CHECK(std::system((nonStationary + " -id " + testIdText(105) + " -o " + N("kriging_test_out") + " 2>/dev/null").c_str()) == 0); // auto -> local
	// Kernel + octant search, grid and k-d tree give the same result (dense data -> grid; -ks large -> tree).
	CHECK(run("-method local -n 12 -os 3 -ks 21", testId(106)) == 0);
	g2s::DataImage windowed = load("im_1_" + testIdText(106));
	CHECK(!windowed.isEmpty());
	if (!windowed.isEmpty()) {
		auto pos = [&](size_t cell, double* x) { long c[kMaxDims]; cellCoordinates(cell, g.dims, c); x[0] = double(c[0]); x[1] = double(c[1]); };
		KdTreeSearchOptions treeOptions;
		treeOptions.halfExtent = { 10, 10 };
		treeOptions.maxPerSector = 3;
		KdTreeNeighborSearch tree(2, g.values.size(), pos, g.ranks.data(), 0, treeOptions);
		LocalKrigingOptions options;
		options.maxNeighbors = 12;
		std::vector<double> estimate(g.targetCells.size());
		localKriging(tree, stationarySet(model, 2), g.targetCells.data(), g.targetCells.size(), g.values.data(), [](size_t) { return 0.0; }, options, estimate.data(), nullptr);
		CHECK_MSG(maxAbsDifference(targetsOf(windowed), estimate) < 1e-5, "-ks 21 -os 3 (grid search) vs k-d tree engine");
	}
	// Memory guard.
	CHECK(run("-method direct -maxmem 0.00001", testId(107)) != 0);
	// Invalid arguments.
	CHECK(run("-kt uk", testId(108)) != 0);
	CHECK(run("-method global", testId(109)) != 0);
}

} // namespace

int main(int argc, char** argv) {
	const std::string kriging = argc > 1 ? argv[1] : "../../build/c++-build/kriging";
	testDense();
	testGlobal();
	testKdTree();
	testLocal();
	testExecutable(kriging);
	printf("%d checks, %d failures\n", checks, failures);
	return failures == 0 ? 0 : 1;
}
