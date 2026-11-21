// Tests of the SGS engine (include/geostat/sgsEngine.hpp, weightTable.hpp,
// gridAddressing.hpp) and of the sgs executable.
//
//   make -C test/geostat_sgs                     # build and run
//   make -C test/geostat_sgs SGS=/path/to/sgs    # other sgs binary
//
// Statistical tolerances (unit sill, 2 paths x 32 realisations of a 128 x 128
// grid, ranges 20-40 cells, so a few thousand independent values per
// estimate): mean within 0.08, variance within 0.12 of the sill and the
// experimental covariance within 0.10 of the model at every tested lag. The
// sampling standard deviation of each estimate is about 0.02-0.04, so these
// are 2.5-4 sigma bounds that also leave room for the small bias of a limited
// neighbourhood.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "DataImage.hpp"
#include "geostat/covarianceModel.hpp"
#include "geostat/gridAddressing.hpp"
#include "geostat/neighborSearch.hpp"
#include "geostat/sgsEngine.hpp"
#include "geostat/weightTable.hpp"

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
#define CHECK_NEAR(a, b, tolerance) do { const double va_ = (a), vb_ = (b); \
	report(std::fabs(va_ - vb_) <= (tolerance), std::string(#a " ~ " #b ": ") + formatNumber(va_) + " vs " + formatNumber(vb_), __FILE__, __LINE__); } while (0)

void section(const std::string& name) { currentSection = name; printf("-- %s\n", name.c_str()); }

// ------------------------------------------------------------ grid helpers

struct GridCase {
	std::vector<unsigned> dims;
	std::vector<bool> known;
	CovarianceModelSet models;
	KrigingType type = KrigingType::Simple;
	size_t neighbors = 32;
	unsigned kernel = 0;   // kernel size per axis; 0 = 2 * practical range + 1
	bool circular = false;
	unsigned perSector = 0;
	bool share = true;
	unsigned threads = 1;

	size_t nodes() const { size_t n = 1; for (unsigned d : dims) n *= d; return n; }
	std::vector<std::vector<int> > offsets() const {
		std::vector<unsigned> k(dims.size());
		const std::vector<double> range = models.maxPracticalRange();
		for (size_t d = 0; d < dims.size(); ++d) {
			unsigned size = kernel ? kernel : 2 * unsigned(std::ceil(range[d])) + 1;
			size = std::min(size, circular ? dims[d] : 2 * dims[d] - 1);
			if (size % 2 == 0) size -= 1;
			k[d] = size;
		}
		return kernelOffsets(k);
	}
	std::vector<uint64_t> randomPath(uint64_t seed) const {
		std::vector<uint64_t> path;
		for (size_t i = 0; i < nodes(); ++i) if (known.empty() || !known[i]) path.push_back(i);
		shufflePath(path, seed);
		return path;
	}
};

CovarianceModelSet buildModels(const std::vector<std::string>& texts, unsigned ndim, const Bindings& bindings = Bindings(),
                               std::vector<unsigned> index = std::vector<unsigned>()) {
	return buildCovarianceModelSet(texts, bindings, ndim, index);
}

struct Planned {
	WeightTable table;
	SgsPlanStats stats;
	std::vector<std::vector<int> > offsets;
};

Planned plan(const GridCase& c, const std::vector<uint64_t>& path) {
	Planned result;
	result.offsets = c.offsets();
	const size_t n = c.nodes();
	const std::vector<int64_t> ranks = pathRanks(n, path.data(), path.size(), [&](size_t node) { return !c.known.empty() && bool(c.known[node]); });
	GridSearchOptions options;
	options.circular = c.circular;
	options.maxPerSector = c.perSector;
	GridNeighborSearch search(c.dims, result.offsets, ranks.data(), options);
	GridAddressing addressing(c.dims, result.offsets, c.circular);
	SgsPlanOptions planOptions;
	planOptions.type = c.type;
	planOptions.maxNeighbors = c.neighbors;
	planOptions.shareStencils = c.share;
	planOptions.threads = c.threads;
	std::string error;
	CHECK_MSG(planSgs(search, addressing, c.models, ranks.data(), path.data(), path.size(), planOptions, result.table, result.stats, error), error);
	return result;
}

std::vector<float> replay(const GridCase& c, const Planned& p, const std::vector<float>& hard, unsigned R, uint64_t seed, uint32_t first, unsigned threads) {
	const size_t n = c.nodes();
	std::vector<float> values(n * R, std::numeric_limits<float>::quiet_NaN());
	for (size_t i = 0; i < n; ++i) if (!c.known.empty() && c.known[i]) std::fill(values.begin() + long(i * R), values.begin() + long((i + 1) * R), hard[i]);
	GridAddressing addressing(c.dims, p.offsets, c.circular);
	SgsReplayOptions options;
	options.seed = seed;
	options.firstRealisation = first;
	options.realisations = R;
	options.threads = threads;
	replaySgs(p.table.view(), addressing, values.data(), options);
	return values;
}

bool sameBits(const std::vector<float>& a, const std::vector<float>& b) {
	return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

template <class T>
bool sameVector(const std::vector<T>& a, const std::vector<T>& b) {
	return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

bool sameTable(const WeightTable& a, const WeightTable& b) {
	return sameVector(a.levelStart, b.levelStart) && sameVector(a.entryTarget, b.entryTarget) && sameVector(a.entryStencil, b.entryStencil) &&
	       sameVector(a.stencilStart, b.stencilStart) && sameVector(a.stencilSigma, b.stencilSigma) && sameVector(a.neighborRef, b.neighborRef) &&
	       sameVector(a.neighborWeight, b.neighborWeight) && a.pathHash == b.pathHash;
}

// ------------------------------------------------------------ engine

void testEngine() {
	section("engine: plan invariants");
	GridCase c;
	c.dims = { 40, 30 };
	c.models = buildModels({ "0.1*nugget + 0.9*spherical(range=[12,6], angle=30)" }, 2);
	c.neighbors = 16;
	c.known.assign(c.nodes(), false);
	std::vector<float> hard(c.nodes(), 0.f);
	for (size_t i = 0; i < c.nodes(); i += 37) { c.known[i] = true; hard[i] = float(std::sin(0.3 * double(i))); }
	const std::vector<uint64_t> path = c.randomPath(11);
	const Planned base = plan(c, path);
	CHECK(base.table.entryTarget.size() == path.size());
	// Every simulated neighbour of an entry belongs to an earlier level.
	{
		std::vector<int64_t> levelOf(c.nodes(), -1);
		const WeightTableView v = base.table.view();
		GridAddressing addressing(c.dims, base.offsets, false);
		bool ordered = true, exhaustive = true;
		std::vector<size_t> ids(64);
		for (uint64_t l = 0; l < v.levelCount; ++l)
			for (uint64_t e = v.levelStart[l]; e < v.levelStart[l + 1]; ++e) levelOf[size_t(v.entryTarget[e])] = int64_t(l);
		for (uint64_t l = 0; l < v.levelCount; ++l) {
			for (uint64_t e = v.levelStart[l]; e < v.levelStart[l + 1]; ++e) {
				const uint32_t s = v.entryStencil[e];
				const size_t count = size_t(v.stencilStart[s + 1] - v.stencilStart[s]);
				addressing.decode(size_t(v.entryTarget[e]), v.neighborRef + v.stencilStart[s], count, ids.data());
				for (size_t j = 0; j < count; ++j) {
					if (c.known[ids[j]]) continue;
					if (levelOf[ids[j]] < 0) exhaustive = false;
					if (levelOf[ids[j]] >= int64_t(l)) ordered = false;
				}
			}
		}
		CHECK(ordered);
		CHECK(exhaustive);
	}

	section("engine: thread invariance of plan and replay");
	{
		GridCase c4 = c;
		c4.threads = 4;
		const Planned p4 = plan(c4, path);
		CHECK(sameTable(base.table, p4.table));
		const std::vector<float> a = replay(c, base, hard, 5, 99, 0, 1);
		const std::vector<float> b = replay(c, base, hard, 5, 99, 0, 4);
		const std::vector<float> d = replay(c, base, hard, 5, 99, 0, 8);
		CHECK(sameBits(a, b));
		CHECK(sameBits(a, d));
	}

	section("engine: sharing does not change the weights");
	{
		GridCase u = c;
		u.share = false;
		const Planned unshared = plan(u, path);
		CHECK(unshared.stats.stencils == path.size());
		CHECK(base.stats.stencils < unshared.stats.stencils);
		const std::vector<float> a = replay(c, base, hard, 3, 5, 0, 1);
		const std::vector<float> b = replay(u, unshared, hard, 3, 5, 0, 1);
		CHECK(sameBits(a, b));
	}

	section("engine: R realisations at once = R single runs");
	{
		const unsigned R = 7;
		const std::vector<float> all = replay(c, base, hard, R, 1234, 10, 2);
		bool same = true;
		for (unsigned r = 0; r < R; ++r) {
			const std::vector<float> one = replay(c, base, hard, 1, 1234, 10 + r, 1);
			for (size_t i = 0; i < c.nodes(); ++i)
				if (std::memcmp(&one[i], &all[i * R + r], sizeof(float)) != 0) same = false;
		}
		CHECK(same);
	}

	section("engine: hard data are left untouched, every path node is simulated");
	{
		const std::vector<float> v = replay(c, base, hard, 4, 3, 0, 3);
		bool hardOk = true, simulated = true;
		for (size_t i = 0; i < c.nodes(); ++i)
			for (unsigned r = 0; r < 4; ++r) {
				if (c.known[i] && v[i * 4 + r] != hard[i]) hardOk = false;
				if (!c.known[i] && !std::isfinite(v[i * 4 + r])) simulated = false;
			}
		CHECK(hardOk);
		CHECK(simulated);
	}

	section("engine: ordinary kriging weights sum to 1");
	{
		GridCase o = c;
		o.type = KrigingType::Ordinary;
		const Planned p = plan(o, path);
		double worst = 0.0;
		size_t withNeighbours = 0;
		for (size_t s = 0; s + 1 < p.table.stencilStart.size(); ++s) {
			const size_t b = size_t(p.table.stencilStart[s]), e = size_t(p.table.stencilStart[s + 1]);
			if (b == e) continue;
			++withNeighbours;
			double sum = 0.0;
			for (size_t j = b; j < e; ++j) sum += p.table.neighborWeight[j];
			worst = std::max(worst, std::fabs(sum - 1.0));
		}
		CHECK(withNeighbours > 0);
		CHECK_MSG(worst < 1e-5, formatNumber(worst));
		// Simple kriging weights do not sum to 1 in general.
		double simpleWorst = 0.0;
		for (size_t s = 0; s + 1 < base.table.stencilStart.size(); ++s) {
			double sum = 0.0;
			for (size_t j = size_t(base.table.stencilStart[s]); j < size_t(base.table.stencilStart[s + 1]); ++j) sum += base.table.neighborWeight[j];
			simpleWorst = std::max(simpleWorst, std::fabs(sum - 1.0));
		}
		CHECK(simpleWorst > 1e-3);
		// Isolated nodes take the mean.
		GridCase iso = o;
		iso.known.assign(o.nodes(), false);
		iso.kernel = 1; // no neighbour at all
		const Planned pi = plan(iso, iso.randomPath(1));
		CHECK(pi.stats.isolated == iso.nodes());
		std::vector<float> values(iso.nodes(), 0.f);
		GridAddressing addressing(iso.dims, pi.offsets, false);
		SgsReplayOptions options;
		options.seed = 1;
		options.isolatedMean = 100.0;
		replaySgs(pi.table.view(), addressing, values.data(), options);
		double mean = 0.0;
		for (float v : values) mean += v;
		CHECK_NEAR(mean / double(values.size()), 100.0, 0.2);
	}

	section("engine: weight file round trip");
	{
		std::string error;
		WeightTable t = base.table;
		t.inputHash = 42;
		t.geometry = { 40, 30 };
		CHECK_MSG(writeWeightFile("sgs_engine_test.w", t.view(), error), error);
		WeightFile file;
		CHECK_MSG(file.open("sgs_engine_test.w", error), error);
		const WeightTableView v = file.view();
		CHECK(v.inputHash == 42 && v.pathHash == t.pathHash && v.geometryCount == 2 && v.geometry[1] == 30);
		CHECK(v.entryCount == t.entryTarget.size() && v.stencilCount == t.stencilSigma.size());
		CHECK(std::memcmp(v.neighborWeight, t.neighborWeight.data(), t.neighborWeight.size() * sizeof(float)) == 0);
		CHECK(reinterpret_cast<uintptr_t>(v.neighborWeight) % 64 == 0 && reinterpret_cast<uintptr_t>(v.entryTarget) % 64 == 0);
		GridAddressing addressing(c.dims, base.offsets, false);
		std::vector<float> a(c.nodes(), 0.f), b(c.nodes(), 0.f);
		for (size_t i = 0; i < c.nodes(); ++i) if (c.known[i]) a[i] = b[i] = hard[i];
		SgsReplayOptions options;
		options.seed = 17;
		replaySgs(t.view(), addressing, a.data(), options);
		replaySgs(v, addressing, b.data(), options);
		CHECK(sameBits(a, b));
		// Truncated and corrupted files are refused.
		FILE* in = fopen("sgs_engine_test.w", "rb");
		std::vector<char> bytes;
		if (in) { fseek(in, 0, SEEK_END); bytes.resize(size_t(ftell(in))); fseek(in, 0, SEEK_SET); if (fread(bytes.data(), 1, bytes.size(), in) != bytes.size()) bytes.clear(); fclose(in); }
		CHECK(bytes.size() > 512);
		FILE* out = fopen("sgs_engine_truncated.w", "wb");
		if (out) { fwrite(bytes.data(), 1, bytes.size() - 100, out); fclose(out); }
		WeightFile bad;
		CHECK(!bad.open("sgs_engine_truncated.w", error));
		CHECK_MSG(error.find("truncated") != std::string::npos, error);
		std::vector<char> corrupt = bytes;
		corrupt[0] = 'X';
		out = fopen("sgs_engine_corrupt.w", "wb");
		if (out) { fwrite(corrupt.data(), 1, corrupt.size(), out); fclose(out); }
		CHECK(!bad.open("sgs_engine_corrupt.w", error));
		CHECK_MSG(error.find("magic") != std::string::npos, error);
		corrupt = bytes;
		const WeightFileHeader* h = reinterpret_cast<const WeightFileHeader*>(bytes.data());
		uint32_t wrongRef = 1u << 30;
		std::memcpy(corrupt.data() + h->sections[5].offset, &wrongRef, sizeof(wrongRef));
		out = fopen("sgs_engine_corrupt.w", "wb");
		if (out) { fwrite(corrupt.data(), 1, corrupt.size(), out); fclose(out); }
		CHECK(!bad.open("sgs_engine_corrupt.w", error));
		CHECK_MSG(error.find("reference") != std::string::npos, error);
	}

	section("engine: circular and octant search");
	{
		GridCase k = c;
		k.circular = true;
		k.kernel = 15;
		k.perSector = 4;
		const Planned p = plan(k, path);
		const std::vector<float> a = replay(k, p, hard, 2, 8, 0, 1);
		const std::vector<float> b = replay(k, p, hard, 2, 8, 0, 4);
		CHECK(sameBits(a, b));
		CHECK(p.stats.neighbors <= 16 * path.size());
		// Nodes on the border find neighbours across it.
		GridAddressing addressing(k.dims, p.offsets, true);
		bool wraps = false;
		for (uint32_t ref = 0; ref < p.offsets.size(); ++ref) {
			size_t id = 0;
			addressing.decode(0, &ref, 1, &id);
			if (p.offsets[ref][0] < 0 && id % 40 > 30) wraps = true;
		}
		CHECK(wraps);
	}

	section("engine: point addressing (no grid)");
	{
		// A 1D line of points with explicit coordinates; brute-force search.
		struct LineSearch : NeighborSearch {
			const std::vector<double>& x;
			const std::vector<int64_t>& ranks;
			LineSearch(const std::vector<double>& x_, const std::vector<int64_t>& r_) : x(x_), ranks(r_) {}
			unsigned ndim() const override { return 1; }
			size_t nodeCount() const override { return x.size(); }
			void find(size_t node, int64_t limit, size_t maxCount, NeighborSet& out) const override {
				out.ndim = 1;
				out.clear();
				std::vector<std::pair<double, size_t> > candidates;
				for (size_t i = 0; i < x.size(); ++i) if (i != node && ranks[i] < limit) candidates.push_back({ std::fabs(x[i] - x[node]), i });
				std::sort(candidates.begin(), candidates.end());
				for (size_t k = 0; k < candidates.size() && k < maxCount; ++k) {
					out.ids.push_back(candidates[k].second);
					out.lags.push_back(x[candidates[k].second] - x[node]);
				}
			}
		};
		std::vector<double> x(300);
		for (size_t i = 0; i < x.size(); ++i) x[i] = 0.37 * double(i) + 0.1 * std::sin(double(i));
		std::vector<uint64_t> path(x.size());
		for (size_t i = 0; i < x.size(); ++i) path[i] = i;
		shufflePath(path, 3);
		const std::vector<int64_t> ranks = pathRanks(x.size(), path.data(), path.size(), [](size_t) { return false; });
		LineSearch search(x, ranks);
		PointAddressing addressing(x.data(), x.size(), 1);
		const CovarianceModelSet models = buildModels({ "exponential(range=10)" }, 1);
		SgsPlanOptions options;
		options.maxNeighbors = 8;
		options.threads = 3;
		WeightTable table;
		SgsPlanStats stats;
		std::string error;
		CHECK_MSG(planSgs(search, addressing, models, ranks.data(), path.data(), path.size(), options, table, stats, error), error);
		CHECK(stats.stencils == x.size()); // no sharing with node-id references
		std::vector<float> a(x.size() * 2, 0.f), b(x.size() * 2, 0.f);
		SgsReplayOptions replayOptions;
		replayOptions.realisations = 2;
		replaySgs(table.view(), addressing, a.data(), replayOptions);
		replayOptions.threads = 4;
		replaySgs(table.view(), addressing, b.data(), replayOptions);
		CHECK(sameBits(a, b));
	}
}

// ------------------------------------------------------------ statistics

struct Moments {
	double mean = 0, variance = 0;
	std::vector<double> covariance;
};

// Ensemble moments over all realisations of `values` ([node x R]), restricted
// to nodes whose x coordinate (axis 0) is in [x0, x1).
Moments moments(const std::vector<float>& values, unsigned R, const std::vector<unsigned>& dims, const std::vector<std::vector<int> >& lags,
                unsigned x0 = 0, unsigned x1 = ~0u) {
	Moments m;
	const unsigned nx = dims[0], ny = dims[1];
	x1 = std::min(x1, nx);
	double sum = 0, sum2 = 0;
	size_t count = 0;
	for (unsigned y = 0; y < ny; ++y)
		for (unsigned x = x0; x < x1; ++x)
			for (unsigned r = 0; r < R; ++r) {
				const double v = values[(size_t(y) * nx + x) * R + r];
				sum += v; sum2 += v * v; ++count;
			}
	m.mean = sum / double(count);
	m.variance = sum2 / double(count) - m.mean * m.mean;
	for (const std::vector<int>& lag : lags) {
		double s = 0, sa = 0, sb = 0;
		size_t n = 0;
		for (unsigned y = 0; y < ny; ++y)
			for (unsigned x = x0; x < x1; ++x) {
				const long xb = long(x) + lag[0], yb = long(y) + lag[1];
				if (xb < long(x0) || xb >= long(x1) || yb < 0 || yb >= long(ny)) continue;
				for (unsigned r = 0; r < R; ++r) {
					const double a = values[(size_t(y) * nx + x) * R + r], b = values[(size_t(yb) * nx + size_t(xb)) * R + r];
					s += a * b; sa += a; sb += b; ++n;
				}
			}
		m.covariance.push_back(s / double(n) - (sa / double(n)) * (sb / double(n)));
	}
	return m;
}

// Unconditional simulation of `paths` x R realisations; [node x (paths*R)].
std::vector<float> simulate(GridCase& c, unsigned paths, unsigned R, uint64_t seed) {
	const size_t n = c.nodes();
	std::vector<float> all(n * paths * R);
	for (unsigned p = 0; p < paths; ++p) {
		const Planned planned = plan(c, c.randomPath(pathSeed(seed, p)));
		const std::vector<float> v = replay(c, planned, std::vector<float>(), R, pathSeed(seed, p), 0, 4);
		for (size_t i = 0; i < n; ++i) for (unsigned r = 0; r < R; ++r) all[i * paths * R + p * R + r] = v[i * R + r];
	}
	return all;
}

void checkStatistics(const std::string& model, size_t neighbors = 32) {
	section("statistics: " + model);
	GridCase c;
	c.dims = { 128, 128 };
	c.models = buildModels({ model }, 2);
	c.neighbors = neighbors;
	c.threads = 4;
	const unsigned paths = 2, R = 32;
	const std::vector<float> values = simulate(c, paths, R, 2024);
	const std::vector<std::vector<int> > lags = { { 1, 0 }, { 0, 1 }, { 4, 0 }, { 0, 4 }, { 8, 0 }, { 0, 8 }, { 6, 6 }, { 6, -6 }, { 15, 0 }, { 0, 15 }, { 25, 0 } };
	const Moments m = moments(values, paths * R, c.dims, lags);
	const ResolvedModel& resolved = c.models.stationaryModel();
	CHECK_NEAR(m.mean, 0.0, 0.08);
	CHECK_NEAR(m.variance, resolved.sill(), 0.12);
	std::string line = "   mean " + formatNumber(m.mean) + ", variance " + formatNumber(m.variance) + "; C(h) experimental/model:";
	for (size_t i = 0; i < lags.size(); ++i) {
		const double lag[2] = { double(lags[i][0]), double(lags[i][1]) };
		const double expected = resolved.covariance(lag);
		line += " (" + std::to_string(lags[i][0]) + "," + std::to_string(lags[i][1]) + ") " + formatNumber(std::round(m.covariance[i] * 1000) / 1000) + "/" + formatNumber(std::round(expected * 1000) / 1000);
		report(std::fabs(m.covariance[i] - expected) <= 0.10, "covariance at lag (" + std::to_string(lags[i][0]) + "," + std::to_string(lags[i][1]) + "): " +
		       formatNumber(m.covariance[i]) + " vs model " + formatNumber(expected), __FILE__, __LINE__);
	}
	printf("%s\n", line.c_str());
}

void testStatistics() {
	checkStatistics("spherical(range=20)");
	checkStatistics("exponential(range=30)");
	checkStatistics("spherical(range=[40,12], angle=30)");
	checkStatistics("0.2*nugget + 0.8*exponential(range=[36,12], angle=-45)");
}

// Two halves with different covariances; each half is 64 x 64 cells, 2 paths
// x 32 realisations. Short ranges keep a few hundred independent values per
// half and realisation set, so the bounds below are about 3 sampling sigmas
// (variance within 0.2 of the sill, covariance within 0.15 of the model).
void testNonStationary() {
	section("non-stationary: range map");
	{
		GridCase c;
		c.dims = { 128, 64 };
		std::vector<float> range(c.nodes());
		for (size_t i = 0; i < range.size(); ++i) range[i] = (i % 128) < 64 ? 3.f : 20.f;
		Bindings bindings;
		bindings.setField("L", range);
		c.models = buildModels({ "spherical(range=L)" }, 2, bindings);
		CHECK(!c.models.isStationary());
		c.threads = 4;
		const std::vector<float> v = simulate(c, 2, 32, 5);
		const std::vector<std::vector<int> > lags = { { 5, 0 }, { 0, 5 } };
		const Moments left = moments(v, 64, c.dims, lags, 0, 56);    // away from the switch at x = 64
		const Moments right = moments(v, 64, c.dims, lags, 72, 128);
		const double model = 1.0 - 1.5 * 0.25 + 0.5 * 0.25 * 0.25 * 0.25; // spherical(5 / 20)
		printf("   lag 5 covariance: range-3 half %.3f %.3f (model 0), range-20 half %.3f %.3f (model %.3f); variances %.3f %.3f\n",
		       left.covariance[0], left.covariance[1], right.covariance[0], right.covariance[1], model, left.variance, right.variance);
		CHECK_NEAR(left.covariance[0], 0.0, 0.05);
		CHECK_NEAR(left.covariance[1], 0.0, 0.05);
		CHECK_NEAR(right.covariance[0], model, 0.15);
		CHECK_NEAR(right.covariance[1], model, 0.15);
		CHECK_NEAR(left.variance, 1.0, 0.2);
		CHECK_NEAR(right.variance, 1.0, 0.2);
	}
	section("non-stationary: two models with an index map");
	{
		GridCase c;
		c.dims = { 128, 64 };
		std::vector<unsigned> index(c.nodes());
		for (size_t i = 0; i < index.size(); ++i) index[i] = (i % 128) < 64 ? 0u : 1u;
		c.models = buildModels({ "exponential(range=3)", "cubic(range=20)" }, 2, Bindings(), index);
		c.threads = 4;
		const std::vector<float> v = simulate(c, 2, 32, 6);
		const std::vector<std::vector<int> > lags = { { 2, 0 }, { 0, 2 } };
		const Moments left = moments(v, 64, c.dims, lags, 0, 56);
		const Moments right = moments(v, 64, c.dims, lags, 72, 128);
		const ResolvedModel cubic = CovarianceModel(ModelSpec::parse("cubic(range=20)"), Bindings(), 2).stationaryModel();
		const double lag2[2] = { 2.0, 0.0 };
		const double modelRight = cubic.covariance(lag2), modelLeft = std::exp(-3.0 * 2.0 / 3.0);
		printf("   lag 2 covariance: exponential(3) half %.3f %.3f (model %.3f), cubic(20) half %.3f %.3f (model %.3f); variances %.3f %.3f\n",
		       left.covariance[0], left.covariance[1], modelLeft, right.covariance[0], right.covariance[1], modelRight, left.variance, right.variance);
		CHECK_NEAR(left.covariance[0], modelLeft, 0.1);
		CHECK_NEAR(left.covariance[1], modelLeft, 0.1);
		CHECK_NEAR(right.covariance[0], modelRight, 0.15);
		CHECK_NEAR(right.covariance[1], modelRight, 0.15);
		CHECK_NEAR(left.variance, 1.0, 0.2);
		CHECK_NEAR(right.variance, 1.0, 0.2);
	}
}

// ------------------------------------------------------------ executable

bool writeImage(const std::string& name, const std::vector<unsigned>& dims, const std::vector<float>& values) {
	std::vector<unsigned> d = dims;
	g2s::DataImage image(unsigned(d.size()), d.data(), 1);
	for (size_t i = 0; i < values.size(); ++i) image._data[i] = values[i];
	image.write(name);
	return !g2s::DataImage::createFromFile(name).isEmpty();
}

int runSgs(const std::string& binary, const std::string& arguments) {
	const std::string command = binary + " " + arguments + " 2>sgs_test_last.log";
	const int status = std::system(command.c_str());
	return status;
}

// DataImage::isEmpty() is not const: take the images by forwarding reference.
template <class A, class B>
bool sameImage(A&& a, B&& b) {
	return !a.isEmpty() && !b.isEmpty() && a._dims == b._dims && a._nbVariable == b._nbVariable &&
	       std::memcmp(a._data, b._data, a.dataSize() * sizeof(float)) == 0;
}

g2s::DataImage output(unsigned id) { return g2s::DataImage::createFromFile("im_1_" + std::to_string(id)); }

void testExecutable(const std::string& binary) {
	section("sgs executable");
	FILE* probe = fopen(binary.c_str(), "r");
	if (!probe) { printf("   skipped: %s not built\n", binary.c_str()); return; }
	fclose(probe);
	const std::vector<unsigned> dims = { 60, 45 };
	const size_t n = 60 * 45;
	std::vector<float> di(n, std::numeric_limits<float>::quiet_NaN());
	for (size_t i = 0; i < n; i += 29) di[i] = float(1.5 * std::sin(0.11 * double(i)));
	CHECK(writeImage("sgstest_di", dims, di));
	std::vector<float> di2 = di;
	di2[5] = 0.25f; // one more hard datum: another mask
	CHECK(writeImage("sgstest_di2", dims, di2));
	std::vector<float> priority(n);
	for (size_t i = 0; i < n; ++i) priority[i] = float((i * 7919) % n);
	CHECK(writeImage("sgstest_sp", dims, priority));
	std::vector<float> priority2(n);
	for (size_t i = 0; i < n; ++i) priority2[i] = float((i * 104729) % n);
	CHECK(writeImage("sgstest_sp2", dims, priority2));
	std::vector<float> mean(n);
	for (size_t i = 0; i < n; ++i) mean[i] = float(i % 60) / 30.f;
	CHECK(writeImage("sgstest_mean", dims, mean));

	const std::string model = "-cov '0.1*nugget + 0.9*spherical(range=[15,8], angle=20)'";
	const std::string common = "-di sgstest_di " + model + " -n 20 -s 77 -nr 3";
	// One shot, then plan to a file, then replay the file.
	CHECK(runSgs(binary, common + " -j 4 -id 770001 -o sgstest_a") == 0);
	std::remove("sgstest.w");
	CHECK(runSgs(binary, common + " -j 3 -wo sgstest.w -id 770002 -o sgstest_b") == 0);
	CHECK(runSgs(binary, common + " -j 2 -wi sgstest.w -id 770003 -o sgstest_c") == 0);
	g2s::DataImage a = output(770001), b = output(770002), c = output(770003);
	CHECK(a._nbVariable == 3 && a._dims == dims);
	CHECK(sameImage(a, b));
	CHECK_MSG(sameImage(a, c), "replay from the weight file differs from the one-shot run");
	// A new seed at replay gives other realisations from the same plan.
	CHECK(runSgs(binary, common + " -wi sgstest.w -s 78 -id 770004 -o sgstest_d") == 0);
	CHECK(!sameImage(a, output(770004)));
	// New hard-data values at the same locations are accepted and honoured.
	{
		std::vector<float> moved = di;
		for (float& v : moved) if (!std::isnan(v)) v += 3.f;
		CHECK(writeImage("sgstest_di3", dims, moved));
		CHECK(runSgs(binary, "-di sgstest_di3 " + model + " -n 20 -s 77 -nr 3 -wi sgstest.w -id 770005 -o sgstest_e") == 0);
		g2s::DataImage e = output(770005);
		bool ok = !e.isEmpty();
		for (size_t i = 0; ok && i < n; ++i) if (!std::isnan(moved[i])) for (unsigned r = 0; r < 3; ++r) ok &= e._data[i * 3 + r] == moved[i];
		CHECK(ok);
	}

	section("sgs executable: threads, realisations, paths");
	CHECK(runSgs(binary, common + " -j 1 -id 770010 -o sgstest_j1") == 0);
	CHECK(sameImage(a, output(770010)));
	{
		bool same = true;
		for (unsigned r = 0; r < 3; ++r) {
			CHECK(runSgs(binary, "-di sgstest_di " + model + " -n 20 -s 77 -nr 1 -ri " + std::to_string(r) + " -j 2 -id " + std::to_string(770020 + r) + " -o sgstest_r") == 0);
			g2s::DataImage one = output(770020 + r);
			if (one.isEmpty() || one._nbVariable != 1) { same = false; continue; }
			for (size_t i = 0; i < n; ++i) if (std::memcmp(&one._data[i], &a._data[i * 3 + r], sizeof(float)) != 0) same = false;
		}
		CHECK_MSG(same, "-nr 3 differs from three -nr 1 -ri r runs");
	}
	{
		CHECK(runSgs(binary, common + " -np 2 -j 4 -id 770030 -o sgstest_np") == 0);
		g2s::DataImage p2 = output(770030);
		CHECK(p2._nbVariable == 6);
		bool firstPath = !p2.isEmpty(), differ = false;
		for (size_t i = 0; firstPath && i < n; ++i)
			for (unsigned r = 0; r < 3; ++r) {
				firstPath &= std::memcmp(&p2._data[i * 6 + r], &a._data[i * 3 + r], sizeof(float)) == 0;
				differ |= p2._data[i * 6 + 3 + r] != p2._data[i * 6 + r];
			}
		CHECK_MSG(firstPath, "path 0 of -np 2 differs from -np 1");
		CHECK(differ);
		g2s::DataImage index = g2s::DataImage::createFromFile("im_2_770030");
		CHECK(!index.isEmpty() && index._nbVariable == 2);
		if (!index.isEmpty()) {
			const unsigned* count = reinterpret_cast<const unsigned*>(index._data);
			bool ok = true;
			unsigned largest = 0;
			for (size_t i = 0; i < n; ++i) {
				if (!std::isnan(di[i])) ok &= count[i * 2] == 0;
				largest = std::max(largest, count[i * 2]);
			}
			CHECK(ok);
			CHECK(largest == 20);
		}
	}

	section("sgs executable: hard data and means");
	{
		bool exact = !a.isEmpty(), finite = true;
		for (size_t i = 0; exact && i < n; ++i) {
			for (unsigned r = 0; r < 3; ++r) {
				if (!std::isnan(di[i])) exact &= a._data[i * 3 + r] == di[i];
				else finite &= std::isfinite(a._data[i * 3 + r]);
			}
		}
		CHECK(exact);
		CHECK(finite);
		CHECK(runSgs(binary, "-di sgstest_di " + model + " -n 20 -s 77 -nr 3 -m sgstest_mean -id 770040 -o sgstest_m") == 0);
		g2s::DataImage m = output(770040);
		exact = !m.isEmpty();
		for (size_t i = 0; exact && i < n; ++i) if (!std::isnan(di[i])) for (unsigned r = 0; r < 3; ++r) exact &= m._data[i * 3 + r] == di[i];
		CHECK(exact);
		CHECK(writeImage("sgstest_empty", dims, std::vector<float>(n, std::numeric_limits<float>::quiet_NaN())));
		CHECK(runSgs(binary, "-di sgstest_empty " + model + " -n 20 -s 77 -nr 4 -m -4.5 -id 770041 -o sgstest_m2") == 0);
		g2s::DataImage neg = output(770041);
		double sum = 0;
		for (size_t i = 0; !neg.isEmpty() && i < 4 * n; ++i) sum += neg._data[i];
		CHECK_NEAR(sum / double(4 * n), -4.5, 0.4); // unconditional, negative mean given as "-m -4.5"
		CHECK(runSgs(binary, "-di sgstest_di " + model + " -n 20 -s 77 -kt ok -j 3 -id 770042 -o sgstest_ok") == 0);
		CHECK(!output(770042).isEmpty());
	}

	section("sgs executable: -sp and -pt");
	{
		CHECK(runSgs(binary, common + " -sp sgstest_sp -wo sgstest_sp.w -id 770050 -o sgstest_sp_a") == 0);
		CHECK(runSgs(binary, common + " -sp sgstest_sp -wi sgstest_sp.w -id 770051 -o sgstest_sp_b") == 0);
		CHECK(sameImage(output(770050), output(770051)));
		CHECK(runSgs(binary, common + " -sp sgstest_sp2 -wi sgstest_sp.w -id 770052 -o sgstest_sp_c") != 0); // another path
		CHECK(runSgs(binary, common + " -pt multigrid -j 4 -id 770053 -o sgstest_mg") == 0);
		CHECK(runSgs(binary, common + " -pt regular -j 4 -os 3 -csim -id 770054 -o sgstest_reg") == 0);
	}

	section("sgs executable: mismatched weight files are refused");
	{
		CHECK(runSgs(binary, "-di sgstest_di -cov 'spherical(range=[15,8], angle=20)' -n 20 -s 77 -wi sgstest.w -id 770060 -o sgstest_x") != 0);
		CHECK(runSgs(binary, "-di sgstest_di " + model + " -n 19 -s 77 -wi sgstest.w -id 770061 -o sgstest_x") != 0);
		CHECK(runSgs(binary, "-di sgstest_di2 " + model + " -n 20 -s 77 -wi sgstest.w -id 770062 -o sgstest_x") != 0);
		CHECK(runSgs(binary, "-di sgstest_di " + model + " -n 20 -s 77 -kt ok -wi sgstest.w -id 770063 -o sgstest_x") != 0);
		CHECK(runSgs(binary, "-di sgstest_di " + model + " -n 20 -s 77 -ks 9 -wi sgstest.w -id 770064 -o sgstest_x") != 0);
		CHECK(runSgs(binary, "-di sgstest_di " + model + " -n 20 -s 77 -os 2 -wi sgstest.w -id 770065 -o sgstest_x") != 0);
		CHECK(runSgs(binary, "-di sgstest_di " + model + " -n 20 -s 77 -csim -wi sgstest.w -id 770066 -o sgstest_x") != 0);
		CHECK(runSgs(binary, "-di sgstest_di " + model + " -n 20 -wi missing_file.w -id 770067 -o sgstest_x") != 0);
		// The refusal states the reason.
		CHECK(runSgs(binary, "-di sgstest_di " + model + " -n 19 -s 77 -wi sgstest.w -id 770068 -o sgstest_x") != 0);
		FILE* log = fopen("sgs_test_last.log", "r");
		std::string text;
		if (log) { char buffer[4096]; size_t k; while ((k = fread(buffer, 1, sizeof(buffer), log)) > 0) text.append(buffer, k); fclose(log); }
		CHECK_MSG(text.find("does not match") != std::string::npos, text);
		// A model variable bound to a different map is refused too.
		std::vector<float> range(n, 12.f), range2(n, 12.f);
		for (size_t i = 0; i < n; ++i) { range[i] = 8.f + float(i % 60) / 6.f; range2[i] = range[i] + (i == 100 ? 1.f : 0.f); }
		CHECK(writeImage("sgstest_range", dims, range));
		CHECK(writeImage("sgstest_range2", dims, range2));
		CHECK(runSgs(binary, "-di sgstest_di -cov 'spherical(range=L)' -cvar L sgstest_range -ks 31 -n 20 -s 1 -wo sgstest_ns.w -id 770070 -o sgstest_ns") == 0);
		CHECK(runSgs(binary, "-di sgstest_di -cov 'spherical(range=L)' -cvar L sgstest_range -ks 31 -n 20 -s 1 -wi sgstest_ns.w -id 770071 -o sgstest_ns2") == 0);
		CHECK(sameImage(output(770070), output(770071)));
		CHECK(runSgs(binary, "-di sgstest_di -cov 'spherical(range=L)' -cvar L sgstest_range2 -ks 31 -n 20 -s 1 -wi sgstest_ns.w -id 770072 -o sgstest_x") != 0);
	}

	section("sgs executable: non-stationary inputs");
	{
		std::vector<float> index(n);
		for (size_t i = 0; i < n; ++i) index[i] = (i % 60) < 30 ? 0.f : 1.f;
		CHECK(writeImage("sgstest_index", dims, index));
		CHECK(runSgs(binary, "-di sgstest_di -cov 'exponential(range=4)' -cov 'cubic(range=25)' -ii sgstest_index -ks 31 -n 20 -s 3 -nr 2 -j 4 -id 770080 -o sgstest_ii") == 0);
		CHECK(!output(770080).isEmpty());
		// Errors: missing -cov, bad -kt, -wo with -np 2.
		CHECK(runSgs(binary, "-di sgstest_di -n 20 -id 770081 -o sgstest_x") != 0);
		CHECK(runSgs(binary, "-di sgstest_di " + model + " -kt uk -id 770082 -o sgstest_x") != 0);
		CHECK(runSgs(binary, "-di sgstest_di " + model + " -np 2 -wo sgstest_np.w -id 770083 -o sgstest_x") != 0);
	}
}

} // namespace

int main(int argc, char** argv) {
	const std::string sgs = argc > 1 ? argv[1] : "../../build/c++-build/sgs";
	testEngine();
	testStatistics();
	testNonStationary();
	testExecutable(sgs);
	printf("%d checks, %d failures\n", checks, failures);
	return failures == 0 ? 0 : 1;
}
