/*
 * G2S - sgs: kriging-based sequential Gaussian simulation.
 * Copyright (C) 2018-2026, Mathieu Gravey (gravey.mathieu@gmail.com) and UNIL (University of Lausanne)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// sgs: grid front end of the two-phase SGS engine (geostat/sgsEngine.hpp).
// For each path, the plan (neighbour search and kriging weights) is computed
// in parallel, then replayed for R realisations at once. The values are
// simulated as given: transform to normal scores beforehand if needed.
//
//   g2s('-a','sgs','-di',grid,'-cov','spherical(range=[40,20], angle=30)','-n',32,'-nr',8,'-j',8)
//
// Outputs: 1 the simulations, P*R variables in (path, realisation) order;
//          2 the number of neighbours used per node, one variable per path
//            (0 for hard data and nodes that were not simulated).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "utils.hpp"
#include "DataImage.hpp"
#include "jobManager.hpp"
#include "jobReporting.hpp"
#include "geostat/covarianceArguments.hpp"
#include "geostat/gridAddressing.hpp"
#include "geostat/neighborSearch.hpp"
#include "geostat/sgsEngine.hpp"
#include "geostat/weightTable.hpp"

namespace {

void printHelp() {
	printf("sgs: sequential Gaussian simulation (kriging-based, plan then replay).\n"
	       "  -di   destination grid (mandatory); non-NaN cells are hard data, honoured exactly\n"
	       "  -cov  covariance model(s); -cvar name value binds a variable; -ii model index image\n"
	       "  -kt   kriging type: sk (simple, default) or ok (ordinary)\n"
	       "  -m    simple-kriging mean: a number or an image on the grid (default 0)\n"
	       "  -n    maximum number of neighbours (default 32)\n"
	       "  -ki   kernel image (neighbourhood); -ks kernel size when -ki is omitted\n"
	       "        (default: twice the largest practical range of the model, + 1)\n"
	       "  -wd   order the kernel by weight instead of Euclidean distance\n"
	       "  -os k octant search: at most k neighbours per sector (default off)\n"
	       "  -csim circular simulation (the grid wraps around)\n"
	       "  -nr R realisations per path (default 1); -np P paths (default 1); -ri first realisation index (default 0)\n"
	       "  -sp   simulation path image (fixed path); -pt random|multigrid|regular path type (default random)\n"
	       "  -s    seed; -j threads\n"
	       "  -wo file  write the plan (weights) of the path; -wi file  replay a written plan\n"
	       "  -noshare  solve every node, without sharing identical systems\n"
	       "  -o / -oi  output names (simulation / neighbour count)\n");
}

std::string millisecondsText(double ms) {
	char text[64];
	snprintf(text, sizeof(text), "%.1f", ms);
	return text;
}

std::string ratioText(double value) {
	char text[64];
	snprintf(text, sizeof(text), "%.4f", value);
	return text;
}

// Path from a priority image (pyDev convention): ascending, NaN last, -inf not simulated.
std::vector<uint64_t> pathFromImage(const g2s::DataImage& image, const std::vector<bool>& known) {
	const size_t count = known.size();
	std::vector<uint64_t> order(count);
	std::iota(order.begin(), order.end(), uint64_t(0));
	const float* priority = image._data;
	std::stable_sort(order.begin(), order.end(), [priority](uint64_t a, uint64_t b) {
		const bool nanA = std::isnan(priority[a]), nanB = std::isnan(priority[b]);
		if (nanA != nanB) return !nanA;
		return priority[a] < priority[b];
	});
	std::vector<uint64_t> path;
	path.reserve(count);
	for (uint64_t cell : order) {
		const float value = priority[cell];
		if (std::isinf(value) && value < 0) continue;
		if (known[size_t(cell)]) continue;
		path.push_back(cell);
	}
	return path;
}

// Multigrid level of a cell: the largest k <= maxLevel with every coordinate a multiple of 2^k.
unsigned multigridLevel(uint64_t cell, const std::vector<unsigned>& dims, unsigned maxLevel) {
	unsigned level = maxLevel;
	for (unsigned d : dims) {
		const uint64_t c = cell % d;
		cell /= d;
		unsigned zeros = 0;
		if (c == 0) zeros = maxLevel;
		else while (zeros < maxLevel && ((c >> zeros) & 1u) == 0) ++zeros;
		level = std::min(level, zeros);
	}
	return level;
}

// random: a shuffle of the unknown cells; multigrid: coarse to fine levels
// (stride 2^k), random order inside each level; regular: the same with raster
// order inside each level.
std::vector<uint64_t> generatePath(const std::string& type, const std::vector<unsigned>& dims, const std::vector<bool>& known, uint64_t seed) {
	std::vector<uint64_t> path;
	path.reserve(known.size());
	for (size_t cell = 0; cell < known.size(); ++cell) if (!known[cell]) path.push_back(cell);
	if (type != "regular") g2s::geostat::shufflePath(path, seed);
	if (type == "multigrid" || type == "regular") {
		unsigned largest = 1;
		for (unsigned d : dims) largest = std::max(largest, d);
		unsigned maxLevel = 0;
		while ((2u << maxLevel) < largest) ++maxLevel;
		std::vector<unsigned char> level(path.size());
		for (size_t i = 0; i < path.size(); ++i) level[i] = (unsigned char)multigridLevel(path[i], dims, maxLevel);
		std::vector<size_t> order(path.size());
		std::iota(order.begin(), order.end(), size_t(0));
		std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return level[a] > level[b]; });
		std::vector<uint64_t> sorted(path.size());
		for (size_t i = 0; i < order.size(); ++i) sorted[i] = path[order[i]];
		path.swap(sorted);
	}
	return path;
}

} // namespace

int main(int argc, char const* argv[]) {
	std::multimap<std::string, std::string> arg = g2s::argumentReader(argc, argv);

	jobIdType uniqueID = -1;
	FILE* reportFile = nullptr;
	bool run = true;
	if (arg.count("-r") > 1) {
		fprintf(stderr, "only one report file is possible\n");
		run = false;
	} else if (arg.count("-r") == 1) {
		reportFile = g2s::reporting::openReportFile((arg.find("-r")->second).c_str(), uniqueID);
		if (!reportFile) { fprintf(stderr, "Impossible to open the report file\n"); run = false; }
	}
	arg.erase("-r");
	if (!reportFile) reportFile = stderr;
	if (arg.count("-id") == 1) {
		const long parsedId = atol((arg.find("-id")->second).c_str());
		if (parsedId >= 0) uniqueID = jobIdType(parsedId);
	}
	arg.erase("-id");
	g2s::reporting::markStarted(reportFile, "sgs");

	auto failWith = [&](const std::string& message, int code) {
		fprintf(reportFile, "error: %s\n", message.c_str());
		g2s::reporting::recordError(reportFile, message, code);
		return code;
	};

	if (arg.count("-h") >= 1 || arg.count("--help") >= 1) { printHelp(); return 0; }
	for (int i = 0; i < argc; ++i) fprintf(reportFile, "%s ", argv[i]);
	fprintf(reportFile, "\n");

	std::string error;
	g2s::geostat::CovarianceArguments covarianceArguments;
	if (!g2s::geostat::readCovarianceArguments(arg, argc, argv, covarianceArguments, error)) {
		fprintf(reportFile, "error: %s\n", error.c_str());
		run = false;
	}
	if (covarianceArguments.models.empty()) { fprintf(reportFile, "error: -cov is mandatory\n"); run = false; }

	unsigned nbThreads = 1;
	{
		unsigned available = std::max(1u, std::thread::hardware_concurrency());
		for (const char* flag : { "-j", "--jobs" }) {
			auto it = arg.lower_bound(flag);
			if (it != arg.upper_bound(flag)) {
				float value = float(atof(it->second.c_str()));
				if (std::roundf(value) != value) value = std::max(std::floor(value * float(available)), 1.f);
				nbThreads = unsigned(std::max(1.f, value));
			}
			arg.erase(flag);
		}
	}

	auto single = [&](const char* flag, std::string& value) {
		if (arg.count(flag) > 1) { fprintf(reportFile, "error: only one %s is possible\n", flag); run = false; }
		else if (arg.count(flag) == 1) value = arg.find(flag)->second;
		arg.erase(flag);
	};
	auto positive = [&](const char* flag, long& value, long minimum) {
		std::string text;
		single(flag, text);
		if (text.empty()) return;
		char* end = nullptr;
		const long parsed = strtol(text.c_str(), &end, 10);
		if (*end != '\0' || parsed < minimum) { fprintf(reportFile, "error: %s expects an integer >= %ld\n", flag, minimum); run = false; }
		else value = parsed;
	};

	std::string targetFileName, kernelFileName, simulationPathFileName, meanText, krigingTypeText = "sk", pathType = "random";
	std::string weightOutput, weightInput;
	single("-di", targetFileName);
	if (targetFileName.empty()) { fprintf(reportFile, "error: -di is mandatory\n"); run = false; }
	single("-ki", kernelFileName);
	single("-sp", simulationPathFileName);
	single("-m", meanText);
	single("-kt", krigingTypeText);
	single("-pt", pathType);
	single("-wo", weightOutput);
	single("-wi", weightInput);
	for (char& c : krigingTypeText) c = char(std::tolower(static_cast<unsigned char>(c)));
	if (krigingTypeText != "sk" && krigingTypeText != "ok") { fprintf(reportFile, "error: -kt expects sk or ok\n"); run = false; }
	if (pathType != "random" && pathType != "multigrid" && pathType != "regular") { fprintf(reportFile, "error: -pt expects random, multigrid or regular\n"); run = false; }
	// A negative mean is read by argumentReader as a flag: take it back from argv.
	for (int i = 0; i + 1 < argc; ++i) {
		if (std::string(argv[i]) != "-m") continue;
		if (meanText.empty() && g2s::geostat::parseNumberText(argv[i + 1])) { meanText = argv[i + 1]; arg.erase(argv[i + 1]); }
	}

	std::string outputFilename = std::to_string(uniqueID);
	std::string outputIndexFilename = std::string("id_") + std::to_string(uniqueID);
	single("-o", outputFilename);
	single("-oi", outputIndexFilename);

	long nbNeighbors = 32, kernelSize = -1, maxPerSector = 0, realisations = 1, paths = 1, firstRealisation = 0;
	positive("-n", nbNeighbors, 1);
	positive("-ks", kernelSize, 1);
	positive("-os", maxPerSector, 1);
	positive("-nr", realisations, 1);
	positive("-np", paths, 1);
	positive("-ri", firstRealisation, 0);

	uint64_t seed = uint64_t(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	if (arg.count("-s") == 1) seed = uint64_t(strtoull(arg.find("-s")->second.c_str(), nullptr, 10));
	arg.erase("-s");
	const bool byWeight = arg.count("-wd") >= 1;
	arg.erase("-wd");
	arg.erase("-ed");
	const bool circularSimulation = arg.count("-csim") >= 1;
	arg.erase("-csim");
	const bool shareStencils = arg.count("-noshare") == 0;
	arg.erase("-noshare");
	arg.erase("-dt");

	for (auto it = arg.begin(); it != arg.end(); ++it)
		fprintf(reportFile, "%s %s <== ignored !\n", it->first.c_str(), it->second.c_str());

	if (!weightInput.empty() && !weightOutput.empty()) { fprintf(reportFile, "error: -wi and -wo cannot be combined\n"); run = false; }
	if (!weightOutput.empty() && paths > 1) { fprintf(reportFile, "error: -wo stores the plan of one path; use -np 1 (and -nr for more realisations)\n"); run = false; }
	if (!weightInput.empty() && paths > 1) { fprintf(reportFile, "error: -wi replays the path of the file; use -nr for more realisations, not -np\n"); run = false; }
	if (uint64_t(realisations) * uint64_t(paths) > 65535) { fprintf(reportFile, "error: -nr x -np must not exceed 65535\n"); run = false; }
	if (uint64_t(firstRealisation) + uint64_t(realisations) > uint64_t(UINT32_MAX)) { fprintf(reportFile, "error: -ri + -nr is too large\n"); run = false; }

	if (!run) return failWith(error.empty() ? "invalid parameters, simulation interrupted" : error, 2);

	const auto begin = std::chrono::high_resolution_clock::now();

	// ---- inputs
	g2s::DataImage DI = g2s::DataImage::createFromFile(targetFileName);
	g2s::reporting::logInput(reportFile, "-di", targetFileName, DI);
	if (DI.isEmpty()) return failWith("failed to load -di " + targetFileName, 3);
	if (DI._nbVariable != 1) return failWith("-di must have a single variable (sgs is univariate)", 2);
	std::vector<unsigned> grid = DI._dims;
	const size_t nodeCount = DI.dataSize();
	const unsigned ndim = unsigned(grid.size());
	if (ndim > g2s::geostat::kMaxDims) return failWith("-di must have 1 to " + std::to_string(g2s::geostat::kMaxDims) + " dimensions", 2);
	if (maxPerSector > 0 && ndim > 4) return failWith("-os supports at most 4 dimensions", 2);

	g2s::geostat::CovarianceModelSet models;
	if (!g2s::geostat::loadCovarianceModels(reportFile, covarianceArguments, grid, ndim, models, error)) return failWith(error, 2);

	std::vector<bool> known(nodeCount);
	size_t knownCount = 0;
	for (size_t i = 0; i < nodeCount; ++i) { known[i] = !std::isnan(DI._data[i]); knownCount += known[i]; }

	// Mean: a number or an image on the grid.
	double meanConstant = 0.0;
	g2s::DataImage meanImage;
	if (!meanText.empty() && !g2s::geostat::parseNumberText(meanText, &meanConstant)) {
		meanImage = g2s::DataImage::createFromFile(meanText);
		g2s::reporting::logInput(reportFile, "-m", meanText, meanImage);
		if (meanImage.isEmpty()) return failWith("failed to load -m " + meanText, 3);
		if (meanImage._nbVariable != 1 || meanImage._dims != grid) return failWith("-m must be a number or a single-variable image on the -di grid", 2);
		for (size_t i = 0; i < nodeCount; ++i) if (!std::isfinite(meanImage._data[i])) return failWith("-m contains non-finite values", 2);
	}
	const float* meanMap = meanImage.isEmpty() ? nullptr : meanImage._data;
	auto meanAt = [&](size_t node) { return meanMap ? double(meanMap[node]) : meanConstant; };
	const g2s::geostat::KrigingType krigingType = krigingTypeText == "ok" ? g2s::geostat::KrigingType::Ordinary : g2s::geostat::KrigingType::Simple;
	if (krigingType == g2s::geostat::KrigingType::Ordinary && !meanText.empty())
		g2s::reporting::recordWarning(reportFile, "-m with ordinary kriging is only used for nodes without any neighbour");

	// Kernel: -ki, -ks, or twice the largest practical range.
	std::vector<unsigned> kernelDims(ndim);
	std::vector<float> kernelWeights;
	unsigned kernelVariables = 1;
	if (!kernelFileName.empty()) {
		g2s::DataImage kernel = g2s::DataImage::createFromFile(kernelFileName);
		g2s::reporting::logInput(reportFile, "-ki", kernelFileName, kernel);
		if (kernel.isEmpty()) return failWith("failed to load -ki " + kernelFileName, 3);
		if (kernel._dims.size() == ndim + 1) kernel.convertFirstDimInVariable();
		if (kernel._dims.size() != ndim) return failWith("-ki must have the same number of dimensions as -di", 2);
		kernelDims = kernel._dims;
		kernelVariables = kernel._nbVariable;
		kernelWeights.assign(kernel._data, kernel._data + kernel.dataSize());
	} else {
		const std::vector<double> range = models.maxPracticalRange();
		for (unsigned d = 0; d < ndim; ++d) {
			long size = kernelSize > 0 ? kernelSize : 2 * long(std::ceil(range[d])) + 1;
			const long limit = circularSimulation ? long(grid[d]) : 2 * long(grid[d]) - 1;
			size = std::max(1L, std::min(size, limit));
			if (size % 2 == 0) size -= 1;
			kernelDims[d] = unsigned(std::max(1L, size));
		}
	}
	const std::vector<std::vector<int> > offsets = g2s::geostat::kernelOffsets(kernelDims, kernelWeights.empty() ? nullptr : kernelWeights.data(), kernelVariables, byWeight);
	if (offsets.empty()) g2s::reporting::recordWarning(reportFile, "the neighbourhood kernel is empty: every node is drawn from the unconditional distribution");
	const g2s::geostat::GridAddressing addressing(grid, offsets, circularSimulation);

	// Everything the weights depend on, except the path.
	uint64_t inputHash = 0;
	{
		g2s::geostat::Hasher h;
		h.text("g2s-sgs-grid-1");
		h.u64(ndim);
		for (unsigned d : grid) h.u64(d);
		std::vector<uint8_t> mask((nodeCount + 7) / 8, 0);
		for (size_t i = 0; i < nodeCount; ++i) if (known[i]) mask[i / 8] |= uint8_t(1u << (i % 8));
		h.bytes(mask.data(), mask.size());
		h.u64(g2s::geostat::covarianceModelSetHash(models, nodeCount));
		h.u64(krigingType == g2s::geostat::KrigingType::Ordinary ? 1 : 0);
		h.u64(uint64_t(nbNeighbors));
		h.u64(offsets.size());
		for (const std::vector<int>& o : offsets) for (int v : o) h.i64(v);
		h.u64(uint64_t(maxPerSector));
		h.u64(circularSimulation ? 1 : 0);
		inputHash = h.value();
	}

	g2s::DataImage simulationPath;
	if (!simulationPathFileName.empty()) {
		simulationPath = g2s::DataImage::createFromFile(simulationPathFileName);
		g2s::reporting::logInput(reportFile, "-sp", simulationPathFileName, simulationPath);
		if (simulationPath.isEmpty()) return failWith("failed to load -sp " + simulationPathFileName, 3);
		if (simulationPath._dims != grid || simulationPath._nbVariable != 1) return failWith("-sp must have the same grid as -di with one value per cell", 2);
	}

	g2s::geostat::WeightFile weightFile;
	if (!weightInput.empty()) {
		if (!weightFile.open(weightInput, error)) return failWith(error, 3);
		const g2s::geostat::WeightTableView& v = weightFile.view();
		std::string mismatch;
		std::vector<uint64_t> fileGrid(v.geometry, v.geometry + v.geometryCount);
		if (fileGrid != std::vector<uint64_t>(grid.begin(), grid.end())) mismatch = "the grid differs";
		else if (v.krigingType != (krigingType == g2s::geostat::KrigingType::Ordinary ? 1u : 0u)) mismatch = "the kriging type (-kt) differs";
		else if (bool(v.flags & g2s::geostat::WeightTableCircular) != circularSimulation) mismatch = "-csim differs";
		else if (v.ndim != ndim || v.refTable == nullptr || addressing.referenceTable() != std::vector<int32_t>(v.refTable, v.refTable + v.refCount * v.ndim)) mismatch = "the neighbourhood kernel (-ki/-ks/-wd) differs";
		else if (v.inputHash != inputHash) mismatch = "the covariance model, the hard-data locations, -n or -os differ";
		if (!mismatch.empty()) return failWith("the weight file " + weightInput + " does not match this simulation: " + mismatch + " (it must be replayed with the inputs it was planned with)", 2);
		g2s::reporting::logParameter(reportFile, "weight_file", weightInput + (weightFile.mapped() ? " (memory-mapped)" : " (read)"));
	}

	g2s::reporting::logParameter(reportFile, "kriging_type", krigingTypeText);
	g2s::reporting::logParameter(reportFile, "neighbors", std::to_string(nbNeighbors));
	g2s::reporting::logParameter(reportFile, "kernel", g2s::reporting::joinUnsignedVector(kernelDims) + " (" + std::to_string(offsets.size()) + " offsets)");
	g2s::reporting::logParameter(reportFile, "octant_search", maxPerSector > 0 ? std::to_string(maxPerSector) + " per sector" : std::string("off"));
	g2s::reporting::logParameter(reportFile, "circular", circularSimulation ? "true" : "false");
	g2s::reporting::logParameter(reportFile, "realisations_per_path", std::to_string(realisations));
	g2s::reporting::logParameter(reportFile, "paths", std::to_string(paths));
	g2s::reporting::logParameter(reportFile, "first_realisation", std::to_string(firstRealisation));
	g2s::reporting::logParameter(reportFile, "path_type", simulationPathFileName.empty() ? (weightInput.empty() ? pathType : std::string("from the weight file")) : std::string("-sp"));
	g2s::reporting::logParameter(reportFile, "stencil_sharing", shareStencils ? "true" : "false");
	g2s::reporting::logParameter(reportFile, "seed", std::to_string((unsigned long long)seed));
	g2s::reporting::logParameter(reportFile, "threads", std::to_string(nbThreads));
	g2s::reporting::logParameter(reportFile, "hard_data", std::to_string(knownCount));

	// ---- outputs
	const unsigned R = unsigned(realisations), P = unsigned(paths);
	g2s::DataImage simulation(ndim, grid.data(), P * R);
	g2s::DataImage indexImage(ndim, grid.data(), P);
	indexImage.setEncoding(g2s::DataImage::UInteger);
	unsigned* neighborCount = reinterpret_cast<unsigned*>(indexImage._data);
	std::fill(neighborCount, neighborCount + size_t(P) * nodeCount, 0u);

	std::vector<float> values(nodeCount * R);
	g2s::geostat::WeightTable table;
	bool tableForFixedPath = false;
	double planMs = 0, replayMs = 0;
	size_t totalEntries = 0, totalStencils = 0;
	const bool simple = krigingType == g2s::geostat::KrigingType::Simple;

	for (unsigned p = 0; p < P; ++p) {
		const uint64_t seedOfPath = g2s::geostat::pathSeed(seed, p);
		g2s::geostat::WeightTableView view;
		if (!weightInput.empty()) {
			view = weightFile.view();
			if (!simulationPathFileName.empty()) {
				const std::vector<uint64_t> path = pathFromImage(simulationPath, known);
				if (g2s::geostat::pathHash(path.data(), path.size()) != view.pathHash)
					return failWith("the weight file " + weightInput + " was planned with another path than -sp " + simulationPathFileName, 2);
			}
		} else if (tableForFixedPath) {
			view = table.view(); // -sp: the same path, hence the same plan, for every path index
		} else {
			g2s::reporting::setProgress(reportFile, 100.0 * p / P, "plan", "path " + std::to_string(p + 1) + " of " + std::to_string(P));
			const std::vector<uint64_t> path = simulationPathFileName.empty() ? generatePath(pathType, grid, known, seedOfPath) : pathFromImage(simulationPath, known);
			const std::vector<int64_t> ranks = g2s::geostat::pathRanks(nodeCount, path.data(), path.size(), [&](size_t node) { return bool(known[node]); });
			g2s::geostat::GridSearchOptions searchOptions;
			searchOptions.circular = circularSimulation;
			searchOptions.maxPerSector = unsigned(maxPerSector);
			const g2s::geostat::GridNeighborSearch search(grid, offsets, ranks.data(), searchOptions);
			g2s::geostat::SgsPlanOptions planOptions;
			planOptions.type = krigingType;
			planOptions.maxNeighbors = size_t(nbNeighbors);
			planOptions.shareStencils = shareStencils;
			planOptions.threads = nbThreads;
			g2s::geostat::SgsPlanStats stats;
			if (!g2s::geostat::planSgs(search, addressing, models, ranks.data(), path.data(), path.size(), planOptions, table, stats, error)) return failWith(error, 5);
			table.inputHash = inputHash;
			if (circularSimulation) table.flags |= g2s::geostat::WeightTableCircular;
			table.geometry.assign(grid.begin(), grid.end());
			tableForFixedPath = !simulationPathFileName.empty();
			planMs += stats.totalMs;
			fprintf(reportFile, "plan (path %u): %zu nodes, %zu systems solved (%.4f per node), %zu levels (largest %zu), %zu without neighbour, %zu jittered, %zu failed; "
			        "search %.0f ms, levels %.0f ms, sharing %.0f ms, solve %.0f ms, total %.0f ms\n",
			        p, stats.entries, stats.stencils, stats.sharingRatio(), stats.levels, stats.largestLevel, stats.isolated, stats.jittered, stats.failed,
			        stats.searchMs, stats.levelMs, stats.shareMs, stats.solveMs, stats.totalMs);
			if (stats.failed > 0) g2s::reporting::recordWarning(reportFile, std::to_string(stats.failed) + " kriging systems could not be solved; these nodes were drawn from the unconditional distribution");
			if (p == 0) {
				g2s::reporting::recordMetric(reportFile, "plan_systems_solved", std::to_string(stats.stencils));
				g2s::reporting::recordMetric(reportFile, "plan_nodes", std::to_string(stats.entries));
				g2s::reporting::recordMetric(reportFile, "plan_sharing_ratio", ratioText(stats.sharingRatio()));
				g2s::reporting::recordMetric(reportFile, "plan_levels", std::to_string(stats.levels));
			}
			view = table.view();
			if (!weightOutput.empty()) {
				if (!g2s::geostat::writeWeightFile(weightOutput, view, error)) return failWith(error, 6);
				g2s::reporting::logParameter(reportFile, "weight_file_written", weightOutput);
			}
		}
		totalEntries += view.entryCount;
		totalStencils += view.stencilCount;

		// Values: residuals (simple kriging) or raw values, [node x R].
		for (size_t node = 0; node < nodeCount; ++node) {
			const float value = known[node] ? float(double(DI._data[node]) - (simple ? meanAt(node) : 0.0)) : std::numeric_limits<float>::quiet_NaN();
			std::fill(values.begin() + long(node * R), values.begin() + long((node + 1) * R), value);
		}
		g2s::reporting::setProgress(reportFile, 100.0 * (p + 0.5) / P, "replay", "path " + std::to_string(p + 1) + " of " + std::to_string(P));
		g2s::geostat::SgsReplayOptions replayOptions;
		replayOptions.seed = seedOfPath;
		replayOptions.firstRealisation = uint32_t(firstRealisation);
		replayOptions.realisations = R;
		replayOptions.threads = nbThreads;
		if (!simple) { replayOptions.isolatedMeanMap = meanMap; replayOptions.isolatedMean = meanConstant; }
		g2s::geostat::SgsReplayStats replayStats;
		g2s::geostat::replaySgs(view, addressing, values.data(), replayOptions, &replayStats);
		replayMs += replayStats.totalMs;
		fprintf(reportFile, "replay (path %u): %u realisations, %u threads, %zu waits, %zu stolen, noise %.0f ms, total %.0f ms\n", p, R, replayStats.threads, replayStats.waits, replayStats.stolen, replayStats.noiseMs, replayStats.totalMs);

		// Simulated nodes from the table; hard data copied exactly.
		const unsigned variables = P * R;
		for (size_t node = 0; node < nodeCount; ++node) {
			float* out = simulation._data + node * variables + size_t(p) * R;
			if (known[node]) { std::fill(out, out + R, DI._data[node]); continue; }
			std::fill(out, out + R, std::numeric_limits<float>::quiet_NaN());
		}
		for (uint64_t e = 0; e < view.entryCount; ++e) {
			const size_t node = size_t(view.entryTarget[e]);
			float* out = simulation._data + node * variables + size_t(p) * R;
			const float* in = values.data() + node * R;
			const float shift = simple ? float(meanAt(node)) : 0.f;
			for (unsigned r = 0; r < R; ++r) out[r] = in[r] + shift;
			neighborCount[node * P + p] = unsigned(view.neighborsOf(e));
		}
	}

	const auto end = std::chrono::high_resolution_clock::now();
	const double time = 1.0e-6 * double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
	fprintf(reportFile, "computation time: %7.2f s\n", time / 1000);
	fprintf(reportFile, "computation time: %.0f ms\n", time);
	g2s::reporting::recordMetric(reportFile, "duration_ms", std::to_string((long long)time));
	g2s::reporting::recordMetric(reportFile, "plan_ms", millisecondsText(planMs));
	g2s::reporting::recordMetric(reportFile, "replay_ms", millisecondsText(replayMs));
	if (totalEntries > 0) g2s::reporting::recordMetric(reportFile, "systems_per_node", ratioText(double(totalStencils) / double(totalEntries)));
	g2s::reporting::markFinished(reportFile, time);

	indexImage.write(outputIndexFilename);
	simulation.write(outputFilename);
	indexImage.write(std::string("im_2_") + std::to_string(uniqueID));
	simulation.write(std::string("im_1_") + std::to_string(uniqueID));
	g2s::reporting::recordOutputDescriptor(reportFile, 1, "simulation", std::string("im_1_") + std::to_string(uniqueID));
	g2s::reporting::recordOutputDescriptor(reportFile, 2, "indexmap", std::string("im_2_") + std::to_string(uniqueID));
	g2s::reporting::logOutput(reportFile, "simulation_image_runtime", std::string("im_1_") + std::to_string(uniqueID), simulation);
	g2s::reporting::logOutput(reportFile, "index_image_runtime", std::string("im_2_") + std::to_string(uniqueID), indexImage);
	return 0;
}
