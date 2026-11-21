/*
 * G2S - kriging: simple and ordinary kriging estimate and variance on a grid.
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

// kriging: estimation (not simulation). The informed (non-NaN) cells of -di
// are the data, the NaN cells the targets; data cells keep their value.
//
//   g2s('-a','kriging','-di',grid,'-cov','0.1*nugget + 0.9*spherical(range=40)','-var')
//   g2s('-a','kriging','-di',grid,'-cov',model,'-kt','sk','-m',meanMap,'-method','local','-n',30,'-os',8)
//
// Two strategies (the engines are include/geostat/krigingGlobal.hpp and
// krigingLocal.hpp; this file only maps the grid onto them):
//   direct  unique neighbourhood: every datum at every target, dual form, one
//           Cholesky of the n x n data covariance; stationary models only.
//   local   moving neighbourhood: the -n nearest data in the -ki/-ks kernel,
//           one small system per target, shared between targets with the same
//           neighbour lags and parameters; spatially varying models allowed.
//   auto    direct when the model is stationary, n <= kAutoDirectMaxData and,
//           with -var, n^2 x targets <= kAutoDirectVarianceWork; local otherwise.
// Output 1: estimate; output 2 (with -var): kriging variance (0 at data cells).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "utils.hpp"
#include "DataImage.hpp"
#include "jobManager.hpp"
#include "jobReporting.hpp"
#include "geostat/covarianceArguments.hpp"
#include "geostat/kdTreeSearch.hpp"
#include "geostat/krigingGlobal.hpp"
#include "geostat/krigingLocal.hpp"
#include "geostat/neighborSearch.hpp"

namespace {

// -method auto: direct up to g2s::geostat::kAutoDirectMaxData data (krigingGlobal.hpp,
// the rule the conditioning of simulations shares) ...
using g2s::geostat::kAutoDirectMaxData;
// ... and, with -var, while n^2 x targets stays below this: the variance costs n^2 per
// target (1000 data on 1000 x 1000 cells: 7.9 s, about the cost of local kriging).
constexpr double kAutoDirectVarianceWork = 1.0e12;
// Default -maxmem (GiB): direct refuses larger n x n matrices.
constexpr double kDefaultMaxMemoryGiB = g2s::geostat::kDefaultDirectMaxMemoryGiB;
// Grid search is used when it walks at most this many kernel offsets per target, the k-d tree otherwise.
constexpr double kGridSearchMaxWalk = 4096.0;

void printHelp() {
	printf("kriging: simple / ordinary kriging of the NaN cells of -di from its informed cells.\n"
	       "  -di   data image, one variable, NaN = cell to estimate (mandatory)\n"
	       "  -cov  covariance model(s) (mandatory); -cvar name value; -ii model index image (see covEval)\n"
	       "  -kt   sk | ok (default ok)\n"
	       "  -m    SK mean: number or image on the -di grid (default 0); OK uses it only where a target has no neighbour\n"
	       "  -method direct | local | auto (default auto)\n"
	       "  -var  also compute the kriging variance (output 2)\n"
	       "  -n    local: number of neighbours (default 50)\n"
	       "  -ki   local: kernel image (neighbourhood window); -ks kernel size; default: unrestricted\n"
	       "  -os k local: octant/quadrant search, at most k neighbours per sector\n"
	       "  -csim local: circular (periodic) grid\n"
	       "  -maxmem GiB  direct: largest n x n matrix allowed (default %.0f)\n"
	       "  -j    threads (a fraction = share of the cores)\n"
	       "  -o / -oi  output names (estimate / variance)\n", kDefaultMaxMemoryGiB);
}

std::string number(double v) {
	char text[64];
	snprintf(text, sizeof(text), "%.6g", v);
	return text;
}

// Value after `flag` in argv that may be a negative number (argumentReader takes "-3" for a flag).
bool numericValueAfter(int argc, char const* argv[], const std::string& flag, std::multimap<std::string, std::string>& arg, std::string& value) {
	for (int i = 0; i + 1 < argc; ++i) {
		if (flag != argv[i]) continue;
		const std::string token = argv[i + 1];
		if (token.size() > 1 && token[0] == '-' && g2s::geostat::parseNumberText(token)) {
			arg.erase(token);
			value = token;
			return true;
		}
	}
	return false;
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
	g2s::reporting::markStarted(reportFile, "kriging");

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

	std::string targetFileName, kernelFileName, meanText;
	if (arg.count("-di") == 1) targetFileName = arg.find("-di")->second;
	else { fprintf(reportFile, "error: -di is mandatory\n"); run = false; }
	arg.erase("-di");
	if (arg.count("-ki") == 1) kernelFileName = arg.find("-ki")->second;
	arg.erase("-ki");
	if (!numericValueAfter(argc, argv, "-m", arg, meanText) && arg.count("-m") == 1) meanText = arg.find("-m")->second;
	if (arg.count("-m") > 1 || (arg.count("-m") == 1 && meanText.empty())) { fprintf(reportFile, "error: -m expects one number or image\n"); run = false; }
	arg.erase("-m");

	std::string outputFilename = std::to_string(uniqueID);
	std::string outputIndexFilename = std::string("id_") + std::to_string(uniqueID);
	if (arg.count("-o") == 1) outputFilename = arg.find("-o")->second;
	arg.erase("-o");
	if (arg.count("-oi") == 1) outputIndexFilename = arg.find("-oi")->second;
	arg.erase("-oi");

	g2s::geostat::KrigingType krigingType = g2s::geostat::KrigingType::Ordinary;
	if (arg.count("-kt") == 1) {
		const std::string value = arg.find("-kt")->second;
		if (value == "sk" || value == "SK") krigingType = g2s::geostat::KrigingType::Simple;
		else if (value == "ok" || value == "OK") krigingType = g2s::geostat::KrigingType::Ordinary;
		else { fprintf(reportFile, "error: -kt expects sk or ok\n"); run = false; }
	}
	arg.erase("-kt");

	std::string method = "auto";
	if (arg.count("-method") == 1) method = arg.find("-method")->second;
	if (method != "auto" && method != "direct" && method != "local") { fprintf(reportFile, "error: -method expects direct, local or auto\n"); run = false; }
	arg.erase("-method");

	const bool withVariance = arg.count("-var") > 0;
	arg.erase("-var");

	size_t nbNeighbors = 50;
	const bool neighborsGiven = arg.count("-n") > 0;
	if (arg.count("-n") == 1) {
		const long value = atol(arg.find("-n")->second.c_str());
		if (value <= 0) { fprintf(reportFile, "error: -n expects a positive integer\n"); run = false; }
		else nbNeighbors = size_t(value);
	} else if (arg.count("-n") > 1) { fprintf(reportFile, "error: -n takes one value (univariate kriging)\n"); run = false; }
	arg.erase("-n");

	int kernelSize = -1;
	if (arg.count("-ks") == 1) {
		kernelSize = atoi(arg.find("-ks")->second.c_str());
		if (kernelSize <= 0) { fprintf(reportFile, "error: -ks expects a positive integer\n"); run = false; }
	}
	arg.erase("-ks");
	unsigned maxPerSector = 0;
	if (arg.count("-os") == 1) {
		const long value = atol(arg.find("-os")->second.c_str());
		if (value <= 0) { fprintf(reportFile, "error: -os expects a positive integer (neighbours per sector)\n"); run = false; }
		else maxPerSector = unsigned(value);
	}
	arg.erase("-os");
	const bool circular = arg.count("-csim") >= 1;
	arg.erase("-csim");
	double maxMemoryGiB = kDefaultMaxMemoryGiB;
	if (arg.count("-maxmem") == 1) {
		maxMemoryGiB = atof(arg.find("-maxmem")->second.c_str());
		if (!(maxMemoryGiB > 0)) { fprintf(reportFile, "error: -maxmem expects a positive number of GiB\n"); run = false; }
	}
	arg.erase("-maxmem");
	arg.erase("-dt");

	for (auto it = arg.begin(); it != arg.end(); ++it)
		fprintf(reportFile, "%s %s <== ignored !\n", it->first.c_str(), it->second.c_str());

	if (!run) return failWith(error.empty() ? "invalid parameters, kriging interrupted" : error, 2);

	// ---- load data
	g2s::DataImage DI = g2s::DataImage::createFromFile(targetFileName);
	g2s::reporting::logInput(reportFile, "-di", targetFileName, DI);
	if (DI.isEmpty()) return failWith("failed to load -di " + targetFileName, 3);
	if (DI._nbVariable != 1) return failWith("-di must have a single variable (univariate kriging), it has " + std::to_string(DI._nbVariable), 2);
	const std::vector<unsigned> dims = DI._dims;
	const unsigned ndim = unsigned(dims.size());
	if (ndim < 1 || ndim > g2s::geostat::kMaxDims) return failWith("-di must have 1 to " + std::to_string(g2s::geostat::kMaxDims) + " dimensions", 2);
	const size_t cellCount = DI.dataSize();

	g2s::geostat::CovarianceModelSet models;
	{
		std::vector<unsigned> grid = dims;
		if (!g2s::geostat::loadCovarianceModels(reportFile, covarianceArguments, grid, ndim, models, error)) return failWith(error, 2);
	}

	std::vector<double> values(cellCount);
	std::vector<size_t> dataCells, targetCells;
	for (size_t cell = 0; cell < cellCount; ++cell) {
		const float v = DI._data[cell];
		values[cell] = double(v);
		if (std::isnan(v)) targetCells.push_back(cell);
		else if (!std::isfinite(v)) return failWith("-di contains an infinite value at cell " + std::to_string(cell), 2);
		else dataCells.push_back(cell);
	}

	// SK mean (number or map); OK uses it where a target has no neighbour.
	double constantMean = 0.0;
	std::vector<double> meanMap;
	if (!meanText.empty()) {
		if (!g2s::geostat::parseNumberText(meanText, &constantMean)) {
			g2s::DataImage meanImage = g2s::DataImage::createFromFile(meanText);
			g2s::reporting::logInput(reportFile, "-m", meanText, meanImage);
			if (meanImage.isEmpty()) return failWith("failed to load -m " + meanText, 3);
			if (meanImage._nbVariable != 1 || meanImage._dims != dims)
				return failWith("-m must be a number or a single-variable image with the size of -di (" + g2s::reporting::joinUnsignedVector(dims) + ")", 2);
			meanMap.assign(meanImage._data, meanImage._data + cellCount);
			for (size_t cell = 0; cell < cellCount; ++cell)
				if (!std::isfinite(meanMap[cell])) return failWith("-m must be finite everywhere; cell " + std::to_string(cell) + " is not", 2);
		} else {
			g2s::reporting::logParameter(reportFile, "mean", meanText);
		}
	}
	auto meanAt = [&](size_t cell) { return meanMap.empty() ? constantMean : meanMap[cell]; };
	const bool simple = krigingType == g2s::geostat::KrigingType::Simple;
	if (!simple && !meanText.empty())
		g2s::reporting::recordWarning(reportFile, "-m is only used by OK where a target has no neighbour (SK uses it everywhere)");

	// ---- strategy
	const bool stationary = models.isStationary();
	const size_t n = dataCells.size();
	const size_t maxMatrixBytes = size_t(maxMemoryGiB * double(size_t(1) << 30));
	std::string reason;
	if (method == "auto") {
		if (!stationary) { method = "local"; reason = "the covariance model varies in space"; }
		else if (n == 0) { method = "local"; reason = "no data"; }
		else if (n > kAutoDirectMaxData) { method = "local"; reason = std::to_string(n) + " data > " + std::to_string(kAutoDirectMaxData) + " (direct limit of auto)"; }
		else if (g2s::geostat::GlobalKriging::matrixBytes(n) > maxMatrixBytes) { method = "local"; reason = "the n x n matrix exceeds -maxmem"; }
		else if (withVariance && double(n) * double(n) * double(targetCells.size()) > kAutoDirectVarianceWork) {
			method = "local";
			reason = "-var with " + std::to_string(n) + " data at " + std::to_string(targetCells.size()) + " targets (n^2 x targets > " + number(kAutoDirectVarianceWork) + ")";
		} else {
			method = "direct";
			reason = "stationary model and " + std::to_string(n) + " data <= " + std::to_string(kAutoDirectMaxData);
		}
		fprintf(reportFile, "method auto: %s (%s)\n", method.c_str(), reason.c_str());
	} else {
		reason = "requested";
	}
	if (method == "direct" && !stationary)
		return failWith("-method direct needs a stationary covariance model (no spatially varying -cvar, a single model index); use -method local", 2);
	if (method == "direct" && neighborsGiven) g2s::reporting::recordWarning(reportFile, "-n is ignored by -method direct (unique neighbourhood: every datum is used)");
	if (method == "direct" && (circular || maxPerSector > 0 || kernelSize > 0 || !kernelFileName.empty()))
		g2s::reporting::recordWarning(reportFile, "-ki, -ks, -os and -csim are ignored by -method direct");

	g2s::reporting::logParameter(reportFile, "kriging_type", simple ? "sk" : "ok");
	g2s::reporting::logParameter(reportFile, "method", method);
	g2s::reporting::logParameter(reportFile, "method_reason", reason);
	g2s::reporting::logParameter(reportFile, "data", std::to_string(n));
	g2s::reporting::logParameter(reportFile, "targets", std::to_string(targetCells.size()));
	g2s::reporting::logParameter(reportFile, "threads", std::to_string(nbThreads));
	g2s::reporting::logParameter(reportFile, "variance", withVariance ? "true" : "false");
	g2s::reporting::recordMetric(reportFile, "kriging_method", method);

	std::vector<double> estimate(targetCells.size()), variance(withVariance ? targetCells.size() : 0);
	const auto begin = std::chrono::high_resolution_clock::now();

	if (method == "direct") {
		if (n == 0) return failWith("-method direct needs at least one informed cell in -di", 2);
		const g2s::geostat::ResolvedModel& model = models.stationaryModel();
		std::vector<double> coordinates(n * ndim);
		for (size_t i = 0; i < n; ++i) {
			long c[g2s::geostat::kMaxDims];
			g2s::geostat::cellCoordinates(dataCells[i], dims, c);
			for (unsigned d = 0; d < ndim; ++d) coordinates[i * ndim + d] = double(c[d]);
		}
		g2s::geostat::GlobalKriging kriging;
		g2s::geostat::GlobalKriging::Options options;
		options.type = krigingType;
		options.nbThreads = nbThreads;
		options.maxMatrixBytes = maxMatrixBytes;
		g2s::reporting::setProgress(reportFile, 5, "factorisation", std::to_string(n) + " x " + std::to_string(n));
		const auto factorBegin = std::chrono::high_resolution_clock::now();
		if (!kriging.factorize(model, coordinates.data(), n, ndim, options, error)) return failWith(error, 5);
		const double factorTime = 1.0e-6 * double(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - factorBegin).count());
		fprintf(reportFile, "factorisation of the %zu x %zu data covariance: %.0f ms\n", n, n, factorTime);
		g2s::reporting::recordMetric(reportFile, "factorization_ms", std::to_string((long long)factorTime));
		if (kriging.jitter() > 0) g2s::reporting::recordWarning(reportFile, "the data covariance matrix needed a diagonal jitter of " + number(kriging.jitter()) + " (duplicate or nearly collinear data, or a very smooth model)");

		std::vector<double> data(n);
		for (size_t i = 0; i < n; ++i) data[i] = values[dataCells[i]] - (simple ? meanAt(dataCells[i]) : 0.0);
		g2s::geostat::GlobalKrigingDual dual;
		kriging.solve(data.data(), 0.0, dual);
		if (!simple) g2s::reporting::logParameter(reportFile, "ok_gls_mean", number(dual.constant[0]));

		g2s::reporting::setProgress(reportFile, 20, "estimate", std::to_string(targetCells.size()) + " targets");
		g2s::geostat::GridCovarianceTable table;
		const bool useTable = table.build(model, dims, nbThreads, maxMatrixBytes / 2);
		g2s::reporting::logParameter(reportFile, "covariance_table", useTable ? "true" : "false (too large, the model is evaluated)");
		if (useTable) {
			std::vector<double> field(cellCount);
			g2s::geostat::estimateGridFields(table, dataCells.data(), dual, field.data(), cellCount, nbThreads);
			for (size_t t = 0; t < targetCells.size(); ++t) estimate[t] = field[targetCells[t]];
		} else {
			kriging.estimate(g2s::geostat::CellCrossCovariance(kriging, dims, targetCells.data()), targetCells.size(), dual, estimate.data(), targetCells.size());
		}
		if (simple) for (size_t t = 0; t < targetCells.size(); ++t) estimate[t] += meanAt(targetCells[t]);
		if (withVariance) {
			g2s::reporting::setProgress(reportFile, 40, "variance", std::to_string(targetCells.size()) + " targets");
			if (useTable) kriging.variance(g2s::geostat::GridCrossCovariance(table, dataCells.data(), n, targetCells.data()), targetCells.size(), variance.data());
			else kriging.variance(g2s::geostat::CellCrossCovariance(kriging, dims, targetCells.data()), targetCells.size(), variance.data());
		}
	} else {
		// ---- local: neighbour search over the data cells
		std::vector<int64_t> ranks(cellCount, g2s::geostat::kRankNever);
		for (size_t cell : dataCells) ranks[cell] = g2s::geostat::kRankKnown;
		if (maxPerSector > 0 && ndim > 4) return failWith("-os supports at most 4 dimensions", 2);

		g2s::DataImage kernel;
		std::vector<unsigned> kernelDims;
		if (!kernelFileName.empty()) {
			kernel = g2s::DataImage::createFromFile(kernelFileName);
			g2s::reporting::logInput(reportFile, "-ki", kernelFileName, kernel);
			if (kernel.isEmpty()) return failWith("failed to load -ki " + kernelFileName, 3);
			if (kernel._dims.size() == dims.size() + 1) kernel.convertFirstDimInVariable();
			if (kernel._dims.size() != dims.size()) return failWith("-ki must have the same number of dimensions as -di", 2);
			kernelDims = kernel._dims;
		} else if (kernelSize > 0) {
			kernelDims.assign(ndim, unsigned(kernelSize) | 1u);
		} else if (circular) {
			kernelDims = dims; // whole periodic grid, each cell reached once
		}
		double offsetCount = 1.0;
		for (unsigned d : kernelDims) offsetCount *= double(d);
		const double density = double(std::max<size_t>(n, 1)) / double(cellCount);
		// Unrestricted search (no kernel) always uses the k-d tree; with a kernel the grid
		// walk is used when it is short (dense data), the k-d tree otherwise. Same neighbours.
		const double expectedWalk = std::min(offsetCount, double(nbNeighbors) / density);
		const bool useGrid = circular || (!kernelDims.empty() && expectedWalk <= kGridSearchMaxWalk);

		std::unique_ptr<g2s::geostat::NeighborSearch> search;
		if (useGrid) {
			const std::vector<std::vector<int> > offsets = g2s::geostat::kernelOffsets(kernelDims, kernel.isEmpty() ? nullptr : kernel._data, kernel.isEmpty() ? 1u : kernel._nbVariable);
			g2s::geostat::GridSearchOptions searchOptions;
			searchOptions.circular = circular;
			searchOptions.maxPerSector = maxPerSector;
			search.reset(new g2s::geostat::GridNeighborSearch(dims, offsets, ranks.data(), searchOptions));
			g2s::reporting::logParameter(reportFile, "neighbor_search", "grid (" + std::to_string(offsets.size()) + " kernel offsets)");
		} else {
			g2s::geostat::KdTreeNeighborSearch::Options searchOptions;
			searchOptions.maxPerSector = maxPerSector;
			if (!kernelDims.empty()) {
				for (unsigned d : kernelDims) searchOptions.halfExtent.push_back(double(d / 2));
				const float* weights = kernel.isEmpty() ? nullptr : kernel._data;
				const unsigned nbVariable = kernel.isEmpty() ? 1u : kernel._nbVariable;
				searchOptions.accept = [kernelDims, weights, nbVariable](const double* lag) {
					size_t flat = 0, stride = 1;
					for (size_t d = 0; d < kernelDims.size(); ++d) {
						const long position = long(kernelDims[d] / 2) + std::lround(lag[d]);
						if (position < 0 || position >= long(kernelDims[d])) return false;
						flat += size_t(position) * stride;
						stride *= kernelDims[d];
					}
					if (!weights) return true;
					for (unsigned v = 0; v < nbVariable; ++v) {
						const float w = weights[flat * nbVariable + v];
						if (std::isfinite(w) && w != 0.f) return true;
					}
					return false;
				};
			}
			auto position = [dims](size_t cell, double* x) {
				for (size_t d = 0; d < dims.size(); ++d) { x[d] = double(cell % dims[d]); cell /= dims[d]; }
			};
			search.reset(new g2s::geostat::KdTreeNeighborSearch(ndim, cellCount, position, ranks.data(), 0, searchOptions));
			g2s::reporting::logParameter(reportFile, "neighbor_search", std::string("k-d tree") + (kernelDims.empty() ? " (unrestricted)" : " (kernel window)"));
		}
		g2s::reporting::logParameter(reportFile, "neighbors", std::to_string(nbNeighbors));
		if (maxPerSector > 0) g2s::reporting::logParameter(reportFile, "neighbors_per_sector", std::to_string(maxPerSector));

		g2s::geostat::LocalKrigingOptions options;
		options.type = krigingType;
		options.maxNeighbors = nbNeighbors;
		options.rankLimit = 0;
		options.computeVariance = withVariance;
		options.nbThreads = nbThreads;
		g2s::reporting::setProgress(reportFile, 5, "local kriging", std::to_string(targetCells.size()) + " targets");
		const g2s::geostat::LocalKrigingStats stats = g2s::geostat::localKriging(*search, models, targetCells.data(), targetCells.size(), values.data(), meanAt,
		                                                                         options, estimate.data(), withVariance ? variance.data() : nullptr);
		char ratio[32];
		snprintf(ratio, sizeof(ratio), "%.4f", stats.sharingRatio());
		fprintf(reportFile, "local kriging: %zu targets, %zu systems solved, %zu shared (ratio %s), %zu without neighbour\n",
		        stats.targets, stats.solved, stats.shared, ratio, stats.noNeighbor);
		g2s::reporting::recordMetric(reportFile, "systems_solved", std::to_string(stats.solved));
		g2s::reporting::recordMetric(reportFile, "systems_shared", std::to_string(stats.shared));
		g2s::reporting::recordMetric(reportFile, "sharing_ratio", ratio);
		if (stats.noNeighbor > 0)
			g2s::reporting::recordWarning(reportFile, std::to_string(stats.noNeighbor) + " target(s) have no datum in their neighbourhood: they get the mean (-m, default 0) and the sill as variance");
		if (stats.jittered > 0) g2s::reporting::recordWarning(reportFile, std::to_string(stats.jittered) + " kriging system(s) needed a diagonal jitter");
		if (stats.failed > 0) g2s::reporting::recordWarning(reportFile, std::to_string(stats.failed) + " kriging system(s) could not be solved: those targets get the mean and the sill");
	}

	const auto end = std::chrono::high_resolution_clock::now();
	const double time = 1.0e-6 * double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
	fprintf(reportFile, "computation time: %7.2f s\n", time / 1000);
	fprintf(reportFile, "computation time: %.0f ms\n", time);
	g2s::reporting::recordMetric(reportFile, "duration_ms", std::to_string((long long)time));
	g2s::reporting::markFinished(reportFile, time);

	// ---- outputs: estimate (data cells keep their value) and variance (0 at data cells)
	g2s::DataImage estimateImage = DI.emptyCopy(true);
	memcpy(estimateImage._data, DI._data, sizeof(float) * cellCount);
	for (size_t t = 0; t < targetCells.size(); ++t) estimateImage._data[targetCells[t]] = float(estimate[t]);
	estimateImage.writeUnderNames({ outputFilename, std::string("im_1_") + std::to_string(uniqueID) });
	g2s::reporting::recordOutputDescriptor(reportFile, 1, "estimate", std::string("im_1_") + std::to_string(uniqueID));
	g2s::reporting::logOutput(reportFile, "estimate_image_runtime", std::string("im_1_") + std::to_string(uniqueID), estimateImage);
	if (withVariance) {
		g2s::DataImage varianceImage = DI.emptyCopy(true);
		for (size_t t = 0; t < targetCells.size(); ++t) varianceImage._data[targetCells[t]] = float(variance[t]);
		varianceImage.writeUnderNames({ outputIndexFilename, std::string("im_2_") + std::to_string(uniqueID) });
		g2s::reporting::recordOutputDescriptor(reportFile, 2, "variance", std::string("im_2_") + std::to_string(uniqueID));
		g2s::reporting::logOutput(reportFile, "variance_image_runtime", std::string("im_2_") + std::to_string(uniqueID), varianceImage);
	}
	return 0;
}
