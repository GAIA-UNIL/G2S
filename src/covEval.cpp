/*
 * G2S - covEval: evaluate a covariance model at given lags.
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

// covEval: evaluates covariance models (geostat/covarianceModel.hpp) so that
// clients can plot or check a model without re-implementing it.
//
//   g2s('-a','covEval','-cov','0.1*nugget + 0.9*spherical(range=[40,20], angle=30)','-lag',lags)
//   g2s('-a','covEval','-cov',model,'-cvar','L',rangeMap,'-lag',lags,'-pos',[12,30],'-vario')
//
// -lag  N x ndim array of lags (one lag per row, components in DataImage
//       axis order: column 0 = shift along the LAST array index); a vector is
//       N lags in 1D. Mandatory.
// -pos  node where the parameters are resolved (non-stationary models): a
//       1 x ndim row for every lag, or N x ndim (one node per lag); integer
//       cell coordinates in the same component order as -lag.
// -di   optional grid; otherwise the grid is the size of the -cvar / -ii images.
// -vario  also return the variogram C(0) - C(h) (output 2).
// Output 1: C(h), a vector of N values (output 2 with -vario: gamma(h)).
// The canonical text of each model is logged ([PARAM] cov_i) and recorded as
// metadata (covariance_model_i).

#include <chrono>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "utils.hpp"
#include "DataImage.hpp"
#include "jobManager.hpp"
#include "jobReporting.hpp"
#include "geostat/covarianceArguments.hpp"

namespace {

void printHelp() {
	printf("covEval: evaluate a covariance model at given lags.\n"
	       "  -cov  covariance model(s), e.g. \"0.1*nugget + 0.9*spherical(range=[40,20], angle=30)\"\n"
	       "  -cvar name value  binds a model variable (number, or image on the grid); repeat as needed\n"
	       "  -ii   model index image (0-based) when several -cov are given\n"
	       "  -lag  N x ndim lags, column 0 = shift along the last array index (mandatory)\n"
	       "  -pos  1 x ndim or N x ndim nodes where non-stationary parameters are resolved\n"
	       "  -di   grid (optional when a -cvar or -ii image gives it)\n"
	       "  -vario  also output the variogram C(0) - C(h)\n");
}

// N x ndim array (DataImage dims [ndim, N]) or vector (dims [N], ndim = 1).
bool readRows(g2s::DataImage& image, unsigned& ndim, size_t& count) {
	if (image._nbVariable != 1) return false;
	if (image._dims.size() == 1) { ndim = 1; count = image._dims[0]; return true; }
	if (image._dims.size() == 2) { ndim = image._dims[0]; count = image._dims[1]; return true; }
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
	g2s::reporting::markStarted(reportFile, "covEval");

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

	std::string lagFileName, positionFileName, gridFileName;
	if (arg.count("-lag") == 1) lagFileName = arg.find("-lag")->second;
	else { fprintf(reportFile, "error: -lag (N x ndim lags) is mandatory\n"); run = false; }
	arg.erase("-lag");
	if (arg.count("-pos") == 1) positionFileName = arg.find("-pos")->second;
	arg.erase("-pos");
	if (arg.count("-di") == 1) gridFileName = arg.find("-di")->second;
	arg.erase("-di");
	const bool withVariogram = arg.count("-vario") > 0;
	arg.erase("-vario");
	std::string outputFilename = std::to_string(uniqueID);
	if (arg.count("-o") == 1) outputFilename = arg.find("-o")->second;
	arg.erase("-o");
	arg.erase("-j");
	arg.erase("--jobs");
	arg.erase("-dt");

	for (auto it = arg.begin(); it != arg.end(); ++it)
		fprintf(reportFile, "%s %s <== ignored !\n", it->first.c_str(), it->second.c_str());

	if (!run) return failWith(error.empty() ? "invalid parameters" : error, 2);

	const auto begin = std::chrono::high_resolution_clock::now();

	g2s::DataImage lagImage = g2s::DataImage::createFromFile(lagFileName);
	g2s::reporting::logInput(reportFile, "-lag", lagFileName, lagImage);
	if (lagImage.isEmpty()) return failWith("failed to load -lag " + lagFileName, 3);
	unsigned ndim = 0;
	size_t count = 0;
	if (!readRows(lagImage, ndim, count)) return failWith("-lag must be an N x ndim array (or a vector of 1D lags)", 2);
	if (ndim < 1 || ndim > g2s::geostat::kMaxDims) return failWith("-lag must have 1 to " + std::to_string(g2s::geostat::kMaxDims) + " columns", 2);

	std::vector<unsigned> grid;
	if (!gridFileName.empty()) {
		g2s::DataImage gridImage = g2s::DataImage::createFromFile(gridFileName);
		g2s::reporting::logInput(reportFile, "-di", gridFileName, gridImage);
		if (gridImage.isEmpty()) return failWith("failed to load -di " + gridFileName, 3);
		grid = gridImage._dims;
	}

	g2s::geostat::CovarianceModelSet models;
	if (!g2s::geostat::loadCovarianceModels(reportFile, covarianceArguments, grid, ndim, models, error)) return failWith(error, 2);
	const std::vector<std::string> canonical = models.canonical();
	for (size_t i = 0; i < canonical.size(); ++i) g2s::reporting::recordMetric(reportFile, "covariance_model_" + std::to_string(i), canonical[i]);

	// Nodes where the parameters are resolved.
	std::vector<size_t> nodes;
	if (!positionFileName.empty()) {
		g2s::DataImage positionImage = g2s::DataImage::createFromFile(positionFileName);
		g2s::reporting::logInput(reportFile, "-pos", positionFileName, positionImage);
		if (positionImage.isEmpty()) return failWith("failed to load -pos " + positionFileName, 3);
		unsigned positionDims = 0;
		size_t positionCount = 0;
		if (!readRows(positionImage, positionDims, positionCount) || positionDims != ndim || (positionCount != 1 && positionCount != count))
			return failWith("-pos must be 1 x ndim or N x ndim, with the same ndim as -lag", 2);
		if (grid.empty()) return failWith("-pos needs a grid: give -di, or bind a -cvar / -ii image", 2);
		for (size_t i = 0; i < positionCount; ++i) {
			size_t node = 0, stride = 1;
			for (unsigned d = 0; d < ndim; ++d) {
				const long c = std::lround(positionImage._data[i * ndim + d]);
				if (c < 0 || c >= long(grid[d])) return failWith("-pos row " + std::to_string(i) + " is outside the grid", 2);
				node += size_t(c) * stride;
				stride *= grid[d];
			}
			nodes.push_back(node);
		}
	} else if (!models.isStationary()) {
		return failWith("the model varies in space (spatially varying -cvar or -ii): give -pos to say where to evaluate it", 2);
	} else {
		nodes.push_back(0);
	}

	unsigned outputSize = unsigned(count);
	g2s::DataImage covarianceImage(1, &outputSize, 1);
	g2s::DataImage variogramImage(1, &outputSize, 1);
	g2s::geostat::ResolvedModel scratch;
	std::vector<double> lag(ndim);
	for (size_t i = 0; i < count; ++i) {
		const g2s::geostat::ResolvedModel& model = models.at(nodes[nodes.size() == 1 ? 0 : i], scratch);
		for (unsigned d = 0; d < ndim; ++d) lag[d] = double(lagImage._data[i * ndim + d]);
		const double c = model.covariance(lag.data());
		covarianceImage._data[i] = float(c);
		variogramImage._data[i] = float(model.sill() - c);
	}

	const auto end = std::chrono::high_resolution_clock::now();
	const double time = 1.0e-6 * double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
	fprintf(reportFile, "computation time: %.0f ms\n", time);
	g2s::reporting::recordMetric(reportFile, "duration_ms", std::to_string((long long)time));
	g2s::reporting::markFinished(reportFile, time);

	covarianceImage.write(outputFilename);
	covarianceImage.write(std::string("im_1_") + std::to_string(uniqueID));
	g2s::reporting::recordOutputDescriptor(reportFile, 1, "covariance", std::string("im_1_") + std::to_string(uniqueID));
	g2s::reporting::logOutput(reportFile, "covariance_runtime", std::string("im_1_") + std::to_string(uniqueID), covarianceImage);
	if (withVariogram) {
		variogramImage.write(std::string("im_2_") + std::to_string(uniqueID));
		g2s::reporting::recordOutputDescriptor(reportFile, 2, "variogram", std::string("im_2_") + std::to_string(uniqueID));
		g2s::reporting::logOutput(reportFile, "variogram_runtime", std::string("im_2_") + std::to_string(uniqueID), variogramImage);
	}
	return 0;
}
