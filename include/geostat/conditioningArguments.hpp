/*
 * G2S - geostat: shared foundation of the Gaussian geostatistics algorithms.
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

// Grid front end of the conditioning engine (conditioning.hpp), shared by the
// unconditional simulation executables (fftma / spectral, tb) so that they
// read the same flags and report the same way. The informed (non-NaN) cells
// of -di are the hard data; an all-NaN -di leaves the simulation
// unconditional (bit-identical to a run without conditioning).
//
//   -kt sk|ok              kriging type of the residuals (default sk)
//   -km direct|local|auto  kriging method (default auto: the kriging
//                          executable's rule, direct up to kAutoDirectMaxData data)
//   -n  k                  local: number of neighbours (default 50)
//   -ki image / -ks size   local: search window (kernel image, or size); default unrestricted
//   -os k                  local: at most k neighbours per quadrant / octant
//   -maxmem GiB            direct: largest n x n matrix (default kDefaultDirectMaxMemoryGiB)
// The same flags and defaults as the kriging executable, apart from -kt
// (ok there: estimation with an unknown mean) and the method flag, -km here
// because fftma already uses -method for fftma | spectral.
//
// Usage in an executable:
//   g2s::geostat::ConditioningArguments conditioning;
//   if (!g2s::geostat::readConditioningArguments(arg, conditioning, error)) ...  // before the "ignored" loop
//   ... load -di, the model ...
//   g2s::geostat::Conditioner conditioner;
//   if (!g2s::geostat::prepareGridConditioning(reportFile, DI, model, periodic, conditioning, threads, conditioner, error)) ...
//   ... simulate R realisations into a DataImage (R variables) ...
//   conditioner.condition({ output._data, R, 1 }, R);
//   g2s::geostat::reportConditioning(reportFile, conditioner);

#ifndef G2S_GEOSTAT_CONDITIONING_ARGUMENTS_HPP
#define G2S_GEOSTAT_CONDITIONING_ARGUMENTS_HPP

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "DataImage.hpp"
#include "jobReporting.hpp"
#include "geostat/conditioning.hpp"

namespace g2s {
namespace geostat {

struct ConditioningArguments {
	KrigingType type = KrigingType::Simple;
	ConditioningMethod method = ConditioningMethod::Auto;
	size_t maxNeighbors = 50;
	bool neighborsGiven = false;
	std::string kernelFileName;
	int kernelSize = -1;
	unsigned maxPerSector = 0;
	double maxMemoryGiB = kDefaultDirectMaxMemoryGiB;
};

// Help lines of the conditioning flags (for printHelp).
inline const char* conditioningHelp() {
	return "  conditioning: the informed (non-NaN) cells of -di are hard data, honoured exactly; each\n"
	       "  realisation is conditioned by kriging its residuals at the data (all-NaN -di: unconditional)\n"
	       "  -kt   sk | ok (default sk: the residuals have mean 0; ok re-estimates the mean from the data)\n"
	       "  -km   direct | local | auto: kriging method (default auto: direct up to 10000 data, local beyond)\n"
	       "  -n    local: number of neighbours (default 50)\n"
	       "  -ki   local: kernel image (neighbourhood window); -ks kernel size, one value for every axis\n"
	       "        (an even size is increased by 1); default: unrestricted\n"
	       "  -os k local: octant (3D) / quadrant (2D) search, at most k neighbours per sector\n"
	       "  -maxmem GiB  direct: largest n x n matrix allowed (default 2)\n";
}

// Reads and removes -kt, -km, -n, -ki, -ks, -os and -maxmem from `arg`.
// Returns false with the first error in `error`.
inline bool readConditioningArguments(std::multimap<std::string, std::string>& arg, ConditioningArguments& out, std::string& error) {
	bool ok = true;
	auto fail = [&](const std::string& message) { if (ok) error = message; ok = false; };
	if (arg.count("-kt") == 1) {
		const std::string value = arg.find("-kt")->second;
		if (value == "sk" || value == "SK") out.type = KrigingType::Simple;
		else if (value == "ok" || value == "OK") out.type = KrigingType::Ordinary;
		else fail("-kt expects sk or ok");
	} else if (arg.count("-kt") > 1) fail("-kt expects sk or ok");
	arg.erase("-kt");
	if (arg.count("-km") == 1) {
		const std::string value = arg.find("-km")->second;
		if (value == "auto") out.method = ConditioningMethod::Auto;
		else if (value == "direct") out.method = ConditioningMethod::Direct;
		else if (value == "local") out.method = ConditioningMethod::Local;
		else fail("-km expects direct, local or auto");
	} else if (arg.count("-km") > 1) fail("-km expects direct, local or auto");
	arg.erase("-km");
	out.neighborsGiven = arg.count("-n") > 0;
	if (arg.count("-n") == 1) {
		const long value = atol(arg.find("-n")->second.c_str());
		if (value <= 0) fail("-n expects a positive integer");
		else out.maxNeighbors = size_t(value);
	} else if (arg.count("-n") > 1) fail("-n takes one value (univariate kriging)");
	arg.erase("-n");
	if (arg.count("-ki") == 1) out.kernelFileName = arg.find("-ki")->second;
	arg.erase("-ki");
	if (arg.count("-ks") == 1) {
		out.kernelSize = atoi(arg.find("-ks")->second.c_str());
		if (out.kernelSize <= 0) fail("-ks expects a positive integer");
	}
	arg.erase("-ks");
	if (arg.count("-os") == 1) {
		const long value = atol(arg.find("-os")->second.c_str());
		if (value <= 0) fail("-os expects a positive integer (neighbours per sector)");
		else out.maxPerSector = unsigned(value);
	}
	arg.erase("-os");
	if (arg.count("-maxmem") == 1) {
		out.maxMemoryGiB = atof(arg.find("-maxmem")->second.c_str());
		if (!(out.maxMemoryGiB > 0)) fail("-maxmem expects a positive number of GiB");
	}
	arg.erase("-maxmem");
	return ok;
}

// Prepares `conditioner` for the informed cells of DI (its grid, cell
// coordinates; periodic: -csim). Reports the parameters; leaves the
// conditioner inactive when DI has no informed cell. Returns false with a
// message (bad -di values or -ki, failed factorisation).
inline bool prepareGridConditioning(FILE* reportFile, DataImage& DI, const ResolvedModel& model, bool periodic, const ConditioningArguments& args,
                                    unsigned nbThreads, Conditioner& conditioner, std::string& error) {
	const std::vector<unsigned> dims = DI._dims;
	const unsigned ndim = unsigned(dims.size());
	const size_t cellCount = DI.dataSize() / std::max(1u, DI._nbVariable);
	std::vector<size_t> dataCells;
	std::vector<double> dataValues;
	size_t informed = 0;
	for (size_t i = 0; i < DI.dataSize(); ++i) if (!std::isnan(DI._data[i])) ++informed;
	if (informed == 0) {
		g2s::reporting::logParameter(reportFile, "conditioning", "none (no informed cell in -di)");
		std::string unused;
		return conditioner.prepare(model, ConditioningGeometry::regularGrid(dims, periodic), nullptr, nullptr, 0, ConditioningOptions(), unused);
	}
	if (DI._nbVariable != 1) { error = "-di must have a single variable to hold hard data, it has " + std::to_string(DI._nbVariable); return false; }
	for (size_t cell = 0; cell < cellCount; ++cell) {
		const float v = DI._data[cell];
		if (std::isnan(v)) continue;
		if (!std::isfinite(v)) { error = "-di contains an infinite value at cell " + std::to_string(cell); return false; }
		dataCells.push_back(cell);
		dataValues.push_back(double(v));
	}

	ConditioningOptions options;
	options.type = args.type;
	options.method = args.method;
	options.maxNeighbors = args.maxNeighbors;
	options.maxPerSector = args.maxPerSector;
	options.maxMatrixBytes = size_t(args.maxMemoryGiB * double(size_t(1) << 30));
	options.nbThreads = nbThreads;
	if (args.maxPerSector > 0 && ndim > 4) { error = "-os supports at most 4 dimensions"; return false; }
	DataImage kernel;
	std::vector<unsigned> kernelDims;
	if (!args.kernelFileName.empty()) {
		kernel = DataImage::createFromFile(args.kernelFileName);
		g2s::reporting::logInput(reportFile, "-ki", args.kernelFileName, kernel);
		if (kernel.isEmpty()) { error = "failed to load -ki " + args.kernelFileName; return false; }
		if (kernel._dims.size() == dims.size() + 1) kernel.convertFirstDimInVariable();
		if (kernel._dims.size() != dims.size()) { error = "-ki must have the same number of dimensions as -di"; return false; }
		kernelDims = kernel._dims;
	} else if (args.kernelSize > 0) {
		kernelDims.assign(ndim, unsigned(args.kernelSize) | 1u);
	}
	if (!kernelDims.empty()) {
		for (unsigned d : kernelDims) options.halfExtent.push_back(double(d / 2));
		if (!kernel.isEmpty()) {
			// The kernel weights are copied into the filter (the image goes out of scope).
			std::shared_ptr<std::vector<float> > weights(new std::vector<float>(kernel._data, kernel._data + kernel.dataSize()));
			const unsigned nbVariable = kernel._nbVariable;
			options.accept = [kernelDims, weights, nbVariable](const double* lag) {
				size_t flat = 0, stride = 1;
				for (size_t d = 0; d < kernelDims.size(); ++d) {
					const long position = long(kernelDims[d] / 2) + std::lround(lag[d]);
					if (position < 0 || position >= long(kernelDims[d])) return false;
					flat += size_t(position) * stride;
					stride *= kernelDims[d];
				}
				for (unsigned v = 0; v < nbVariable; ++v) {
					const float w = (*weights)[flat * nbVariable + v];
					if (std::isfinite(w) && w != 0.f) return true;
				}
				return false;
			};
		}
	}

	if (!conditioner.prepare(model, ConditioningGeometry::regularGrid(dims, periodic), dataCells.data(), dataValues.data(), dataCells.size(), options, error)) return false;
	const ConditioningReport& report = conditioner.report();
	const bool direct = report.method == ConditioningMethod::Direct;
	g2s::reporting::logParameter(reportFile, "conditioning", "kriging of the residuals");
	g2s::reporting::logParameter(reportFile, "conditioning_data", std::to_string(dataCells.size()));
	g2s::reporting::logParameter(reportFile, "kriging_type", args.type == KrigingType::Simple ? "sk" : "ok");
	g2s::reporting::logParameter(reportFile, "kriging_method", conditioningMethodName(report.method));
	g2s::reporting::logParameter(reportFile, "kriging_method_reason", report.reason);
	g2s::reporting::recordMetric(reportFile, "kriging_method", conditioningMethodName(report.method));
	if (direct) {
		g2s::reporting::logParameter(reportFile, "covariance_table", report.covarianceTable ? "true" : "false (too large, the model is evaluated)");
		if (args.neighborsGiven) g2s::reporting::recordWarning(reportFile, "-n is ignored by -km direct (unique neighbourhood: every datum is used)");
		if (args.maxPerSector > 0 || args.kernelSize > 0 || !args.kernelFileName.empty())
			g2s::reporting::recordWarning(reportFile, "-ki, -ks and -os are ignored by -km direct");
		if (report.jitter > 0) {
			char text[64];
			snprintf(text, sizeof(text), "%.6g", report.jitter);
			g2s::reporting::recordWarning(reportFile, std::string("the data covariance matrix needed a diagonal jitter of ") + text + " (duplicate or nearly collinear data, or a very smooth model)");
		}
		fprintf(reportFile, "factorisation of the %zu x %zu data covariance: %.0f ms\n", dataCells.size(), dataCells.size(), report.prepareMs);
		g2s::reporting::recordMetric(reportFile, "factorization_ms", std::to_string((long long)report.prepareMs));
	} else {
		g2s::reporting::logParameter(reportFile, "neighbors", std::to_string(args.maxNeighbors));
		g2s::reporting::logParameter(reportFile, "neighbor_search", std::string("k-d tree") + (kernelDims.empty() ? " (unrestricted)" : " (kernel window)") + (periodic ? ", periodic" : ""));
		if (args.maxPerSector > 0) g2s::reporting::logParameter(reportFile, "neighbors_per_sector", std::to_string(args.maxPerSector));
	}
	return true;
}

// Reports what condition() did (call after the last condition()).
inline void reportConditioning(FILE* reportFile, const Conditioner& conditioner) {
	if (!conditioner.active()) return;
	const ConditioningReport& report = conditioner.report();
	fprintf(reportFile, "conditioning (%s kriging of the residuals at %zu data): %.0f ms\n", conditioningMethodName(report.method), report.data, report.conditionMs);
	g2s::reporting::recordMetric(reportFile, "conditioning_ms", std::to_string((long long)report.conditionMs));
	if (report.method != ConditioningMethod::Local) return;
	const LocalKrigingStats& stats = report.local;
	char ratio[32];
	snprintf(ratio, sizeof(ratio), "%.4f", stats.sharingRatio());
	fprintf(reportFile, "local kriging: %zu targets, %zu systems solved, %zu shared (ratio %s), %zu without neighbour\n",
	        stats.targets, stats.solved, stats.shared, ratio, stats.noNeighbor);
	g2s::reporting::recordMetric(reportFile, "systems_solved", std::to_string(stats.solved));
	g2s::reporting::recordMetric(reportFile, "systems_shared", std::to_string(stats.shared));
	g2s::reporting::recordMetric(reportFile, "sharing_ratio", ratio);
	if (stats.noNeighbor > 0)
		g2s::reporting::recordWarning(reportFile, std::to_string(stats.noNeighbor) + " target(s) have no datum in their neighbourhood: they stay unconditional");
	if (stats.jittered > 0) g2s::reporting::recordWarning(reportFile, std::to_string(stats.jittered) + " kriging system(s) needed a diagonal jitter");
	if (stats.failed > 0) g2s::reporting::recordWarning(reportFile, std::to_string(stats.failed) + " kriging system(s) could not be solved: those targets stay unconditional");
}

} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_CONDITIONING_ARGUMENTS_HPP
