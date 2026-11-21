/*
 * G2S - fftma / spectral: unconditional Gaussian simulation with FFTs.
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

// fftma and spectral: stationary Gaussian simulation on the -di grid
// (engine: geostat/spectralEngine.hpp). One executable; the method comes
// from the name it is run under (the build links `spectral` to `fftma`) and
// -method overrides it.
//
//   g2s('-a','fftma','-di',grid,'-cov','1*spherical(range=[40,20], angle=30)','-nr',16,'-s',1,'-j',8)
//   g2s('-a','spectral','-di',grid,'-cov','1*gaussian(range=30)','-csim')
//   g2s('-a','fftma','-di',grid,'-cov',model,'-ni',noise,'-on')   % FFT-MA with a given noise
//   g2s('-a','fftma','-di',data,'-cov',model,'-nr',16,'-km','local')  % conditional
//
// The informed (non-NaN) cells of -di are hard data: each realisation is
// conditioned by kriging of its residuals (geostat/conditioning.hpp, flags in
// geostat/conditioningArguments.hpp). With -csim the kriging uses the
// covariance of the periodic simulated field (minimal-image lags, the same
// circulant covariance as the FFTs). An all-NaN -di gives the unconditional
// simulation.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "utils.hpp"
#include "DataImage.hpp"
#include "jobManager.hpp"
#include "jobReporting.hpp"
#include "geostat/conditioningArguments.hpp"
#include "geostat/covarianceArguments.hpp"
#include "geostat/spectralEngine.hpp"

namespace {

void printHelp() {
	printf("fftma / spectral: Gaussian simulation with FFTs, conditioned to the informed cells of -di.\n"
	       "  -di   grid of the simulation (mandatory); non-NaN cells are hard data (all NaN: unconditional)\n"
	       "  -cov  covariance model, e.g. \"0.1*nugget + 0.9*spherical(range=[40,20], angle=30)\" (mandatory, stationary)\n"
	       "  -cvar name value  binds a model variable (number, or map image); repeat as needed\n"
	       "  -method fftma|spectral  default: the name the executable is run under\n"
	       "  -nr   number of realisations (default 1; the number of -ni variables when -ni is given)\n"
	       "  -ri   index of the first realisation (default 0): realisation r of a run with -ri 0\n"
	       "        equals realisation 0 of a run with -ri r\n"
	       "  -m    mean (default 0)\n"
	       "  -s    seed; -j threads (a fraction = share of the cores); with a given FFTW library on a\n"
	       "        given machine the output is bit-identical for any -j and any split of -nr / -ri\n"
	       "        (not across FFTW builds or machines)\n"
	       "  -buf  buffer in cells, one value or one per axis (default: the largest practical\n"
	       "        range along each axis); the grid is then rounded up to an FFT-friendly size\n"
	       "  -csim periodic simulation on the -di grid (no buffer); the conditioning uses periodic distances\n"
	       "  -ni   fftma only: white noise image on the extended grid, one variable per realisation\n"
	       "        (its size sets the extended grid)\n"
	       "  -on   fftma only: also output the white noise of the unconditional field (output 2, extended grid)\n"
	       "%s"
	       "  -o / -oi  output names (simulation / noise)\n", g2s::geostat::conditioningHelp());
}

std::string baseName(const char* path) {
	if (!path) return std::string();
	std::string name(path);
	const size_t slash = name.find_last_of("/\\");
	if (slash != std::string::npos) name = name.substr(slash + 1);
	return name;
}

std::string gridText(const std::vector<unsigned>& dims) { return g2s::reporting::joinUnsignedVector(dims); }

} // namespace

int main(int argc, char const* argv[]) {
	namespace gs = g2s::geostat;
	namespace sp = g2s::geostat::spectral;
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

	sp::Method method = gs::toLower(baseName(argv[0])).find("spectral") != std::string::npos ? sp::Method::Spectral : sp::Method::FftMovingAverage;
	std::string methodError;
	if (arg.count("-method") == 1) {
		const std::string value = gs::toLower(arg.find("-method")->second);
		if (value == "fftma" || value == "fft-ma") method = sp::Method::FftMovingAverage;
		else if (value == "spectral") method = sp::Method::Spectral;
		else methodError = "-method must be fftma or spectral, got '" + arg.find("-method")->second + "'";
	} else if (arg.count("-method") > 1) {
		methodError = "only one -method is possible";
	}
	arg.erase("-method");
	g2s::reporting::markStarted(reportFile, sp::methodName(method));

	auto failWith = [&](const std::string& message, int code) {
		fprintf(reportFile, "error: %s\n", message.c_str());
		g2s::reporting::recordError(reportFile, message, code);
		return code;
	};
	auto warn = [&](const std::string& message) {
		fprintf(reportFile, "warning: %s\n", message.c_str());
		g2s::reporting::recordWarning(reportFile, message);
	};

	if (arg.count("-h") >= 1 || arg.count("--help") >= 1) { printHelp(); return 0; }
	for (int i = 0; i < argc; ++i) fprintf(reportFile, "%s ", argv[i]);
	fprintf(reportFile, "\n");
	if (!methodError.empty()) { fprintf(reportFile, "error: %s\n", methodError.c_str()); run = false; }

	std::string error;
	gs::CovarianceArguments covarianceArguments;
	if (!gs::readCovarianceArguments(arg, argc, argv, covarianceArguments, error)) {
		fprintf(reportFile, "error: %s\n", error.c_str());
		run = false;
	}
	gs::ConditioningArguments conditioning;
	{
		std::string conditioningError;
		if (!gs::readConditioningArguments(arg, conditioning, conditioningError)) {
			fprintf(reportFile, "error: %s\n", conditioningError.c_str());
			if (error.empty()) error = conditioningError;
			run = false;
		}
	}
	if (covarianceArguments.models.empty()) { fprintf(reportFile, "error: -cov is mandatory\n"); run = false; }

	unsigned nbThreads = 1;
	{
		const unsigned available = std::max(1u, std::thread::hardware_concurrency());
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

	std::string targetFileName, noiseFileName;
	if (arg.count("-di") == 1) targetFileName = arg.find("-di")->second;
	else { fprintf(reportFile, "error: -di is mandatory\n"); run = false; }
	arg.erase("-di");
	if (arg.count("-ni") == 1) noiseFileName = arg.find("-ni")->second;
	else if (arg.count("-ni") > 1) { fprintf(reportFile, "error: only one -ni (noise image) is possible\n"); run = false; }
	arg.erase("-ni");
	const bool outputNoise = arg.count("-on") > 0;
	arg.erase("-on");
	const bool periodic = arg.count("-csim") > 0;
	arg.erase("-csim");

	std::string outputFilename = std::to_string(uniqueID);
	std::string outputIndexFilename = std::string("id_") + std::to_string(uniqueID);
	if (arg.count("-o") == 1) outputFilename = arg.find("-o")->second;
	arg.erase("-o");
	if (arg.count("-oi") == 1) outputIndexFilename = arg.find("-oi")->second;
	arg.erase("-oi");

	// argumentReader() takes "-2" for a flag: read the value from argv instead
	// and drop the bogus key.
	auto readNumber = [&](const char* flag, double& value) {
		if (arg.count(flag) == 0) return false;
		std::string text = arg.find(flag)->second;
		for (int i = 0; i + 1 < argc; ++i) {
			if (std::string(argv[i]) != flag) continue;
			const std::string next = argv[i + 1];
			if (next.size() > 1 && next[0] == '-' && gs::parseNumberText(next)) { text = next; arg.erase(next); }
		}
		if (arg.count(flag) > 1 || !gs::parseNumberText(text, &value)) {
			fprintf(reportFile, "error: %s expects one number\n", flag);
			run = false;
		}
		arg.erase(flag);
		return true;
	};
	double value = 0.0;
	unsigned nbRealisations = 1;
	const bool realisationsGiven = readNumber("-nr", value);
	if (realisationsGiven) {
		if (!(value >= 1 && value <= 1e6 && value == std::floor(value))) { fprintf(reportFile, "error: -nr expects an integer >= 1\n"); run = false; }
		else nbRealisations = unsigned(value);
	}
	uint32_t firstRealisation = 0;
	for (const char* flag : { "-ri", "-fr" }) { // -fr: former name, kept as an alias
		if (!readNumber(flag, value)) continue;
		if (!(value >= 0 && value <= 4294967295.0 && value == std::floor(value))) { fprintf(reportFile, "error: -ri expects an integer >= 0\n"); run = false; }
		else firstRealisation = uint32_t(value);
		break;
	}
	arg.erase("-fr");
	double mean = 0.0;
	readNumber("-m", mean);

	uint64_t seed = uint64_t(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	if (arg.count("-s") == 1) seed = uint64_t(strtoull(arg.find("-s")->second.c_str(), nullptr, 10));
	arg.erase("-s");

	std::vector<double> buffer;
	for (auto it = arg.lower_bound("-buf"); it != arg.upper_bound("-buf"); ++it) {
		double cells = 0.0;
		if (!gs::parseNumberText(it->second, &cells) || cells < 0) { fprintf(reportFile, "error: -buf expects numbers of cells >= 0\n"); run = false; }
		buffer.push_back(cells);
	}
	const bool bufferGiven = arg.count("-buf") > 0;
	arg.erase("-buf");
	arg.erase("-dt");

	for (auto it = arg.begin(); it != arg.end(); ++it)
		fprintf(reportFile, "%s %s <== ignored !\n", it->first.c_str(), it->second.c_str());

	if (method == sp::Method::Spectral && (!noiseFileName.empty() || outputNoise)) {
		fprintf(reportFile, "error: -ni and -on are only available with fftma (the spectral method draws its noise in frequency)\n");
		run = false;
	}
	if (!run) return failWith(error.empty() ? "invalid parameters, simulation interrupted" : error, 2);

	// ---- grid and model
	g2s::DataImage DI = g2s::DataImage::createFromFile(targetFileName);
	g2s::reporting::logInput(reportFile, "-di", targetFileName, DI);
	if (DI.isEmpty()) return failWith("failed to load -di " + targetFileName, 3);
	const std::vector<unsigned> grid = DI._dims;
	const unsigned ndim = unsigned(grid.size());
	if (ndim < 1 || ndim > 3) return failWith("-di must be a 1D, 2D or 3D grid, got " + std::to_string(ndim) + " dimensions", 2);

	gs::CovarianceModelSet models;
	std::vector<unsigned> modelGrid = grid;
	if (!gs::loadCovarianceModels(reportFile, covarianceArguments, modelGrid, ndim, models, error)) return failWith(error, 2);
	if (!models.isStationary())
		return failWith(std::string(sp::methodName(method)) + " needs a stationary covariance model: spatially varying -cvar images and several models with -ii are not supported", 2);
	const gs::ResolvedModel& model = models.stationaryModel();
	g2s::reporting::recordMetric(reportFile, "covariance_model", models.canonical()[0]);

	// ---- noise input (fftma)
	g2s::DataImage noiseImage;
	if (!noiseFileName.empty()) {
		noiseImage = g2s::DataImage::createFromFile(noiseFileName);
		g2s::reporting::logInput(reportFile, "-ni", noiseFileName, noiseImage);
		if (noiseImage.isEmpty()) return failWith("failed to load -ni " + noiseFileName, 3);
		if (noiseImage._dims.size() != ndim) return failWith("-ni must have the same number of dimensions as -di", 2);
		for (unsigned d = 0; d < ndim; ++d) {
			if (noiseImage._dims[d] < grid[d] || (periodic && noiseImage._dims[d] != grid[d]))
				return failWith("-ni has size " + gridText(noiseImage._dims) + "; it must cover the grid " + gridText(grid) + (periodic ? " exactly with -csim" : ""), 2);
		}
		if (!realisationsGiven) nbRealisations = noiseImage._nbVariable;
		else if (noiseImage._nbVariable != nbRealisations)
			return failWith("-ni has " + std::to_string(noiseImage._nbVariable) + " variable(s) but -nr is " + std::to_string(nbRealisations) + " (one noise variable per realisation)", 2);
		for (size_t i = 0; i < noiseImage.dataSize(); ++i)
			if (!std::isfinite(noiseImage._data[i])) return failWith("-ni contains non-finite values", 2);
	}

	// ---- extended grid
	const std::vector<double> practicalRange = models.maxPracticalRange();
	std::vector<double> bufferCells(ndim, 0.0);
	if (bufferGiven) {
		if (buffer.size() == 1) bufferCells.assign(ndim, buffer[0]);
		else if (buffer.size() == ndim) bufferCells = buffer;
		else return failWith("-buf takes one value or one per axis (" + std::to_string(ndim) + ")", 2);
	} else {
		for (unsigned d = 0; d < ndim; ++d) bufferCells[d] = std::ceil(practicalRange[d]);
	}
	std::vector<unsigned> extended;
	if (!noiseFileName.empty()) {
		extended = noiseImage._dims;
		if (bufferGiven) warn("-buf is ignored: the extended grid is the size of -ni");
		for (unsigned d = 0; d < ndim && !periodic; ++d) {
			if (grid[d] > 1 && double(extended[d]) < double(grid[d]) + std::ceil(practicalRange[d]))
				warn("-ni is smaller than the grid plus the practical range along axis " + std::to_string(d) + ": the periodic edge effect reaches the output");
		}
	} else {
		try {
			extended = sp::extendedGrid(grid, bufferCells, periodic);
		} catch (const gs::Error& e) {
			return failWith(e.what(), 2);
		}
	}
	if (periodic && bufferGiven) warn("-buf is ignored with -csim");
	double extendedCells = 1.0;
	for (unsigned d = 0; d < ndim; ++d) extendedCells *= double(extended[d]);
	if (extendedCells > 2147483648.0)
		return failWith("the extended grid " + gridText(extended) + " is too large; reduce the buffer (-buf) or use -csim", 2);
	if (periodic) {
		std::vector<unsigned> unfriendly;
		for (unsigned d = 0; d < ndim; ++d) if (!sp::isFftFriendly(grid[d])) unfriendly.push_back(d);
		if (!unfriendly.empty())
			warn("with -csim the grid size " + gridText(grid) + " has prime factors above 7: the FFTs are slower (the size is kept)");
	}

	std::string bufferText;
	for (unsigned d = 0; d < ndim; ++d) bufferText += (d ? "x" : "") + std::to_string(extended[d] - grid[d]);
	std::string rangeText;
	for (unsigned d = 0; d < ndim; ++d) rangeText += (d ? "x" : "") + gs::formatNumber(practicalRange[d]);
	g2s::reporting::logParameter(reportFile, "method", sp::methodName(method));
	g2s::reporting::logParameter(reportFile, "grid", gridText(grid));
	g2s::reporting::logParameter(reportFile, "practical_range", rangeText);
	g2s::reporting::logParameter(reportFile, "periodic", periodic ? "true" : "false");
	g2s::reporting::logParameter(reportFile, "extended_grid", gridText(extended));
	g2s::reporting::logParameter(reportFile, "buffer", bufferText);
	g2s::reporting::logParameter(reportFile, "realisations", std::to_string(nbRealisations));
	g2s::reporting::logParameter(reportFile, "first_realisation", std::to_string(firstRealisation));
	g2s::reporting::logParameter(reportFile, "mean", gs::formatNumber(mean));
	g2s::reporting::logParameter(reportFile, "seed", std::to_string((unsigned long long)seed));
	g2s::reporting::logParameter(reportFile, "threads", std::to_string(nbThreads));
	g2s::reporting::recordMetric(reportFile, "extended_grid", gridText(extended));

	// ---- simulation
	const auto begin = std::chrono::high_resolution_clock::now();
	gs::Conditioner conditioner;
	try {
		if (!gs::prepareGridConditioning(reportFile, DI, model, periodic, conditioning, nbThreads, conditioner, error)) return failWith(error, 2);
	} catch (const std::bad_alloc&) {
		return failWith("not enough memory to prepare the conditioning", 5);
	}
	g2s::DataImage simulation(ndim, const_cast<unsigned*>(grid.data()), nbRealisations);
	g2s::DataImage noiseOutput;
	if (outputNoise) noiseOutput = g2s::DataImage(ndim, extended.data(), nbRealisations);
	double spectrumTime = 0.0;
	try {
		const auto spectrumBegin = std::chrono::high_resolution_clock::now();
		sp::SpectralSimulator simulator(model, grid, extended, method, nbThreads);
		spectrumTime = 1.0e-6 * double(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - spectrumBegin).count());
		const sp::SpectrumReport& spectrum = simulator.report();
		char text[256];
		snprintf(text, sizeof(text), "%.3e", spectrum.clippedFraction);
		g2s::reporting::logParameter(reportFile, "clipped_energy_fraction", text);
		g2s::reporting::recordMetric(reportFile, "clipped_energy_fraction", text);
		g2s::reporting::logParameter(reportFile, "clipped_frequencies", std::to_string((unsigned long long)spectrum.clippedCount));
		g2s::reporting::logParameter(reportFile, "model_sill", gs::formatNumber(model.sill()));
		snprintf(text, sizeof(text), "%.6g", spectrum.effectiveSill);
		g2s::reporting::logParameter(reportFile, "simulated_variance", text);
		const double clippedWarning = 1e-3;
		if (spectrum.clippedFraction > clippedWarning) {
			snprintf(text, sizeof(text), "%.2f%%", 100.0 * spectrum.clippedFraction);
			warn(std::string("the covariance is not positive definite on the extended grid ") + gridText(extended) + ": " + text +
			     " of the spectral energy was clipped, so the simulated covariance deviates from the model; increase the buffer (-buf)" +
			     (periodic ? " or drop -csim" : ""));
		}
		sp::ConstFieldView noiseIn;
		if (!noiseImage.isEmpty()) noiseIn = { noiseImage._data, noiseImage._nbVariable, 1 };
		sp::FieldView noiseOut;
		if (outputNoise) noiseOut = { noiseOutput._data, nbRealisations, 1 };
		simulator.simulate(seed, firstRealisation, nbRealisations, mean, { simulation._data, nbRealisations, 1 }, noiseIn, noiseOut);
		conditioner.condition({ simulation._data, nbRealisations, 1 }, nbRealisations);
	} catch (const gs::Error& e) {
		return failWith(e.what(), 5);
	} catch (const std::bad_alloc&) {
		return failWith("not enough memory for the extended grid " + gridText(extended), 5);
	}
	const auto end = std::chrono::high_resolution_clock::now();
	const double time = 1.0e-6 * double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
	gs::reportConditioning(reportFile, conditioner);
	fprintf(reportFile, "spectrum time: %.0f ms\n", spectrumTime);
	fprintf(reportFile, "computation time: %.0f ms\n", time);
	g2s::reporting::recordMetric(reportFile, "duration_ms", std::to_string((long long)time));
	g2s::reporting::recordMetric(reportFile, "spectrum_ms", std::to_string((long long)spectrumTime));
	g2s::reporting::markFinished(reportFile, time);

	simulation.writeUnderNames({ outputFilename, std::string("im_1_") + std::to_string(uniqueID) });
	g2s::reporting::recordOutputDescriptor(reportFile, 1, "simulation", std::string("im_1_") + std::to_string(uniqueID));
	g2s::reporting::logOutput(reportFile, "simulation_image_runtime", std::string("im_1_") + std::to_string(uniqueID), simulation);
	if (outputNoise) {
		noiseOutput.writeUnderNames({ outputIndexFilename, std::string("im_2_") + std::to_string(uniqueID) });
		g2s::reporting::recordOutputDescriptor(reportFile, 2, "noise", std::string("im_2_") + std::to_string(uniqueID));
		g2s::reporting::logOutput(reportFile, "noise_image_runtime", std::string("im_2_") + std::to_string(uniqueID), noiseOutput);
	}
	return 0;
}
