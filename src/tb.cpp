/*
 * G2S - tb: unconditional Gaussian simulation by turning bands.
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

// tb: grid front end of the turning-bands engine (geostat/turningBands.hpp).
//
//   g2s('-a','tb','-di',nan(200,300),'-dt',[0],'-cov','0.1*nugget + 0.9*spherical(range=[40,20], angle=30)','-nr',10,'-s',7)
//   g2s('-a','tb','-di',data,'-dt',[0],'-cov',model,'-nr',16,'-km','local','-n',30)   % conditional
//   (-dt: the interfaces need the data type of -di, one 0 = continuous)
//
// The grid is the -di image; node coordinates are cell indices in DataImage
// axis order, the same units as the covariance ranges. Its informed (non-NaN)
// cells are hard data: each realisation is conditioned by kriging of its
// residuals (geostat/conditioning.hpp, flags in
// geostat/conditioningArguments.hpp); an all-NaN -di gives the unconditional
// simulation. Output 1 has the grid of -di and one variable per realisation.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "utils.hpp"
#include "DataImage.hpp"
#include "jobManager.hpp"
#include "jobReporting.hpp"
#include "geostat/conditioningArguments.hpp"
#include "geostat/covarianceArguments.hpp"
#include "geostat/turningBands.hpp"
#include "geostat/gpu/deviceEngines.hpp"
#include "geostat/gpu/deviceSelection.hpp"

namespace {

void printHelp() {
	printf("tb: Gaussian simulation by turning bands, conditioned to the informed cells of -di.\n"
	       "  -di   destination image: gives the grid (1D, 2D or 3D) (mandatory); non-NaN cells are hard data\n"
	       "        (all NaN: unconditional)\n"
	       "  -cov  covariance model, e.g. \"0.1*nugget + 0.9*spherical(range=[40,20], angle=30)\" (mandatory)\n"
	       "        supported: nugget, spherical, exponential, gaussian, cubic, matern, stable (not map)\n"
	       "  -cvar name value  binds a model variable (constant; spatially varying values are not supported)\n"
	       "        (-ii and several -cov are not supported either)\n"
	       "  -nb   number of lines per structure (default 1000, at most 16777215)\n"
	       "  -nr   number of realisations, output as variables (default 1, at most 100000)\n"
	       "  -ri   index of the first realisation (default 0): realisation r of a run with -ri 0\n"
	       "        equals realisation 0 of a run with -ri r\n"
	       "  -m    mean (default 0)\n"
	       "  -s    seed, an integer in [0, 2^64 - 1] (default: from the clock)\n"
	       "  -j    threads (a fraction = share of the cores); the result does not depend on -j\n"
	       "  -csim is refused: turning bands fields are not periodic\n"
	       "%s"
	       "%s"
	       "        tb: the sum over the lines (grid path) runs on the device; line parameters, nugget,\n"
	       "        mean and conditioning stay on the CPU. Phases are exact (64-bit fixed point), so no\n"
	       "        double precision is needed; |gpu - cpu| <= 2e-5 sqrt(sill). On devices sharing the\n"
	       "        host memory (Apple silicon) spectral lines (exponential, gaussian, matern, stable)\n"
	       "        are about 1.2-2x faster than 8 CPU threads and dilution lines (spherical, cubic)\n"
	       "        are not faster; with ~0.3 s to start the device, it is used from 1e11 node-lines\n"
	       "        in total (e.g. 200x200x200, 1000 lines, 16 realisations); below, the CPU\n"
	       "  -gpuforce  use the device even where the CPU is expected to be faster\n"
	       "  -o    output name\n", g2s::geostat::conditioningHelp(), g2s::geostat::gpu::gpuHelp());
}

// Reads "-flag value" from argv when value may be a negative number
// (argumentReader takes "-3" for a flag); removes the bogus key.
bool readSignedNumber(std::multimap<std::string, std::string>& arg, int argc, char const* argv[], const std::string& flag, double& value, bool& given) {
	given = false;
	for (int i = 0; i + 1 < argc; ++i) {
		if (flag != argv[i]) continue;
		const std::string token = argv[i + 1];
		if (!g2s::geostat::parseNumberText(token, &value)) return false;
		if (token.size() > 1 && token[0] == '-') arg.erase(token);
		given = true;
	}
	if (!given && arg.count(flag) > 0) return false; // flag without value
	arg.erase(flag);
	return true;
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
	g2s::reporting::markStarted(reportFile, "tb");

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
	g2s::geostat::ConditioningArguments conditioning;
	{
		std::string conditioningError;
		if (!g2s::geostat::readConditioningArguments(arg, conditioning, conditioningError)) {
			fprintf(reportFile, "error: %s\n", conditioningError.c_str());
			if (error.empty()) error = conditioningError;
			run = false;
		}
	}

	double mean = 0.0;
	bool meanGiven = false;
	if (!readSignedNumber(arg, argc, argv, "-m", mean, meanGiven)) {
		error = "-m expects one number";
		fprintf(reportFile, "error: %s\n", error.c_str());
		run = false;
	}

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

	std::string targetFileName;
	if (arg.count("-di") == 1) targetFileName = arg.find("-di")->second;
	else { fprintf(reportFile, "error: -di is mandatory\n"); run = false; }
	arg.erase("-di");

	unsigned lineCount = 1000;
	if (arg.count("-nb") == 1) {
		const long value = atol(arg.find("-nb")->second.c_str());
		if (value < 1 || value >= (1L << 24)) { error = "-nb expects an integer in [1, 16777215]"; fprintf(reportFile, "error: %s\n", error.c_str()); run = false; }
		else lineCount = unsigned(value);
	}
	arg.erase("-nb");

	unsigned nbRealisations = 1;
	if (arg.count("-nr") == 1) {
		const long value = atol(arg.find("-nr")->second.c_str());
		if (value < 1 || value > 100000) { error = "-nr expects an integer in [1, 100000]"; fprintf(reportFile, "error: %s\n", error.c_str()); run = false; }
		else nbRealisations = unsigned(value);
	}
	arg.erase("-nr");

	unsigned firstRealisation = 0;
	if (arg.count("-ri") == 1) {
		const std::string text = arg.find("-ri")->second;
		double value = -1.0;
		if (!g2s::geostat::parseNumberText(text, &value) || !(value >= 0 && value <= 4294967295.0 - double(nbRealisations) && value == std::floor(value))) {
			error = "-ri expects an integer >= 0";
			fprintf(reportFile, "error: %s\n", error.c_str());
			run = false;
		} else firstRealisation = unsigned(value);
	} else if (arg.count("-ri") > 1) { error = "-ri expects an integer >= 0"; fprintf(reportFile, "error: %s\n", error.c_str()); run = false; }
	arg.erase("-ri");

	uint64_t seed = 0;
	if (!g2s::geostat::readSeedArgument(arg, seed, error)) { fprintf(reportFile, "error: %s\n", error.c_str()); run = false; }

	if (arg.count("-csim") >= 1) {
		error = "-csim (periodic simulation) is not available with turning bands: the fields are not periodic; remove -csim";
		fprintf(reportFile, "error: %s\n", error.c_str());
		run = false;
	}
	arg.erase("-csim");

	g2s::geostat::gpu::GpuRequest gpuRequest;
	{
		std::string gpuError;
		if (!g2s::geostat::gpu::readGpuArguments(arg, gpuRequest, gpuError)) { error = gpuError; fprintf(reportFile, "error: %s\n", error.c_str()); run = false; }
	}
	const bool gpuForce = arg.count("-gpuforce") > 0;
	arg.erase("-gpuforce");

	std::string outputFilename = std::to_string(uniqueID);
	if (arg.count("-o") == 1) outputFilename = arg.find("-o")->second;
	arg.erase("-o");
	arg.erase("-dt");

	for (auto it = arg.begin(); it != arg.end(); ++it)
		fprintf(reportFile, "%s %s <== ignored !\n", it->first.c_str(), it->second.c_str());

	if (!run) return failWith(error.empty() ? "invalid parameters, simulation interrupted" : error, 2);

	// ---- grid and model
	g2s::DataImage DI = g2s::DataImage::createFromFile(targetFileName);
	g2s::reporting::logInput(reportFile, "-di", targetFileName, DI);
	if (DI.isEmpty()) return failWith("failed to load -di " + targetFileName, 3);
	std::vector<unsigned> grid = DI._dims;
	const unsigned ndim = unsigned(grid.size());
	if (ndim < 1 || ndim > 3) return failWith("tb simulates 1D, 2D or 3D grids; -di has " + std::to_string(ndim) + " dimensions", 2);

	g2s::geostat::CovarianceModelSet models;
	if (!g2s::geostat::loadCovarianceModels(reportFile, covarianceArguments, grid, ndim, models, error)) return failWith(error, 2);
	const g2s::geostat::ResolvedModel* modelPointer = nullptr;
	try {
		modelPointer = &g2s::geostat::turningBandsModel(models);
	} catch (const g2s::geostat::Error& e) {
		return failWith(e.what(), 2);
	}
	const g2s::geostat::ResolvedModel& model = *modelPointer;
	const std::vector<std::string> canonical = models.canonical();
	for (size_t i = 0; i < canonical.size(); ++i) g2s::reporting::recordMetric(reportFile, "covariance_model_" + std::to_string(i), canonical[i]);

	g2s::geostat::TurningBandsSettings settings;
	settings.lineCount = lineCount;
	settings.seed = seed;
	settings.threads = nbThreads;
	settings.mean = mean;
	g2s::reporting::logParameter(reportFile, "seed", std::to_string((unsigned long long)seed));
	g2s::reporting::logParameter(reportFile, "threads", std::to_string(nbThreads));
	g2s::reporting::logParameter(reportFile, "lines_per_structure", std::to_string(lineCount));
	g2s::reporting::logParameter(reportFile, "realisations", std::to_string(nbRealisations));
	g2s::reporting::logParameter(reportFile, "first_realisation", std::to_string(firstRealisation));
	g2s::reporting::logParameter(reportFile, "mean", g2s::geostat::formatNumber(mean));
	for (size_t s = 0; s < model.structures.size(); ++s) {
		g2s::geostat::LineProcess process = g2s::geostat::LineProcess::Spectral;
		g2s::geostat::turningBandsLineProcess(model.structures[s], process);
		const bool nugget = model.structures[s].type == g2s::geostat::StructureType::Nugget;
		g2s::reporting::logParameter(reportFile, "structure_" + std::to_string(s + 1),
		                             std::string(g2s::geostat::structureName(model.structures[s].type)) + ": " + (nugget ? "independent noise" : g2s::geostat::lineProcessName(process)));
	}

	// ---- simulation: the grid path of the engine (blocks of rows evaluated with
	// per-axis tables, see geostat/turningBands.hpp) on a persistent pool of -j
	// threads; several realisations are prepared and evaluated together when the
	// grid has few blocks. Values do not depend on -j.
	unsigned outputDims[3] = { 0, 0, 0 };
	for (unsigned d = 0; d < ndim; ++d) outputDims[d] = grid[d];
	g2s::DataImage output(ndim, outputDims, nbRealisations);
	output._types.assign(nbRealisations, g2s::DataImage::Continuous);
	const g2s::geostat::RegularGrid regularGrid = g2s::geostat::RegularGrid::cells(ndim, grid.data());

	// ---- conditioning to the informed cells of -di (factorisation or neighbour search)
	const auto start = std::chrono::high_resolution_clock::now();
	g2s::geostat::Conditioner conditioner;
	try {
		if (!g2s::geostat::prepareGridConditioning(reportFile, DI, model, false, conditioning, nbThreads, conditioner, error)) return failWith(error, 2);
	} catch (const std::bad_alloc&) {
		return failWith("not enough memory to prepare the conditioning", 5);
	}

	// ---- optional device (-W_GPU / -W_CUDA): the sum over the lines on the GPU
	g2s::geostat::gpu::SelectedDevice device;
	std::unique_ptr<g2s::geostat::gpu::TurningBandsKernel> deviceKernel;
	if (gpuRequest.any()) {
		device = g2s::geostat::gpu::openDevice(gpuRequest, argv[0], reportFile);
		if (device) {
			std::string reason;
			deviceKernel.reset(device.get()->turningBands(reason));
			if (!deviceKernel) fprintf(reportFile, "gpu: turning bands unavailable on the device (%s); running on the CPU\n", reason.c_str());
		}
		if (deviceKernel && !gpuForce) {
			// Measured on an Apple M1 Pro (OpenCL) against 8 CPU threads: spectral lines 1.2-2x
			// faster (about 3 ms saved per 1e9 node-lines), dilution lines slightly slower, and
			// about 0.3 s to open the device and compile the kernels: worth it from 1e11
			// node-lines in total. Devices with their own memory (discrete GPUs, CUDA; not
			// measured here) are expected to win from 1e9.
			size_t spectral = 0, dilution = 0;
			for (const g2s::geostat::ResolvedStructure& structure : model.structures) {
				g2s::geostat::LineProcess process = g2s::geostat::LineProcess::Spectral;
				if (structure.type == g2s::geostat::StructureType::Nugget || !(structure.sill > 0)) continue;
				g2s::geostat::turningBandsLineProcess(structure, process);
				(process == g2s::geostat::LineProcess::Spectral ? spectral : dilution) += 1;
			}
			const double work = double(regularGrid.nodeCount()) * double(lineCount) * double(spectral + dilution) * double(nbRealisations);
			std::string reason;
			if (spectral + dilution == 0) reason = "the model is a pure nugget";
			else if (device.get()->info().unifiedMemory && dilution > spectral) reason = "dilution lines (spherical, cubic) are not faster on a device sharing the host memory";
			else if (work < (device.get()->info().unifiedMemory ? 1e11 : 1e9)) reason = "the work (nodes x lines x realisations) is too small for the device to pay off";
			if (!reason.empty()) {
				fprintf(reportFile, "gpu: %s; running on the CPU (-gpuforce to override)\n", reason.c_str());
				deviceKernel.reset();
			}
		}
	}
	g2s::reporting::logParameter(reportFile, "device", deviceKernel ? g2s::geostat::gpu::describe(device.get()->info()) : std::string("cpu"));

	const auto begin = std::chrono::high_resolution_clock::now();
	bool simulatedOnDevice = false;
	if (deviceKernel) {
		g2s::geostat::WorkerPool pool(nbThreads);
		float* data = output._data;
		const size_t columns = regularGrid.size[0];
		std::string deviceError;
		simulatedOnDevice = g2s::geostat::gpu::simulateTurningBandsGridDevice(model, settings, regularGrid, firstRealisation, nbRealisations, *deviceKernel, pool,
			[&](unsigned r, size_t rowBegin, size_t rowEnd, const float* values) {
				const size_t count = (rowEnd - rowBegin) * columns;
				float* target = data + rowBegin * columns * nbRealisations + r;
				for (size_t i = 0; i < count; ++i) target[i * nbRealisations] = values[i];
			},
			[&](unsigned done) {
				g2s::reporting::setProgress(reportFile, 100.0 * double(done) / double(nbRealisations), "simulation",
				                            "realisation " + std::to_string(done) + "/" + std::to_string(nbRealisations), done, nbRealisations);
			}, deviceError);
		if (!simulatedOnDevice) fprintf(reportFile, "gpu: the device failed (%s); simulating on the CPU\n", deviceError.c_str());
		else {
			const g2s::geostat::gpu::DeviceTimes times = deviceKernel->times();
			fprintf(reportFile, "gpu: device time %.0f ms (upload %.0f, kernels %.0f, download %.0f)\n", 1e3 * times.total(), 1e3 * times.upload, 1e3 * times.compute, 1e3 * times.download);
		}
	}
	if (!simulatedOnDevice) {
		g2s::geostat::WorkerPool pool(nbThreads);
		float* data = output._data;
		const size_t columns = regularGrid.size[0];
		g2s::geostat::simulateTurningBandsGrid(model, settings, regularGrid, firstRealisation, nbRealisations, pool,
			[&](unsigned r, const g2s::geostat::GridBlock& block, const double* values) {
				const size_t width = block.width();
				for (size_t row = block.rowBegin; row < block.rowEnd; ++row) {
					const double* source = values + (row - block.rowBegin) * width;
					float* target = data + (row * columns + block.columnBegin) * nbRealisations + r;
					for (size_t i = 0; i < width; ++i) target[i * nbRealisations] = float(source[i]);
				}
			},
			[&](unsigned done) {
				g2s::reporting::setProgress(reportFile, 100.0 * double(done) / double(nbRealisations), "simulation",
				                            "realisation " + std::to_string(done) + "/" + std::to_string(nbRealisations), done, nbRealisations);
			});
	}
	const auto simulated = std::chrono::high_resolution_clock::now();
	conditioner.condition({ output._data, nbRealisations, 1 }, nbRealisations);
	const auto end = std::chrono::high_resolution_clock::now();
	g2s::geostat::reportConditioning(reportFile, conditioner);
	const double simulationTime = 1.0e-6 * double(std::chrono::duration_cast<std::chrono::nanoseconds>(simulated - begin).count());
	fprintf(reportFile, "simulation time: %.0f ms\n", simulationTime);
	g2s::reporting::recordMetric(reportFile, "simulation_ms", std::to_string((long long)simulationTime));

	const double time = 1.0e-6 * double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
	fprintf(reportFile, "computation time: %7.2f s\n", time / 1000);
	fprintf(reportFile, "computation time: %.0f ms\n", time);
	g2s::reporting::recordMetric(reportFile, "duration_ms", std::to_string((long long)time));
	g2s::reporting::markFinished(reportFile, time);

	output.writeUnderNames({ outputFilename, std::string("im_1_") + std::to_string(uniqueID) }, true, true);
	g2s::reporting::recordOutputDescriptor(reportFile, 1, "simulation", std::string("im_1_") + std::to_string(uniqueID));
	g2s::reporting::logOutput(reportFile, "simulation_image_runtime", std::string("im_1_") + std::to_string(uniqueID), output);
	return 0;
}
