/*
 * G2S - geostat: -W_GPU / -W_CUDA device selection.
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

#include "geostat/gpu/deviceSelection.hpp"

#include <chrono>
#include <cstdlib>
#include <dlfcn.h>

#include "geostat/gpu/openclDevice.hpp"

namespace g2s {
namespace geostat {
namespace gpu {

namespace {

bool parseIndex(const std::string& text, int& value) {
	if (text.empty()) return false;
	char* end = nullptr;
	const long parsed = std::strtol(text.c_str(), &end, 10);
	if (*end != '\0' || parsed < 0 || parsed > 1023) return false;
	value = int(parsed);
	return true;
}

std::string directoryOf(const char* path) {
	if (!path) return std::string();
	const std::string text(path);
	const size_t slash = text.rfind('/');
	return slash == std::string::npos ? std::string() : text.substr(0, slash + 1);
}

} // namespace

bool readGpuArguments(std::multimap<std::string, std::string>& arg, GpuRequest& request, std::string& error) {
	request = GpuRequest();
	auto range = arg.equal_range("-W_GPU");
	for (auto it = range.first; it != range.second; ++it) {
		request.openCL = true;
		if (it->second.empty()) continue;
		int index = 0;
		if (!parseIndex(it->second, index)) { error = "-W_GPU expects nothing or an OpenCL device index, got '" + it->second + "'"; return false; }
		if (request.openCLIndex < 0) request.openCLIndex = index;
	}
	arg.erase("-W_GPU");
	range = arg.equal_range("-W_CUDA");
	for (auto it = range.first; it != range.second; ++it) {
		request.cuda = true;
		if (it->second.empty()) continue;
		int index = 0;
		if (!parseIndex(it->second, index)) { error = "-W_CUDA expects nothing or CUDA device ids, got '" + it->second + "'"; return false; }
		request.cudaIndices.push_back(index);
	}
	arg.erase("-W_CUDA");
	return true;
}

std::string describe(const DeviceInfo& info) {
	return info.api + " " + info.name + " (device " + std::to_string(info.index) + ", " + std::to_string(info.globalMemory >> 20) + " MiB" +
	       (info.unifiedMemory ? ", unified memory" : "") + (info.fp64 ? ", fp64" : ", no fp64") + ")";
}

SelectedDevice::~SelectedDevice() { reset(); }

void SelectedDevice::reset() {
	_device.reset();
	if (_library) dlclose(_library);
	_library = nullptr;
}

SelectedDevice openDevice(const GpuRequest& request, const char* executablePath, FILE* report) {
	SelectedDevice selected;
	if (!report) report = stderr;
	const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	auto elapsed = [&]() { return 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); };
	if (request.cuda) {
		const int index = request.cudaIndices.empty() ? 0 : request.cudaIndices.front();
		if (request.cudaIndices.size() > 1) fprintf(report, "gpu: -W_CUDA lists %zu devices; this algorithm uses one, device %d\n", request.cudaIndices.size(), index);
		void* library = nullptr;
		std::string tried;
		std::vector<std::string> paths;
		const std::string beside = directoryOf(executablePath) + "g2s_cuda.so";
		if (beside != "g2s_cuda.so" && beside != "./g2s_cuda.so") paths.push_back(beside);
		paths.push_back("./g2s_cuda.so");
		for (const std::string& path : paths) {
			library = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
			if (library) break;
			const char* message = dlerror();
			tried += (tried.empty() ? "" : "; ") + path + ": " + (message ? message : "not found");
		}
		if (!library) {
			fprintf(report, "gpu: -W_CUDA: could not load g2s_cuda.so (%s); %s\n", tried.c_str(), request.openCL ? "trying OpenCL" : "running on the CPU");
		} else {
			CudaDeviceFunction create = reinterpret_cast<CudaDeviceFunction>(dlsym(library, kCudaDeviceSymbol));
			if (!create) {
				fprintf(report, "gpu: -W_CUDA: g2s_cuda.so has no %s (built before the geostat back end?); %s\n", kCudaDeviceSymbol, request.openCL ? "trying OpenCL" : "running on the CPU");
				dlclose(library);
			} else {
				char message[1024] = { 0 };
				Device* device = create(index, message, sizeof(message));
				if (!device) {
					fprintf(report, "gpu: -W_CUDA: device %d unavailable: %s; %s\n", index, message, request.openCL ? "trying OpenCL" : "running on the CPU");
					dlclose(library);
				} else {
					selected._device.reset(device);
					selected._library = library;
					fprintf(report, "gpu: using %s, ready in %.0f ms\n", describe(device->info()).c_str(), elapsed());
					return selected;
				}
			}
		}
	}
	if (request.openCL) {
		std::string message;
		Device* device = openOpenClDevice(request.openCLIndex, message);
		if (!device) {
			fprintf(report, "gpu: -W_GPU: %s; running on the CPU\n", message.c_str());
		} else {
			selected._device.reset(device);
			fprintf(report, "gpu: using %s, ready in %.0f ms (context and kernel compilation)\n", describe(device->info()).c_str(), elapsed());
		}
	}
	return selected;
}

const char* gpuHelp() {
	return "  -W_GPU [n]  OpenCL GPU device n (default: the first); -W_CUDA [n]  CUDA device n (default 0;\n"
	       "        needs g2s_cuda.so next to the executable). The CPU stays the reference: without a\n"
	       "        device, library or capability, or where the device is slower, the run uses the CPU\n"
	       "        and says so in the log. GPU results match the CPU within a documented tolerance,\n"
	       "        not bit for bit; on one device they are reproducible for a given seed.\n";
}

} // namespace gpu
} // namespace geostat
} // namespace g2s
