/*
 * G2S - geostat: -W_GPU / -W_CUDA device selection (src/geostatGpu.cpp).
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

// The flags of qs, for tb, fftma/spectral and kriging:
//   -W_GPU          OpenCL: the first GPU device (of any platform);
//   -W_GPU n        OpenCL: GPU device n (in the order of `openclDeviceNames`);
//   -W_CUDA         CUDA: device 0 (g2s_cuda.so, loaded with dlopen);
//   -W_CUDA n ...   CUDA: device n (the first id is used; one device per run).
// With both, CUDA is tried first. Any failure (no library, no device, a
// kernel the device cannot run) is logged and the run continues on the CPU.

#ifndef G2S_GEOSTAT_GPU_DEVICE_SELECTION_HPP
#define G2S_GEOSTAT_GPU_DEVICE_SELECTION_HPP

#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "geostat/gpu/backend.hpp"

namespace g2s {
namespace geostat {
namespace gpu {

struct GpuRequest {
	bool openCL = false;
	int openCLIndex = -1;          // -1: the first device
	bool cuda = false;
	std::vector<int> cudaIndices;  // empty: device 0
	bool any() const { return openCL || cuda; }
};

// Reads and removes -W_GPU and -W_CUDA from the arguments. False (with the
// error) for a malformed device index.
bool readGpuArguments(std::multimap<std::string, std::string>& arg, GpuRequest& request, std::string& error);

// A device and, for CUDA, the library it lives in (closed after the device).
class SelectedDevice {
public:
	SelectedDevice() = default;
	SelectedDevice(const SelectedDevice&) = delete;
	SelectedDevice& operator=(const SelectedDevice&) = delete;
	SelectedDevice(SelectedDevice&& other) noexcept : _device(std::move(other._device)), _library(other._library) { other._library = nullptr; }
	SelectedDevice& operator=(SelectedDevice&& other) noexcept {
		if (this != &other) {
			reset();
			_device = std::move(other._device);
			_library = other._library;
			other._library = nullptr;
		}
		return *this;
	}
	~SelectedDevice();
	Device* get() const { return _device.get(); }
	explicit operator bool() const { return bool(_device); }
	void reset();

private:
	friend SelectedDevice openDevice(const GpuRequest&, const char*, FILE*);
	std::unique_ptr<Device> _device;
	void* _library = nullptr;
};

// The requested device, or an empty handle (CPU). Logs to `report`: the device
// chosen, or why there is none. executablePath (argv[0]) locates g2s_cuda.so
// next to the executable (then ./g2s_cuda.so, as qs).
SelectedDevice openDevice(const GpuRequest& request, const char* executablePath, FILE* report);

// "OpenCL Apple M1 Pro (device 0, 25559 MiB, unified memory, no fp64)"
std::string describe(const DeviceInfo& info);

// The help text of the flags, for the -h of the executables.
const char* gpuHelp();

} // namespace gpu
} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_GPU_DEVICE_SELECTION_HPP
