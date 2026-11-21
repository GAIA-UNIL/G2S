/*
 * G2S - geostat: OpenCL back end of the GPU kernels (gpu/backend.hpp).
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

// OpenCL 1.2 (the version Apple provides), kernels from
// gpu/geostatKernels.inc compiled at run time, transforms with clFFT (as the
// QS OpenCL device, src/OpenCLGPUDevice.cpp). Compiled in under WITH_OPENCL;
// without it the functions report that OpenCL is not available.
//
// Apple notes (macOS OpenCL is deprecated but present on Apple silicon):
//   * no cl_khr_fp64: the kriging kernels are refused (KrigingKernel needs double);
//   * OpenCL 1.2 only: clCreateCommandQueue (deprecated in 2.0) is the call to use;
//   * device memory is host memory (CL_DEVICE_HOST_UNIFIED_MEMORY), transfers
//     are memory copies.

#include "geostat/gpu/openclDevice.hpp"

#ifndef WITH_OPENCL

namespace g2s {
namespace geostat {
namespace gpu {
bool openclAvailable() { return false; }
std::vector<std::string> openclDeviceNames() { return {}; }
Device* openOpenClDevice(int, std::string& message) {
	message = "this executable was built without OpenCL (WITH_OPENCL; see build/Makefile, `make clfft`)";
	return nullptr;
}
KrigingKernel* openclKrigingFloatForTests(Device*, std::string& message) {
	message = "built without OpenCL";
	return nullptr;
}
} // namespace gpu
} // namespace geostat
} // namespace g2s

#else

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/cl.h>
#endif
#include <clFFT.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <vector>

#include "geostat/gpu/kernelPrelude.hpp"

namespace g2s {
namespace geostat {
namespace gpu {

namespace {

#define G2S_GPU_SOURCE(...) #__VA_ARGS__
const char* const kKernelSource =
#include "geostat/gpu/geostatKernels.inc"
;
#undef G2S_GPU_SOURCE

using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point since) { return std::chrono::duration<double>(Clock::now() - since).count(); }

uint64_t roundUp(uint64_t value, uint64_t multiple) { return (value + multiple - 1) / multiple * multiple; }

std::string clError(const char* what, cl_int code) { return std::string("OpenCL: ") + what + " failed (error " + std::to_string(code) + ")"; }

std::vector<std::pair<cl_platform_id, cl_device_id> > gpuDevices() {
	std::vector<std::pair<cl_platform_id, cl_device_id> > result;
	cl_uint platformCount = 0;
	if (clGetPlatformIDs(0, nullptr, &platformCount) != CL_SUCCESS || platformCount == 0) return result;
	std::vector<cl_platform_id> platforms(platformCount);
	clGetPlatformIDs(platformCount, platforms.data(), nullptr);
	for (cl_platform_id platform : platforms) {
		cl_uint count = 0;
		if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count) != CL_SUCCESS || count == 0) continue;
		std::vector<cl_device_id> devices(count);
		clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, count, devices.data(), nullptr);
		for (cl_device_id device : devices) result.push_back({ platform, device });
	}
	return result;
}

std::string deviceString(cl_device_id device, cl_device_info what) {
	size_t size = 0;
	clGetDeviceInfo(device, what, 0, nullptr, &size);
	std::string text(size, '\0');
	clGetDeviceInfo(device, what, size, &text[0], nullptr);
	while (!text.empty() && text.back() == '\0') text.pop_back();
	return text;
}

// clFFT is a process-wide library: set up once, torn down with the last device.
std::mutex clfftMutex;
int clfftUsers = 0;

class Buffer {
public:
	Buffer() = default;
	Buffer(const Buffer&) = delete;
	Buffer& operator=(const Buffer&) = delete;
	~Buffer() { release(); }
	bool ensure(cl_context context, size_t bytes, std::string& error) {
		if (bytes <= _bytes && _mem) return true;
		release();
		cl_int status = CL_SUCCESS;
		_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, std::max<size_t>(bytes, 16), nullptr, &status);
		if (status != CL_SUCCESS || !_mem) { _mem = nullptr; error = clError("clCreateBuffer", status) + " for " + std::to_string(bytes >> 20) + " MiB"; return false; }
		_bytes = bytes;
		return true;
	}
	cl_mem get() const { return _mem; }
	cl_mem* address() { return &_mem; }
	void release() {
		if (_mem) clReleaseMemObject(_mem);
		_mem = nullptr;
		_bytes = 0;
	}

private:
	cl_mem _mem = nullptr;
	size_t _bytes = 0;
};

template <class T>
cl_int setArg(cl_kernel kernel, cl_uint index, const T& value) { return clSetKernelArg(kernel, index, sizeof(T), &value); }
inline cl_int setArg(cl_kernel kernel, cl_uint index, const Buffer& buffer) { cl_mem mem = buffer.get(); return clSetKernelArg(kernel, index, sizeof(cl_mem), &mem); }

template <class... Args>
cl_int setArgs(cl_kernel kernel, const Args&... args) {
	cl_int status = CL_SUCCESS;
	cl_uint index = 0;
	using expand = int[];
	(void)expand{ 0, (status = (status == CL_SUCCESS ? setArg(kernel, index++, args) : status), 0)... };
	return status;
}

class OpenClDevice;

// A kernel object of the program, released with its owner.
class Kernel {
public:
	Kernel() = default;
	Kernel(const Kernel&) = delete;
	Kernel& operator=(const Kernel&) = delete;
	~Kernel() { if (_kernel) clReleaseKernel(_kernel); }
	bool create(cl_program program, const char* name, std::string& error) {
		cl_int status = CL_SUCCESS;
		_kernel = clCreateKernel(program, name, &status);
		if (status != CL_SUCCESS) { _kernel = nullptr; error = clError((std::string("clCreateKernel ") + name).c_str(), status); return false; }
		return true;
	}
	cl_kernel get() const { return _kernel; }

private:
	cl_kernel _kernel = nullptr;
};

class OpenClDevice : public Device {
public:
	OpenClDevice(cl_platform_id platform, cl_device_id device, int index) : _platform(platform), _device(device) {
		_info.api = "OpenCL";
		_info.index = index;
		_info.name = deviceString(device, CL_DEVICE_NAME);
		cl_ulong memory = 0, allocation = 0;
		clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memory), &memory, nullptr);
		clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(allocation), &allocation, nullptr);
		_info.globalMemory = memory;
		_info.maxAllocation = allocation;
		cl_device_fp_config fp64 = 0;
		clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp64), &fp64, nullptr);
		const std::string extensions = deviceString(device, CL_DEVICE_EXTENSIONS);
		_info.fp64 = fp64 != 0 && extensions.find("cl_khr_fp64") != std::string::npos;
		cl_bool unified = CL_FALSE;
		clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified), &unified, nullptr);
		_info.unifiedMemory = unified == CL_TRUE;
		cl_device_fp_config single = 0;
		clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(single), &single, nullptr);
		_correctlyRoundedSqrt = (single & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) != 0;
	}
	~OpenClDevice() override {
		if (_program) clReleaseProgram(_program);
		if (_queue) clReleaseCommandQueue(_queue);
		if (_context) clReleaseContext(_context);
		if (_clfft) {
			std::lock_guard<std::mutex> lock(clfftMutex);
			if (--clfftUsers == 0) clfftTeardown();
		}
	}

	bool initialise(std::string& error) {
		cl_int status = CL_SUCCESS;
		const cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM, cl_context_properties(_platform), 0 };
		_context = clCreateContext(properties, 1, &_device, nullptr, nullptr, &status);
		if (status != CL_SUCCESS) { _context = nullptr; error = clError("clCreateContext", status); return false; }
		_queue = clCreateCommandQueue(_context, _device, 0, &status);
		if (status != CL_SUCCESS) { _queue = nullptr; error = clError("clCreateCommandQueue", status); return false; }
		const std::string source = std::string(kOpenClPrelude) + kKernelSource;
		const char* text = source.c_str();
		_program = clCreateProgramWithSource(_context, 1, &text, nullptr, &status);
		if (status != CL_SUCCESS) { _program = nullptr; error = clError("clCreateProgramWithSource", status); return false; }
		std::string options = "-cl-std=CL1.2";
		if (_info.fp64) options += " -DG2S_FP64";
		if (_correctlyRoundedSqrt) options += " -cl-fp32-correctly-rounded-divide-sqrt";
		status = clBuildProgram(_program, 1, &_device, options.c_str(), nullptr, nullptr);
		if (status != CL_SUCCESS) {
			size_t size = 0;
			clGetProgramBuildInfo(_program, _device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
			std::string log(size, '\0');
			clGetProgramBuildInfo(_program, _device, CL_PROGRAM_BUILD_LOG, size, &log[0], nullptr);
			error = clError("clBuildProgram", status) + ": " + log;
			return false;
		}
		return true;
	}

	bool useClfft(std::string& error) {
		if (_clfft) return true;
		std::lock_guard<std::mutex> lock(clfftMutex);
		if (clfftUsers == 0) {
			clfftSetupData setup;
			clfftStatus status = clfftInitSetupData(&setup);
			if (status == CLFFT_SUCCESS) status = clfftSetup(&setup);
			if (status != CLFFT_SUCCESS) { error = "clFFT: setup failed (error " + std::to_string(int(status)) + ")"; return false; }
		}
		++clfftUsers;
		_clfft = true;
		return true;
	}

	const DeviceInfo& info() const override { return _info; }
	TurningBandsKernel* turningBands(std::string& reason) override;
	SpectralKernel* spectral(std::string& reason) override;
	KrigingKernel* kriging(std::string& reason) override;
	KrigingKernel* krigingFloat(std::string& reason);

	cl_context context() const { return _context; }
	cl_command_queue queue() const { return _queue; }
	cl_program program() const { return _program; }
	// Device memory one kernel object may use (buffers of one chunk).
	uint64_t budget() const {
		uint64_t budget = std::min<uint64_t>(_info.globalMemory / 4, uint64_t(1) << 30);
		return std::max<uint64_t>(budget, uint64_t(64) << 20);
	}

	cl_int launch(const Kernel& kernel, unsigned dims, const size_t* global, const size_t* local) const {
		return clEnqueueNDRangeKernel(_queue, kernel.get(), dims, nullptr, global, local, 0, nullptr, nullptr);
	}

private:
	cl_platform_id _platform;
	cl_device_id _device;
	DeviceInfo _info;
	bool _correctlyRoundedSqrt = false;
	bool _clfft = false;
	cl_context _context = nullptr;
	cl_command_queue _queue = nullptr;
	cl_program _program = nullptr;
};

// ============================================================ turning bands

class OpenClTurningBands : public TurningBandsKernel {
public:
	explicit OpenClTurningBands(OpenClDevice& device) : _device(device) {}

	bool initialise(std::string& error) {
		return _rowTable.create(_device.program(), "tbRowTable", error) && _columnTable.create(_device.program(), "tbColumnTable", error) &&
		       _spectral.create(_device.program(), "tbSpectralProduct", error) && _dilution.create(_device.program(), "tbDilutionProduct", error);
	}

	bool sum(const TurningBandsLines* sets, size_t setCount, const uint64_t size[3], uint64_t rowBegin, uint64_t rowEnd, float* out, std::string& error) override {
		if (rowEnd <= rowBegin) return true;
		const uint64_t columns = size[0], columnPitch = roundUp(columns, 64);
		if (columns > 0xFFFFFFFFull) { error = "turning bands on the GPU: more than 2^32 columns"; return false; }
		uint32_t maxLines = 16;
		for (size_t s = 0; s < setCount; ++s) maxLines = std::max(maxLines, sets[s].count);
		const uint64_t maxPitch = roundUp(maxLines, 16);
		const uint64_t columnBytes = maxPitch * columnPitch * 8;
		const uint64_t rowBytes = maxPitch * 8 + columnPitch * 4;
		const uint64_t available = _device.budget() > columnBytes ? _device.budget() - columnBytes : 0;
		uint64_t chunkRows = std::min<uint64_t>(roundUp(rowEnd - rowBegin, 128), std::max<uint64_t>(128, available / rowBytes / 128 * 128));
		chunkRows = std::min<uint64_t>(chunkRows, uint64_t(1) << 20);
		const cl_context context = _device.context();
		if (!_whole.ensure(context, 4 * maxPitch * 4, error) || !_fraction.ensure(context, 4 * maxPitch * 8, error) || !_keys.ensure(context, 2 * maxPitch * 4, error) ||
		    !_columns.ensure(context, columnBytes, error) || !_rows.ensure(context, maxPitch * chunkRows * 8, error) || !_out.ensure(context, chunkRows * columnPitch * 4, error))
			return false;
		std::vector<uint32_t> whole, keys;
		std::vector<uint64_t> fraction;
		const cl_command_queue queue = _device.queue();
		for (uint64_t first = rowBegin; first < rowEnd; first += chunkRows) {
			const uint64_t rows = std::min(chunkRows, rowEnd - first), rowPitch = roundUp(rows, 128);
			Clock::time_point start = Clock::now();
			bool any = false;
			for (size_t s = 0; s < setCount; ++s) {
				const TurningBandsLines& set = sets[s];
				if (set.count == 0) continue;
				const uint32_t count = set.count, pitch = uint32_t(roundUp(count, 16));
				const cl_uint spectral = set.process == 0 ? 1u : 0u;
				// Plain arrays padded to the pitch.
				start = Clock::now();
				whole.assign(4 * size_t(pitch), 0u);
				fraction.assign(4 * size_t(pitch), 0u);
				keys.assign(2 * size_t(pitch), 0u);
				for (unsigned j = 0; j < 4; ++j)
					for (uint32_t l = 0; l < count; ++l) {
						whole[j * pitch + l] = set.whole[j * size_t(count) + l];
						fraction[j * pitch + l] = set.fraction[j * size_t(count) + l];
					}
				if (!spectral && set.signKey)
					for (unsigned j = 0; j < 2; ++j)
						for (uint32_t l = 0; l < count; ++l) keys[j * pitch + l] = set.signKey[j * size_t(count) + l];
				cl_int status = clEnqueueWriteBuffer(queue, _whole.get(), CL_FALSE, 0, whole.size() * 4, whole.data(), 0, nullptr, nullptr);
				if (status == CL_SUCCESS) status = clEnqueueWriteBuffer(queue, _fraction.get(), CL_FALSE, 0, fraction.size() * 8, fraction.data(), 0, nullptr, nullptr);
				if (status == CL_SUCCESS) status = clEnqueueWriteBuffer(queue, _keys.get(), CL_FALSE, 0, keys.size() * 4, keys.data(), 0, nullptr, nullptr);
				if (status == CL_SUCCESS) status = clFinish(queue);
				if (status != CL_SUCCESS) { error = clError("clEnqueueWriteBuffer (turning bands lines)", status); return false; }
				_times.upload += seconds(start);
				start = Clock::now();
				{
					const size_t global[2] = { size_t(columnPitch), size_t(pitch) }, local[2] = { 64, 1 };
					status = setArgs(_columnTable.get(), _whole, _fraction, cl_uint(count), cl_uint(pitch), cl_uint(columns), cl_uint(columnPitch), spectral, _columns, _columns);
					if (status == CL_SUCCESS) status = _device.launch(_columnTable, 2, global, local);
				}
				if (status == CL_SUCCESS) {
					const size_t global[2] = { size_t(rowPitch), size_t(pitch) }, local[2] = { 64, 1 };
					status = setArgs(_rowTable.get(), _whole, _fraction, cl_uint(count), cl_uint(pitch), cl_ulong(size[1]), cl_ulong(first), cl_uint(rows), cl_uint(rowPitch), spectral, _rows, _rows);
					if (status == CL_SUCCESS) status = _device.launch(_rowTable, 2, global, local);
				}
				if (status == CL_SUCCESS) {
					// Spectral: tiles of 128 rows x 64 columns; dilution: 64 x 64.
					const size_t global[2] = { size_t(columnPitch / 64 * 16), size_t(rowPitch / (spectral ? 128 : 64) * 16) }, local[2] = { 16, 16 };
					const cl_uint accumulate = any ? 1u : 0u;
					if (spectral) {
						status = setArgs(_spectral.get(), _rows, _columns, cl_uint(2 * pitch), cl_uint(rowPitch), cl_uint(columnPitch), cl_float(set.amplitude), accumulate, _out);
						if (status == CL_SUCCESS) status = _device.launch(_spectral, 2, global, local);
					} else {
						status = setArgs(_dilution.get(), _rows, _columns, _keys, cl_uint(count), cl_uint(pitch), cl_uint(rowPitch), cl_uint(columnPitch),
						                 cl_uint(set.process == 2 ? 1u : 0u), cl_float(set.amplitude), accumulate, _out);
						if (status == CL_SUCCESS) status = _device.launch(_dilution, 2, global, local);
					}
				}
				if (status == CL_SUCCESS) status = clFinish(queue);
				if (status != CL_SUCCESS) { error = clError("turning bands kernels", status); return false; }
				_times.compute += seconds(start);
				any = true;
			}
			start = Clock::now();
			float* target = out + (first - rowBegin) * columns;
			if (!any) {
				std::fill(target, target + rows * columns, 0.0f);
				continue;
			}
			const size_t origin[3] = { 0, 0, 0 }, region[3] = { size_t(columns * 4), size_t(rows), 1 };
			const cl_int status = clEnqueueReadBufferRect(queue, _out.get(), CL_TRUE, origin, origin, region, size_t(columnPitch * 4), 0, size_t(columns * 4), 0, target, 0, nullptr, nullptr);
			if (status != CL_SUCCESS) { error = clError("clEnqueueReadBufferRect (turning bands)", status); return false; }
			_times.download += seconds(start);
		}
		return true;
	}

	DeviceTimes times() const override { return _times; }

private:
	OpenClDevice& _device;
	Kernel _rowTable, _columnTable, _spectral, _dilution;
	Buffer _whole, _fraction, _keys, _columns, _rows, _out;
	DeviceTimes _times;
};

// ============================================================ FFT-MA / spectral

class OpenClSpectral : public SpectralKernel {
public:
	explicit OpenClSpectral(OpenClDevice& device) : _device(device) {}
	~OpenClSpectral() override {
		if (_forwardPlan) clfftDestroyPlan(&_forwardPlan);
		if (_backwardPlan) clfftDestroyPlan(&_backwardPlan);
	}

	bool initialise(std::string& error) {
		return _device.useClfft(error) && _noise.create(_device.program(), "fftWhiteNoise", error) && _multiply.create(_device.program(), "fftMultiplyAmplitude", error) &&
		       _spectralNoise.create(_device.program(), "fftSpectralNoise", error) && _crop.create(_device.program(), "fftCrop", error);
	}

	bool prepare(const SpectralLayout& layout, const float* amplitude, bool movingAverage, std::string& error) override {
		_layout = layout;
		_movingAverage = movingAverage;
		const DeviceInfo& info = _device.info();
		const uint64_t realBytes = layout.realCount * 4, halfBytes = layout.halfCount * 4, outBytes = layout.gridCount() * 4;
		if (realBytes > info.maxAllocation || halfBytes > info.maxAllocation || outBytes > info.maxAllocation ||
		    realBytes + halfBytes + outBytes > uint64_t(double(info.globalMemory) * 0.6)) {
			error = "one realisation needs " + std::to_string((realBytes + halfBytes + outBytes) >> 20) + " MiB of device memory (device: " +
			        std::to_string(info.globalMemory >> 20) + " MiB, largest buffer " + std::to_string(info.maxAllocation >> 20) + " MiB)";
			return false;
		}
		const cl_context context = _device.context();
		if (!_data.ensure(context, realBytes, error) || !_amplitude.ensure(context, halfBytes, error) || !_out.ensure(context, outBytes, error)) return false;
		const Clock::time_point start = Clock::now();
		cl_int status = clEnqueueWriteBuffer(_device.queue(), _amplitude.get(), CL_TRUE, 0, halfBytes, amplitude, 0, nullptr, nullptr);
		if (status != CL_SUCCESS) { error = clError("clEnqueueWriteBuffer (amplitude)", status); return false; }
		_times.upload += seconds(start);
		const clfftDim dim = layout.ndim == 1 ? CLFFT_1D : (layout.ndim == 2 ? CLFFT_2D : CLFFT_3D);
		size_t lengths[3] = { size_t(layout.extended[0]), size_t(layout.extended[1]), size_t(layout.extended[2]) };
		size_t realStrides[3] = { 1, size_t(layout.rowPitch), size_t(layout.rowPitch * layout.extended[1]) };
		size_t complexStrides[3] = { 1, size_t(layout.halfRow), size_t(layout.halfRow * layout.extended[1]) };
		auto makePlan = [&](clfftPlanHandle& plan, bool forward) -> bool {
			cl_context planContext = context;
			clfftStatus s = clfftCreateDefaultPlan(&plan, planContext, dim, lengths);
			if (s == CLFFT_SUCCESS) s = clfftSetPlanPrecision(plan, CLFFT_SINGLE);
			if (s == CLFFT_SUCCESS) s = forward ? clfftSetLayout(plan, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED) : clfftSetLayout(plan, CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL);
			if (s == CLFFT_SUCCESS) s = clfftSetResultLocation(plan, CLFFT_INPLACE);
			if (s == CLFFT_SUCCESS) s = clfftSetPlanInStride(plan, dim, forward ? realStrides : complexStrides);
			if (s == CLFFT_SUCCESS) s = clfftSetPlanOutStride(plan, dim, forward ? complexStrides : realStrides);
			if (s == CLFFT_SUCCESS) s = forward ? clfftSetPlanDistance(plan, layout.realCount, layout.halfCount) : clfftSetPlanDistance(plan, layout.halfCount, layout.realCount);
			// Unnormalised in both directions, as FFTW (spectralEngine.hpp normalisation).
			if (s == CLFFT_SUCCESS) s = clfftSetPlanScale(plan, forward ? CLFFT_FORWARD : CLFFT_BACKWARD, 1.0f);
			cl_command_queue queue = _device.queue();
			if (s == CLFFT_SUCCESS) s = clfftBakePlan(plan, 1, &queue, nullptr, nullptr);
			if (s != CLFFT_SUCCESS) { error = "clFFT: plan for " + std::to_string(layout.extended[0]) + " x " + std::to_string(layout.extended[1]) + " x " + std::to_string(layout.extended[2]) + " failed (error " + std::to_string(int(s)) + ")"; return false; }
			size_t temporary = 0;
			clfftGetTmpBufSize(plan, &temporary);
			if (temporary > 0 && !_temporary.ensure(context, temporary, error)) return false;
			return true;
		};
		if (movingAverage && !makePlan(_forwardPlan, true)) return false;
		if (!makePlan(_backwardPlan, false)) return false;
		return true;
	}

	bool simulate(uint64_t seed, uint32_t realisation, float mean, const float* noiseIn, float* noiseOut, float* out, std::string& error) override {
		const SpectralLayout& L = _layout;
		const cl_command_queue queue = _device.queue();
		cl_command_queue queues[1] = { queue };
		const cl_uint k0 = cl_uint(uint32_t(seed) ^ 0xA4093822u), k1 = cl_uint(uint32_t(seed >> 32) ^ 0x299F31D0u);
		const double M = double(L.cells);
		cl_int status = CL_SUCCESS;
		cl_mem temporary = _temporary.get();
		Clock::time_point start = Clock::now();
		if (_movingAverage) {
			if (noiseIn) {
				status = clEnqueueWriteBuffer(queue, _data.get(), CL_TRUE, 0, L.realCount * 4, noiseIn, 0, nullptr, nullptr);
				if (status != CL_SUCCESS) { error = clError("clEnqueueWriteBuffer (noise)", status); return false; }
				_times.upload += seconds(start);
				start = Clock::now();
			} else {
				const size_t blocks = size_t((L.cells + 3) / 4), local = 64, global = size_t(roundUp(blocks, local));
				status = setArgs(_noise.get(), k0, k1, cl_uint(realisation), cl_uint(0), cl_ulong(L.extended[0]), cl_ulong(L.rowPitch), cl_ulong(L.cells), _data);
				if (status == CL_SUCCESS) status = _device.launch(_noise, 1, &global, &local);
			}
			if (status == CL_SUCCESS && noiseOut) {
				status = clEnqueueReadBuffer(queue, _data.get(), CL_TRUE, 0, L.realCount * 4, noiseOut, 0, nullptr, nullptr);
			}
			if (status == CL_SUCCESS) {
				const clfftStatus s = clfftEnqueueTransform(_forwardPlan, CLFFT_FORWARD, 1, queues, 0, nullptr, nullptr, _data.address(), nullptr, temporary);
				if (s != CLFFT_SUCCESS) { error = "clFFT: forward transform failed (error " + std::to_string(int(s)) + ")"; return false; }
				const size_t local = 64, global = size_t(roundUp(L.halfCount, local));
				status = setArgs(_multiply.get(), _amplitude, cl_float(1.0 / M), cl_ulong(L.halfCount), _data);
				if (status == CL_SUCCESS) status = _device.launch(_multiply, 1, &global, &local);
			}
		} else {
			const size_t local = 64, global = size_t(roundUp(L.halfCount, local));
			status = setArgs(_spectralNoise.get(), k0, k1, cl_uint(realisation), cl_uint(1), cl_uint(L.ndim), cl_ulong(L.extended[0]), cl_ulong(L.extended[1]),
			                 cl_ulong(L.extended[2]), cl_ulong(L.halfRow), cl_ulong(L.rows), _amplitude, cl_float(1.0 / std::sqrt(M)), _data);
			if (status == CL_SUCCESS) status = _device.launch(_spectralNoise, 1, &global, &local);
		}
		if (status == CL_SUCCESS) {
			const clfftStatus s = clfftEnqueueTransform(_backwardPlan, CLFFT_BACKWARD, 1, queues, 0, nullptr, nullptr, _data.address(), nullptr, temporary);
			if (s != CLFFT_SUCCESS) { error = "clFFT: backward transform failed (error " + std::to_string(int(s)) + ")"; return false; }
			const size_t local = 64, global = size_t(roundUp(L.gridCount(), local));
			status = setArgs(_crop.get(), _data, cl_uint(L.ndim), cl_ulong(L.grid[0]), cl_ulong(L.grid[1]), cl_ulong(L.grid[2]), cl_ulong(L.extended[1]),
			                 cl_ulong(L.rowPitch), cl_float(mean), _out);
			if (status == CL_SUCCESS) status = _device.launch(_crop, 1, &global, &local);
		}
		if (status == CL_SUCCESS) status = clFinish(queue);
		if (status != CL_SUCCESS) { error = clError("FFT kernels", status); return false; }
		_times.compute += seconds(start);
		start = Clock::now();
		status = clEnqueueReadBuffer(queue, _out.get(), CL_TRUE, 0, L.gridCount() * 4, out, 0, nullptr, nullptr);
		if (status != CL_SUCCESS) { error = clError("clEnqueueReadBuffer (simulation)", status); return false; }
		_times.download += seconds(start);
		return true;
	}

	DeviceTimes times() const override { return _times; }

private:
	OpenClDevice& _device;
	SpectralLayout _layout;
	bool _movingAverage = true;
	Kernel _noise, _multiply, _spectralNoise, _crop;
	Buffer _data, _amplitude, _out, _temporary;
	clfftPlanHandle _forwardPlan = 0, _backwardPlan = 0;
	DeviceTimes _times;
};

// ============================================================ global kriging variance

class OpenClKriging : public KrigingKernel {
public:
	OpenClKriging(OpenClDevice& device, bool doubles) : _device(device), _doubles(doubles) {}

	bool initialise(std::string& error) {
		return _gather.create(_device.program(), "krigingGather", error) && _gemm.create(_device.program(), "krigingGemmSubtract", error) &&
		       _diagonal.create(_device.program(), "krigingDiagonalSolve", error) && _reduce.create(_device.program(), "krigingReduce", error);
	}

	bool doublePrecision() const override { return _doubles; }

	bool prepare(const double* factor, size_t n, const double* w, std::string& error) override {
		_n = n;
		_nPad = roundUp(n, 64);
		const size_t real = realSize();
		const uint64_t factorBytes = _nPad * _nPad * real;
		if (factorBytes > _device.info().maxAllocation || factorBytes > _device.info().globalMemory / 2) {
			error = "the " + std::to_string(n) + " x " + std::to_string(n) + " factor needs " + std::to_string(factorBytes >> 20) + " MiB of device memory";
			return false;
		}
		const uint64_t free = _device.budget();
		_pitch = std::max<uint64_t>(64, std::min<uint64_t>(16384, free / (_nPad * real) / 64 * 64));
		const cl_context context = _device.context();
		if (!_factor.ensure(context, factorBytes, error) || !_w.ensure(context, _nPad * real, error) || !_y.ensure(context, _nPad * _pitch * real, error) ||
		    !_offsets.ensure(context, _pitch * 8, error) || !_sums.ensure(context, _pitch * real, error) || !_dots.ensure(context, _pitch * real, error))
			return false;
		std::vector<double> padded(_nPad * _nPad, 0.0);
		for (size_t i = 0; i < _nPad; ++i) {
			if (i < n) for (size_t j = 0; j <= i; ++j) padded[i * _nPad + j] = factor[i * n + j];
			else padded[i * _nPad + i] = 1.0;
		}
		std::vector<double> wPadded(_nPad, 0.0);
		std::copy(w, w + n, wPadded.begin());
		const Clock::time_point start = Clock::now();
		if (!upload(_factor, padded, error) || !upload(_w, wPadded, error)) return false;
		_times.upload += seconds(start);
		return true;
	}

	bool setTable(const double* table, size_t tableCount, const uint64_t* dataBase, std::string& error) override {
		const cl_context context = _device.context();
		if (tableCount * realSize() > _device.info().maxAllocation) { error = "the covariance table does not fit one device buffer"; return false; }
		if (!_table.ensure(context, tableCount * realSize(), error) || !_base.ensure(context, _nPad * 8, error)) return false;
		std::vector<double> values(table, table + tableCount);
		std::vector<uint64_t> base(_nPad, 0);
		std::copy(dataBase, dataBase + _n, base.begin());
		const Clock::time_point start = Clock::now();
		if (!upload(_table, values, error)) return false;
		const cl_int status = clEnqueueWriteBuffer(_device.queue(), _base.get(), CL_TRUE, 0, _nPad * 8, base.data(), 0, nullptr, nullptr);
		if (status != CL_SUCCESS) { error = clError("clEnqueueWriteBuffer (table indices)", status); return false; }
		_times.upload += seconds(start);
		_hasTable = true;
		return true;
	}

	size_t maxTargets() const override { return size_t(_pitch); }

	bool varianceTerms(const uint64_t* targetOffsets, const double* kColumns, size_t m, double* sumSquares, double* dotW, std::string& error) override {
		if (m == 0) return true;
		if (m > _pitch) { error = "too many targets for one call"; return false; }
		const uint64_t pitch = roundUp(m, 64);
		const cl_command_queue queue = _device.queue();
		cl_int status = CL_SUCCESS;
		Clock::time_point start = Clock::now();
		if (targetOffsets) {
			if (!_hasTable) { error = "no covariance table"; return false; }
			std::vector<uint64_t> offsets(pitch, 0);
			std::copy(targetOffsets, targetOffsets + m, offsets.begin());
			status = clEnqueueWriteBuffer(queue, _offsets.get(), CL_TRUE, 0, pitch * 8, offsets.data(), 0, nullptr, nullptr);
			if (status != CL_SUCCESS) { error = clError("clEnqueueWriteBuffer (targets)", status); return false; }
			_times.upload += seconds(start);
			start = Clock::now();
			const size_t global[2] = { size_t(pitch), size_t(_nPad) }, local[2] = { 64, 1 };
			status = setArgs(_gather.get(), _table, _base, _offsets, cl_uint(_n), cl_uint(m), cl_uint(pitch), _y);
			if (status == CL_SUCCESS) status = _device.launch(_gather, 2, global, local);
		} else {
			std::vector<double> y(_nPad * pitch, 0.0);
			for (size_t i = 0; i < _n; ++i) std::copy(kColumns + i * m, kColumns + (i + 1) * m, y.begin() + i * pitch);
			if (!upload(_y, y, error)) return false;
			_times.upload += seconds(start);
			start = Clock::now();
		}
		for (uint64_t i0 = 0; i0 < _nPad && status == CL_SUCCESS; i0 += 64) {
			if (i0 > 0) {
				const size_t global[2] = { size_t(pitch / 64 * 16), 16 }, local[2] = { 16, 16 };
				status = setArgs(_gemm.get(), _factor, cl_uint(_nPad), _y, cl_uint(pitch), cl_uint(i0), cl_uint(i0));
				if (status == CL_SUCCESS) status = _device.launch(_gemm, 2, global, local);
			}
			if (status == CL_SUCCESS) {
				const size_t global = size_t(pitch), local = 64;
				status = setArgs(_diagonal.get(), _factor, cl_uint(_nPad), _y, cl_uint(pitch), cl_uint(i0), cl_uint(i0 + 64));
				if (status == CL_SUCCESS) status = _device.launch(_diagonal, 1, &global, &local);
			}
		}
		if (status == CL_SUCCESS) {
			const size_t global = size_t(pitch), local = 64;
			status = setArgs(_reduce.get(), _y, _w, cl_uint(_n), cl_uint(m), cl_uint(pitch), _sums, _dots);
			if (status == CL_SUCCESS) status = _device.launch(_reduce, 1, &global, &local);
		}
		if (status == CL_SUCCESS) status = clFinish(queue);
		if (status != CL_SUCCESS) { error = clError("kriging kernels", status); return false; }
		_times.compute += seconds(start);
		start = Clock::now();
		if (!download(_sums, m, sumSquares, error) || !download(_dots, m, dotW, error)) return false;
		_times.download += seconds(start);
		return true;
	}

	DeviceTimes times() const override { return _times; }

private:
	OpenClDevice& _device;
	bool _doubles;
	size_t _n = 0;
	uint64_t _nPad = 0, _pitch = 0;
	bool _hasTable = false;
	Kernel _gather, _gemm, _diagonal, _reduce;
	Buffer _factor, _w, _y, _table, _base, _offsets, _sums, _dots;
	DeviceTimes _times;

	size_t realSize() const { return _doubles ? 8 : 4; }

	bool upload(Buffer& buffer, const std::vector<double>& values, std::string& error) {
		cl_int status;
		if (_doubles) status = clEnqueueWriteBuffer(_device.queue(), buffer.get(), CL_TRUE, 0, values.size() * 8, values.data(), 0, nullptr, nullptr);
		else {
			std::vector<float> narrow(values.begin(), values.end());
			status = clEnqueueWriteBuffer(_device.queue(), buffer.get(), CL_TRUE, 0, narrow.size() * 4, narrow.data(), 0, nullptr, nullptr);
		}
		if (status != CL_SUCCESS) { error = clError("clEnqueueWriteBuffer (kriging)", status); return false; }
		return true;
	}
	bool download(Buffer& buffer, size_t count, double* out, std::string& error) {
		cl_int status;
		if (_doubles) status = clEnqueueReadBuffer(_device.queue(), buffer.get(), CL_TRUE, 0, count * 8, out, 0, nullptr, nullptr);
		else {
			std::vector<float> narrow(count);
			status = clEnqueueReadBuffer(_device.queue(), buffer.get(), CL_TRUE, 0, count * 4, narrow.data(), 0, nullptr, nullptr);
			std::copy(narrow.begin(), narrow.end(), out);
		}
		if (status != CL_SUCCESS) { error = clError("clEnqueueReadBuffer (kriging)", status); return false; }
		return true;
	}
};

TurningBandsKernel* OpenClDevice::turningBands(std::string& reason) {
	OpenClTurningBands* kernel = new OpenClTurningBands(*this);
	if (!kernel->initialise(reason)) { delete kernel; return nullptr; }
	return kernel;
}

SpectralKernel* OpenClDevice::spectral(std::string& reason) {
	OpenClSpectral* kernel = new OpenClSpectral(*this);
	if (!kernel->initialise(reason)) { delete kernel; return nullptr; }
	return kernel;
}

KrigingKernel* OpenClDevice::kriging(std::string& reason) {
	if (!_info.fp64) {
		reason = "the device has no double precision (no cl_khr_fp64); kriging systems are ill-conditioned and are not solved in float";
		return nullptr;
	}
	OpenClKriging* kernel = new OpenClKriging(*this, true);
	if (!kernel->initialise(reason)) { delete kernel; return nullptr; }
	return kernel;
}

KrigingKernel* OpenClDevice::krigingFloat(std::string& reason) {
	if (_info.fp64) { reason = "the device has double precision: use Device::kriging"; return nullptr; }
	OpenClKriging* kernel = new OpenClKriging(*this, false);
	if (!kernel->initialise(reason)) { delete kernel; return nullptr; }
	return kernel;
}

} // namespace

bool openclAvailable() { return true; }

std::vector<std::string> openclDeviceNames() {
	std::vector<std::string> names;
	for (const auto& entry : gpuDevices()) names.push_back(deviceString(entry.second, CL_DEVICE_NAME));
	return names;
}

Device* openOpenClDevice(int index, std::string& message) {
	const std::vector<std::pair<cl_platform_id, cl_device_id> > devices = gpuDevices();
	if (devices.empty()) { message = "no OpenCL GPU device"; return nullptr; }
	if (index < 0) index = 0;
	if (size_t(index) >= devices.size()) { message = "no OpenCL GPU device " + std::to_string(index) + " (" + std::to_string(devices.size()) + " found)"; return nullptr; }
	OpenClDevice* device = new OpenClDevice(devices[index].first, devices[index].second, index);
	if (!device->initialise(message)) { delete device; return nullptr; }
	return device;
}

KrigingKernel* openclKrigingFloatForTests(Device* device, std::string& message) {
	OpenClDevice* opencl = dynamic_cast<OpenClDevice*>(device);
	if (!opencl) { message = "not an OpenCL device"; return nullptr; }
	return opencl->krigingFloat(message);
}

} // namespace gpu
} // namespace geostat
} // namespace g2s

#endif // WITH_OPENCL
