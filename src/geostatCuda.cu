/*
 * G2S - geostat: CUDA back end of the GPU kernels (gpu/backend.hpp).
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

// ************************************************************************
// NOT COMPILED WITH NVCC AND NEVER RUN. This file was written on a machine
// without CUDA (Apple silicon). It mirrors src/geostatOpenCL.cpp, which is
// tested on an Apple GPU, call for call; the kernels are the same source
// (gpu/geostatKernels.inc), tested through OpenCL. What has been checked:
// clang in CUDA mode (-x cuda --cuda-gpu-arch=sm_70) with stub headers for
// the runtime and cuFFT parses and type-checks this file and the kernels
// (see the commit that adds it). What has not: nvcc, the real CUDA headers,
// linking against cudart / cufft, and every run-time behaviour (launch
// configurations, cuFFT layouts, results). Test it on an Nvidia GPU with
// test/geostat_gpu adapted to CUDA before relying on it.
// ************************************************************************
//
// Built into g2s_cuda.so with the QS CUDA device (build/c++-build/Makefile,
// g2s_cuda.so rule, when nvcc is found) and loaded by
// gpu/deviceSelection (dlopen, -W_CUDA). Entry points: g2s_geostatCudaDevice,
// g2s_geostatCudaDeviceCount (extern "C").
//
// Differences from OpenCL: CUDA devices are used in double for the kriging
// kernels (REAL = double, every CUDA GPU has fp64); transforms with cuFFT
// (unnormalised like FFTW, in place with the padded r2c layout).

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "geostat/gpu/backend.hpp"
#include "geostat/gpu/kernelPrelude.hpp"

#define G2S_GPU_SOURCE(...) __VA_ARGS__
#include "geostat/gpu/geostatKernels.inc"
#undef G2S_GPU_SOURCE

namespace g2s {
namespace geostat {
namespace gpu {

namespace {

using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point since) { return std::chrono::duration<double>(Clock::now() - since).count(); }

uint64_t roundUp(uint64_t value, uint64_t multiple) { return (value + multiple - 1) / multiple * multiple; }

std::string cudaMessage(const char* what, cudaError_t code) { return std::string("CUDA: ") + what + " failed (" + cudaGetErrorString(code) + ")"; }

// Last launch error, then completion.
bool finish(const char* what, std::string& error) {
	cudaError_t status = cudaGetLastError();
	if (status == cudaSuccess) status = cudaDeviceSynchronize();
	if (status != cudaSuccess) { error = cudaMessage(what, status); return false; }
	return true;
}

class Buffer {
public:
	Buffer() = default;
	Buffer(const Buffer&) = delete;
	Buffer& operator=(const Buffer&) = delete;
	~Buffer() { release(); }
	bool ensure(size_t bytes, std::string& error) {
		if (bytes <= _bytes && _pointer) return true;
		release();
		const cudaError_t status = cudaMalloc(&_pointer, std::max<size_t>(bytes, 16));
		if (status != cudaSuccess) { _pointer = nullptr; error = cudaMessage("cudaMalloc", status) + " for " + std::to_string(bytes >> 20) + " MiB"; return false; }
		_bytes = bytes;
		return true;
	}
	template <class T> T* as() const { return static_cast<T*>(_pointer); }
	void release() {
		if (_pointer) cudaFree(_pointer);
		_pointer = nullptr;
		_bytes = 0;
	}

private:
	void* _pointer = nullptr;
	size_t _bytes = 0;
};

bool upload(Buffer& buffer, const void* data, size_t bytes, std::string& error) {
	const cudaError_t status = cudaMemcpy(buffer.as<void>(), data, bytes, cudaMemcpyHostToDevice);
	if (status != cudaSuccess) { error = cudaMessage("cudaMemcpy to the device", status); return false; }
	return true;
}

bool download(void* data, const Buffer& buffer, size_t bytes, std::string& error) {
	const cudaError_t status = cudaMemcpy(data, buffer.as<void>(), bytes, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) { error = cudaMessage("cudaMemcpy from the device", status); return false; }
	return true;
}

class CudaDevice : public Device {
public:
	CudaDevice(int index, const cudaDeviceProp& properties) : _index(index) {
		_info.api = "CUDA";
		_info.index = index;
		_info.name = properties.name;
		_info.globalMemory = properties.totalGlobalMem;
		_info.maxAllocation = properties.totalGlobalMem;
		_info.fp64 = true;
		_info.unifiedMemory = properties.integrated != 0;
	}
	const DeviceInfo& info() const override { return _info; }
	TurningBandsKernel* turningBands(std::string& reason) override;
	SpectralKernel* spectral(std::string& reason) override;
	KrigingKernel* kriging(std::string& reason) override;
	uint64_t budget() const {
		size_t free = 0, total = 0;
		if (cudaMemGetInfo(&free, &total) != cudaSuccess) free = size_t(_info.globalMemory / 2);
		return std::max<uint64_t>(uint64_t(64) << 20, std::min<uint64_t>(uint64_t(free) / 2, uint64_t(2) << 30));
	}

private:
	int _index;
	DeviceInfo _info;
};

// ============================================================ turning bands

class CudaTurningBands : public TurningBandsKernel {
public:
	explicit CudaTurningBands(CudaDevice& device) : _device(device) {}

	bool sum(const TurningBandsLines* sets, size_t setCount, const uint64_t size[3], uint64_t rowBegin, uint64_t rowEnd, float* out, std::string& error) override {
		if (rowEnd <= rowBegin) return true;
		const uint64_t columns = size[0], columnPitch = roundUp(columns, 64);
		if (columns > 0xFFFFFFFFull) { error = "turning bands on the GPU: more than 2^32 columns"; return false; }
		uint32_t maxLines = 16;
		for (size_t s = 0; s < setCount; ++s) maxLines = std::max(maxLines, sets[s].count);
		const uint64_t maxPitch = roundUp(maxLines, 16);
		if (maxPitch > 65535) { error = "more than 65535 lines per structure (CUDA grid limit of the table kernels)"; return false; }
		const uint64_t columnBytes = maxPitch * columnPitch * 8;
		const uint64_t rowBytes = maxPitch * 8 + columnPitch * 4;
		const uint64_t available = _device.budget() > columnBytes ? _device.budget() - columnBytes : 0;
		uint64_t chunkRows = std::min<uint64_t>(roundUp(rowEnd - rowBegin, 128), std::max<uint64_t>(128, available / rowBytes / 128 * 128));
		chunkRows = std::min<uint64_t>(chunkRows, uint64_t(1) << 20);
		if (!_whole.ensure(4 * maxPitch * 4, error) || !_fraction.ensure(4 * maxPitch * 8, error) || !_keys.ensure(2 * maxPitch * 4, error) ||
		    !_columns.ensure(columnBytes, error) || !_rows.ensure(maxPitch * chunkRows * 8, error) || !_out.ensure(chunkRows * columnPitch * 4, error))
			return false;
		std::vector<uint32_t> whole, keys;
		std::vector<uint64_t> fraction;
		for (uint64_t first = rowBegin; first < rowEnd; first += chunkRows) {
			const uint64_t rows = std::min(chunkRows, rowEnd - first), rowPitch = roundUp(rows, 128);
			Clock::time_point start = Clock::now();
			bool any = false;
			for (size_t s = 0; s < setCount; ++s) {
				const TurningBandsLines& set = sets[s];
				if (set.count == 0) continue;
				const uint32_t count = set.count, pitch = uint32_t(roundUp(count, 16));
				const uint32_t spectral = set.process == 0 ? 1u : 0u;
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
				if (!upload(_whole, whole.data(), whole.size() * 4, error) || !upload(_fraction, fraction.data(), fraction.size() * 8, error) ||
				    !upload(_keys, keys.data(), keys.size() * 4, error))
					return false;
				_times.upload += seconds(start);
				start = Clock::now();
				tbColumnTable<<<dim3(unsigned(columnPitch / 64), pitch), dim3(64, 1)>>>(_whole.as<g2s_u32>(), _fraction.as<g2s_u64>(), count, pitch, uint32_t(columns),
				                                                                       uint32_t(columnPitch), spectral, _columns.as<float>(), _columns.as<g2s_u64>());
				tbRowTable<<<dim3(unsigned(rowPitch / 64), pitch), dim3(64, 1)>>>(_whole.as<g2s_u32>(), _fraction.as<g2s_u64>(), count, pitch, g2s_u64(size[1]), g2s_u64(first),
				                                                                 uint32_t(rows), uint32_t(rowPitch), spectral, _rows.as<float>(), _rows.as<g2s_u64>());
				const uint32_t accumulate = any ? 1u : 0u;
				if (spectral) {
					tbSpectralProduct<<<dim3(unsigned(columnPitch / 64), unsigned(rowPitch / 128)), dim3(16, 16)>>>(
						_rows.as<float>(), _columns.as<float>(), 2 * pitch, uint32_t(rowPitch), uint32_t(columnPitch), set.amplitude, accumulate, _out.as<float>());
				} else {
					tbDilutionProduct<<<dim3(unsigned(columnPitch / 64), unsigned(rowPitch / 64)), dim3(16, 16)>>>(
						_rows.as<g2s_u64>(), _columns.as<g2s_u64>(), _keys.as<g2s_u32>(), count, pitch, uint32_t(rowPitch), uint32_t(columnPitch),
						set.process == 2 ? 1u : 0u, set.amplitude, accumulate, _out.as<float>());
				}
				if (!finish("turning bands kernels", error)) return false;
				_times.compute += seconds(start);
				any = true;
			}
			start = Clock::now();
			float* target = out + (first - rowBegin) * columns;
			if (!any) {
				std::fill(target, target + rows * columns, 0.0f);
				continue;
			}
			const cudaError_t status = cudaMemcpy2D(target, size_t(columns * 4), _out.as<float>(), size_t(columnPitch * 4), size_t(columns * 4), size_t(rows), cudaMemcpyDeviceToHost);
			if (status != cudaSuccess) { error = cudaMessage("cudaMemcpy2D (turning bands)", status); return false; }
			_times.download += seconds(start);
		}
		return true;
	}

	DeviceTimes times() const override { return _times; }

private:
	CudaDevice& _device;
	Buffer _whole, _fraction, _keys, _columns, _rows, _out;
	DeviceTimes _times;
};

// ============================================================ FFT-MA / spectral

class CudaSpectral : public SpectralKernel {
public:
	explicit CudaSpectral(CudaDevice& device) : _device(device) {}
	~CudaSpectral() override {
		if (_forwardPlan) cufftDestroy(_forwardPlan);
		if (_backwardPlan) cufftDestroy(_backwardPlan);
	}

	bool prepare(const SpectralLayout& layout, const float* amplitude, bool movingAverage, std::string& error) override {
		_layout = layout;
		_movingAverage = movingAverage;
		const uint64_t realBytes = layout.realCount * 4, halfBytes = layout.halfCount * 4, outBytes = layout.gridCount() * 4;
		if (realBytes + halfBytes + outBytes > uint64_t(double(_device.info().globalMemory) * 0.6)) {
			error = "one realisation needs " + std::to_string((realBytes + halfBytes + outBytes) >> 20) + " MiB of device memory (device: " +
			        std::to_string(_device.info().globalMemory >> 20) + " MiB)";
			return false;
		}
		if (!_data.ensure(realBytes, error) || !_amplitude.ensure(halfBytes, error) || !_out.ensure(outBytes, error)) return false;
		const Clock::time_point start = Clock::now();
		if (!upload(_amplitude, amplitude, halfBytes, error)) return false;
		_times.upload += seconds(start);
		// cuFFT dimensions are slowest first; DataImage axis 0 is the fastest.
		const int rank = int(layout.ndim);
		int n[3], realEmbed[3], complexEmbed[3];
		for (int d = 0; d < rank; ++d) {
			n[d] = int(layout.extended[rank - 1 - d]);
			realEmbed[d] = n[d];
			complexEmbed[d] = n[d];
		}
		realEmbed[rank - 1] = int(layout.rowPitch);
		complexEmbed[rank - 1] = int(layout.halfRow);
		auto makePlan = [&](cufftHandle& plan, bool forward) -> bool {
			const cufftResult result = forward
				? cufftPlanMany(&plan, rank, n, realEmbed, 1, int(layout.realCount), complexEmbed, 1, int(layout.halfCount), CUFFT_R2C, 1)
				: cufftPlanMany(&plan, rank, n, complexEmbed, 1, int(layout.halfCount), realEmbed, 1, int(layout.realCount), CUFFT_C2R, 1);
			if (result != CUFFT_SUCCESS) { error = "cuFFT: plan failed (error " + std::to_string(int(result)) + ")"; return false; }
			return true;
		};
		if (movingAverage && !makePlan(_forwardPlan, true)) return false;
		if (!makePlan(_backwardPlan, false)) return false;
		return true;
	}

	bool simulate(uint64_t seed, uint32_t realisation, float mean, const float* noiseIn, float* noiseOut, float* out, std::string& error) override {
		const SpectralLayout& L = _layout;
		const uint32_t k0 = uint32_t(seed) ^ 0xA4093822u, k1 = uint32_t(seed >> 32) ^ 0x299F31D0u;
		const double M = double(L.cells);
		Clock::time_point start = Clock::now();
		const unsigned threads = 64;
		auto blocks = [&](uint64_t count) { return unsigned((count + threads - 1) / threads); };
		if (_movingAverage) {
			if (noiseIn) {
				if (!upload(_data, noiseIn, L.realCount * 4, error)) return false;
				_times.upload += seconds(start);
				start = Clock::now();
			} else {
				fftWhiteNoise<<<blocks((L.cells + 3) / 4), threads>>>(k0, k1, realisation, 0u, g2s_u64(L.extended[0]), g2s_u64(L.rowPitch), g2s_u64(L.cells), _data.as<float>());
			}
			if (noiseOut && (!finish("white noise", error) || !download(noiseOut, _data, L.realCount * 4, error))) return false;
			if (cufftExecR2C(_forwardPlan, _data.as<cufftReal>(), _data.as<cufftComplex>()) != CUFFT_SUCCESS) { error = "cuFFT: forward transform failed"; return false; }
			fftMultiplyAmplitude<<<blocks(L.halfCount), threads>>>(_amplitude.as<float>(), float(1.0 / M), g2s_u64(L.halfCount), _data.as<float>());
		} else {
			fftSpectralNoise<<<blocks(L.halfCount), threads>>>(k0, k1, realisation, 1u, L.ndim, g2s_u64(L.extended[0]), g2s_u64(L.extended[1]), g2s_u64(L.extended[2]),
			                                                   g2s_u64(L.halfRow), g2s_u64(L.rows), _amplitude.as<float>(), float(1.0 / std::sqrt(M)), _data.as<float>());
		}
		if (cufftExecC2R(_backwardPlan, _data.as<cufftComplex>(), _data.as<cufftReal>()) != CUFFT_SUCCESS) { error = "cuFFT: backward transform failed"; return false; }
		fftCrop<<<blocks(L.gridCount()), threads>>>(_data.as<float>(), L.ndim, g2s_u64(L.grid[0]), g2s_u64(L.grid[1]), g2s_u64(L.grid[2]), g2s_u64(L.extended[1]),
		                                            g2s_u64(L.rowPitch), mean, _out.as<float>());
		if (!finish("FFT kernels", error)) return false;
		_times.compute += seconds(start);
		start = Clock::now();
		if (!download(out, _out, L.gridCount() * 4, error)) return false;
		_times.download += seconds(start);
		return true;
	}

	DeviceTimes times() const override { return _times; }

private:
	CudaDevice& _device;
	SpectralLayout _layout;
	bool _movingAverage = true;
	Buffer _data, _amplitude, _out;
	cufftHandle _forwardPlan = 0, _backwardPlan = 0;
	DeviceTimes _times;
};

// ============================================================ global kriging variance

class CudaKriging : public KrigingKernel {
public:
	explicit CudaKriging(CudaDevice& device) : _device(device) {}

	bool doublePrecision() const override { return true; }

	bool prepare(const double* factor, size_t n, const double* w, std::string& error) override {
		_n = n;
		_nPad = roundUp(n, 64);
		if (_nPad > 65535) { error = "more than 65535 data (CUDA grid limit of the gather kernel)"; return false; }
		const uint64_t factorBytes = _nPad * _nPad * 8;
		if (factorBytes > _device.info().globalMemory / 2) {
			error = "the " + std::to_string(n) + " x " + std::to_string(n) + " factor needs " + std::to_string(factorBytes >> 20) + " MiB of device memory";
			return false;
		}
		_pitch = std::max<uint64_t>(64, std::min<uint64_t>(16384, _device.budget() / (_nPad * 8) / 64 * 64));
		if (!_factor.ensure(factorBytes, error) || !_w.ensure(_nPad * 8, error) || !_y.ensure(_nPad * _pitch * 8, error) || !_offsets.ensure(_pitch * 8, error) ||
		    !_sums.ensure(_pitch * 8, error) || !_dots.ensure(_pitch * 8, error))
			return false;
		std::vector<double> padded(_nPad * _nPad, 0.0);
		for (size_t i = 0; i < _nPad; ++i) {
			if (i < n) for (size_t j = 0; j <= i; ++j) padded[i * _nPad + j] = factor[i * n + j];
			else padded[i * _nPad + i] = 1.0;
		}
		std::vector<double> wPadded(_nPad, 0.0);
		std::copy(w, w + n, wPadded.begin());
		const Clock::time_point start = Clock::now();
		if (!upload(_factor, padded.data(), padded.size() * 8, error) || !upload(_w, wPadded.data(), wPadded.size() * 8, error)) return false;
		_times.upload += seconds(start);
		return true;
	}

	bool setTable(const double* table, size_t tableCount, const uint64_t* dataBase, std::string& error) override {
		if (!_table.ensure(tableCount * 8, error) || !_base.ensure(_nPad * 8, error)) return false;
		std::vector<uint64_t> base(_nPad, 0);
		std::copy(dataBase, dataBase + _n, base.begin());
		const Clock::time_point start = Clock::now();
		if (!upload(_table, table, tableCount * 8, error) || !upload(_base, base.data(), _nPad * 8, error)) return false;
		_times.upload += seconds(start);
		_hasTable = true;
		return true;
	}

	size_t maxTargets() const override { return size_t(_pitch); }

	bool varianceTerms(const uint64_t* targetOffsets, const double* kColumns, size_t m, double* sumSquares, double* dotW, std::string& error) override {
		if (m == 0) return true;
		if (m > _pitch) { error = "too many targets for one call"; return false; }
		const uint64_t pitch = roundUp(m, 64);
		Clock::time_point start = Clock::now();
		if (targetOffsets) {
			if (!_hasTable) { error = "no covariance table"; return false; }
			std::vector<uint64_t> offsets(pitch, 0);
			std::copy(targetOffsets, targetOffsets + m, offsets.begin());
			if (!upload(_offsets, offsets.data(), pitch * 8, error)) return false;
			_times.upload += seconds(start);
			start = Clock::now();
			krigingGather<<<dim3(unsigned(pitch / 64), unsigned(_nPad)), dim3(64, 1)>>>(_table.as<double>(), _base.as<g2s_u64>(), _offsets.as<g2s_u64>(), uint32_t(_n),
			                                                                             uint32_t(m), uint32_t(pitch), _y.as<double>());
		} else {
			std::vector<double> y(_nPad * pitch, 0.0);
			for (size_t i = 0; i < _n; ++i) std::copy(kColumns + i * m, kColumns + (i + 1) * m, y.begin() + i * pitch);
			if (!upload(_y, y.data(), y.size() * 8, error)) return false;
			_times.upload += seconds(start);
			start = Clock::now();
		}
		for (uint64_t i0 = 0; i0 < _nPad; i0 += 64) {
			if (i0 > 0)
				krigingGemmSubtract<<<dim3(unsigned(pitch / 64), 1), dim3(16, 16)>>>(_factor.as<double>(), uint32_t(_nPad), _y.as<double>(), uint32_t(pitch), uint32_t(i0), uint32_t(i0));
			krigingDiagonalSolve<<<unsigned(pitch / 64), 64>>>(_factor.as<double>(), uint32_t(_nPad), _y.as<double>(), uint32_t(pitch), uint32_t(i0), uint32_t(i0 + 64));
		}
		krigingReduce<<<unsigned(pitch / 64), 64>>>(_y.as<double>(), _w.as<double>(), uint32_t(_n), uint32_t(m), uint32_t(pitch), _sums.as<double>(), _dots.as<double>());
		if (!finish("kriging kernels", error)) return false;
		_times.compute += seconds(start);
		start = Clock::now();
		if (!download(sumSquares, _sums, m * 8, error) || !download(dotW, _dots, m * 8, error)) return false;
		_times.download += seconds(start);
		return true;
	}

	DeviceTimes times() const override { return _times; }

private:
	CudaDevice& _device;
	size_t _n = 0;
	uint64_t _nPad = 0, _pitch = 0;
	bool _hasTable = false;
	Buffer _factor, _w, _y, _table, _base, _offsets, _sums, _dots;
	DeviceTimes _times;
};

TurningBandsKernel* CudaDevice::turningBands(std::string&) { return new CudaTurningBands(*this); }
SpectralKernel* CudaDevice::spectral(std::string&) { return new CudaSpectral(*this); }
KrigingKernel* CudaDevice::kriging(std::string&) { return new CudaKriging(*this); }

} // namespace

} // namespace gpu
} // namespace geostat
} // namespace g2s

extern "C" {

int g2s_geostatCudaDeviceCount() {
	int count = 0;
	if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
	return count;
}

g2s::geostat::gpu::Device* g2s_geostatCudaDevice(int index, char* message, size_t messageSize) {
	auto fail = [&](const std::string& text) -> g2s::geostat::gpu::Device* {
		if (message && messageSize > 0) snprintf(message, messageSize, "%s", text.c_str());
		return nullptr;
	};
	int count = 0;
	cudaError_t status = cudaGetDeviceCount(&count);
	if (status != cudaSuccess) return fail(std::string("no CUDA device (") + cudaGetErrorString(status) + ")");
	if (index < 0 || index >= count) return fail("no CUDA device " + std::to_string(index) + " (" + std::to_string(count) + " found)");
	status = cudaSetDevice(index);
	if (status != cudaSuccess) return fail(std::string("cudaSetDevice failed (") + cudaGetErrorString(status) + ")");
	cudaDeviceProp properties;
	status = cudaGetDeviceProperties(&properties, index);
	if (status != cudaSuccess) return fail(std::string("cudaGetDeviceProperties failed (") + cudaGetErrorString(status) + ")");
	return new g2s::geostat::gpu::CudaDevice(index, properties);
}

}
