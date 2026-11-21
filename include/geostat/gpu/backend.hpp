/*
 * G2S - geostat: interfaces of the GPU back ends (OpenCL, CUDA).
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

// One small interface per offloaded kernel, each with three implementations:
//   CPU     the engines themselves (always the default and the reference; the
//           CPU versions of these interfaces in gpu/cpuKernels.hpp are thin
//           wrappers of the engine kernels, used as references by the tests);
//   OpenCL  src/geostatOpenCL.cpp, compiled in under WITH_OPENCL (clFFT);
//   CUDA    src/geostatCuda.cu, in g2s_cuda.so, loaded with dlopen (cuFFT).
// The device kernels are one source (gpu/geostatKernels.inc). This header
// depends on nothing but the standard library, so that g2s_cuda.so does not
// compile the engines.
//
//   TurningBandsKernel  sum over the lines of turning bands on rows of a grid
//                       (the grid path, sumLinesGrid);
//   SpectralKernel      one FFT-MA / spectral realisation: noise, transforms,
//                       amplitude, crop (fillWhiteNoise ... cropToGrid);
//   KrigingKernel       the per-target part of the global kriging variance:
//                       L^-1 k for blocks of targets and the two reductions
//                       (GlobalKriging::variance), double precision only.
//
// Precision per kernel (tolerances: gpu/deviceEngines.hpp and the tests):
//   turning bands   no double needed: phases in exact 64-bit fixed point,
//                   sums in float with a compensated total;
//   FFT             float, as the CPU path (FFTW float);
//   kriging         double: offered only by devices with fp64 (CUDA, OpenCL
//                   with cl_khr_fp64). Never in float: the systems are
//                   ill-conditioned (gaussian models, clustered data).
//
// Errors: every call returns false with a message; the callers then fall back
// to the CPU and log why.

#ifndef G2S_GEOSTAT_GPU_BACKEND_HPP
#define G2S_GEOSTAT_GPU_BACKEND_HPP

#include <cstddef>
#include <cstdint>
#include <string>

namespace g2s {
namespace geostat {
namespace gpu {

struct DeviceInfo {
	std::string api;            // "OpenCL" or "CUDA"
	std::string name;
	int index = -1;             // index in the selection syntax (-W_GPU n / -W_CUDA n)
	uint64_t globalMemory = 0;  // bytes
	uint64_t maxAllocation = 0; // largest single buffer, bytes
	bool fp64 = false;
	bool unifiedMemory = false; // device memory is host memory (Apple silicon, integrated GPUs)
};

// Time spent by a kernel object, seconds (host wall clock around blocking calls).
struct DeviceTimes {
	double upload = 0.0, compute = 0.0, download = 0.0;
	double total() const { return upload + compute + download; }
};

// ------------------------------------------------------------ turning bands

// The lines of one structure in fixed point (built by gpu/deviceEngines.hpp
// from a LineSet and a RegularGrid). Value j in {0: shift, 1: step along
// axis 0, 2: axis 1, 3: axis 2} of line l is whole[j * count + l] +
// fraction[j * count + l] / 2^64 (whole modulo 2^32, two's complement), so
// that t_l(node) = shift + sum_d step_d i_d is computed exactly modulo 2^32
// segments (or turns). signKey: 2 x count (dilution only), as LineSet.
struct TurningBandsLines {
	int process = 0;            // 0 spectral, 1 dilution spherical, 2 dilution cubic
	uint32_t count = 0;
	float amplitude = 0.0f;
	const uint32_t* whole = nullptr;
	const uint64_t* fraction = nullptr;
	const uint32_t* signKey = nullptr;
};

class TurningBandsKernel {
public:
	virtual ~TurningBandsKernel() {}
	// out[(row - rowBegin) * size[0] + i0] = sum over the sets of amplitude *
	// sum over the lines of the line's value at node (i0, row), for rows
	// rowBegin .. rowEnd - 1 (row = i1 + size[1] i2). The work is split into
	// chunks of rows that fit the device; any row range is accepted.
	virtual bool sum(const TurningBandsLines* sets, size_t setCount, const uint64_t size[3], uint64_t rowBegin, uint64_t rowEnd, float* out,
	                 std::string& error) = 0;
	virtual DeviceTimes times() const = 0;
};

// ------------------------------------------------------------ FFT-MA / spectral

// Same meaning as spectral::GridLayout (spectralEngine.hpp); unused axes 1.
struct SpectralLayout {
	unsigned ndim = 0;
	uint64_t grid[3] = { 1, 1, 1 };
	uint64_t extended[3] = { 1, 1, 1 };
	uint64_t cells = 0, rows = 0, halfRow = 0, halfCount = 0, rowPitch = 0, realCount = 0;
	uint64_t gridCount() const { return grid[0] * grid[1] * grid[2]; }
};

class SpectralKernel {
public:
	virtual ~SpectralKernel() {}
	// Plans the transforms and uploads the amplitude (halfCount floats).
	// movingAverage: FFT-MA (forward and backward transforms), else spectral.
	// False when the device cannot hold one realisation (the caller then
	// stays on the CPU).
	virtual bool prepare(const SpectralLayout& layout, const float* amplitude, bool movingAverage, std::string& error) = 0;
	// One realisation into out (output grid, gridCount floats, axis 0
	// fastest), mean added. FFT-MA: noiseIn (padded layout, realCount floats)
	// replaces the drawn white noise when not null; noiseOut (padded) receives
	// the white noise used when not null.
	virtual bool simulate(uint64_t seed, uint32_t realisation, float mean, const float* noiseIn, float* noiseOut, float* out, std::string& error) = 0;
	virtual DeviceTimes times() const = 0;
};

// ------------------------------------------------------------ global kriging variance

class KrigingKernel {
public:
	virtual ~KrigingKernel() {}
	// True when the kernels compute in double (always the case for results;
	// float only in the OpenCL test mode, see openclKrigingFloatForTests).
	virtual bool doublePrecision() const = 0;
	// The factor L (n x n row-major, lower triangle) and w = L^-1 1.
	virtual bool prepare(const double* factor, size_t n, const double* w, std::string& error) = 0;
	// Optional table of covariances (GridCovarianceTable values) and, per
	// datum, the table index of the zero offset (GridCovarianceTable::base):
	// then varianceTerms may take target offsets instead of k columns.
	virtual bool setTable(const double* table, size_t tableCount, const uint64_t* dataBase, std::string& error) = 0;
	// Largest number of targets per varianceTerms call.
	virtual size_t maxTargets() const = 0;
	// For m targets: y = L^-1 k_t, sumSquares[t] = y.y, dotW[t] = w.y, with
	// k_t gathered from the table (targetOffsets[t] = GridCovarianceTable::offset)
	// or given (kColumns, n x m row-major, when targetOffsets is null).
	virtual bool varianceTerms(const uint64_t* targetOffsets, const double* kColumns, size_t m, double* sumSquares, double* dotW,
	                           std::string& error) = 0;
	virtual DeviceTimes times() const = 0;
};

// ------------------------------------------------------------ device

class Device {
public:
	virtual ~Device() {}
	virtual const DeviceInfo& info() const = 0;
	// New kernel objects (owned by the caller), or nullptr with the reason
	// when the device cannot run them.
	virtual TurningBandsKernel* turningBands(std::string& reason) = 0;
	virtual SpectralKernel* spectral(std::string& reason) = 0;
	virtual KrigingKernel* kriging(std::string& reason) = 0;
};

// Entry points of g2s_cuda.so (dlsym names), see src/geostatCuda.cu.
// g2s_geostatCudaDevice: the device `index`, or nullptr with a message.
typedef int (*CudaDeviceCountFunction)();
typedef Device* (*CudaDeviceFunction)(int index, char* message, size_t messageSize);
constexpr const char* kCudaDeviceCountSymbol = "g2s_geostatCudaDeviceCount";
constexpr const char* kCudaDeviceSymbol = "g2s_geostatCudaDevice";

} // namespace gpu
} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_GPU_BACKEND_HPP
