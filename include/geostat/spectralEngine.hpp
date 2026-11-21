/*
 * G2S - geostat: FFT moving average and spectral simulation engine.
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

// Unconditional stationary Gaussian simulation on a periodic grid, shared by
// the two FFT methods:
//   * FFT moving average (FFT-MA, Le Ravalec et al. 2000): white noise w drawn
//     in space, Y = g (*) w with g the convolution square root of the
//     covariance, computed as Y = IFFT(sqrt(S) FFT(w)). The noise can be given
//     and returned, which is what local perturbation and gradual deformation
//     need.
//   * spectral (Pardo-Iguzquiza & Chica-Olmo 1993; circulant embedding,
//     Dietrich & Newsam 1993/1997): complex Gaussian noise drawn directly in
//     frequency with Hermitian symmetry, Y = IFFT(sqrt(S) Z).
// Both share the amplitude sqrt(S): S is the DFT of the covariance sampled on
// the extended periodic grid (minimal-image lags, computed in double), negative
// values are clipped to zero and the clipped energy is reported.
//
// ---------------------------------------------------------------- normalisation
// FFTW transforms are unnormalised: F x(k) = sum_x x e^{-2 pi i k.x/N},
// B X(x) = sum_k X e^{+2 pi i k.x/N}, B F = M (M = number of cells of the
// extended grid). With c the sampled covariance, S = F c is real and even and
// sum_k S(k) = M c(0). Let s = sqrt(max(S, 0)).
//   FFT-MA:   Y = B(s/M * F w), w iid N(0,1): E|F w(k)|^2 = M, so
//             Var Y = sum_k (s/M)^2 M = sum_k S/M = c(0).
//   spectral: Y = B(s/sqrt(M) * zeta), zeta(k) standard complex normal
//             (real and imaginary parts of variance 1/2) with
//             zeta(-k) = conj(zeta(k)), real N(0,1) where k = -k:
//             Var Y = sum_k S/M = c(0).
// Both give exactly the circulant covariance c on the extended grid (up to
// the clipping), so the variance equals the sill and, inside the output grid,
// the covariance equals the model when the buffer covers the model's reach.
//
// ---------------------------------------------------------------- layout
// Grids use DataImage axis order (axis 0 fastest). Arrays given to the FFT are
// FFTW in-place r2c arrays: real rows padded to rowPitch = 2 (N0/2 + 1) floats,
// the half spectrum has N0/2 + 1 complex values per row (interleaved re, im).
// A "row" is one line along axis 0; rows are numbered over axes 1..ndim-1
// (axis 1 fastest). Random numbers are indexed by the unpadded flat index of
// the extended grid (white noise) or of the full spectrum (spectral), with
// geostat/random.hpp, so results do not depend on the number of threads, the
// way realisations are batched, or on the back end.
//
// ---------------------------------------------------------------- back ends
// The hot loops are free functions on plain arrays (sampleCovariance,
// amplitudeFromSpectrum, fillWhiteNoise, importNoise, exportNoise,
// multiplyAmplitude, fillSpectralNoise, cropToGrid) and the transforms sit in
// FftwBackend (forward r2c / backward c2r on one buffer). A GPU back end
// replaces these with device kernels and cuFFT / clFFT plans with the same
// layout; SpectralSimulator only sequences them.
//
// FFTW plans are made with FFTW_ESTIMATE, which picks the same algorithm for a
// given size whatever the number of threads; checked bitwise in
// test/geostat_fft (FFTW does not formally promise it).

#ifndef G2S_GEOSTAT_SPECTRAL_ENGINE_HPP
#define G2S_GEOSTAT_SPECTRAL_ENGINE_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef WITH_MKL
	#include <fftw/fftw3_mkl.h>
#else
	#include <fftw3.h>
#endif

#include "geostat/covarianceModel.hpp"
#include "geostat/random.hpp"

namespace g2s {
namespace geostat {
namespace spectral {

enum class Method { FftMovingAverage, Spectral };

inline const char* methodName(Method method) { return method == Method::Spectral ? "spectral" : "fftma"; }

// Stream numbers of geostat/random.hpp used by the two methods.
constexpr uint32_t kWhiteNoiseStream = 0;
constexpr uint32_t kSpectralNoiseStream = 1;

// ============================================================ threads

// Runs f(begin, end) over [0, count) split in `threads` contiguous chunks
// (inline when threads <= 1). Every element is computed the same way whatever
// the split, so results do not depend on the thread count.
template <class F>
void parallelFor(size_t count, unsigned threads, F f) {
	if (count == 0) return;
	const size_t chunks = std::min<size_t>(std::max(1u, threads), count);
	if (chunks <= 1) { f(size_t(0), count); return; }
	std::vector<std::thread> workers;
	workers.reserve(chunks - 1);
	for (size_t c = 1; c < chunks; ++c) workers.emplace_back([=]() { f(count * c / chunks, count * (c + 1) / chunks); });
	f(0, count / chunks);
	for (std::thread& t : workers) t.join();
}

// ============================================================ grid sizes

// True when n has no prime factor above 7 (fast FFT sizes).
inline bool isFftFriendly(size_t n) {
	if (n == 0) return false;
	for (size_t p : { 2, 3, 5, 7 }) while (n % p == 0) n /= p;
	return n == 1;
}

// Smallest FFT-friendly size >= n.
inline size_t nextFftFriendly(size_t n) {
	if (n <= 1) return 1;
	while (!isFftFriendly(n)) ++n;
	return n;
}

// Extended periodic grid: grid[d] + buffer[d] rounded up to an FFT-friendly
// size, or the grid itself when periodic. Axes of size 1 stay 1 (a slice of a
// higher-dimensional field: only lag 0 occurs along them).
inline std::vector<unsigned> extendedGrid(const std::vector<unsigned>& grid, const std::vector<double>& buffer, bool periodic) {
	std::vector<unsigned> extended(grid);
	if (periodic) return extended;
	for (size_t d = 0; d < grid.size(); ++d) {
		if (grid[d] <= 1) continue;
		const double cells = std::ceil(std::max(0.0, d < buffer.size() ? buffer[d] : 0.0));
		if (double(grid[d]) + cells > 2147483647.0) throw Error("the buffer along axis " + std::to_string(d) + " is too large (" + formatNumber(cells) + " cells)");
		extended[d] = unsigned(nextFftFriendly(size_t(grid[d]) + size_t(cells)));
	}
	return extended;
}

// Geometry of the extended grid and of its FFTW in-place arrays.
struct GridLayout {
	unsigned ndim = 0;
	size_t grid[kMaxDims] = {};      // output grid
	size_t extended[kMaxDims] = {};  // periodic simulation grid (N)
	size_t cells = 0;                // M = prod N
	size_t rows = 0;                 // M / N0
	size_t halfRow = 0;              // N0/2 + 1 complex values per row
	size_t halfCount = 0;            // halfRow * rows complex values
	size_t rowPitch = 0;             // 2 * halfRow floats per padded real row
	size_t realCount = 0;            // rowPitch * rows floats (= 2 halfCount)

	GridLayout() = default;
	GridLayout(const std::vector<unsigned>& outputGrid, const std::vector<unsigned>& extendedGrid) {
		if (outputGrid.empty() || outputGrid.size() > kMaxDims || outputGrid.size() != extendedGrid.size())
			throw Error("the grid must have 1 to " + std::to_string(kMaxDims) + " dimensions");
		ndim = unsigned(outputGrid.size());
		cells = 1;
		for (unsigned d = 0; d < ndim; ++d) {
			grid[d] = outputGrid[d];
			extended[d] = extendedGrid[d];
			if (grid[d] == 0 || extended[d] < grid[d]) throw Error("the extended grid must cover the grid on every axis");
			cells *= extended[d];
		}
		rows = cells / extended[0];
		halfRow = extended[0] / 2 + 1;
		halfCount = halfRow * rows;
		rowPitch = 2 * halfRow;
		realCount = rowPitch * rows;
	}
	// Coordinates along axes 1..ndim-1 of row `row`.
	void rowCoordinates(size_t row, size_t* coordinates) const {
		for (unsigned d = 1; d < ndim; ++d) { coordinates[d] = row % extended[d]; row /= extended[d]; }
	}
	// Row of the given coordinates along axes 1..ndim-1.
	size_t rowIndex(const size_t* coordinates) const {
		size_t row = 0;
		for (unsigned d = ndim; d-- > 1;) row = row * extended[d] + coordinates[d];
		return row;
	}
	// FFTW dimensions (slowest first).
	std::vector<int> fftwDims() const {
		std::vector<int> n(ndim);
		for (unsigned d = 0; d < ndim; ++d) n[ndim - 1 - d] = int(extended[d]);
		return n;
	}
};

// ============================================================ hot loops

// Covariance sampled on the extended periodic grid with minimal-image lags,
// written to an in-place r2c array of doubles (padding set to 0). At the
// Nyquist index of an even axis the lag +N/2 and -N/2 coincide: the value is
// averaged over both signs so that c(k) = c(-k) exactly (rotated anisotropy is
// not symmetric per axis) and the spectrum is real.
inline void sampleCovariance(const ResolvedModel& model, const GridLayout& L, double* padded, unsigned threads) {
	const unsigned ndim = L.ndim;
	auto lagOf = [](size_t k, size_t n, bool& nyquist) {
		nyquist = (n % 2 == 0) && n > 1 && k == n / 2;
		return k <= n / 2 ? double(k) : double(k) - double(n);
	};
	parallelFor(L.rows, threads, [&](size_t begin, size_t end) {
		size_t coordinates[kMaxDims] = {};
		double lag[kMaxDims] = {}, flipped[kMaxDims] = {};
		bool nyquist[kMaxDims] = {};
		for (size_t row = begin; row < end; ++row) {
			L.rowCoordinates(row, coordinates);
			unsigned nyquistMask = 0;
			for (unsigned d = 1; d < ndim; ++d) {
				lag[d] = lagOf(coordinates[d], L.extended[d], nyquist[d]);
				if (nyquist[d]) nyquistMask |= 1u << d;
			}
			double* out = padded + row * L.rowPitch;
			for (size_t k0 = 0; k0 < L.extended[0]; ++k0) {
				lag[0] = lagOf(k0, L.extended[0], nyquist[0]);
				const unsigned mask = nyquistMask | (nyquist[0] ? 1u : 0u);
				if (mask == 0) { out[k0] = model.covariance(lag); continue; }
				// Average over the sign combinations of the Nyquist components.
				double sum = 0.0;
				unsigned combinations = 0;
				for (unsigned signs = 0; signs < (1u << ndim); ++signs) {
					if ((signs & ~mask) != 0) continue;
					for (unsigned d = 0; d < ndim; ++d) flipped[d] = (signs >> d & 1u) ? -lag[d] : lag[d];
					sum += model.covariance(flipped);
					++combinations;
				}
				out[k0] = sum / double(combinations);
			}
			for (size_t k0 = L.extended[0]; k0 < L.rowPitch; ++k0) out[k0] = 0.0;
		}
	});
}

// What the clipping of negative spectral values removed.
struct SpectrumReport {
	double sill = 0.0;             // c(0) of the sampled covariance (sum S / M)
	double effectiveSill = 0.0;    // variance of the simulated field (sum max(S,0) / M)
	double positiveEnergy = 0.0;   // sum of S over S > 0 (full spectrum)
	double clippedEnergy = 0.0;    // sum of -S over S < 0 (full spectrum)
	double clippedFraction = 0.0;  // clippedEnergy / (positiveEnergy + clippedEnergy)
	size_t clippedCount = 0;       // number of negative frequencies (full spectrum)
};

// Whether the half-spectrum entry (k0, row) is on a plane k0 = 0 or k0 = N0/2
// (N0 even), where its Hermitian partner -k is stored in the half spectrum
// too; partnerRow receives the row of -k.
inline bool onSelfPartnerPlane(const GridLayout& L, size_t k0, size_t row, size_t& partnerRow) {
	if (!(k0 == 0 || (L.extended[0] % 2 == 0 && k0 == L.extended[0] / 2))) return false;
	size_t coordinates[kMaxDims] = {};
	L.rowCoordinates(row, coordinates);
	for (unsigned d = 1; d < L.ndim; ++d) coordinates[d] = (L.extended[d] - coordinates[d]) % L.extended[d];
	partnerRow = L.rowIndex(coordinates);
	return true;
}

// From the half spectrum S (FFTW r2c output of the sampled covariance, complex
// doubles interleaved) to the amplitude sqrt(max(S, 0)) in float (halfCount
// values). Also makes the amplitude exactly even on the self-partner planes.
inline SpectrumReport amplitudeFromSpectrum(const double* halfSpectrum, const GridLayout& L, float* amplitude, unsigned threads) {
	// Per-row sums, added in row order: the report does not depend on the thread count.
	std::vector<double> positive(L.rows, 0.0), negative(L.rows, 0.0);
	std::vector<size_t> negativeCount(L.rows, 0);
	parallelFor(L.rows, threads, [&](size_t begin, size_t end) {
		for (size_t row = begin; row < end; ++row) {
			double sumPositive = 0.0, sumNegative = 0.0;
			size_t countNegative = 0;
			for (size_t k0 = 0; k0 < L.halfRow; ++k0) {
				const size_t i = row * L.halfRow + k0;
				const double s = halfSpectrum[2 * i];
				// Each interior entry stands for itself and its implicit conjugate.
				const double weight = (k0 == 0 || (L.extended[0] % 2 == 0 && k0 == L.extended[0] / 2)) ? 1.0 : 2.0;
				if (s > 0) sumPositive += weight * s;
				else if (s < 0) { sumNegative -= weight * s; countNegative += size_t(weight); }
				amplitude[i] = float(std::sqrt(std::max(s, 0.0)));
			}
			positive[row] = sumPositive;
			negative[row] = sumNegative;
			negativeCount[row] = countNegative;
		}
	});
	// Exact evenness on the self-partner planes (rounding can break it slightly).
	for (size_t row = 0; row < L.rows; ++row) {
		for (size_t k0 : { size_t(0), L.extended[0] / 2 }) {
			if (k0 >= L.halfRow) continue;
			size_t partner = 0;
			if (!onSelfPartnerPlane(L, k0, row, partner) || partner >= row) continue;
			amplitude[row * L.halfRow + k0] = amplitude[partner * L.halfRow + k0];
		}
	}
	SpectrumReport report;
	for (size_t row = 0; row < L.rows; ++row) {
		report.positiveEnergy += positive[row];
		report.clippedEnergy += negative[row];
		report.clippedCount += negativeCount[row];
	}
	const double M = double(L.cells);
	report.sill = (report.positiveEnergy - report.clippedEnergy) / M;
	report.effectiveSill = report.positiveEnergy / M;
	const double total = report.positiveEnergy + report.clippedEnergy;
	report.clippedFraction = total > 0 ? report.clippedEnergy / total : 0.0;
	return report;
}

// White noise N(0,1) of realisation `realisation` on the extended grid,
// written to an in-place r2c float array (padding set to 0). Cell x (unpadded
// flat index) takes fastNormalByIndex(seed, realisation, x, kWhiteNoiseStream).
inline void fillWhiteNoise(uint64_t seed, uint32_t realisation, const GridLayout& L, float* padded, unsigned threads) {
	const size_t N0 = L.extended[0];
	parallelFor(L.rows, threads, [&](size_t begin, size_t end) {
		for (size_t row = begin; row < end; ++row) {
			float* out = padded + row * L.rowPitch;
			fillFastNormalsByIndex(seed, realisation, row * N0, N0, kWhiteNoiseStream, out);
			for (size_t k = N0; k < L.rowPitch; ++k) out[k] = 0.f;
		}
	});
}

// Noise given on the extended grid (cell x at noise[x * cellStride]) into the
// padded array.
inline void importNoise(const float* noise, size_t cellStride, const GridLayout& L, float* padded, unsigned threads) {
	const size_t N0 = L.extended[0];
	parallelFor(L.rows, threads, [&](size_t begin, size_t end) {
		for (size_t row = begin; row < end; ++row) {
			float* out = padded + row * L.rowPitch;
			const float* in = noise + row * N0 * cellStride;
			for (size_t x = 0; x < N0; ++x) out[x] = in[x * cellStride];
			for (size_t k = N0; k < L.rowPitch; ++k) out[k] = 0.f;
		}
	});
}

// The padded noise back to an extended-grid array (cell x at noise[x * cellStride]).
inline void exportNoise(const float* padded, const GridLayout& L, float* noise, size_t cellStride, unsigned threads) {
	const size_t N0 = L.extended[0];
	parallelFor(L.rows, threads, [&](size_t begin, size_t end) {
		for (size_t row = begin; row < end; ++row) {
			const float* in = padded + row * L.rowPitch;
			float* out = noise + row * N0 * cellStride;
			for (size_t x = 0; x < N0; ++x) out[x * cellStride] = in[x];
		}
	});
}

// spectrum[i] *= scale * amplitude[i] (complex interleaved, halfCount values).
inline void multiplyAmplitude(const float* amplitude, float scale, float* spectrum, size_t halfCount, unsigned threads) {
	parallelFor(halfCount, threads, [&](size_t begin, size_t end) {
		for (size_t i = begin; i < end; ++i) {
			const float a = scale * amplitude[i];
			spectrum[2 * i] *= a;
			spectrum[2 * i + 1] *= a;
		}
	});
}

// Complex Gaussian noise times scale * amplitude in the half spectrum, with
// Hermitian symmetry: frequency k (flat index f over the full extended grid)
// uses the normals 2 g and 2 g + 1 of fastNormalByIndex(stream
// kSpectralNoiseStream), g = min(f, f(-k)), as (real, imaginary) parts;
// interior entries (0 < k0 < N0/2) are independent, entries on the
// self-partner planes are the conjugate of their partner, and self-conjugate
// entries are real N(0,1).
inline void fillSpectralNoise(uint64_t seed, uint32_t realisation, const GridLayout& L, const float* amplitude, float scale,
                              float* spectrum, unsigned threads) {
	const size_t N0 = L.extended[0];
	const double half = std::sqrt(0.5);
	parallelFor(L.rows, threads, [&](size_t begin, size_t end) {
		std::vector<float> normals(2 * L.halfRow);
		for (size_t row = begin; row < end; ++row) {
			fillFastNormalsByIndex(seed, realisation, 2 * row * N0, 2 * L.halfRow, kSpectralNoiseStream, normals.data());
			for (size_t k0 = 0; k0 < L.halfRow; ++k0) {
				const size_t i = row * L.halfRow + k0;
				const size_t flat = row * N0 + k0;
				size_t partnerRow = row;
				double re = double(normals[2 * k0]), im = double(normals[2 * k0 + 1]);
				if (!onSelfPartnerPlane(L, k0, row, partnerRow)) {
					re *= half; im *= half;
				} else {
					const size_t partnerFlat = partnerRow * N0 + k0;
					if (partnerFlat == flat) {
						im = 0.0;
					} else if (flat < partnerFlat) {
						re *= half; im *= half;
					} else {
						re = double(fastNormalByIndex(seed, realisation, 2 * partnerFlat, kSpectralNoiseStream));
						im = double(fastNormalByIndex(seed, realisation, 2 * partnerFlat + 1, kSpectralNoiseStream));
						re *= half; im *= -half;
					}
				}
				const double a = double(scale) * double(amplitude[i]);
				spectrum[2 * i] = float(a * re);
				spectrum[2 * i + 1] = float(a * im);
			}
		}
	});
}

// out[x * cellStride] = mean + padded value, for every cell x of the output
// grid (the corner of the extended grid at the origin).
inline void cropToGrid(const float* padded, const GridLayout& L, double mean, float* out, size_t cellStride, unsigned threads) {
	size_t outRows = 1;
	for (unsigned d = 1; d < L.ndim; ++d) outRows *= L.grid[d];
	const float m = float(mean);
	parallelFor(outRows, threads, [&](size_t begin, size_t end) {
		size_t coordinates[kMaxDims] = {};
		for (size_t outRow = begin; outRow < end; ++outRow) {
			size_t r = outRow;
			for (unsigned d = 1; d < L.ndim; ++d) { coordinates[d] = r % L.grid[d]; r /= L.grid[d]; }
			const float* in = padded + L.rowIndex(coordinates) * L.rowPitch;
			float* o = out + outRow * L.grid[0] * cellStride;
			for (size_t x = 0; x < L.grid[0]; ++x) o[x * cellStride] = m + in[x];
		}
	});
}

// ============================================================ FFTW back end

inline void initFftwThreads() {
	static std::once_flag once;
	std::call_once(once, []() { fftwf_init_threads(); fftw_init_threads(); });
}

// The amplitude sqrt(max(S, 0)) of `model` on the extended grid; the spectrum
// is computed in double with FFTW.
inline SpectrumReport computeAmplitude(const ResolvedModel& model, const GridLayout& L, std::vector<float>& amplitude, unsigned threads) {
	initFftwThreads();
	double* buffer = static_cast<double*>(fftw_malloc(sizeof(double) * L.realCount));
	if (!buffer) throw Error("not enough memory for the covariance spectrum (" + std::to_string(L.realCount * sizeof(double) >> 20) + " MiB)");
	const std::vector<int> n = L.fftwDims();
	fftw_plan_with_nthreads(int(std::max(1u, threads)));
	fftw_plan plan = fftw_plan_dft_r2c(int(L.ndim), n.data(), buffer, reinterpret_cast<fftw_complex*>(buffer), FFTW_ESTIMATE);
	if (!plan) { fftw_free(buffer); throw Error("FFTW could not plan the covariance transform"); }
	sampleCovariance(model, L, buffer, threads);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
	amplitude.assign(L.halfCount, 0.f);
	const SpectrumReport report = amplitudeFromSpectrum(buffer, L, amplitude.data(), threads);
	fftw_free(buffer);
	return report;
}

// In-place single-precision r2c / c2r transforms of one buffer.
class FftwBackend {
public:
	FftwBackend() = default;
	FftwBackend(const FftwBackend&) = delete;
	FftwBackend& operator=(const FftwBackend&) = delete;
	~FftwBackend() { release(); }

	// Plans the transforms (not thread-safe: call from one thread at a time).
	void create(const GridLayout& L, bool withForward, unsigned threads) {
		release();
		initFftwThreads();
		_buffer = static_cast<float*>(fftwf_malloc(sizeof(float) * L.realCount));
		if (!_buffer) throw Error("not enough memory for the FFT buffer (" + std::to_string(L.realCount * sizeof(float) >> 20) + " MiB)");
		const std::vector<int> n = L.fftwDims();
		fftwf_plan_with_nthreads(int(std::max(1u, threads)));
		fftwf_complex* complexBuffer = reinterpret_cast<fftwf_complex*>(_buffer);
		if (withForward) _forward = fftwf_plan_dft_r2c(int(L.ndim), n.data(), _buffer, complexBuffer, FFTW_ESTIMATE);
		_backward = fftwf_plan_dft_c2r(int(L.ndim), n.data(), complexBuffer, _buffer, FFTW_ESTIMATE);
		if ((withForward && !_forward) || !_backward) { release(); throw Error("FFTW could not plan the transforms"); }
	}
	float* buffer() const { return _buffer; }
	void forward() const { fftwf_execute(_forward); }
	void backward() const { fftwf_execute(_backward); }

private:
	float* _buffer = nullptr;
	fftwf_plan _forward = nullptr, _backward = nullptr;
	void release() {
		if (_forward) fftwf_destroy_plan(_forward);
		if (_backward) fftwf_destroy_plan(_backward);
		if (_buffer) fftwf_free(_buffer);
		_forward = _backward = nullptr;
		_buffer = nullptr;
	}
};

// ============================================================ simulator

// Strided view of R fields: value of cell x of realisation r at
// data[x * cellStride + r * realisationStride]. For a DataImage with R
// variables: { image._data, R, 1 }.
struct FieldView {
	float* data = nullptr;
	size_t cellStride = 1;
	size_t realisationStride = 0;
};
struct ConstFieldView {
	const float* data = nullptr;
	size_t cellStride = 1;
	size_t realisationStride = 0;
};

// Batch of realisations with one amplitude. Typical use (a conditioning step
// calls the same thing and krigs the residuals afterwards):
//   SpectralSimulator simulator(model, grid, extendedGrid(grid, buffer, periodic), Method::FftMovingAverage, threads);
//   simulator.simulate(seed, 0, R, mean, { out, R, 1 });
class SpectralSimulator {
public:
	SpectralSimulator(const ResolvedModel& model, const std::vector<unsigned>& grid, const std::vector<unsigned>& extended,
	                  Method method, unsigned threads)
		: _layout(grid, extended), _method(method), _threads(std::max(1u, threads)) {
		if (model.ndim != _layout.ndim)
			throw Error("the covariance model has " + std::to_string(model.ndim) + " dimensions, the grid " + std::to_string(_layout.ndim));
		_report = computeAmplitude(model, _layout, _amplitude, _threads);
	}

	const GridLayout& layout() const { return _layout; }
	const SpectrumReport& report() const { return _report; }
	Method method() const { return _method; }
	const std::vector<float>& amplitude() const { return _amplitude; }

	// Number of realisations computed concurrently (each with its own buffer
	// and threads / workers FFT threads); 0 = automatic: min(threads, count),
	// within memoryBudget bytes of FFT buffers.
	unsigned realisationWorkers = 0;
	size_t memoryBudget = size_t(4) << 30;

	// Simulates realisations first .. first+count-1 into `out` (output grid).
	// FFT-MA only: noiseIn (extended grid) replaces the drawn white noise;
	// noiseOut receives the white noise used (extended grid).
	void simulate(uint64_t seed, uint32_t first, unsigned count, double mean, FieldView out,
	              ConstFieldView noiseIn = ConstFieldView(), FieldView noiseOut = FieldView()) {
		if (count == 0) return;
		if (_method == Method::Spectral && (noiseIn.data || noiseOut.data))
			throw Error("noise input and output are only available with fftma");
		const unsigned workers = workerCount(count);
		const unsigned inner = std::max(1u, _threads / workers);
		std::vector<FftwBackend> backends(workers);
		for (FftwBackend& backend : backends) backend.create(_layout, _method == Method::FftMovingAverage, inner);
		std::atomic<unsigned> next(0);
		auto work = [&](unsigned w) {
			for (unsigned i = next++; i < count; i = next++) one(backends[w], inner, seed, first + i, mean, out, noiseIn, noiseOut, i);
		};
		if (workers == 1) { work(0); return; }
		std::vector<std::thread> pool;
		for (unsigned w = 1; w < workers; ++w) pool.emplace_back(work, w);
		work(0);
		for (std::thread& t : pool) t.join();
	}

private:
	GridLayout _layout;
	Method _method;
	unsigned _threads;
	std::vector<float> _amplitude;
	SpectrumReport _report;

	unsigned workerCount(unsigned count) const {
		unsigned workers = realisationWorkers > 0 ? realisationWorkers : std::min(_threads, count);
		const size_t perWorker = _layout.realCount * sizeof(float);
		const size_t affordable = std::max<size_t>(1, memoryBudget / std::max<size_t>(1, perWorker));
		if (realisationWorkers == 0) workers = unsigned(std::min<size_t>(workers, affordable));
		return std::max(1u, std::min(workers, count));
	}

	void one(const FftwBackend& fft, unsigned threads, uint64_t seed, uint32_t realisation, double mean, FieldView out,
	         ConstFieldView noiseIn, FieldView noiseOut, unsigned i) const {
		float* buffer = fft.buffer();
		const double M = double(_layout.cells);
		if (_method == Method::FftMovingAverage) {
			if (noiseIn.data) importNoise(noiseIn.data + i * noiseIn.realisationStride, noiseIn.cellStride, _layout, buffer, threads);
			else fillWhiteNoise(seed, realisation, _layout, buffer, threads);
			if (noiseOut.data) exportNoise(buffer, _layout, noiseOut.data + i * noiseOut.realisationStride, noiseOut.cellStride, threads);
			fft.forward();
			multiplyAmplitude(_amplitude.data(), float(1.0 / M), buffer, _layout.halfCount, threads);
		} else {
			fillSpectralNoise(seed, realisation, _layout, _amplitude.data(), float(1.0 / std::sqrt(M)), buffer, threads);
		}
		fft.backward();
		cropToGrid(buffer, _layout, mean, out.data + i * out.realisationStride, out.cellStride, threads);
	}
};

} // namespace spectral
} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_SPECTRAL_ENGINE_HPP
