/*
 * G2S - geostat: the engines on a GPU back end (gpu/backend.hpp).
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

// Host side of the device paths: the same sequencing as the CPU engines with
// the hot loops replaced by a kernel object. Everything that is cheap or
// needs double precision stays on the host: line parameters, random
// rotations and frequencies (turning bands), the covariance spectrum and its
// square root (FFT), the factorisation (kriging), the nugget and the mean.
//
// ------------------------------------------------------------- tolerances
// GPU results are not bit-identical to the CPU, they agree within (checked by
// test/geostat_gpu, GPU against CPU on the same seed):
//   turning bands     |gpu - cpu| <= 2e-5 sqrt(sill) at every node. The phase
//                     of every line is exact (64-bit fixed point; the CPU
//                     rounds it in double), cosines and dilution profiles
//                     are evaluated in float as on the CPU, sums are float
//                     with a compensated total. Exception, dilution
//                     (spherical, cubic) only: a node lying within ~1e-9 of a
//                     segment boundary of a line may be given to the other
//                     segment (as already between the CPU grid and meshless
//                     paths), which changes it by up to 2 g(1/2) sqrt(sill/L);
//                     the tests bound the fraction of such nodes (1e-5).
//   FFT-MA, spectral  |gpu - cpu| <= 1e-4 sqrt(sill) (float transforms, as
//                     FFTW float; different rounding). The white noise itself
//                     is the CPU's bit for bit when the device has a
//                     correctly rounded sqrt (checked with -on).
//   kriging variance  |gpu - cpu| <= 1e-9 sill + 1e-6 |cpu| (double, blocked
//                     substitution in another order).
// On a given device, results are reproducible for a given seed (no atomics,
// no order-dependent reductions).

#ifndef G2S_GEOSTAT_GPU_DEVICE_ENGINES_HPP
#define G2S_GEOSTAT_GPU_DEVICE_ENGINES_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "geostat/gpu/backend.hpp"
#include "geostat/krigingGlobal.hpp"
#include "geostat/spectralEngine.hpp"
#include "geostat/turningBands.hpp"

namespace g2s {
namespace geostat {
namespace gpu {

// ============================================================ turning bands

// v = whole + fraction / 2^64 with whole = floor(v) modulo 2^32: exact for
// every double |v| < 2^63 (the fraction keeps all its bits).
inline void toFixedPoint(double v, uint32_t& whole, uint64_t& fraction) {
	const double floor = std::floor(v);
	const double rest = v - floor; // exact
	whole = uint32_t(uint64_t(int64_t(floor)));
	const double scaled = std::ldexp(rest, 64);
	fraction = scaled >= 18446744073709551616.0 ? ~uint64_t(0) : uint64_t(scaled);
}

// The lines of a LineSet on a grid in fixed point: the doubles of
// prepareGridLines (shift + sum_d w_d origin_d, w_d spacing_d) converted
// exactly. Same t_l(node) as the CPU grid path, without its double rounding.
struct FixedLines {
	std::vector<uint32_t> whole;
	std::vector<uint64_t> fraction;
	TurningBandsLines view;

	void build(const LineSet& lines, const RegularGrid& grid) {
		const size_t L = lines.count;
		whole.assign(4 * L, 0u);
		fraction.assign(4 * L, 0u);
		for (size_t l = 0; l < L; ++l) {
			double shift = lines.shift[l];
			for (unsigned d = 0; d < lines.ndim; ++d) {
				const double w = lines.direction[d * L + l];
				shift += w * grid.origin[d];
				toFixedPoint(w * grid.spacing[d], whole[(1 + d) * L + l], fraction[(1 + d) * L + l]);
			}
			toFixedPoint(shift, whole[l], fraction[l]);
		}
		view.process = lines.process == LineProcess::Spectral ? 0 : (lines.process == LineProcess::DilutionSpherical ? 1 : 2);
		view.count = uint32_t(L);
		view.amplitude = float(lines.amplitude);
		view.whole = whole.data();
		view.fraction = fraction.data();
		view.signKey = lines.signKey;
	}
};

// Bytes of host memory per piece of rows handed to the device.
constexpr size_t kDevicePieceBytes = size_t(64) << 20;

// simulateTurningBandsGrid on a device: realisations firstRealisation ..
// + realisationCount - 1; for every realisation r (relative) and piece of
// rows, calls
//     store(r, rowBegin, rowEnd, values)   values[(row - rowBegin) * size[0] + i0], float,
//                                          mean and nugget included
// from the pool's threads (concurrently for disjoint rows of one
// realisation), and progress(done) after each realisation. Returns false
// with the error when the device fails (nothing is retried here).
template <class Store, class Progress>
inline bool simulateTurningBandsGridDevice(const ResolvedModel& model, const TurningBandsSettings& settings, const RegularGrid& grid, uint32_t firstRealisation,
                                           unsigned realisationCount, TurningBandsKernel& kernel, WorkerPool& pool, Store store, Progress progress,
                                           std::string& error) {
	checkTurningBandsModel(model);
	checkTurningBandsSettings(settings);
	if (grid.ndim != model.ndim) throw Error("turning bands: the grid has " + std::to_string(grid.ndim) + " dimensions, the model " + std::to_string(model.ndim));
	const size_t columns = grid.size[0], rows = grid.rowCount();
	if (columns == 0 || rows == 0) return true;
	const uint64_t size[3] = { grid.size[0], grid.size[1], grid.size[2] };
	const size_t piece = std::max<size_t>(1, std::min(rows, kDevicePieceBytes / (columns * sizeof(float))));
	std::vector<float> values(piece * columns);
	for (unsigned r = 0; r < realisationCount; ++r) {
		const TurningBandsRealisation realisation(model, settings, firstRealisation + r);
		std::vector<FixedLines> fixed(realisation.lineSetCount());
		std::vector<TurningBandsLines> sets(fixed.size());
		for (size_t s = 0; s < fixed.size(); ++s) {
			fixed[s].build(realisation.lineSet(s), grid);
			sets[s] = fixed[s].view;
		}
		const double mean = settings.mean, nugget = realisation.nuggetDeviation();
		for (size_t rowBegin = 0; rowBegin < rows; rowBegin += piece) {
			const size_t rowEnd = std::min(rows, rowBegin + piece), count = rowEnd - rowBegin;
			if (!kernel.sum(sets.data(), sets.size(), size, rowBegin, rowEnd, values.data(), error)) return false;
			// Mean and nugget (keyed by the node id, as evaluateGrid), then the store, in parallel over rows.
			const size_t groups = std::min<size_t>(count, 4 * size_t(pool.size()));
			pool.run(groups, [&](size_t g, unsigned) {
				const size_t r0 = count * g / groups, r1 = count * (g + 1) / groups;
				std::vector<float> noise(nugget > 0 ? columns : 0);
				for (size_t row = r0; row < r1; ++row) {
					float* v = values.data() + row * columns;
					if (nugget > 0) {
						fillFastNormalsByIndex(settings.seed, firstRealisation + r, uint64_t(rowBegin + row) * columns, columns, 0, noise.data());
						for (size_t i = 0; i < columns; ++i) v[i] = float(mean + double(v[i]) + nugget * double(noise[i]));
					} else if (mean != 0.0) {
						for (size_t i = 0; i < columns; ++i) v[i] = float(mean + double(v[i]));
					}
				}
				if (r1 > r0) store(r, rowBegin + r0, rowBegin + r1, static_cast<const float*>(values.data() + r0 * columns));
			});
		}
		progress(r + 1);
	}
	return true;
}

// ============================================================ FFT-MA / spectral

// False when the layout has more axes than the device kernels (3).
inline bool toDeviceLayout(const spectral::GridLayout& L, SpectralLayout& out) {
	if (L.ndim < 1 || L.ndim > 3) return false;
	out = SpectralLayout();
	out.ndim = L.ndim;
	for (unsigned d = 0; d < L.ndim; ++d) { out.grid[d] = L.grid[d]; out.extended[d] = L.extended[d]; }
	out.cells = L.cells; out.rows = L.rows; out.halfRow = L.halfRow; out.halfCount = L.halfCount;
	out.rowPitch = L.rowPitch; out.realCount = L.realCount;
	return true;
}

// SpectralSimulator::simulate on a device prepared with
// kernel.prepare(layout, simulator.amplitude(), method == FftMovingAverage).
inline bool simulateSpectralDevice(const spectral::SpectralSimulator& simulator, SpectralKernel& kernel, uint64_t seed, uint32_t first, unsigned count, double mean,
                                   spectral::FieldView out, spectral::ConstFieldView noiseIn, spectral::FieldView noiseOut, unsigned threads, std::string& error) {
	const spectral::GridLayout& L = simulator.layout();
	SpectralLayout layout;
	if (!toDeviceLayout(L, layout)) { error = "the device kernels handle 1 to 3 dimensions"; return false; }
	const size_t cells = size_t(layout.gridCount());
	std::vector<float> result(cells);
	std::vector<float> padded((noiseIn.data || noiseOut.data) ? L.realCount : 0);
	for (unsigned i = 0; i < count; ++i) {
		if (noiseIn.data) spectral::importNoise(noiseIn.data + i * noiseIn.realisationStride, noiseIn.cellStride, L, padded.data(), threads);
		if (!kernel.simulate(seed, first + i, float(mean), noiseIn.data ? padded.data() : nullptr, noiseOut.data ? padded.data() : nullptr, result.data(), error)) return false;
		if (noiseOut.data) spectral::exportNoise(padded.data(), L, noiseOut.data + i * noiseOut.realisationStride, noiseOut.cellStride, threads);
		float* target = out.data + i * out.realisationStride;
		const size_t stride = out.cellStride;
		spectral::parallelFor(cells, threads, [&](size_t begin, size_t end) {
			for (size_t c = begin; c < end; ++c) target[c * stride] = result[c];
		});
	}
	return true;
}

// ============================================================ global kriging variance

// GlobalKriging::variance on a device. With a table (and the data cells given
// to factorize, in order) the covariances are gathered on the device from the
// target cells; otherwise `cross` fills them on the host (in parallel) and
// they are uploaded. out[t] for t < targetCount.
template <class CrossCovariance>
inline bool globalKrigingVarianceDevice(const GlobalKriging& kriging, KrigingKernel& kernel, const CrossCovariance& cross, const GridCovarianceTable* table,
                                        const size_t* dataCells, const size_t* targetCells, size_t targetCount, double* out, std::string& error) {
	if (!kernel.doublePrecision()) { error = "the kriging kernels need double precision"; return false; }
	const size_t n = kriging.dataCount();
	if (!kernel.prepare(kriging.factor(), n, kriging.forwardOnes(), error)) return false;
	const bool gather = table && table->ready() && dataCells && targetCells;
	if (gather) {
		std::vector<uint64_t> base(n);
		for (size_t i = 0; i < n; ++i) base[i] = table->base(dataCells[i]);
		size_t tableCount = 1;
		for (size_t d = 0; d < table->dims().size(); ++d) tableCount *= 2 * size_t(table->dims()[d]) - 1;
		if (!kernel.setTable(table->values(), tableCount, base.data(), error)) return false;
	}
	const double c0 = kriging.model().sill(), sumU = kriging.onesQuadraticForm();
	const bool simple = kriging.type() == KrigingType::Simple;
	const size_t chunk = kernel.maxTargets();
	std::vector<double> sums(chunk), dots(chunk), columns(gather ? 0 : n * chunk);
	std::vector<uint64_t> offsets(gather ? chunk : 0);
	const unsigned threads = kriging.threads();
	for (size_t t0 = 0; t0 < targetCount; t0 += chunk) {
		const size_t m = std::min(chunk, targetCount - t0);
		if (gather) {
			for (size_t t = 0; t < m; ++t) offsets[t] = table->offset(targetCells[t0 + t]);
		} else {
			std::vector<std::vector<double> > k(threads, std::vector<double>(n));
			parallelFor(m, threads, [&](size_t t, unsigned thread) {
				cross(t0 + t, k[thread].data());
				for (size_t i = 0; i < n; ++i) columns[i * m + t] = k[thread][i];
			});
		}
		if (!kernel.varianceTerms(gather ? offsets.data() : nullptr, gather ? nullptr : columns.data(), m, sums.data(), dots.data(), error)) return false;
		for (size_t t = 0; t < m; ++t) {
			double v = c0 - sums[t];
			if (!simple) v += (1.0 - dots[t]) * (1.0 - dots[t]) / sumU;
			out[t0 + t] = std::max(0.0, v);
		}
	}
	return true;
}

} // namespace gpu
} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_GPU_DEVICE_ENGINES_HPP
