/*
 * G2S - geostat: turning-bands Gaussian simulation engine.
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

// Unconditional Gaussian simulation by turning bands (Matheron 1973), in the
// modern formulation of Emery & Lantuejoul (2006, TBSIM, Computers &
// Geosciences 32:1615-1628). Two evaluation paths share the same lines:
//   * meshless: an array of point coordinates (N x ndim doubles, row-major,
//     in the same units and axis order as the covariance lags);
//   * regular grid: nodes origin_d + spacing_d i_d, evaluated by blocks of
//     rows with per-axis tables (much faster; src/tb.cpp uses it).
// Nothing here knows about DataImage.
//
// ------------------------------------------------------------------ method
// Each nested structure s of the model (sill c_s, anisotropy transform A_s,
// see covarianceModel.hpp) is simulated separately, in its own isotropic
// space y = A_s x where its correlation is rho_s(|y|) with unit practical
// range, with its own L lines:
//     Z(x) = mean + sum_s sqrt(c_s / L) sum_l Y_sl(<A_s x, u_sl>) + nugget,
// where the Y_sl are independent unit-variance one-dimensional processes and
// the nugget is independent N(0, c_nugget) noise per node (keyed by the node
// id, so it does not depend on the order or batching of the points).
//
// Directions u_sl: an equidistributed set (2D: L regularly spaced directions
// on the half circle; 3D: the Fibonacci / golden-angle lattice on the
// half sphere, z_l = (l + 1/2)/L, which is equal-area) turned by a uniform
// random rotation drawn per realisation and per structure. The rotation makes
// every u_sl uniformly distributed, so the covariance is exact in expectation
// for any L; the low-discrepancy set makes each realisation close to ergodic
// (no banding).
//
// One-dimensional processes:
//   * spherical, cubic: dilution (piecewise) line processes in R^3 (Matheron
//     1973; Lantuejoul 2002, Geostatistical Simulation, Springer, ch. 15;
//     Emery & Lantuejoul 2006, table 1). The line covariance of a 3D
//     isotropic covariance C3 is C1(r) = d/dr [r C3(r)]; with the line cut
//     into unit segments with a uniform random origin and an independent
//     random sign e_k per segment,
//         Y(t) = e_k g(t - k - 1/2), k = floor(t),
//     has covariance C1(r) = int g(s) g(s + r) ds. For unit practical range:
//         spherical  C1 = 1 - 3r + 2r^3,              g(s) = 2 sqrt(3) s
//         cubic      C1 = 1 - 21r^2 + 35r^3 - 21r^5 + 6r^7,
//                                                    g(s) = sqrt(105/2) s (1 - 4 s^2)
//     on |s| < 1/2 (both checked by direct integration). The spectral
//     densities of these models have heavy, oscillating tails, so the
//     spectral method converges badly for them; the dilution is exact.
//     These line processes are 3D ones: in 2D and 1D the points are embedded
//     in R^3 (third coordinates 0) and the directions are 3D. This is exact
//     here because the foundation defines spherical and cubic by the same
//     formula of |A h| in every dimension, which is precisely the restriction
//     of the 3D isotropic covariance to a plane or a line, and the restriction
//     of a stationary isotropic 3D field to a plane has the restricted
//     covariance.
//     Signs: e_k = -1 when the top bit of mix32(uint32(k) * a_l + b_l) is
//     set, +1 otherwise, with mix32 the MurmurHash3 finaliser (a bijection of
//     the 32-bit integers with full avalanche) and (a_l odd, b_l) 64 random
//     bits per line. Stateless, so no table and no branch: it vectorises.
//     Different multipliers keep two lines from having shifted copies of the
//     same sign sequence. The sequence repeats after 2^32 segments.
//   * exponential, gaussian, matern, stable: continuous spectral method
//     (Shinozuka 1971; Emery & Lantuejoul 2006, sec. 3.2; Lantuejoul 2002):
//         Y(t) = sqrt(2) cos(w t + phi), phi ~ U(0, 2 pi),
//     with the frequency vector w u drawn from the d-dimensional spectral
//     measure of the structure (d = ndim, the native dimension: the 2D field
//     has exactly the 2D covariance). All four are Gaussian scale mixtures,
//     so with chi_d = |G| (G a standard normal d-vector, giving the modulus)
//     and, in unit-practical-range space (k = practical range / scale, i.e.
//     ResolvedStructure::scaleToRange):
//         gaussian     rho = exp(-3t^2)           |w| = sqrt(6) chi_d
//         exponential  rho = exp(-3t)             |w| = 3 chi_d / |N(0,1)|
//                      (multivariate Cauchy: matern with nu = 1/2)
//         matern       rho = M_nu(k t)            |w| = k chi_d / sqrt(2 Gamma(nu, 1))
//                      (multivariate Student t with 2 nu degrees of freedom)
//         stable       rho = exp(-(k t)^alpha)    |w| = k sqrt(2 S) chi_d,
//                      S positive (alpha/2)-stable, E exp(-s S) = exp(-s^(alpha/2)),
//                      drawn with Kanter's (1975) representation
//                      (Chambers, Mallows & Stuck 1976).
//     Check: E cos(<w u, h>) = E exp(-W |h|^2 / 2) gives back rho for each
//     mixing variable W. Frequencies are capped at 1e9 per practical range
//     (heavy-tailed draws beyond any grid resolution behave as white noise).
//     The phase is kept in turns (t / 2 pi).
//   * nugget: independent noise. map(...): refused (no line process).
//
// Random numbers are counter-based (random.hpp): every draw is a function of
// (seed, realisation, index, stream), so results do not depend on the number
// of threads, on how the points are split into blocks, or on how realisations
// are batched. Streams: 0 nugget (fastNormalByIndex, index = node id), 1 rotations (index =
// structure), 2 + 8 s + k the k-th draw of structure s (index = line, or
// line * 64 + attempt; k = 4 is the sign key of a dilution line).
//
// ------------------------------------------------------------------ precision
// The outputs are meant for float images. Phases are computed and reduced in
// double (t - round(t) in turns, valid for |t| < 2^51 turns or segments, far
// beyond the point where a double phase means anything), the reduced value
// (|.| <= 1/2) is then rounded to float, which costs at most 3e-8 turns,
// independently of the size of the coordinates or of the frequency: there is
// no loss of accuracy far from the origin or at high frequency. Cosines and
// dilution profiles are evaluated in float (cosine: degree-9 odd minimax
// polynomial, 3.4e-9 truncation, < 2.5e-7 total error per term); each line's
// term is added in float over tiles of 64 lines (kLineTile), and the tiles in
// double. Measured against a long-double reference (test/geostat_tb, 1000
// lines): |error| <= 7e-7 sqrt(sill) on the meshless path (test bound 1.5e-6),
// also with coordinates near 1e6 and frequencies of several turns per unit
// (about 1 float ulp of a unit-variance value; the error is the float
// rounding of each line's term, summed as a random walk). The grid path:
//   * spectral: cos(2 pi (row phase + column phase)) = cos_r cos_c - sin_r sin_c
//     with both factors evaluated directly (no rotation recurrence, so no error
//     growth along a row: every table entry is a direct evaluation); same
//     error level as the meshless path.
//   * dilution: along a row, each line's profile is a polynomial in the column
//     index between segment crossings; the polynomial coefficients of every
//     line are added into difference arrays at the crossings and the row is
//     rebuilt with running (prefix) sums in double. The running sums restart
//     at every column tile of kGridColumnTile = 512 nodes (the periodic
//     resynchronisation), with the polynomial variable centred on the tile, so
//     |u| <= 256; lines steeper than 1/4 segment per node use the direct
//     evaluation. Bound: each running sum adds at most L (1 + 128) terms of
//     magnitude <= 4 c u^3 w^3 ~ 2e6 (cubic, |w| = 1/4) or <= 250 (spherical),
//     i.e. an error below 1e-6 (cubic, worst case) and 1e-12 (spherical) per
//     unit amplitude: in practice the dilution grid path is accurate to about
//     1e-9 (double arithmetic throughout).
// Grid and meshless paths therefore agree to within 3e-6 sqrt(sill) (test
// bound; measured <= 1.5e-6),
// but not bit for bit; each path is bit-for-bit deterministic on its own
// (any thread count, any batching of realisations, any order of points).
//
// ----------------------------------------------------------------- usage
//   const ResolvedModel& model = turningBandsModel(modelSet);   // throws Error
//   TurningBandsSettings settings; settings.seed = 7; settings.threads = 8;
//   simulateTurningBands(model, settings, points, count, 0, R, out);   // out[r * count + i]
// or one realisation at a time, on points generated on the fly:
//   TurningBandsRealisation realisation(model, settings, r);
//   realisation.evaluate(blockPoints, blockCount, nullptr, firstNodeId, blockOut);
// Regular grid (node id = i0 + n0 (i1 + n1 i2)):
//   WorkerPool pool(threads);
//   simulateTurningBandsGrid(model, settings, grid, 0, R, pool, store, progress);
// where store(r, block, values) receives the values of every block of
// planGridBlocks(grid) (called concurrently, for disjoint blocks).
//
// ----------------------------------------------------------------- kernels
// The innermost loops have plain-array interfaces so that other back ends
// (OpenCL, CUDA) can replace them:
//   sumLines(LineSet, points, count, out)                       meshless
//   sumLinesGrid(LineSet, GridLines, RegularGrid, GridBlock, out, scratch)   grid
// Memory layouts are documented at LineSet, RegularGrid, GridBlock and
// GridLines below. A GPU back end needs only LineSet, RegularGrid and
// GridBlock (GridLines are CPU lookup tables derived from them).

#ifndef G2S_GEOSTAT_TURNING_BANDS_HPP
#define G2S_GEOSTAT_TURNING_BANDS_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "geostat/covarianceModel.hpp"
#include "geostat/random.hpp"
#include "geostat/simdLanes.hpp"

namespace g2s {
namespace geostat {

// ============================================================ support

enum class LineProcess { Spectral, DilutionSpherical, DilutionCubic };

inline const char* lineProcessName(LineProcess process) {
	switch (process) {
		case LineProcess::Spectral: return "continuous spectral";
		case LineProcess::DilutionSpherical: return "dilution (spherical)";
		case LineProcess::DilutionCubic: return "dilution (cubic)";
	}
	return "?";
}

// Line process of a structure; false (with the reason) when turning bands
// cannot simulate it. Nugget is supported (as independent noise) and returns
// true with process left unchanged.
inline bool turningBandsLineProcess(const ResolvedStructure& structure, LineProcess& process, std::string* reason = nullptr) {
	switch (structure.type) {
		case StructureType::Nugget: return true;
		case StructureType::Spherical: process = LineProcess::DilutionSpherical; return true;
		case StructureType::Cubic: process = LineProcess::DilutionCubic; return true;
		case StructureType::Exponential:
		case StructureType::Gaussian:
		case StructureType::Matern:
		case StructureType::Stable: process = LineProcess::Spectral; return true;
		case StructureType::Map:
			if (reason) *reason = "map(...) has no turning-bands line process (its spectral measure is not known in closed form); use a parametric structure";
			return false;
	}
	if (reason) *reason = "unknown structure";
	return false;
}

// Throws Error naming the first structure turning bands cannot simulate.
inline void checkTurningBandsModel(const ResolvedModel& model) {
	if (model.ndim < 1 || model.ndim > 3) throw Error("turning bands: 1D, 2D or 3D only, got " + std::to_string(model.ndim) + "D");
	for (size_t s = 0; s < model.structures.size(); ++s) {
		LineProcess process = LineProcess::Spectral;
		std::string reason;
		if (!turningBandsLineProcess(model.structures[s], process, &reason))
			throw Error("turning bands: structure " + std::to_string(s + 1) + " (" + structureName(model.structures[s].type) + ") is not supported: " + reason);
	}
}

// The stationary model of a set, checked; throws Error otherwise.
inline const ResolvedModel& turningBandsModel(const CovarianceModelSet& models) {
	if (!models.isStationary())
		throw Error("turning bands simulates stationary models only (the model varies in space through a -cvar image or -ii)");
	checkTurningBandsModel(models.stationaryModel());
	return models.stationaryModel();
}

struct TurningBandsSettings {
	unsigned lineCount = 1000; // lines per structure
	uint64_t seed = 0;
	unsigned threads = 1;      // simulateTurningBands only; results do not depend on it
	double mean = 0.0;
};

inline void checkTurningBandsSettings(const TurningBandsSettings& settings) {
	if (settings.lineCount < 1) throw Error("turning bands: the number of lines must be >= 1");
	if (settings.lineCount >= (1u << 24)) throw Error("turning bands: too many lines (max 16777215)");
}

// ============================================================ plain-array descriptions

// The lines of one structure in one realisation: everything a back end needs
// to evaluate the sum over lines. All arrays are plain, contiguous, indexed by
// the line l < count. Line l maps a point x (ndim components) to
//     t_l(x) = sum_d direction[d * count + l] x_d + shift[l]
// (direction: ndim x count doubles, structure of arrays; it contains
// A^T u_l, times the frequency in turns per unit for the spectral process;
// shift: count doubles, the phase in turns in [0, 1) or the segment origin
// in [0, 1)), and contributes
//     spectral: amplitude * cos(2 pi t_l)
//     dilution: amplitude * e_l(k) * g(t_l - k - 1/2),  k = floor(t_l),
//               e_l(k) = top bit of mix32(uint32(k) * signKey[l] + signKey[count + l]) ? -1 : +1
// (signKey: 2 x count uint32, dilution only; signKey[l] is odd).
struct LineSet {
	LineProcess process = LineProcess::Spectral;
	unsigned ndim = 0;
	size_t count = 0;
	double amplitude = 0.0;
	const double* direction = nullptr;  // ndim x count
	const double* shift = nullptr;      // count
	const uint32_t* signKey = nullptr;  // 2 x count (dilution only)
};

// A regular grid: node (i0, i1, i2), i_d < size[d], is at
// x_d = origin[d] + spacing[d] i_d; unused axes have size 1. Node id
// (row-major, axis 0 fastest, the DataImage order) = i0 + size[0] * row,
// row = i1 + size[1] i2.
struct RegularGrid {
	unsigned ndim = 0;
	size_t size[3] = { 1, 1, 1 };
	double origin[3] = { 0, 0, 0 };
	double spacing[3] = { 1, 1, 1 };

	size_t rowCount() const { return size[1] * size[2]; }
	size_t nodeCount() const { return size[0] * size[1] * size[2]; }
	// Nodes at the cell indices (origin 0, spacing 1), as tb uses them.
	static RegularGrid cells(unsigned ndim, const unsigned* dims) {
		RegularGrid grid;
		grid.ndim = ndim;
		for (unsigned d = 0; d < ndim && d < 3; ++d) grid.size[d] = dims[d];
		return grid;
	}
};

// Grid path: values depend on the column tile (resynchronisation of the
// dilution running sums), so blocks never straddle a tile boundary.
constexpr size_t kGridColumnTile = 512;

// A block of grid nodes: rows rowBegin .. rowEnd - 1, columns (i0)
// columnBegin .. columnEnd - 1, inside one column tile
// (columnBegin / kGridColumnTile == (columnEnd - 1) / kGridColumnTile).
// Block values are stored row-major: out[(row - rowBegin) * width() + (i0 - columnBegin)].
struct GridBlock {
	size_t rowBegin = 0, rowEnd = 0, columnBegin = 0, columnEnd = 0;
	size_t rows() const { return rowEnd - rowBegin; }
	size_t width() const { return columnEnd - columnBegin; }
	size_t count() const { return rows() * width(); }
};

// The lines of one LineSet on one grid (CPU lookup tables, built by
// buildGridLines): t_l(node) = shift[l] + sum_d step[d * padded + l] i_d,
// padded = count rounded up to simd::kLanes (padding lines are zero). For the
// spectral process, shift is reduced to [-1/2, 1/2] turns and
// columnCos / columnSin[l * stride + i0] = cos / sin(2 pi step[0][l] i0) for
// i0 < stride (stride >= size[0] + 64).
struct GridLines {
	size_t count = 0, padded = 0, stride = 0;
	std::vector<double> shift, step;
	std::vector<float> columnCos, columnSin;
};

// Per-thread workspace of sumLinesGrid (grows on first use, then reused).
struct GridScratch {
	std::vector<float> rowCos, rowSin, partial, noise;
	std::vector<double> accumulator, lineStart, coefficients, columns, values;
	template <class T> static T* reserve(std::vector<T>& v, size_t n) { if (v.size() < n) v.resize(n); return v.data(); }
};

// ============================================================ scalar helpers

constexpr double kTwoPi = 6.28318530717958647693;

namespace detail {
// sin(2 pi a) ~ a (c0 + c1 a^2 + ... + c4 a^8) on |a| <= 1/4 (minimax-like fit,
// max error 3.4e-9).
constexpr float kSinTurns0 = 6.283185158278849f, kSinTurns1 = -41.34165469233216f, kSinTurns2 = 81.60098593229426f,
                kSinTurns3 = -76.54941360513733f, kSinTurns4 = 39.53417324387686f;
constexpr float kSphericalFactor = 3.46410161513775459f;   // 2 sqrt(3)
constexpr float kCubicFactor = 7.24568837309471540f;       // sqrt(105 / 2)
constexpr size_t kLineTile = 64;    // lines added in float before the double accumulator
constexpr size_t kPointBlock = 256; // meshless points per block (a multiple of every kLanes)
constexpr double kDirectSlope = 0.25; // grid dilution: lines steeper than this (segments per node) are evaluated directly
} // namespace detail

// cos(2 pi y) in float, |error| < 2.5e-7 for any double y (|y| < 2^51).
inline float turningBandsCosTurns(double y) {
	y -= std::nearbyint(y);
	const float a = float(0.25 - std::fabs(y));
	const float x = a * a;
	using namespace detail;
	return a * ((((kSinTurns4 * x + kSinTurns3) * x + kSinTurns2) * x + kSinTurns1) * x + kSinTurns0);
}

// MurmurHash3 32-bit finaliser.
inline uint32_t mixBits32(uint32_t h) {
	h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
	return h;
}

// Sign e_l(k) of segment k of a dilution line with key (multiplier, offset).
inline double dilutionSign(uint32_t multiplier, uint32_t offset, int64_t k) {
	return (mixBits32(uint32_t(uint64_t(k)) * multiplier + offset) >> 31) ? -1.0 : 1.0;
}

// ============================================================ vector building blocks

namespace detail {

// cos(2 pi y) from a = 1/4 - |y| (|y| <= 1/2), i.e. sin(2 pi a).
inline simd::F sinTurnsQuarter(simd::F a) {
	using namespace simd;
	const F x = fMul(a, a);
	F p = fMulAdd(fSet(kSinTurns4), x, fSet(kSinTurns3));
	p = fMulAdd(p, x, fSet(kSinTurns2));
	p = fMulAdd(p, x, fSet(kSinTurns1));
	p = fMulAdd(p, x, fSet(kSinTurns0));
	return fMul(a, p);
}

// cos(2 pi t), t in turns (double), as float.
inline simd::F cosTurns(simd::D t) {
	using namespace simd;
	const D y = dSub(t, dRound(t));
	return sinTurnsQuarter(dToF(dSub(dSet(0.25), dAbs(y))));
}

// sin(2 pi t) = cos(2 pi (t - 1/4)).
inline simd::F sinTurns(simd::D t) { return cosTurns(simd::dSub(t, simd::dSet(0.25))); }

// e_l(k) g(t - k - 1/2), k = floor(t), for one dilution line.
template <bool CUBIC>
inline simd::F dilutionValue(simd::D t, simd::U multiplier, simd::U offset) {
	using namespace simd;
	const D k = dFloor(t);
	const F s = dToF(dSub(dSub(t, k), dSet(0.5)));
	U h = uAdd(uMul(dLow32(k), multiplier), offset);
	h = uXor(h, uShr<16>(h));
	h = uMul(h, uSet(0x85ebca6bu));
	h = uXor(h, uShr<13>(h));
	h = uMul(h, uSet(0xc2b2ae35u));
	h = uXor(h, uShr<16>(h));
	const U sign = uAnd(h, uSet(0x80000000u));
	const F g = CUBIC ? fMul(s, fNegMulAdd(fMul(s, s), fSet(4.0f * kCubicFactor), fSet(kCubicFactor))) : fMul(s, fSet(kSphericalFactor));
	return fXorBits(g, sign);
}

// acc[i] = sum over the lines of their (unit-amplitude) value at the points
// x[d][i], i < n (n a multiple of kLanes). Points are the outer loop: the
// coordinates and the accumulator stay in registers while the lines stream.
template <unsigned NDIM, LineProcess P>
inline void meshlessBlock(const LineSet& lines, const double* const* x, size_t n, double* acc) {
	using namespace simd;
	const size_t L = lines.count;
	const double* w0 = lines.direction;
	const double* w1 = lines.direction + (NDIM > 1 ? L : 0);
	const double* w2 = lines.direction + (NDIM > 2 ? 2 * L : 0);
	const double* shift = lines.shift;
	const uint32_t* key = lines.signKey;
	for (size_t i = 0; i < n; i += kLanes) {
		const D x0 = dLoad(x[0] + i);
		const D x1 = NDIM > 1 ? dLoad(x[1] + i) : x0;
		const D x2 = NDIM > 2 ? dLoad(x[2] + i) : x0;
		D total = dSet(0.0);
		for (size_t l0 = 0; l0 < L; l0 += kLineTile) {
			const size_t l1 = std::min(L, l0 + kLineTile);
			F sum = fZero();
			for (size_t l = l0; l < l1; ++l) {
				D t = dMulAdd(dSet(w0[l]), x0, dSet(shift[l]));
				if (NDIM > 1) t = dMulAdd(dSet(w1[l]), x1, t);
				if (NDIM > 2) t = dMulAdd(dSet(w2[l]), x2, t);
				if (P == LineProcess::Spectral) sum = fAdd(sum, cosTurns(t));
				else sum = fAdd(sum, dilutionValue<P == LineProcess::DilutionCubic>(t, uSet(key[l]), uSet(key[L + l])));
			}
			total = dAdd(total, fToD(sum));
		}
		dStore(acc + i, total);
	}
}

template <unsigned NDIM>
inline void meshlessDispatch(const LineSet& lines, const double* const* x, size_t n, double* acc) {
	switch (lines.process) {
		case LineProcess::Spectral: meshlessBlock<NDIM, LineProcess::Spectral>(lines, x, n, acc); break;
		case LineProcess::DilutionSpherical: meshlessBlock<NDIM, LineProcess::DilutionSpherical>(lines, x, n, acc); break;
		case LineProcess::DilutionCubic: meshlessBlock<NDIM, LineProcess::DilutionCubic>(lines, x, n, acc); break;
	}
}

} // namespace detail

// ============================================================ meshless kernel

// out[i] += sum over the lines of their value at points[i] (row-major
// count x lines.ndim). Single-threaded; any count; a point's value does not
// depend on the other points of the call (nor on its position in the array).
inline void sumLines(const LineSet& lines, const double* points, size_t count, double* out) {
	using detail::kPointBlock;
	const unsigned ndim = lines.ndim;
	alignas(64) double buffer[3][kPointBlock];
	alignas(64) double accumulator[kPointBlock];
	const double* coordinates[3] = { buffer[0], buffer[1], buffer[2] };
	for (size_t begin = 0; begin < count; begin += kPointBlock) {
		const size_t n = std::min(kPointBlock, count - begin);
		const size_t padded = (n + simd::kLanes - 1) / simd::kLanes * simd::kLanes;
		for (size_t i = 0; i < n; ++i)
			for (unsigned d = 0; d < ndim; ++d) buffer[d][i] = points[(begin + i) * ndim + d];
		for (size_t i = n; i < padded; ++i)
			for (unsigned d = 0; d < ndim; ++d) buffer[d][i] = 0.0;
		if (ndim == 1) detail::meshlessDispatch<1>(lines, coordinates, padded, accumulator);
		else if (ndim == 2) detail::meshlessDispatch<2>(lines, coordinates, padded, accumulator);
		else detail::meshlessDispatch<3>(lines, coordinates, padded, accumulator);
		for (size_t i = 0; i < n; ++i) out[begin + i] += lines.amplitude * accumulator[i];
	}
}

// ============================================================ grid kernel

// Builds the CPU tables of a line set on a grid (see GridLines).
inline void buildGridLines(const LineSet& lines, const RegularGrid& grid, GridLines& table) {
	using namespace simd;
	const size_t L = lines.count, P = (L + kLanes - 1) / kLanes * kLanes;
	table.count = L;
	table.padded = P;
	table.shift.assign(P, 0.0);
	table.step.assign(3 * P, 0.0);
	const bool spectral = lines.process == LineProcess::Spectral;
	for (size_t l = 0; l < L; ++l) {
		double shift = lines.shift[l];
		for (unsigned d = 0; d < lines.ndim; ++d) {
			const double w = lines.direction[d * L + l];
			shift += w * grid.origin[d];
			table.step[d * P + l] = w * grid.spacing[d];
		}
		table.shift[l] = spectral ? shift - std::nearbyint(shift) : shift;
	}
	table.columnCos.clear();
	table.columnSin.clear();
	table.stride = 0;
	if (!spectral) return;
	table.stride = (grid.size[0] + 63) / 64 * 64 + 64;
	const size_t stride = table.stride;
	table.columnCos.resize(L * stride);
	table.columnSin.resize(L * stride);
	std::vector<double> columns(stride);
	for (size_t i = 0; i < stride; ++i) columns[i] = double(i);
	for (size_t l = 0; l < L; ++l) {
		const D w = dSet(table.step[l]);
		float* c = table.columnCos.data() + l * stride;
		float* s = table.columnSin.data() + l * stride;
		for (size_t i = 0; i < stride; i += kLanes) {
			const D t = dMul(w, dLoad(columns.data() + i));
			fStore(c + i, detail::cosTurns(t));
			fStore(s + i, detail::sinTurns(t));
		}
	}
}

namespace detail {

#if G2S_GEOSTAT_SIMD_AVX2 || G2S_GEOSTAT_SIMD_SSE2
constexpr unsigned kGridVectors = 2; // 16 registers: 4 x 2 accumulators + 4 table vectors + 2 broadcasts
#else
constexpr unsigned kGridVectors = 4; // 32 registers (NEON, AVX-512): 4 x 4 accumulators + 8 table vectors + 2 broadcasts
#endif
constexpr size_t kGridChunk = kGridVectors * simd::kLanes; // columns per micro-kernel step
constexpr size_t kGridColumnGroup = 256;                   // columns whose table slice stays in L1 across row groups

// acc[r * accStride + x] += sum_{l in [l0, l1)} rowCos[r][l] columnCos[l][x] - rowSin[r][l] columnSin[l][x]
// for r < ROWS, x < width (a multiple of kGridChunk); the line sum is in float.
template <unsigned ROWS>
inline void gridSpectralRows(const float* columnCos, const float* columnSin, size_t stride, size_t l0, size_t l1, const float* rowCos,
                             const float* rowSin, size_t rowStride, size_t width, double* acc, size_t accStride) {
	using namespace simd;
	for (size_t x = 0; x < width; x += kGridChunk) {
		F a[ROWS][kGridVectors];
		for (unsigned r = 0; r < ROWS; ++r)
			for (unsigned j = 0; j < kGridVectors; ++j) a[r][j] = fZero();
		for (size_t l = l0; l < l1; ++l) {
			const float* ec = columnCos + l * stride + x;
			const float* es = columnSin + l * stride + x;
			F c[kGridVectors], s[kGridVectors];
			for (unsigned j = 0; j < kGridVectors; ++j) { c[j] = fLoad(ec + j * kLanes); s[j] = fLoad(es + j * kLanes); }
			for (unsigned r = 0; r < ROWS; ++r) {
				const F fc = fSet(rowCos[r * rowStride + l]), fs = fSet(rowSin[r * rowStride + l]);
				for (unsigned j = 0; j < kGridVectors; ++j) a[r][j] = fNegMulAdd(fs, s[j], fMulAdd(fc, c[j], a[r][j]));
			}
		}
		for (unsigned r = 0; r < ROWS; ++r)
			for (unsigned j = 0; j < kGridVectors; ++j) {
				double* target = acc + r * accStride + x + j * kLanes;
				dStore(target, dAdd(dLoad(target), fToD(a[r][j])));
			}
	}
}

inline void gridRowIndices(const RegularGrid& grid, size_t row, double& i1, double& i2) {
	i1 = double(row % grid.size[1]);
	i2 = double(row / grid.size[1]);
}

inline void sumLinesGridSpectral(const LineSet& lines, const GridLines& table, const RegularGrid& grid, const GridBlock& block, double* out, GridScratch& scratch) {
	using namespace simd;
	const size_t L = table.count, P = table.padded, rows = block.rows(), width = block.width();
	const size_t widthPadded = (width + kGridChunk - 1) / kGridChunk * kGridChunk;
	float* rowCos = GridScratch::reserve(scratch.rowCos, rows * P);
	float* rowSin = GridScratch::reserve(scratch.rowSin, rows * P);
	double* acc = GridScratch::reserve(scratch.accumulator, rows * widthPadded);
	std::fill(acc, acc + rows * widthPadded, 0.0);
	const double* step1 = table.step.data() + P;
	const double* step2 = table.step.data() + 2 * P;
	for (size_t r = 0; r < rows; ++r) {
		double i1, i2;
		gridRowIndices(grid, block.rowBegin + r, i1, i2);
		const D v1 = dSet(i1), v2 = dSet(i2);
		for (size_t l = 0; l < P; l += kLanes) {
			const D t = dMulAdd(dLoad(step2 + l), v2, dMulAdd(dLoad(step1 + l), v1, dLoad(table.shift.data() + l)));
			fStore(rowCos + r * P + l, cosTurns(t));
			fStore(rowSin + r * P + l, sinTurns(t));
		}
	}
	const float* columnCos = table.columnCos.data() + block.columnBegin;
	const float* columnSin = table.columnSin.data() + block.columnBegin;
	for (size_t x0 = 0; x0 < widthPadded; x0 += kGridColumnGroup) {
		const size_t groupWidth = std::min(kGridColumnGroup, widthPadded - x0);
		for (size_t l0 = 0; l0 < L; l0 += kLineTile) {
			const size_t l1 = std::min(L, l0 + kLineTile);
			size_t r = 0;
			for (; r + 4 <= rows; r += 4)
				gridSpectralRows<4>(columnCos + x0, columnSin + x0, table.stride, l0, l1, rowCos + r * P, rowSin + r * P, P, groupWidth, acc + r * widthPadded + x0, widthPadded);
			for (; r < rows; ++r)
				gridSpectralRows<1>(columnCos + x0, columnSin + x0, table.stride, l0, l1, rowCos + r * P, rowSin + r * P, P, groupWidth, acc + r * widthPadded + x0, widthPadded);
		}
	}
	for (size_t r = 0; r < rows; ++r)
		for (size_t x = 0; x < width; ++x) out[r * width + x] += lines.amplitude * acc[r * widthPadded + x];
}

// One dilution line along one row piece of n nodes, t(xi) = t0 + w xi
// (0 < |w| <= kDirectSlope, or w == 0). Adds to the difference arrays the
// coefficients of the profile (a polynomial in u = xi - centre) of the first
// segment at node 0 and, at the first node of every later segment, the change
// of coefficients. A node whose t lies within rounding of a segment boundary
// may be given to either segment (its s is then +-1/2 within rounding).
template <bool CUBIC>
inline void dilutionCrossings(double t0, double w, uint32_t multiplier, uint32_t offset, size_t n, double centre, double* c0, double* c1, double* c2, double* c3) {
	const double factor = CUBIC ? double(kCubicFactor) : double(kSphericalFactor);
	const double base = (t0 - 0.5) + w * centre; // s at u = 0 is base - k
	double a0, a1, a2 = 0, a3 = 0;               // coefficients of the current segment
	auto coefficients = [&](double k, double& b0, double& b1, double& b2, double& b3) {
		const double e = dilutionSign(multiplier, offset, int64_t(k)) * factor;
		const double beta = base - k;
		if (!CUBIC) {
			b0 = e * beta;
			b1 = e * w;
		} else {
			const double b = beta * beta;
			b0 = e * (beta - 4.0 * b * beta);
			b1 = e * w * (1.0 - 12.0 * b);
			b2 = -12.0 * e * beta * w * w;
			b3 = -4.0 * e * w * w * w;
		}
	};
	double k = std::floor(t0);
	coefficients(k, a0, a1, a2, a3);
	c0[0] += a0; c1[0] += a1;
	if (CUBIC) { c2[0] += a2; c3[0] += a3; }
	if (w == 0.0) return;
	const double inverse = 1.0 / w, direction = w > 0 ? 1.0 : -1.0, limit = double(n);
	size_t xi = 0;
	for (;;) {
		// First node of the next segment: t >= k + 1 (w > 0) or t < k (w < 0).
		const double crossing = ((w > 0 ? k + 1.0 : k) - t0) * inverse;
		const double first = w > 0 ? std::ceil(crossing) : std::floor(crossing) + 1.0;
		if (!(first < limit)) break;
		const size_t next = first > double(xi) ? size_t(first) : xi + 1;
		if (next >= n) break;
		k += direction;
		double b0, b1, b2 = 0, b3 = 0;
		coefficients(k, b0, b1, b2, b3);
		c0[next] += b0 - a0; c1[next] += b1 - a1;
		if (CUBIC) { c2[next] += b2 - a2; c3[next] += b3 - a3; }
		a0 = b0; a1 = b1; a2 = b2; a3 = b3;
		xi = next;
	}
}

template <bool CUBIC>
inline void sumLinesGridDilution(const LineSet& lines, const GridLines& table, const RegularGrid& grid, const GridBlock& block, double* out, GridScratch& scratch) {
	using namespace simd;
	const size_t L = table.count, P = table.padded, rows = block.rows(), width = block.width();
	const size_t widthPadded = (width + kLanes - 1) / kLanes * kLanes;
	const size_t tileBegin = block.columnBegin / kGridColumnTile * kGridColumnTile;
	const size_t tileEnd = std::min(grid.size[0], tileBegin + kGridColumnTile);
	// The polynomial variable is centred on the column tile and the running
	// sums start at the block's first column: blocks must start at a tile
	// (planGridBlocks) for the values not to depend on the blocks.
	const double centre = 0.5 * double(tileEnd - tileBegin - 1) - double(block.columnBegin - tileBegin);
	double* lineStart = GridScratch::reserve(scratch.lineStart, P);
	double* coefficients = GridScratch::reserve(scratch.coefficients, 4 * (width + 1));
	double* columns = GridScratch::reserve(scratch.columns, widthPadded);
	float* partial = GridScratch::reserve(scratch.partial, widthPadded);
	double* c0 = coefficients, *c1 = c0 + width + 1, *c2 = c1 + width + 1, *c3 = c2 + width + 1;
	for (size_t i = 0; i < widthPadded; ++i) columns[i] = double(i);
	const double* step0 = table.step.data();
	const double* step1 = table.step.data() + P;
	const double* step2 = table.step.data() + 2 * P;
	const uint32_t* key = lines.signKey;
	for (size_t r = 0; r < rows; ++r) {
		double i1, i2;
		gridRowIndices(grid, block.rowBegin + r, i1, i2);
		const D v0 = dSet(double(block.columnBegin)), v1 = dSet(i1), v2 = dSet(i2);
		for (size_t l = 0; l < P; l += kLanes)
			dStore(lineStart + l, dMulAdd(dLoad(step0 + l), v0, dMulAdd(dLoad(step2 + l), v2, dMulAdd(dLoad(step1 + l), v1, dLoad(table.shift.data() + l)))));
		std::fill(coefficients, coefficients + 4 * (width + 1), 0.0);
		std::fill(partial, partial + widthPadded, 0.0f);
		double* row = out + r * width;
		size_t direct = 0;
		for (size_t l = 0; l < L; ++l) {
			const double w = step0[l];
			if (std::fabs(w) <= kDirectSlope) {
				dilutionCrossings<CUBIC>(lineStart[l], w, key[l], key[L + l], width, centre, c0, c1, c2, c3);
				continue;
			}
			const D t0 = dSet(lineStart[l]), vw = dSet(w);
			const U multiplier = uSet(key[l]), offset = uSet(key[L + l]);
			for (size_t i = 0; i < widthPadded; i += kLanes)
				fStore(partial + i, fAdd(fLoad(partial + i), dilutionValue<CUBIC>(dMulAdd(vw, dLoad(columns + i), t0), multiplier, offset)));
			if (++direct % kLineTile == 0) {
				for (size_t i = 0; i < width; ++i) row[i] += lines.amplitude * double(partial[i]);
				std::fill(partial, partial + widthPadded, 0.0f);
			}
		}
		double a0 = 0, a1 = 0, a2 = 0, a3 = 0;
		for (size_t i = 0; i < width; ++i) {
			a0 += c0[i]; a1 += c1[i];
			if (CUBIC) { a2 += c2[i]; a3 += c3[i]; }
			const double u = double(i) - centre;
			const double value = CUBIC ? a0 + u * (a1 + u * (a2 + u * a3)) : a0 + u * a1;
			row[i] += lines.amplitude * (value + double(partial[i]));
		}
	}
}

} // namespace detail

// out[(row - rowBegin) * width + (i0 - columnBegin)] += sum over the lines of
// their value at the nodes of the block (layout: GridBlock). `table` comes
// from buildGridLines(lines, grid). Single-threaded; the value of a node does
// not depend on the block it is computed in, as long as the blocks start at
// column tiles (planGridBlocks).
inline void sumLinesGrid(const LineSet& lines, const GridLines& table, const RegularGrid& grid, const GridBlock& block, double* out, GridScratch& scratch) {
	switch (lines.process) {
		case LineProcess::Spectral: detail::sumLinesGridSpectral(lines, table, grid, block, out, scratch); break;
		case LineProcess::DilutionSpherical: detail::sumLinesGridDilution<false>(lines, table, grid, block, out, scratch); break;
		case LineProcess::DilutionCubic: detail::sumLinesGridDilution<true>(lines, table, grid, block, out, scratch); break;
	}
}

// Blocks covering the grid: column tiles of kGridColumnTile nodes, rows grouped
// so that a block has about 8192 nodes (4 to 64 rows). Depends on the grid
// only.
inline std::vector<GridBlock> planGridBlocks(const RegularGrid& grid) {
	std::vector<GridBlock> blocks;
	const size_t rows = grid.rowCount(), columns = grid.size[0];
	if (rows == 0 || columns == 0) return blocks;
	const size_t tileWidth = std::min(columns, kGridColumnTile);
	size_t rowsPerBlock = std::max<size_t>(4, std::min<size_t>(64, 8192 / tileWidth / 4 * 4));
	rowsPerBlock = std::min(rowsPerBlock, rows);
	for (size_t row = 0; row < rows; row += rowsPerBlock)
		for (size_t column = 0; column < columns; column += kGridColumnTile) {
			GridBlock block;
			block.rowBegin = row;
			block.rowEnd = std::min(rows, row + rowsPerBlock);
			block.columnBegin = column;
			block.columnEnd = std::min(columns, column + kGridColumnTile);
			blocks.push_back(block);
		}
	return blocks;
}

// ============================================================ random draws

namespace detail {

// Gamma(shape, 1), Marsaglia & Tsang (2000); attempts at index line*64 + a.
inline double gammaVariate(double shape, uint64_t seed, uint32_t realisation, size_t line, uint32_t stream) {
	double boost = 1.0;
	if (shape < 1.0) {
		boost = std::pow(uniform(seed, realisation, uint64_t(line) * 64 + 63, stream), 1.0 / shape);
		shape += 1.0;
	}
	const double d = shape - 1.0 / 3.0, c = 1.0 / std::sqrt(9.0 * d);
	for (unsigned attempt = 0; attempt < 63; ++attempt) {
		const uint64_t index = uint64_t(line) * 64 + attempt;
		const double x = standardNormal(seed, realisation, index, stream);
		const double u = uniform2(seed, realisation, index, stream);
		const double v = 1.0 + c * x;
		if (v <= 0.0) continue;
		const double v3 = v * v * v;
		if (std::log(u) < 0.5 * x * x + d - d * v3 + d * std::log(v3)) return d * v3 * boost;
	}
	return d * boost; // probability < 0.05^63
}

// log of a positive stable variable S with E exp(-s S) = exp(-s^beta), 0 < beta < 1
// (Kanter 1975; Chambers, Mallows & Stuck 1976).
inline double logPositiveStable(double beta, double uniformAngle, double uniformExponential) {
	const double pi = 3.14159265358979323846;
	const double v = pi * uniformAngle;
	const double e = -std::log(uniformExponential);
	return std::log(std::sin(beta * v)) - std::log(std::sin(v)) / beta + (1.0 - beta) / beta * (std::log(std::sin((1.0 - beta) * v)) - std::log(e));
}

// Random rotation (row-major dim x dim, dim = 2 or 3), uniform on SO(dim).
inline void randomRotation(unsigned dim, uint64_t seed, uint32_t realisation, size_t structure, double* R) {
	if (dim == 2) {
		const double angle = kTwoPi * uniform(seed, realisation, structure, 1);
		const double c = std::cos(angle), s = std::sin(angle);
		R[0] = c; R[1] = -s; R[2] = s; R[3] = c;
		return;
	}
	double q[4];
	standardNormalPair(seed, realisation, 2 * uint64_t(structure), 1, q[0], q[1]);
	standardNormalPair(seed, realisation, 2 * uint64_t(structure) + 1, 1, q[2], q[3]);
	const double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
	const double w = q[0] / norm, x = q[1] / norm, y = q[2] / norm, z = q[3] / norm;
	R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - w * z);     R[2] = 2 * (x * z + w * y);
	R[3] = 2 * (x * y + w * z);     R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - w * x);
	R[6] = 2 * (x * z - w * y);     R[7] = 2 * (y * z + w * x);     R[8] = 1 - 2 * (x * x + y * y);
}

// Direction l of the equidistributed set (before rotation), in dim = 1, 2 or 3.
inline void lineDirection(unsigned dim, size_t l, size_t L, double* u) {
	if (dim == 1) { u[0] = 1.0; return; }
	if (dim == 2) {
		const double angle = 3.14159265358979323846 * (double(l) + 0.5) / double(L);
		u[0] = std::cos(angle); u[1] = std::sin(angle);
		return;
	}
	const double golden = 0.61803398874989484820; // (sqrt(5) - 1) / 2
	const double z = (double(l) + 0.5) / double(L);
	double turns = double(l) * golden;
	turns -= std::floor(turns);
	const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
	u[0] = r * std::cos(kTwoPi * turns); u[1] = r * std::sin(kTwoPi * turns); u[2] = z;
}

// Runs f(block) for block < blockCount on `threads` new threads (kept for
// callers that need no pool; the engine uses WorkerPool).
template <class F>
inline void parallelBlocks(size_t blockCount, unsigned threads, F f) {
	if (threads <= 1 || blockCount <= 1) {
		for (size_t b = 0; b < blockCount; ++b) f(b);
		return;
	}
	std::atomic<size_t> next(0);
	std::vector<std::thread> pool;
	const unsigned count = unsigned(std::min<size_t>(threads, blockCount));
	for (unsigned t = 0; t < count; ++t)
		pool.emplace_back([&]() { for (size_t b = next++; b < blockCount; b = next++) f(b); });
	for (std::thread& thread : pool) thread.join();
}

} // namespace detail

// ============================================================ thread pool

// A persistent pool of size() - 1 worker threads plus the calling thread.
// run(count, f) calls f(index, worker) for every index < count (dynamic
// scheduling: each thread takes the next index when it is done, so slower
// cores simply take fewer) and returns when all are done; worker < size()
// identifies the thread (for per-thread scratch). The first exception thrown
// by f is rethrown by run. Not reentrant; use from one thread at a time.
class WorkerPool {
public:
	explicit WorkerPool(unsigned threads) : _size(std::max(1u, threads)) {
		for (unsigned t = 1; t < _size; ++t) _threads.emplace_back([this, t]() { work(t); });
	}
	~WorkerPool() {
		{
			std::lock_guard<std::mutex> lock(_mutex);
			_stop = true;
		}
		_wake.notify_all();
		for (std::thread& thread : _threads) thread.join();
	}
	WorkerPool(const WorkerPool&) = delete;
	WorkerPool& operator=(const WorkerPool&) = delete;

	unsigned size() const { return _size; }

	template <class F>
	void run(size_t count, F&& f) {
		if (count == 0) return;
		if (_size == 1 || count == 1) {
			for (size_t i = 0; i < count; ++i) f(i, 0u);
			return;
		}
		using Function = typename std::remove_reference<F>::type;
		{
			std::lock_guard<std::mutex> lock(_mutex);
			_invoke = [](void* context, size_t index, unsigned worker) { (*static_cast<Function*>(context))(index, worker); };
			_context = static_cast<void*>(&f);
			_count = count;
			_next.store(0);
			_busy = _size - 1;
			_error = nullptr;
			++_generation;
		}
		_wake.notify_all();
		drain(0);
		std::unique_lock<std::mutex> lock(_mutex);
		_done.wait(lock, [this]() { return _busy == 0; });
		if (_error) std::rethrow_exception(_error);
	}

private:
	void drain(unsigned worker) {
		for (size_t i = _next.fetch_add(1); i < _count; i = _next.fetch_add(1)) {
			try {
				_invoke(_context, i, worker);
			} catch (...) {
				std::lock_guard<std::mutex> lock(_mutex);
				if (!_error) _error = std::current_exception();
				_next.store(_count);
			}
		}
	}
	void work(unsigned worker) {
		uint64_t seen = 0;
		std::unique_lock<std::mutex> lock(_mutex);
		for (;;) {
			_wake.wait(lock, [&]() { return _stop || _generation != seen; });
			if (_stop) return;
			seen = _generation;
			lock.unlock();
			drain(worker);
			lock.lock();
			if (--_busy == 0) _done.notify_one();
		}
	}

	unsigned _size;
	std::vector<std::thread> _threads;
	std::mutex _mutex;
	std::condition_variable _wake, _done;
	bool _stop = false;
	uint64_t _generation = 0;
	unsigned _busy = 0;
	void (*_invoke)(void*, size_t, unsigned) = nullptr;
	void* _context = nullptr;
	size_t _count = 0;
	std::atomic<size_t> _next{ 0 };
	std::exception_ptr _error;
};

// ============================================================ one realisation

// The lines of every structure for one realisation.
class TurningBandsRealisation {
public:
	// boxMin / boxMax are accepted for compatibility and unused (the dilution
	// signs need no table).
	TurningBandsRealisation(const ResolvedModel& model, const TurningBandsSettings& settings, uint32_t realisation,
	                        const double* boxMin = nullptr, const double* boxMax = nullptr)
		: _ndim(model.ndim), _seed(settings.seed), _realisation(realisation), _mean(settings.mean) {
		(void)boxMin;
		(void)boxMax;
		checkTurningBandsModel(model);
		checkTurningBandsSettings(settings);
		const unsigned ndim = _ndim;
		const size_t L = settings.lineCount;
		for (size_t s = 0; s < model.structures.size(); ++s) {
			const ResolvedStructure& rs = model.structures[s];
			if (rs.type == StructureType::Nugget) { _nuggetDeviation = std::sqrt(_nuggetDeviation * _nuggetDeviation + rs.sill); continue; }
			if (!(rs.sill > 0)) continue;
			LineProcess process = LineProcess::Spectral;
			turningBandsLineProcess(rs, process);
			Structure st;
			st.process = process;
			st.structureIndex = s;
			st.amplitude = std::sqrt((process == LineProcess::Spectral ? 2.0 : 1.0) * rs.sill / double(L));
			st.direction.assign(size_t(ndim) * L, 0.0);
			st.shift.assign(L, 0.0);
			const unsigned dim = process == LineProcess::Spectral ? ndim : 3; // direction space
			double R[9];
			if (dim >= 2) detail::randomRotation(dim, _seed, realisation, s, R);
			const uint32_t base = 2 + 8 * uint32_t(s);
			if (process != LineProcess::Spectral) st.signKey.assign(2 * L, 0u);
			for (size_t l = 0; l < L; ++l) {
				double u0[3], u[3] = { 0, 0, 0 };
				detail::lineDirection(dim, l, L, u0);
				if (dim == 1) u[0] = u0[0];
				else for (unsigned i = 0; i < dim; ++i) { u[i] = 0; for (unsigned j = 0; j < dim; ++j) u[i] += R[i * dim + j] * u0[j]; }
				double frequency = 1.0;
				if (process == LineProcess::Spectral) {
					frequency = spectralModulus(rs, ndim, l, base) / kTwoPi; // turns per unit
					st.shift[l] = uniform(_seed, realisation, l, base);
				} else {
					st.shift[l] = uniform(_seed, realisation, l, base);
					const std::array<uint32_t, 4> bits = randomBlock(_seed, realisation, l, base + 4);
					st.signKey[l] = bits[0] | 1u;
					st.signKey[L + l] = bits[1];
				}
				// t = <frequency u, A x> = <frequency A^T u, x> (u truncated to ndim: embedding in R^3).
				for (unsigned d = 0; d < ndim; ++d) {
					double w = 0.0;
					for (unsigned k = 0; k < ndim; ++k) w += rs.transform[k * ndim + d] * u[k];
					st.direction[d * L + l] = frequency * w;
				}
			}
			_structures.push_back(std::move(st));
		}
	}

	unsigned ndim() const { return _ndim; }
	uint32_t realisation() const { return _realisation; }
	double nuggetDeviation() const { return _nuggetDeviation; }
	size_t lineSetCount() const { return _structures.size(); }

	// The plain-array view of the lines of the i-th simulated (non-nugget) structure.
	LineSet lineSet(size_t i) const {
		const Structure& st = _structures[i];
		LineSet set;
		set.process = st.process;
		set.ndim = _ndim;
		set.count = st.shift.size();
		set.amplitude = st.amplitude;
		set.direction = st.direction.data();
		set.shift = st.shift.data();
		set.signKey = st.signKey.empty() ? nullptr : st.signKey.data();
		return set;
	}

	// out[i] = value at points[i] (row-major count x ndim). nodeIds (may be
	// null, then the id of point i is firstId + i) key the nugget noise.
	// Single-threaded and const: call it from several threads on disjoint blocks.
	void evaluate(const double* points, size_t count, const uint64_t* nodeIds, uint64_t firstId, double* out) const {
		for (size_t i = 0; i < count; ++i) out[i] = 0.0;
		for (size_t s = 0; s < _structures.size(); ++s) sumLines(lineSet(s), points, count, out);
		if (!(_nuggetDeviation > 0)) {
			for (size_t i = 0; i < count; ++i) out[i] += _mean;
			return;
		}
		float noise[256];
		for (size_t begin = 0; begin < count; begin += 256) {
			const size_t n = std::min<size_t>(256, count - begin);
			if (nodeIds) for (size_t i = 0; i < n; ++i) noise[i] = fastNormalByIndex(_seed, _realisation, nodeIds[begin + i], 0);
			else fillFastNormalsByIndex(_seed, _realisation, firstId + begin, n, 0, noise);
			for (size_t i = 0; i < n; ++i) out[begin + i] = _mean + out[begin + i] + _nuggetDeviation * double(noise[i]);
		}
	}

	// Prepares the grid path (tables of every structure on `grid`, whose ndim
	// must be the model's).
	void attachGrid(const RegularGrid& grid) {
		if (grid.ndim != _ndim) throw Error("turning bands: the grid has " + std::to_string(grid.ndim) + " dimensions, the model " + std::to_string(_ndim));
		_grid = grid;
		_gridLines.resize(_structures.size());
		for (size_t s = 0; s < _structures.size(); ++s) buildGridLines(lineSet(s), grid, _gridLines[s]);
	}
	const GridLines& gridLines(size_t i) const { return _gridLines[i]; }

	// out (block layout, see GridBlock) = values at the nodes of the block;
	// node ids (i0 + size[0] * row) key the nugget. Needs attachGrid. Const and
	// single-threaded: call it from several threads with their own scratch.
	void evaluateGrid(const GridBlock& block, double* out, GridScratch& scratch) const {
		const size_t width = block.width(), count = block.count();
		std::fill(out, out + count, 0.0);
		for (size_t s = 0; s < _structures.size(); ++s) sumLinesGrid(lineSet(s), _gridLines[s], _grid, block, out, scratch);
		if (!(_nuggetDeviation > 0)) {
			for (size_t i = 0; i < count; ++i) out[i] += _mean;
			return;
		}
		float* noise = GridScratch::reserve(scratch.noise, width);
		for (size_t r = 0; r < block.rows(); ++r) {
			fillFastNormalsByIndex(_seed, _realisation, uint64_t(block.rowBegin + r) * _grid.size[0] + block.columnBegin, width, 0, noise);
			double* row = out + r * width;
			for (size_t i = 0; i < width; ++i) row[i] = _mean + row[i] + _nuggetDeviation * double(noise[i]);
		}
	}

	// Bytes of the grid tables (attachGrid) of a realisation of this model.
	static size_t gridTableBytes(const ResolvedModel& model, const TurningBandsSettings& settings, const RegularGrid& grid) {
		size_t spectral = 0;
		for (const ResolvedStructure& rs : model.structures) {
			LineProcess process = LineProcess::DilutionSpherical;
			if (rs.type != StructureType::Nugget && rs.sill > 0 && turningBandsLineProcess(rs, process) && process == LineProcess::Spectral) ++spectral;
		}
		return spectral * size_t(settings.lineCount) * ((grid.size[0] + 63) / 64 * 64 + 64) * 2 * sizeof(float);
	}

private:
	struct Structure {
		LineProcess process = LineProcess::Spectral;
		size_t structureIndex = 0;
		double amplitude = 0.0;
		std::vector<double> direction, shift;
		std::vector<uint32_t> signKey;
	};

	unsigned _ndim = 0;
	uint64_t _seed = 0;
	uint32_t _realisation = 0;
	double _mean = 0.0;
	double _nuggetDeviation = 0.0;
	std::vector<Structure> _structures;
	RegularGrid _grid;
	std::vector<GridLines> _gridLines;

	// |w| for line l (unit-practical-range space, see the header comment).
	double spectralModulus(const ResolvedStructure& rs, unsigned ndim, size_t l, uint32_t base) const {
		double n[4];
		standardNormalPair(_seed, _realisation, l, base + 1, n[0], n[1]);
		standardNormalPair(_seed, _realisation, l, base + 2, n[2], n[3]);
		double chi2 = 0.0;
		for (unsigned d = 0; d < ndim; ++d) chi2 += n[d] * n[d];
		const double chi = std::sqrt(chi2);
		const double k = rs.scaleToRange;
		double modulus = 0.0;
		switch (rs.type) {
			case StructureType::Gaussian: modulus = std::sqrt(6.0) * chi; break;
			case StructureType::Exponential: modulus = 3.0 * chi / std::fabs(n[3]); break;
			case StructureType::Matern:
				modulus = k * chi / std::sqrt(2.0 * detail::gammaVariate(rs.shape, _seed, _realisation, l, base + 3));
				break;
			case StructureType::Stable:
				if (rs.shape >= 2.0) {
					modulus = std::sqrt(6.0) * chi;
				} else {
					const double logS = detail::logPositiveStable(0.5 * rs.shape, uniform(_seed, _realisation, l, base + 3), uniform2(_seed, _realisation, l, base + 3));
					modulus = k * chi * std::exp(0.5 * (0.69314718055994530942 + logS));
				}
				break;
			default: break;
		}
		if (!(modulus < 1e9)) modulus = 1e9; // also catches NaN / inf
		return modulus;
	}
};

// ============================================================ batch entry points

namespace detail {

// Realisations prepared together: enough (block, realisation) tasks to keep
// every thread busy, at least one per thread (their setup runs in parallel),
// within a memory budget for the grid tables. Never changes the values.
inline unsigned realisationBatch(unsigned realisationCount, unsigned threads, size_t blocks, size_t bytesPerRealisation) {
	size_t batch = std::max<size_t>(threads, (4 * size_t(threads) + blocks - 1) / std::max<size_t>(blocks, 1));
	if (bytesPerRealisation > 0) batch = std::min(batch, std::max<size_t>(1, (size_t(1) << 29) / bytesPerRealisation));
	return unsigned(std::max<size_t>(1, std::min<size_t>(batch, realisationCount)));
}

} // namespace detail

// Fills realisations firstRealisation .. firstRealisation + realisationCount - 1
// at `count` points (row-major count x model.ndim) into out[r * count + i]
// (r relative to firstRealisation). nodeIds (may be null: point i has id i)
// key the nugget noise. Uses settings.threads; the result does not depend on
// it. Throws Error for an unsupported model.
inline void simulateTurningBands(const ResolvedModel& model, const TurningBandsSettings& settings, const double* points, size_t count,
                                 uint32_t firstRealisation, unsigned realisationCount, double* out, const uint64_t* nodeIds = nullptr) {
	checkTurningBandsModel(model);
	checkTurningBandsSettings(settings);
	const unsigned ndim = model.ndim;
	const size_t blockSize = 4096;
	const size_t blocks = (count + blockSize - 1) / blockSize;
	if (blocks == 0 || realisationCount == 0) return;
	WorkerPool pool(settings.threads);
	const unsigned batch = detail::realisationBatch(realisationCount, pool.size(), blocks, 0);
	std::vector<std::unique_ptr<TurningBandsRealisation> > slots(batch);
	for (unsigned r0 = 0; r0 < realisationCount; r0 += batch) {
		const unsigned n = std::min(batch, realisationCount - r0);
		pool.run(n, [&](size_t i, unsigned) { slots[i].reset(new TurningBandsRealisation(model, settings, firstRealisation + r0 + unsigned(i))); });
		// Realisation-major: concurrent tasks mostly write different blocks.
		pool.run(size_t(n) * blocks, [&](size_t task, unsigned) {
			const size_t i = task / blocks, b = task % blocks;
			const size_t begin = b * blockSize, m = std::min(blockSize, count - begin);
			slots[i]->evaluate(points + begin * ndim, m, nodeIds ? nodeIds + begin : nullptr, begin, out + (r0 + i) * count + begin);
		});
	}
}

// Grid path: realisations firstRealisation .. + realisationCount - 1 on
// `grid` (grid.ndim == model.ndim), on `pool`. For every realisation r
// (relative) and every block of planGridBlocks(grid), calls
//     store(r, block, values)   values in block layout (GridBlock), valid during the call
// from the pool's threads (concurrently, for different (r, block)), and
// progress(done) after each batch of realisations (calling thread). Values
// depend neither on the pool size nor on the batching.
template <class Store, class Progress>
inline void simulateTurningBandsGrid(const ResolvedModel& model, const TurningBandsSettings& settings, const RegularGrid& grid, uint32_t firstRealisation,
                                     unsigned realisationCount, WorkerPool& pool, Store store, Progress progress) {
	checkTurningBandsModel(model);
	checkTurningBandsSettings(settings);
	if (grid.ndim != model.ndim) throw Error("turning bands: the grid has " + std::to_string(grid.ndim) + " dimensions, the model " + std::to_string(model.ndim));
	const std::vector<GridBlock> blocks = planGridBlocks(grid);
	if (blocks.empty() || realisationCount == 0) return;
	size_t largest = 0;
	for (const GridBlock& block : blocks) largest = std::max(largest, block.count());
	const unsigned batch = detail::realisationBatch(realisationCount, pool.size(), blocks.size(), TurningBandsRealisation::gridTableBytes(model, settings, grid));
	std::vector<std::unique_ptr<TurningBandsRealisation> > slots(batch);
	std::vector<GridScratch> scratch(pool.size());
	for (GridScratch& s : scratch) s.values.resize(largest);
	for (unsigned r0 = 0; r0 < realisationCount; r0 += batch) {
		const unsigned n = std::min(batch, realisationCount - r0);
		pool.run(n, [&](size_t i, unsigned) {
			slots[i].reset();
			slots[i].reset(new TurningBandsRealisation(model, settings, firstRealisation + r0 + unsigned(i)));
			slots[i]->attachGrid(grid);
		});
		pool.run(size_t(n) * blocks.size(), [&](size_t task, unsigned worker) {
			const size_t i = task / blocks.size();
			const GridBlock& block = blocks[task % blocks.size()];
			GridScratch& own = scratch[worker];
			slots[i]->evaluateGrid(block, own.values.data(), own);
			store(r0 + unsigned(i), block, static_cast<const double*>(own.values.data()));
		});
		progress(r0 + n);
	}
}

// Grid path into out[r * grid.nodeCount() + node id], on settings.threads threads.
inline void simulateTurningBandsGrid(const ResolvedModel& model, const TurningBandsSettings& settings, const RegularGrid& grid, uint32_t firstRealisation,
                                     unsigned realisationCount, double* out) {
	WorkerPool pool(settings.threads);
	const size_t nodes = grid.nodeCount(), columns = grid.size[0];
	simulateTurningBandsGrid(model, settings, grid, firstRealisation, realisationCount, pool,
		[&](unsigned r, const GridBlock& block, const double* values) {
			for (size_t row = block.rowBegin; row < block.rowEnd; ++row)
				std::copy(values + (row - block.rowBegin) * block.width(), values + (row - block.rowBegin + 1) * block.width(),
				          out + size_t(r) * nodes + row * columns + block.columnBegin);
		},
		[](unsigned) {});
}

} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_TURNING_BANDS_HPP
