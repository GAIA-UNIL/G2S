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
// Geosciences 32:1615-1628). The engine is meshless: it works on an array of
// point coordinates (N x ndim doubles, row-major, in the same units and axis
// order as the covariance lags) and fills a caller-provided output. Nothing
// here knows about grids or DataImage (src/tb.cpp is the grid front end).
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
//   * nugget: independent noise. map(...): refused (no line process).
//
// Random numbers are counter-based (random.hpp): every draw is a function of
// (seed, realisation, index, stream), so results do not depend on the number
// of threads, on how the points are split into blocks, or on how realisations
// are batched. Streams: 0 nugget (fastNormalByIndex, index = node id), 1 rotations (index =
// structure), 2 + 8 s + k the k-th draw of structure s (index = line, or
// line * 64 + attempt, or (line << 40) + segment).
//
// ----------------------------------------------------------------- usage
//   const ResolvedModel& model = turningBandsModel(modelSet);   // throws Error
//   TurningBandsSettings settings; settings.seed = 7; settings.threads = 8;
//   simulateTurningBands(model, settings, points, count, 0, R, out);   // out[r * count + i]
// or, for blocks of points generated on the fly (grid front end):
//   TurningBandsRealisation realisation(model, settings, r, boxMin, boxMax);
//   realisation.evaluate(blockPoints, blockCount, nullptr, firstNodeId, blockOut);
// The innermost loop, sumLines(LineSet, points, count, out), has a plain array
// interface (LineSet) so that other back ends (GPU) can replace it.

#ifndef G2S_GEOSTAT_TURNING_BANDS_HPP
#define G2S_GEOSTAT_TURNING_BANDS_HPP

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "geostat/covarianceModel.hpp"
#include "geostat/random.hpp"

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

// ============================================================ inner loop

// Plain-array description of the lines of one structure in one realisation:
// everything a back end needs to evaluate the sum over lines. Line l maps a
// point x (ndim components) to
//     t_l(x) = sum_d direction[d * count + l] x_d + shift[l]
// (direction already contains A^T u_l, times the frequency for the spectral
// process; shift is the phase or the segment origin), and contributes
//     spectral: amplitude * cos(t_l)
//     dilution: amplitude * e(l, k) * g(t_l - k - 1/2), k = floor(t_l),
// where e(l, k) = signs[tableOffset[l] + k - tableFirst[l]] when
// 0 <= k - tableFirst[l] < tableSize[l], else dilutionSign(seed,
// realisation, signStream, l, k) (identical values; the table covers the
// bounding box given at construction).
struct LineSet {
	LineProcess process = LineProcess::Spectral;
	unsigned ndim = 0;
	size_t count = 0;
	double amplitude = 0.0;
	const double* direction = nullptr; // ndim x count (structure of arrays)
	const double* shift = nullptr;     // count
	// dilution only
	const int64_t* tableFirst = nullptr;
	const uint32_t* tableSize = nullptr;
	const size_t* tableOffset = nullptr;
	const int8_t* signs = nullptr;
	uint64_t seed = 0;
	uint32_t realisation = 0;
	uint32_t signStream = 0;
};

constexpr double kTwoPi = 6.28318530717958647693;

// cos(x) to ~1e-11, branch-free so that the point loop vectorises: reduce to
// a whole number of turns, then cos(2 pi y) = sin(2 pi (1/4 - |y|)) with a
// degree-15 Taylor polynomial on [-pi/2, pi/2].
inline double turningBandsCos(double x) {
	double y = x * (1.0 / kTwoPi);
	y -= std::round(y);
	const double a = kTwoPi * (0.25 - std::fabs(y));
	const double a2 = a * a;
	return a * (1.0 + a2 * (-1.0 / 6 + a2 * (1.0 / 120 + a2 * (-1.0 / 5040 + a2 * (1.0 / 362880 + a2 * (-1.0 / 39916800 +
	       a2 * (1.0 / 6227020800.0 + a2 * (-1.0 / 1307674368000.0))))))));
}

// Sign of segment k of dilution line `line` (+1 or -1).
inline int dilutionSign(uint64_t seed, uint32_t realisation, uint32_t stream, size_t line, int64_t k) {
	const uint64_t segment = (uint64_t(k) + (uint64_t(1) << 39)) & ((uint64_t(1) << 40) - 1);
	return uniform(seed, realisation, (uint64_t(line) << 40) | segment, stream) < 0.5 ? -1 : 1;
}

namespace detail {

constexpr size_t kPointBlock = 256;

template <unsigned NDIM>
inline void spectralBlock(const LineSet& lines, const double* const* coordinates, size_t count, double* accumulator) {
	const size_t L = lines.count;
	for (size_t l = 0; l < L; ++l) {
		const double w0 = lines.direction[l];
		const double w1 = NDIM > 1 ? lines.direction[L + l] : 0.0;
		const double w2 = NDIM > 2 ? lines.direction[2 * L + l] : 0.0;
		const double phase = lines.shift[l];
		const double* x0 = coordinates[0];
		const double* x1 = coordinates[NDIM > 1 ? 1 : 0];
		const double* x2 = coordinates[NDIM > 2 ? 2 : 0];
		for (size_t i = 0; i < count; ++i) {
			double t = phase + w0 * x0[i];
			if (NDIM > 1) t += w1 * x1[i];
			if (NDIM > 2) t += w2 * x2[i];
			accumulator[i] += turningBandsCos(t);
		}
	}
}

template <unsigned NDIM>
inline void dilutionBlock(const LineSet& lines, const double* const* coordinates, size_t count, double* accumulator) {
	const size_t L = lines.count;
	const bool cubic = lines.process == LineProcess::DilutionCubic;
	const double sphericalFactor = 2.0 * std::sqrt(3.0), cubicFactor = std::sqrt(52.5);
	for (size_t l = 0; l < L; ++l) {
		const double w0 = lines.direction[l];
		const double w1 = NDIM > 1 ? lines.direction[L + l] : 0.0;
		const double w2 = NDIM > 2 ? lines.direction[2 * L + l] : 0.0;
		const double origin = lines.shift[l];
		const int64_t first = lines.tableFirst[l];
		const uint64_t size = lines.tableSize[l];
		const int8_t* signs = lines.signs + lines.tableOffset[l];
		const double* x0 = coordinates[0];
		const double* x1 = coordinates[NDIM > 1 ? 1 : 0];
		const double* x2 = coordinates[NDIM > 2 ? 2 : 0];
		for (size_t i = 0; i < count; ++i) {
			double t = origin + w0 * x0[i];
			if (NDIM > 1) t += w1 * x1[i];
			if (NDIM > 2) t += w2 * x2[i];
			const double k = std::floor(t);
			const double s = t - k - 0.5;
			const int64_t segment = int64_t(k);
			const uint64_t index = uint64_t(segment - first);
			const int sign = index < size ? signs[index] : dilutionSign(lines.seed, lines.realisation, lines.signStream, l, segment);
			const double g = cubic ? cubicFactor * s * (1.0 - 4.0 * s * s) : sphericalFactor * s;
			accumulator[i] += sign * g;
		}
	}
}

} // namespace detail

// out[i] += sum over the lines of their value at points[i] (row-major
// count x lines.ndim). Single-threaded; any count; for a given point the
// result does not depend on the other points of the call.
inline void sumLines(const LineSet& lines, const double* points, size_t count, double* out) {
	const unsigned ndim = lines.ndim;
	double buffer[3][detail::kPointBlock];
	double accumulator[detail::kPointBlock];
	const double* coordinates[3] = { buffer[0], buffer[1], buffer[2] };
	for (size_t begin = 0; begin < count; begin += detail::kPointBlock) {
		const size_t n = std::min(detail::kPointBlock, count - begin);
		for (size_t i = 0; i < n; ++i) {
			for (unsigned d = 0; d < ndim; ++d) buffer[d][i] = points[(begin + i) * ndim + d];
			accumulator[i] = 0.0;
		}
		if (lines.process == LineProcess::Spectral) {
			if (ndim == 1) detail::spectralBlock<1>(lines, coordinates, n, accumulator);
			else if (ndim == 2) detail::spectralBlock<2>(lines, coordinates, n, accumulator);
			else detail::spectralBlock<3>(lines, coordinates, n, accumulator);
		} else {
			if (ndim == 1) detail::dilutionBlock<1>(lines, coordinates, n, accumulator);
			else if (ndim == 2) detail::dilutionBlock<2>(lines, coordinates, n, accumulator);
			else detail::dilutionBlock<3>(lines, coordinates, n, accumulator);
		}
		for (size_t i = 0; i < n; ++i) out[begin + i] += lines.amplitude * accumulator[i];
	}
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

// Runs f(block) for block < blockCount on `threads` threads.
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

// ============================================================ one realisation

// The lines of every structure for one realisation. boxMin / boxMax (ndim
// values each, may be null) bound the points that will be evaluated; they
// only size the dilution sign tables (points outside still get the correct
// values, computed on the fly), never change the result.
class TurningBandsRealisation {
public:
	// Dilution tables larger than this (segments per line) are not stored.
	static constexpr uint64_t kMaxSegmentsPerLine = uint64_t(1) << 18;

	TurningBandsRealisation(const ResolvedModel& model, const TurningBandsSettings& settings, uint32_t realisation,
	                        const double* boxMin = nullptr, const double* boxMax = nullptr)
		: _ndim(model.ndim), _seed(settings.seed), _realisation(realisation), _mean(settings.mean) {
		checkTurningBandsModel(model);
		if (settings.lineCount < 1) throw Error("turning bands: the number of lines must be >= 1");
		if (settings.lineCount >= (1u << 24)) throw Error("turning bands: too many lines (max 16777215)");
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
			st.signStream = 2 + 8 * uint32_t(s) + 4;
			st.amplitude = std::sqrt((process == LineProcess::Spectral ? 2.0 : 1.0) * rs.sill / double(L));
			st.direction.assign(size_t(ndim) * L, 0.0);
			st.shift.assign(L, 0.0);
			const unsigned dim = process == LineProcess::Spectral ? ndim : 3; // direction space
			double R[9];
			if (dim >= 2) detail::randomRotation(dim, _seed, realisation, s, R);
			const uint32_t base = 2 + 8 * uint32_t(s);
			for (size_t l = 0; l < L; ++l) {
				double u0[3], u[3] = { 0, 0, 0 };
				detail::lineDirection(dim, l, L, u0);
				if (dim == 1) u[0] = u0[0];
				else for (unsigned i = 0; i < dim; ++i) { u[i] = 0; for (unsigned j = 0; j < dim; ++j) u[i] += R[i * dim + j] * u0[j]; }
				double frequency = 1.0;
				if (process == LineProcess::Spectral) {
					frequency = spectralModulus(rs, ndim, l, base);
					st.shift[l] = kTwoPi * uniform(_seed, realisation, l, base);
				} else {
					st.shift[l] = uniform(_seed, realisation, l, base);
				}
				// t = <frequency u, A x> = <frequency A^T u, x> (u truncated to ndim: embedding in R^3).
				for (unsigned d = 0; d < ndim; ++d) {
					double w = 0.0;
					for (unsigned k = 0; k < ndim; ++k) w += rs.transform[k * ndim + d] * u[k];
					st.direction[d * L + l] = frequency * w;
				}
			}
			if (process != LineProcess::Spectral) buildSignTables(st, L, boxMin, boxMax);
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
		set.tableFirst = st.tableFirst.data();
		set.tableSize = st.tableSize.data();
		set.tableOffset = st.tableOffset.data();
		set.signs = st.signs.data();
		set.seed = _seed;
		set.realisation = _realisation;
		set.signStream = st.signStream;
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

private:
	struct Structure {
		LineProcess process = LineProcess::Spectral;
		size_t structureIndex = 0;
		uint32_t signStream = 0;
		double amplitude = 0.0;
		std::vector<double> direction, shift;
		std::vector<int64_t> tableFirst;
		std::vector<uint32_t> tableSize;
		std::vector<size_t> tableOffset;
		std::vector<int8_t> signs;
	};

	unsigned _ndim = 0;
	uint64_t _seed = 0;
	uint32_t _realisation = 0;
	double _mean = 0.0;
	double _nuggetDeviation = 0.0;
	std::vector<Structure> _structures;

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

	void buildSignTables(Structure& st, size_t L, const double* boxMin, const double* boxMax) const {
		st.tableFirst.assign(L, 0);
		st.tableSize.assign(L, 0);
		st.tableOffset.assign(L, 0);
		if (!boxMin || !boxMax) return;
		size_t total = 0;
		for (size_t l = 0; l < L; ++l) {
			double lo = st.shift[l], hi = st.shift[l];
			for (unsigned d = 0; d < _ndim; ++d) {
				const double w = st.direction[d * L + l];
				lo += std::min(w * boxMin[d], w * boxMax[d]);
				hi += std::max(w * boxMin[d], w * boxMax[d]);
			}
			const double first = std::floor(lo), last = std::floor(hi);
			if (!(last - first + 1 <= double(kMaxSegmentsPerLine))) continue; // on the fly
			st.tableFirst[l] = int64_t(first);
			st.tableSize[l] = uint32_t(last - first + 1);
			st.tableOffset[l] = total;
			total += st.tableSize[l];
		}
		st.signs.resize(total);
		for (size_t l = 0; l < L; ++l)
			for (uint32_t k = 0; k < st.tableSize[l]; ++k)
				st.signs[st.tableOffset[l] + k] = int8_t(dilutionSign(_seed, _realisation, st.signStream, l, st.tableFirst[l] + int64_t(k)));
	}
};

// ============================================================ batch entry point

// Fills realisations firstRealisation .. firstRealisation + realisationCount - 1
// at `count` points (row-major count x model.ndim) into out[r * count + i]
// (r relative to firstRealisation). nodeIds (may be null: point i has id i)
// key the nugget noise. Uses settings.threads; the result does not depend on
// it. Throws Error for an unsupported model.
inline void simulateTurningBands(const ResolvedModel& model, const TurningBandsSettings& settings, const double* points, size_t count,
                                 uint32_t firstRealisation, unsigned realisationCount, double* out, const uint64_t* nodeIds = nullptr) {
	checkTurningBandsModel(model);
	const unsigned ndim = model.ndim;
	double boxMin[3] = { 0, 0, 0 }, boxMax[3] = { 0, 0, 0 };
	if (count > 0) {
		for (unsigned d = 0; d < ndim; ++d) boxMin[d] = boxMax[d] = points[d];
		for (size_t i = 1; i < count; ++i)
			for (unsigned d = 0; d < ndim; ++d) {
				boxMin[d] = std::min(boxMin[d], points[i * ndim + d]);
				boxMax[d] = std::max(boxMax[d], points[i * ndim + d]);
			}
	}
	const size_t blockSize = 4096;
	const size_t blocks = (count + blockSize - 1) / blockSize;
	for (unsigned r = 0; r < realisationCount; ++r) {
		const TurningBandsRealisation realisation(model, settings, firstRealisation + r, boxMin, boxMax);
		double* target = out + size_t(r) * count;
		detail::parallelBlocks(blocks, settings.threads, [&](size_t b) {
			const size_t begin = b * blockSize, n = std::min(blockSize, count - begin);
			realisation.evaluate(points + begin * ndim, n, nodeIds ? nodeIds + begin : nullptr, begin, target + begin);
		});
	}
}

} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_TURNING_BANDS_HPP
