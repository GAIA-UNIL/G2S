/*
 * G2S - geostat: shared foundation of the Gaussian geostatistics algorithms.
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

// Global (unique-neighbourhood) kriging in dual form, for a stationary model.
// Every datum takes part in every estimate. The data-to-data covariance K
// (n x n) is factorised once; afterwards each right-hand side (a vector of
// data values) costs two triangular solves, and each target one dot product.
//
//   SK, mean m:  alpha = K^-1 (z - m 1),            Z*(x) = m    + k(x).alpha
//   OK:          a = K^-1 z, u = K^-1 1,
//                beta = 1'a / 1'u (GLS mean),      Z*(x) = beta + k(x).alpha,
//                alpha = a - beta u = K^-1 (z - beta 1)
//   with k_i(x) = C(x_i - x) and K_ij = C(x_i - x_j) (kriging.hpp conventions).
//   This is the same estimate as solving the full SK / OK system with all data
//   at every target. The constant (m or beta) and alpha form a GlobalKrigingDual.
//
//   Variance (needs one forward solve per target, v = L^-1 k with K = L L^T):
//   SK: c0 - v.v      OK: c0 - v.v + (1 - w.v)^2 / (w.w), w = L^-1 1 (w.w = 1'u)
//   clamped at 0. Costs n^2 per target: compute it only when needed.
//
// Usage (one factorisation, many right-hand sides, e.g. conditioning R
// unconditional realisations by kriging of residuals):
//
//   using namespace g2s::geostat;
//   GlobalKriging kriging;
//   GlobalKriging::Options options;           // type Ordinary, 1 thread, 2 GiB limit
//   options.type = KrigingType::Simple;  options.nbThreads = 8;
//   std::string error;
//   if (!kriging.factorize(model, dataCoordinates, n, ndim, options, error)) ...  // n x ndim, row-major
//   // values: R vectors of n values, realisation-major (values[r*n + i]); means: R SK means or nullptr (0)
//   GlobalKrigingDual dual;
//   kriging.solve(values, R, n, nullptr, dual);
//   // Estimates at T points: out[r*T + t]
//   kriging.estimate(PointCrossCovariance(kriging, targetCoordinates), T, dual, out, T);
//   // On a regular grid: every cell of each realisation in one pass (fast path, see GridCovarianceTable)
//   GridCovarianceTable table;
//   if (table.build(model, dims, 8)) estimateGridFields(table, dataCells, dual, fields, cellCount, 8);
//   // Kriging variance (does not depend on the values): variance[t]
//   kriging.variance(PointCrossCovariance(kriging, targetCoordinates), T, variance);
//
// A CrossCovariance is any object with `void operator()(size_t target, double* k) const`
// filling k[i] = C(x_i - x_target) for the n data; PointCrossCovariance
// (coordinates), CellCrossCovariance (grid cells, model evaluated) and
// GridCrossCovariance (grid cells, table lookup) are provided.
//
// All member functions are const after factorize() and thread-safe; results
// do not depend on the number of threads. Memory: one n x n double matrix
// (8 n^2 bytes; factorize refuses beyond Options::maxMatrixBytes).
// The model is copied, but a map(...) structure points to the table owned by
// its CovarianceModel: keep the CovarianceModelSet alive while the engine is used.

#ifndef G2S_GEOSTAT_KRIGING_GLOBAL_HPP
#define G2S_GEOSTAT_KRIGING_GLOBAL_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "geostat/covarianceModel.hpp"
#include "geostat/denseSolver.hpp"
#include "geostat/kriging.hpp"

namespace g2s {
namespace geostat {

// Dual weights of nrhs right-hand sides: Z*_r(x) = constant[r] + sum_i k_i(x) alpha[i*nrhs + r].
struct GlobalKrigingDual {
	size_t count = 0;              // number of data
	size_t nrhs = 0;               // number of right-hand sides
	std::vector<double> alpha;     // count x nrhs, row-major
	std::vector<double> constant;  // nrhs: SK mean, or OK generalised-least-squares mean
};

class GlobalKriging {
public:
	struct Options {
		KrigingType type = KrigingType::Ordinary;
		unsigned nbThreads = 1;
		size_t maxMatrixBytes = size_t(2) << 30; // refuse when 8 n^2 exceeds this
	};

	static size_t matrixBytes(size_t count) { return count * count * sizeof(double); }

	// Builds and factorises K for the n data at `coordinates` (n x ndim,
	// row-major, the units of the model's lags; copied). Same jitter policy as
	// kriging.hpp: none, then 1e-10 ... 1e-2 times max(diag K). Returns false
	// with a message when n is 0, ndim is wrong, the matrix exceeds the memory
	// limit, or K cannot be factorised even with the largest jitter.
	bool factorize(const ResolvedModel& model, const double* coordinates, size_t count, unsigned ndim, const Options& options, std::string& error) {
		_count = 0;
		_factor.clear();
		if (count == 0) { error = "global kriging needs at least one datum"; return false; }
		if (ndim != model.ndim || ndim < 1 || ndim > kMaxDims) { error = "global kriging: the coordinates have " + std::to_string(ndim) + " components but the model has " + std::to_string(model.ndim); return false; }
		if (options.maxMatrixBytes > 0 && (count > (size_t(1) << 31) || matrixBytes(count) > options.maxMatrixBytes)) {
			error = "global kriging with " + std::to_string(count) + " data needs a " + std::to_string(count) + " x " + std::to_string(count) +
			        " matrix of " + formatBytes(double(count) * double(count) * 8.0) + ", above the limit of " + formatBytes(double(options.maxMatrixBytes)) +
			        "; use the moving neighbourhood (local) method or raise the limit";
			return false;
		}
		_model = model;
		_ndim = ndim;
		_type = options.type;
		_threads = std::max(1u, options.nbThreads);
		_coordinates.assign(coordinates, coordinates + count * ndim);
		_factor.resize(count * count);
		const double maxDiagonal = model.sill();
		if (!(maxDiagonal > 0) || !std::isfinite(maxDiagonal)) { error = "global kriging: the covariance at lag 0 must be positive"; return false; }
		static const double relativeJitter[] = { 0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2 };
		bool factorised = false;
		for (double relative : relativeJitter) {
			_jitter = relative * maxDiagonal;
			fillMatrix(_jitter);
			if (blockedCholesky(_factor.data(), count, count, 1e-12 * maxDiagonal, _threads) == 0) { factorised = true; break; }
		}
		if (!factorised) { _factor.clear(); error = "global kriging: the data covariance matrix is not positive definite, even with a diagonal jitter"; return false; }
		_count = count;
		_w.assign(count, 1.0);
		blockedForwardSubstitution(_factor.data(), count, count, _w.data(), 1, 1);
		_sumU = dotProduct(_w.data(), _w.data(), count); // 1'K^-1 1 = |L^-1 1|^2
		_u = _w;
		backSubstitution(_u.data());
		return true;
	}

	bool ready() const { return _count > 0; }
	size_t dataCount() const { return _count; }
	unsigned ndim() const { return _ndim; }
	KrigingType type() const { return _type; }
	double jitter() const { return _jitter; }           // absolute value added to diag(K), 0 when not needed
	const ResolvedModel& model() const { return _model; }
	const double* coordinates() const { return _coordinates.data(); } // dataCount x ndim
	unsigned threads() const { return _threads; }

	// Dual weights for nrhs right-hand sides. values: realisation-major, value i
	// of right-hand side r at values[r*ldValues + i] (ldValues >= dataCount).
	// means: nrhs SK means (nullptr = 0); ignored by OK.
	void solve(const double* values, size_t nrhs, size_t ldValues, const double* means, GlobalKrigingDual& out) const {
		const size_t n = _count;
		out.count = n;
		out.nrhs = nrhs;
		out.alpha.assign(n * nrhs, 0.0);
		out.constant.assign(nrhs, 0.0);
		if (n == 0 || nrhs == 0) return;
		const bool simple = _type == KrigingType::Simple;
		for (size_t r = 0; r < nrhs; ++r) {
			const double m = simple && means ? means[r] : 0.0;
			if (simple) out.constant[r] = m;
			for (size_t i = 0; i < n; ++i) out.alpha[i * nrhs + r] = values[r * ldValues + i] - m;
		}
		// Columns in independent chunks, solved in parallel.
		const size_t chunk = 32;
		const size_t chunks = (nrhs + chunk - 1) / chunk;
		parallelFor(chunks, _threads, [&](size_t c, unsigned) {
			const size_t c0 = c * chunk, w = std::min(chunk, nrhs - c0);
			if (nrhs == 1) choleskySolve(_factor.data(), n, n, out.alpha.data());
			else {
				blockedForwardSubstitution(_factor.data(), n, n, out.alpha.data() + c0, w, nrhs);
				backSubstitution(out.alpha.data() + c0, w, nrhs);
			}
		});
		if (simple) return;
		for (size_t r = 0; r < nrhs; ++r) {
			double sum = 0.0;
			for (size_t i = 0; i < n; ++i) sum += out.alpha[i * nrhs + r];
			const double beta = sum / _sumU;
			out.constant[r] = beta;
			for (size_t i = 0; i < n; ++i) out.alpha[i * nrhs + r] -= beta * _u[i];
		}
	}
	// One right-hand side.
	void solve(const double* values, double mean, GlobalKrigingDual& out) const { solve(values, 1, _count, &mean, out); }

	// Estimates at targetCount targets: out[r*ldOut + t] for every right-hand side r of `dual`.
	template <class CrossCovariance>
	void estimate(const CrossCovariance& cross, size_t targetCount, const GlobalKrigingDual& dual, double* out, size_t ldOut) const {
		const size_t n = _count, nrhs = dual.nrhs;
		const size_t block = 256;
		std::vector<std::vector<double> > k(_threads, std::vector<double>(n)), acc(_threads, std::vector<double>(nrhs));
		parallelFor((targetCount + block - 1) / block, _threads, [&](size_t b, unsigned thread) {
			double* kt = k[thread].data();
			double* sum = acc[thread].data();
			const size_t t1 = std::min(targetCount, (b + 1) * block);
			for (size_t t = b * block; t < t1; ++t) {
				cross(t, kt);
				if (nrhs == 1) { out[t] = dual.constant[0] + dotProduct(kt, dual.alpha.data(), n); continue; }
				for (size_t r = 0; r < nrhs; ++r) sum[r] = 0.0;
				for (size_t i = 0; i < n; ++i) {
					const double ki = kt[i];
					const double* ai = dual.alpha.data() + i * nrhs;
					for (size_t r = 0; r < nrhs; ++r) sum[r] += ki * ai[r];
				}
				for (size_t r = 0; r < nrhs; ++r) out[r * ldOut + t] = dual.constant[r] + sum[r];
			}
		});
	}

	// Kriging variance at targetCount targets (independent of the data values).
	template <class CrossCovariance>
	void variance(const CrossCovariance& cross, size_t targetCount, double* out) const {
		const size_t n = _count;
		const size_t block = 64; // targets per forward substitution
		const double c0 = _model.sill();
		const bool simple = _type == KrigingType::Simple;
		struct Scratch { std::vector<double> y, k, s, q; };
		std::vector<Scratch> scratch(_threads);
		parallelFor((targetCount + block - 1) / block, _threads, [&](size_t b, unsigned thread) {
			Scratch& w = scratch[thread];
			const size_t t0 = b * block, t1 = std::min(targetCount, t0 + block), m = t1 - t0;
			w.y.resize(n * m);
			w.k.resize(n);
			w.s.assign(m, 0.0);
			w.q.assign(m, 0.0);
			for (size_t t = 0; t < m; ++t) {
				cross(t0 + t, w.k.data());
				for (size_t i = 0; i < n; ++i) w.y[i * m + t] = w.k[i];
			}
			blockedForwardSubstitution(_factor.data(), n, n, w.y.data(), m, m);
			for (size_t i = 0; i < n; ++i) {
				const double* yi = w.y.data() + i * m;
				const double wi = _w[i];
				for (size_t t = 0; t < m; ++t) { w.s[t] += yi[t] * yi[t]; w.q[t] += wi * yi[t]; }
			}
			for (size_t t = 0; t < m; ++t) {
				double v = c0 - w.s[t];
				if (!simple) v += (1.0 - w.q[t]) * (1.0 - w.q[t]) / _sumU;
				out[t0 + t] = std::max(0.0, v);
			}
		});
	}

	// The factor L (n x n, row-major, lower triangle) for callers that need more.
	const double* factor() const { return _factor.data(); }

private:
	ResolvedModel _model;
	std::vector<double> _coordinates;
	size_t _count = 0;
	unsigned _ndim = 0;
	unsigned _threads = 1;
	KrigingType _type = KrigingType::Ordinary;
	std::vector<double> _factor; // L, lower triangle
	std::vector<double> _w;      // L^-1 1
	std::vector<double> _u;      // K^-1 1
	double _sumU = 0.0;          // 1'K^-1 1
	double _jitter = 0.0;

	static std::string formatBytes(double bytes) {
		char text[64];
		if (bytes >= double(size_t(1) << 30)) snprintf(text, sizeof(text), "%.2f GiB", bytes / double(size_t(1) << 30));
		else snprintf(text, sizeof(text), "%.1f MiB", bytes / double(size_t(1) << 20));
		return text;
	}

	void fillMatrix(double jitter) {
		const size_t n = _coordinates.size() / _ndim;
		const unsigned nd = _ndim;
		const size_t rowsPerBlock = 64;
		parallelFor((n + rowsPerBlock - 1) / rowsPerBlock, _threads, [&](size_t b, unsigned) {
			double lag[kMaxDims];
			const size_t i1 = std::min(n, (b + 1) * rowsPerBlock);
			for (size_t i = b * rowsPerBlock; i < i1; ++i) {
				const double* xi = _coordinates.data() + i * nd;
				double* row = _factor.data() + i * n;
				for (size_t j = 0; j <= i; ++j) {
					const double* xj = _coordinates.data() + j * nd;
					for (unsigned d = 0; d < nd; ++d) lag[d] = xi[d] - xj[d];
					row[j] = _model.covariance(lag);
				}
				row[i] += jitter;
			}
		});
	}

	// L^T x = b in place (b: one vector).
	void backSubstitution(double* b) const {
		const size_t n = _count == 0 ? _coordinates.size() / _ndim : _count;
		const double* l = _factor.data();
		for (size_t i = n; i-- > 0;) {
			const double* li = l + i * n;
			b[i] /= li[i];
			const double x = b[i];
			for (size_t k = 0; k < i; ++k) b[k] -= li[k] * x;
		}
	}
	// L^T X = B in place, B row-major n x nrhs with leading dimension ldb.
	void backSubstitution(double* b, size_t nrhs, size_t ldb) const {
		const size_t n = _count;
		const double* l = _factor.data();
		for (size_t i = n; i-- > 0;) {
			const double* li = l + i * n;
			double* bi = b + i * ldb;
			const double inverse = 1.0 / li[i];
			for (size_t r = 0; r < nrhs; ++r) bi[r] *= inverse;
			for (size_t k = 0; k < i; ++k) {
				const double f = li[k];
				double* bk = b + k * ldb;
				for (size_t r = 0; r < nrhs; ++r) bk[r] -= f * bi[r];
			}
		}
	}
};

// ============================================================ cross covariances

// Targets given by coordinates (targetCount x ndim, row-major, not copied).
class PointCrossCovariance {
public:
	PointCrossCovariance(const GlobalKriging& kriging, const double* targetCoordinates)
		: _model(kriging.model()), _data(kriging.coordinates()), _count(kriging.dataCount()), _ndim(kriging.ndim()), _targets(targetCoordinates) {}
	void operator()(size_t target, double* k) const {
		double lag[kMaxDims];
		const double* x = _targets + target * _ndim;
		for (size_t i = 0; i < _count; ++i) {
			const double* xi = _data + i * _ndim;
			for (unsigned d = 0; d < _ndim; ++d) lag[d] = xi[d] - x[d];
			k[i] = _model.covariance(lag);
		}
	}
private:
	const ResolvedModel& _model;
	const double* _data;
	size_t _count;
	unsigned _ndim;
	const double* _targets;
};

// Cell coordinates of a flat index on a grid (cell = x0 + d0*(x1 + d1*(x2 ...))).
inline void cellCoordinates(size_t cell, const std::vector<unsigned>& dims, long* coordinate) {
	for (size_t d = 0; d < dims.size(); ++d) { coordinate[d] = long(cell % dims[d]); cell /= dims[d]; }
}

// Targets given by grid cell ids; data positions are the kriging coordinates.
// Evaluates the model (use GridCrossCovariance when the table fits in memory).
class CellCrossCovariance {
public:
	CellCrossCovariance(const GlobalKriging& kriging, const std::vector<unsigned>& dims, const size_t* targetCells)
		: _model(kriging.model()), _data(kriging.coordinates()), _count(kriging.dataCount()), _ndim(kriging.ndim()), _dims(dims), _targets(targetCells) {}
	void operator()(size_t target, double* k) const {
		long c[kMaxDims];
		cellCoordinates(_targets[target], _dims, c);
		double lag[kMaxDims];
		for (size_t i = 0; i < _count; ++i) {
			const double* xi = _data + i * _ndim;
			for (unsigned d = 0; d < _ndim; ++d) lag[d] = xi[d] - double(c[d]);
			k[i] = _model.covariance(lag);
		}
	}
private:
	const ResolvedModel& _model;
	const double* _data;
	size_t _count;
	unsigned _ndim;
	std::vector<unsigned> _dims;
	const size_t* _targets;
};

// ============================================================ grid fast path

// Covariance of every cell offset of a grid: value(o) = C(-o) for the offset
// o = target cell - data cell, o_d in [-(dims_d - 1), dims_d - 1]; prod(2 dims - 1)
// doubles (32 MiB for 1000 x 1000). Lookups replace covariance evaluations,
// and the scatter in estimateGridFields streams contiguous rows of the table.
class GridCovarianceTable {
public:
	// Returns false (and stays empty) when the table would exceed maxBytes.
	bool build(const ResolvedModel& model, const std::vector<unsigned>& dims, unsigned nbThreads, size_t maxBytes = size_t(1) << 30) {
		_values.clear();
		if (dims.empty() || dims.size() != model.ndim || dims.size() > kMaxDims) return false;
		_dims = dims;
		double size = 1.0;
		_extent.resize(dims.size());
		_stride.resize(dims.size());
		for (size_t d = 0; d < dims.size(); ++d) {
			_extent[d] = 2 * size_t(dims[d]) - 1;
			_stride[d] = size_t(size);
			size *= double(_extent[d]);
		}
		if (size * sizeof(double) > double(maxBytes)) return false;
		const size_t total = size_t(size);
		_values.resize(total);
		const size_t rowLength = _extent[0];
		const unsigned nd = unsigned(dims.size());
		parallelFor(total / rowLength, nbThreads, [&](size_t row, unsigned) {
			double lag[kMaxDims];
			size_t rest = row;
			for (unsigned d = 1; d < nd; ++d) { lag[d] = -(double(rest % _extent[d]) - double(_dims[d] - 1)); rest /= _extent[d]; }
			double* out = _values.data() + row * rowLength;
			for (size_t x = 0; x < rowLength; ++x) {
				lag[0] = -(double(x) - double(_dims[0] - 1));
				out[x] = model.covariance(lag);
			}
		});
		return true;
	}
	bool ready() const { return !_values.empty(); }
	const std::vector<unsigned>& dims() const { return _dims; }
	const double* values() const { return _values.data(); }
	size_t stride(size_t d) const { return _stride[d]; }
	size_t bytes() const { return _values.size() * sizeof(double); }
	// Index of the zero offset minus the data cell: index(o) = base(data) + sum_d target_d stride_d.
	size_t base(size_t dataCell) const {
		long c[kMaxDims];
		cellCoordinates(dataCell, _dims, c);
		size_t index = 0;
		for (size_t d = 0; d < _dims.size(); ++d) index += size_t(long(_dims[d]) - 1 - c[d]) * _stride[d];
		return index;
	}
	size_t offset(size_t targetCell) const {
		long c[kMaxDims];
		cellCoordinates(targetCell, _dims, c);
		size_t index = 0;
		for (size_t d = 0; d < _dims.size(); ++d) index += size_t(c[d]) * _stride[d];
		return index;
	}
private:
	std::vector<unsigned> _dims;
	std::vector<size_t> _extent, _stride;
	std::vector<double> _values;
};

// k_i = C(x_i - x_target) by table lookup; dataCells must be the cells whose
// coordinates were given to GlobalKriging::factorize, in the same order.
class GridCrossCovariance {
public:
	GridCrossCovariance(const GridCovarianceTable& table, const size_t* dataCells, size_t dataCount, const size_t* targetCells)
		: _table(table), _targets(targetCells) {
		_base.resize(dataCount);
		for (size_t i = 0; i < dataCount; ++i) _base[i] = table.base(dataCells[i]);
	}
	void operator()(size_t target, double* k) const {
		const double* values = _table.values() + _table.offset(_targets[target]);
		const size_t n = _base.size();
		for (size_t i = 0; i < n; ++i) k[i] = values[_base[i]];
	}
private:
	const GridCovarianceTable& _table;
	std::vector<size_t> _base;
	const size_t* _targets;
};

// Estimates every cell of the grid for every right-hand side of `dual`:
// out[r*ldOut + cell] (ldOut >= cell count). dataCells: as for
// GridCrossCovariance. Scatter form: each datum adds alpha_i times a
// contiguous row of the table to each row of the output; parallel over
// row segments, n * cells * nrhs multiply-adds. Data cells receive their
// kriged value (the datum up to rounding and jitter).
inline void estimateGridFields(const GridCovarianceTable& table, const size_t* dataCells, const GlobalKrigingDual& dual, double* out, size_t ldOut, unsigned nbThreads) {
	const std::vector<unsigned>& dims = table.dims();
	const size_t n = dual.count, nrhs = dual.nrhs;
	size_t cellCount = 1;
	for (unsigned d : dims) cellCount *= d;
	const size_t rowLength = dims[0], rows = cellCount / rowLength;
	const size_t segment = 2048, segments = (rowLength + segment - 1) / segment;
	// Data sorted by cell (rows of the same table row become consecutive).
	std::vector<size_t> order(n), base(n);
	std::iota(order.begin(), order.end(), size_t(0));
	std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return dataCells[a] < dataCells[b]; });
	for (size_t i = 0; i < n; ++i) base[i] = table.base(dataCells[order[i]]);
	const double* values = table.values();
	parallelFor(rows * segments, nbThreads, [&](size_t item, unsigned) {
		const size_t row = item / segments, x0 = (item % segments) * segment, w = std::min(segment, rowLength - x0);
		const size_t firstCell = row * rowLength + x0;
		const size_t rowOffset = table.offset(firstCell);
		for (size_t r = 0; r < nrhs; ++r) {
			double* o = out + r * ldOut + firstCell;
			for (size_t x = 0; x < w; ++x) o[x] = dual.constant[r];
		}
		for (size_t s = 0; s < n; ++s) {
			const double* t = values + base[s] + rowOffset;
			const double* a = dual.alpha.data() + order[s] * nrhs;
			for (size_t r = 0; r < nrhs; ++r) {
				const double alpha = a[r];
				double* o = out + r * ldOut + firstCell;
				for (size_t x = 0; x < w; ++x) o[x] += alpha * t[x];
			}
		}
	});
}

} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_KRIGING_GLOBAL_HPP
