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

// Dense Cholesky factorisation and the simple / ordinary kriging solver.
// Double precision, no BLAS/LAPACK. Every function is pure apart from the
// caller-provided workspace and output, so concurrent calls with distinct
// workspaces are thread-safe; reuse one workspace per thread to avoid
// allocations per node.
//
// Conventions
//   lags: row-major count x ndim, lag_i = x_i - x_0 (neighbour minus target),
//     in grid-cell units for the grid front end;
//   K_ij = C(lag_i - lag_j), k0_i = C(lag_i), c0 = C(0) (total sill);
//   estimate = sum_i w_i z_i + meanWeight * mean   (krigingEstimate below).
//
// Simple kriging (SK): K w = k0, variance = c0 - w.k0, meanWeight = 1 - sum w.
// Ordinary kriging (OK): [K 1; 1' 0][w; m] = [k0; 1], solved through the
//   Schur complement: K a = k0, K u = 1, m = (1'a - 1) / (1'u), w = a - m u;
//   variance = c0 - w.k0 - m, meanWeight = 0, lagrange = m.
// Variances are clamped at 0.
//
// Zero neighbours: weights empty, variance = c0, meanWeight = 1, status
//   KrigingNoNeighbor, for SK and for OK alike: OK without data has no
//   unbiasedness constraint to satisfy and falls back to simple kriging with
//   the mean the caller supplies (0 for a standard Gaussian simulation).
//
// Singular or non positive-definite K (duplicate or collinear neighbours,
// clipped covariance maps, Gaussian models at short lags): the factorisation
// rejects pivots <= 1e-12 * max(diag K); the solver then retries with a
// diagonal jitter of 1e-10, 1e-8, 1e-6, 1e-4 and 1e-2 times max(diag K),
// sets KrigingJittered and reports the jitter used. If even the last one
// fails, status is KrigingFailed, weights are 0, variance = c0 and
// meanWeight = 1 (the unconditional distribution). Nothing throws.

#ifndef G2S_GEOSTAT_KRIGING_HPP
#define G2S_GEOSTAT_KRIGING_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "geostat/covarianceModel.hpp"

namespace g2s {
namespace geostat {

// ============================================================ dense Cholesky

inline double dotProduct(const double* a, const double* b, size_t n) {
	double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
	size_t k = 0;
	for (; k + 4 <= n; k += 4) {
		s0 += a[k] * b[k];
		s1 += a[k + 1] * b[k + 1];
		s2 += a[k + 2] * b[k + 2];
		s3 += a[k + 3] * b[k + 3];
	}
	for (; k < n; ++k) s0 += a[k] * b[k];
	return (s0 + s1) + (s2 + s3);
}

// In-place Cholesky A = L L^T of a symmetric n x n row-major matrix with
// leading dimension lda (row i starts at a + i*lda). Only the lower triangle
// (diagonal included) is read; it is overwritten with L, the strict upper
// triangle is left untouched. Returns 0 on success, otherwise 1 + the row of
// the first pivot <= pivotTolerance (the matrix is then partly overwritten).
// Row-oriented (Cholesky-Banachiewicz): every inner product runs over two
// contiguous rows; O(n^3/3) flops, fine for a few thousand rows.
inline size_t choleskyFactor(double* a, size_t n, size_t lda, double pivotTolerance = 0.0) {
	for (size_t i = 0; i < n; ++i) {
		double* ai = a + i * lda;
		for (size_t j = 0; j < i; ++j) {
			const double* aj = a + j * lda;
			ai[j] = (ai[j] - dotProduct(ai, aj, j)) / aj[j];
		}
		const double pivot = ai[i] - dotProduct(ai, ai, i);
		if (!(pivot > pivotTolerance)) return i + 1;
		ai[i] = std::sqrt(pivot);
	}
	return 0;
}

// Solves L L^T x = b in place (b becomes x) with the factor of choleskyFactor.
inline void choleskySolve(const double* l, size_t n, size_t lda, double* b) {
	for (size_t i = 0; i < n; ++i) b[i] = (b[i] - dotProduct(l + i * lda, b, i)) / l[i * lda + i];
	for (size_t i = n; i-- > 0;) {
		const double* li = l + i * lda;
		b[i] /= li[i];
		const double x = b[i];
		for (size_t k = 0; k < i; ++k) b[k] -= li[k] * x;
	}
}

// Same for nrhs right-hand sides at once: b is row-major n x nrhs with leading dimension ldb.
inline void choleskySolve(const double* l, size_t n, size_t lda, double* b, size_t nrhs, size_t ldb) {
	for (size_t i = 0; i < n; ++i) {
		const double* li = l + i * lda;
		double* bi = b + i * ldb;
		for (size_t k = 0; k < i; ++k) {
			const double f = li[k];
			const double* bk = b + k * ldb;
			for (size_t r = 0; r < nrhs; ++r) bi[r] -= f * bk[r];
		}
		const double inverse = 1.0 / li[i];
		for (size_t r = 0; r < nrhs; ++r) bi[r] *= inverse;
	}
	for (size_t i = n; i-- > 0;) {
		const double* li = l + i * lda;
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

// Factorises a copy of the symmetric matrix `a` (lower triangle read) into
// `factor` (n x n, leading dimension n), adding the escalating jitter described
// at the top of this file when needed. Returns the absolute jitter added
// (0 when none was needed), or a negative value when every attempt failed.
inline double choleskyFactorWithJitter(const double* a, size_t n, size_t lda, std::vector<double>& factor) {
	factor.resize(n * n);
	double maxDiagonal = 0.0;
	for (size_t i = 0; i < n; ++i) maxDiagonal = std::max(maxDiagonal, a[i * lda + i]);
	if (!(maxDiagonal > 0) || !std::isfinite(maxDiagonal)) return -1.0;
	static const double relativeJitter[] = { 0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2 };
	for (double relative : relativeJitter) {
		const double jitter = relative * maxDiagonal;
		for (size_t i = 0; i < n; ++i) {
			const double* source = a + i * lda;
			double* target = factor.data() + i * n;
			for (size_t j = 0; j < i; ++j) target[j] = source[j];
			target[i] = source[i] + jitter;
		}
		if (choleskyFactor(factor.data(), n, n, 1e-12 * maxDiagonal) == 0) return jitter;
	}
	return -1.0;
}

// ============================================================ kriging

enum class KrigingType { Simple, Ordinary };

// Status flags (combined with |).
enum KrigingStatus : unsigned {
	KrigingOk = 0u,
	KrigingNoNeighbor = 1u, // count == 0: weights empty, variance = c0
	KrigingJittered = 2u,   // K needed a diagonal jitter (see `jitter`)
	KrigingFailed = 4u      // K could not be factorised: weights 0, variance = c0
};

struct KrigingSolution {
	std::vector<double> weights; // one per neighbour
	double variance = 0.0;       // kriging variance, >= 0
	double meanWeight = 0.0;     // weight of the mean: SK 1 - sum(w), OK 0 (1 without neighbours or on failure)
	double lagrange = 0.0;       // OK Lagrange multiplier m (K w + m 1 = k0); 0 for SK
	double jitter = 0.0;         // absolute value added to diag(K), 0 when not needed
	unsigned status = KrigingOk;
};

// Reusable buffers; one per thread.
struct KrigingWorkspace {
	std::vector<double> matrix, rhs, factor, a, u;
};

// sum_i w_i values_i + meanWeight * mean.
inline double krigingEstimate(const KrigingSolution& solution, const double* values, double mean = 0.0) {
	return dotProduct(solution.weights.data(), values, solution.weights.size()) + solution.meanWeight * mean;
}

// Solves an assembled system: matrix (count x count, leading dimension ld,
// lower triangle read), rhs = k0, c0 = C(0). Returns out.status.
inline unsigned solveKrigingSystem(KrigingType type, const double* matrix, size_t ld, const double* rhs, double c0, size_t count,
                                   KrigingWorkspace& workspace, KrigingSolution& out) {
	out.weights.assign(count, 0.0);
	out.variance = std::max(c0, 0.0);
	out.meanWeight = 1.0;
	out.lagrange = 0.0;
	out.jitter = 0.0;
	if (count == 0) { out.status = KrigingNoNeighbor; return out.status; }
	const double jitter = choleskyFactorWithJitter(matrix, count, ld, workspace.factor);
	if (jitter < 0) { out.status = KrigingFailed; return out.status; }
	out.status = jitter > 0 ? KrigingJittered : KrigingOk;
	out.jitter = jitter;
	const double* factor = workspace.factor.data();

	workspace.a.assign(rhs, rhs + count);
	choleskySolve(factor, count, count, workspace.a.data());
	if (type == KrigingType::Simple) {
		double sum = 0.0;
		for (size_t i = 0; i < count; ++i) { out.weights[i] = workspace.a[i]; sum += workspace.a[i]; }
		out.meanWeight = 1.0 - sum;
		out.variance = std::max(0.0, c0 - dotProduct(out.weights.data(), rhs, count));
		return out.status;
	}
	workspace.u.assign(count, 1.0);
	choleskySolve(factor, count, count, workspace.u.data());
	double sumA = 0.0, sumU = 0.0;
	for (size_t i = 0; i < count; ++i) { sumA += workspace.a[i]; sumU += workspace.u[i]; }
	if (!(sumU > 0) || !std::isfinite(sumU)) {
		out.weights.assign(count, 0.0);
		out.status |= KrigingFailed;
		return out.status;
	}
	const double m = (sumA - 1.0) / sumU;
	for (size_t i = 0; i < count; ++i) out.weights[i] = workspace.a[i] - m * workspace.u[i];
	out.lagrange = m;
	out.meanWeight = 0.0;
	out.variance = std::max(0.0, c0 - dotProduct(out.weights.data(), rhs, count) - m);
	return out.status;
}

// Kriging at a target from the lags of its neighbours with a resolved model.
inline unsigned krige(KrigingType type, const ResolvedModel& model, const double* lags, size_t count,
                      KrigingWorkspace& workspace, KrigingSolution& out) {
	workspace.matrix.resize(count * count);
	workspace.rhs.resize(count);
	model.covarianceMatrix(lags, count, workspace.matrix.data(), count, workspace.rhs.data());
	return solveKrigingSystem(type, workspace.matrix.data(), count, workspace.rhs.data(), model.sill(), count, workspace, out);
}

// Same with any covariance callable: double covariance(const double* lag),
// c0 = covariance(0), ndim <= kMaxDims.
template <class Covariance>
inline unsigned krige(KrigingType type, const Covariance& covariance, double c0, unsigned ndim, const double* lags, size_t count,
                      KrigingWorkspace& workspace, KrigingSolution& out) {
	workspace.matrix.resize(count * count);
	workspace.rhs.resize(count);
	double difference[kMaxDims];
	if (ndim > kMaxDims) {
		out.weights.assign(count, 0.0);
		out.variance = c0; out.meanWeight = 1.0; out.lagrange = 0.0; out.jitter = 0.0;
		out.status = KrigingFailed;
		return out.status;
	}
	for (size_t i = 0; i < count; ++i) {
		const double* li = lags + i * ndim;
		workspace.rhs[i] = covariance(li);
		for (size_t j = 0; j <= i; ++j) {
			const double* lj = lags + j * ndim;
			for (unsigned d = 0; d < ndim; ++d) difference[d] = li[d] - lj[d];
			const double c = covariance(static_cast<const double*>(difference));
			workspace.matrix[i * count + j] = c;
			workspace.matrix[j * count + i] = c;
		}
	}
	return solveKrigingSystem(type, workspace.matrix.data(), count, workspace.rhs.data(), c0, count, workspace, out);
}

} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_KRIGING_HPP
