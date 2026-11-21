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

// Dense kernels for large kriging systems (thousands of data), multithreaded,
// double precision, no BLAS/LAPACK. kriging.hpp holds the small-system solver;
// this file adds what the global (unique-neighbourhood) engine needs:
//
//   parallelFor        static work sharing over std::thread with an atomic counter;
//   gemmSubtract       C -= A B (row-major), a register-blocked micro-kernel;
//   blockedCholesky    in-place right-looking blocked Cholesky, parallel over
//                      the panel solve and the trailing update;
//   blockedForwardSubstitution  L Y = B for many right-hand sides at once.
//
// Plugging in a BLAS/LAPACK later: blockedCholesky is LAPACK dpotrf('L') on a
// row-major matrix (= dpotrf('U') column-major), gemmSubtract is dgemm and
// blockedForwardSubstitution is dtrsm('L','L','N','N'). Each function is a
// single replacement point; callers do not depend on the implementation.
//
// Every result is independent of the number of threads: each output element
// is computed by one thread with a fixed operation order.

#ifndef G2S_GEOSTAT_DENSE_SOLVER_HPP
#define G2S_GEOSTAT_DENSE_SOLVER_HPP

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "geostat/kriging.hpp"

namespace g2s {
namespace geostat {

// Calls f(index, threadIndex) for index in [0, count), indices handed out in
// increasing order; threadIndex < nbThreads lets callers keep per-thread
// workspaces. Runs inline when nbThreads <= 1 or count <= 1.
template <class F>
inline void parallelFor(size_t count, unsigned nbThreads, F f) {
	if (count == 0) return;
	const unsigned threads = unsigned(std::min<size_t>(std::max(1u, nbThreads), count));
	if (threads <= 1) {
		for (size_t i = 0; i < count; ++i) f(i, 0u);
		return;
	}
	std::atomic<size_t> next(0);
	auto worker = [&](unsigned threadIndex) {
		for (size_t i = next.fetch_add(1); i < count; i = next.fetch_add(1)) f(i, threadIndex);
	};
	std::vector<std::thread> pool;
	pool.reserve(threads - 1);
	for (unsigned t = 1; t < threads; ++t) pool.emplace_back(worker, t);
	worker(0);
	for (std::thread& thread : pool) thread.join();
}

namespace detail {

// One full 4 x 8 tile: C[0..4, 0..8] -= A[0..4, 0..k] B[0..k, 0..8].
#if defined(__GNUC__) || defined(__clang__)
// GNU vector extensions (gcc, clang, icpx): 2-double vectors map to NEON / SSE2
// registers, 16 accumulators stay in registers. Same per-element operations as
// the portable loop below.
typedef double DoubleVector2 __attribute__((vector_size(16)));
inline DoubleVector2 loadVector(const double* p) { DoubleVector2 v; __builtin_memcpy(&v, p, sizeof(v)); return v; }
inline void storeVector(double* p, DoubleVector2 v) { __builtin_memcpy(p, &v, sizeof(v)); }
inline void gemmTile4x8(double* c, size_t ldc, const double* a0, const double* a1, const double* a2, const double* a3,
                        const double* b, size_t ldb, size_t k) {
	DoubleVector2 acc[4][4] = {};
	for (size_t p = 0; p < k; ++p) {
		const double* bp = b + p * ldb;
		const DoubleVector2 b0 = loadVector(bp), b1 = loadVector(bp + 2), b2 = loadVector(bp + 4), b3 = loadVector(bp + 6);
		const double v[4] = { a0[p], a1[p], a2[p], a3[p] };
		for (int r = 0; r < 4; ++r) {
			const DoubleVector2 a = { v[r], v[r] };
			acc[r][0] += a * b0;
			acc[r][1] += a * b1;
			acc[r][2] += a * b2;
			acc[r][3] += a * b3;
		}
	}
	for (int r = 0; r < 4; ++r) {
		double* cr = c + size_t(r) * ldc;
		for (int q = 0; q < 4; ++q) storeVector(cr + 2 * q, loadVector(cr + 2 * q) - acc[r][q]);
	}
}
#else
inline void gemmTile4x8(double* c, size_t ldc, const double* a0, const double* a1, const double* a2, const double* a3,
                        const double* b, size_t ldb, size_t k) {
	double acc[4][8] = {};
	const double* rows[4] = { a0, a1, a2, a3 };
	for (size_t p = 0; p < k; ++p) {
		const double* bp = b + p * ldb;
		for (int r = 0; r < 4; ++r) {
			const double v = rows[r][p];
			for (int j = 0; j < 8; ++j) acc[r][j] += v * bp[j];
		}
	}
	for (int r = 0; r < 4; ++r)
		for (int j = 0; j < 8; ++j) c[size_t(r) * ldc + j] -= acc[r][j];
}
#endif

} // namespace detail

// C (m x n, ldc) -= A (m x k, lda) * B (k x n, ldb), all row-major.
// 4 x 8 register tile: per step of k, 4 broadcasts of A and 8 loads of B feed 32 FMAs.
inline void gemmSubtract(double* c, size_t ldc, const double* a, size_t lda, const double* b, size_t ldb, size_t m, size_t n, size_t k) {
	constexpr size_t MR = 4, NR = 8;
	size_t i0 = 0;
	for (; i0 + MR <= m; i0 += MR) {
		const double* a0 = a + (i0 + 0) * lda;
		const double* a1 = a + (i0 + 1) * lda;
		const double* a2 = a + (i0 + 2) * lda;
		const double* a3 = a + (i0 + 3) * lda;
		size_t j0 = 0;
		for (; j0 + NR <= n; j0 += NR) {
			detail::gemmTile4x8(c + i0 * ldc + j0, ldc, a0, a1, a2, a3, b + j0, ldb, k);
		}
		if (j0 < n) {
			const size_t w = n - j0;
			double acc[MR][NR] = {};
			for (size_t p = 0; p < k; ++p) {
				const double* bp = b + p * ldb + j0;
				const double v[MR] = { a0[p], a1[p], a2[p], a3[p] };
				for (size_t r = 0; r < MR; ++r)
					for (size_t j = 0; j < w; ++j) acc[r][j] += v[r] * bp[j];
			}
			for (size_t r = 0; r < MR; ++r)
				for (size_t j = 0; j < w; ++j) c[(i0 + r) * ldc + j0 + j] -= acc[r][j];
		}
	}
	for (; i0 < m; ++i0) {
		const double* ai = a + i0 * lda;
		double* ci = c + i0 * ldc;
		for (size_t j0 = 0; j0 < n; j0 += NR) {
			const size_t w = std::min(NR, n - j0);
			double acc[NR] = {};
			for (size_t p = 0; p < k; ++p) {
				const double* bp = b + p * ldb + j0;
				const double v = ai[p];
				for (size_t j = 0; j < w; ++j) acc[j] += v * bp[j];
			}
			for (size_t j = 0; j < w; ++j) ci[j0 + j] -= acc[j];
		}
	}
}

// In-place Cholesky A = L L^T of the symmetric n x n row-major matrix `a`
// (leading dimension lda; lower triangle read and overwritten with L; the
// strict upper triangle is used as scratch and left with garbage).
// Returns 0 on success, otherwise 1 + the row of the first pivot <= pivotTolerance
// (the matrix is then partly overwritten), like choleskyFactor.
// Right-looking, block size 128: factor the diagonal block (choleskyFactor),
// solve the panel below it (parallel over rows), update the trailing lower
// triangle with gemmSubtract (parallel over tiles). n^3/3 flops.
inline size_t blockedCholesky(double* a, size_t n, size_t lda, double pivotTolerance, unsigned nbThreads) {
	constexpr size_t NB = 128; // panel width
	constexpr size_t TB = 128; // trailing-update tile
	if (n <= NB) return choleskyFactor(a, n, lda, pivotTolerance);
	std::vector<double> panelT; // panel transposed: NB x (rows below)
	for (size_t k0 = 0; k0 < n; k0 += NB) {
		const size_t k1 = std::min(n, k0 + NB), bk = k1 - k0;
		const size_t failed = choleskyFactor(a + k0 * lda + k0, bk, lda, pivotTolerance);
		if (failed) return k0 + failed;
		if (k1 == n) break;
		const size_t rows = n - k1;
		// Panel: L[i, k0:k1] = A[i, k0:k1] L_kk^-T, row by row.
		const size_t rowBlocks = (rows + 63) / 64;
		parallelFor(rowBlocks, nbThreads, [&](size_t block, unsigned) {
			const size_t r1 = std::min(rows, (block + 1) * 64);
			for (size_t r = block * 64; r < r1; ++r) {
				double* li = a + (k1 + r) * lda + k0;
				for (size_t j = 0; j < bk; ++j) {
					const double* lj = a + (k0 + j) * lda + k0;
					li[j] = (li[j] - dotProduct(li, lj, j)) / lj[j];
				}
			}
		});
		// Transposed copy of the panel so that the update is a plain row-major gemm.
		panelT.resize(bk * rows);
		for (size_t r = 0; r < rows; ++r) {
			const double* li = a + (k1 + r) * lda + k0;
			for (size_t j = 0; j < bk; ++j) panelT[j * rows + r] = li[j];
		}
		// Trailing update of the lower triangle: tiles (I, J) with J <= I.
		const size_t tiles = (rows + TB - 1) / TB;
		const size_t pairs = tiles * (tiles + 1) / 2;
		parallelFor(pairs, nbThreads, [&](size_t pair, unsigned) {
			// Pairs handed out last-tile-row first; q -> (I, J) with J <= I.
			const size_t q = pairs - 1 - pair;
			size_t I = size_t((std::sqrt(8.0 * double(q) + 1.0) - 1.0) / 2.0);
			while (I * (I + 1) / 2 > q) --I;
			while ((I + 1) * (I + 2) / 2 <= q) ++I;
			const size_t J = q - I * (I + 1) / 2;
			const size_t i0 = I * TB, i1 = std::min(rows, i0 + TB);
			const size_t j0 = J * TB, j1 = std::min(rows, j0 + TB);
			gemmSubtract(a + (k1 + i0) * lda + k1 + j0, lda, a + (k1 + i0) * lda + k0, lda, panelT.data() + j0, rows, i1 - i0, j1 - j0, bk);
		});
	}
	return 0;
}

// Solves L Y = B in place for nrhs right-hand sides: L lower triangular n x n
// (leading dimension ldl, from choleskyFactor / blockedCholesky), B row-major
// n x nrhs (leading dimension ldb). Blocked: per block of 64 rows, one gemm
// with the rows above, then a small triangular solve. Single-threaded (callers
// parallelise over independent blocks of right-hand sides).
inline void blockedForwardSubstitution(const double* l, size_t n, size_t ldl, double* b, size_t nrhs, size_t ldb) {
	constexpr size_t NB = 64;
	for (size_t i0 = 0; i0 < n; i0 += NB) {
		const size_t i1 = std::min(n, i0 + NB);
		if (i0 > 0) gemmSubtract(b + i0 * ldb, ldb, l + i0 * ldl, ldl, b, ldb, i1 - i0, nrhs, i0);
		for (size_t i = i0; i < i1; ++i) {
			const double* li = l + i * ldl;
			double* bi = b + i * ldb;
			for (size_t k = i0; k < i; ++k) {
				const double f = li[k];
				const double* bk = b + k * ldb;
				for (size_t r = 0; r < nrhs; ++r) bi[r] -= f * bk[r];
			}
			const double inverse = 1.0 / li[i];
			for (size_t r = 0; r < nrhs; ++r) bi[r] *= inverse;
		}
	}
}

} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_DENSE_SOLVER_HPP
