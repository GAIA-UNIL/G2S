/*
 * G2S - geostat: minimal explicit SIMD layer for the turning-bands kernels.
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

// Three vector types with the same number of lanes, kLanes:
//   F  kLanes floats             (one native register)
//   D  kLanes doubles            (two native registers)
//   U  kLanes unsigned 32-bit ints
// and the few operations the turning-bands kernels need. The instruction set
// is chosen at compile time from the compiler's target macros, as in
// fKs.hpp / fKb.hpp / complexMulti.hpp:
//   ARM64 (__aarch64__, __arm64__, _M_ARM64)   NEON          kLanes 4
//   __AVX512F__                                AVX-512F      kLanes 16
//   __AVX2__                                   AVX2 (+FMA)   kLanes 8
//   __SSE2__ / x86-64                          SSE2 (+SSE4.1, +FMA when enabled) kLanes 4
//   otherwise, or G2S_GEOSTAT_SIMD_SCALAR      scalar        kLanes 1
// (an x86-64 build without -m flags is SSE2: build with -march=native or
// -mavx2 -mfma to get the wider paths.) Every operation gives the same value
// in every lane and in the scalar build up to the use of fused multiply-adds
// (fMulAdd / dMulAdd are fused where the instruction set has FMA, a multiply
// then an add otherwise): a given binary is deterministic, two builds for
// different instruction sets may differ in the last bits.

#ifndef G2S_GEOSTAT_SIMD_LANES_HPP
#define G2S_GEOSTAT_SIMD_LANES_HPP

#include <cmath>
#include <cstdint>
#include <cstring>

#if !defined(G2S_GEOSTAT_SIMD_SCALAR)
	#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
		#include <arm_neon.h>
		#define G2S_GEOSTAT_SIMD_NEON 1
	#elif defined(__AVX512F__)
		#include <immintrin.h>
		#define G2S_GEOSTAT_SIMD_AVX512 1
	#elif defined(__AVX2__)
		#include <immintrin.h>
		#define G2S_GEOSTAT_SIMD_AVX2 1
	#elif defined(__SSE2__) || defined(_M_X64)
		#include <immintrin.h>
		#define G2S_GEOSTAT_SIMD_SSE2 1
	#else
		#define G2S_GEOSTAT_SIMD_SCALAR 1
	#endif
#endif

namespace g2s {
namespace geostat {
namespace simd {

// 1.5 * 2^52: x + kRoundMagic - kRoundMagic rounds x to the nearest integer
// (ties to even) for |x| < 2^51, and the low 32 bits of x + kRoundMagic are
// x mod 2^32 for an integral x.
constexpr double kRoundMagic = 6755399441055744.0;

#if G2S_GEOSTAT_SIMD_NEON
// ------------------------------------------------------------------ NEON
constexpr unsigned kLanes = 4;
constexpr const char* kInstructionSet = "NEON";
struct F { float32x4_t v; };
struct D { float64x2_t lo, hi; };
struct U { uint32x4_t v; };

inline F fZero() { return { vdupq_n_f32(0.0f) }; }
inline F fSet(float a) { return { vdupq_n_f32(a) }; }
inline F fLoad(const float* p) { return { vld1q_f32(p) }; }
inline void fStore(float* p, F a) { vst1q_f32(p, a.v); }
inline F fAdd(F a, F b) { return { vaddq_f32(a.v, b.v) }; }
inline F fSub(F a, F b) { return { vsubq_f32(a.v, b.v) }; }
inline F fMul(F a, F b) { return { vmulq_f32(a.v, b.v) }; }
inline F fMulAdd(F a, F b, F c) { return { vfmaq_f32(c.v, a.v, b.v) }; }    // a * b + c
inline F fNegMulAdd(F a, F b, F c) { return { vfmsq_f32(c.v, a.v, b.v) }; } // c - a * b
inline F fXorBits(F a, U b) { return { vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(a.v), b.v)) }; }

inline D dSet(double a) { return { vdupq_n_f64(a), vdupq_n_f64(a) }; }
inline D dLoad(const double* p) { return { vld1q_f64(p), vld1q_f64(p + 2) }; }
inline void dStore(double* p, D a) { vst1q_f64(p, a.lo); vst1q_f64(p + 2, a.hi); }
inline D dAdd(D a, D b) { return { vaddq_f64(a.lo, b.lo), vaddq_f64(a.hi, b.hi) }; }
inline D dSub(D a, D b) { return { vsubq_f64(a.lo, b.lo), vsubq_f64(a.hi, b.hi) }; }
inline D dMul(D a, D b) { return { vmulq_f64(a.lo, b.lo), vmulq_f64(a.hi, b.hi) }; }
inline D dMulAdd(D a, D b, D c) { return { vfmaq_f64(c.lo, a.lo, b.lo), vfmaq_f64(c.hi, a.hi, b.hi) }; }
inline D dAbs(D a) { return { vabsq_f64(a.lo), vabsq_f64(a.hi) }; }
inline D dRound(D a) { return { vrndnq_f64(a.lo), vrndnq_f64(a.hi) }; }
inline D dFloor(D a) { return { vrndmq_f64(a.lo), vrndmq_f64(a.hi) }; }
inline F dToF(D a) { return { vcvt_high_f32_f64(vcvt_f32_f64(a.lo), a.hi) }; }
inline D fToD(F a) { return { vcvt_f64_f32(vget_low_f32(a.v)), vcvt_high_f64_f32(a.v) }; }
inline U dLow32(D integral) {
	const float64x2_t m = vdupq_n_f64(kRoundMagic);
	return { vuzp1q_u32(vreinterpretq_u32_f64(vaddq_f64(integral.lo, m)), vreinterpretq_u32_f64(vaddq_f64(integral.hi, m))) };
}

inline U uSet(uint32_t a) { return { vdupq_n_u32(a) }; }
inline U uAdd(U a, U b) { return { vaddq_u32(a.v, b.v) }; }
inline U uMul(U a, U b) { return { vmulq_u32(a.v, b.v) }; }
inline U uXor(U a, U b) { return { veorq_u32(a.v, b.v) }; }
inline U uAnd(U a, U b) { return { vandq_u32(a.v, b.v) }; }
template <int N> inline U uShr(U a) { return { vshrq_n_u32(a.v, N) }; }

#elif G2S_GEOSTAT_SIMD_AVX512
// ------------------------------------------------------------------ AVX-512F
constexpr unsigned kLanes = 16;
constexpr const char* kInstructionSet = "AVX-512F";
struct F { __m512 v; };
struct D { __m512d lo, hi; };
struct U { __m512i v; };

inline F fZero() { return { _mm512_setzero_ps() }; }
inline F fSet(float a) { return { _mm512_set1_ps(a) }; }
inline F fLoad(const float* p) { return { _mm512_loadu_ps(p) }; }
inline void fStore(float* p, F a) { _mm512_storeu_ps(p, a.v); }
inline F fAdd(F a, F b) { return { _mm512_add_ps(a.v, b.v) }; }
inline F fSub(F a, F b) { return { _mm512_sub_ps(a.v, b.v) }; }
inline F fMul(F a, F b) { return { _mm512_mul_ps(a.v, b.v) }; }
inline F fMulAdd(F a, F b, F c) { return { _mm512_fmadd_ps(a.v, b.v, c.v) }; }
inline F fNegMulAdd(F a, F b, F c) { return { _mm512_fnmadd_ps(a.v, b.v, c.v) }; }
inline F fXorBits(F a, U b) { return { _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(a.v), b.v)) }; }

inline D dSet(double a) { return { _mm512_set1_pd(a), _mm512_set1_pd(a) }; }
inline D dLoad(const double* p) { return { _mm512_loadu_pd(p), _mm512_loadu_pd(p + 8) }; }
inline void dStore(double* p, D a) { _mm512_storeu_pd(p, a.lo); _mm512_storeu_pd(p + 8, a.hi); }
inline D dAdd(D a, D b) { return { _mm512_add_pd(a.lo, b.lo), _mm512_add_pd(a.hi, b.hi) }; }
inline D dSub(D a, D b) { return { _mm512_sub_pd(a.lo, b.lo), _mm512_sub_pd(a.hi, b.hi) }; }
inline D dMul(D a, D b) { return { _mm512_mul_pd(a.lo, b.lo), _mm512_mul_pd(a.hi, b.hi) }; }
inline D dMulAdd(D a, D b, D c) { return { _mm512_fmadd_pd(a.lo, b.lo, c.lo), _mm512_fmadd_pd(a.hi, b.hi, c.hi) }; }
inline D dAbs(D a) { return { _mm512_abs_pd(a.lo), _mm512_abs_pd(a.hi) }; }
inline D dRound(D a) {
	return { _mm512_roundscale_pd(a.lo, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _mm512_roundscale_pd(a.hi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) };
}
inline D dFloor(D a) {
	return { _mm512_roundscale_pd(a.lo, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC), _mm512_roundscale_pd(a.hi, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC) };
}
inline F dToF(D a) {
	const __m256 lo = _mm512_cvtpd_ps(a.lo), hi = _mm512_cvtpd_ps(a.hi);
	return { _mm512_castpd_ps(_mm512_insertf64x4(_mm512_castps_pd(_mm512_castps256_ps512(lo)), _mm256_castps_pd(hi), 1)) };
}
inline D fToD(F a) {
	return { _mm512_cvtps_pd(_mm512_castps512_ps256(a.v)), _mm512_cvtps_pd(_mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(a.v), 1))) };
}
inline U dLow32(D integral) {
	const __m512d m = _mm512_set1_pd(kRoundMagic);
	const __m256i lo = _mm512_cvtepi64_epi32(_mm512_castpd_si512(_mm512_add_pd(integral.lo, m)));
	const __m256i hi = _mm512_cvtepi64_epi32(_mm512_castpd_si512(_mm512_add_pd(integral.hi, m)));
	return { _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1) };
}

inline U uSet(uint32_t a) { return { _mm512_set1_epi32(int(a)) }; }
inline U uAdd(U a, U b) { return { _mm512_add_epi32(a.v, b.v) }; }
inline U uMul(U a, U b) { return { _mm512_mullo_epi32(a.v, b.v) }; }
inline U uXor(U a, U b) { return { _mm512_xor_si512(a.v, b.v) }; }
inline U uAnd(U a, U b) { return { _mm512_and_si512(a.v, b.v) }; }
template <int N> inline U uShr(U a) { return { _mm512_srli_epi32(a.v, N) }; }

#elif G2S_GEOSTAT_SIMD_AVX2
// ------------------------------------------------------------------ AVX2
constexpr unsigned kLanes = 8;
constexpr const char* kInstructionSet = "AVX2";
struct F { __m256 v; };
struct D { __m256d lo, hi; };
struct U { __m256i v; };

inline F fZero() { return { _mm256_setzero_ps() }; }
inline F fSet(float a) { return { _mm256_set1_ps(a) }; }
inline F fLoad(const float* p) { return { _mm256_loadu_ps(p) }; }
inline void fStore(float* p, F a) { _mm256_storeu_ps(p, a.v); }
inline F fAdd(F a, F b) { return { _mm256_add_ps(a.v, b.v) }; }
inline F fSub(F a, F b) { return { _mm256_sub_ps(a.v, b.v) }; }
inline F fMul(F a, F b) { return { _mm256_mul_ps(a.v, b.v) }; }
#if defined(__FMA__)
inline F fMulAdd(F a, F b, F c) { return { _mm256_fmadd_ps(a.v, b.v, c.v) }; }
inline F fNegMulAdd(F a, F b, F c) { return { _mm256_fnmadd_ps(a.v, b.v, c.v) }; }
#else
inline F fMulAdd(F a, F b, F c) { return { _mm256_add_ps(_mm256_mul_ps(a.v, b.v), c.v) }; }
inline F fNegMulAdd(F a, F b, F c) { return { _mm256_sub_ps(c.v, _mm256_mul_ps(a.v, b.v)) }; }
#endif
inline F fXorBits(F a, U b) { return { _mm256_xor_ps(a.v, _mm256_castsi256_ps(b.v)) }; }

inline D dSet(double a) { return { _mm256_set1_pd(a), _mm256_set1_pd(a) }; }
inline D dLoad(const double* p) { return { _mm256_loadu_pd(p), _mm256_loadu_pd(p + 4) }; }
inline void dStore(double* p, D a) { _mm256_storeu_pd(p, a.lo); _mm256_storeu_pd(p + 4, a.hi); }
inline D dAdd(D a, D b) { return { _mm256_add_pd(a.lo, b.lo), _mm256_add_pd(a.hi, b.hi) }; }
inline D dSub(D a, D b) { return { _mm256_sub_pd(a.lo, b.lo), _mm256_sub_pd(a.hi, b.hi) }; }
inline D dMul(D a, D b) { return { _mm256_mul_pd(a.lo, b.lo), _mm256_mul_pd(a.hi, b.hi) }; }
#if defined(__FMA__)
inline D dMulAdd(D a, D b, D c) { return { _mm256_fmadd_pd(a.lo, b.lo, c.lo), _mm256_fmadd_pd(a.hi, b.hi, c.hi) }; }
#else
inline D dMulAdd(D a, D b, D c) { return dAdd(dMul(a, b), c); }
#endif
inline D dAbs(D a) {
	const __m256d sign = _mm256_set1_pd(-0.0);
	return { _mm256_andnot_pd(sign, a.lo), _mm256_andnot_pd(sign, a.hi) };
}
inline D dRound(D a) {
	return { _mm256_round_pd(a.lo, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _mm256_round_pd(a.hi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) };
}
inline D dFloor(D a) { return { _mm256_floor_pd(a.lo), _mm256_floor_pd(a.hi) }; }
inline F dToF(D a) { return { _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(a.lo)), _mm256_cvtpd_ps(a.hi), 1) }; }
inline D fToD(F a) { return { _mm256_cvtps_pd(_mm256_castps256_ps128(a.v)), _mm256_cvtps_pd(_mm256_extractf128_ps(a.v, 1)) }; }
inline U dLow32(D integral) {
	const __m256d m = _mm256_set1_pd(kRoundMagic);
	// per 128-bit half: (lo0 lo1 hi0 hi1 | lo2 lo3 hi2 hi3), then the 64-bit pairs in order.
	const __m256 both = _mm256_shuffle_ps(_mm256_castpd_ps(_mm256_add_pd(integral.lo, m)), _mm256_castpd_ps(_mm256_add_pd(integral.hi, m)), _MM_SHUFFLE(2, 0, 2, 0));
	return { _mm256_castpd_si256(_mm256_permute4x64_pd(_mm256_castps_pd(both), _MM_SHUFFLE(3, 1, 2, 0))) };
}

inline U uSet(uint32_t a) { return { _mm256_set1_epi32(int(a)) }; }
inline U uAdd(U a, U b) { return { _mm256_add_epi32(a.v, b.v) }; }
inline U uMul(U a, U b) { return { _mm256_mullo_epi32(a.v, b.v) }; }
inline U uXor(U a, U b) { return { _mm256_xor_si256(a.v, b.v) }; }
inline U uAnd(U a, U b) { return { _mm256_and_si256(a.v, b.v) }; }
template <int N> inline U uShr(U a) { return { _mm256_srli_epi32(a.v, N) }; }

#elif G2S_GEOSTAT_SIMD_SSE2
// ------------------------------------------------------------------ SSE2 (SSE4.1, FMA when enabled)
constexpr unsigned kLanes = 4;
#if defined(__SSE4_1__)
constexpr const char* kInstructionSet = "SSE4.1";
#else
constexpr const char* kInstructionSet = "SSE2";
#endif
struct F { __m128 v; };
struct D { __m128d lo, hi; };
struct U { __m128i v; };

inline F fZero() { return { _mm_setzero_ps() }; }
inline F fSet(float a) { return { _mm_set1_ps(a) }; }
inline F fLoad(const float* p) { return { _mm_loadu_ps(p) }; }
inline void fStore(float* p, F a) { _mm_storeu_ps(p, a.v); }
inline F fAdd(F a, F b) { return { _mm_add_ps(a.v, b.v) }; }
inline F fSub(F a, F b) { return { _mm_sub_ps(a.v, b.v) }; }
inline F fMul(F a, F b) { return { _mm_mul_ps(a.v, b.v) }; }
#if defined(__FMA__)
inline F fMulAdd(F a, F b, F c) { return { _mm_fmadd_ps(a.v, b.v, c.v) }; }
inline F fNegMulAdd(F a, F b, F c) { return { _mm_fnmadd_ps(a.v, b.v, c.v) }; }
#else
inline F fMulAdd(F a, F b, F c) { return { _mm_add_ps(_mm_mul_ps(a.v, b.v), c.v) }; }
inline F fNegMulAdd(F a, F b, F c) { return { _mm_sub_ps(c.v, _mm_mul_ps(a.v, b.v)) }; }
#endif
inline F fXorBits(F a, U b) { return { _mm_xor_ps(a.v, _mm_castsi128_ps(b.v)) }; }

inline D dSet(double a) { return { _mm_set1_pd(a), _mm_set1_pd(a) }; }
inline D dLoad(const double* p) { return { _mm_loadu_pd(p), _mm_loadu_pd(p + 2) }; }
inline void dStore(double* p, D a) { _mm_storeu_pd(p, a.lo); _mm_storeu_pd(p + 2, a.hi); }
inline D dAdd(D a, D b) { return { _mm_add_pd(a.lo, b.lo), _mm_add_pd(a.hi, b.hi) }; }
inline D dSub(D a, D b) { return { _mm_sub_pd(a.lo, b.lo), _mm_sub_pd(a.hi, b.hi) }; }
inline D dMul(D a, D b) { return { _mm_mul_pd(a.lo, b.lo), _mm_mul_pd(a.hi, b.hi) }; }
#if defined(__FMA__)
inline D dMulAdd(D a, D b, D c) { return { _mm_fmadd_pd(a.lo, b.lo, c.lo), _mm_fmadd_pd(a.hi, b.hi, c.hi) }; }
#else
inline D dMulAdd(D a, D b, D c) { return dAdd(dMul(a, b), c); }
#endif
inline D dAbs(D a) {
	const __m128d sign = _mm_set1_pd(-0.0);
	return { _mm_andnot_pd(sign, a.lo), _mm_andnot_pd(sign, a.hi) };
}
#if defined(__SSE4_1__)
inline D dRound(D a) { return { _mm_round_pd(a.lo, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _mm_round_pd(a.hi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) }; }
inline D dFloor(D a) { return { _mm_floor_pd(a.lo), _mm_floor_pd(a.hi) }; }
#else
inline D dRound(D a) {
	const __m128d m = _mm_set1_pd(kRoundMagic);
	return { _mm_sub_pd(_mm_add_pd(a.lo, m), m), _mm_sub_pd(_mm_add_pd(a.hi, m), m) };
}
inline D dFloor(D a) {
	const D r = dRound(a);
	const __m128d one = _mm_set1_pd(1.0);
	return { _mm_sub_pd(r.lo, _mm_and_pd(_mm_cmpgt_pd(r.lo, a.lo), one)), _mm_sub_pd(r.hi, _mm_and_pd(_mm_cmpgt_pd(r.hi, a.hi), one)) };
}
#endif
inline F dToF(D a) { return { _mm_movelh_ps(_mm_cvtpd_ps(a.lo), _mm_cvtpd_ps(a.hi)) }; }
inline D fToD(F a) { return { _mm_cvtps_pd(a.v), _mm_cvtps_pd(_mm_movehl_ps(a.v, a.v)) }; }
inline U dLow32(D integral) {
	const __m128d m = _mm_set1_pd(kRoundMagic);
	return { _mm_castps_si128(_mm_shuffle_ps(_mm_castpd_ps(_mm_add_pd(integral.lo, m)), _mm_castpd_ps(_mm_add_pd(integral.hi, m)), _MM_SHUFFLE(2, 0, 2, 0))) };
}

inline U uSet(uint32_t a) { return { _mm_set1_epi32(int(a)) }; }
inline U uAdd(U a, U b) { return { _mm_add_epi32(a.v, b.v) }; }
#if defined(__SSE4_1__)
inline U uMul(U a, U b) { return { _mm_mullo_epi32(a.v, b.v) }; }
#else
inline U uMul(U a, U b) {
	const __m128i even = _mm_mul_epu32(a.v, b.v);
	const __m128i odd = _mm_mul_epu32(_mm_srli_epi64(a.v, 32), _mm_srli_epi64(b.v, 32));
	return { _mm_unpacklo_epi32(_mm_shuffle_epi32(even, _MM_SHUFFLE(0, 0, 2, 0)), _mm_shuffle_epi32(odd, _MM_SHUFFLE(0, 0, 2, 0))) };
}
#endif
inline U uXor(U a, U b) { return { _mm_xor_si128(a.v, b.v) }; }
inline U uAnd(U a, U b) { return { _mm_and_si128(a.v, b.v) }; }
template <int N> inline U uShr(U a) { return { _mm_srli_epi32(a.v, N) }; }

#else
// ------------------------------------------------------------------ scalar
constexpr unsigned kLanes = 1;
constexpr const char* kInstructionSet = "scalar";
struct F { float v; };
struct D { double v; };
struct U { uint32_t v; };

inline F fZero() { return { 0.0f }; }
inline F fSet(float a) { return { a }; }
inline F fLoad(const float* p) { return { *p }; }
inline void fStore(float* p, F a) { *p = a.v; }
inline F fAdd(F a, F b) { return { a.v + b.v }; }
inline F fSub(F a, F b) { return { a.v - b.v }; }
inline F fMul(F a, F b) { return { a.v * b.v }; }
inline F fMulAdd(F a, F b, F c) { const float p = a.v * b.v; return { p + c.v }; }
inline F fNegMulAdd(F a, F b, F c) { const float p = a.v * b.v; return { c.v - p }; }
inline F fXorBits(F a, U b) { uint32_t bits; std::memcpy(&bits, &a.v, 4); bits ^= b.v; float r; std::memcpy(&r, &bits, 4); return { r }; }

inline D dSet(double a) { return { a }; }
inline D dLoad(const double* p) { return { *p }; }
inline void dStore(double* p, D a) { *p = a.v; }
inline D dAdd(D a, D b) { return { a.v + b.v }; }
inline D dSub(D a, D b) { return { a.v - b.v }; }
inline D dMul(D a, D b) { return { a.v * b.v }; }
inline D dMulAdd(D a, D b, D c) { const double p = a.v * b.v; return { p + c.v }; }
inline D dAbs(D a) { return { std::fabs(a.v) }; }
inline D dRound(D a) { return { std::nearbyint(a.v) }; }
inline D dFloor(D a) { return { std::floor(a.v) }; }
inline F dToF(D a) { return { float(a.v) }; }
inline D fToD(F a) { return { double(a.v) }; }
inline U dLow32(D integral) { const double b = integral.v + kRoundMagic; uint64_t bits; std::memcpy(&bits, &b, 8); return { uint32_t(bits) }; }

inline U uSet(uint32_t a) { return { a }; }
inline U uAdd(U a, U b) { return { a.v + b.v }; }
inline U uMul(U a, U b) { return { a.v * b.v }; }
inline U uXor(U a, U b) { return { a.v ^ b.v }; }
inline U uAnd(U a, U b) { return { a.v & b.v }; }
template <int N> inline U uShr(U a) { return { a.v >> N }; }
#endif

} // namespace simd
} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_SIMD_LANES_HPP
