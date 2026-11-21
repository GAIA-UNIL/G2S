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

// Counter-based random numbers: every draw is a pure function of
// (seed, realisation, index, stream), so a simulation gives the same result
// whatever the number of threads, the order in which nodes are processed or
// the way realisations are batched.
//
// Generator: Philox4x32-10 (Salmon et al., SC'11, as in Random123), 128 random
// bits per call. Counter = (index low 32 bits, index high 32 bits,
// realisation, stream), key = (seed low 32 bits, seed high 32 bits).
//   index:   the path position (SGS), a cell or band index, ... (64 bits)
//   stream:  separates independent uses at the same index (e.g. 0 = the node's
//            normal deviate, 1 = path shuffling, 2+ = algorithm specific).
// uniform() uses the first 64 bits of the block, uniform2() the last 64.
//
// Example: const double z = g2s::geostat::standardNormal(seed, realisation, pathIndex);
//
// Which normal generator:
//   standardNormal(), standardNormalPair()
//       double precision, inverse CDF of one uniform: monotone in the uniform
//       and exact in the tails. About 25 ns per draw. For draws that are few
//       or need the uniform-to-normal link (line parameters, path decisions).
//   fastNormalByIndex(), fillFastNormalsByIndex()
//       float, Box-Muller on 32-bit uniforms, four normals per Philox block,
//       written to vectorise: about 1 ns per draw in bulk. Tails are cut at
//       6.7 standard deviations (|z| <= sqrt(66 ln 2)). For bulk noise over
//       consecutive indices: white noise of a grid (fftma, spectral), nugget.
//   fastNormalByRealisation(), fillFastNormalsByRealisation()
//       the same, with the four normals of a block on four consecutive
//       realisations of one index: for noise drawn for R realisations of a
//       node at once (SGS replay).
// Every generator is a pure function of (seed, realisation, index, stream);
// the three families use different Philox keys, so they never share a block,
// and a fill gives exactly the values of the scalar function (whatever the
// first index or realisation and the count, so whatever the batching).

#ifndef G2S_GEOSTAT_RANDOM_HPP
#define G2S_GEOSTAT_RANDOM_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <cmath>

#include "geostat/specialFunctions.hpp"

namespace g2s {
namespace geostat {

// Philox4x32 with 10 rounds.
inline std::array<uint32_t, 4> philox4x32(std::array<uint32_t, 4> counter, std::array<uint32_t, 2> key) {
	const uint32_t m0 = 0xD2511F53u, m1 = 0xCD9E8D57u;
	const uint32_t w0 = 0x9E3779B9u, w1 = 0xBB67AE85u;
	for (int round = 0; round < 10; ++round) {
		if (round > 0) { key[0] += w0; key[1] += w1; }
		const uint64_t p0 = uint64_t(m0) * counter[0];
		const uint64_t p1 = uint64_t(m1) * counter[2];
		const uint32_t hi0 = uint32_t(p0 >> 32), lo0 = uint32_t(p0);
		const uint32_t hi1 = uint32_t(p1 >> 32), lo1 = uint32_t(p1);
		counter = { hi1 ^ counter[1] ^ key[0], lo1, hi0 ^ counter[3] ^ key[1], lo0 };
	}
	return counter;
}

// The 128 random bits of (seed, realisation, index, stream).
inline std::array<uint32_t, 4> randomBlock(uint64_t seed, uint32_t realisation, uint64_t index, uint32_t stream = 0) {
	return philox4x32({ uint32_t(index), uint32_t(index >> 32), realisation, stream }, { uint32_t(seed), uint32_t(seed >> 32) });
}

// 64 random bits to a double in the open interval (0, 1): (k + 0.5) / 2^53
// with k the top 53 bits; never 0 or 1.
inline double bitsToUniform(uint32_t high, uint32_t low) {
	const uint64_t k = ((uint64_t(high) << 32) | low) >> 11;
	return (double(k) + 0.5) * (1.0 / 9007199254740992.0);
}

// Uniform in (0, 1).
inline double uniform(uint64_t seed, uint32_t realisation, uint64_t index, uint32_t stream = 0) {
	const std::array<uint32_t, 4> bits = randomBlock(seed, realisation, index, stream);
	return bitsToUniform(bits[0], bits[1]);
}

// A second uniform from the same block, independent of uniform().
inline double uniform2(uint64_t seed, uint32_t realisation, uint64_t index, uint32_t stream = 0) {
	const std::array<uint32_t, 4> bits = randomBlock(seed, realisation, index, stream);
	return bitsToUniform(bits[2], bits[3]);
}

// Standard normal: inverseNormalCdf(uniform(...)), so a node's deviate is
// monotone in its uniform (useful for antithetic or quantile-based variants).
inline double standardNormal(uint64_t seed, uint32_t realisation, uint64_t index, uint32_t stream = 0) {
	return inverseNormalCdf(uniform(seed, realisation, index, stream));
}

// Both normals of one block: inverseNormalCdf of uniform() and uniform2().
inline void standardNormalPair(uint64_t seed, uint32_t realisation, uint64_t index, uint32_t stream, double& first, double& second) {
	const std::array<uint32_t, 4> bits = randomBlock(seed, realisation, index, stream);
	first = inverseNormalCdf(bitsToUniform(bits[0], bits[1]));
	second = inverseNormalCdf(bitsToUniform(bits[2], bits[3]));
}

// ============================================================ fast normals

namespace detail {

// Philox key of the fast generators: the seed with a per-family tweak, so that
// they never reuse a block of uniform() / standardNormal().
inline std::array<uint32_t, 2> fastByIndexKey(uint64_t seed) { return { uint32_t(seed) ^ 0xA4093822u, uint32_t(seed >> 32) ^ 0x299F31D0u }; }
inline std::array<uint32_t, 2> fastByRealisationKey(uint64_t seed) { return { uint32_t(seed) ^ 0x082EFA98u, uint32_t(seed >> 32) ^ 0xEC4E6C89u }; }

inline float floatFromBits(uint32_t bits) { float f; std::memcpy(&f, &bits, sizeof(f)); return f; }
inline uint32_t bitsFromFloat(float f) { uint32_t bits; std::memcpy(&bits, &f, sizeof(bits)); return bits; }

// a * b + c rounded twice. The product is a separate statement so that the
// compiler does not fuse it into an FMA (fp-contract=on fuses within one
// expression only): fused or not would depend on whether the loop around is
// vectorised, and the values must not depend on that.
inline float mulAdd(float a, float b, float c) {
	const float product = a * b;
	return product + c;
}

// ln(x) for a normal float x in (0, 1], branch-free (Cephes logf, ~1 ulp).
inline float fastLog(float x) {
	const uint32_t bits = bitsFromFloat(x);
	int32_t e = int32_t(bits >> 23) - 126; // x = m 2^e, m in [0.5, 1)
	float m = floatFromBits((bits & 0x007FFFFFu) | 0x3F000000u);
	const bool low = m < 0.707106781186547524f;
	e -= low ? 1 : 0;
	m = low ? m + m - 1.0f : m - 1.0f;
	const float z = m * m;
	float y = 7.0376836292e-2f;
	y = mulAdd(y, m, -1.1514610310e-1f);
	y = mulAdd(y, m, 1.1676998740e-1f);
	y = mulAdd(y, m, -1.2420140846e-1f);
	y = mulAdd(y, m, 1.4249322787e-1f);
	y = mulAdd(y, m, -1.6668057665e-1f);
	y = mulAdd(y, m, 2.0000714765e-1f);
	y = mulAdd(y, m, -2.4999993993e-1f);
	y = mulAdd(y, m, 3.3333331174e-1f);
	const float mz = m * z;
	y = y * mz;
	const float fe = float(e);
	y = mulAdd(-2.12194440e-4f, fe, y);
	y = mulAdd(-0.5f, z, y);
	const float sum = m + y;
	return mulAdd(0.693359375f, fe, sum);
}

// Box-Muller on two 32-bit words: radius from u1 = (k1 + 1/2) / 2^32 in
// (0, 1], angle 2 pi k2 / 2^32 reduced to the nearest quadrant in integers.
// first = radius cos(angle), second = radius sin(angle).
inline void boxMuller(uint32_t k1, uint32_t k2, float& first, float& second) {
	const float u1 = mulAdd(float(k1), 0x1p-32f, 0x1p-33f);
	const float radius = std::sqrt(-2.0f * fastLog(u1));
	const uint32_t shifted = k2 + (1u << 29);
	const uint32_t quadrant = shifted >> 30;
	const float a = float(int32_t(shifted & 0x3FFFFFFFu) - int32_t(1u << 29)) * 1.46291807926715968e-9f; // 2 pi / 2^32, |a| <= pi/4
	const float z = a * a;
	float s = mulAdd(-1.9515295891e-4f, z, 8.3321608736e-3f);
	s = mulAdd(s, z, -1.6666654611e-1f);
	const float za = z * a;
	s = mulAdd(s, za, a);
	float c = mulAdd(2.443315711809948e-5f, z, -1.388731625493765e-3f);
	c = mulAdd(c, z, 4.166664568298827e-2f);
	const float zz = z * z;
	c = mulAdd(c, zz, 1.0f);
	c = mulAdd(-0.5f, z, c);
	// angle = quadrant * pi/2 + a
	const float cq = (quadrant & 1u) ? s : c;
	const float sq = (quadrant & 1u) ? c : s;
	const float cosine = (quadrant == 1u || quadrant == 2u) ? -cq : cq;
	const float sine = (quadrant >= 2u) ? -sq : sq;
	first = radius * cosine;
	second = radius * sine;
}

// Normals of lanes [begin, end) of the Philox block `counter` into out[0 ..
// end - begin) (lanes 0-1 and 2-3 are two Box-Muller pairs; only the pairs
// needed are computed).
inline void fastNormalLanes(const std::array<uint32_t, 4>& counter, const uint32_t key[2], unsigned begin, unsigned end, float* out) {
	const std::array<uint32_t, 4> bits = philox4x32(counter, { key[0], key[1] });
	float lanes[4];
	if (begin < 2) boxMuller(bits[0], bits[1], lanes[0], lanes[1]);
	if (end > 2) boxMuller(bits[2], bits[3], lanes[2], lanes[3]);
	for (unsigned lane = begin; lane < end; ++lane) out[lane - begin] = lanes[lane];
}

// Normals of blocks firstBlock .. firstBlock + blockCount - 1 (blockCount <=
// kFastBatch) into out[4 * blockCount]; counter(b, c) fills the counter of
// block b. Structure-of-arrays loops, so that the compiler vectorises Philox
// over blocks and Box-Muller over pairs; the arithmetic is that of
// fastNormalLanes, so the values are identical.
constexpr size_t kFastBatch = 32;
template <class Counter>
inline void fastNormalBlocks(uint64_t firstBlock, size_t blockCount, const uint32_t key[2], Counter counter, float* out) {
	uint32_t c0[kFastBatch], c1[kFastBatch], c2[kFastBatch], c3[kFastBatch];
	for (size_t i = 0; i < blockCount; ++i) {
		std::array<uint32_t, 4> c;
		counter(firstBlock + i, c);
		c0[i] = c[0]; c1[i] = c[1]; c2[i] = c[2]; c3[i] = c[3];
	}
	uint32_t k0 = key[0], k1 = key[1];
	for (int round = 0; round < 10; ++round) {
		if (round > 0) { k0 += 0x9E3779B9u; k1 += 0xBB67AE85u; }
		for (size_t i = 0; i < blockCount; ++i) {
			const uint64_t p0 = uint64_t(0xD2511F53u) * c0[i];
			const uint64_t p1 = uint64_t(0xCD9E8D57u) * c2[i];
			const uint32_t n0 = uint32_t(p1 >> 32) ^ c1[i] ^ k0;
			const uint32_t n2 = uint32_t(p0 >> 32) ^ c3[i] ^ k1;
			c0[i] = n0; c1[i] = uint32_t(p1); c2[i] = n2; c3[i] = uint32_t(p0);
		}
	}
	float a0[kFastBatch], a1[kFastBatch], a2[kFastBatch], a3[kFastBatch];
	for (size_t i = 0; i < blockCount; ++i) boxMuller(c0[i], c1[i], a0[i], a1[i]);
	for (size_t i = 0; i < blockCount; ++i) boxMuller(c2[i], c3[i], a2[i], a3[i]);
	for (size_t i = 0; i < blockCount; ++i) {
		out[4 * i] = a0[i]; out[4 * i + 1] = a1[i]; out[4 * i + 2] = a2[i]; out[4 * i + 3] = a3[i];
	}
}

// out[k] = normal number first + k of a sequence whose block b has counter(b):
// value v is lane v % 4 of block v / 4.
template <class Counter>
inline void fillFastNormals(uint64_t first, size_t count, const uint32_t key[2], Counter counter, float* out) {
	float buffer[4 * kFastBatch];
	uint64_t block = first >> 2;
	size_t skip = size_t(first & 3u), done = 0;
	while (done < count) {
		const size_t needed = (skip + (count - done) + 3) / 4;
		if (needed == 1) { // within one block: scalar, only the pairs needed
			std::array<uint32_t, 4> c;
			counter(block, c);
			fastNormalLanes(c, key, unsigned(skip), unsigned(skip + count - done), out + done);
			return;
		}
		const size_t blocks = needed < kFastBatch ? needed : kFastBatch;
		fastNormalBlocks(block, blocks, key, counter, buffer);
		size_t take = 4 * blocks - skip;
		if (take > count - done) take = count - done;
		std::memcpy(out + done, buffer + skip, take * sizeof(float));
		done += take;
		block += blocks;
		skip = 0;
	}
}

} // namespace detail

// N(0,1) number `index` of (seed, realisation, stream), fast family by index:
// lane index % 4 of the Philox block (index / 4, realisation, stream).
inline float fastNormalByIndex(uint64_t seed, uint32_t realisation, uint64_t index, uint32_t stream = 0) {
	const uint64_t block = index >> 2;
	const unsigned lane = unsigned(index & 3u);
	float out;
	detail::fastNormalLanes({ uint32_t(block), uint32_t(block >> 32), realisation, stream }, detail::fastByIndexKey(seed).data(), lane, lane + 1, &out);
	return out;
}

// out[k] = fastNormalByIndex(seed, realisation, firstIndex + k, stream), k < count.
inline void fillFastNormalsByIndex(uint64_t seed, uint32_t realisation, uint64_t firstIndex, size_t count, uint32_t stream, float* out) {
	const std::array<uint32_t, 2> key = detail::fastByIndexKey(seed);
	detail::fillFastNormals(firstIndex, count, key.data(), [&](uint64_t block, std::array<uint32_t, 4>& c) {
		c = { uint32_t(block), uint32_t(block >> 32), realisation, stream };
	}, out);
}

// N(0,1) of realisation `realisation` at `index`, fast family by realisation:
// lane realisation % 4 of the Philox block (realisation / 4, index, stream).
inline float fastNormalByRealisation(uint64_t seed, uint32_t realisation, uint64_t index, uint32_t stream = 0) {
	const unsigned lane = realisation & 3u;
	float out;
	detail::fastNormalLanes({ realisation >> 2, uint32_t(index), uint32_t(index >> 32), stream }, detail::fastByRealisationKey(seed).data(), lane, lane + 1, &out);
	return out;
}

// out[r] = fastNormalByRealisation(seed, firstRealisation + r, index, stream), r < count
// (firstRealisation + count <= 2^32).
inline void fillFastNormalsByRealisation(uint64_t seed, uint32_t firstRealisation, size_t count, uint64_t index, uint32_t stream, float* out) {
	const std::array<uint32_t, 2> key = detail::fastByRealisationKey(seed);
	detail::fillFastNormals(firstRealisation, count, key.data(), [&](uint64_t block, std::array<uint32_t, 4>& c) {
		c = { uint32_t(block), uint32_t(index), uint32_t(index >> 32), stream };
	}, out);
}

} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_RANDOM_HPP
