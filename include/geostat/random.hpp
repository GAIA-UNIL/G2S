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

#ifndef G2S_GEOSTAT_RANDOM_HPP
#define G2S_GEOSTAT_RANDOM_HPP

#include <array>
#include <cstdint>

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

} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_RANDOM_HPP
