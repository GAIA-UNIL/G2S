/*
 * Unit tests for the QS inner kernels:
 *  - g2s::complexAddAlphaxCxD (spectrum multiply-accumulate) against a scalar reference,
 *    for every size, so that SIMD paths and their scalar tails are both exercised;
 *  - the generic fKst::findKbiggest scans against the original element-by-element loop:
 *    same values, same positions and the same number of random draws.
 *
 * Build with different SIMD flags to cover each code path, e.g.
 *   make                                  # native (NEON on arm64, SSE2 on x86-64)
 *   make SIMD_FLAGS="-msse3"
 *   make SIMD_FLAGS="-mavx2 -mfma"
 */

#include <climits>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

#include "complexMulti.hpp"
#include "fKst.hpp"

static int failures = 0;

#define CHECK(cond, ...)                                  \
	do {                                                  \
		if (!(cond)) {                                    \
			failures++;                                   \
			fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
			fprintf(stderr, __VA_ARGS__);                 \
			fprintf(stderr, "\n");                        \
		}                                                 \
	} while (0)

template <typename T>
static void testComplexAddAlphaxCxD(const char* typeName) {
	std::mt19937 gen(42);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);
	for (unsigned size = 0; size < 68; ++size) {
		std::vector<T> C(2 * size + 2), D(2 * size + 2), dst(2 * size + 2), ref(2 * size + 2);
		for (size_t i = 0; i < C.size(); ++i) {
			C[i] = T(dist(gen));
			D[i] = T(dist(gen));
			dst[i] = ref[i] = T(dist(gen));
		}
		const T alpha = T(0.75);
		for (unsigned i = 0; i < size; ++i) {
			ref[2 * i + 0] += alpha * (C[2 * i + 0] * D[2 * i + 0] - C[2 * i + 1] * D[2 * i + 1]);
			ref[2 * i + 1] += alpha * (C[2 * i + 0] * D[2 * i + 1] + C[2 * i + 1] * D[2 * i + 0]);
		}
		g2s::complexAddAlphaxCxD(dst.data(), C.data(), D.data(), alpha, size);
		for (size_t i = 0; i < dst.size(); ++i) {
			CHECK(std::fabs(double(dst[i]) - double(ref[i])) < 1e-5, "complexAddAlphaxCxD<%s> size %u index %zu: %g vs %g",
			      typeName, size, i, double(dst[i]), double(ref[i]));
		}
	}
}

// The scans as they were before the block-skip optimisation.
template <typename T>
static void refFindKbiggest(const T* data, unsigned N, unsigned short k, T* output) {
	std::fill(output, output + k, -INFINITY);
	for (unsigned i = 0; i < N; ++i)
		if (data[i] > output[k - 1]) fKst::addValueB(data, N, k, output, i);
}

template <typename T>
static void refFindKbiggest(const T* data, unsigned N, unsigned short k, T* output, unsigned* position) {
	std::fill(output, output + k, -INFINITY);
	std::fill(position, position + k, UINT_MAX);
	for (unsigned i = 0; i < N; ++i)
		if (data[i] > output[k - 1]) fKst::addValueB(data, N, k, output, position, i);
}

template <typename T, typename G>
static void refFindKbiggest(const T* data, unsigned N, unsigned short k, T* output, unsigned* position, G generator) {
	std::fill(output, output + k, -INFINITY);
	std::fill(position, position + k, UINT_MAX);
	unsigned cpt = 0;
	for (unsigned i = 0; i < N; ++i)
		if (data[i] >= output[k - 1]) fKst::addValueB(data, N, k, output, position, generator, cpt, i);
}

static std::vector<float> makeData(unsigned N, int pattern, std::mt19937& gen) {
	std::uniform_real_distribution<float> dist(-100.f, 0.f);
	std::uniform_int_distribution<int> small(-6, 0);
	std::vector<float> data(N);
	for (unsigned i = 0; i < N; ++i) {
		switch (pattern) {
			case 0: data[i] = dist(gen); break;                                   // distinct values
			case 1: data[i] = float(small(gen)); break;                           // many ties
			case 2: data[i] = (i % 37) < 20 ? -INFINITY : dist(gen); break;       // masked bands
			case 3: data[i] = (i % 11 == 0) ? std::nanf("") : float(small(gen)); break; // NaN + ties
			default: data[i] = -float(N - i); break;                              // increasing: worst case
		}
	}
	return data;
}

static void testFindKbiggest() {
	std::mt19937 dataGen(7);
	const unsigned sizes[] = {0, 1, 15, 16, 17, 31, 33, 100, 1000, 4099};
	const unsigned short ks[] = {1, 2, 3, 5};
	for (unsigned N : sizes)
		for (unsigned short k : ks)
			for (int pattern = 0; pattern < 5; ++pattern) {
				std::vector<float> data = makeData(N, pattern, dataGen);
				std::vector<float> out(k), refOut(k);
				std::vector<unsigned> pos(k), refPos(k);

				std::fill(out.begin(), out.end(), -INFINITY);
				fKst::findKbiggest(data.data(), N, k, out.data());
				refFindKbiggest(data.data(), N, k, refOut.data());
				for (int j = 0; j < k; ++j)
					CHECK(out[j] == refOut[j] || (std::isnan(out[j]) && std::isnan(refOut[j])),
					      "findKbiggest N=%u k=%d pattern=%d", N, k, pattern);

				std::fill(out.begin(), out.end(), -INFINITY);
				std::fill(pos.begin(), pos.end(), UINT_MAX);
				fKst::findKbiggest(data.data(), N, k, out.data(), pos.data());
				refFindKbiggest(data.data(), N, k, refOut.data(), refPos.data());
				for (int j = 0; j < k; ++j)
					CHECK(out[j] == refOut[j] && pos[j] == refPos[j], "findKbiggest+position N=%u k=%d pattern=%d", N, k, pattern);

				std::mt19937 engine(1234), refEngine(1234);
				std::uniform_real_distribution<float> uniform(0.f, 1.f), refUniform(0.f, 1.f);
				auto rng = std::bind(uniform, std::ref(engine));
				auto refRng = std::bind(refUniform, std::ref(refEngine));
				std::fill(out.begin(), out.end(), -INFINITY);
				std::fill(pos.begin(), pos.end(), UINT_MAX);
				fKst::findKbiggest(data.data(), N, k, out.data(), pos.data(), rng);
				refFindKbiggest(data.data(), N, k, refOut.data(), refPos.data(), refRng);
				for (int j = 0; j < k; ++j)
					CHECK(out[j] == refOut[j] && pos[j] == refPos[j], "findKbiggest+generator N=%u k=%d pattern=%d", N, k, pattern);
				CHECK(engine() == refEngine(), "findKbiggest+generator N=%u k=%d pattern=%d: different number of draws", N, k, pattern);
			}
}

int main() {
	testComplexAddAlphaxCxD<float>("float");
	testComplexAddAlphaxCxD<double>("double");
	testFindKbiggest();
	if (failures) {
		fprintf(stderr, "%d check(s) failed\n", failures);
		return 1;
	}
	printf("qs kernels: all checks passed\n");
	return 0;
}
