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

// Sequential Gaussian simulation engine in two phases (Nunes & Almeida 2010;
// Schoedl, MacKie, ..., Gravey 2023): neighbour search and kriging weights
// depend only on the geometry, the path and the covariance model, never on
// values, so they are computed ahead in parallel (plan, planSgs) into a
// WeightTable (weightTable.hpp); the simulation then only applies the stored
// weights (replay, replaySgs) to any number of realisations at once, which is
// the constant-path idea of Nussbaumer et al. (2018).
//
// Nothing here knows about grids or images: nodes are ids into flat arrays,
// neighbours come from a NeighborSearch, and an Addressing object turns a
// neighbour into a 32-bit reference and back:
//   unsigned ndim() const;
//   size_t   referenceLimit() const;      // upper bound on neighbours per node
//   uint32_t encode(size_t target, size_t neighbor, const double* lag, uint32_t searchRef) const;
//                                          // searchRef: NeighborSet::ref() of the neighbour;
//                                          // kNoReference when not representable
//   void     decode(size_t target, const uint32_t* refs, size_t count, size_t* ids) const;
//   void     lag(size_t target, uint32_t ref, double* out) const; // neighbour - target
//   bool     shareable() const;           // equal references mean equal lags
//   bool     integerLags() const;         // every lag has integer components
//   std::vector<int32_t> referenceTable() const; // stored in the weight file
// GridAddressing (gridAddressing.hpp) uses kernel-offset ids, so that every
// node with the same neighbour pattern and covariance shares one stencil;
// PointAddressing below uses node ids (no sharing) for scattered nodes.
//
// Plan (planSgs), for path positions i = 0 .. L-1 (rank of node path[i] is i):
//   1. search (parallel): the nearest maxNeighbors nodes with rank < i
//      (hard data have rank -1), as references, and a signature hash of
//      (resolved covariance key, references);
//   2. levels (sequential, O(L n)): level = 1 + max level of the simulated
//      neighbours, 0 without any; entries are then counting-sorted by level
//      (stable, so path order inside a level);
//   3. sharing (parallel over signature shards, hash maps): an entry reuses
//      the stencil of an earlier entry with the same references and the same
//      resolved model (ResolvedModel::key() then sameParameters()); without
//      sharing, one stencil per entry;
//   4. solve (parallel over stencils): kriging (kriging.hpp) in double,
//      stored as float32 weights and standard deviation. With a stationary
//      model and integer lags (grids), the covariance is tabulated once over
//      the box of lag differences and the systems are assembled by lookup
//      (same values as ResolvedModel::covarianceMatrix, bit for bit).
// The result depends on the inputs only, not on the number of threads.
//
// Replay (replaySgs): values are float, [node x R] (realisation fastest),
// residuals for simple kriging (the front end subtracts the mean beforehand
// and adds it back), raw values for ordinary kriging. Hard-data rows must be
// filled; simulated rows are overwritten. For each entry,
//   v[t, r] = sum_j w_j v[n_j, r] + sigma * fastNormalByRealisation(seed, first + r, t)
//             (the isolated mean instead of the sum when the entry has no
//             neighbour); the noise term depends on no value and is written
//             first, in a fully parallel pass
// The weights of an entry are read once for all R realisations (the inner
// loop over r is contiguous and vectorised). Threads take consecutive chunks
// of the level-ordered entries and wait, per neighbour, only for nodes that
// are still being computed (see replaySgs). The random
// numbers are counter-based (random.hpp), indexed by the target node, so the
// result is the same for any number of threads and any batching of
// realisations (realisation r of a batch starting at `first` is realisation
// first + r).

#ifndef G2S_GEOSTAT_SGS_ENGINE_HPP
#define G2S_GEOSTAT_SGS_ENGINE_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "geostat/covarianceModel.hpp"
#include "geostat/kriging.hpp"
#include "geostat/neighborSearch.hpp"
#include "geostat/random.hpp"
#include "geostat/weightTable.hpp"

namespace g2s {
namespace geostat {

// ============================================================ threads

// f(begin, end, worker) over [0, count) in chunks of `grain`, dynamically
// scheduled over `threads` workers (the calling thread is worker 0).
template <class F>
inline void parallelFor(size_t count, unsigned threads, size_t grain, F f) {
	if (grain == 0) grain = 1;
	if (threads <= 1 || count <= grain) { if (count > 0) f(size_t(0), count, 0u); return; }
	threads = unsigned(std::min<size_t>(threads, (count + grain - 1) / grain));
	std::atomic<size_t> next(0);
	auto work = [&](unsigned worker) {
		for (;;) {
			const size_t begin = next.fetch_add(grain);
			if (begin >= count) break;
			f(begin, std::min(count, begin + grain), worker);
		}
	};
	std::vector<std::thread> pool;
	for (unsigned w = 1; w < threads; ++w) pool.emplace_back(work, w);
	work(0);
	for (std::thread& t : pool) t.join();
}

// Hint to the core that we are spinning.
inline void cpuRelax() {
#if defined(__aarch64__) || defined(_M_ARM64)
	__asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
	__asm__ __volatile__("pause");
#endif
}

inline double millisecondsSince(std::chrono::steady_clock::time_point start) {
	return 1e-6 * double(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

// ============================================================ paths and seeds

// Seed of path p: independent noise and path for every path index.
inline uint64_t pathSeed(uint64_t seed, uint64_t pathIndex) {
	uint64_t z = seed + 0x9E3779B97F4A7C15ull * (pathIndex + 1);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
	return z ^ (z >> 31);
}

// Fisher-Yates shuffle driven by uniform(seed, 0, i, stream 1): the same
// permutation on every platform and for any number of threads.
inline void shufflePath(std::vector<uint64_t>& nodes, uint64_t seed) {
	for (size_t i = nodes.size(); i > 1; --i) {
		size_t j = size_t(uniform(seed, 0, i - 1, 1) * double(i));
		if (j >= i) j = i - 1;
		std::swap(nodes[i - 1], nodes[j]);
	}
}

inline uint64_t pathHash(const uint64_t* path, size_t length) {
	Hasher h;
	h.u64(length);
	h.bytes(path, length * sizeof(uint64_t));
	return h.value();
}

// ============================================================ point addressing

// References are node ids; lags from explicit coordinates (nodeCount x ndim).
// For scattered nodes, where neighbour patterns do not repeat.
class PointAddressing {
public:
	PointAddressing(const double* coordinates, size_t nodeCount, unsigned ndim) : _coordinates(coordinates), _nodeCount(nodeCount), _ndim(ndim) {}
	unsigned ndim() const { return _ndim; }
	size_t referenceLimit() const { return _nodeCount; }
	uint32_t encode(size_t, size_t neighbor, const double*, uint32_t) const { return neighbor < size_t(kNoReference) ? uint32_t(neighbor) : kNoReference; }
	void decode(size_t, const uint32_t* refs, size_t count, size_t* ids) const { for (size_t i = 0; i < count; ++i) ids[i] = refs[i]; }
	void lag(size_t target, uint32_t ref, double* out) const {
		for (unsigned d = 0; d < _ndim; ++d) out[d] = _coordinates[size_t(ref) * _ndim + d] - _coordinates[target * _ndim + d];
	}
	bool shareable() const { return false; }
	bool integerLags() const { return false; }
	std::vector<int32_t> referenceTable() const { return std::vector<int32_t>(); }
private:
	const double* _coordinates;
	size_t _nodeCount;
	unsigned _ndim;
};

// ============================================================ plan

struct SgsPlanOptions {
	KrigingType type = KrigingType::Simple;
	size_t maxNeighbors = 32;
	bool shareStencils = true;
	unsigned threads = 1;
};

struct SgsPlanStats {
	size_t entries = 0, stencils = 0, neighbors = 0, levels = 0, largestLevel = 0;
	size_t isolated = 0;          // entries without any neighbour
	size_t jittered = 0, failed = 0; // stencils whose system needed a jitter / could not be solved
	double searchMs = 0, levelMs = 0, shareMs = 0, solveMs = 0, totalMs = 0;
	// Solved systems per simulated node: 1 without sharing.
	double sharingRatio() const { return entries ? double(stencils) / double(entries) : 1.0; }
};

// Builds the plan of `path` (pathLength node ids; ranks from pathRanks with
// the same path). table.inputHash and table.geometry are left to the caller.
// Returns false (and sets error) when a neighbour cannot be addressed.
template <class Addressing>
inline bool planSgs(const NeighborSearch& search, const Addressing& addressing, const CovarianceModelSet& models,
                    const int64_t* ranks, const uint64_t* path, size_t pathLength, const SgsPlanOptions& options,
                    WeightTable& table, SgsPlanStats& stats, std::string& error) {
	const auto start = std::chrono::steady_clock::now();
	stats = SgsPlanStats();
	const unsigned threads = std::max(1u, options.threads);
	const unsigned ndim = addressing.ndim();
	const size_t stride = std::max<size_t>(1, std::min(options.maxNeighbors, addressing.referenceLimit()));
	const bool stationary = models.isStationary();
	const size_t L = pathLength;

	table = WeightTable();
	table.krigingType = options.type == KrigingType::Ordinary ? 1u : 0u;
	table.ndim = ndim;
	table.nodeCount = search.nodeCount();
	table.pathHash = pathHash(path, L);
	table.refTable = addressing.referenceTable();
	const bool share = options.shareStencils && addressing.shareable();
	if (share) table.flags |= WeightTableShared;

	// ---- 1. search
	std::vector<uint32_t> refs(L * stride);
	std::vector<uint32_t> counts(L);
	std::vector<uint64_t> signature(L);
	std::atomic<bool> unaddressable(false);
	{
		std::vector<NeighborSet> sets(threads);
		std::vector<ResolvedModel> scratch(threads);
		parallelFor(L, threads, 256, [&](size_t begin, size_t end, unsigned worker) {
			NeighborSet& set = sets[worker];
			for (size_t i = begin; i < end; ++i) {
				const size_t node = size_t(path[i]);
				search.find(node, int64_t(i), options.maxNeighbors, set);
				const size_t count = std::min(set.size(), stride);
				uint32_t* row = refs.data() + i * stride;
				Hasher h;
				h.u64(stationary ? 0 : models.at(node, scratch[worker]).key());
				h.u64(count);
				for (size_t j = 0; j < count; ++j) {
					row[j] = addressing.encode(node, set.ids[j], set.lag(j), set.ref(j));
					if (row[j] == kNoReference) unaddressable = true;
				}
				h.bytes(row, count * sizeof(uint32_t));
				counts[i] = uint32_t(count);
				signature[i] = h.value();
			}
		});
	}
	if (unaddressable) { error = "a neighbour could not be expressed as a reference (kernel offset)"; return false; }
	stats.searchMs = millisecondsSince(start);

	// ---- 2. levels (sequential: no thread ever waits for another, which
	// keeps the plan fast on a busy machine), then the level order of the
	// entries (counting sort, stable)
	auto t1 = std::chrono::steady_clock::now();
	std::vector<uint32_t> levelOf(L);
	uint32_t maxLevel = 0;
	{
		std::vector<size_t> ids(stride);
		for (size_t i = 0; i < L; ++i) {
			addressing.decode(size_t(path[i]), refs.data() + i * stride, counts[i], ids.data());
			uint32_t value = 0;
			for (size_t j = 0; j < counts[i]; ++j) {
				const int64_t rank = ranks[ids[j]];
				if (rank >= 0) value = std::max(value, levelOf[size_t(rank)] + 1);
			}
			levelOf[i] = value;
			maxLevel = std::max(maxLevel, value);
			if (counts[i] == 0) ++stats.isolated;
		}
	}
	const size_t levelCount = L ? size_t(maxLevel) + 1 : 0;
	table.levelStart.assign(levelCount + 1, 0);
	for (size_t i = 0; i < L; ++i) ++table.levelStart[levelOf[i] + 1];
	for (size_t l = 0; l < levelCount; ++l) {
		stats.largestLevel = std::max<size_t>(stats.largestLevel, table.levelStart[l + 1]);
		table.levelStart[l + 1] += table.levelStart[l];
	}
	std::vector<uint64_t> order(L); // entry -> path position
	{
		std::vector<uint64_t> cursor(table.levelStart.begin(), table.levelStart.end() - 1);
		for (size_t i = 0; i < L; ++i) order[cursor[levelOf[i]]++] = i;
	}
	std::vector<uint32_t>().swap(levelOf);
	stats.levelMs = millisecondsSince(t1);

	// ---- 3. stencils: share identical systems. Entries are split into
	// shards by signature; inside a shard (in entry order) an entry adopts the
	// first earlier entry with the same references and model. Stencil ids are
	// then numbered in entry order, so the result does not depend on threads.
	auto t2 = std::chrono::steady_clock::now();
	table.entryTarget.resize(L);
	table.entryStencil.resize(L);
	std::vector<uint64_t> representative; // stencil -> path position solved for it
	for (size_t e = 0; e < L; ++e) table.entryTarget[e] = path[size_t(order[e])];
	if (share) {
		const size_t shardCount = size_t(threads) * 8;
		std::vector<uint64_t> shardStart(shardCount + 1, 0), shardEntries(L);
		auto shardOf = [&](size_t e) { return size_t((signature[size_t(order[e])] >> 32) % shardCount); };
		for (size_t e = 0; e < L; ++e) ++shardStart[shardOf(e) + 1];
		for (size_t k = 0; k < shardCount; ++k) shardStart[k + 1] += shardStart[k];
		{
			std::vector<uint64_t> cursor(shardStart.begin(), shardStart.end() - 1);
			for (size_t e = 0; e < L; ++e) shardEntries[cursor[shardOf(e)]++] = e;
		}
		std::vector<uint64_t> owner(L); // entry -> first entry with the same system
		std::vector<ResolvedModel> scratchA(threads), scratchB(threads);
		parallelFor(shardCount, threads, 1, [&](size_t begin, size_t end, unsigned worker) {
			for (size_t k = begin; k < end; ++k) {
				std::unordered_map<uint64_t, uint32_t> first; // signature -> first candidate
				std::vector<std::pair<uint64_t, uint32_t> > candidates; // (entry, next candidate with the signature)
				first.reserve(size_t(shardStart[k + 1] - shardStart[k]));
				for (uint64_t q = shardStart[k]; q < shardStart[k + 1]; ++q) {
					const size_t e = size_t(shardEntries[q]);
					const size_t i = size_t(order[e]);
					const uint32_t* row = refs.data() + i * stride;
					uint64_t found = e;
					uint32_t last = kNoReference;
					auto it = first.find(signature[i]);
					if (it != first.end()) {
						for (uint32_t c = it->second; c != kNoReference; c = candidates[c].second) {
							last = c;
							const size_t r = size_t(order[size_t(candidates[c].first)]);
							if (counts[r] != counts[i] || std::memcmp(refs.data() + r * stride, row, counts[i] * sizeof(uint32_t)) != 0) continue;
							if (!stationary) {
								const ResolvedModel& ma = models.at(size_t(path[r]), scratchA[worker]);
								const ResolvedModel& mb = models.at(size_t(path[i]), scratchB[worker]);
								if (ma.key() != mb.key() || !ma.sameParameters(mb)) continue;
							}
							found = candidates[c].first;
							break;
						}
					}
					if (found == e) {
						candidates.push_back({ e, kNoReference });
						const uint32_t index = uint32_t(candidates.size() - 1);
						if (last == kNoReference) first.emplace(signature[i], index);
						else candidates[last].second = index;
					}
					owner[e] = found;
				}
			}
		});
		for (size_t e = 0; e < L; ++e) {
			if (owner[e] == e) {
				if (representative.size() >= size_t(kNoReference)) { error = "too many distinct stencils for 32-bit stencil ids"; return false; }
				table.entryStencil[e] = uint32_t(representative.size());
				representative.push_back(order[e]);
			} else {
				table.entryStencil[e] = table.entryStencil[size_t(owner[e])]; // owner[e] < e
			}
		}
	} else {
		if (L >= size_t(kNoReference)) { error = "too many nodes for 32-bit stencil ids"; return false; }
		representative.assign(order.begin(), order.end());
		for (size_t e = 0; e < L; ++e) table.entryStencil[e] = uint32_t(e);
	}
	std::vector<uint64_t>().swap(signature);
	const size_t S = representative.size();
	table.stencilStart.assign(S + 1, 0);
	for (size_t s = 0; s < S; ++s) table.stencilStart[s + 1] = table.stencilStart[s] + counts[size_t(representative[s])];
	stats.shareMs = millisecondsSince(t2);

	// ---- 4. solve
	auto t3 = std::chrono::steady_clock::now();
	const size_t N = size_t(table.stencilStart[S]);
	table.neighborRef.resize(N);
	table.neighborWeight.resize(N);
	table.stencilSigma.resize(S);
	std::vector<size_t> jittered(threads, 0), failed(threads, 0);
	// Covariance table over the lag differences (stationary model, integer lags).
	std::vector<double> covarianceTable;
	std::vector<long> boxHalf(ndim, 0), boxStride(ndim, 0);
	if (stationary && addressing.integerLags() && ndim <= kMaxDims && S > 0) {
		std::vector<double> lag(ndim);
		for (size_t r = 0; r < addressing.referenceLimit(); ++r) {
			addressing.lag(0, uint32_t(r), lag.data());
			for (unsigned d = 0; d < ndim; ++d) boxHalf[d] = std::max(boxHalf[d], 2 * long(std::lround(std::fabs(lag[d]))));
		}
		size_t size = 1;
		for (unsigned d = 0; d < ndim; ++d) { boxStride[d] = long(size); size *= size_t(2 * boxHalf[d] + 1); }
		if (size <= (size_t(1) << 24) && size < 8 * S * stride * stride) {
			covarianceTable.resize(size);
			const ResolvedModel& model = models.stationaryModel();
			parallelFor(size, threads, 4096, [&](size_t begin, size_t end, unsigned) {
				double l[kMaxDims];
				for (size_t index = begin; index < end; ++index) {
					size_t rest = index;
					for (unsigned d = 0; d < ndim; ++d) {
						const size_t extent = size_t(2 * boxHalf[d] + 1);
						l[d] = double(long(rest % extent) - boxHalf[d]);
						rest /= extent;
					}
					covarianceTable[index] = model.covariance(l);
				}
			});
		}
	}
	const bool tabulated = !covarianceTable.empty();
	long centre = 0;
	for (unsigned d = 0; d < ndim && tabulated; ++d) centre += boxHalf[d] * boxStride[d];
	{
		std::vector<KrigingWorkspace> workspaces(threads);
		std::vector<KrigingSolution> solutions(threads);
		std::vector<ResolvedModel> scratch(threads);
		std::vector<std::vector<double> > lags(threads, std::vector<double>(stride * ndim));
		std::vector<std::vector<long> > cells(threads, std::vector<long>(stride));
		parallelFor(S, threads, 64, [&](size_t begin, size_t end, unsigned worker) {
			for (size_t s = begin; s < end; ++s) {
				const size_t i = size_t(representative[s]);
				const size_t node = size_t(path[i]);
				const size_t count = counts[i];
				const uint32_t* row = refs.data() + i * stride;
				double* lag = lags[worker].data();
				for (size_t j = 0; j < count; ++j) addressing.lag(node, row[j], lag + j * ndim);
				KrigingSolution& solution = solutions[worker];
				KrigingWorkspace& workspace = workspaces[worker];
				unsigned status;
				if (tabulated) {
					// Lag j as an offset in the table, so C(lag_i - lag_j) = table[centre + cell_i - cell_j].
					long* cell = cells[worker].data();
					for (size_t j = 0; j < count; ++j) {
						long c = 0;
						for (unsigned d = 0; d < ndim; ++d) c += long(std::lround(lag[j * ndim + d])) * boxStride[d];
						cell[j] = c;
					}
					workspace.matrix.resize(count * count);
					workspace.rhs.resize(count);
					double* matrix = workspace.matrix.data();
					for (size_t a = 0; a < count; ++a) {
						workspace.rhs[a] = covarianceTable[size_t(centre + cell[a])];
						for (size_t b = 0; b <= a; ++b) {
							const double c = covarianceTable[size_t(centre + cell[a] - cell[b])];
							matrix[a * count + b] = c;
							matrix[b * count + a] = c;
						}
					}
					status = solveKrigingSystem(options.type, matrix, count, workspace.rhs.data(), models.stationaryModel().sill(), count, workspace, solution);
				} else {
					status = krige(options.type, models.at(node, scratch[worker]), lag, count, workspace, solution);
				}
				if (status & KrigingJittered) ++jittered[worker];
				if (status & KrigingFailed) ++failed[worker];
				const size_t offset = size_t(table.stencilStart[s]);
				for (size_t j = 0; j < count; ++j) {
					table.neighborRef[offset + j] = row[j];
					table.neighborWeight[offset + j] = float(solution.weights[j]);
				}
				table.stencilSigma[s] = float(std::sqrt(std::max(0.0, solution.variance)));
			}
		});
	}
	for (unsigned w = 0; w < threads; ++w) { stats.jittered += jittered[w]; stats.failed += failed[w]; }
	stats.solveMs = millisecondsSince(t3);

	stats.entries = L;
	stats.stencils = S;
	stats.neighbors = 0;
	for (size_t i = 0; i < L; ++i) stats.neighbors += counts[i];
	stats.levels = levelCount;
	stats.totalMs = millisecondsSince(start);
	return true;
}

// ============================================================ replay

struct SgsReplayOptions {
	uint64_t seed = 0;
	uint32_t firstRealisation = 0;   // realisation r of this batch draws the noise of first + r
	unsigned realisations = 1;       // R: values are [node x R]
	unsigned threads = 1;
	// Mean of entries without any neighbour (ordinary kriging falls back to
	// it); per node when isolatedMeanMap is set, otherwise isolatedMean.
	// Leave at 0 for simple kriging on residuals.
	const float* isolatedMeanMap = nullptr;
	double isolatedMean = 0.0;
};

struct SgsReplayStats {
	unsigned threads = 1;
	size_t waits = 0;  // neighbours being computed by another worker when an entry needed them
	size_t stolen = 0; // pending neighbours computed by the worker that needed them
	double noiseMs = 0, totalMs = 0; // noise pass, and both passes

};

namespace detail {

// The two passes of the replay. noise() writes sigma * eps (plus the mean of
// isolated entries) into every target row; it depends on no value, so it runs
// fully in parallel. compute() then adds the weighted neighbours, in level
// order (serially, or through ParallelApply). Both passes use the same float
// operations whatever R, the batch or the thread count.
template <class Addressing>
struct ReplayKernel {
	const WeightTableView& table;
	const Addressing& addressing;
	const SgsReplayOptions& options;
	float* values;

	void noise(uint64_t begin, uint64_t end) const {
		const size_t R = options.realisations;
		for (uint64_t e = begin; e < end; ++e) {
			const size_t target = size_t(table.entryTarget[e]);
			const uint32_t s = table.entryStencil[e];
			const float sigma = table.stencilSigma[s];
			const bool isolated = table.stencilStart[s + 1] == table.stencilStart[s];
			const float mean = isolated ? float(options.isolatedMeanMap ? double(options.isolatedMeanMap[target]) : options.isolatedMean) : 0.f;
			float* out = values + target * R;
			fillFastNormalsByRealisation(options.seed, options.firstRealisation, R, target, 0, out);
			for (size_t r = 0; r < R; ++r) {
				const float deviation = sigma * out[r];
				out[r] = isolated ? mean + deviation : deviation;
			}
		}
	}

	// Weighted neighbours of entry e added to its row (neighbours final).
	void compute(uint64_t e, const size_t* ids, size_t count, float* acc) const {
		const size_t R = options.realisations;
		const size_t target = size_t(table.entryTarget[e]);
		const float* weight = table.neighborWeight + table.stencilStart[table.entryStencil[e]];
		float* out = values + target * R;
		if (count == 0) return;
		// Products and sums are separate statements so that the compiler
		// cannot fuse them (fp-contract=on): whether a loop is vectorised
		// (an ordered reduction is not fused) depends on where it is inlined,
		// and the result must not.
		if (R == 1) {
			float sum = 0.f;
			for (size_t j = 0; j < count; ++j) {
				const float product = weight[j] * values[ids[j]];
				sum += product;
			}
			out[0] = sum + out[0];
			return;
		}
		for (size_t r = 0; r < R; ++r) acc[r] = 0.f;
		for (size_t j = 0; j < count; ++j) {
			const float w = weight[j];
			const float* v = values + ids[j] * R;
			for (size_t r = 0; r < R; ++r) {
				const float product = w * v[r];
				acc[r] += product;
			}
		}
		for (size_t r = 0; r < R; ++r) out[r] = acc[r] + out[r];
	}

	size_t decode(uint64_t e, std::vector<size_t>& ids) const {
		const uint32_t s = table.entryStencil[e];
		const uint64_t first = table.stencilStart[s];
		const size_t count = size_t(table.stencilStart[s + 1] - first);
		if (ids.size() < count) ids.resize(count);
		addressing.decode(size_t(table.entryTarget[e]), table.neighborRef + first, count, ids.data());
		return count;
	}

	// Serial pass, in table order.
	void applyAll(std::vector<size_t>& ids, std::vector<float>& acc) const {
		for (uint64_t e = 0; e < table.entryCount; ++e) {
			const size_t count = decode(e, ids); // resizes ids: before taking ids.data()
			compute(e, ids.data(), count, acc.data());
		}
	}
};

// Parallel dependent pass. Node states: 0 pending, 1 being computed, 2 final
// (nodes outside the table are final from the start). A worker claims an
// entry (0 -> 1) before computing it; when a neighbour is still pending, the
// worker claims and computes that neighbour itself (up to kMaxDepth nested
// entries) instead of waiting for the worker whose chunk holds it. So a
// worker that the system preempts delays others only through the one entry
// it is computing. Waiting always terminates: neighbours come earlier in the
// level order, which is the order in which entries are taken.
template <class Addressing>
struct ParallelApply {
	static constexpr unsigned kMaxDepth = 32;
	const ReplayKernel<Addressing>& kernel;
	std::atomic<uint8_t>* state;
	const uint64_t* entryOf; // node -> entry (valid for table targets)

	struct Scratch {
		std::vector<std::vector<size_t> > ids;
		std::vector<std::vector<float> > acc;
		size_t waits = 0, stolen = 0;
		Scratch(size_t R) : ids(kMaxDepth + 1), acc(kMaxDepth + 1, std::vector<float>(R)) {}
	};

	// Computes entry e when it can be claimed.
	void run(uint64_t e, unsigned depth, Scratch& scratch) const {
		const size_t target = size_t(kernel.table.entryTarget[e]);
		uint8_t expected = 0;
		if (!state[target].compare_exchange_strong(expected, 1, std::memory_order_acquire)) return;
		std::vector<size_t>& ids = scratch.ids[depth];
		const size_t count = kernel.decode(e, ids);
		for (size_t j = 0; j < count; ++j) {
			const size_t id = ids[j];
			uint8_t current = state[id].load(std::memory_order_acquire);
			if (current == 2) continue;
			if (current == 0 && depth < kMaxDepth) {
				run(entryOf[id], depth + 1, scratch);
				++scratch.stolen;
				current = state[id].load(std::memory_order_acquire);
				if (current == 2) continue;
			}
			++scratch.waits;
			for (unsigned spin = 0; state[id].load(std::memory_order_acquire) != 2; ++spin) {
				if (spin < 4096) cpuRelax();
				else std::this_thread::yield();
			}
		}
		kernel.compute(e, ids.data(), count, scratch.acc[depth].data());
		state[target].store(2, std::memory_order_release);
	}
};

} // namespace detail

// Applies the table to `values` ([nodeCount x R] floats, see the top of the
// file). The noise pass is split over the threads freely. In the dependent
// pass, workers take consecutive chunks of the level-ordered entries and
// synchronise per node (ParallelApply), without any barrier between levels.
template <class Addressing>
inline void replaySgs(const WeightTableView& table, const Addressing& addressing, float* values, const SgsReplayOptions& options,
                      SgsReplayStats* stats = nullptr) {
	const auto start = std::chrono::steady_clock::now();
	const size_t R = std::max(1u, options.realisations);
	SgsReplayOptions local = options;
	local.realisations = unsigned(R);
	const unsigned threads = unsigned(std::min<uint64_t>(std::max(1u, options.threads), std::max<uint64_t>(1, table.entryCount / 64)));
	SgsReplayStats result;
	result.threads = threads;

	const detail::ReplayKernel<Addressing> kernel{ table, addressing, local, values };
	if (threads == 1) {
		std::vector<size_t> ids;
		std::vector<float> acc(R);
		kernel.noise(0, table.entryCount);
		result.noiseMs = millisecondsSince(start);
		kernel.applyAll(ids, acc);
	} else {
		std::vector<std::atomic<uint8_t> > state(size_t(table.nodeCount));
		std::vector<uint64_t> entryOf(size_t(table.nodeCount));
		parallelFor(size_t(table.nodeCount), threads, size_t(1) << 16, [&](size_t b, size_t e, unsigned) {
			for (size_t i = b; i < e; ++i) state[i].store(2, std::memory_order_relaxed);
		});
		parallelFor(size_t(table.entryCount), threads, 512, [&](size_t b, size_t e, unsigned) {
			for (size_t i = b; i < e; ++i) {
				state[size_t(table.entryTarget[i])].store(0, std::memory_order_relaxed);
				entryOf[size_t(table.entryTarget[i])] = i;
			}
			kernel.noise(b, e);
		});
		result.noiseMs = millisecondsSince(start);
		const detail::ParallelApply<Addressing> apply{ kernel, state.data(), entryOf.data() };
		const uint64_t grain = 16;
		std::atomic<uint64_t> cursor(0);
		std::atomic<size_t> waits(0), stolen(0);
		auto work = [&](unsigned) {
			typename detail::ParallelApply<Addressing>::Scratch scratch(R);
			for (;;) {
				const uint64_t b = cursor.fetch_add(grain, std::memory_order_relaxed);
				if (b >= table.entryCount) break;
				const uint64_t e = std::min<uint64_t>(table.entryCount, b + grain);
				for (uint64_t i = b; i < e; ++i) apply.run(i, 0, scratch);
			}
			waits += scratch.waits;
			stolen += scratch.stolen;
		};
		std::vector<std::thread> pool;
		for (unsigned w = 1; w < threads; ++w) pool.emplace_back(work, w);
		work(0);
		for (std::thread& t : pool) t.join();
		result.waits = waits;
		result.stolen = stolen;
	}
	result.totalMs = millisecondsSince(start);
	if (stats) *stats = result;
}

} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_SGS_ENGINE_HPP
