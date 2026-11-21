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

// Moving-neighbourhood (local) kriging: at each target, the nearest data given
// by a NeighborSearch (grid, k-d tree, ...) and one small SK / OK system
// (kriging.hpp) with the covariance model resolved at the target (local
// stationarity, so spatially varying models are supported).
//
// Written against node ids only: targets are node ids, values[] and mean() are
// indexed by node id, and the model set resolves parameters by node id. On a
// grid the node id is the flat cell index; a point front end numbers its
// points instead.
//
// Sharing: the weights depend only on the neighbour lags and the resolved
// model, so targets with the same lag pattern (regular sampling, borders,
// repeated configurations) reuse one solved system. Each thread keeps a cache
// keyed by a hash of the lags combined with ResolvedModel::key(), confirmed by
// comparing the lags and ResolvedModel::sameParameters(). A cached solution is
// bit-identical to a fresh one, so the results do not depend on the number of
// threads or on the cache (only the statistics do).
//
//   SK: Z*(x0) = m(x0) + sum_i w_i (z_i - m(x_i)),  variance c0 - w.k0
//   OK: Z*(x0) = sum_i w_i z_i,                     variance c0 - w.k0 - mu
// A target without neighbour (or whose system cannot be solved) gets m(x0)
// and the sill as variance, for SK and OK alike (as kriging.hpp).

#ifndef G2S_GEOSTAT_KRIGING_LOCAL_HPP
#define G2S_GEOSTAT_KRIGING_LOCAL_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "geostat/covarianceModel.hpp"
#include "geostat/denseSolver.hpp"
#include "geostat/kriging.hpp"
#include "geostat/neighborSearch.hpp"

namespace g2s {
namespace geostat {

struct LocalKrigingOptions {
	KrigingType type = KrigingType::Ordinary;
	size_t maxNeighbors = 50;      // -n
	int64_t rankLimit = 0;         // neighbours are nodes with rank < rankLimit (0: data only)
	bool computeVariance = true;   // the variance costs almost nothing here
	size_t cacheCapacity = 8192;   // solved systems kept per thread; 0 disables sharing
	unsigned nbThreads = 1;
};

struct LocalKrigingStats {
	size_t targets = 0;     // targets processed
	size_t solved = 0;      // systems factorised
	size_t shared = 0;      // targets that reused a solved system
	size_t noNeighbor = 0;  // targets without neighbour
	size_t jittered = 0;    // solved systems that needed a diagonal jitter
	size_t failed = 0;      // solved systems that could not be factorised
	size_t neighbors = 0;   // total neighbours used
	// Fraction of the targets with neighbours that reused a solved system.
	double sharingRatio() const { return targets > noNeighbor ? double(shared) / double(targets - noNeighbor) : 0.0; }
};

namespace detail {

class LocalSystemCache {
public:
	struct Entry {
		std::vector<double> lags;
		ResolvedModel model;
		KrigingSolution solution;
	};

	static uint64_t hashLags(const double* lags, size_t size, uint64_t seed) {
		uint64_t hash = 1469598103934665603ull ^ seed;
		for (size_t i = 0; i < size; ++i) {
			double v = lags[i];
			if (v == 0.0) v = 0.0; // -0 == +0
			uint64_t bits;
			std::memcpy(&bits, &v, sizeof(bits));
			hash ^= bits + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
		}
		return hash * 1099511628211ull ^ uint64_t(size);
	}

	const KrigingSolution* find(uint64_t hash, const double* lags, size_t size, const ResolvedModel& model) const {
		auto range = _index.equal_range(hash);
		for (auto it = range.first; it != range.second; ++it) {
			const Entry& e = _entries[it->second];
			if (e.lags.size() == size && std::equal(e.lags.begin(), e.lags.end(), lags) && e.model.sameParameters(model)) return &e.solution;
		}
		return nullptr;
	}

	void insert(uint64_t hash, const double* lags, size_t size, const ResolvedModel& model, const KrigingSolution& solution, size_t capacity) {
		if (capacity == 0) return;
		if (_entries.size() >= capacity) { _entries.clear(); _index.clear(); }
		_entries.push_back(Entry{ std::vector<double>(lags, lags + size), model, solution });
		_index.emplace(hash, _entries.size() - 1);
	}

private:
	std::vector<Entry> _entries;
	std::unordered_multimap<uint64_t, size_t> _index;
};

} // namespace detail

// Kriging at targets[0 .. targetCount) from the neighbours found by `search`.
// values: indexed by node id, read at neighbour nodes only. mean: callable
// double(size_t node), the SK mean (OK uses it only where a target has no
// neighbour). estimate[t] and, when not null and options.computeVariance,
// variance[t] receive the results for targets[t].
template <class Mean>
inline LocalKrigingStats localKriging(const NeighborSearch& search, const CovarianceModelSet& models, const size_t* targets, size_t targetCount,
                                      const double* values, Mean mean, const LocalKrigingOptions& options, double* estimate, double* variance) {
	const unsigned threads = std::max(1u, options.nbThreads);
	struct ThreadState {
		NeighborSet neighbors;
		KrigingWorkspace workspace;
		KrigingSolution solution;
		ResolvedModel scratch;
		detail::LocalSystemCache cache;
		LocalKrigingStats stats;
	};
	std::vector<ThreadState> states(threads);
	const bool simple = options.type == KrigingType::Simple;
	const size_t block = 256;
	parallelFor((targetCount + block - 1) / block, threads, [&](size_t b, unsigned thread) {
		ThreadState& s = states[thread];
		const size_t t1 = std::min(targetCount, (b + 1) * block);
		for (size_t t = b * block; t < t1; ++t) {
			const size_t node = targets[t];
			++s.stats.targets;
			search.find(node, options.rankLimit, options.maxNeighbors, s.neighbors);
			const ResolvedModel& model = models.at(node, s.scratch);
			const size_t count = s.neighbors.size();
			const double m0 = mean(node);
			if (count == 0) {
				++s.stats.noNeighbor;
				estimate[t] = m0;
				if (variance && options.computeVariance) variance[t] = std::max(model.sill(), 0.0);
				continue;
			}
			s.stats.neighbors += count;
			const double* lags = s.neighbors.lags.data();
			const size_t lagSize = s.neighbors.lags.size();
			const KrigingSolution* solution = nullptr;
			uint64_t hash = 0;
			if (options.cacheCapacity > 0) {
				hash = detail::LocalSystemCache::hashLags(lags, lagSize, model.key());
				solution = s.cache.find(hash, lags, lagSize, model);
			}
			if (solution) {
				++s.stats.shared;
			} else {
				krige(options.type, model, lags, count, s.workspace, s.solution);
				++s.stats.solved;
				if (s.solution.status & KrigingJittered) ++s.stats.jittered;
				if (s.solution.status & KrigingFailed) ++s.stats.failed;
				s.cache.insert(hash, lags, lagSize, model, s.solution, options.cacheCapacity);
				solution = &s.solution;
			}
			double z = 0.0;
			const double* w = solution->weights.data();
			if (simple) {
				for (size_t i = 0; i < count; ++i) z += w[i] * (values[s.neighbors.ids[i]] - mean(s.neighbors.ids[i]));
				z += m0;
			} else if (solution->status & KrigingFailed) {
				z = m0;
			} else {
				for (size_t i = 0; i < count; ++i) z += w[i] * values[s.neighbors.ids[i]];
			}
			estimate[t] = z;
			if (variance && options.computeVariance) variance[t] = solution->variance;
		}
	});
	LocalKrigingStats total;
	for (const ThreadState& s : states) {
		total.targets += s.stats.targets;
		total.solved += s.stats.solved;
		total.shared += s.stats.shared;
		total.noNeighbor += s.stats.noNeighbor;
		total.jittered += s.stats.jittered;
		total.failed += s.stats.failed;
		total.neighbors += s.stats.neighbors;
	}
	return total;
}

} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_KRIGING_LOCAL_HPP
