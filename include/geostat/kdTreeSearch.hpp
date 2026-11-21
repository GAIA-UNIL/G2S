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

// k-d tree implementation of NeighborSearch (neighborSearch.hpp) over the
// nodes that are informed from the start (e.g. hard data). GridNeighborSearch
// walks kernel offsets from the target, which costs about maxCount / density
// steps: fine for dense data, prohibitive for sparse data (1000 data on a
// 1000 x 1000 grid need ~50000 steps per target for 50 neighbours). The tree
// costs O(log m + maxCount) per query whatever the density.
//
// Same result as GridNeighborSearch (non-circular, distance-sorted offsets)
// when the coordinates are cell coordinates: the maxCount nearest indexed
// nodes with rank < rankLimit, ordered by (Euclidean distance, then the lag
// lexicographically, component 0 first) — exactly the tie order of
// kernelOffsets — restricted to a window |lag_d| <= halfExtent[d] and an
// optional accept(lag) filter (kernel weights), with the same per-sector limit
// (sectorOf). Coordinates may be any real values (point sets).
// Periodic (-csim) search is not supported: use GridNeighborSearch for it.
//
// Only nodes with rank < indexRankLimit (given at construction) are in the
// tree; find() further keeps those with rank < rankLimit. const and
// thread-safe after construction.

#ifndef G2S_GEOSTAT_KD_TREE_SEARCH_HPP
#define G2S_GEOSTAT_KD_TREE_SEARCH_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "geostat/covarianceModel.hpp"
#include "geostat/neighborSearch.hpp"

namespace g2s {
namespace geostat {

struct KdTreeSearchOptions {
	std::vector<double> halfExtent;                // per axis; empty = unbounded
	std::function<bool(const double*)> accept;     // optional filter on the lag
	unsigned maxPerSector = 0;                     // 0 = off
};

class KdTreeNeighborSearch : public NeighborSearch {
public:
	typedef KdTreeSearchOptions Options;

	// position(node, x) writes the ndim coordinates of any node (targets included).
	KdTreeNeighborSearch(unsigned ndim, size_t nodeCount, std::function<void(size_t, double*)> position, const int64_t* ranks,
	                     int64_t indexRankLimit, Options options = Options())
		: _ndim(ndim), _nodeCount(nodeCount), _position(std::move(position)), _ranks(ranks), _options(std::move(options)) {
		if (ndim < 1 || ndim > kMaxDims) throw std::invalid_argument("KdTreeNeighborSearch: 1 to " + std::to_string(kMaxDims) + " dimensions");
		if (!_options.halfExtent.empty() && _options.halfExtent.size() != ndim) throw std::invalid_argument("KdTreeNeighborSearch: halfExtent must have ndim values");
		for (size_t node = 0; node < nodeCount; ++node) if (ranks[node] < indexRankLimit) _ids.push_back(node);
		_points.resize(_ids.size() * ndim);
		for (size_t i = 0; i < _ids.size(); ++i) _position(_ids[i], _points.data() + i * ndim);
		if (!_ids.empty()) build(0, _ids.size());
	}

	unsigned ndim() const override { return _ndim; }
	size_t nodeCount() const override { return _nodeCount; }
	size_t indexedCount() const { return _ids.size(); }

	void find(size_t node, int64_t rankLimit, size_t maxCount, NeighborSet& out) const override {
		out.ndim = _ndim;
		out.clear();
		if (maxCount == 0 || _ids.empty() || node >= _nodeCount) return;
		Query q;
		q.node = node;
		q.rankLimit = rankLimit;
		_position(node, q.x);
		const bool sectorMode = _options.maxPerSector > 0;
		q.sectorCount = sectorMode ? (1u << _ndim) : 1u;
		q.capacity = sectorMode ? std::min<size_t>(_options.maxPerSector, maxCount) : maxCount;
		q.heaps.assign(q.sectorCount, std::vector<Candidate>());
		search(0, q);
		std::vector<Candidate> all;
		for (const std::vector<Candidate>& heap : q.heaps) all.insert(all.end(), heap.begin(), heap.end());
		std::sort(all.begin(), all.end(), [this](const Candidate& a, const Candidate& b) { return less(a, b); });
		if (all.size() > maxCount) all.resize(maxCount);
		for (const Candidate& c : all) {
			out.ids.push_back(c.id);
			out.lags.insert(out.lags.end(), c.lag, c.lag + _ndim);
		}
	}

private:
	struct Node {
		size_t begin, end;              // points [begin, end)
		size_t left = 0, right = 0;     // children (0 = leaf)
		double low[kMaxDims], high[kMaxDims];
	};
	struct Candidate {
		double distance2;
		size_t id;
		double lag[kMaxDims];
	};
	struct Query {
		size_t node;
		int64_t rankLimit;
		double x[kMaxDims];
		unsigned sectorCount;
		size_t capacity;
		std::vector<std::vector<Candidate> > heaps; // max-heaps: front = worst kept
	};

	unsigned _ndim;
	size_t _nodeCount;
	std::function<void(size_t, double*)> _position;
	const int64_t* _ranks;
	Options _options;
	std::vector<size_t> _ids;     // indexed node ids, reordered by the build
	std::vector<double> _points;  // their coordinates
	std::vector<Node> _nodes;

	bool less(const Candidate& a, const Candidate& b) const {
		if (a.distance2 != b.distance2) return a.distance2 < b.distance2;
		for (unsigned d = 0; d < _ndim; ++d) if (a.lag[d] != b.lag[d]) return a.lag[d] < b.lag[d];
		return a.id < b.id;
	}

	size_t build(size_t begin, size_t end) {
		const size_t index = _nodes.size();
		_nodes.push_back(Node());
		{
			Node& node = _nodes[index];
			node.begin = begin;
			node.end = end;
			for (unsigned d = 0; d < _ndim; ++d) { node.low[d] = std::numeric_limits<double>::infinity(); node.high[d] = -node.low[d]; }
			for (size_t i = begin; i < end; ++i)
				for (unsigned d = 0; d < _ndim; ++d) {
					node.low[d] = std::min(node.low[d], _points[i * _ndim + d]);
					node.high[d] = std::max(node.high[d], _points[i * _ndim + d]);
				}
		}
		if (end - begin <= 16) return index;
		unsigned axis = 0;
		double spread = -1.0;
		for (unsigned d = 0; d < _ndim; ++d) {
			const double s = _nodes[index].high[d] - _nodes[index].low[d];
			if (s > spread) { spread = s; axis = d; }
		}
		if (!(spread > 0)) return index; // all points equal
		const size_t middle = begin + (end - begin) / 2;
		std::vector<size_t> order(end - begin);
		std::iota(order.begin(), order.end(), begin);
		std::nth_element(order.begin(), order.begin() + long(middle - begin), order.end(), [&](size_t a, size_t b) {
			return _points[a * _ndim + axis] < _points[b * _ndim + axis];
		});
		std::vector<size_t> ids(end - begin);
		std::vector<double> points((end - begin) * _ndim);
		for (size_t i = 0; i < order.size(); ++i) {
			ids[i] = _ids[order[i]];
			for (unsigned d = 0; d < _ndim; ++d) points[i * _ndim + d] = _points[order[i] * _ndim + d];
		}
		std::copy(ids.begin(), ids.end(), _ids.begin() + long(begin));
		std::copy(points.begin(), points.end(), _points.begin() + long(begin * _ndim));
		const size_t left = build(begin, middle);
		const size_t right = build(middle, end);
		_nodes[index].left = left;
		_nodes[index].right = right;
		return index;
	}

	// Pruning bound for a box: the worst kept distance over the sectors the box
	// can reach, or infinity while one of them is not full.
	double bound(const Node& node, const Query& q) const {
		if (q.sectorCount == 1) {
			const std::vector<Candidate>& heap = q.heaps[0];
			return heap.size() < q.capacity ? std::numeric_limits<double>::infinity() : heap.front().distance2;
		}
		// Signs a lag inside the box can take (zero counts as both: sectorOf tilts it).
		bool negative[kMaxDims], positive[kMaxDims];
		for (unsigned d = 0; d < _ndim; ++d) {
			const double lowLag = node.low[d] - q.x[d], highLag = node.high[d] - q.x[d];
			const double tolerance = 1e-6 * (1.0 + std::fabs(lowLag) + std::fabs(highLag));
			negative[d] = lowLag <= tolerance;
			positive[d] = highLag >= -tolerance;
		}
		double worst = 0.0;
		for (unsigned s = 0; s < q.sectorCount; ++s) {
			bool reachable = true;
			for (unsigned d = 0; d < _ndim && reachable; ++d) reachable = (s >> d & 1u) ? negative[d] : positive[d];
			if (!reachable) continue;
			const std::vector<Candidate>& heap = q.heaps[s];
			if (heap.size() < q.capacity) return std::numeric_limits<double>::infinity();
			worst = std::max(worst, heap.front().distance2);
		}
		return worst;
	}

	// Squared distance from the query to the box, or infinity when the box misses the window.
	double boxDistance2(const Node& node, const Query& q) const {
		double sum = 0.0;
		for (unsigned d = 0; d < _ndim; ++d) {
			const double lowLag = node.low[d] - q.x[d], highLag = node.high[d] - q.x[d];
			if (!_options.halfExtent.empty() && (lowLag > _options.halfExtent[d] || highLag < -_options.halfExtent[d]))
				return std::numeric_limits<double>::infinity();
			const double gap = lowLag > 0 ? lowLag : (highLag < 0 ? -highLag : 0.0);
			sum += gap * gap;
		}
		return sum;
	}

	void search(size_t index, Query& q) const {
		const Node& node = _nodes[index];
		if (node.left == 0) {
			auto worse = [this](const Candidate& a, const Candidate& b) { return less(a, b); };
			for (size_t i = node.begin; i < node.end; ++i) {
				const size_t id = _ids[i];
				if (id == q.node || !(_ranks[id] < q.rankLimit)) continue;
				Candidate c;
				c.id = id;
				c.distance2 = 0.0;
				bool inside = true;
				for (unsigned d = 0; d < _ndim; ++d) {
					c.lag[d] = _points[i * _ndim + d] - q.x[d];
					if (!_options.halfExtent.empty() && std::fabs(c.lag[d]) > _options.halfExtent[d]) inside = false;
					c.distance2 += c.lag[d] * c.lag[d];
				}
				if (!inside || (_options.accept && !_options.accept(c.lag))) continue;
				std::vector<Candidate>& heap = q.heaps[q.sectorCount > 1 ? sectorOf(c.lag, _ndim) : 0];
				if (heap.size() < q.capacity) {
					heap.push_back(c);
					std::push_heap(heap.begin(), heap.end(), worse);
				} else if (less(c, heap.front())) {
					std::pop_heap(heap.begin(), heap.end(), worse);
					heap.back() = c;
					std::push_heap(heap.begin(), heap.end(), worse);
				}
			}
			return;
		}
		const double dl = boxDistance2(_nodes[node.left], q), dr = boxDistance2(_nodes[node.right], q);
		const size_t first = dl <= dr ? node.left : node.right, second = dl <= dr ? node.right : node.left;
		const double d1 = std::min(dl, dr), d2 = std::max(dl, dr);
		const double infinity = std::numeric_limits<double>::infinity();
		if (d1 < infinity && d1 <= bound(_nodes[first], q)) search(first, q);
		if (d2 < infinity && d2 <= bound(_nodes[second], q)) search(second, q);
	}
};

} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_KD_TREE_SEARCH_HPP
