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

// Neighbour search for sequential simulation and moving-neighbourhood kriging.
//
// Ranks. Every node has a rank (std::vector<int64_t>, one per node):
//   kRankKnown (-1)  known before the simulation (hard data);
//   i >= 0           the i-th node of the simulation path;
//   kRankNever       never informed (outside the path and without data).
// A node is an informed neighbour of a target when rank < rankLimit; for a
// sequential simulation rankLimit is the target's path position, for kriging
// of unknown nodes from data only it is 0. The result therefore depends only on
// the ranks and the geometry, never on timing or on the number of threads. It
// says which nodes to use, not that their values are ready: in a parallel
// simulation, the caller waits until every returned node with rank >= 0 has
// been written before reading its value (as pyDevSimulation.hpp does).
//
// NeighborSearch is the abstract interface; GridNeighborSearch implements it
// on a regular grid from kernel offsets sorted by distance (-ki/-ks, -csim).
// A future point (k-d tree) implementation would provide the same find():
// node ids are point indices, lags are coordinate differences (any real
// values), neighbours are the maxCount nearest points with rank < rankLimit
// ordered by (distance, id) so ties stay deterministic, optionally limited per
// sector with the same sectorOf() rule applied to the lag, and a maximum
// search radius in place of the kernel extent.

#ifndef G2S_GEOSTAT_NEIGHBOR_SEARCH_HPP
#define G2S_GEOSTAT_NEIGHBOR_SEARCH_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace g2s {
namespace geostat {

constexpr int64_t kRankKnown = -1;
// No reference (NeighborSet::ref, sgsEngine.hpp addressing).
constexpr uint32_t kNoReference = std::numeric_limits<uint32_t>::max();
constexpr int64_t kRankNever = std::numeric_limits<int64_t>::max();

// Ranks from a path: kRankKnown where isKnown(node), else the path position,
// else kRankNever. A known node listed in the path keeps kRankKnown.
template <class PathIndex, class IsKnown>
inline std::vector<int64_t> pathRanks(size_t nodeCount, const PathIndex* path, size_t pathLength, IsKnown isKnown) {
	std::vector<int64_t> ranks(nodeCount, kRankNever);
	for (size_t i = 0; i < pathLength; ++i) ranks[size_t(path[i])] = int64_t(i);
	for (size_t node = 0; node < nodeCount; ++node) if (isKnown(node)) ranks[node] = kRankKnown;
	return ranks;
}

// The neighbours of one target, nearest first.
struct NeighborSet {
	unsigned ndim = 0;
	std::vector<size_t> ids;    // node ids (flat cell index on a grid)
	std::vector<double> lags;   // size() x ndim, row-major: neighbour position - target position
	std::vector<uint32_t> refs; // optional, one per neighbour when filled: what the search knows of it
	                            // (GridNeighborSearch: index of the kernel offset; KdTreeSearch: node id)

	size_t size() const { return ids.size(); }
	const double* lag(size_t i) const { return lags.data() + i * ndim; }
	// The reference of neighbour i, kNoReference when the search gives none.
	uint32_t ref(size_t i) const { return i < refs.size() ? refs[i] : kNoReference; }
	void clear() { ids.clear(); lags.clear(); refs.clear(); }
};

class NeighborSearch {
public:
	virtual ~NeighborSearch() {}
	virtual unsigned ndim() const = 0;
	virtual size_t nodeCount() const = 0;
	// Fills `out` (cleared first) with at most maxCount neighbours of `node`
	// whose rank is < rankLimit, nearest first. The node itself is never
	// returned. const and thread-safe: use one NeighborSet per thread.
	virtual void find(size_t node, int64_t rankLimit, size_t maxCount, NeighborSet& out) const = 0;
};

// Sector (quadrant in 2D, octant in 3D, 2^ndim sectors) of a lag: bit d is set
// when component d is negative. Lags on an axis or a sector boundary are
// assigned as if rotated by an infinitesimal angle, counter-clockwise (axis 0
// towards axis 1) in 2D and about the direction (1, 2, 3) in 3D, so that the
// 2 x ndim axis neighbours fall into distinct sectors. In 1D and above 3D a
// zero component counts as positive.
inline unsigned sectorOf(const double* lag, unsigned ndim) {
	double key[3];
	const double tilt = 1e-9; // much smaller than one grid cell, larger than rounding
	if (ndim == 2) {
		key[0] = lag[0] - tilt * lag[1];
		key[1] = lag[1] + tilt * lag[0];
	} else if (ndim == 3) {
		// lag + tilt * (w x lag), w = (1, 2, 3)
		key[0] = lag[0] + tilt * (2.0 * lag[2] - 3.0 * lag[1]);
		key[1] = lag[1] + tilt * (3.0 * lag[0] - 1.0 * lag[2]);
		key[2] = lag[2] + tilt * (1.0 * lag[1] - 2.0 * lag[0]);
	}
	unsigned sector = 0;
	for (unsigned d = 0; d < ndim && d < 31; ++d) {
		const double value = (ndim == 2 || ndim == 3) ? key[d] : lag[d];
		if (value < 0) sector |= 1u << d;
	}
	return sector;
}

// ============================================================ grid offsets

// Offsets of a kernel of size kernelDims (DataImage axis order), the node
// itself excluded, sorted as pyDev/QS do: offsets -dims/2..dims/2 on each axis
// (clipped to the kernel), enumerated with the last axis fastest, kept when at
// least one variable has a finite non-zero weight (all kept when weights is
// null), then stable-sorted nearest first (Euclidean), or by decreasing
// largest |weight| when byWeight (-wd). weights: kernel values, nbVariable per cell.
inline std::vector<std::vector<int> > kernelOffsets(const std::vector<unsigned>& kernelDims, const float* weights = nullptr,
                                                    unsigned nbVariable = 1, bool byWeight = false) {
	const size_t ndim = kernelDims.size();
	std::vector<std::vector<int> > all(1);
	for (size_t dim = 0; dim < ndim; ++dim) {
		const int radius = int(kernelDims[dim] / 2);
		std::vector<std::vector<int> > next;
		next.reserve(all.size() * size_t(2 * radius + 1));
		for (const std::vector<int>& prefix : all) {
			for (int delta = -radius; delta <= radius; ++delta) {
				next.push_back(prefix);
				next.back().push_back(delta);
			}
		}
		all.swap(next);
	}
	struct Entry { size_t index; double priority; };
	std::vector<Entry> entries;
	entries.reserve(all.size());
	for (size_t i = 0; i < all.size(); ++i) {
		bool zero = true, inside = true;
		size_t flat = 0, stride = 1;
		for (size_t d = 0; d < ndim; ++d) {
			zero &= all[i][d] == 0;
			const int position = int(kernelDims[d] / 2) + all[i][d];
			if (position < 0 || position >= int(kernelDims[d])) inside = false;
			flat += size_t(std::max(position, 0)) * stride;
			stride *= kernelDims[d];
		}
		if (zero || !inside) continue;
		double priority = 0.0;
		if (weights) {
			bool usable = false;
			for (unsigned v = 0; v < nbVariable; ++v) {
				const float w = weights[flat * nbVariable + v];
				if (std::isfinite(w) && w != 0.f) usable = true;
				if (std::isfinite(w)) priority = std::max(priority, double(std::fabs(w)));
			}
			if (!usable) continue;
		}
		if (!byWeight || !weights) {
			double distance2 = 0.0;
			for (int value : all[i]) distance2 += double(value) * double(value);
			priority = -distance2;
		}
		entries.push_back({ i, priority });
	}
	std::stable_sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.priority > b.priority; });
	std::vector<std::vector<int> > offsets;
	offsets.reserve(entries.size());
	for (const Entry& entry : entries) offsets.push_back(all[entry.index]);
	return offsets;
}

// Stable re-sort nearest first by the anisotropic distance |M o|, M row-major
// ndim x ndim (e.g. ResolvedStructure::transform of the main structure).
inline void sortOffsetsByMetric(std::vector<std::vector<int> >& offsets, const double* metric, unsigned ndim) {
	std::vector<std::pair<double, size_t> > keys(offsets.size());
	for (size_t i = 0; i < offsets.size(); ++i) {
		double sum = 0.0;
		for (unsigned k = 0; k < ndim; ++k) {
			double u = 0.0;
			for (unsigned d = 0; d < ndim; ++d) u += metric[k * ndim + d] * offsets[i][d];
			sum += u * u;
		}
		keys[i] = { sum, i };
	}
	std::stable_sort(keys.begin(), keys.end(), [](const std::pair<double, size_t>& a, const std::pair<double, size_t>& b) { return a.first < b.first; });
	std::vector<std::vector<int> > sorted;
	sorted.reserve(offsets.size());
	for (const std::pair<double, size_t>& key : keys) sorted.push_back(offsets[key.second]);
	offsets.swap(sorted);
}

// ============================================================ grid search

struct GridSearchOptions {
	bool circular = false;      // -csim: offsets wrap around the grid (periodic)
	unsigned maxPerSector = 0;  // octant/quadrant search: at most this many per sector; 0 = off
};

// Grid implementation: walks the sorted offsets from the target and keeps the
// first maxCount cells with rank < rankLimit. NeighborSet::refs holds the
// index of each neighbour's offset in the offsets given to the constructor. In circular mode an offset
// leaving the grid wraps around (the lag stays the offset); keep the kernel
// smaller than the grid or a cell can be reached twice. In sector mode, an
// offset whose sector already holds maxPerSector neighbours is skipped, and
// the search stops when every sector is full or maxCount is reached.
class GridNeighborSearch : public NeighborSearch {
public:
	// dims: grid size in DataImage axis order (cell = x0 + d0*(x1 + d1*(x2 ...)));
	// offsets: as returned by kernelOffsets, nearest first; ranks: one per cell,
	// not copied (must outlive the search).
	GridNeighborSearch(const std::vector<unsigned>& dims, const std::vector<std::vector<int> >& offsets, const int64_t* ranks,
	                   GridSearchOptions options = GridSearchOptions())
		: _dims(dims), _ranks(ranks), _options(options) {
		const size_t nd = dims.size();
		if (nd == 0 || nd > 16) throw std::invalid_argument("GridNeighborSearch: the grid must have 1 to 16 dimensions");
		if (options.maxPerSector > 0 && nd > 4) throw std::invalid_argument("GridNeighborSearch: sector search supports at most 4 dimensions");
		_nodeCount = 1;
		for (unsigned d : dims) { _strides.push_back(_nodeCount); _nodeCount *= d; }
		_offsets.reserve(offsets.size() * nd);
		_lags.reserve(offsets.size() * nd);
		std::vector<double> lag(nd);
		for (size_t i = 0; i < offsets.size(); ++i) {
			const std::vector<int>& offset = offsets[i];
			if (offset.size() != nd) throw std::invalid_argument("GridNeighborSearch: an offset does not match the grid dimension");
			bool zero = true;
			for (int v : offset) zero &= v == 0;
			if (zero) continue;
			_offsetIndex.push_back(i < size_t(kNoReference) ? uint32_t(i) : kNoReference);
			for (size_t d = 0; d < nd; ++d) { _offsets.push_back(offset[d]); _lags.push_back(double(offset[d])); lag[d] = double(offset[d]); }
			_sectors.push_back((unsigned char)sectorOf(lag.data(), unsigned(nd)));
		}
	}

	unsigned ndim() const override { return unsigned(_dims.size()); }
	size_t nodeCount() const override { return _nodeCount; }
	size_t offsetCount() const { return _sectors.size(); }

	void find(size_t node, int64_t rankLimit, size_t maxCount, NeighborSet& out) const override {
		const unsigned nd = unsigned(_dims.size());
		out.ndim = nd;
		out.clear();
		if (maxCount == 0 || node >= _nodeCount) return;
		long coordinate[16];
		size_t rest = node;
		for (unsigned d = 0; d < nd; ++d) { coordinate[d] = long(rest % _dims[d]); rest /= _dims[d]; }
		const bool sectorMode = _options.maxPerSector > 0;
		const unsigned sectorCount = 1u << (sectorMode ? nd : 0);
		unsigned perSector[16] = { 0 };
		unsigned fullSectors = 0;
		const size_t count = _sectors.size();
		for (size_t p = 0; p < count; ++p) {
			if (sectorMode && perSector[_sectors[p]] >= _options.maxPerSector) continue;
			const int* offset = _offsets.data() + p * nd;
			size_t id = 0;
			bool skip = false;
			for (unsigned d = 0; d < nd; ++d) {
				long c = coordinate[d] + offset[d];
				const long size = long(_dims[d]);
				if (c < 0 || c >= size) {
					if (!_options.circular) { skip = true; break; }
					c = ((c % size) + size) % size;
				}
				id += size_t(c) * _strides[d];
			}
			if (skip || id == node) continue;
			if (!(_ranks[id] < rankLimit)) continue;
			out.ids.push_back(id);
			out.lags.insert(out.lags.end(), _lags.begin() + long(p * nd), _lags.begin() + long((p + 1) * nd));
			out.refs.push_back(_offsetIndex[p]);
			if (out.ids.size() >= maxCount) break;
			if (sectorMode && ++perSector[_sectors[p]] == _options.maxPerSector && ++fullSectors == sectorCount) break;
		}
	}

private:
	std::vector<unsigned> _dims;
	std::vector<size_t> _strides;
	size_t _nodeCount = 0;
	std::vector<int> _offsets;     // offsetCount x ndim
	std::vector<double> _lags;     // same, as doubles
	std::vector<unsigned char> _sectors;
	std::vector<uint32_t> _offsetIndex; // index in the offsets given to the constructor
	const int64_t* _ranks;
	GridSearchOptions _options;
};

} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_NEIGHBOR_SEARCH_HPP
