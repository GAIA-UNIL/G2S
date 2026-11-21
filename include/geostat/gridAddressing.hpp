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

// Grid addressing for the SGS engine (sgsEngine.hpp): a neighbour is stored
// as the id of its kernel offset, which GridNeighborSearch returns in
// NeighborSet::refs when it is built with the same offsets, so the stencil of
// a node depends only on its neighbour pattern and every node with the same
// pattern and covariance shares one solved system. encode() checks that the
// offset matches the lag (another search, or other offsets, give
// kNoReference). Decoding is target + offset, with wrapping in circular mode
// (-csim).

#ifndef G2S_GEOSTAT_GRID_ADDRESSING_HPP
#define G2S_GEOSTAT_GRID_ADDRESSING_HPP

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "geostat/sgsEngine.hpp"

namespace g2s {
namespace geostat {

class GridAddressing {
public:
	// dims: grid size in DataImage axis order; offsets: kernel offsets (without the zero offset).
	GridAddressing(const std::vector<unsigned>& dims, const std::vector<std::vector<int> >& offsets, bool circular)
		: _dims(dims), _circular(circular) {
		const size_t nd = dims.size();
		if (nd == 0 || nd > 16) throw std::invalid_argument("GridAddressing: the grid must have 1 to 16 dimensions");
		size_t stride = 1;
		for (unsigned d : dims) { _strides.push_back(stride); stride *= d; }
		for (const std::vector<int>& o : offsets) {
			if (o.size() != nd) throw std::invalid_argument("GridAddressing: an offset does not match the grid dimension");
			int64_t delta = 0;
			for (size_t d = 0; d < nd; ++d) {
				delta += int64_t(o[d]) * int64_t(_strides[d]);
				_offsets.push_back(o[d]);
			}
			_deltas.push_back(delta);
		}
	}

	unsigned ndim() const { return unsigned(_dims.size()); }
	size_t referenceLimit() const { return _deltas.size(); }
	bool shareable() const { return true; }
	bool integerLags() const { return true; }
	std::vector<int32_t> referenceTable() const { return std::vector<int32_t>(_offsets.begin(), _offsets.end()); }

	// searchRef: the offset index given by GridNeighborSearch.
	uint32_t encode(size_t, size_t, const double* lag, uint32_t searchRef) const {
		if (searchRef >= _deltas.size()) return kNoReference;
		const size_t nd = _dims.size();
		const int* offset = _offsets.data() + size_t(searchRef) * nd;
		for (size_t d = 0; d < nd; ++d) if (double(offset[d]) != lag[d]) return kNoReference;
		return searchRef;
	}

	void decode(size_t target, const uint32_t* refs, size_t count, size_t* ids) const {
		if (!_circular) {
			for (size_t i = 0; i < count; ++i) ids[i] = size_t(int64_t(target) + _deltas[refs[i]]);
			return;
		}
		const size_t nd = _dims.size();
		long coordinate[16];
		size_t rest = target;
		for (size_t d = 0; d < nd; ++d) { coordinate[d] = long(rest % _dims[d]); rest /= _dims[d]; }
		for (size_t i = 0; i < count; ++i) {
			const int* offset = _offsets.data() + size_t(refs[i]) * nd;
			size_t id = 0;
			for (size_t d = 0; d < nd; ++d) {
				long c = coordinate[d] + offset[d];
				const long size = long(_dims[d]);
				if (c < 0 || c >= size) c = ((c % size) + size) % size;
				id += size_t(c) * _strides[d];
			}
			ids[i] = id;
		}
	}

	void lag(size_t, uint32_t ref, double* out) const {
		const size_t nd = _dims.size();
		for (size_t d = 0; d < nd; ++d) out[d] = double(_offsets[size_t(ref) * nd + d]);
	}

private:
	std::vector<unsigned> _dims;
	std::vector<size_t> _strides;
	std::vector<int> _offsets;
	std::vector<int64_t> _deltas;
	bool _circular;
};

} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_GRID_ADDRESSING_HPP
