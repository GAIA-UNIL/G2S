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
// as the id of its kernel offset (the order of the offsets given, as for
// GridNeighborSearch), so the stencil of a node depends only on its
// neighbour pattern and every node with the same pattern and covariance
// shares one solved system. Decoding is target + offset, with wrapping in
// circular mode (-csim).

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
		_half.assign(nd, 0);
		for (const std::vector<int>& o : offsets) {
			if (o.size() != nd) throw std::invalid_argument("GridAddressing: an offset does not match the grid dimension");
			for (size_t d = 0; d < nd; ++d) _half[d] = std::max(_half[d], std::abs(o[d]));
		}
		size_t box = 1;
		for (size_t d = 0; d < nd; ++d) { _boxStrides.push_back(box); box *= size_t(2 * _half[d] + 1); }
		_lookup.assign(box, kNoReference);
		for (size_t i = 0; i < offsets.size(); ++i) {
			size_t index = 0;
			int64_t delta = 0;
			for (size_t d = 0; d < nd; ++d) {
				index += size_t(offsets[i][d] + _half[d]) * _boxStrides[d];
				delta += int64_t(offsets[i][d]) * int64_t(_strides[d]);
				_offsets.push_back(offsets[i][d]);
			}
			if (_lookup[index] == kNoReference) _lookup[index] = uint32_t(i); // first one wins for duplicates
			_deltas.push_back(delta);
		}
	}

	unsigned ndim() const { return unsigned(_dims.size()); }
	size_t referenceLimit() const { return _deltas.size(); }
	bool shareable() const { return true; }
	std::vector<int32_t> referenceTable() const { return std::vector<int32_t>(_offsets.begin(), _offsets.end()); }

	uint32_t encode(size_t, size_t, const double* lag) const {
		size_t index = 0;
		for (size_t d = 0; d < _dims.size(); ++d) {
			const long v = std::lround(lag[d]);
			if (v < -_half[d] || v > _half[d]) return kNoReference;
			index += size_t(v + _half[d]) * _boxStrides[d];
		}
		return _lookup[index];
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
	std::vector<size_t> _strides, _boxStrides;
	std::vector<int> _half, _offsets;
	std::vector<int64_t> _deltas;
	std::vector<uint32_t> _lookup;
	bool _circular;
};

} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_GRID_ADDRESSING_HPP
