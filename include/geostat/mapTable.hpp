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

// Validity of a covariance map table (the image bound to map(name)).
//
// A map table C (lag 0 at dims/2, 0 outside the table) is a valid covariance
// on the grid when it is symmetric, C(h) = C(-h), and positive definite: its
// discrete spectrum (the Fourier transform of the table seen as a function on
// the integer lags) is >= 0 everywhere. A raw experimental covariance map
// (cropped from an image) usually is not, and kriging systems built from it
// can be indefinite: SGS then produces values far outside the data range.
//
//   diagnoseMapTable(values, dims)
//       asymmetry: ||C(h) - C(-h)|| / (2 ||C||) (L2; 0 for a symmetric table);
//       negativeFraction: sum of the negative part of the spectrum of the
//       symmetric part over the sum of its absolute values (0 for a positive
//       definite table). The spectrum is sampled on a zero-padded grid of at
//       least twice the table along each axis (power-of-two sizes), which
//       gives the exact transform at those frequencies.
//   clipMapTable(values, dims)
//       projects the table onto a valid covariance of the same size by
//       alternating projections: zero the negative (and the imaginary, i.e.
//       antisymmetric) part of the spectrum, then the values outside the
//       table, until the negative fraction is below kClipTarget or
//       kClipMaxIterations. The result keeps the table size; C(0) usually
//       grows a little (the negative spectral values removed were part of it).
//
// No external FFT library: radix-2 transforms in double precision, cheap for
// the table sizes involved (a 161 x 161 table is transformed on 512 x 512).

#ifndef G2S_GEOSTAT_MAP_TABLE_HPP
#define G2S_GEOSTAT_MAP_TABLE_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace g2s {
namespace geostat {

struct MapTableDiagnostics {
	bool checked = false;          // false when the table is too large to transform
	double negativeFraction = 0.0;  // see above
	double asymmetry = 0.0;         // see above
	double centre = 0.0;            // C(0), the value at dims/2
	double largestAbsolute = 0.0;   // max |C(h)|
};

// Above these the map is reported (warning): about the level where the
// kriging systems of SGS start to be visibly indefinite.
constexpr double kMapNegativeFractionWarning = 1e-2;
constexpr double kMapAsymmetryWarning = 1e-2;
constexpr double kClipTarget = 1e-4;
constexpr unsigned kClipMaxIterations = 200;
// Largest padded grid transformed (cells); larger tables are not checked.
constexpr size_t kMapCheckMaxCells = size_t(1) << 25;

namespace detail {

inline size_t nextPowerOfTwo(size_t n) {
	size_t p = 1;
	while (p < n) p <<= 1;
	return p;
}

// In-place radix-2 FFT of `count` complex values with stride `stride`.
inline void fftLine(std::complex<double>* data, size_t count, size_t stride, bool inverse, std::vector<std::complex<double> >& line) {
	if (count < 2) return;
	line.resize(count);
	for (size_t i = 0; i < count; ++i) line[i] = data[i * stride];
	for (size_t i = 1, j = 0; i < count; ++i) {
		size_t bit = count >> 1;
		for (; j & bit; bit >>= 1) j ^= bit;
		j ^= bit;
		if (i < j) std::swap(line[i], line[j]);
	}
	const double pi = 3.14159265358979323846;
	for (size_t length = 2; length <= count; length <<= 1) {
		const double angle = (inverse ? 2.0 : -2.0) * pi / double(length);
		const std::complex<double> step(std::cos(angle), std::sin(angle));
		for (size_t begin = 0; begin < count; begin += length) {
			std::complex<double> w(1.0, 0.0);
			for (size_t k = 0; k < length / 2; ++k) {
				const std::complex<double> a = line[begin + k], b = line[begin + k + length / 2] * w;
				line[begin + k] = a + b;
				line[begin + k + length / 2] = a - b;
				w *= step;
			}
		}
	}
	for (size_t i = 0; i < count; ++i) data[i * stride] = line[i];
}

inline void fftAll(std::vector<std::complex<double> >& data, const std::vector<size_t>& size, bool inverse) {
	std::vector<std::complex<double> > line;
	size_t stride = 1;
	const size_t total = data.size();
	for (size_t d = 0; d < size.size(); ++d) {
		const size_t n = size[d];
		if (n > 1) {
			// lines along axis d: every start index whose coordinate d is 0
			for (size_t outer = 0; outer < total; outer += stride * n)
				for (size_t inner = 0; inner < stride; ++inner) fftLine(&data[outer + inner], n, stride, inverse, line);
		}
		stride *= n;
	}
	if (inverse) for (std::complex<double>& v : data) v /= double(total);
}

// Padded-grid geometry of a table: lag l of axis d sits at (l mod size[d]).
struct MapEmbedding {
	std::vector<unsigned> dims;
	std::vector<size_t> size;
	size_t total = 1;
	bool valid = false;

	explicit MapEmbedding(const std::vector<unsigned>& tableDims) : dims(tableDims) {
		for (int pass = 0; pass < 2; ++pass) {
			size.clear();
			total = 1;
			for (unsigned n : dims) {
				const size_t s = n > 1 ? nextPowerOfTwo(pass == 0 ? 2 * size_t(n) : size_t(n)) : 1;
				size.push_back(s);
				total *= s;
			}
			if (total <= kMapCheckMaxCells) { valid = true; return; }
		}
	}

	// Calls f(tableIndex, paddedIndex, mirrorTableIndex or SIZE_MAX) for every table cell.
	template <class F>
	void forEachCell(F f) const {
		size_t count = 1;
		for (unsigned n : dims) count *= n;
		std::vector<long> lag(dims.size());
		for (size_t index = 0; index < count; ++index) {
			size_t rest = index, padded = 0, paddedStride = 1, mirror = 0, tableStride = 1;
			bool mirrorInside = true;
			for (size_t d = 0; d < dims.size(); ++d) {
				const long n = long(dims[d]), centre = n / 2;
				const long l = long(rest % size_t(n)) - centre;
				rest /= size_t(n);
				const long s = long(size[d]);
				padded += size_t(((l % s) + s) % s) * paddedStride;
				paddedStride *= size_t(s);
				const long m = centre - l;
				if (m < 0 || m >= n) mirrorInside = false;
				else mirror += size_t(m) * tableStride;
				tableStride *= size_t(n);
			}
			f(index, padded, mirrorInside ? mirror : size_t(-1));
		}
	}
};

inline double negativeFractionOf(const std::vector<std::complex<double> >& spectrum) {
	double negative = 0.0, absolute = 0.0;
	for (const std::complex<double>& v : spectrum) {
		absolute += std::fabs(v.real());
		if (v.real() < 0) negative -= v.real();
	}
	return absolute > 0 ? negative / absolute : 0.0;
}

} // namespace detail

inline MapTableDiagnostics diagnoseMapTable(const std::vector<float>& values, const std::vector<unsigned>& dims) {
	MapTableDiagnostics result;
	const detail::MapEmbedding embedding(dims);
	size_t centre = 0, stride = 1;
	for (unsigned n : dims) { centre += size_t(n / 2) * stride; stride *= n; }
	if (centre < values.size()) result.centre = double(values[centre]);
	for (float v : values) result.largestAbsolute = std::max(result.largestAbsolute, double(std::fabs(v)));

	double difference = 0.0, norm = 0.0;
	embedding.forEachCell([&](size_t index, size_t, size_t mirror) {
		const double v = double(values[index]);
		const double m = mirror == size_t(-1) ? 0.0 : double(values[mirror]);
		difference += (v - m) * (v - m);
		norm += v * v;
	});
	result.asymmetry = norm > 0 ? 0.5 * std::sqrt(difference / norm) : 0.0;

	if (!embedding.valid) return result;
	std::vector<std::complex<double> > grid(embedding.total);
	embedding.forEachCell([&](size_t index, size_t padded, size_t) { grid[padded] = double(values[index]); });
	detail::fftAll(grid, embedding.size, false);
	result.negativeFraction = detail::negativeFractionOf(grid);
	result.checked = true;
	return result;
}

// Returns false (values unchanged) when the table is too large to transform.
inline bool clipMapTable(std::vector<float>& values, const std::vector<unsigned>& dims, MapTableDiagnostics& after, unsigned& iterations) {
	const detail::MapEmbedding embedding(dims);
	iterations = 0;
	if (!embedding.valid) return false;
	std::vector<std::complex<double> > grid(embedding.total);
	std::vector<char> inside(embedding.total, 0);
	embedding.forEachCell([&](size_t index, size_t padded, size_t) { grid[padded] = double(values[index]); inside[padded] = 1; });
	for (;;) {
		detail::fftAll(grid, embedding.size, false);
		const double negative = detail::negativeFractionOf(grid);
		if ((negative <= kClipTarget && iterations > 0) || iterations >= kClipMaxIterations) {
			detail::fftAll(grid, embedding.size, true);
			break;
		}
		for (std::complex<double>& v : grid) v = std::complex<double>(std::max(v.real(), 0.0), 0.0);
		detail::fftAll(grid, embedding.size, true);
		for (size_t i = 0; i < grid.size(); ++i) grid[i] = inside[i] ? std::complex<double>(grid[i].real(), 0.0) : std::complex<double>(0.0, 0.0);
		++iterations;
	}
	embedding.forEachCell([&](size_t index, size_t padded, size_t) { values[index] = float(grid[padded].real()); });
	after = diagnoseMapTable(values, dims);
	return true;
}

} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_MAP_TABLE_HPP
