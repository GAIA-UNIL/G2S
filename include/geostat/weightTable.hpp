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

// Weight table of a sequential Gaussian simulation (the "plan") and its
// binary file. Nothing here knows about grids: nodes are ids into flat value
// arrays, neighbours are 32-bit references that an addressing object
// (sgsEngine.hpp) turns into node ids given the target node.
//
// ---------------------------------------------------------------- table
// entries   one per simulated node, in LEVEL order: a node's level is one more
//           than the highest level among its simulated neighbours (0 when it
//           has none), so the entries of one level are independent and can be
//           replayed in parallel. Inside a level, entries keep path order.
//   levelStart[levelCount + 1]  entries of level l: [levelStart[l], levelStart[l+1])
//   entryTarget[entryCount]     node id
//   entryStencil[entryCount]    stencil id
// stencils  one per distinct solved system (shared by every entry with the
//           same neighbour references and the same resolved covariance), CSR:
//   stencilStart[stencilCount + 1]  neighbours of stencil s: [stencilStart[s], stencilStart[s+1])
//   stencilSigma[stencilCount]      kriging standard deviation (float32)
//   neighborRef[neighborCount]      reference, nearest first (uint32)
//   neighborWeight[neighborCount]   kriging weight (float32, computed in double)
// refTable  int32 [refCount x ndim]: what a reference means for the front end
//           (the grid kernel offsets), empty when references are node ids.
// geometry  uint64 [geometryCount]: front-end description (grid dims).
// Stencils are numbered in order of first use by the level-ordered entries,
// so without sharing stencil e is entry e and replay streams through memory.
//
// ---------------------------------------------------------------- file
// Little-endian host order of the writer; all integers are fixed width; the
// file is meant to be memory-mapped and used in place (no parsing).
//   offset 0: header, 256 bytes:
//     char     magic[8]        "G2SSGSWT"
//     uint32   version         1
//     uint32   endianTag       0x01020304 as written (a reader on a host of
//                              the other endianness sees 0x04030201 and refuses)
//     uint32   headerSize      256
//     uint32   krigingType     0 simple, 1 ordinary
//     uint32   ndim            lag components of refTable
//     uint32   flags           bit 0: stencils shared; bit 1: circular (-csim)
//     uint64   inputHash       hash of everything the weights depend on except
//                              the path: geometry, hard-data mask, model
//                              (canonical text and every resolved parameter),
//                              kriging type, neighbour count, kernel offsets,
//                              search options (computed by the front end)
//     uint64   pathHash        hash of the path (node ids in path order)
//     uint64   nodeCount, entryCount, levelCount, stencilCount, neighborCount,
//              refCount, geometryCount
//     uint64   reserved        0
//     section  sections[9]     {uint64 offset, uint64 bytes} of, in order:
//              levelStart (uint64), entryTarget (uint64), entryStencil (uint32),
//              stencilStart (uint64), stencilSigma (float32),
//              neighborRef (uint32), neighborWeight (float32),
//              refTable (int32), geometry (uint64)
//   then the sections, each starting on a 64-byte boundary (zero padding).
// A reader checks magic, version, endianTag, headerSize, that every section
// lies inside the file with the expected size, and the consistency of the
// counts; the front end then compares inputHash (and pathHash when the path is
// given again) with its own inputs and refuses a mismatch.

#ifndef G2S_GEOSTAT_WEIGHT_TABLE_HPP
#define G2S_GEOSTAT_WEIGHT_TABLE_HPP

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "geostat/covarianceModel.hpp"

namespace g2s {
namespace geostat {

// ============================================================ hashing

// FNV-1a 64 over bytes, deterministic across runs and platforms of the same
// endianness. Doubles are normalised (-0 -> 0) before hashing.
class Hasher {
public:
	void bytes(const void* data, size_t size) {
		const unsigned char* p = static_cast<const unsigned char*>(data);
		for (size_t i = 0; i < size; ++i) { _hash ^= p[i]; _hash *= 1099511628211ull; }
	}
	void u64(uint64_t v) { bytes(&v, sizeof(v)); }
	void i64(int64_t v) { bytes(&v, sizeof(v)); }
	void f64(double v) { if (v == 0.0) v = 0.0; bytes(&v, sizeof(v)); }
	void text(const std::string& s) { u64(s.size()); bytes(s.data(), s.size()); }
	uint64_t value() const { return _hash; }
private:
	uint64_t _hash = 1469598103934665603ull;
};

// Hash of what a model set means: canonical texts, per-node model index and
// every resolved parameter (map tables by content), visiting every node when
// the set is not stationary. ResolvedModel::key() is not used because it
// hashes the address of map tables, which changes from run to run.
inline uint64_t covarianceModelSetHash(const CovarianceModelSet& models, size_t nodeCount) {
	Hasher h;
	for (const std::string& text : models.canonical()) h.text(text);
	std::vector<std::pair<const float*, uint64_t> > tables;
	auto tableHash = [&](const ResolvedStructure& s) -> uint64_t {
		for (const std::pair<const float*, uint64_t>& t : tables) if (t.first == s.table) return t.second;
		Hasher th;
		size_t count = 1;
		for (unsigned d = 0; d < s.ndim; ++d) { th.u64(s.tableDims[d]); count *= s.tableDims[d]; }
		th.bytes(s.table, count * sizeof(float));
		tables.push_back({ s.table, th.value() });
		return th.value();
	};
	auto mixModel = [&](const ResolvedModel& m) {
		h.u64(m.modelIndex);
		h.u64(m.structures.size());
		for (const ResolvedStructure& s : m.structures) {
			h.u64(uint64_t(s.type));
			h.f64(s.coefficient);
			h.f64(s.sill);
			h.f64(s.shape);
			for (unsigned k = 0; k < s.ndim; ++k) h.f64(s.range[k]);
			for (unsigned k = 0; k < 3; ++k) h.f64(s.angle[k]);
			if (s.table) h.u64(tableHash(s));
		}
	};
	h.u64(models.ndim());
	if (models.isStationary()) {
		h.u64(1);
		mixModel(models.stationaryModel());
	} else {
		h.u64(0);
		h.u64(nodeCount);
		ResolvedModel scratch;
		for (size_t node = 0; node < nodeCount; ++node) mixModel(models.at(node, scratch));
	}
	return h.value();
}

// ============================================================ table

struct WeightTableView {
	uint32_t krigingType = 0; // 0 simple, 1 ordinary
	uint32_t ndim = 0;
	uint32_t flags = 0;
	uint64_t inputHash = 0, pathHash = 0;
	uint64_t nodeCount = 0, entryCount = 0, levelCount = 0, stencilCount = 0, neighborCount = 0, refCount = 0, geometryCount = 0;
	const uint64_t* levelStart = nullptr;
	const uint64_t* entryTarget = nullptr;
	const uint32_t* entryStencil = nullptr;
	const uint64_t* stencilStart = nullptr;
	const float* stencilSigma = nullptr;
	const uint32_t* neighborRef = nullptr;
	const float* neighborWeight = nullptr;
	const int32_t* refTable = nullptr;
	const uint64_t* geometry = nullptr;

	size_t neighborsOf(uint64_t entry) const { const uint32_t s = entryStencil[entry]; return size_t(stencilStart[s + 1] - stencilStart[s]); }
};

enum WeightTableFlags : uint32_t { WeightTableShared = 1u, WeightTableCircular = 2u };

// Owning, in-memory table (the result of a plan).
struct WeightTable {
	uint32_t krigingType = 0, ndim = 0, flags = 0;
	uint64_t inputHash = 0, pathHash = 0, nodeCount = 0;
	std::vector<uint64_t> levelStart, entryTarget, stencilStart, geometry;
	std::vector<uint32_t> entryStencil, neighborRef;
	std::vector<float> stencilSigma, neighborWeight;
	std::vector<int32_t> refTable;

	WeightTableView view() const {
		WeightTableView v;
		v.krigingType = krigingType; v.ndim = ndim; v.flags = flags;
		v.inputHash = inputHash; v.pathHash = pathHash; v.nodeCount = nodeCount;
		v.entryCount = entryTarget.size();
		v.levelCount = levelStart.empty() ? 0 : levelStart.size() - 1;
		v.stencilCount = stencilSigma.size();
		v.neighborCount = neighborRef.size();
		v.refCount = ndim ? refTable.size() / ndim : 0;
		v.geometryCount = geometry.size();
		v.levelStart = levelStart.data(); v.entryTarget = entryTarget.data(); v.entryStencil = entryStencil.data();
		v.stencilStart = stencilStart.data(); v.stencilSigma = stencilSigma.data();
		v.neighborRef = neighborRef.data(); v.neighborWeight = neighborWeight.data();
		v.refTable = refTable.data(); v.geometry = geometry.data();
		return v;
	}
};

// ============================================================ file

constexpr char kWeightFileMagic[8] = { 'G', '2', 'S', 'S', 'G', 'S', 'W', 'T' };
constexpr uint32_t kWeightFileVersion = 1;
constexpr uint32_t kWeightFileEndianTag = 0x01020304u;
constexpr size_t kWeightFileAlignment = 64;
constexpr unsigned kWeightFileSections = 9;

struct WeightFileSection { uint64_t offset, bytes; };

struct WeightFileHeader {
	char magic[8];
	uint32_t version, endianTag, headerSize, krigingType, ndim, flags;
	uint64_t inputHash, pathHash;
	uint64_t nodeCount, entryCount, levelCount, stencilCount, neighborCount, refCount, geometryCount;
	uint64_t reserved;
	WeightFileSection sections[kWeightFileSections];
};
static_assert(sizeof(WeightFileHeader) == 256, "the weight file header must be 256 bytes");

namespace detail {
inline uint64_t alignUp(uint64_t v) { return (v + kWeightFileAlignment - 1) / kWeightFileAlignment * kWeightFileAlignment; }

// Expected byte size of each section from the counts.
inline void sectionSizes(const WeightFileHeader& h, uint64_t* bytes) {
	bytes[0] = (h.levelCount + 1) * sizeof(uint64_t);
	bytes[1] = h.entryCount * sizeof(uint64_t);
	bytes[2] = h.entryCount * sizeof(uint32_t);
	bytes[3] = (h.stencilCount + 1) * sizeof(uint64_t);
	bytes[4] = h.stencilCount * sizeof(float);
	bytes[5] = h.neighborCount * sizeof(uint32_t);
	bytes[6] = h.neighborCount * sizeof(float);
	bytes[7] = h.refCount * h.ndim * sizeof(int32_t);
	bytes[8] = h.geometryCount * sizeof(uint64_t);
}
} // namespace detail

// Writes the table; returns false and sets `error` on failure.
inline bool writeWeightFile(const std::string& path, const WeightTableView& t, std::string& error) {
	WeightFileHeader h;
	std::memset(&h, 0, sizeof(h));
	std::memcpy(h.magic, kWeightFileMagic, 8);
	h.version = kWeightFileVersion; h.endianTag = kWeightFileEndianTag; h.headerSize = sizeof(WeightFileHeader);
	h.krigingType = t.krigingType; h.ndim = t.ndim; h.flags = t.flags;
	h.inputHash = t.inputHash; h.pathHash = t.pathHash;
	h.nodeCount = t.nodeCount; h.entryCount = t.entryCount; h.levelCount = t.levelCount; h.stencilCount = t.stencilCount;
	h.neighborCount = t.neighborCount; h.refCount = t.refCount; h.geometryCount = t.geometryCount;
	uint64_t bytes[kWeightFileSections];
	detail::sectionSizes(h, bytes);
	const void* data[kWeightFileSections] = { t.levelStart, t.entryTarget, t.entryStencil, t.stencilStart, t.stencilSigma,
	                                          t.neighborRef, t.neighborWeight, t.refTable, t.geometry };
	uint64_t offset = detail::alignUp(sizeof(h));
	for (unsigned s = 0; s < kWeightFileSections; ++s) {
		h.sections[s].offset = offset;
		h.sections[s].bytes = bytes[s];
		offset = detail::alignUp(offset + bytes[s]);
	}
	FILE* file = fopen(path.c_str(), "wb");
	if (!file) { error = "cannot open " + path + " for writing"; return false; }
	static const char zeros[kWeightFileAlignment] = { 0 };
	bool ok = fwrite(&h, sizeof(h), 1, file) == 1;
	uint64_t position = sizeof(h);
	for (unsigned s = 0; s < kWeightFileSections && ok; ++s) {
		if (h.sections[s].offset > position) ok = fwrite(zeros, 1, size_t(h.sections[s].offset - position), file) == size_t(h.sections[s].offset - position);
		position = h.sections[s].offset;
		if (ok && bytes[s] > 0) ok = fwrite(data[s], 1, size_t(bytes[s]), file) == size_t(bytes[s]);
		position += bytes[s];
	}
	if (fclose(file) != 0) ok = false;
	if (!ok) error = "failed to write " + path;
	return ok;
}

// A weight file opened for replay: memory-mapped (POSIX) or read into memory.
class WeightFile {
public:
	WeightFile() = default;
	WeightFile(const WeightFile&) = delete;
	WeightFile& operator=(const WeightFile&) = delete;
	~WeightFile() { close(); }

	bool open(const std::string& path, std::string& error) {
		close();
		const char* base = nullptr;
		size_t size = 0;
#if !defined(_WIN32)
		const int fd = ::open(path.c_str(), O_RDONLY);
		if (fd < 0) { error = "cannot open the weight file " + path; return false; }
		struct stat info;
		if (fstat(fd, &info) != 0) { ::close(fd); error = "cannot read the size of " + path; return false; }
		size = size_t(info.st_size);
		if (size >= sizeof(WeightFileHeader)) {
			void* map = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
			if (map != MAP_FAILED) { _map = map; _mapSize = size; base = static_cast<const char*>(map); }
		}
		::close(fd);
#endif
		if (!base) {
			FILE* file = fopen(path.c_str(), "rb");
			if (!file) { error = "cannot open the weight file " + path; return false; }
			fseek(file, 0, SEEK_END);
			const long length = ftell(file);
			fseek(file, 0, SEEK_SET);
			if (length < 0) { fclose(file); error = "cannot read " + path; return false; }
			size = size_t(length);
			_buffer.resize((size + 7) / 8 + 1);
			const bool read = size == 0 || fread(_buffer.data(), 1, size, file) == size;
			fclose(file);
			if (!read) { error = "cannot read " + path; return false; }
			base = reinterpret_cast<const char*>(_buffer.data());
		}
		const std::string what = path + " is not a valid sgs weight file: ";
		if (size < sizeof(WeightFileHeader)) { error = what + "too short"; close(); return false; }
		WeightFileHeader h;
		std::memcpy(&h, base, sizeof(h));
		if (std::memcmp(h.magic, kWeightFileMagic, 8) != 0) { error = what + "bad magic"; close(); return false; }
		if (h.endianTag != kWeightFileEndianTag) { error = path + " was written on a host of the other endianness"; close(); return false; }
		if (h.version != kWeightFileVersion) { error = path + " has format version " + std::to_string(h.version) + ", this sgs reads version " + std::to_string(kWeightFileVersion); close(); return false; }
		if (h.headerSize != sizeof(WeightFileHeader)) { error = what + "unexpected header size"; close(); return false; }
		uint64_t bytes[kWeightFileSections];
		detail::sectionSizes(h, bytes);
		for (unsigned s = 0; s < kWeightFileSections; ++s) {
			const WeightFileSection& section = h.sections[s];
			if (section.bytes != bytes[s] || section.offset % kWeightFileAlignment != 0 || section.offset > size || section.bytes > size - section.offset) {
				error = what + "section " + std::to_string(s) + " is truncated or inconsistent with the counts";
				close();
				return false;
			}
		}
		WeightTableView& v = _view;
		v.krigingType = h.krigingType; v.ndim = h.ndim; v.flags = h.flags;
		v.inputHash = h.inputHash; v.pathHash = h.pathHash;
		v.nodeCount = h.nodeCount; v.entryCount = h.entryCount; v.levelCount = h.levelCount; v.stencilCount = h.stencilCount;
		v.neighborCount = h.neighborCount; v.refCount = h.refCount; v.geometryCount = h.geometryCount;
		v.levelStart = reinterpret_cast<const uint64_t*>(base + h.sections[0].offset);
		v.entryTarget = reinterpret_cast<const uint64_t*>(base + h.sections[1].offset);
		v.entryStencil = reinterpret_cast<const uint32_t*>(base + h.sections[2].offset);
		v.stencilStart = reinterpret_cast<const uint64_t*>(base + h.sections[3].offset);
		v.stencilSigma = reinterpret_cast<const float*>(base + h.sections[4].offset);
		v.neighborRef = reinterpret_cast<const uint32_t*>(base + h.sections[5].offset);
		v.neighborWeight = reinterpret_cast<const float*>(base + h.sections[6].offset);
		v.refTable = reinterpret_cast<const int32_t*>(base + h.sections[7].offset);
		v.geometry = reinterpret_cast<const uint64_t*>(base + h.sections[8].offset);
		_size = size;
		if (!checkStructure(error)) { error = what + error; close(); return false; }
		return true;
	}

	const WeightTableView& view() const { return _view; }
	size_t fileSize() const { return _size; }
	bool mapped() const { return _map != nullptr; }

	void close() {
#if !defined(_WIN32)
		if (_map) munmap(_map, _mapSize);
#endif
		_map = nullptr;
		_mapSize = 0;
		_buffer.clear();
		_view = WeightTableView();
		_size = 0;
	}

private:
	void* _map = nullptr;
	size_t _mapSize = 0, _size = 0;
	std::vector<uint64_t> _buffer;
	WeightTableView _view;

	// Index bounds, so that a corrupt file cannot make replay read or write out of range.
	bool checkStructure(std::string& error) const {
		const WeightTableView& v = _view;
		if (v.levelStart[0] != 0 || v.levelStart[v.levelCount] != v.entryCount) { error = "level offsets do not cover the entries"; return false; }
		for (uint64_t l = 0; l < v.levelCount; ++l) if (v.levelStart[l] > v.levelStart[l + 1]) { error = "level offsets decrease"; return false; }
		if (v.stencilStart[0] != 0 || v.stencilStart[v.stencilCount] != v.neighborCount) { error = "stencil offsets do not cover the neighbours"; return false; }
		for (uint64_t s = 0; s < v.stencilCount; ++s) if (v.stencilStart[s] > v.stencilStart[s + 1]) { error = "stencil offsets decrease"; return false; }
		for (uint64_t e = 0; e < v.entryCount; ++e) {
			if (v.entryTarget[e] >= v.nodeCount) { error = "an entry targets a node outside the grid"; return false; }
			if (v.entryStencil[e] >= v.stencilCount) { error = "an entry refers to a missing stencil"; return false; }
		}
		const uint64_t refLimit = v.refCount > 0 ? v.refCount : v.nodeCount; // without a table, references are node ids
		for (uint64_t i = 0; i < v.neighborCount; ++i) if (v.neighborRef[i] >= refLimit) { error = "a neighbour reference is out of range"; return false; }
		return true;
	}
};

} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_WEIGHT_TABLE_HPP
