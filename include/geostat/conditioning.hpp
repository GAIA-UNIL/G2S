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

// Conditioning of unconditional Gaussian realisations to hard data by kriging
// of the residuals (Journel & Huijbregts 1978; Chiles & Delfiner 2012, sec.
// 7.3). With z_u an unconditional realisation of the model and d the data at
// the nodes x_i,
//     z_c(x) = z_u(x) + K[d - z_u](x),
// where K[r](x) = sum_i w_i(x) r_i is the kriging estimate at x from the
// residuals r_i = d_i - z_u(x_i), with the same covariance model as z_u.
// z_c honours the data exactly (data nodes are set to d_i, which is also what
// the kriging returns there), and, for simple kriging with the exact
// covariance, z_c has the conditional distribution: mean = the SK estimate
// from d, covariance = the SK error covariance (variance = the SK variance).
//
// Kriging type
//   SK (default): the residuals have mean 0 whatever the mean m of z_u (both
//     d and z_u have mean m), so the residuals are kriged with mean 0 and m
//     does not enter the conditioning. This is the theoretically consistent
//     choice: the simulation assumes a known mean, and SK of the residuals
//     gives exactly the conditional Gaussian distribution. Far from the data
//     (beyond the range) z_c = z_u, with the unconditional mean, variance and
//     covariance.
//   OK: the weights sum to 1, so z_c = z_u - OK[z_u] + OK[d]: the mean of z_u
//     is filtered out and replaced by the mean estimated from the data (the
//     generalised-least-squares mean with the direct method, a local mean in
//     each neighbourhood with the local method). m then plays no role. Use it
//     when the mean is unknown or drifts slowly (each neighbourhood re-
//     estimates it); far from the data z_c has the data's (local) mean rather
//     than m, and the conditional variance is the SK variance plus the
//     uncertainty of the estimated mean (exact for the direct method: z_c then
//     has the OK error covariance).
//
// Methods (ConditioningMethod)
//   Direct  every datum at every node, dual form (krigingGlobal.hpp): one
//           Cholesky of the n x n data covariance, shared by every realisation
//           (only the right-hand side, the residuals, changes), then
//           n x nodes multiply-adds per realisation (on a grid through the
//           covariance table, GridCovarianceTable). Exact.
//   Local   moving neighbourhood: the maxNeighbors nearest data of each node
//           (k-d tree, kdTreeSearch.hpp, with the optional window / filter /
//           per-sector limit of the kriging executable). The weights do not
//           depend on the realisation: they are computed once per node (and
//           shared between nodes with identical neighbour lags, as in
//           krigingLocal.hpp) and applied to every realisation passed to the
//           same condition() call; a realisation costs maxNeighbors
//           multiply-adds per node. Approximate where the neighbourhood
//           leaves out data that matter.
//   Auto    the rule of the kriging executable's -method auto
//           (kAutoDirectMaxData, krigingGlobal.hpp): Direct up to that many
//           data while the n x n matrix fits maxMatrixBytes, Local otherwise.
//           It does not depend on the number of realisations, so realisation r
//           of a batch is the same as a run of realisation r alone.
//
// Geometry (ConditioningGeometry): written against node ids and coordinates,
// not images. Node ids run over [0, nodeCount); position(node, x) gives the
// coordinates of any node (in the units of the model's lags). A regular grid
// (node = flat cell index x0 + d0*(x1 + d1*x2), coordinates = cell indices)
// sets `grid` and gets the covariance-table fast path of the direct method; a
// point set gives coordinates only. `period` (one value per axis, 0 = not
// periodic) makes the domain a torus: the covariance of a lag is
// periodicCovariance (krigingGlobal.hpp: the model at the minimal image,
// averaged over both signs at exactly half a period), which is the covariance
// of a periodic (-csim) FFT simulation on that grid; the local search then
// also finds the data through the periodic boundary (nearest image of each datum).
//
// Fields (ConditioningFields): float values, node i of realisation r at
// data[i * cellStride + r * realisationStride] (DataImage with R variables:
// cellStride R, realisationStride 1; realisation-major: 1, nodeCount).
//
// Determinism: every value is computed by one thread in a fixed order, each
// realisation is solved on its own, so the result does not depend on the
// number of threads nor on how realisations are batched.
//
// Usage:
//   using namespace g2s::geostat;
//   Conditioner conditioner;
//   ConditioningOptions options;   // SK, auto, 50 neighbours, 1 thread
//   std::string error;
//   if (!conditioner.prepare(model, ConditioningGeometry::regularGrid(dims, periodic), dataNodes, dataValues, n, options, error)) ...
//   conditioner.condition({ fields, R, 1 }, R);   // in place
//   conditioner.report();          // method, reason, jitter, local statistics, timings

#ifndef G2S_GEOSTAT_CONDITIONING_HPP
#define G2S_GEOSTAT_CONDITIONING_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "geostat/covarianceModel.hpp"
#include "geostat/denseSolver.hpp"
#include "geostat/kdTreeSearch.hpp"
#include "geostat/kriging.hpp"
#include "geostat/krigingGlobal.hpp"
#include "geostat/krigingLocal.hpp"
#include "geostat/neighborSearch.hpp"

namespace g2s {
namespace geostat {

enum class ConditioningMethod { Auto, Direct, Local };

inline const char* conditioningMethodName(ConditioningMethod method) {
	return method == ConditioningMethod::Direct ? "direct" : method == ConditioningMethod::Local ? "local" : "auto";
}

struct ConditioningGeometry {
	unsigned ndim = 0;
	size_t nodeCount = 0;
	std::function<void(size_t, double*)> position; // coordinates of any node
	std::vector<unsigned> grid;                    // non-empty: nodes are the cells of this grid
	std::vector<double> period;                    // per axis, 0 = not periodic; empty: none

	bool periodic() const {
		for (double p : period) if (p > 0) return true;
		return false;
	}

	// Cells of a regular grid (DataImage axis order); periodic: every axis wraps (-csim).
	static ConditioningGeometry regularGrid(const std::vector<unsigned>& dims, bool periodic) {
		ConditioningGeometry g;
		g.ndim = unsigned(dims.size());
		g.nodeCount = 1;
		for (unsigned d : dims) g.nodeCount *= d;
		g.grid = dims;
		g.position = [dims](size_t cell, double* x) {
			for (size_t d = 0; d < dims.size(); ++d) { x[d] = double(cell % dims[d]); cell /= dims[d]; }
		};
		if (periodic) for (unsigned d : dims) g.period.push_back(double(d));
		return g;
	}

	// Points: count x ndim coordinates, row-major (not copied: keep them alive).
	static ConditioningGeometry points(const double* coordinates, size_t count, unsigned ndim) {
		ConditioningGeometry g;
		g.ndim = ndim;
		g.nodeCount = count;
		g.position = [coordinates, ndim](size_t node, double* x) {
			for (unsigned d = 0; d < ndim; ++d) x[d] = coordinates[node * ndim + d];
		};
		return g;
	}
};

struct ConditioningOptions {
	KrigingType type = KrigingType::Simple;
	ConditioningMethod method = ConditioningMethod::Auto;
	// Local method.
	size_t maxNeighbors = 50;                    // -n
	std::vector<double> halfExtent;              // search window per axis (-ki / -ks); empty = unbounded
	std::function<bool(const double*)> accept;   // optional filter on the lag (kernel weights)
	unsigned maxPerSector = 0;                   // -os; 0 = off
	size_t cacheCapacity = 8192;                 // shared systems kept per thread
	// Direct method.
	size_t maxMatrixBytes = size_t(kDefaultDirectMaxMemoryGiB * double(size_t(1) << 30)); // -maxmem
	size_t fieldBufferBytes = size_t(512) << 20; // double buffer of kriged fields (realisations are processed in groups that fit)
	unsigned nbThreads = 1;
};

// Values of R realisations: node i of realisation r at data[i * cellStride + r * realisationStride].
struct ConditioningFields {
	float* data = nullptr;
	size_t cellStride = 1;
	size_t realisationStride = 0;
	float& at(size_t node, size_t r) const { return data[node * cellStride + r * realisationStride]; }
};

struct ConditioningReport {
	ConditioningMethod method = ConditioningMethod::Auto; // the method used (Direct or Local)
	std::string reason;                // why (auto) or "requested"
	size_t data = 0;                   // number of data
	size_t targets = 0;                // nodes that are not data
	double jitter = 0.0;               // direct: diagonal jitter of the n x n matrix
	bool covarianceTable = false;      // direct on a grid: the covariance table is used
	LocalKrigingStats local;           // local: neighbour systems, summed over condition() calls
	size_t duplicateImages = 0;        // local, periodic: neighbours dropped as a second image of a datum
	double prepareMs = 0.0;            // prepare(): factorisation / search structure
	double conditionMs = 0.0;          // condition(): summed over calls
};

// The automatic choice (see the header): Direct when 0 < n <= kAutoDirectMaxData
// and the n x n matrix fits maxMatrixBytes, Local otherwise.
inline ConditioningMethod chooseConditioningMethod(size_t dataCount, size_t maxMatrixBytes, std::string& reason) {
	if (dataCount > kAutoDirectMaxData) {
		reason = std::to_string(dataCount) + " data > " + std::to_string(kAutoDirectMaxData) + " (direct limit of auto)";
		return ConditioningMethod::Local;
	}
	if (GlobalKriging::matrixBytes(dataCount) > maxMatrixBytes) {
		reason = "the " + std::to_string(dataCount) + " x " + std::to_string(dataCount) + " matrix exceeds the memory limit (-maxmem)";
		return ConditioningMethod::Local;
	}
	reason = std::to_string(dataCount) + " data <= " + std::to_string(kAutoDirectMaxData);
	return ConditioningMethod::Direct;
}

// Not copyable nor movable: the local search refers to members.
class Conditioner {
public:
	Conditioner() {}
	Conditioner(const Conditioner&) = delete;
	Conditioner& operator=(const Conditioner&) = delete;

	// Sets up the conditioning to the `count` data at node ids dataNodes
	// (distinct, < geometry.nodeCount) with values dataValues (finite), for the
	// stationary `model` (copied; keep its CovarianceModelSet alive for map()
	// structures). Direct: factorises the data covariance; Local: builds the
	// neighbour search. count == 0 is valid: active() is then false and
	// condition() does nothing. Returns false with a message on failure.
	bool prepare(const ResolvedModel& model, const ConditioningGeometry& geometry, const size_t* dataNodes, const double* dataValues, size_t count,
	             const ConditioningOptions& options, std::string& error) {
		const auto begin = std::chrono::steady_clock::now();
		reset();
		_report.data = count;
		if (count == 0) return true;
		if (geometry.ndim < 1 || geometry.ndim > kMaxDims || geometry.ndim != model.ndim) { error = "conditioning: the geometry and the model do not have the same number of dimensions"; return false; }
		if (!geometry.position) { error = "conditioning: the geometry has no node positions"; return false; }
		if (!geometry.period.empty() && geometry.period.size() != geometry.ndim) { error = "conditioning: the period needs one value per axis"; return false; }
		if (!geometry.grid.empty()) {
			size_t cells = 1;
			for (unsigned d : geometry.grid) cells *= d;
			if (geometry.grid.size() != geometry.ndim || cells != geometry.nodeCount) { error = "conditioning: the grid does not match the node count"; return false; }
		}
		_model = model;
		_geometry = geometry;
		_options = options;
		_options.nbThreads = std::max(1u, options.nbThreads);
		_nodes.assign(dataNodes, dataNodes + count);
		_values.assign(dataValues, dataValues + count);
		const uint32_t none = std::numeric_limits<uint32_t>::max();
		if (count >= size_t(none)) { error = "conditioning: too many data"; return false; }
		_dataIndex.assign(geometry.nodeCount, none);
		for (size_t i = 0; i < count; ++i) {
			if (_nodes[i] >= geometry.nodeCount) { error = "conditioning: a datum has a node id outside the domain"; return false; }
			if (_dataIndex[_nodes[i]] != none) { error = "conditioning: node " + std::to_string(_nodes[i]) + " holds two data"; return false; }
			if (!std::isfinite(_values[i])) { error = "conditioning: datum " + std::to_string(i) + " is not finite"; return false; }
			_dataIndex[_nodes[i]] = uint32_t(i);
		}
		_report.targets = geometry.nodeCount - count;
		if (geometry.periodic()) _period = geometry.period;

		ConditioningMethod method = options.method;
		if (method == ConditioningMethod::Auto) method = chooseConditioningMethod(count, options.maxMatrixBytes, _report.reason);
		else _report.reason = "requested";
		_report.method = method;
		const bool ok = method == ConditioningMethod::Direct ? prepareDirect(error) : prepareLocal(error);
		_report.prepareMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
		if (ok) _active = true;
		return ok;
	}

	bool active() const { return _active; }
	const ConditioningReport& report() const { return _report; }

	// Replaces the `realisations` unconditional fields by conditional ones, in place.
	void condition(ConditioningFields fields, size_t realisations) {
		if (!_active || realisations == 0) return;
		const auto begin = std::chrono::steady_clock::now();
		if (_report.method == ConditioningMethod::Direct) conditionDirect(fields, realisations);
		else conditionLocal(fields, realisations);
		// Hard data, exactly.
		for (size_t i = 0; i < _nodes.size(); ++i)
			for (size_t r = 0; r < realisations; ++r) fields.at(_nodes[i], r) = float(_values[i]);
		_report.conditionMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
	}

private:
	ResolvedModel _model;
	ConditioningGeometry _geometry;
	ConditioningOptions _options;
	ConditioningReport _report;
	bool _active = false;
	std::vector<size_t> _nodes;       // data node ids
	std::vector<double> _values;      // data values
	std::vector<uint32_t> _dataIndex; // node id -> data index (max = not a datum)
	std::vector<double> _period;      // empty when not periodic
	// Direct.
	GlobalKriging _kriging;
	GridCovarianceTable _table;
	// Local.
	std::vector<int64_t> _ranks;            // real nodes, then the periodic images of the data
	std::vector<std::vector<double> > _shifts; // periodic image shifts (the identity excluded)
	std::unique_ptr<KdTreeNeighborSearch> _search;

	const double* period() const { return _period.empty() ? nullptr : _period.data(); }

	void reset() {
		_report = ConditioningReport();
		_active = false;
		_nodes.clear(); _values.clear(); _dataIndex.clear(); _period.clear();
		_kriging = GlobalKriging();
		_table = GridCovarianceTable();
		_search.reset();
		_ranks.clear(); _shifts.clear();
	}

	double residual(size_t i, const ConditioningFields& fields, size_t r) const { return _values[i] - double(fields.at(_nodes[i], r)); }

	// ---------------------------------------------------------------- direct

	bool prepareDirect(std::string& error) {
		const size_t n = _nodes.size();
		const unsigned nd = _geometry.ndim;
		std::vector<double> coordinates(n * nd);
		for (size_t i = 0; i < n; ++i) _geometry.position(_nodes[i], coordinates.data() + i * nd);
		GlobalKriging::Options options;
		options.type = _options.type;
		options.nbThreads = _options.nbThreads;
		options.maxMatrixBytes = _options.maxMatrixBytes;
		options.period = _period;
		if (!_kriging.factorize(_model, coordinates.data(), n, nd, options, error)) return false;
		_report.jitter = _kriging.jitter();
		if (!_geometry.grid.empty()) _report.covarianceTable = _table.build(_model, _geometry.grid, _options.nbThreads, _options.maxMatrixBytes / 2, !_period.empty());
		return true;
	}

	// k_i = C(x_i - x_node) for every datum, node coordinates from the geometry.
	class NodeCrossCovariance {
	public:
		NodeCrossCovariance(const GlobalKriging& kriging, const ConditioningGeometry& geometry, const double* period)
			: _kriging(kriging), _geometry(geometry), _period(period) {}
		void operator()(size_t node, double* k) const {
			const unsigned nd = _kriging.ndim();
			double x[kMaxDims], lag[kMaxDims];
			_geometry.position(node, x);
			const double* data = _kriging.coordinates();
			for (size_t i = 0; i < _kriging.dataCount(); ++i) {
				for (unsigned d = 0; d < nd; ++d) lag[d] = data[i * nd + d] - x[d];
				k[i] = periodicCovariance(_kriging.model(), lag, _period, nd);
			}
		}
	private:
		const GlobalKriging& _kriging;
		const ConditioningGeometry& _geometry;
		const double* _period;
	};

	void conditionDirect(const ConditioningFields& fields, size_t realisations) {
		const size_t n = _nodes.size(), nodes = _geometry.nodeCount;
		const unsigned threads = _options.nbThreads;
		// Dual weights, one realisation at a time (the same operations whatever the batch).
		std::vector<GlobalKrigingDual> duals(realisations);
		parallelFor(realisations, threads, [&](size_t r, unsigned) {
			std::vector<double> rhs(n);
			for (size_t i = 0; i < n; ++i) rhs[i] = residual(i, fields, r);
			_kriging.solve(rhs.data(), 1, n, nullptr, duals[r]);
		});
		// Kriged residual fields, in groups of realisations that fit the buffer.
		const size_t group = std::max<size_t>(1, std::min<size_t>(realisations, _options.fieldBufferBytes / std::max<size_t>(1, nodes * sizeof(double))));
		std::vector<double> kriged(group * nodes);
		GlobalKrigingDual dual;
		for (size_t r0 = 0; r0 < realisations; r0 += group) {
			const size_t m = std::min(group, realisations - r0);
			dual.count = n;
			dual.nrhs = m;
			dual.alpha.assign(n * m, 0.0);
			dual.constant.assign(m, 0.0);
			for (size_t r = 0; r < m; ++r) {
				dual.constant[r] = duals[r0 + r].constant[0];
				for (size_t i = 0; i < n; ++i) dual.alpha[i * m + r] = duals[r0 + r].alpha[i];
			}
			if (_report.covarianceTable) estimateGridFields(_table, _nodes.data(), dual, kriged.data(), nodes, threads);
			else _kriging.estimate(NodeCrossCovariance(_kriging, _geometry, period()), nodes, dual, kriged.data(), nodes);
			const size_t block = 4096;
			parallelFor((nodes + block - 1) / block, threads, [&](size_t b, unsigned) {
				const size_t end = std::min(nodes, (b + 1) * block);
				for (size_t node = b * block; node < end; ++node)
					for (size_t r = 0; r < m; ++r) {
						float& value = fields.at(node, r0 + r);
						value = float(double(value) + kriged[r * nodes + node]);
					}
			});
		}
	}

	// ---------------------------------------------------------------- local

	bool prepareLocal(std::string& error) {
		const size_t n = _nodes.size(), nodes = _geometry.nodeCount;
		const unsigned nd = _geometry.ndim;
		if (_options.maxNeighbors == 0) { error = "conditioning: the number of neighbours must be positive"; return false; }
		if (_options.maxPerSector > 0 && nd > 4) { error = "conditioning: the per-sector search supports at most 4 dimensions"; return false; }
		// Periodic: the data are repeated at every combination of -1, 0, +1
		// period along the periodic axes, so that the k-d tree finds the
		// nearest image of each datum; images are virtual nodes after the real ones.
		_shifts.clear();
		if (!_period.empty()) {
			std::vector<std::vector<double> > all(1, std::vector<double>());
			for (unsigned d = 0; d < nd; ++d) {
				std::vector<std::vector<double> > next;
				for (const std::vector<double>& prefix : all)
					for (int k = -1; k <= 1; ++k) {
						if (k != 0 && !(_period[d] > 0)) continue;
						next.push_back(prefix);
						next.back().push_back(double(k) * _period[d]);
					}
				all.swap(next);
			}
			for (const std::vector<double>& shift : all) {
				bool zero = true;
				for (double v : shift) zero &= v == 0.0;
				if (!zero) _shifts.push_back(shift);
			}
		}
		_ranks.assign(nodes + _shifts.size() * n, kRankKnown);
		for (size_t node = 0; node < nodes; ++node) if (_dataIndex[node] == std::numeric_limits<uint32_t>::max()) _ranks[node] = kRankNever;
		const ConditioningGeometry* geometry = &_geometry;
		const std::vector<size_t>* dataNodes = &_nodes;
		const std::vector<std::vector<double> >* shifts = &_shifts;
		auto position = [geometry, dataNodes, shifts, nodes, n, nd](size_t node, double* x) {
			if (node < nodes) { geometry->position(node, x); return; }
			const size_t image = (node - nodes) / n, i = (node - nodes) % n;
			geometry->position((*dataNodes)[i], x);
			for (unsigned d = 0; d < nd; ++d) x[d] += (*shifts)[image][d];
		};
		KdTreeSearchOptions searchOptions;
		searchOptions.halfExtent = _options.halfExtent;
		searchOptions.accept = _options.accept;
		searchOptions.maxPerSector = _options.maxPerSector;
		try {
			_search.reset(new KdTreeNeighborSearch(nd, _ranks.size(), position, _ranks.data(), 0, searchOptions));
		} catch (const std::exception& e) {
			error = std::string("conditioning: ") + e.what();
			return false;
		}
		return true;
	}

	size_t dataOfNeighbor(size_t id) const {
		const size_t nodes = _geometry.nodeCount;
		return id < nodes ? size_t(_dataIndex[id]) : (id - nodes) % _nodes.size();
	}

	void conditionLocal(const ConditioningFields& fields, size_t realisations) {
		const size_t n = _nodes.size(), nodes = _geometry.nodeCount, R = realisations;
		const unsigned threads = _options.nbThreads, nd = _geometry.ndim;
		const bool periodic = !_period.empty();
		const double* periodValues = period();
		const ResolvedModel& model = _model;
		auto covariance = [&model, periodValues, nd](const double* lag) { return periodicCovariance(model, lag, periodValues, nd); };
		// Residuals, datum-major: residual of datum i in realisation r at [i * R + r].
		std::vector<double> residuals(n * R);
		for (size_t i = 0; i < n; ++i)
			for (size_t r = 0; r < R; ++r) residuals[i * R + r] = residual(i, fields, r);

		struct ThreadState {
			NeighborSet neighbors;
			KrigingWorkspace workspace;
			KrigingSolution solution;
			detail::LocalSystemCache cache;
			LocalKrigingStats stats;
			std::vector<size_t> data; // data index of each neighbour
			size_t duplicates = 0;
		};
		std::vector<ThreadState> states(threads);
		const size_t maxCount = std::min(_options.maxNeighbors, n);
		const size_t block = 256;
		parallelFor((nodes + block - 1) / block, threads, [&](size_t b, unsigned thread) {
			ThreadState& s = states[thread];
			const size_t end = std::min(nodes, (b + 1) * block);
			for (size_t node = b * block; node < end; ++node) {
				if (_ranks[node] == kRankKnown) continue;
				++s.stats.targets;
				findNeighbors(node, maxCount, s.neighbors, s.data, s.duplicates);
				const size_t count = s.neighbors.size();
				if (count == 0) { ++s.stats.noNeighbor; continue; } // the residual mean, 0: z_c = z_u
				s.stats.neighbors += count;
				const double* lags = s.neighbors.lags.data();
				const size_t lagSize = s.neighbors.lags.size();
				const KrigingSolution* solution = nullptr;
				uint64_t hash = 0;
				if (_options.cacheCapacity > 0) {
					hash = detail::LocalSystemCache::hashLags(lags, lagSize, model.key());
					solution = s.cache.find(hash, lags, lagSize, model);
				}
				if (solution) {
					++s.stats.shared;
				} else {
					if (periodic) krige(_options.type, covariance, model.sill(), nd, lags, count, s.workspace, s.solution);
					else krige(_options.type, model, lags, count, s.workspace, s.solution);
					++s.stats.solved;
					if (s.solution.status & KrigingJittered) ++s.stats.jittered;
					if (s.solution.status & KrigingFailed) ++s.stats.failed;
					s.cache.insert(hash, lags, lagSize, model, s.solution, _options.cacheCapacity);
					solution = &s.solution;
				}
				// Failed systems have zero weights: z_c = z_u there.
				const double* w = solution->weights.data();
				for (size_t r = 0; r < R; ++r) {
					double correction = 0.0;
					for (size_t k = 0; k < count; ++k) correction += w[k] * residuals[s.data[k] * R + r];
					float& value = fields.at(node, r);
					value = float(double(value) + correction);
				}
			}
		});
		for (const ThreadState& s : states) {
			_report.local.targets += s.stats.targets;
			_report.local.solved += s.stats.solved;
			_report.local.shared += s.stats.shared;
			_report.local.noNeighbor += s.stats.noNeighbor;
			_report.local.jittered += s.stats.jittered;
			_report.local.failed += s.stats.failed;
			_report.local.neighbors += s.stats.neighbors;
			_report.duplicateImages += s.duplicates;
		}
	}

	// The maxCount nearest data of `node` (periodic: the nearest image of each
	// datum only; a farther image of a datum already found is dropped and the
	// search widened until maxCount distinct data or every image is seen).
	void findNeighbors(size_t node, size_t maxCount, NeighborSet& neighbors, std::vector<size_t>& data, size_t& duplicates) const {
		const unsigned nd = _geometry.ndim;
		size_t request = maxCount;
		for (;;) {
			_search->find(node, 0, request, neighbors);
			const size_t found = neighbors.size();
			data.clear();
			size_t kept = 0;
			for (size_t k = 0; k < found && data.size() < maxCount; ++k) {
				const size_t i = dataOfNeighbor(neighbors.ids[k]);
				if (std::find(data.begin(), data.end(), i) != data.end()) { ++duplicates; continue; }
				data.push_back(i);
				if (kept != k) {
					neighbors.ids[kept] = neighbors.ids[k];
					for (unsigned d = 0; d < nd; ++d) neighbors.lags[kept * nd + d] = neighbors.lags[k * nd + d];
				}
				++kept;
			}
			const bool exhausted = found < request || request >= _ranks.size();
			if (kept == maxCount || exhausted || kept == found) {
				neighbors.ids.resize(kept);
				neighbors.lags.resize(kept * nd);
				neighbors.refs.clear();
				return;
			}
			request = std::min(request * 2, _ranks.size());
		}
	}
};

} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_CONDITIONING_HPP
