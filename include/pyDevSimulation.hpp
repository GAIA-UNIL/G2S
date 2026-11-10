/*
 * G2S - pyDev: prototype sequential-simulation algorithms in Python.
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

// Path-driven vector simulation for pyDev: C++ walks the path, gathers the
// informed neighbours of each node, and Python predicts the node.
//
// Neighbour rows passed to Python:
//   * row 0 is always the node itself (offset 0); its values are NaN except
//     for variables that are already known (hard data);
//   * the following rows are the closest informed cells (kernel order) until
//     every variable has n informed values or the kernel is exhausted; a value
//     that is not needed (its variable already has n values) is NaN.
// A cell is informed when it is hard data or when it comes strictly earlier in
// the path; workers wait (detached from Python) for earlier cells to finish, so
// the result does not depend on the number of threads or on the batching.
//
// Two modes:
//   * per node: each worker calls sample() for its node;
//   * batch: workers queue complete neighbourhoods in a shared batch and move
//     on to the next node. A batch is sent to sample_batch() by the worker that
//     fills it (batchSize nodes) or by the worker that finds every other worker
//     blocked (no further node can join), so batches are as full as possible
//     without waiting for nothing. While a batch is being predicted, the
//     "all blocked" rule waits for it to finish, since the next call cannot
//     start earlier anyway. Batch mode runs 2 x batchSize workers (fixed
//     oversubscription) so that enough nodes are in flight; blocked workers
//     sleep until the next batch is written.

#ifndef PY_DEV_SIMULATION_HPP
#define PY_DEV_SIMULATION_HPP

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "DataImage.hpp"
#include "jobReporting.hpp"
#include "pathIndexType.hpp"
#include "pyDevAlgorithm.hpp"

namespace g2s {
namespace pydev {

struct SimulationSetup {
	unsigned nbThreads = 1;      // per-node mode workers (-j)
	unsigned batchSize = 0;      // 0 = per-node mode; otherwise batch mode with 2 x batchSize workers
	bool circularSimulation = false;
	std::vector<unsigned> nbNeighbors;
	// Sorted neighbour offsets in DataImage axis order, without the zero offset.
	std::vector<std::vector<int> > offsets;
	const g2s_path_index_t* path = nullptr; // cells to simulate, in order
	g2s_path_index_t pathLength = 0;
	const float* uniformSeeds = nullptr;    // one per path position
	g2s::DataImage* trainingIndexImage = nullptr; // -ii, optional
};

struct SimulationStats {
	unsigned workers = 0;
	unsigned long long pythonCalls = 0;
	unsigned long long predictedNodes = 0;
};

inline bool simulate(FILE* reportFile, g2s::DataImage& di, Algorithm& algorithm, const SimulationSetup& setup,
                     unsigned* importDataIndex, SimulationStats& stats, std::string& error) {
	const unsigned nbVariable = di._nbVariable;
	const unsigned nbDims = unsigned(di._dims.size());
	const size_t cellCount = di.dataSize() / nbVariable;
	const int64_t neverInformed = std::numeric_limits<int64_t>::max();
	const bool batchMode = setup.batchSize > 0;
	const unsigned nbWorkers = batchMode ? 2 * setup.batchSize : std::max(1u, setup.nbThreads);

	// order[cell]: -1 = known before the simulation, k = k-th node of the path,
	// neverInformed = stays unknown (not in the path and no data).
	std::vector<int64_t> order(cellCount, neverInformed);
	for (size_t cell = 0; cell < cellCount; ++cell) {
		for (unsigned v = 0; v < nbVariable; ++v) {
			if (!std::isnan(di._data[cell * nbVariable + v])) { order[cell] = -1; break; }
		}
	}
	for (g2s_path_index_t i = 0; i < setup.pathLength; ++i) order[setup.path[i]] = int64_t(i);

	std::vector<std::atomic<uint8_t> > done(cellCount);
	for (size_t cell = 0; cell < cellCount; ++cell) done[cell].store(order[cell] < 0 ? 1 : 0, std::memory_order_relaxed);

	std::atomic<g2s_path_index_t> nextIndex(0);
	std::atomic<g2s_path_index_t> completed(0);
	std::atomic<bool> failed(false);
	std::atomic<unsigned long long> pythonCalls(0), predictedNodes(0);
	std::mutex errorMutex;
	const g2s_path_index_t displayRatio = std::max(setup.pathLength / g2s_path_index_t(100), g2s_path_index_t(1));

	auto fail = [&](const std::string& message) {
		std::lock_guard<std::mutex> lock(errorMutex);
		if (!failed.exchange(true)) error = message;
	};

	auto reportProgress = [&](g2s_path_index_t count) {
		const g2s_path_index_t before = completed.fetch_add(count, std::memory_order_relaxed);
		const g2s_path_index_t after = before + count;
		if (after / displayRatio != before / displayRatio || after == setup.pathLength) {
			g2s::reporting::setProgress(reportFile, 100.0 * double(after) / double(setup.pathLength), "simulation_vector",
				"cell " + std::to_string((unsigned long long)after) + " of " + std::to_string((unsigned long long)setup.pathLength),
				(long long)after, (long long)setup.pathLength);
		}
	};

	auto writeResult = [&](unsigned cell, const SampleResult& result) {
		for (unsigned v = 0; v < nbVariable; ++v) {
			if (std::isnan(di._data[cell * nbVariable + v]))
				di._data[cell * nbVariable + v] = float(result.values[v]);
		}
		importDataIndex[cell] = result.index;
		done[cell].store(1, std::memory_order_release);
	};

	// ---- batch state (batch mode only)
	std::mutex batchMutex;          // guards openBatch, openCells, searching, inFlight
	std::mutex predictMutex;        // one sample_batch() call at a time
	std::mutex writtenMutex;        // with batchWritten: wakes blocked workers
	std::condition_variable batchWritten;
	std::vector<Algorithm::BatchItem> openBatch;
	std::vector<unsigned> openCells;
	unsigned searching = nbWorkers; // workers neither blocked nor finished
	unsigned inFlight = 0;          // batches taken but not yet written
	std::vector<int64_t> waitingOn(nbWorkers, -1); // cell a blocked worker waits for

	// Must be called with batchMutex held; returns the batch to send, if any.
	auto takeBatchIfDue = [&](bool force, std::vector<Algorithm::BatchItem>& items, std::vector<unsigned>& cells) {
		if (openBatch.empty()) return false;
		const bool full = openBatch.size() >= setup.batchSize;
		const bool nobodyCanAdd = searching == 0 && inFlight == 0;
		if (!force && !full && !nobodyCanAdd) return false;
		inFlight++;
		items.swap(openBatch);
		cells.swap(openCells);
		openBatch.clear();
		openCells.clear();
		return true;
	};

	// With batchMutex held: workers whose awaited cell is now done count as
	// searching again right away (before they actually wake up), so the
	// "nobody can add" rule does not fire while they are still waking.
	auto reactivateUnblocked = [&]() {
		for (unsigned w = 0; w < nbWorkers; ++w) {
			if (waitingOn[w] >= 0 && done[size_t(waitingOn[w])].load(std::memory_order_acquire)) {
				waitingOn[w] = -1;
				searching++;
			}
		}
	};

	auto wakeBlocked = [&]() {
		{ std::lock_guard<std::mutex> lock(writtenMutex); }
		batchWritten.notify_all();
	};

	// Predicts a taken batch, writes it, then sends the next one if the batch
	// that just finished was the only thing everybody was waiting for.
	auto sendBatch = [&](AttachedWorker& python, std::vector<Algorithm::BatchItem>& items, std::vector<unsigned>& cells) {
		while (!items.empty()) {
			std::vector<SampleResult> results;
			std::string localError;
			bool ok = false;
			{
				std::lock_guard<std::mutex> lock(predictMutex);
				if (!failed.load()) {
					AttachedWorker::Attach attached(python);
					ok = algorithm.sampleBatch(items, results, localError);
				}
			}
			if (!ok) {
				if (!failed.load()) fail(localError);
				wakeBlocked();
				return;
			}
			for (size_t b = 0; b < cells.size(); ++b) writeResult(cells[b], results[b]);
			pythonCalls.fetch_add(1, std::memory_order_relaxed);
			predictedNodes.fetch_add(items.size(), std::memory_order_relaxed);
			reportProgress(g2s_path_index_t(items.size()));
			items.clear();
			cells.clear();
			{
				std::lock_guard<std::mutex> lock(batchMutex);
				inFlight--;
				reactivateUnblocked();
				takeBatchIfDue(false, items, cells);
			}
			wakeBlocked();
		}
	};

	// Batch mode: a worker blocks on `cell`; if that leaves nobody able to add a
	// node, it sends the open batch itself.
	auto blockOn = [&](AttachedWorker& python, unsigned threadId, unsigned cell) {
		std::vector<Algorithm::BatchItem> items;
		std::vector<unsigned> cells;
		bool send;
		{
			std::lock_guard<std::mutex> lock(batchMutex);
			searching--;
			waitingOn[threadId] = int64_t(cell);
			send = takeBatchIfDue(false, items, cells);
		}
		if (send) sendBatch(python, items, cells);
	};
	auto unblock = [&](unsigned threadId) {
		std::lock_guard<std::mutex> lock(batchMutex);
		if (waitingOn[threadId] >= 0) { // not reactivated by the writer (e.g. failure)
			waitingOn[threadId] = -1;
			searching++;
		}
	};

	auto worker = [&](unsigned threadId) {
		AttachedWorker python;
		std::vector<int> deltaMemory(nbDims);
		std::vector<int> rowOffsets;
		std::vector<double> rowValues;
		std::vector<unsigned> counts(nbVariable);
		SampleResult result;
		std::string localError;

		while (!failed.load(std::memory_order_relaxed)) {
			const g2s_path_index_t indexPath = nextIndex.fetch_add(1, std::memory_order_relaxed);
			if (indexPath >= setup.pathLength) break;
			const unsigned currentCell = unsigned(setup.path[indexPath]);
			const int64_t currentOrder = int64_t(indexPath);

			bool complete = true;
			for (unsigned v = 0; v < nbVariable; ++v) complete &= !std::isnan(di._data[currentCell * nbVariable + v]);
			if (complete) {
				done[currentCell].store(1, std::memory_order_release);
				if (batchMode) {
					{ std::lock_guard<std::mutex> lock(batchMutex); reactivateUnblocked(); }
					wakeBlocked();
				}
				reportProgress(1);
				continue;
			}

			// ---- neighbour search (detached from Python)
			rowOffsets.assign(nbDims, 0);
			rowValues.assign(nbVariable, std::nan(""));
			for (unsigned v = 0; v < nbVariable; ++v) rowValues[v] = di._data[currentCell * nbVariable + v];
			size_t rows = 1;
			std::fill(counts.begin(), counts.end(), 0u);
			auto needed = [&](unsigned v) { return counts[v] < setup.nbNeighbors[v % setup.nbNeighbors.size()]; };
			auto anyNeeded = [&]() { for (unsigned v = 0; v < nbVariable; ++v) if (needed(v)) return true; return false; };

			for (size_t p = 0; p < setup.offsets.size() && anyNeeded(); ++p) {
				unsigned dataIndex = 0;
				const bool inside = di.indexWithDelta(dataIndex, currentCell, setup.offsets[p], deltaMemory.data());
				if (!inside && !setup.circularSimulation) continue;
				const int64_t neighborOrder = order[dataIndex];
				if (neighborOrder >= currentOrder) continue;
				if (neighborOrder >= 0 && !done[dataIndex].load(std::memory_order_acquire)) {
					if (batchMode) {
						blockOn(python, threadId, dataIndex);
						std::unique_lock<std::mutex> lock(writtenMutex);
						batchWritten.wait(lock, [&]() { return done[dataIndex].load(std::memory_order_acquire) || failed.load(); });
						lock.unlock();
						unblock(threadId);
					} else {
						unsigned spins = 0;
						while (!done[dataIndex].load(std::memory_order_acquire) && !failed.load(std::memory_order_relaxed)) {
							if (++spins < 64) std::this_thread::yield();
							else std::this_thread::sleep_for(std::chrono::microseconds(50));
						}
					}
					if (failed.load(std::memory_order_relaxed)) break;
				}
				bool contributes = false;
				const size_t rowStart = rowValues.size();
				rowValues.resize(rowStart + nbVariable, std::nan(""));
				for (unsigned v = 0; v < nbVariable; ++v) {
					const float value = di._data[dataIndex * nbVariable + v];
					if (needed(v) && !std::isnan(value)) {
						rowValues[rowStart + v] = value;
						counts[v]++;
						contributes = true;
					}
				}
				if (!contributes) { rowValues.resize(rowStart); continue; }
				rowOffsets.insert(rowOffsets.end(), setup.offsets[p].begin(), setup.offsets[p].end());
				rows++;
			}
			if (failed.load(std::memory_order_relaxed)) break;

			const int tiIndex = setup.trainingIndexImage ? int(setup.trainingIndexImage->_data[currentCell * setup.trainingIndexImage->_nbVariable]) : -1;

			if (!batchMode) {
				bool ok;
				{
					AttachedWorker::Attach attached(python);
					ok = algorithm.sample(rowOffsets, rowValues, rows, setup.uniformSeeds[indexPath], uint64_t(indexPath), threadId, tiIndex, result, localError);
				}
				if (!ok) { fail(localError); break; }
				writeResult(currentCell, result);
				pythonCalls.fetch_add(1, std::memory_order_relaxed);
				predictedNodes.fetch_add(1, std::memory_order_relaxed);
				reportProgress(1);
				continue;
			}

			// ---- batch mode: queue the node and move on
			Algorithm::BatchItem item;
			item.offsets.swap(rowOffsets);
			item.values.swap(rowValues);
			item.rows = rows;
			item.randUniform = setup.uniformSeeds[indexPath];
			item.indexInThePath = uint64_t(indexPath);
			item.idTI4Sampling = tiIndex;
			std::vector<Algorithm::BatchItem> items;
			std::vector<unsigned> cells;
			bool send;
			{
				std::lock_guard<std::mutex> lock(batchMutex);
				openBatch.push_back(std::move(item));
				openCells.push_back(currentCell);
				send = takeBatchIfDue(false, items, cells);
			}
			if (send) sendBatch(python, items, cells);
		}

		if (batchMode) {
			// Finished (or failing): leave the searching pool; the last one out
			// sends what remains.
			std::vector<Algorithm::BatchItem> items;
			std::vector<unsigned> cells;
			bool send;
			{
				std::lock_guard<std::mutex> lock(batchMutex);
				searching--;
				send = takeBatchIfDue(searching == 0, items, cells);
			}
			if (send) sendBatch(python, items, cells);
		}
	};

	std::vector<std::thread> threads;
	threads.reserve(nbWorkers - 1);
	for (unsigned t = 1; t < nbWorkers; ++t) threads.emplace_back(worker, t);
	worker(0);
	for (std::thread& thread : threads) thread.join();

	stats.workers = nbWorkers;
	stats.pythonCalls = pythonCalls.load();
	stats.predictedNodes = predictedNodes.load();
	return !failed.load();
}

} // namespace pydev
} // namespace g2s

#endif // PY_DEV_SIMULATION_HPP
