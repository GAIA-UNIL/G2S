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
// informed neighbours of each node, and Python's sample() predicts the node.
//
// Neighbour rows passed to sample():
//   * row 0 is always the node itself (offset 0); its values are NaN except
//     for variables that are already known (hard data);
//   * the following rows are the closest informed cells (kernel order) until
//     every variable has n informed values or the kernel is exhausted; a value
//     that is not needed (its variable already has n values) is NaN.
// A cell is informed when it is hard data or when it comes strictly earlier in
// the path; workers wait (detached from Python) for earlier cells to finish, so
// the result does not depend on the number of threads.

#ifndef PY_DEV_SIMULATION_HPP
#define PY_DEV_SIMULATION_HPP

#include <atomic>
#include <chrono>
#include <cmath>
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
	unsigned nbThreads = 1;
	bool circularSimulation = false;
	std::vector<unsigned> nbNeighbors;
	// Sorted neighbour offsets in DataImage axis order, without the zero offset.
	std::vector<std::vector<int> > offsets;
	const g2s_path_index_t* path = nullptr; // cells to simulate, in order
	g2s_path_index_t pathLength = 0;
	const float* uniformSeeds = nullptr;    // one per path position
	g2s::DataImage* trainingIndexImage = nullptr; // -ii, optional
};

inline bool simulate(FILE* reportFile, g2s::DataImage& di, Algorithm& algorithm, const SimulationSetup& setup,
                     unsigned* importDataIndex, std::string& error) {
	const unsigned nbVariable = di._nbVariable;
	const unsigned nbDims = unsigned(di._dims.size());
	const size_t cellCount = di.dataSize() / nbVariable;
	const int64_t neverInformed = std::numeric_limits<int64_t>::max();

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
	std::mutex errorMutex;
	const g2s_path_index_t displayRatio = std::max(setup.pathLength / g2s_path_index_t(100), g2s_path_index_t(1));

	auto fail = [&](const std::string& message) {
		std::lock_guard<std::mutex> lock(errorMutex);
		if (!failed.exchange(true)) error = message;
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

			if (!complete) {
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
					if (neighborOrder >= 0) {
						unsigned spins = 0;
						while (!done[dataIndex].load(std::memory_order_acquire)) {
							if (failed.load(std::memory_order_relaxed)) return;
							if (++spins < 64) std::this_thread::yield();
							else std::this_thread::sleep_for(std::chrono::microseconds(50));
						}
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

				const int tiIndex = setup.trainingIndexImage ? int(setup.trainingIndexImage->_data[currentCell * setup.trainingIndexImage->_nbVariable]) : -1;
				bool ok;
				{
					AttachedWorker::Attach attached(python);
					ok = algorithm.sample(rowOffsets, rowValues, rows, setup.uniformSeeds[indexPath], uint64_t(indexPath), threadId, tiIndex, result, localError);
				}
				if (!ok) { fail(localError); return; }

				for (unsigned v = 0; v < nbVariable; ++v) {
					if (std::isnan(di._data[currentCell * nbVariable + v]))
						di._data[currentCell * nbVariable + v] = float(result.values[v]);
				}
				importDataIndex[currentCell] = result.index;
			}
			done[currentCell].store(1, std::memory_order_release);

			const g2s_path_index_t finished = completed.fetch_add(1, std::memory_order_relaxed) + 1;
			if (finished % displayRatio == 0 || finished == setup.pathLength) {
				g2s::reporting::setProgress(reportFile, 100.0 * double(finished) / double(setup.pathLength), "simulation_vector",
					"cell " + std::to_string((unsigned long long)finished) + " of " + std::to_string((unsigned long long)setup.pathLength),
					(long long)finished, (long long)setup.pathLength);
			}
		}
	};

	const unsigned nbThreads = std::max(1u, setup.nbThreads);
	std::vector<std::thread> threads;
	threads.reserve(nbThreads - 1);
	for (unsigned t = 1; t < nbThreads; ++t) threads.emplace_back(worker, t);
	worker(0);
	for (std::thread& thread : threads) thread.join();
	return !failed.load();
}

} // namespace pydev
} // namespace g2s

#endif // PY_DEV_SIMULATION_HPP
