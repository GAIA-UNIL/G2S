#include "snesimCPUThreadDevice.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <sstream>

namespace {

int compareOffsetLexicographic(const std::vector<int>& left, const std::vector<int>& right) {
	const size_t commonSize = std::min(left.size(), right.size());
	for (size_t i = 0; i < commonSize; ++i) {
		if (left[i] < right[i]) {
			return -1;
		}
		if (left[i] > right[i]) {
			return 1;
		}
	}
	if (left.size() < right.size()) {
		return -1;
	}
	if (left.size() > right.size()) {
		return 1;
	}
	return 0;
}

void updateTotalStatFromNode(const snesim::TreeNode& node,
	int currentDepth,
	std::vector<unsigned long long>& totalStat,
	int& maxDepth) {
	if (currentDepth > maxDepth) {
		for (size_t classIndex = 0; classIndex < totalStat.size(); ++classIndex) {
			totalStat[classIndex] = static_cast<unsigned long long>(node.categoryCounts[classIndex]);
		}
		maxDepth = currentDepth;
		return;
	}

	if (currentDepth == maxDepth) {
		for (size_t classIndex = 0; classIndex < totalStat.size(); ++classIndex) {
			totalStat[classIndex] += static_cast<unsigned long long>(node.categoryCounts[classIndex]);
		}
	}
}

void depthSearchTreeRecursive(const std::vector<snesim::TreeNode>& nodes,
	int nodeIndex,
	int currentDepth,
	const std::vector<std::vector<int> >& pathPositionArray,
	size_t pathIndex,
	const std::vector<std::vector<int> >& neighborArrayVector,
	const std::vector<std::vector<float> >& neighborValueArrayVector,
	size_t neighborIndex,
	std::vector<unsigned long long>& totalStat,
	int& maxDepth) {
	if (nodeIndex < 0 || static_cast<size_t>(nodeIndex) >= nodes.size()) {
		return;
	}

	const snesim::TreeNode& node = nodes[static_cast<size_t>(nodeIndex)];
	if (node.categoryCounts.size() != totalStat.size() || node.childNodeIndex.size() != totalStat.size()) {
		return;
	}
	updateTotalStatFromNode(node, currentDepth, totalStat, maxDepth);

	if (pathIndex >= pathPositionArray.size()) {
		return;
	}

	const std::vector<int>& expectedOffset = pathPositionArray[pathIndex];
	const size_t alignedNeighborIndex = neighborIndex;
	const bool sameOffset = alignedNeighborIndex < neighborArrayVector.size()
		&& compareOffsetLexicographic(neighborArrayVector[alignedNeighborIndex], expectedOffset) == 0;

	bool hasKnownNeighbor = false;
	unsigned knownClassIndex = 0u;
	bool branchAll = true;
	if (sameOffset) {
		if (alignedNeighborIndex < neighborValueArrayVector.size() && !neighborValueArrayVector[alignedNeighborIndex].empty()) {
			hasKnownNeighbor = false;
			if (neighborValueArrayVector[alignedNeighborIndex].size() >= totalStat.size() && !totalStat.empty()) {
				int foundClass = -1;
				for (size_t classIndex = 0; classIndex < totalStat.size(); ++classIndex) {
					const float value = neighborValueArrayVector[alignedNeighborIndex][classIndex];
					if (!std::isfinite(value) || value <= 0.5f) {
						continue;
					}
					if (foundClass >= 0) {
						foundClass = -2;
						break;
					}
					foundClass = static_cast<int>(classIndex);
				}
				if (foundClass >= 0) {
					hasKnownNeighbor = true;
					knownClassIndex = static_cast<unsigned>(foundClass);
					branchAll = false;
				}
			}
		}
	}

	if (hasKnownNeighbor) {
		const int childNode = node.childNodeIndex[knownClassIndex];
		if (childNode < 0) {
			return;
		}
		depthSearchTreeRecursive(nodes,
			childNode,
			currentDepth + 1,
				pathPositionArray,
				pathIndex + 1,
				neighborArrayVector,
				neighborValueArrayVector,
				alignedNeighborIndex + 1,
				totalStat,
				maxDepth);
		return;
	}

	if (branchAll) {
		for (size_t classIndex = 0; classIndex < node.childNodeIndex.size(); ++classIndex) {
			const int childNode = node.childNodeIndex[classIndex];
			if (childNode < 0) {
				continue;
			}
			depthSearchTreeRecursive(nodes,
				childNode,
				currentDepth + 1,
				pathPositionArray,
					pathIndex + 1,
					neighborArrayVector,
					neighborValueArrayVector,
					sameOffset ?
							(alignedNeighborIndex + 1) :
							alignedNeighborIndex,
					totalStat,
					maxDepth);
			}
		}
}

std::string buildConditioningSequence(const std::vector<std::vector<int> >& pathPositionArray,
	const std::vector<std::vector<int> >& neighborArrayVector,
	const std::vector<std::vector<float> >& neighborValueArrayVector,
	const std::vector<int>& categories) {
	std::ostringstream stream;
	size_t neighborIndex = 0u;
	for (size_t pathIndex = 0; pathIndex < pathPositionArray.size(); ++pathIndex) {
		if (pathIndex > 0u) {
			stream << ",";
		}

		bool matched = false;
		if (neighborIndex < neighborArrayVector.size()) {
			matched = compareOffsetLexicographic(neighborArrayVector[neighborIndex], pathPositionArray[pathIndex]) == 0;
		}
		if (!matched) {
			stream << "_";
			continue;
		}

		bool isKnownCategory = false;
		int decodedClass = -1;
		if (neighborIndex < neighborValueArrayVector.size() && !neighborValueArrayVector[neighborIndex].empty()) {
			if (neighborValueArrayVector[neighborIndex].size() >= categories.size() && !categories.empty()) {
				for (size_t classIndex = 0; classIndex < categories.size(); ++classIndex) {
					const float value = neighborValueArrayVector[neighborIndex][classIndex];
					if (!std::isfinite(value) || value <= 0.5f) {
						continue;
					}
					if (decodedClass >= 0) {
						decodedClass = -2;
						break;
					}
					decodedClass = static_cast<int>(classIndex);
				}
				isKnownCategory = decodedClass >= 0;
			}
		}
		if (isKnownCategory) {
			stream << categories[static_cast<size_t>(decodedClass)];
		} else {
			stream << "_";
		}

		++neighborIndex;
	}

	return stream.str();
}

std::string buildStatsString(const std::vector<unsigned long long>& values) {
	std::ostringstream stream;
	stream << "[";
	for (size_t i = 0; i < values.size(); ++i) {
		if (i > 0u) {
			stream << ",";
		}
		stream << values[i];
	}
	stream << "]";
	return stream.str();
}

} // namespace

namespace snesim {

std::mutex SNESIMCPUThreadDevice::_sharedTreeMutex;
std::map<unsigned, std::vector<std::shared_ptr<const SearchTree> > > SNESIMCPUThreadDevice::_sharedTreesByLevel;
std::map<unsigned, std::shared_ptr<const SearchTree> > SNESIMCPUThreadDevice::_mergedTreesByLevel;
std::map<unsigned, std::vector<std::vector<int> > > SNESIMCPUThreadDevice::_pathPositionArrayByLevel;
std::atomic<unsigned> SNESIMCPUThreadDevice::_globalGridLevel(0u);
std::atomic<unsigned> SNESIMCPUThreadDevice::_treeSelectionMode(static_cast<unsigned>(TreeSelectionMode::PerTrainingImage));
static FILE* g_traceReportFile = stderr;
static std::mutex g_traceReportMutex;

SNESIMCPUThreadDevice::SNESIMCPUThreadDevice(unsigned workerId, std::shared_ptr<const SearchTree> sharedTree) :
	_workerId(workerId),
	_initialGridLevel(0),
	_fallbackTree(sharedTree) {
	// Keep constructor lightweight. Shared level trees are managed through static methods.
	// This preserves a single immutable tree memory block across all worker objects.
	if (sharedTree && !hasSharedTreeForLevel(0u, 0u)) {
		setSharedTreesForLevel(0u, std::vector<std::shared_ptr<const SearchTree> >(1u, sharedTree));
	}
}

unsigned SNESIMCPUThreadDevice::workerId() const {
	return _workerId;
}

unsigned SNESIMCPUThreadDevice::gridLevel() const {
	return globalGridLevel();
}

void SNESIMCPUThreadDevice::clearSharedTrees() {
	std::lock_guard<std::mutex> guard(_sharedTreeMutex);
	_sharedTreesByLevel.clear();
	_mergedTreesByLevel.clear();
	_pathPositionArrayByLevel.clear();
}

void SNESIMCPUThreadDevice::setSharedTreesForLevel(unsigned gridLevel, const std::vector<std::shared_ptr<const SearchTree> >& treesByTrainingImage) {
	std::lock_guard<std::mutex> guard(_sharedTreeMutex);
	_sharedTreesByLevel[gridLevel] = treesByTrainingImage;
}

void SNESIMCPUThreadDevice::setSharedTrees(const std::map<unsigned, std::vector<std::shared_ptr<const SearchTree> > >& treesByLevel) {
	std::lock_guard<std::mutex> guard(_sharedTreeMutex);
	_sharedTreesByLevel.clear();
	for (std::map<unsigned, std::vector<std::shared_ptr<const SearchTree> > >::const_iterator it = treesByLevel.begin(); it != treesByLevel.end(); ++it) {
		_sharedTreesByLevel[it->first] = it->second;
	}
}

bool SNESIMCPUThreadDevice::hasSharedTreeForLevel(unsigned gridLevel, unsigned trainingImageIndex) {
	std::lock_guard<std::mutex> guard(_sharedTreeMutex);
	std::map<unsigned, std::vector<std::shared_ptr<const SearchTree> > >::const_iterator levelIt = _sharedTreesByLevel.find(gridLevel);
	if (levelIt == _sharedTreesByLevel.end()) {
		return false;
	}
	return trainingImageIndex < levelIt->second.size() && levelIt->second[trainingImageIndex];
}

std::shared_ptr<const SearchTree> SNESIMCPUThreadDevice::sharedTreeForLevel(unsigned gridLevel, unsigned trainingImageIndex) {
	std::lock_guard<std::mutex> guard(_sharedTreeMutex);
	std::map<unsigned, std::vector<std::shared_ptr<const SearchTree> > >::const_iterator levelIt = _sharedTreesByLevel.find(gridLevel);
	if (levelIt == _sharedTreesByLevel.end()) {
		return std::shared_ptr<const SearchTree>();
	}
	if (trainingImageIndex >= levelIt->second.size()) {
		return std::shared_ptr<const SearchTree>();
	}
	return levelIt->second[trainingImageIndex];
}

void SNESIMCPUThreadDevice::setMergedTreeForLevel(unsigned gridLevel, std::shared_ptr<const SearchTree> mergedTree) {
	if (!mergedTree) {
		return;
	}
	std::lock_guard<std::mutex> guard(_sharedTreeMutex);
	_mergedTreesByLevel[gridLevel] = mergedTree;
}

bool SNESIMCPUThreadDevice::hasMergedTreeForLevel(unsigned gridLevel) {
	std::lock_guard<std::mutex> guard(_sharedTreeMutex);
	return _mergedTreesByLevel.find(gridLevel) != _mergedTreesByLevel.end();
}

std::shared_ptr<const SearchTree> SNESIMCPUThreadDevice::mergedTreeForLevel(unsigned gridLevel) {
	std::lock_guard<std::mutex> guard(_sharedTreeMutex);
	std::map<unsigned, std::shared_ptr<const SearchTree> >::const_iterator it = _mergedTreesByLevel.find(gridLevel);
	if (it != _mergedTreesByLevel.end()) {
		return it->second;
	}
	return std::shared_ptr<const SearchTree>();
}

void SNESIMCPUThreadDevice::setPathPositionArrayForLevel(unsigned gridLevel, const std::vector<std::vector<int> >& pathPositionArray) {
	std::lock_guard<std::mutex> guard(_sharedTreeMutex);
	_pathPositionArrayByLevel[gridLevel] = pathPositionArray;
}

bool SNESIMCPUThreadDevice::hasPathPositionArrayForLevel(unsigned gridLevel) {
	std::lock_guard<std::mutex> guard(_sharedTreeMutex);
	return _pathPositionArrayByLevel.find(gridLevel) != _pathPositionArrayByLevel.end();
}

std::vector<std::vector<int> > SNESIMCPUThreadDevice::pathPositionArrayForLevel(unsigned gridLevel) {
	std::lock_guard<std::mutex> guard(_sharedTreeMutex);
	std::map<unsigned, std::vector<std::vector<int> > >::const_iterator it = _pathPositionArrayByLevel.find(gridLevel);
	if (it != _pathPositionArrayByLevel.end()) {
		return it->second;
	}
	return std::vector<std::vector<int> >();
}

void SNESIMCPUThreadDevice::setTraceReportFile(FILE* reportFile) {
	std::lock_guard<std::mutex> guard(g_traceReportMutex);
	g_traceReportFile = (reportFile != nullptr) ? reportFile : stderr;
}

void SNESIMCPUThreadDevice::setGlobalGridLevel(unsigned gridLevel) {
	// Global level switch for all workers before each simulation level pass.
	_globalGridLevel.store(gridLevel, std::memory_order_relaxed);
}

unsigned SNESIMCPUThreadDevice::globalGridLevel() {
	return _globalGridLevel.load(std::memory_order_relaxed);
}

void SNESIMCPUThreadDevice::setTreeSelectionMode(TreeSelectionMode mode) {
	_treeSelectionMode.store(static_cast<unsigned>(mode), std::memory_order_relaxed);
}

TreeSelectionMode SNESIMCPUThreadDevice::treeSelectionMode() {
	return static_cast<TreeSelectionMode>(_treeSelectionMode.load(std::memory_order_relaxed));
}

std::shared_ptr<const SearchTree> SNESIMCPUThreadDevice::resolveActiveTree(unsigned trainingImageIndex) const {
	// Resolution order:
	// 1) merged tree at current level (if merged mode)
	// 2) TI-specific tree at current level
	// 3) constructor fallback tree
	// 4) first available shared tree across levels
	const unsigned activeLevel = globalGridLevel();
	if (treeSelectionMode() == TreeSelectionMode::Merged) {
		std::shared_ptr<const SearchTree> mergedTree = mergedTreeForLevel(activeLevel);
		if (mergedTree) {
			return mergedTree;
		}
	}
	std::shared_ptr<const SearchTree> levelTree = sharedTreeForLevel(activeLevel, trainingImageIndex);
	if (levelTree) {
		return levelTree;
	}
	if (_fallbackTree) {
		return _fallbackTree;
	}
	{
		std::lock_guard<std::mutex> guard(_sharedTreeMutex);
		for (std::map<unsigned, std::vector<std::shared_ptr<const SearchTree> > >::const_iterator levelIt = _sharedTreesByLevel.begin(); levelIt != _sharedTreesByLevel.end(); ++levelIt) {
			for (size_t tiIndex = 0; tiIndex < levelIt->second.size(); ++tiIndex) {
				if (levelIt->second[tiIndex]) {
					return levelIt->second[tiIndex];
				}
			}
		}
	}
	return std::shared_ptr<const SearchTree>();
}

int SNESIMCPUThreadDevice::simulatePixel(const g2s::DataImage& /*simulationGrid*/,
	unsigned /*cellIndex*/,
	unsigned /*variableIndex*/,
	unsigned trainingImageIndex,
	const std::vector<std::vector<int> >& neighborArrayVector,
	const std::vector<std::vector<float> >& neighborValueArrayVector,
	std::mt19937& randomGenerator) const {
	const std::shared_ptr<const SearchTree> activeTree = resolveActiveTree(trainingImageIndex);
	if (!activeTree) {
		return 0;
	}

	const std::vector<int>& categories = activeTree->categories();
	if (categories.empty()) {
		return 0;
	}

	const std::vector<TreeNode>& nodes = activeTree->nodes();
	if (nodes.empty()) {
		return categories[0];
	}

	if (nodes[0].categoryCounts.size() != categories.size() || nodes[0].childNodeIndex.size() != categories.size()) {
		return categories[0];
	}

	// Depth search over tree with branch-all whenever neighborhood has NaN/missing value.
	const unsigned activeLevel = globalGridLevel();
	const std::vector<std::vector<int> > pathPositionArray = pathPositionArrayForLevel(activeLevel);
	std::vector<unsigned long long> totalStat(categories.size(), 0ULL);
	int maxDepth = -1;
	depthSearchTreeRecursive(nodes,
		0,
		0,
		pathPositionArray,
		0u,
		neighborArrayVector,
		neighborValueArrayVector,
		0u,
		totalStat,
		maxDepth);

	std::vector<double> weights(totalStat.size(), 0.0);
	bool hasPositiveWeight = false;
	for (size_t i = 0; i < totalStat.size(); ++i) {
		weights[i] = static_cast<double>(totalStat[i]);
		hasPositiveWeight = hasPositiveWeight || (weights[i] > 0.0);
	}
	if (!hasPositiveWeight) {
		// Safety fallback should not happen, because root depth (0) should always seed totalStat.
		return categories[0];
	}

	std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
	const size_t categoryIndex = distribution(randomGenerator);
	const int sampledCategory = categories[categoryIndex];

	// Debug trace intentionally disabled for regular runs.
	// {
	// 	std::lock_guard<std::mutex> guard(g_traceReportMutex);
	// 	std::fprintf(g_traceReportFile,
	// 		"[SNESIM_TRACE] level=%u worker=%u ti=%u conditioning=%s maxDepth=%d globalStat=%s sample=%d\n",
	// 		activeLevel,
	// 		_workerId,
	// 		trainingImageIndex,
	// 		buildConditioningSequence(pathPositionArray, neighborArrayVector, neighborValueArrayVector, categories).c_str(),
	// 		maxDepth,
	// 		buildStatsString(totalStat).c_str(),
	// 		sampledCategory);
	// }

	return sampledCategory;
}

} // namespace snesim
