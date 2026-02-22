#include "snesimCPUThreadDevice.hpp"

#include <algorithm>

namespace snesim {

std::mutex SNESIMCPUThreadDevice::_sharedTreeMutex;
std::map<unsigned, std::vector<std::shared_ptr<const SearchTree> > > SNESIMCPUThreadDevice::_sharedTreesByLevel;
std::map<unsigned, std::shared_ptr<const SearchTree> > SNESIMCPUThreadDevice::_mergedTreesByLevel;
std::atomic<unsigned> SNESIMCPUThreadDevice::_globalGridLevel(0u);
std::atomic<unsigned> SNESIMCPUThreadDevice::_treeSelectionMode(static_cast<unsigned>(TreeSelectionMode::PerTrainingImage));

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
	const std::vector<ConditioningDatum>& conditioningData,
	std::mt19937& randomGenerator) const {
	// TODO (real SNESIM):
	// - decode conditioningData into template order
	// - traverse child nodes in _sharedTree
	// - retrieve conditional probabilities at reached node
	// - sample category from that conditional distribution
	(void)conditioningData;

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

	const TreeNode& rootNode = nodes[0];
	if (rootNode.categoryCounts.size() != categories.size()) {
		return categories[0];
	}

	// Placeholder behavior: sample from root histogram only.
	// Full node traversal by conditioning pattern is intentionally deferred.
	std::vector<double> weights(rootNode.categoryCounts.size(), 1.0);
	bool hasPositiveWeight = false;
	for (size_t i = 0; i < rootNode.categoryCounts.size(); ++i) {
		weights[i] = static_cast<double>(rootNode.categoryCounts[i]);
		hasPositiveWeight = hasPositiveWeight || (weights[i] > 0.0);
	}
	if (!hasPositiveWeight) {
		return categories[0];
	}

	std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
	const size_t categoryIndex = distribution(randomGenerator);
	return categories[categoryIndex];
}

} // namespace snesim
