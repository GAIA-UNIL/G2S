#include "snesimCPUThreadDevice.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

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

bool decodeKnownClass(const std::vector<float>& encodedNeighbor,
	size_t classCount,
	unsigned& knownClassIndex) {
	if (encodedNeighbor.size() < classCount || classCount == 0u) {
		return false;
	}

	int foundClass = -1;
	for (size_t classIndex = 0; classIndex < classCount; ++classIndex) {
		const float value = encodedNeighbor[classIndex];
		if (!std::isfinite(value) || value <= 0.5f) {
			continue;
		}
		if (foundClass >= 0) {
			return false;
		}
		foundClass = static_cast<int>(classIndex);
	}
	if (foundClass < 0) {
		return false;
	}
	knownClassIndex = static_cast<unsigned>(foundClass);
	return true;
}

bool isWildcardActiveAtDepth(size_t pathIndex,
	size_t pathSize,
	bool wildcardEnabled,
	unsigned wildcardDepth,
	snesim::WildcardMode wildcardMode,
	unsigned wildcardBranchIndex,
	size_t branchCount) {
	if (!wildcardEnabled || wildcardBranchIndex >= branchCount) {
		return false;
	}
	if (wildcardMode == snesim::WildcardMode::Prefix) {
		return pathIndex < static_cast<size_t>(wildcardDepth);
	}
	const size_t depth = static_cast<size_t>(wildcardDepth);
	const size_t suffixStart = (depth >= pathSize) ? 0u : (pathSize - depth);
	return pathIndex >= suffixStart;
}

void depthSearchTreeRecursive(const std::vector<snesim::TreeNode>& nodes,
	int nodeIndex,
	int currentDepth,
	const std::vector<std::vector<int> >& pathPositionArray,
	size_t pathIndex,
	const std::vector<std::vector<int> >& neighborArrayVector,
	const std::vector<std::vector<float> >& neighborValueArrayVector,
	size_t neighborIndex,
	bool wildcardEnabled,
	unsigned wildcardDepth,
	snesim::WildcardMode wildcardMode,
	unsigned wildcardBranchIndex,
	std::vector<unsigned long long>& totalStat,
	int& maxDepth) {
	if (nodeIndex < 0 || static_cast<size_t>(nodeIndex) >= nodes.size()) {
		return;
	}

	const snesim::TreeNode& node = nodes[static_cast<size_t>(nodeIndex)];
	const size_t classCount = totalStat.size();
	if (node.categoryCounts.size() != classCount || node.childNodeIndex.size() < classCount) {
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
	const size_t nextNeighborIndex = sameOffset ? (alignedNeighborIndex + 1u) : alignedNeighborIndex;
	const bool allowWildcardAtDepth = isWildcardActiveAtDepth(pathIndex,
		pathPositionArray.size(),
		wildcardEnabled,
		wildcardDepth,
		wildcardMode,
		wildcardBranchIndex,
		node.childNodeIndex.size());

	bool hasKnownNeighbor = false;
	unsigned knownClassIndex = 0u;
	if (sameOffset) {
		if (alignedNeighborIndex < neighborValueArrayVector.size() && !neighborValueArrayVector[alignedNeighborIndex].empty()) {
			hasKnownNeighbor = decodeKnownClass(neighborValueArrayVector[alignedNeighborIndex], classCount, knownClassIndex);
		}
	}

	if (hasKnownNeighbor) {
		if (knownClassIndex >= classCount) {
			return;
		}
		const int knownChild = node.childNodeIndex[knownClassIndex];
		if (knownChild >= 0) {
			depthSearchTreeRecursive(nodes,
				knownChild,
				currentDepth + 1,
				pathPositionArray,
				pathIndex + 1,
				neighborArrayVector,
				neighborValueArrayVector,
				nextNeighborIndex,
				wildcardEnabled,
				wildcardDepth,
				wildcardMode,
				wildcardBranchIndex,
				totalStat,
				maxDepth);
		}
		// Known neighbor values do not fallback to wildcard.
		// If the known class branch is missing, traversal stops here.
		return;
	}

	if (allowWildcardAtDepth) {
		const int wildcardChild = node.childNodeIndex[wildcardBranchIndex];
		if (wildcardChild < 0) {
			return;
		}
		depthSearchTreeRecursive(nodes,
			wildcardChild,
			currentDepth + 1,
			pathPositionArray,
			pathIndex + 1,
			neighborArrayVector,
				neighborValueArrayVector,
				nextNeighborIndex,
				wildcardEnabled,
				wildcardDepth,
				wildcardMode,
				wildcardBranchIndex,
				totalStat,
				maxDepth);
		return;
	}

	// Strict SNESIM behavior for non-wildcard depths: branch-all on missing/unknown.
	for (size_t classIndex = 0; classIndex < classCount; ++classIndex) {
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
			nextNeighborIndex,
			wildcardEnabled,
			wildcardDepth,
			wildcardMode,
			wildcardBranchIndex,
			totalStat,
			maxDepth);
	}
}

} // namespace

namespace snesim {

std::mutex SNESIMCPUThreadDevice::_sharedTreeMutex;
std::map<unsigned, std::vector<std::shared_ptr<const SearchTree> > > SNESIMCPUThreadDevice::_sharedTreesByLevel;
std::map<unsigned, std::shared_ptr<const SearchTree> > SNESIMCPUThreadDevice::_mergedTreesByLevel;
std::map<unsigned, std::shared_ptr<const std::vector<std::vector<int> > > > SNESIMCPUThreadDevice::_pathPositionArrayByLevel;
std::atomic<unsigned> SNESIMCPUThreadDevice::_globalGridLevel(0u);
std::atomic<unsigned> SNESIMCPUThreadDevice::_treeSelectionMode(static_cast<unsigned>(TreeSelectionMode::PerTrainingImage));
std::atomic<unsigned> SNESIMCPUThreadDevice::_wildcardEnabled(0u);
std::atomic<unsigned> SNESIMCPUThreadDevice::_wildcardDepth(0u);
std::atomic<unsigned> SNESIMCPUThreadDevice::_wildcardMode(static_cast<unsigned>(WildcardMode::Prefix));
std::atomic<unsigned long long> SNESIMCPUThreadDevice::_sharedStateVersion(1u);

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
	_sharedStateVersion.fetch_add(1u, std::memory_order_relaxed);
}

void SNESIMCPUThreadDevice::setSharedTreesForLevel(unsigned gridLevel, const std::vector<std::shared_ptr<const SearchTree> >& treesByTrainingImage) {
	std::lock_guard<std::mutex> guard(_sharedTreeMutex);
	_sharedTreesByLevel[gridLevel] = treesByTrainingImage;
	_sharedStateVersion.fetch_add(1u, std::memory_order_relaxed);
}

void SNESIMCPUThreadDevice::setSharedTrees(const std::map<unsigned, std::vector<std::shared_ptr<const SearchTree> > >& treesByLevel) {
	std::lock_guard<std::mutex> guard(_sharedTreeMutex);
	_sharedTreesByLevel.clear();
	for (std::map<unsigned, std::vector<std::shared_ptr<const SearchTree> > >::const_iterator it = treesByLevel.begin(); it != treesByLevel.end(); ++it) {
		_sharedTreesByLevel[it->first] = it->second;
	}
	_sharedStateVersion.fetch_add(1u, std::memory_order_relaxed);
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
	_sharedStateVersion.fetch_add(1u, std::memory_order_relaxed);
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
	_pathPositionArrayByLevel[gridLevel] = std::make_shared<std::vector<std::vector<int> > >(pathPositionArray);
	_sharedStateVersion.fetch_add(1u, std::memory_order_relaxed);
}

bool SNESIMCPUThreadDevice::hasPathPositionArrayForLevel(unsigned gridLevel) {
	std::lock_guard<std::mutex> guard(_sharedTreeMutex);
	std::map<unsigned, std::shared_ptr<const std::vector<std::vector<int> > > >::const_iterator it = _pathPositionArrayByLevel.find(gridLevel);
	return it != _pathPositionArrayByLevel.end() && static_cast<bool>(it->second);
}

std::shared_ptr<const std::vector<std::vector<int> > > SNESIMCPUThreadDevice::pathPositionArrayHandleForLevel(unsigned gridLevel) {
	std::lock_guard<std::mutex> guard(_sharedTreeMutex);
	std::map<unsigned, std::shared_ptr<const std::vector<std::vector<int> > > >::const_iterator it = _pathPositionArrayByLevel.find(gridLevel);
	if (it != _pathPositionArrayByLevel.end()) {
		return it->second;
	}
	return std::shared_ptr<const std::vector<std::vector<int> > >();
}

std::vector<std::vector<int> > SNESIMCPUThreadDevice::pathPositionArrayForLevel(unsigned gridLevel) {
	const std::shared_ptr<const std::vector<std::vector<int> > > pathPositionArray = pathPositionArrayHandleForLevel(gridLevel);
	if (pathPositionArray) {
		return *pathPositionArray;
	}
	return std::vector<std::vector<int> >();
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

void SNESIMCPUThreadDevice::setWildcardConfig(bool enabled, unsigned depth, WildcardMode mode) {
	_wildcardEnabled.store(enabled ? 1u : 0u, std::memory_order_relaxed);
	_wildcardDepth.store(enabled ? depth : 0u, std::memory_order_relaxed);
	_wildcardMode.store(enabled ? static_cast<unsigned>(mode) : static_cast<unsigned>(WildcardMode::Prefix), std::memory_order_relaxed);
}

bool SNESIMCPUThreadDevice::wildcardEnabled() {
	return _wildcardEnabled.load(std::memory_order_relaxed) != 0u;
}

unsigned SNESIMCPUThreadDevice::wildcardDepth() {
	return _wildcardDepth.load(std::memory_order_relaxed);
}

WildcardMode SNESIMCPUThreadDevice::wildcardMode() {
	return static_cast<WildcardMode>(_wildcardMode.load(std::memory_order_relaxed));
}

void SNESIMCPUThreadDevice::refreshLevelCache(unsigned activeLevel) const {
	const unsigned long long sharedStateVersion = _sharedStateVersion.load(std::memory_order_relaxed);
	if (_cachedLevelValid
		&& _cachedLevel == activeLevel
		&& _cachedSharedStateVersion == sharedStateVersion) {
		return;
	}

	_cachedTreesByTrainingImage.clear();
	_cachedMergedTree.reset();
	_cachedAnySharedTree.reset();
	_cachedPathPositionArray.reset();

	std::lock_guard<std::mutex> guard(_sharedTreeMutex);
	std::map<unsigned, std::vector<std::shared_ptr<const SearchTree> > >::const_iterator levelTreeIt = _sharedTreesByLevel.find(activeLevel);
	if (levelTreeIt != _sharedTreesByLevel.end()) {
		_cachedTreesByTrainingImage = levelTreeIt->second;
		for (size_t tiIndex = 0; tiIndex < _cachedTreesByTrainingImage.size(); ++tiIndex) {
			if (_cachedTreesByTrainingImage[tiIndex]) {
				_cachedAnySharedTree = _cachedTreesByTrainingImage[tiIndex];
				break;
			}
		}
	}

	std::map<unsigned, std::shared_ptr<const SearchTree> >::const_iterator mergedTreeIt = _mergedTreesByLevel.find(activeLevel);
	if (mergedTreeIt != _mergedTreesByLevel.end()) {
		_cachedMergedTree = mergedTreeIt->second;
	}

	std::map<unsigned, std::shared_ptr<const std::vector<std::vector<int> > > >::const_iterator pathIt = _pathPositionArrayByLevel.find(activeLevel);
	if (pathIt != _pathPositionArrayByLevel.end()) {
		_cachedPathPositionArray = pathIt->second;
	}

	if (!_cachedAnySharedTree) {
		for (std::map<unsigned, std::vector<std::shared_ptr<const SearchTree> > >::const_iterator anyLevelIt = _sharedTreesByLevel.begin();
			anyLevelIt != _sharedTreesByLevel.end();
			++anyLevelIt) {
			for (size_t tiIndex = 0; tiIndex < anyLevelIt->second.size(); ++tiIndex) {
				if (anyLevelIt->second[tiIndex]) {
					_cachedAnySharedTree = anyLevelIt->second[tiIndex];
					break;
				}
			}
			if (_cachedAnySharedTree) {
				break;
			}
		}
	}

	_cachedLevel = activeLevel;
	_cachedLevelValid = true;
	_cachedSharedStateVersion = _sharedStateVersion.load(std::memory_order_relaxed);
}

std::shared_ptr<const SearchTree> SNESIMCPUThreadDevice::resolveActiveTree(unsigned trainingImageIndex, TreeSelectionMode mode) const {
	// Resolution order:
	// 1) merged tree at current level (if merged mode)
	// 2) TI-specific tree at current level
	// 3) constructor fallback tree
	// 4) first available shared tree
	if (mode == TreeSelectionMode::Merged && _cachedMergedTree) {
		return _cachedMergedTree;
	}
	if (trainingImageIndex < _cachedTreesByTrainingImage.size() && _cachedTreesByTrainingImage[trainingImageIndex]) {
		return _cachedTreesByTrainingImage[trainingImageIndex];
	}
	if (_fallbackTree) {
		return _fallbackTree;
	}
	return _cachedAnySharedTree;
}

int SNESIMCPUThreadDevice::simulatePixel(const g2s::DataImage& /*simulationGrid*/,
	unsigned /*cellIndex*/,
	unsigned /*variableIndex*/,
	unsigned trainingImageIndex,
	const std::vector<std::vector<int> >& neighborArrayVector,
	const std::vector<std::vector<float> >& neighborValueArrayVector,
	std::mt19937& randomGenerator) const {
	const unsigned activeLevel = globalGridLevel();
	refreshLevelCache(activeLevel);
	const TreeSelectionMode selectionMode = treeSelectionMode();
	const std::shared_ptr<const SearchTree> activeTree = resolveActiveTree(trainingImageIndex, selectionMode);
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

	if (nodes[0].categoryCounts.size() != categories.size() || nodes[0].childNodeIndex.size() < categories.size()) {
		return categories[0];
	}

	// Depth search over tree:
	// - strict mode branches all class children on missing values
	// - wildcard mode keeps known neighbors strict and routes unknown/missing through wildcard in active depths.
	static const std::vector<std::vector<int> > kEmptyPathPositionArray;
	const std::vector<std::vector<int> >& pathPositionArray = _cachedPathPositionArray
		? *_cachedPathPositionArray
		: kEmptyPathPositionArray;
	const bool useWildcard = wildcardEnabled() && wildcardDepth() > 0u;
	const unsigned wildcardDepthValue = wildcardDepth();
	const WildcardMode wildcardModeValue = wildcardMode();
	const unsigned wildcardBranchIndex = static_cast<unsigned>(categories.size());
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
		useWildcard,
		wildcardDepthValue,
		wildcardModeValue,
		wildcardBranchIndex,
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

	return sampledCategory;
}

} // namespace snesim
