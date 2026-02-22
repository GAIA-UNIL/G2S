#ifndef G2S_SNESIM_CPU_THREAD_DEVICE_HPP
#define G2S_SNESIM_CPU_THREAD_DEVICE_HPP

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <vector>

#include "DataImage.hpp"
#include "snesimTree.hpp"

namespace snesim {

struct ConditioningDatum {
	unsigned location = 0;
	int category = 0;
};

enum class TreeSelectionMode {
	PerTrainingImage = 0,
	Merged = 1
};

// Thread-local worker that reads from an immutable shared SNESIM tree.
// Tree ownership is centralized in static members so all workers can
// switch level together without duplicating tree memory.
class SNESIMCPUThreadDevice {
public:
	SNESIMCPUThreadDevice(unsigned workerId, std::shared_ptr<const SearchTree> sharedTree);

	unsigned workerId() const;
	unsigned gridLevel() const;

	// Shared immutable tree memory management for all worker instances.
	static void clearSharedTrees();
	static void setSharedTreesForLevel(unsigned gridLevel, const std::vector<std::shared_ptr<const SearchTree> >& treesByTrainingImage);
	static void setSharedTrees(const std::map<unsigned, std::vector<std::shared_ptr<const SearchTree> > >& treesByLevel);
	static bool hasSharedTreeForLevel(unsigned gridLevel, unsigned trainingImageIndex);
	static std::shared_ptr<const SearchTree> sharedTreeForLevel(unsigned gridLevel, unsigned trainingImageIndex);
	static void setMergedTreeForLevel(unsigned gridLevel, std::shared_ptr<const SearchTree> mergedTree);
	static bool hasMergedTreeForLevel(unsigned gridLevel);
	static std::shared_ptr<const SearchTree> mergedTreeForLevel(unsigned gridLevel);

	// Global level switch: one call updates the active level for all workers.
	static void setGlobalGridLevel(unsigned gridLevel);
	static unsigned globalGridLevel();
	static void setTreeSelectionMode(TreeSelectionMode mode);
	static TreeSelectionMode treeSelectionMode();

	int simulatePixel(const g2s::DataImage& simulationGrid,
		unsigned cellIndex,
		unsigned variableIndex,
		unsigned trainingImageIndex,
		const std::vector<ConditioningDatum>& conditioningData,
		std::mt19937& randomGenerator) const;

private:
	std::shared_ptr<const SearchTree> resolveActiveTree(unsigned trainingImageIndex) const;

	unsigned _workerId = 0;
	unsigned _initialGridLevel = 0;
	std::shared_ptr<const SearchTree> _fallbackTree;

	// Shared tree registry: level -> immutable tree handles by TI index.
	static std::mutex _sharedTreeMutex;
	static std::map<unsigned, std::vector<std::shared_ptr<const SearchTree> > > _sharedTreesByLevel;
	// Optional merged tree per level (recomputed per run).
	static std::map<unsigned, std::shared_ptr<const SearchTree> > _mergedTreesByLevel;
	// One global level value used by all workers in a level pass.
	static std::atomic<unsigned> _globalGridLevel;
	static std::atomic<unsigned> _treeSelectionMode;
};

} // namespace snesim

#endif // G2S_SNESIM_CPU_THREAD_DEVICE_HPP
