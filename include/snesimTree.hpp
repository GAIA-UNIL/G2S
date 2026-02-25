#ifndef G2S_SNESIM_TREE_HPP
#define G2S_SNESIM_TREE_HPP

#include <string>
#include <vector>

#include "DataImage.hpp"

namespace snesim {

struct TreeNode {
	// One histogram entry per class for the central pixel category.
	std::vector<unsigned> categoryCounts;
	// One branch entry per class for the conditioning value.
	// -1 means no branch/subtree for that class.
	std::vector<int> childNodeIndex;
};

enum class MergePolicy {
	SumCounts = 0
};

enum class WildcardMode {
	Prefix = 0,
	Suffix = 1
};

class SearchTree {
public:
	SearchTree();
	SearchTree(const std::vector<int>& categories, const std::vector<TreeNode>& nodes);
	SearchTree(const std::vector<int>& categories,
		unsigned branchCount,
		const std::vector<unsigned>& categoryCountsFlat,
		const std::vector<int>& childNodeIndexFlat);

	const std::vector<int>& categories() const;
	size_t nodeCount() const;
	unsigned branchCount() const;
	const std::vector<unsigned>& categoryCountsFlat() const;
	const std::vector<int>& childNodeIndexFlat() const;
	const std::vector<TreeNode>& nodes() const;
	SearchTree deepCopy() const;
	void addStatisticsFrom(const SearchTree& other, MergePolicy policy = MergePolicy::SumCounts);

private:
	void assignFromNodes(const std::vector<TreeNode>& nodes);

	std::vector<int> _categories;
	unsigned _branchCount = 0u;
	std::vector<unsigned> _categoryCountsFlat;
	std::vector<int> _childNodeIndexFlat;
	mutable std::vector<TreeNode> _nodesCache;
	mutable bool _nodesCacheValid = false;
};

struct TreeBuildConfig {
	std::vector<int> templateRadius;
	unsigned gridLevel = 0;
	unsigned maxConditioningData = 0;
	unsigned branchCount = 0;
	bool wildcardEnabled = false;
	unsigned wildcardDepth = 0;
	WildcardMode wildcardMode = WildcardMode::Prefix;
};

struct TrainingImageSummary {
	std::string sourceName;
	std::string cacheName;
	std::vector<unsigned> dims;
	unsigned nbVariable = 0;
	std::vector<int> categories;
	std::vector<unsigned> categoryFrequency;
};

struct CachedTreeRecord {
	TrainingImageSummary summary;
	TreeBuildConfig config;
	SearchTree tree;
};

bool summarizeCategoricalTrainingImage(const g2s::DataImage& trainingImage,
	const std::string& sourceName,
	TrainingImageSummary& outSummary,
	std::string& errorMessage);

SearchTree buildBasicTree(const g2s::DataImage& trainingImage,
	const TrainingImageSummary& summary,
	const TreeBuildConfig& config);
SearchTree buildTreeForLevel(const g2s::DataImage& trainingImage,
	const TrainingImageSummary& summary,
	const std::vector<std::vector<int> >& pathPositionArray);
SearchTree buildTreeForLevel(const g2s::DataImage& trainingImage,
	const TrainingImageSummary& summary,
	const std::vector<std::vector<int> >& pathPositionArray,
	const TreeBuildConfig& config);
SearchTree mergeTrees(const std::vector<const SearchTree*>& trees,
	MergePolicy policy = MergePolicy::SumCounts);

bool ensureDirectory(const std::string& directory, std::string& errorMessage);
std::string sanitizePathToken(const std::string& rawName);

class TreeCacheRepository {
public:
	explicit TreeCacheRepository(const std::string& rootPath);

	const std::string& rootPath() const;
	std::string getTrainingImageCacheFolder(const std::string& trainingImageSourceName) const;
	std::string getTreeMetadataPath(const std::string& trainingImageSourceName, unsigned gridLevel) const;
	bool ensureCacheFolder(const std::string& trainingImageSourceName,
		std::string& cacheFolder,
		std::string& errorMessage) const;
	bool load(const std::string& trainingImageSourceName,
		CachedTreeRecord& outRecord,
		unsigned gridLevel,
		std::string& errorMessage) const;
	bool save(const std::string& trainingImageSourceName,
		const CachedTreeRecord& record,
		unsigned gridLevel,
		std::string& errorMessage) const;

private:
	std::string _rootPath;
};

} // namespace snesim

#endif // G2S_SNESIM_TREE_HPP
