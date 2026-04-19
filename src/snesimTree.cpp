#include "snesimTree.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

namespace {

const char* kLegacyTreeMetadataFile = "tree.meta";
const char* kTreeMetadataPrefix = "tree_level_";
const char* kTreeMetadataSuffix = ".meta";
const char* kDefaultTreeRoot = "/tmp/G2S/data/snesim_trees";
const double kCategoryTolerance = 1e-5;

std::string trim(const std::string& value) {
	std::string::size_type begin = 0;
	while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
		++begin;
	}

	std::string::size_type end = value.size();
	while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
		--end;
	}
	return value.substr(begin, end - begin);
}

std::string joinUnsignedVector(const std::vector<unsigned>& values) {
	std::ostringstream oss;
	for (size_t i = 0; i < values.size(); ++i) {
		if (i > 0) {
			oss << ",";
		}
		oss << values[i];
	}
	return oss.str();
}

std::string joinIntVector(const std::vector<int>& values) {
	std::ostringstream oss;
	for (size_t i = 0; i < values.size(); ++i) {
		if (i > 0) {
			oss << ",";
		}
		oss << values[i];
	}
	return oss.str();
}

bool parseUnsignedVector(const std::string& raw, std::vector<unsigned>& outValues) {
	outValues.clear();
	std::stringstream ss(raw);
	std::string item;
	while (std::getline(ss, item, ',')) {
		const std::string cleaned = trim(item);
		if (cleaned.empty()) {
			continue;
		}
		char* endPtr = nullptr;
		const unsigned long parsedValue = std::strtoul(cleaned.c_str(), &endPtr, 10);
		if (endPtr == nullptr || *endPtr != '\0' || parsedValue > std::numeric_limits<unsigned>::max()) {
			return false;
		}
		outValues.push_back(static_cast<unsigned>(parsedValue));
	}
	return true;
}

bool parseIntVector(const std::string& raw, std::vector<int>& outValues) {
	outValues.clear();
	std::stringstream ss(raw);
	std::string item;
	while (std::getline(ss, item, ',')) {
		const std::string cleaned = trim(item);
		if (cleaned.empty()) {
			continue;
		}
		char* endPtr = nullptr;
		const long parsedValue = std::strtol(cleaned.c_str(), &endPtr, 10);
		if (endPtr == nullptr || *endPtr != '\0' || parsedValue < std::numeric_limits<int>::min() || parsedValue > std::numeric_limits<int>::max()) {
			return false;
		}
		outValues.push_back(static_cast<int>(parsedValue));
	}
	return true;
}

bool parseUnsignedValue(const std::string& raw, unsigned& outValue) {
	const std::string cleaned = trim(raw);
	if (cleaned.empty()) {
		return false;
	}
	char* endPtr = nullptr;
	const unsigned long parsedValue = std::strtoul(cleaned.c_str(), &endPtr, 10);
	if (endPtr == nullptr || *endPtr != '\0' || parsedValue > std::numeric_limits<unsigned>::max()) {
		return false;
	}
	outValue = static_cast<unsigned>(parsedValue);
	return true;
}

bool parseKeyValueLine(const std::string& line, std::string& outKey, std::string& outValue) {
	const std::string trimmed = trim(line);
	if (trimmed.empty()) {
		return false;
	}
	const std::string::size_type separator = trimmed.find('=');
	if (separator == std::string::npos) {
		return false;
	}
	outKey = trim(trimmed.substr(0, separator));
	outValue = trim(trimmed.substr(separator + 1));
	return !outKey.empty();
}

bool fileExists(const std::string& path) {
	struct stat buffer;
	return stat(path.c_str(), &buffer) == 0;
}

unsigned computeCellCount(const std::vector<unsigned>& dims) {
	unsigned cellCount = 1u;
	for (size_t i = 0; i < dims.size(); ++i) {
		cellCount *= dims[i];
	}
	return cellCount;
}

snesim::TreeNode makeNode(size_t numberOfClasses) {
	snesim::TreeNode node;
	node.categoryCounts.assign(numberOfClasses, 0u);
	node.childNodeIndex.assign(numberOfClasses, -1);
	return node;
}

void incrementCount(unsigned& counter) {
	if (counter < std::numeric_limits<unsigned>::max()) {
		++counter;
	}
}

void flattenIndexToCoordinates(unsigned cellIndex,
	const std::vector<unsigned>& dims,
	std::vector<int>& coordinates) {
	coordinates.assign(dims.size(), 0);
	for (size_t dim = 0; dim < dims.size(); ++dim) {
		coordinates[dim] = static_cast<int>(cellIndex % dims[dim]);
		cellIndex /= dims[dim];
	}
}

bool tryCategoryToClassIndex(float rawValue,
	const std::map<int, unsigned>& categoryToClassIndex,
	unsigned& outClassIndex) {
	if (!std::isfinite(rawValue)) {
		return false;
	}
	const double roundedValue = std::round(static_cast<double>(rawValue));
	if (std::fabs(static_cast<double>(rawValue) - roundedValue) > kCategoryTolerance) {
		return false;
	}
	if (roundedValue < static_cast<double>(std::numeric_limits<int>::min())
		|| roundedValue > static_cast<double>(std::numeric_limits<int>::max())) {
		return false;
	}
	const int category = static_cast<int>(roundedValue);
	std::map<int, unsigned>::const_iterator it = categoryToClassIndex.find(category);
	if (it == categoryToClassIndex.end()) {
		return false;
	}
	outClassIndex = it->second;
	return true;
}

bool extractTemplateSequence(const g2s::DataImage& trainingImage,
	unsigned cellIndex,
	const std::vector<std::vector<int> >& pathPositionArray,
	const std::map<int, unsigned>& categoryToClassIndex,
	std::vector<float>& sequenceValues,
	bool& hasInvalidTemplateValue) {
	hasInvalidTemplateValue = false;
	sequenceValues.assign(pathPositionArray.size(), std::numeric_limits<float>::quiet_NaN());

	std::vector<int> centerCoordinates;
	flattenIndexToCoordinates(cellIndex, trainingImage._dims, centerCoordinates);

	for (size_t pathIndex = 0; pathIndex < pathPositionArray.size(); ++pathIndex) {
		const std::vector<int>& offset = pathPositionArray[pathIndex];
		if (offset.size() != trainingImage._dims.size()) {
			hasInvalidTemplateValue = true;
			continue;
		}

		bool isOutside = false;
		unsigned neighborCellIndex = 0u;
		unsigned linearWeight = 1u;
		for (size_t dim = 0; dim < trainingImage._dims.size(); ++dim) {
			const int shiftedCoordinate = centerCoordinates[dim] + offset[dim];
			if (shiftedCoordinate < 0 || shiftedCoordinate >= static_cast<int>(trainingImage._dims[dim])) {
				isOutside = true;
				break;
			}
			neighborCellIndex += static_cast<unsigned>(shiftedCoordinate) * linearWeight;
			linearWeight *= trainingImage._dims[dim];
		}

		if (isOutside) {
			hasInvalidTemplateValue = true;
			continue;
		}

		const float rawNeighborValue = trainingImage._data[neighborCellIndex * trainingImage._nbVariable];
		unsigned neighborClassIndex = 0u;
		if (!tryCategoryToClassIndex(rawNeighborValue, categoryToClassIndex, neighborClassIndex)) {
			hasInvalidTemplateValue = true;
			continue;
		}
		sequenceValues[pathIndex] = static_cast<float>(neighborClassIndex);
	}

	return true;
}

bool areNodesCompatible(const std::vector<snesim::TreeNode>& nodes, size_t numberOfClasses) {
	for (size_t i = 0; i < nodes.size(); ++i) {
		if (nodes[i].categoryCounts.size() != numberOfClasses || nodes[i].childNodeIndex.size() != numberOfClasses) {
			return false;
		}
	}
	return true;
}

int cloneSubtreeFrom(const snesim::SearchTree& sourceTree,
	int sourceRootIndex,
	std::vector<snesim::TreeNode>& destinationNodes,
	size_t numberOfClasses) {
	if (sourceRootIndex < 0 || static_cast<size_t>(sourceRootIndex) >= sourceTree.nodes().size()) {
		return -1;
	}

	std::map<int, int> indexMap;
	std::vector<int> stack;
	stack.push_back(sourceRootIndex);

	const int destinationRoot = static_cast<int>(destinationNodes.size());
	destinationNodes.push_back(makeNode(numberOfClasses));
	indexMap[sourceRootIndex] = destinationRoot;

	while (!stack.empty()) {
		const int currentSource = stack.back();
		stack.pop_back();

		std::map<int, int>::const_iterator destinationIndexIt = indexMap.find(currentSource);
		if (destinationIndexIt == indexMap.end()) {
			continue;
		}
		const int currentDestination = destinationIndexIt->second;

		const snesim::TreeNode& sourceNode = sourceTree.nodes()[static_cast<size_t>(currentSource)];
		destinationNodes[static_cast<size_t>(currentDestination)].categoryCounts = sourceNode.categoryCounts;
		destinationNodes[static_cast<size_t>(currentDestination)].childNodeIndex.assign(numberOfClasses, -1);

		for (size_t classIndex = 0; classIndex < numberOfClasses; ++classIndex) {
			const int sourceChild = sourceNode.childNodeIndex[classIndex];
			if (sourceChild < 0 || static_cast<size_t>(sourceChild) >= sourceTree.nodes().size()) {
				continue;
			}

			int destinationChild = -1;
			std::map<int, int>::const_iterator childIt = indexMap.find(sourceChild);
			if (childIt == indexMap.end()) {
				destinationChild = static_cast<int>(destinationNodes.size());
				destinationNodes.push_back(makeNode(numberOfClasses));
				indexMap[sourceChild] = destinationChild;
				stack.push_back(sourceChild);
			} else {
				destinationChild = childIt->second;
			}

			destinationNodes[static_cast<size_t>(currentDestination)].childNodeIndex[classIndex] = destinationChild;
		}
	}

	return destinationRoot;
}

} // namespace

namespace snesim {

SearchTree::SearchTree() {
}

SearchTree::SearchTree(const std::vector<int>& categories, const std::vector<TreeNode>& nodes) :
	_categories(categories),
	_nodes(nodes) {
}

const std::vector<int>& SearchTree::categories() const {
	return _categories;
}

const std::vector<TreeNode>& SearchTree::nodes() const {
	return _nodes;
}

SearchTree SearchTree::deepCopy() const {
	return SearchTree(_categories, _nodes);
}

void SearchTree::addStatisticsFrom(const SearchTree& other, MergePolicy policy) {
	if (policy != MergePolicy::SumCounts) {
		return;
	}
	if (other._nodes.empty()) {
		return;
	}
	if (_nodes.empty()) {
		*this = other.deepCopy();
		return;
	}
	if (_categories != other._categories) {
		return;
	}

	const size_t numberOfClasses = _categories.size();
	if (numberOfClasses == 0u) {
		return;
	}
	if (!areNodesCompatible(_nodes, numberOfClasses) || !areNodesCompatible(other._nodes, numberOfClasses)) {
		return;
	}

	std::vector<std::pair<int, int> > mergeStack;
	mergeStack.push_back(std::make_pair(0, 0));

	while (!mergeStack.empty()) {
		const std::pair<int, int> current = mergeStack.back();
		mergeStack.pop_back();

		if (current.first < 0 || static_cast<size_t>(current.first) >= _nodes.size()) {
			continue;
		}
		if (current.second < 0 || static_cast<size_t>(current.second) >= other._nodes.size()) {
			continue;
		}

		const TreeNode& sourceNode = other._nodes[static_cast<size_t>(current.second)];
		for (size_t classIndex = 0; classIndex < numberOfClasses; ++classIndex) {
			const unsigned sourceCount = sourceNode.categoryCounts[classIndex];
			unsigned& destinationCount = _nodes[static_cast<size_t>(current.first)].categoryCounts[classIndex];
			const unsigned long long mergedCount = static_cast<unsigned long long>(destinationCount)
				+ static_cast<unsigned long long>(sourceCount);
			destinationCount = static_cast<unsigned>(std::min<unsigned long long>(
				mergedCount,
				static_cast<unsigned long long>(std::numeric_limits<unsigned>::max())));
		}

		for (size_t classIndex = 0; classIndex < numberOfClasses; ++classIndex) {
			const int sourceChild = sourceNode.childNodeIndex[classIndex];
			if (sourceChild < 0 || static_cast<size_t>(sourceChild) >= other._nodes.size()) {
				continue;
			}

			const int destinationChild = _nodes[static_cast<size_t>(current.first)].childNodeIndex[classIndex];
			if (destinationChild < 0) {
				const int clonedRoot = cloneSubtreeFrom(other, sourceChild, _nodes, numberOfClasses);
				if (clonedRoot >= 0) {
					_nodes[static_cast<size_t>(current.first)].childNodeIndex[classIndex] = clonedRoot;
				}
			} else {
				mergeStack.push_back(std::make_pair(destinationChild, sourceChild));
			}
		}
	}
}

bool ensureDirectory(const std::string& directory, std::string& errorMessage) {
	if (directory.empty()) {
		errorMessage = "directory path is empty";
		return false;
	}

	std::string current;
	if (directory[0] == '/') {
		current = "/";
	}

	std::stringstream ss(directory);
	std::string token;
	while (std::getline(ss, token, '/')) {
		if (token.empty()) {
			continue;
		}

		if (!current.empty() && current[current.size() - 1] != '/') {
			current += "/";
		}
		current += token;

		if (mkdir(current.c_str(), 0775) != 0 && errno != EEXIST) {
			errorMessage = "cannot create directory '" + current + "': " + std::strerror(errno);
			return false;
		}
	}

	return true;
}

std::string sanitizePathToken(const std::string& rawName) {
	std::string sanitizedName;
	sanitizedName.reserve(rawName.size());
	for (size_t i = 0; i < rawName.size(); ++i) {
		const unsigned char ch = static_cast<unsigned char>(rawName[i]);
		if (std::isalnum(ch) || ch == '_' || ch == '-' || ch == '.') {
			sanitizedName.push_back(static_cast<char>(ch));
		} else {
			sanitizedName.push_back('_');
		}
	}

	if (sanitizedName.empty()) {
		sanitizedName = "unnamed_ti";
	}
	return sanitizedName;
}

bool summarizeCategoricalTrainingImage(const g2s::DataImage& trainingImage,
	const std::string& sourceName,
	TrainingImageSummary& outSummary,
	std::string& errorMessage) {
	// SNESIM currently supports categorical simulation only.
	// We validate that TI metadata and TI values are both categorical-compatible.
	if (trainingImage._types.empty() || trainingImage._data == nullptr || trainingImage._nbVariable == 0) {
		errorMessage = "training image '" + sourceName + "' is empty or invalid";
		return false;
	}

	if (trainingImage._types.size() != trainingImage._nbVariable) {
		errorMessage = "training image '" + sourceName + "' has inconsistent variable metadata";
		return false;
	}

	for (unsigned variableIndex = 0; variableIndex < trainingImage._nbVariable; ++variableIndex) {
		if (trainingImage._types[variableIndex] != g2s::DataImage::Categorical) {
			errorMessage = "training image '" + sourceName + "' is not categorical (variable " + std::to_string(variableIndex) + " is continuous)";
			return false;
		}
	}

	std::map<int, unsigned> categoryFrequency;
	unsigned fullSize = trainingImage._nbVariable;
	for (size_t i = 0; i < trainingImage._dims.size(); ++i) {
		fullSize *= trainingImage._dims[i];
	}
	for (unsigned i = 0; i < fullSize; ++i) {
		const float value = trainingImage._data[i];
		if (!std::isfinite(value)) {
			errorMessage = "training image '" + sourceName + "' contains invalid numeric values (NaN/Inf)";
			return false;
		}
		const double roundedValue = std::round(static_cast<double>(value));
		if (std::fabs(static_cast<double>(value) - roundedValue) > kCategoryTolerance) {
			errorMessage = "training image '" + sourceName + "' contains non-integer categories";
			return false;
		}
		if (roundedValue < static_cast<double>(std::numeric_limits<int>::min())
			|| roundedValue > static_cast<double>(std::numeric_limits<int>::max())) {
			errorMessage = "training image '" + sourceName + "' contains category values outside integer range";
			return false;
		}
		const int category = static_cast<int>(roundedValue);
		++categoryFrequency[category];
	}

	if (categoryFrequency.empty()) {
		errorMessage = "training image '" + sourceName + "' does not contain any category";
		return false;
	}

	outSummary = TrainingImageSummary();
	outSummary.sourceName = sourceName;
	outSummary.cacheName = sanitizePathToken(sourceName);
	outSummary.dims = trainingImage._dims;
	outSummary.nbVariable = trainingImage._nbVariable;
	for (std::map<int, unsigned>::const_iterator it = categoryFrequency.begin(); it != categoryFrequency.end(); ++it) {
		outSummary.categories.push_back(it->first);
		outSummary.categoryFrequency.push_back(it->second);
	}

	return true;
}

SearchTree buildBasicTree(const g2s::DataImage& /*trainingImage*/,
	const TrainingImageSummary& summary,
	const TreeBuildConfig& /*config*/) {
	const size_t numberOfClasses = summary.categories.size();
	TreeNode rootNode = makeNode(numberOfClasses);
	for (size_t classIndex = 0; classIndex < numberOfClasses && classIndex < summary.categoryFrequency.size(); ++classIndex) {
		rootNode.categoryCounts[classIndex] = summary.categoryFrequency[classIndex];
	}

	std::vector<TreeNode> nodes;
	nodes.push_back(rootNode);
	return SearchTree(summary.categories, nodes);
}

SearchTree buildTreeForLevel(const g2s::DataImage& trainingImage,
	const TrainingImageSummary& summary,
	const std::vector<std::vector<int> >& pathPositionArray) {
	const size_t numberOfClasses = summary.categories.size();
	if (numberOfClasses == 0u || trainingImage._data == nullptr || trainingImage._dims.empty()) {
		return SearchTree(summary.categories, std::vector<TreeNode>());
	}

	std::map<int, unsigned> categoryToClassIndex;
	for (size_t i = 0; i < summary.categories.size(); ++i) {
		categoryToClassIndex[summary.categories[i]] = static_cast<unsigned>(i);
	}

	std::vector<TreeNode> nodes;
	nodes.reserve(std::max(2u, computeCellCount(trainingImage._dims) / 2u));
	nodes.push_back(makeNode(numberOfClasses));

	const unsigned cellCount = computeCellCount(trainingImage._dims);
	std::vector<float> templateSequence;
	for (unsigned cellIndex = 0; cellIndex < cellCount; ++cellIndex) {
		// Central value drives the histogram update target.
		// Missing center values are skipped immediately.
		const float rawCenterValue = trainingImage._data[cellIndex * trainingImage._nbVariable];
		unsigned centerClassIndex = 0u;
		if (!tryCategoryToClassIndex(rawCenterValue, categoryToClassIndex, centerClassIndex)) {
			continue;
		}

		bool hasInvalidTemplateValue = false;
		// Extract full sequence first. Invalid/outside values are written as NaN and flagged.
		extractTemplateSequence(trainingImage,
			cellIndex,
			pathPositionArray,
			categoryToClassIndex,
			templateSequence,
			hasInvalidTemplateValue);
		if (hasInvalidTemplateValue) {
			continue;
		}

		// Counts are updated at every visited node for the central class.
		// Child node allocation is contiguous: each new branch gets the next free index.
		int currentNode = 0;
		incrementCount(nodes[0].categoryCounts[centerClassIndex]);
		for (size_t pathIndex = 0; pathIndex < templateSequence.size(); ++pathIndex) {
			const float branchClassRaw = templateSequence[pathIndex];
			if (!std::isfinite(branchClassRaw) || branchClassRaw < 0.f) {
				currentNode = -1;
				break;
			}
			const unsigned branchClass = static_cast<unsigned>(branchClassRaw);
			if (branchClass >= numberOfClasses) {
				currentNode = -1;
				break;
			}

			int nextNode = nodes[static_cast<size_t>(currentNode)].childNodeIndex[branchClass];
			if (nextNode < 0) {
				nextNode = static_cast<int>(nodes.size());
				nodes[static_cast<size_t>(currentNode)].childNodeIndex[branchClass] = nextNode;
				nodes.push_back(makeNode(numberOfClasses));
			}
			currentNode = nextNode;
			incrementCount(nodes[static_cast<size_t>(currentNode)].categoryCounts[centerClassIndex]);
		}
	}

	return SearchTree(summary.categories, nodes);
}

SearchTree mergeTrees(const std::vector<const SearchTree*>& trees, MergePolicy policy) {
	SearchTree merged;
	bool hasSeed = false;
	for (size_t i = 0; i < trees.size(); ++i) {
		if (trees[i] == nullptr) {
			continue;
		}
		if (!hasSeed) {
			merged = trees[i]->deepCopy();
			hasSeed = true;
			continue;
		}
		merged.addStatisticsFrom(*trees[i], policy);
	}
	return merged;
}

TreeCacheRepository::TreeCacheRepository(const std::string& rootPath) :
	_rootPath(rootPath.empty() ? std::string(kDefaultTreeRoot) : rootPath) {
}

const std::string& TreeCacheRepository::rootPath() const {
	return _rootPath;
}

std::string TreeCacheRepository::getTrainingImageCacheFolder(const std::string& trainingImageSourceName) const {
	return _rootPath + "/" + sanitizePathToken(trainingImageSourceName);
}

std::string TreeCacheRepository::getTreeMetadataPath(const std::string& trainingImageSourceName, unsigned gridLevel) const {
	return getTrainingImageCacheFolder(trainingImageSourceName)
		+ "/" + kTreeMetadataPrefix + std::to_string(gridLevel) + kTreeMetadataSuffix;
}

bool TreeCacheRepository::ensureCacheFolder(const std::string& trainingImageSourceName,
	std::string& cacheFolder,
	std::string& errorMessage) const {
	if (!ensureDirectory(_rootPath, errorMessage)) {
		return false;
	}

	cacheFolder = getTrainingImageCacheFolder(trainingImageSourceName);
	return ensureDirectory(cacheFolder, errorMessage);
}

bool TreeCacheRepository::save(const std::string& trainingImageSourceName,
	const CachedTreeRecord& record,
	unsigned gridLevel,
	std::string& errorMessage) const {
	std::string cacheFolder;
	if (!ensureCacheFolder(trainingImageSourceName, cacheFolder, errorMessage)) {
		return false;
	}

	const std::string metadataPath = getTreeMetadataPath(trainingImageSourceName, gridLevel);
	std::ofstream outFile(metadataPath.c_str(), std::ios::out | std::ios::trunc);
	if (!outFile.good()) {
		errorMessage = "cannot write tree metadata: " + metadataPath;
		return false;
	}

	outFile << "version=2\n";
	outFile << "source_name=" << record.summary.sourceName << "\n";
	outFile << "cache_name=" << record.summary.cacheName << "\n";
	outFile << "nb_dim=" << record.summary.dims.size() << "\n";
	outFile << "dims=" << joinUnsignedVector(record.summary.dims) << "\n";
	outFile << "nb_variable=" << record.summary.nbVariable << "\n";
	outFile << "categories=" << joinIntVector(record.summary.categories) << "\n";
	outFile << "category_frequency=" << joinUnsignedVector(record.summary.categoryFrequency) << "\n";
	outFile << "template_radius=" << joinIntVector(record.config.templateRadius) << "\n";
	// Backward-compatible key name retained for older readers.
	outFile << "template_size=" << joinIntVector(record.config.templateRadius) << "\n";
	outFile << "grid_level=" << record.config.gridLevel << "\n";
	outFile << "max_conditioning_data=" << record.config.maxConditioningData << "\n";

	const std::vector<TreeNode>& nodes = record.tree.nodes();
	outFile << "node_count=" << nodes.size() << "\n";
	for (size_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
		outFile << "node_" << nodeIndex << "_counts=" << joinUnsignedVector(nodes[nodeIndex].categoryCounts) << "\n";
		outFile << "node_" << nodeIndex << "_children=" << joinIntVector(nodes[nodeIndex].childNodeIndex) << "\n";
	}

	if (!outFile.good()) {
		errorMessage = "error while writing tree metadata: " + metadataPath;
		return false;
	}

	return true;
}

bool TreeCacheRepository::load(const std::string& trainingImageSourceName,
	CachedTreeRecord& outRecord,
	unsigned gridLevel,
	std::string& errorMessage) const {
	std::string metadataPath = getTreeMetadataPath(trainingImageSourceName, gridLevel);
	if (!fileExists(metadataPath)) {
		if (gridLevel == 0u) {
			const std::string legacyMetadataPath = getTrainingImageCacheFolder(trainingImageSourceName)
				+ "/" + kLegacyTreeMetadataFile;
			if (fileExists(legacyMetadataPath)) {
				metadataPath = legacyMetadataPath;
			} else {
				errorMessage = "tree cache not found: " + metadataPath;
				return false;
			}
		} else {
			errorMessage = "tree cache not found: " + metadataPath;
			return false;
		}
	}

	std::ifstream inFile(metadataPath.c_str());
	if (!inFile.good()) {
		errorMessage = "cannot open tree metadata: " + metadataPath;
		return false;
	}

	std::map<std::string, std::string> fields;
	std::string line;
	while (std::getline(inFile, line)) {
		std::string key;
		std::string value;
		if (!parseKeyValueLine(line, key, value)) {
			continue;
		}
		fields[key] = value;
	}

	outRecord = CachedTreeRecord();
	outRecord.summary.sourceName = fields["source_name"];
	outRecord.summary.cacheName = fields["cache_name"];

	if (!parseUnsignedVector(fields["dims"], outRecord.summary.dims)) {
		errorMessage = "invalid dims in tree cache: " + metadataPath;
		return false;
	}

	if (!parseUnsignedValue(fields["nb_variable"], outRecord.summary.nbVariable)) {
		errorMessage = "invalid nb_variable in tree cache: " + metadataPath;
		return false;
	}

	if (!parseIntVector(fields["categories"], outRecord.summary.categories)) {
		errorMessage = "invalid categories in tree cache: " + metadataPath;
		return false;
	}

	if (!parseUnsignedVector(fields["category_frequency"], outRecord.summary.categoryFrequency)) {
		errorMessage = "invalid category_frequency in tree cache: " + metadataPath;
		return false;
	}

	if (!fields["template_radius"].empty()) {
		if (!parseIntVector(fields["template_radius"], outRecord.config.templateRadius)) {
			errorMessage = "invalid template_radius in tree cache: " + metadataPath;
			return false;
		}
	} else {
		// Backward-compatible load path.
		if (!parseIntVector(fields["template_size"], outRecord.config.templateRadius)) {
			errorMessage = "invalid template_size in tree cache: " + metadataPath;
			return false;
		}
	}

	if (!parseUnsignedValue(fields["max_conditioning_data"], outRecord.config.maxConditioningData)) {
		outRecord.config.maxConditioningData = 0u;
	}
	if (!parseUnsignedValue(fields["grid_level"], outRecord.config.gridLevel)) {
		outRecord.config.gridLevel = gridLevel;
	}

	unsigned parsedNodeCount = 0u;
	if (!parseUnsignedValue(fields["node_count"], parsedNodeCount)) {
		parsedNodeCount = 0u;
	}

	const size_t numberOfClasses = outRecord.summary.categories.size();
	if (numberOfClasses == 0u) {
		errorMessage = "invalid categories in tree cache: " + metadataPath;
		return false;
	}

	std::vector<TreeNode> nodes;
	if (parsedNodeCount > 0u) {
		nodes.assign(parsedNodeCount, makeNode(numberOfClasses));
		for (unsigned nodeIndex = 0u; nodeIndex < parsedNodeCount; ++nodeIndex) {
			const std::string countKey = "node_" + std::to_string(nodeIndex) + "_counts";
			const std::string childrenKey = "node_" + std::to_string(nodeIndex) + "_children";
			if (fields[countKey].empty() || fields[childrenKey].empty()) {
				errorMessage = "missing node fields in tree cache: " + metadataPath;
				return false;
			}

			std::vector<unsigned> nodeCounts;
			std::vector<int> nodeChildren;
			if (!parseUnsignedVector(fields[countKey], nodeCounts) || nodeCounts.size() != numberOfClasses) {
				errorMessage = "invalid " + countKey + " in tree cache: " + metadataPath;
				return false;
			}
			if (!parseIntVector(fields[childrenKey], nodeChildren) || nodeChildren.size() != numberOfClasses) {
				errorMessage = "invalid " + childrenKey + " in tree cache: " + metadataPath;
				return false;
			}

			nodes[nodeIndex].categoryCounts = nodeCounts;
			nodes[nodeIndex].childNodeIndex = nodeChildren;
		}
		for (size_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
			for (size_t classIndex = 0; classIndex < numberOfClasses; ++classIndex) {
				const int child = nodes[nodeIndex].childNodeIndex[classIndex];
				if (child < -1 || child >= static_cast<int>(nodes.size())) {
					errorMessage = "invalid child index in tree cache: " + metadataPath;
					return false;
				}
			}
		}
	} else {
		// Legacy root-only metadata fallback.
		std::vector<unsigned> rootCounts;
		if (!fields["root_counts"].empty()) {
			if (!parseUnsignedVector(fields["root_counts"], rootCounts) || rootCounts.size() != numberOfClasses) {
				errorMessage = "invalid root_counts in tree cache: " + metadataPath;
				return false;
			}
		} else {
			rootCounts = outRecord.summary.categoryFrequency;
		}

		TreeNode rootNode = makeNode(numberOfClasses);
		for (size_t classIndex = 0; classIndex < numberOfClasses && classIndex < rootCounts.size(); ++classIndex) {
			rootNode.categoryCounts[classIndex] = rootCounts[classIndex];
		}
		nodes.push_back(rootNode);
	}

	outRecord.tree = SearchTree(outRecord.summary.categories, nodes);
	return true;
}

} // namespace snesim
