/*
 * G2S
 * Copyright (C) 2018, Mathieu Gravey (gravey.mathieu@gmail.com) and UNIL (University of Lausanne)
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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "computeDeviceModule.hpp"
#include "DataImage.hpp"
#include "samplingModule.hpp"
#include "simulation.hpp"
#include "snesimCPUThreadDevice.hpp"
#include "snesimTree.hpp"
#include "utils.hpp"

namespace {

struct SimulationRunConfig {
	unsigned nbThreads = 1;
	unsigned seed = 0;
	unsigned maxGridLevel = 0;
	std::vector<int> templateRadius;
};

enum class TreeStrategy {
	First = 0,
	Ii = 1,
	Merged = 2
};

struct CliOptions {
	FILE* reportFile = stdout;
	bool closeReportFile = false;
	unsigned uniqueID = std::numeric_limits<unsigned>::max();

	bool verbose = false;
	bool forceTreeRebuild = false;

	std::vector<std::string> trainingImageNames;
	std::string destinationImageName;
	std::string trainingImageIndexName;
	std::string outputName;
	std::string treeRoot = "/tmp/G2S/data/snesim_trees";
	TreeStrategy treeStrategy = TreeStrategy::Merged;
	bool treeStrategyExplicit = false;

	snesim::TreeBuildConfig treeBuildConfig;
	SimulationRunConfig simulationConfig;
};

struct GridLevelPlan {
	unsigned level = 0;
	std::vector<g2s_path_index_t> simulationPath;
	std::vector<std::vector<int> > pathPositionArray;
};

void printHelp() {
	printf("SNESIM scaffold options:\n");
	printf("  -ti <hash_or_name> [repeat]   Training image ids (required)\n");
	printf("  -di <hash_or_name>            Destination image id (required)\n");
	printf("  -ii <hash_or_name>            TI index image (optional, selects tree/TI per node)\n");
	printf("  -o  <output_name>             Output id (optional)\n");
	printf("  --tree-strategy <mode>        first|ii|merged (default=merged)\n");
	printf("  -j  <threads>                 Worker thread count (optional)\n");
	printf("  -mg <level>                   Max multi-grid level (levels run from <level> down to 0)\n");
	printf("  -tpl <radius> [repeat]        Template radius (3 means offsets in [-3,+3])\n");
	printf("  --template-radius <radius>    Same as -tpl\n");
	printf("  -tree-root <path>             Tree cache root (optional)\n");
	printf("  -force-tree                   Force tree rebuild and overwrite cache\n");
	printf("  -s <seed>                     Random seed (optional)\n");
	printf("  -r <stdout|stderr|path>       Report file (optional)\n");
}

unsigned parseThreadCount(const std::string& rawValue, unsigned totalThreadsAvailable) {
	const float parsedValue = std::atof(rawValue.c_str());
	if (std::isnan(parsedValue) || parsedValue <= 0.f) {
		return 1;
	}
	if (std::round(parsedValue) == parsedValue) {
		return std::max(1u, static_cast<unsigned>(parsedValue));
	}
	const float scaledValue = std::max(1.f, std::floor(parsedValue * static_cast<float>(totalThreadsAvailable)));
	return std::max(1u, static_cast<unsigned>(scaledValue));
}

bool parseUnsignedFromString(const std::string& rawValue, unsigned& outValue) {
	char* endPtr = nullptr;
	const unsigned long parsedValue = std::strtoul(rawValue.c_str(), &endPtr, 10);
	if (endPtr == nullptr || *endPtr != '\0' || parsedValue > std::numeric_limits<unsigned>::max()) {
		return false;
	}
	outValue = static_cast<unsigned>(parsedValue);
	return true;
}

bool parseIntFromString(const std::string& rawValue, int& outValue) {
	char* endPtr = nullptr;
	const long parsedValue = std::strtol(rawValue.c_str(), &endPtr, 10);
	if (endPtr == nullptr || *endPtr != '\0' || parsedValue < std::numeric_limits<int>::min() || parsedValue > std::numeric_limits<int>::max()) {
		return false;
	}
	outValue = static_cast<int>(parsedValue);
	return true;
}

const char* treeStrategyName(TreeStrategy strategy) {
	switch (strategy) {
	case TreeStrategy::First:
		return "first";
	case TreeStrategy::Ii:
		return "ii";
	case TreeStrategy::Merged:
	default:
		return "merged";
	}
}

bool parseTreeStrategy(const std::string& rawValue, TreeStrategy& outStrategy) {
	if (rawValue == "first") {
		outStrategy = TreeStrategy::First;
		return true;
	}
	if (rawValue == "ii") {
		outStrategy = TreeStrategy::Ii;
		return true;
	}
	if (rawValue == "merged") {
		outStrategy = TreeStrategy::Merged;
		return true;
	}
	return false;
}

bool setupReportFile(std::multimap<std::string, std::string>& args,
	CliOptions& options,
	std::string& errorMessage) {
	if (args.count("-r") > 1) {
		errorMessage = "only one report file is supported";
		return false;
	}

	if (args.count("-r") == 1) {
		const std::string reportName = args.find("-r")->second;
		if (reportName == "stdout") {
			options.reportFile = stdout;
		} else if (reportName == "stderr") {
			options.reportFile = stderr;
		} else {
			options.reportFile = fopen(reportName.c_str(), "a");
			if (!options.reportFile) {
				errorMessage = "cannot open report file: " + reportName;
				return false;
			}
			options.closeReportFile = true;
			setvbuf(options.reportFile, nullptr, _IOLBF, 0);

			unsigned logId;
			if (sscanf(reportName.c_str(), "/tmp/G2S/logs/%u.log", &logId) == 1) {
				options.uniqueID = logId;
			}
		}
	}

	args.erase("-r");
	return true;
}

bool parseCliOptions(int argc, const char* argv[], CliOptions& outOptions) {
	std::multimap<std::string, std::string> args = g2s::argumentReader(argc, argv);

	if (args.count("-h") == 1 || args.count("--help") == 1) {
		printHelp();
		return false;
	}
	args.erase("-h");
	args.erase("--help");

	std::string errorMessage;
	if (!setupReportFile(args, outOptions, errorMessage)) {
		fprintf(stderr, "[SNESIM] %s\n", errorMessage.c_str());
		return false;
	}

	for (int i = 0; i < argc; ++i) {
		fprintf(outOptions.reportFile, "%s ", argv[i]);
	}
	fprintf(outOptions.reportFile, "\n");

	unsigned totalThreadsAvailable = std::max(1u, std::thread::hardware_concurrency());

	if (args.count("-j") >= 1) {
		std::multimap<std::string, std::string>::iterator jobsString = args.lower_bound("-j");
		if (jobsString != args.upper_bound("-j")) {
			outOptions.simulationConfig.nbThreads = parseThreadCount(jobsString->second, totalThreadsAvailable);
		}
	}
	args.erase("-j");

	if (args.count("--jobs") >= 1) {
		std::multimap<std::string, std::string>::iterator jobsString = args.lower_bound("--jobs");
		if (jobsString != args.upper_bound("--jobs")) {
			outOptions.simulationConfig.nbThreads = parseThreadCount(jobsString->second, totalThreadsAvailable);
		}
	}
	args.erase("--jobs");

	if (args.count("-v") == 1 || args.count("--verbose") == 1) {
		outOptions.verbose = true;
	}
	args.erase("-v");
	args.erase("--verbose");

	if (args.count("-ti") > 0) {
		for (std::multimap<std::string, std::string>::iterator it = args.lower_bound("-ti"); it != args.upper_bound("-ti"); ++it) {
			outOptions.trainingImageNames.push_back(it->second);
		}
	}
	args.erase("-ti");

	if (args.count("-di") == 1) {
		outOptions.destinationImageName = args.find("-di")->second;
	}
	args.erase("-di");

	if (args.count("-ii") == 1) {
		outOptions.trainingImageIndexName = args.find("-ii")->second;
	} else if (args.count("-ii") > 1) {
		fprintf(outOptions.reportFile, "[SNESIM] only one -ii image is supported\n");
		return false;
	}
	args.erase("-ii");

	if (args.count("-o") == 1) {
		outOptions.outputName = args.find("-o")->second;
	}
	args.erase("-o");

	if (args.count("--tree-strategy") == 1) {
		const std::string strategyValue = args.find("--tree-strategy")->second;
		TreeStrategy parsedStrategy;
		if (!parseTreeStrategy(strategyValue, parsedStrategy)) {
			fprintf(outOptions.reportFile, "[SNESIM] invalid --tree-strategy value: %s\n", strategyValue.c_str());
			return false;
		}
		outOptions.treeStrategy = parsedStrategy;
		outOptions.treeStrategyExplicit = true;
	} else if (args.count("--tree-strategy") > 1) {
		fprintf(outOptions.reportFile, "[SNESIM] only one --tree-strategy value is supported\n");
		return false;
	}
	args.erase("--tree-strategy");

	if (args.count("-mg") >= 1) {
		unsigned parsedMaxLevel = outOptions.simulationConfig.maxGridLevel;
		bool hasValidValue = false;
		for (std::multimap<std::string, std::string>::iterator it = args.lower_bound("-mg"); it != args.upper_bound("-mg"); ++it) {
			unsigned level;
			if (!parseUnsignedFromString(it->second, level)) {
				fprintf(outOptions.reportFile, "[SNESIM] invalid -mg value ignored: %s\n", it->second.c_str());
				continue;
			}
			parsedMaxLevel = std::max(parsedMaxLevel, level);
			hasValidValue = true;
		}
		if (hasValidValue) {
			outOptions.simulationConfig.maxGridLevel = parsedMaxLevel;
		}
	}
	args.erase("-mg");

	if (args.count("--mg-level") >= 1) {
		unsigned parsedMaxLevel = outOptions.simulationConfig.maxGridLevel;
		bool hasValidValue = false;
		for (std::multimap<std::string, std::string>::iterator it = args.lower_bound("--mg-level"); it != args.upper_bound("--mg-level"); ++it) {
			unsigned level;
			if (!parseUnsignedFromString(it->second, level)) {
				fprintf(outOptions.reportFile, "[SNESIM] invalid --mg-level value ignored: %s\n", it->second.c_str());
				continue;
			}
			parsedMaxLevel = std::max(parsedMaxLevel, level);
			hasValidValue = true;
		}
		if (hasValidValue) {
			outOptions.simulationConfig.maxGridLevel = parsedMaxLevel;
		}
	}
	args.erase("--mg-level");

	if (args.count("-tpl") >= 1) {
		for (std::multimap<std::string, std::string>::iterator it = args.lower_bound("-tpl"); it != args.upper_bound("-tpl"); ++it) {
			int radiusValue;
			if (!parseIntFromString(it->second, radiusValue)) {
				fprintf(outOptions.reportFile, "[SNESIM] invalid -tpl value ignored: %s\n", it->second.c_str());
				continue;
			}
			outOptions.treeBuildConfig.templateRadius.push_back(radiusValue);
		}
	}
	args.erase("-tpl");

	if (args.count("--template-radius") >= 1) {
		for (std::multimap<std::string, std::string>::iterator it = args.lower_bound("--template-radius"); it != args.upper_bound("--template-radius"); ++it) {
			int radiusValue;
			if (!parseIntFromString(it->second, radiusValue)) {
				fprintf(outOptions.reportFile, "[SNESIM] invalid --template-radius value ignored: %s\n", it->second.c_str());
				continue;
			}
			outOptions.treeBuildConfig.templateRadius.push_back(radiusValue);
		}
	}
	args.erase("--template-radius");

	// Backward-compatible alias. Still interpreted as radius.
	if (args.count("--template-size") >= 1) {
		for (std::multimap<std::string, std::string>::iterator it = args.lower_bound("--template-size"); it != args.upper_bound("--template-size"); ++it) {
			int radiusValue;
			if (!parseIntFromString(it->second, radiusValue)) {
				fprintf(outOptions.reportFile, "[SNESIM] invalid --template-size value ignored: %s\n", it->second.c_str());
				continue;
			}
			outOptions.treeBuildConfig.templateRadius.push_back(radiusValue);
		}
		fprintf(outOptions.reportFile, "[SNESIM] --template-size is deprecated; values are interpreted as radius\n");
	}
	args.erase("--template-size");

	if (args.count("-maxn") > 0) {
		fprintf(outOptions.reportFile, "[SNESIM] -maxn is not used in this SNESIM scaffold and is ignored\n");
	}
	args.erase("-maxn");

	if (args.count("-tree-root") == 1) {
		outOptions.treeRoot = args.find("-tree-root")->second;
	}
	args.erase("-tree-root");

	if (args.count("--tree-root") == 1) {
		outOptions.treeRoot = args.find("--tree-root")->second;
	}
	args.erase("--tree-root");

	if (args.count("-force-tree") == 1 || args.count("--force-tree") == 1 || args.count("-ft") == 1) {
		outOptions.forceTreeRebuild = true;
	}
	args.erase("-force-tree");
	args.erase("--force-tree");
	args.erase("-ft");

	if (args.count("-s") == 1) {
		unsigned seedValue;
		if (parseUnsignedFromString(args.find("-s")->second, seedValue)) {
			outOptions.simulationConfig.seed = seedValue;
		}
	}
	args.erase("-s");

	if (outOptions.simulationConfig.seed == 0) {
		outOptions.simulationConfig.seed = static_cast<unsigned>(
			std::chrono::high_resolution_clock::now().time_since_epoch().count());
	}

	if (outOptions.simulationConfig.nbThreads == 0) {
		outOptions.simulationConfig.nbThreads = 1;
	}
	outOptions.simulationConfig.templateRadius = outOptions.treeBuildConfig.templateRadius;

	if (!outOptions.trainingImageIndexName.empty()) {
		if (outOptions.treeStrategy != TreeStrategy::Ii) {
			fprintf(outOptions.reportFile,
				"[SNESIM] -ii provided: forcing tree strategy to 'ii' (was '%s')\n",
				treeStrategyName(outOptions.treeStrategy));
		}
		outOptions.treeStrategy = TreeStrategy::Ii;
	}

	for (std::multimap<std::string, std::string>::iterator it = args.begin(); it != args.end(); ++it) {
		fprintf(outOptions.reportFile, "%s %s <== ignored !\n", it->first.c_str(), it->second.c_str());
	}

	if (outOptions.trainingImageNames.empty()) {
		fprintf(outOptions.reportFile, "[SNESIM] missing mandatory parameter: -ti\n");
		return false;
	}
	if (outOptions.destinationImageName.empty()) {
		fprintf(outOptions.reportFile, "[SNESIM] missing mandatory parameter: -di\n");
		return false;
	}
	if (outOptions.treeStrategy == TreeStrategy::Ii && outOptions.trainingImageIndexName.empty()) {
		fprintf(outOptions.reportFile, "[SNESIM] --tree-strategy ii requires -ii\n");
		return false;
	}

	return true;
}

unsigned computeCellCount(const g2s::DataImage& image) {
	unsigned cellCount = 1;
	for (size_t dim = 0; dim < image._dims.size(); ++dim) {
		cellCount *= image._dims[dim];
	}
	return cellCount;
}

bool cellHasNaN(const g2s::DataImage& image, unsigned cellIndex) {
	for (unsigned variableIndex = 0; variableIndex < image._nbVariable; ++variableIndex) {
		if (std::isnan(image._data[cellIndex * image._nbVariable + variableIndex])) {
			return true;
		}
	}
	return false;
}

bool validateDestinationImage(const g2s::DataImage& destinationImage,
	unsigned expectedVariableCount,
	const std::set<int>& knownCategories,
	std::string& errorMessage) {
	if (destinationImage._types.empty() || destinationImage._data == nullptr) {
		errorMessage = "destination image cannot be loaded";
		return false;
	}
	if (destinationImage._nbVariable != expectedVariableCount) {
		errorMessage = "destination image variable count does not match the training image";
		return false;
	}
	if (destinationImage._types.size() != destinationImage._nbVariable) {
		errorMessage = "destination image has inconsistent variable metadata";
		return false;
	}
	for (unsigned variableIndex = 0; variableIndex < destinationImage._nbVariable; ++variableIndex) {
		if (destinationImage._types[variableIndex] != g2s::DataImage::Categorical) {
			errorMessage = "destination image is not categorical";
			return false;
		}
	}

	const double tolerance = 1e-5;
	const unsigned fullSize = destinationImage._nbVariable * computeCellCount(destinationImage);
	for (unsigned i = 0; i < fullSize; ++i) {
		const float value = destinationImage._data[i];
		if (std::isnan(value)) {
			continue;
		}
		if (!std::isfinite(value)) {
			errorMessage = "destination image contains NaN/Inf";
			return false;
		}
		const double roundedValue = std::round(static_cast<double>(value));
		if (std::fabs(static_cast<double>(value) - roundedValue) > tolerance) {
			errorMessage = "destination image contains non-integer values";
			return false;
		}
		const int category = static_cast<int>(roundedValue);
		if (knownCategories.find(category) == knownCategories.end()) {
			errorMessage = "destination image contains category " + std::to_string(category) + " not present in TI";
			return false;
		}
	}

	return true;
}

bool validateTrainingImageSelectionImage(g2s::DataImage& tiSelectionImage,
	const g2s::DataImage& destinationImage,
	unsigned trainingImageCount,
	std::string& errorMessage) {
	if (tiSelectionImage._types.empty() || tiSelectionImage._data == nullptr) {
		errorMessage = "TI index image cannot be loaded";
		return false;
	}
	if (tiSelectionImage._dims != destinationImage._dims) {
		errorMessage = "TI index image dimensions do not match destination image";
		return false;
	}
	if (tiSelectionImage._nbVariable != 1u) {
		errorMessage = "TI index image must have exactly one variable";
		return false;
	}
	const unsigned cellCount = computeCellCount(destinationImage);
	const double tolerance = 1e-5;
	for (unsigned cellIndex = 0; cellIndex < cellCount; ++cellIndex) {
		const float rawValue = tiSelectionImage._data[cellIndex];
		if (!std::isfinite(rawValue)) {
			errorMessage = "TI index image contains NaN/Inf";
			return false;
		}
		const double roundedValue = std::round(static_cast<double>(rawValue));
		if (std::fabs(static_cast<double>(rawValue) - roundedValue) > tolerance) {
			errorMessage = "TI index image contains non-integer values";
			return false;
		}
		if (roundedValue < 0.0 || roundedValue >= static_cast<double>(trainingImageCount)) {
			errorMessage = "TI index image contains values outside [0, " + std::to_string(trainingImageCount - 1u) + "]";
			return false;
		}
		// Normalize to exact integer value for downstream casts in simulation().
		tiSelectionImage._data[cellIndex] = static_cast<float>(roundedValue);
	}
	return true;
}

std::shared_ptr<const snesim::SearchTree> loadOrCreateTree(const std::string& trainingImageName,
	const g2s::DataImage& trainingImage,
	const snesim::TrainingImageSummary& summary,
	const snesim::TreeBuildConfig& treeConfig,
	const std::vector<std::vector<int> >& pathPositionArray,
	bool forceTreeRebuild,
	snesim::TreeCacheRepository& cacheRepository,
	FILE* reportFile) {
	if (!forceTreeRebuild) {
		snesim::CachedTreeRecord loadedRecord;
		std::string loadError;
		if (cacheRepository.load(trainingImageName, loadedRecord, treeConfig.gridLevel, loadError)) {
			const bool compatible = loadedRecord.summary.nbVariable == summary.nbVariable
				&& loadedRecord.summary.dims == summary.dims
				&& loadedRecord.summary.categories == summary.categories
				&& loadedRecord.config.templateRadius == treeConfig.templateRadius
				&& loadedRecord.config.maxConditioningData == treeConfig.maxConditioningData
				&& loadedRecord.config.gridLevel == treeConfig.gridLevel;
			if (compatible) {
				fprintf(reportFile, "[SNESIM] reuse cached tree for TI '%s' at level %u (%s)\n",
					trainingImageName.c_str(),
					treeConfig.gridLevel,
					cacheRepository.getTrainingImageCacheFolder(trainingImageName).c_str());
				return std::make_shared<snesim::SearchTree>(loadedRecord.tree);
			}
			fprintf(reportFile, "[SNESIM] cached tree is incompatible for TI '%s' level %u, rebuilding\n",
				trainingImageName.c_str(),
				treeConfig.gridLevel);
		} else {
			fprintf(reportFile, "[SNESIM] no reusable cache for TI '%s' level %u: %s\n",
				trainingImageName.c_str(),
				treeConfig.gridLevel,
				loadError.c_str());
		}
	}

	const snesim::SearchTree tree = snesim::buildTreeForLevel(trainingImage, summary, pathPositionArray);
	snesim::CachedTreeRecord record;
	record.summary = summary;
	record.config = treeConfig;
	record.tree = tree;

	std::string saveError;
	if (!cacheRepository.save(trainingImageName, record, treeConfig.gridLevel, saveError)) {
		fprintf(reportFile, "[SNESIM] warning: cannot cache tree for TI '%s' level %u: %s\n",
			trainingImageName.c_str(),
			treeConfig.gridLevel,
			saveError.c_str());
	} else {
		fprintf(reportFile, "[SNESIM] tree cache updated for TI '%s' level %u in %s\n",
			trainingImageName.c_str(),
			treeConfig.gridLevel,
			cacheRepository.getTrainingImageCacheFolder(trainingImageName).c_str());
	}

	return std::make_shared<snesim::SearchTree>(tree);
}

std::shared_ptr<const snesim::SearchTree> buildMergedTree(const std::vector<std::shared_ptr<const snesim::SearchTree> >& treesByTrainingImage) {
	std::vector<const snesim::SearchTree*> rawTrees;
	rawTrees.reserve(treesByTrainingImage.size());
	for (size_t i = 0; i < treesByTrainingImage.size(); ++i) {
		rawTrees.push_back(treesByTrainingImage[i].get());
	}
	snesim::SearchTree merged = snesim::mergeTrees(rawTrees, snesim::MergePolicy::SumCounts);
	if (merged.categories().empty()) {
		return std::shared_ptr<const snesim::SearchTree>();
	}
	return std::make_shared<snesim::SearchTree>(merged);
}

std::vector<unsigned> buildDescendingGridLevels(unsigned maxGridLevel) {
	std::vector<unsigned> levels;
	for (unsigned level = maxGridLevel + 1; level-- > 0;) {
		levels.push_back(level);
	}
	return levels;
}

bool cellBelongsToGridLevel(unsigned cellIndex, const std::vector<unsigned>& dims, unsigned gridLevel) {
	const unsigned stride = (gridLevel >= (sizeof(unsigned) * 8 - 1)) ? 0u : (1u << gridLevel);
	if (stride == 0u) {
		return false;
	}

	unsigned localIndex = cellIndex;
	for (size_t dim = 0; dim < dims.size(); ++dim) {
		const unsigned coord = localIndex % dims[dim];
		if ((coord % stride) != 0u) {
			return false;
		}
		localIndex /= dims[dim];
	}
	return true;
}

void enumerateOffsets(size_t dimIndex,
	std::vector<int>& currentOffset,
	const std::vector<int>& radiusByDim,
	unsigned stride,
	std::vector<std::vector<int> >& offsets) {
	if (dimIndex == currentOffset.size()) {
		bool isZero = true;
		for (size_t i = 0; i < currentOffset.size(); ++i) {
			isZero = isZero && (currentOffset[i] == 0);
		}
		if (!isZero) {
			offsets.push_back(currentOffset);
		}
		return;
	}

	const int bound = radiusByDim[dimIndex] * int(stride);
	for (int value = -bound; value <= bound; value += int(stride)) {
		currentOffset[dimIndex] = value;
		enumerateOffsets(dimIndex + 1, currentOffset, radiusByDim, stride, offsets);
	}
}

std::vector<std::vector<int> > buildPathPositionArrayForLevel(const std::vector<unsigned>& dims,
	const std::vector<int>& templateRadius,
	unsigned gridLevel) {
	std::vector<std::vector<int> > offsets;
	if (dims.empty()) {
		return offsets;
	}

	const unsigned stride = (gridLevel >= (sizeof(unsigned) * 8 - 1)) ? 0u : (1u << gridLevel);
	if (stride == 0u) {
		return offsets;
	}

	std::vector<int> radiusByDim(dims.size(), 1);
	for (size_t dim = 0; dim < dims.size(); ++dim) {
		int localTemplateRadius = 3;
		if (!templateRadius.empty()) {
			localTemplateRadius = templateRadius[std::min(dim, templateRadius.size() - 1)];
		}
		if (localTemplateRadius < 0) {
			localTemplateRadius *= -1;
		}
		if (localTemplateRadius < 1) {
			localTemplateRadius = 1;
		}
		radiusByDim[dim] = std::max(1, localTemplateRadius);
	}

	std::vector<int> currentOffset(dims.size(), 0);
	enumerateOffsets(0, currentOffset, radiusByDim, stride, offsets);

	std::sort(offsets.begin(), offsets.end(), [](const std::vector<int>& left, const std::vector<int>& right) {
		int leftScore = 0;
		int rightScore = 0;
		for (size_t i = 0; i < left.size(); ++i) {
			leftScore += std::abs(left[i]);
			rightScore += std::abs(right[i]);
		}
		if (leftScore != rightScore) {
			return leftScore < rightScore;
		}
		return left < right;
	});

	return offsets;
}

void logPathPositionArray(FILE* reportFile,
	unsigned gridLevel,
	const std::vector<std::vector<int> >& pathPositionArray) {
	fprintf(reportFile,
		"[SNESIM] level %u pathPositionArray begin (size=%lu)\n",
		gridLevel,
		static_cast<unsigned long>(pathPositionArray.size()));
	for (size_t i = 0; i < pathPositionArray.size(); ++i) {
		fprintf(reportFile, "[SNESIM] level %u pathPositionArray[%lu] = (",
			gridLevel,
			static_cast<unsigned long>(i));
		for (size_t j = 0; j < pathPositionArray[i].size(); ++j) {
			if (j > 0) {
				fprintf(reportFile, ",");
			}
			fprintf(reportFile, "%d", pathPositionArray[i][j]);
		}
		fprintf(reportFile, ")\n");
	}
	fprintf(reportFile, "[SNESIM] level %u pathPositionArray end\n", gridLevel);
}

void buildMultigridPlans(const g2s::DataImage& destinationImage,
	const SimulationRunConfig& config,
	std::vector<GridLevelPlan>& levelPlans,
	std::vector<g2s_path_index_t>& posteriorPath,
	FILE* reportFile) {
	const std::vector<unsigned> levels = buildDescendingGridLevels(config.maxGridLevel);
	fprintf(reportFile, "[SNESIM] max grid level=%u (levels %u..0)\n", config.maxGridLevel, config.maxGridLevel);
	const unsigned cellCount = computeCellCount(destinationImage);
	const g2s_path_index_t maxPathValue = std::numeric_limits<g2s_path_index_t>::max();
	levelPlans.clear();
	posteriorPath.assign(cellCount, maxPathValue);
	std::vector<unsigned char> alreadyPlanned(cellCount, 0u);
	for (unsigned cellIndex = 0; cellIndex < cellCount; ++cellIndex) {
		if (!cellHasNaN(destinationImage, cellIndex)) {
			posteriorPath[cellIndex] = g2s_path_index_t(0);
			alreadyPlanned[cellIndex] = 1u;
		}
	}
	std::mt19937 randomGenerator(config.seed);
	g2s_path_index_t nextOrder = g2s_path_index_t(1);
	for (size_t levelIndex = 0; levelIndex < levels.size(); ++levelIndex) {
		GridLevelPlan plan;
		plan.level = levels[levelIndex];
		plan.pathPositionArray = buildPathPositionArrayForLevel(destinationImage._dims, config.templateRadius, plan.level);
		logPathPositionArray(reportFile, plan.level, plan.pathPositionArray);
		for (unsigned cellIndex = 0; cellIndex < cellCount; ++cellIndex) {
			if (alreadyPlanned[cellIndex] != 0u) {
				continue;
			}
			if (!cellBelongsToGridLevel(cellIndex, destinationImage._dims, plan.level)) {
				continue;
			}
			if (!cellHasNaN(destinationImage, cellIndex)) {
				continue;
			}
			plan.simulationPath.push_back(static_cast<g2s_path_index_t>(cellIndex));
		}
		// Randomized simulation path for this level (seeded for reproducibility).
		std::shuffle(plan.simulationPath.begin(), plan.simulationPath.end(), randomGenerator);
		// Posterior path is global across levels and records simulation order.
		for (size_t i = 0; i < plan.simulationPath.size(); ++i) {
			const g2s_path_index_t cellIndex = plan.simulationPath[i];
			alreadyPlanned[cellIndex] = 1u;
			posteriorPath[cellIndex] = nextOrder++;
		}
		fprintf(reportFile,
			"[SNESIM] level %u planned with %lu nodes and %lu path offsets\n",
			plan.level,
			static_cast<unsigned long>(plan.simulationPath.size()),
			static_cast<unsigned long>(plan.pathPositionArray.size()));
		levelPlans.push_back(plan);
	}
}

bool writePosteriorCsv2D(const std::string& outputCsvPath,
	const std::vector<unsigned>& dims,
	const std::vector<g2s_path_index_t>& posteriorPath,
	std::string& errorMessage) {
	// Temporary debug export to validate multigrid/posterior ordering on 2D test cases.
	// Keep for now; can be gated or removed once tree logic is fully validated.
	if (dims.size() != 2u) {
		errorMessage = "posterior CSV export skipped: destination image is not 2D";
		return false;
	}
	const unsigned nx = dims[0];
	const unsigned ny = dims[1];
	const size_t cellCount = static_cast<size_t>(nx) * static_cast<size_t>(ny);
	if (posteriorPath.size() < cellCount) {
		errorMessage = "posterior CSV export skipped: posterior size does not match 2D grid";
		return false;
	}

	std::ofstream outFile(outputCsvPath.c_str(), std::ios::out | std::ios::trunc);
	if (!outFile.good()) {
		errorMessage = "cannot open posterior CSV file: " + outputCsvPath;
		return false;
	}

	for (unsigned y = 0; y < ny; ++y) {
		for (unsigned x = 0; x < nx; ++x) {
			if (x > 0u) {
				outFile << ",";
			}
			const size_t index = static_cast<size_t>(x) + static_cast<size_t>(y) * static_cast<size_t>(nx);
			outFile << posteriorPath[index];
		}
		outFile << "\n";
	}

	if (!outFile.good()) {
		errorMessage = "error while writing posterior CSV file: " + outputCsvPath;
		return false;
	}
	return true;
}

std::vector<std::vector<float> > buildCategoriesValues(const snesim::TrainingImageSummary& summary) {
	std::vector<std::vector<float> > categoriesValues;
	for (unsigned variableIndex = 0; variableIndex < summary.nbVariable; ++variableIndex) {
		std::vector<float> categories;
		categories.reserve(summary.categories.size());
		for (size_t i = 0; i < summary.categories.size(); ++i) {
			categories.push_back(static_cast<float>(summary.categories[i]));
		}
		categoriesValues.push_back(categories);
	}
	return categoriesValues;
}

class SNESIMSamplingModule : public SamplingModule {
public:
	SNESIMSamplingModule(std::vector<ComputeDeviceModule*>* computeDevices,
		const std::vector<g2s::DataImage>& trainingImages,
		std::vector<snesim::SNESIMCPUThreadDevice>* workers,
		TreeStrategy treeStrategy) :
		SamplingModule(computeDevices, nullptr),
		_trainingImages(trainingImages),
		_workers(workers),
		_treeStrategy(treeStrategy) {
		buildCategoryLookup();
	}

	matchLocation sample(std::vector<std::vector<int> > neighborArrayVector,
		std::vector<std::vector<float> > neighborValueArrayVector,
		float seed,
		matchLocation /*verbatimRecord*/,
		unsigned moduleID = 0,
		bool /*fullStationary*/ = false,
		unsigned /*variableOfInterest*/ = 0,
		float /*localk*/ = 0.f,
		int idTI4Sampling = -1,
		g2s::DataImage* /*localKernel*/ = nullptr) {
		const unsigned trainingImageIndex = resolveTrainingImageIndex(idTI4Sampling);
		matchLocation fallback = (_treeStrategy == TreeStrategy::Merged) ?
			randomFromAll(seed) :
			randomFromTi(trainingImageIndex, seed);
		if (_workers == nullptr || _workers->empty()) {
			return fallback;
		}

		const unsigned workerIndex = moduleID % _workers->size();
		std::mt19937 randomGenerator(static_cast<unsigned>(seed * 4294967295.0f) ^ (workerIndex * 2654435761u));
		const int category = (*_workers)[workerIndex].simulatePixel(_trainingImages[trainingImageIndex],
			0u,
			0u,
			trainingImageIndex,
			neighborArrayVector,
			neighborValueArrayVector,
			randomGenerator);

		if (_treeStrategy == TreeStrategy::Merged) {
			std::map<int, std::vector<matchLocation> >::const_iterator categoryList = _categoryCandidates.find(category);
			if (categoryList == _categoryCandidates.end() || categoryList->second.empty()) {
				return fallback;
			}
			return selectCandidate(categoryList->second, seed);
		}
		if (trainingImageIndex >= _categoryCandidatesByTi.size()) {
			return fallback;
		}
		std::map<int, std::vector<matchLocation> >::const_iterator categoryList = _categoryCandidatesByTi[trainingImageIndex].find(category);
		if (categoryList == _categoryCandidatesByTi[trainingImageIndex].end() || categoryList->second.empty()) {
			return fallback;
		}
		return selectCandidate(categoryList->second, seed);
	}

	narrownessMeasurment narrowness(std::vector<std::vector<int> > neighborArrayVector,
		std::vector<std::vector<float> > neighborValueArrayVector,
		float seed,
		unsigned moduleID = 0,
		bool fullStationary = false) {
		narrownessMeasurment output;
		output.candidate = sample(neighborArrayVector, neighborValueArrayVector, seed, matchLocation{0u, 0u}, moduleID, fullStationary);
		output.narrowness = 0.f;
		return output;
	}

private:
	unsigned resolveTrainingImageIndex(int idTI4Sampling) const {
		if (_trainingImages.empty()) {
			return 0u;
		}
		if (_treeStrategy == TreeStrategy::First) {
			return 0u;
		}
		if (idTI4Sampling >= 0 && idTI4Sampling < static_cast<int>(_trainingImages.size())) {
			return static_cast<unsigned>(idTI4Sampling);
		}
		return 0u;
	}

	void buildCategoryLookup() {
		_allCandidatesByTi.assign(_trainingImages.size(), std::vector<matchLocation>());
		_categoryCandidatesByTi.assign(_trainingImages.size(), std::map<int, std::vector<matchLocation> >());
		for (size_t tiIndex = 0; tiIndex < _trainingImages.size(); ++tiIndex) {
			const g2s::DataImage& ti = _trainingImages[tiIndex];
			if (ti._nbVariable == 0) {
				continue;
			}
			const unsigned cellCount = computeCellCount(ti);
			for (unsigned cellIndex = 0; cellIndex < cellCount; ++cellIndex) {
				const float value = ti._data[cellIndex * ti._nbVariable + 0];
				if (!std::isfinite(value)) {
					continue;
				}
				const int category = static_cast<int>(std::round(value));
				matchLocation location;
				location.TI = static_cast<unsigned>(tiIndex);
				location.index = cellIndex;
				_allCandidates.push_back(location);
				_allCandidatesByTi[tiIndex].push_back(location);
				_categoryCandidates[category].push_back(location);
				_categoryCandidatesByTi[tiIndex][category].push_back(location);
			}
		}
	}

	matchLocation randomFromTi(unsigned tiIndex, float seed) const {
		if (tiIndex < _allCandidatesByTi.size() && !_allCandidatesByTi[tiIndex].empty()) {
			return selectCandidate(_allCandidatesByTi[tiIndex], seed);
		}
		return randomFromAll(seed);
	}

	matchLocation randomFromAll(float seed) const {
		if (_allCandidates.empty()) {
			return matchLocation{0u, 0u};
		}
		return selectCandidate(_allCandidates, seed);
	}

	matchLocation selectCandidate(const std::vector<matchLocation>& candidates, float seed) const {
		if (candidates.empty()) {
			return matchLocation{0u, 0u};
		}
		const float localSeed = std::max(0.f, std::min(seed, std::nextafter(1.f, 0.f)));
		const size_t index = std::min(candidates.size() - 1, static_cast<size_t>(std::floor(localSeed * candidates.size())));
		return candidates[index];
	}

private:
	const std::vector<g2s::DataImage>& _trainingImages;
	std::vector<snesim::SNESIMCPUThreadDevice>* _workers = nullptr;
	TreeStrategy _treeStrategy = TreeStrategy::Merged;
	std::vector<matchLocation> _allCandidates;
	std::vector<std::vector<matchLocation> > _allCandidatesByTi;
	std::map<int, std::vector<matchLocation> > _categoryCandidates;
	std::vector<std::map<int, std::vector<matchLocation> > > _categoryCandidatesByTi;
};

} // namespace

int main(int argc, char const* argv[]) {
	CliOptions options;
	if (!parseCliOptions(argc, argv, options)) {
		if (options.closeReportFile && options.reportFile) {
			fclose(options.reportFile);
		}
		return 0;
	}
	FILE* reportFile = options.reportFile;
	if (options.verbose) {
		fprintf(reportFile, "[SNESIM] verbose mode enabled\n");
	}

	std::vector<g2s::DataImage> trainingImages;
	std::vector<snesim::TrainingImageSummary> trainingSummaries;
	trainingImages.reserve(options.trainingImageNames.size());
	trainingSummaries.reserve(options.trainingImageNames.size());

	snesim::TreeCacheRepository cacheRepository(options.treeRoot);
	for (size_t i = 0; i < options.trainingImageNames.size(); ++i) {
		const std::string& trainingImageName = options.trainingImageNames[i];
		g2s::DataImage trainingImage = g2s::DataImage::createFromFile(trainingImageName);

		snesim::TrainingImageSummary summary;
		std::string summaryError;
		if (!snesim::summarizeCategoricalTrainingImage(trainingImage, trainingImageName, summary, summaryError)) {
			fprintf(reportFile, "[SNESIM] %s\n", summaryError.c_str());
			if (options.closeReportFile) {
				fclose(reportFile);
			}
			return 0;
		}

		if (!trainingSummaries.empty()) {
			if (summary.nbVariable != trainingSummaries[0].nbVariable || summary.categories != trainingSummaries[0].categories) {
				fprintf(reportFile, "[SNESIM] all TIs must share the same variable count and categories in this scaffold\n");
				if (options.closeReportFile) {
					fclose(reportFile);
				}
				return 0;
			}
		}

		trainingImages.push_back(std::move(trainingImage));
		trainingSummaries.push_back(summary);
	}

	g2s::DataImage destinationImage = g2s::DataImage::createFromFile(options.destinationImageName);
	const snesim::TrainingImageSummary& summary = trainingSummaries[0];
	std::set<int> categorySet(summary.categories.begin(), summary.categories.end());
	std::string destinationError;
	if (!validateDestinationImage(destinationImage, summary.nbVariable, categorySet, destinationError)) {
		fprintf(reportFile, "[SNESIM] %s\n", destinationError.c_str());
		if (options.closeReportFile) {
			fclose(reportFile);
		}
		return 0;
	}

	g2s::DataImage tiSelectionImage;
	const bool useTiSelectionImage = !options.trainingImageIndexName.empty();
	if (useTiSelectionImage) {
		tiSelectionImage = g2s::DataImage::createFromFile(options.trainingImageIndexName);
		std::string tiSelectionError;
		if (!validateTrainingImageSelectionImage(tiSelectionImage, destinationImage, static_cast<unsigned>(trainingImages.size()), tiSelectionError)) {
			fprintf(reportFile, "[SNESIM] %s\n", tiSelectionError.c_str());
			if (options.closeReportFile) {
				fclose(reportFile);
			}
			return 0;
		}
	}

	fprintf(reportFile, "[SNESIM] tree strategy: %s\n", treeStrategyName(options.treeStrategy));

	std::vector<GridLevelPlan> levelPlans;
	std::vector<g2s_path_index_t> posteriorPath;
	// Global multigrid planning point:
	// per-level simulation path + deterministic pathPositionArray + shared posterior path.
	buildMultigridPlans(destinationImage, options.simulationConfig, levelPlans, posteriorPath, reportFile);

	std::map<unsigned, std::vector<std::shared_ptr<const snesim::SearchTree> > > treesByLevel;
	std::map<unsigned, std::shared_ptr<const snesim::SearchTree> > mergedTreeByLevel;
	for (size_t i = 0; i < levelPlans.size(); ++i) {
		const GridLevelPlan& levelPlan = levelPlans[i];
		// Build/load one tree per TI for this exact level so each level uses
		// its own deterministic pathPositionArray statistics.
		snesim::TreeBuildConfig levelTreeConfig = options.treeBuildConfig;
		levelTreeConfig.gridLevel = levelPlan.level;

		std::vector<std::shared_ptr<const snesim::SearchTree> > treesByTrainingImage;
		treesByTrainingImage.reserve(trainingImages.size());
		for (size_t tiIndex = 0; tiIndex < trainingImages.size(); ++tiIndex) {
			std::shared_ptr<const snesim::SearchTree> levelTree = loadOrCreateTree(
				options.trainingImageNames[tiIndex],
				trainingImages[tiIndex],
				trainingSummaries[tiIndex],
				levelTreeConfig,
				levelPlan.pathPositionArray,
				options.forceTreeRebuild,
				cacheRepository,
				reportFile);
			if (!levelTree) {
				fprintf(reportFile,
					"[SNESIM] failed to build/load tree for TI '%s' at level %u\n",
					options.trainingImageNames[tiIndex].c_str(),
					levelPlan.level);
				if (options.closeReportFile) {
					fclose(reportFile);
				}
				return 0;
			}
			treesByTrainingImage.push_back(levelTree);
		}

		treesByLevel[levelPlan.level] = treesByTrainingImage;

		if (options.treeStrategy == TreeStrategy::Merged) {
			std::shared_ptr<const snesim::SearchTree> mergedTree = buildMergedTree(treesByTrainingImage);
			if (!mergedTree) {
				fprintf(reportFile, "[SNESIM] failed to build merged tree at level %u\n", levelPlan.level);
				if (options.closeReportFile) {
					fclose(reportFile);
				}
				return 0;
			}
			mergedTreeByLevel[levelPlan.level] = mergedTree;
		}
	}

	snesim::SNESIMCPUThreadDevice::clearSharedTrees();
	snesim::SNESIMCPUThreadDevice::setSharedTrees(treesByLevel);
	for (size_t i = 0; i < levelPlans.size(); ++i) {
		snesim::SNESIMCPUThreadDevice::setPathPositionArrayForLevel(levelPlans[i].level, levelPlans[i].pathPositionArray);
	}
	for (std::map<unsigned, std::shared_ptr<const snesim::SearchTree> >::const_iterator it = mergedTreeByLevel.begin();
		it != mergedTreeByLevel.end();
		++it) {
		snesim::SNESIMCPUThreadDevice::setMergedTreeForLevel(it->first, it->second);
	}

	snesim::SNESIMCPUThreadDevice::setTreeSelectionMode(
		options.treeStrategy == TreeStrategy::Merged ?
			snesim::TreeSelectionMode::Merged :
			snesim::TreeSelectionMode::PerTrainingImage);

	std::shared_ptr<const snesim::SearchTree> workerFallbackTree;
	if (!levelPlans.empty()) {
		const unsigned fallbackLevel = levelPlans[0].level;
		std::map<unsigned, std::vector<std::shared_ptr<const snesim::SearchTree> > >::const_iterator levelTrees = treesByLevel.find(fallbackLevel);
		if (levelTrees != treesByLevel.end() && !levelTrees->second.empty()) {
			workerFallbackTree = levelTrees->second[0];
		}
	}

	std::vector<snesim::SNESIMCPUThreadDevice> workers;
	workers.reserve(std::max(1u, options.simulationConfig.nbThreads));
	for (unsigned workerId = 0; workerId < std::max(1u, options.simulationConfig.nbThreads); ++workerId) {
		workers.push_back(snesim::SNESIMCPUThreadDevice(workerId, workerFallbackTree));
	}

	std::vector<ComputeDeviceModule*> noComputeDevices;
	SNESIMSamplingModule samplingModule(&noComputeDevices, trainingImages, &workers, options.treeStrategy);
	std::vector<g2s::DataImage> kernels;
	std::vector<std::vector<float> > categoriesValues = buildCategoriesValues(summary);
	std::vector<unsigned> numberNeighbor(1, 16u);
	std::vector<unsigned> importDataIndex(computeCellCount(destinationImage), 0u);
	std::mt19937 randomGenerator(options.simulationConfig.seed);
	std::uniform_real_distribution<float> seedDistribution(0.f, std::nextafter(1.f, 0.f));

	for (size_t levelIndex = 0; levelIndex < levelPlans.size(); ++levelIndex) {
		const GridLevelPlan& levelPlan = levelPlans[levelIndex];
		if (levelPlan.simulationPath.empty()) {
			fprintf(reportFile, "[SNESIM] level %u skipped (empty path)\n", levelPlan.level);
			continue;
		}

		snesim::SNESIMCPUThreadDevice::setGlobalGridLevel(levelPlan.level);
		std::vector<float> seedArray(levelPlan.simulationPath.size(), 0.f);
		for (size_t i = 0; i < seedArray.size(); ++i) {
			seedArray[i] = seedDistribution(randomGenerator);
		}

		std::vector<std::vector<std::vector<int> > > pathPositionArray(1);
		pathPositionArray[0] = levelPlan.pathPositionArray;

		fprintf(reportFile,
			"[SNESIM] level %u running default simulation() with %lu path nodes\n",
			levelPlan.level,
			static_cast<unsigned long>(levelPlan.simulationPath.size()));

		simulation(reportFile,
			destinationImage,
			trainingImages,
			kernels,
			samplingModule,
			pathPositionArray,
			const_cast<g2s_path_index_t*>(levelPlan.simulationPath.data()),
			static_cast<g2s_path_index_t>(levelPlan.simulationPath.size()),
			(useTiSelectionImage ? &tiSelectionImage : nullptr),
			nullptr,
			seedArray.data(),
			importDataIndex.data(),
			numberNeighbor,
			nullptr,
			nullptr,
			categoriesValues,
			std::max(1u, options.simulationConfig.nbThreads),
			false,
			false,
			false,
			false,
			posteriorPath.data(),
			nullptr,
			nullptr);
	}

	if (options.outputName.empty()) {
		const unsigned conventionalUniqueID = (options.uniqueID == std::numeric_limits<unsigned>::max()) ? 0u : options.uniqueID;
		options.outputName = std::string("im_1_") + std::to_string(conventionalUniqueID);
	}

	destinationImage.write(options.outputName);
	fprintf(reportFile, "[SNESIM] output written with id '%s'\n", options.outputName.c_str());
	const unsigned conventionalUniqueID = (options.uniqueID == std::numeric_limits<unsigned>::max()) ? 0u : options.uniqueID;
	const std::string conventionalOutputName = std::string("im_1_") + std::to_string(conventionalUniqueID);
	if (options.outputName != conventionalOutputName) {
		destinationImage.write(conventionalOutputName);
		fprintf(reportFile, "[SNESIM] conventional output also written with id '%s'\n", conventionalOutputName.c_str());
	}
	// Temporary debug output: dump posterior path as 2D CSV to inspect ordering.
	const std::string posteriorCsvPath = options.outputName + "_posterior.csv";
	std::string posteriorCsvError;
	if (writePosteriorCsv2D(posteriorCsvPath, destinationImage._dims, posteriorPath, posteriorCsvError)) {
		fprintf(reportFile, "[SNESIM] posterior CSV written to '%s'\n", posteriorCsvPath.c_str());
	} else {
		fprintf(reportFile, "[SNESIM] %s\n", posteriorCsvError.c_str());
	}

	if (options.closeReportFile) {
		fclose(reportFile);
	}
	return 0;
}
