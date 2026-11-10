/*
 * Reusable in-memory QS entry point.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "qsCore.hpp"
#include "DataImage.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <sys/stat.h>

namespace g2s {
namespace qs {

namespace {
std::atomic<unsigned> nextJobId(1000000);
std::mutex runMutex;

void removeBrowserArtifact(const std::string& name){
#ifdef G2S_BROWSER_BUILD
	(void)std::remove(("/tmp/G2S/data/"+name+".bgrid").c_str());
#else
	(void)name;
#endif
}

bool validateArray(const Array& array, std::string& error){
	if(array.name.empty() || array.dimensions.empty() || array.variableTypes.empty()){
		error="QS array metadata is incomplete";
		return false;
	}
	size_t expected=array.variableTypes.size();
	for(unsigned dimension:array.dimensions){
		if(dimension==0 || expected>std::numeric_limits<size_t>::max()/dimension){
			error="QS array dimensions overflow";
			return false;
		}
		expected*=dimension;
	}
	if(expected!=array.values.size()){
		error="QS array element count does not match its dimensions";
		return false;
	}
	for(unsigned type:array.variableTypes){
		if(type>1){
			error="QS variable types must be continuous (0) or categorical (1)";
			return false;
		}
	}
	return true;
}

std::string parameterForArrayName(const std::string& name){
	if(name=="trainingImage") return "-ti";
	if(name=="destination") return "-di";
	if(name=="kernel") return "-ki";
	if(name=="simulationPath") return "-sp";
	if(name=="trainingImageIndex") return "-ii";
	if(name=="neighborCountMap") return "-ni";
	if(name=="kernelIndexMap") return "-kii";
	if(name=="candidateCountMap") return "-kvi";
	if(name=="rotationMap") return "-rmi";
	if(name=="scaleMap") return "-smi";
	return std::string();
}

Array moveDataImage(const std::string& name, g2s::DataImage&& image){
	Array result;
	result.name=name;
	result.dimensions=image._dims;
	result.variableTypes.reserve(image._types.size());
	for(g2s::DataImage::VariableType type:image._types) result.variableTypes.push_back(static_cast<unsigned>(type));
	if(image._encodingType==g2s::DataImage::UInteger) result.encoding="uint32";
	else if(image._encodingType==g2s::DataImage::Integer) result.encoding="int32";
	result.values.assign(image._data,image._data+image.dataSize());
	return result;
}
}

bool runInMemory(const Request& request, Result& result, std::string& error, const Observer* observer){
	std::lock_guard<std::mutex> runLock(runMutex);
	bool hasTrainingImage=false;
	bool hasDestination=false;
	for(const Array& array:request.arrays){
		if(!validateArray(array,error)) return false;
		hasTrainingImage|=(array.name=="trainingImage");
		hasDestination|=(array.name=="destination");
	}
	if(!hasTrainingImage || !hasDestination){
		error="QS requires a trainingImage and destination";
		return false;
	}

	(void)mkdir("/tmp/G2S",0770);
	(void)mkdir("/tmp/G2S/data",0770);
	(void)mkdir("/tmp/G2S/logs",0770);
	const unsigned jobId=nextJobId.fetch_add(1);
	std::vector<std::string> arguments;
	arguments.push_back("qs");
	arguments.push_back("-id");
	arguments.push_back(std::to_string(jobId));
	arguments.push_back("-r");
	arguments.push_back("/tmp/G2S/logs/"+std::to_string(jobId)+".log");

	unsigned arrayIndex=0;
	std::vector<std::string> storageNames;
	for(const Array& array:request.arrays){
		const std::string parameter=parameterForArrayName(array.name);
		if(parameter.empty()){
			error="unsupported QS array name: "+array.name;
			return false;
		}
		const std::string storageName="wasm_"+std::to_string(jobId)+"_"+std::to_string(arrayIndex++);
		storageNames.push_back(storageName);
		g2s::DataImage image(array.dimensions.size(),const_cast<unsigned*>(array.dimensions.data()),array.variableTypes.size());
		for(size_t index=0;index<array.variableTypes.size();++index){
			image._types[index]=array.variableTypes[index]==1 ? g2s::DataImage::Categorical : g2s::DataImage::Continuous;
		}
		std::memcpy(image._data,array.values.data(),array.values.size()*sizeof(float));
		image.write(storageName,false);
		arguments.push_back(parameter);
		arguments.push_back(storageName);
	}
	arguments.insert(arguments.end(),request.arguments.begin(),request.arguments.end());

	std::vector<const char*> argv;
	argv.reserve(arguments.size());
	for(const std::string& argument:arguments) argv.push_back(argument.c_str());
	setActiveObserver(observer);
	const auto start=std::chrono::steady_clock::now();
	const int returnCode=g2sQsProgramMain(static_cast<int>(argv.size()),argv.data());
	const auto end=std::chrono::steady_clock::now();
	if(returnCode!=0){
		setActiveObserver(nullptr);
		for(const std::string& name:storageNames) removeBrowserArtifact(name);
		error="QS returned error code "+std::to_string(returnCode);
		return false;
	}

	g2s::DataImage simulation=g2s::DataImage::createFromFile(std::to_string(jobId));
	g2s::DataImage index=g2s::DataImage::createFromFile("id_"+std::to_string(jobId));
	if(simulation.isEmpty() || index.isEmpty()){
		setActiveObserver(nullptr);
		for(const std::string& name:storageNames) removeBrowserArtifact(name);
		error="QS did not produce its expected simulation and index outputs";
		return false;
	}
	result=Result();
	result.arrays.push_back(moveDataImage("simulation",std::move(simulation)));
	result.arrays.push_back(moveDataImage("index",std::move(index)));
	result.durationMs=std::chrono::duration<double,std::milli>(end-start).count();
	result.metadata["algorithm"]="qs";
	result.metadata["status"]="success";
	result.metadata["duration_ms"]=std::to_string(static_cast<long long>(result.durationMs));
	notifyProgress(100.0,"completed");
	setActiveObserver(nullptr);
	for(const std::string& name:storageNames) removeBrowserArtifact(name);
	removeBrowserArtifact(std::to_string(jobId));
	removeBrowserArtifact("id_"+std::to_string(jobId));
	removeBrowserArtifact("im_1_"+std::to_string(jobId));
	removeBrowserArtifact("im_2_"+std::to_string(jobId));
	(void)std::remove(("/tmp/G2S/logs/"+std::to_string(jobId)+".log").c_str());
	return true;
}

} // namespace qs
} // namespace g2s
