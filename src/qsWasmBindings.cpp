/*
 * Emscripten bindings for the in-memory QS core.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifdef __EMSCRIPTEN__

#include "qsCore.hpp"

#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>

using emscripten::val;

namespace {

bool has(const val& object, const char* key){
	return object.call<bool>("hasOwnProperty",std::string(key));
}

std::vector<unsigned> unsignedVector(const val& array){
	const unsigned length=array["length"].as<unsigned>();
	std::vector<unsigned> values(length);
	for(unsigned index=0;index<length;++index) values[index]=array[index].as<unsigned>();
	return values;
}

std::vector<float> floatVector(const val& array){
	const unsigned length=array["length"].as<unsigned>();
	std::vector<float> values(length);
	for(unsigned index=0;index<length;++index) values[index]=array[index].as<float>();
	return values;
}

void appendFlag(std::vector<std::string>& arguments, const val& options, const char* property, const char* flag){
	if(has(options,property) && options[property].as<bool>()) arguments.push_back(flag);
}

val runQs(val request, val progressCallback){
	if(!has(request,"algorithm") || request["algorithm"].as<std::string>()!="qs"){
		val output=val::object();
		output.set("error","The WebAssembly module supports only QS");
		return output;
	}
	g2s::qs::Request coreRequest;
	const val inputArrays=request["arrays"];
	const unsigned arrayCount=inputArrays["length"].as<unsigned>();
	for(unsigned index=0;index<arrayCount;++index){
		const val input=inputArrays[index];
		g2s::qs::Array array;
		array.name=input["name"].as<std::string>();
		array.dimensions=unsignedVector(input["shape"]);
		array.variableTypes=unsignedVector(input["variableTypes"]);
		array.values=floatVector(input["data"]);
		coreRequest.arrays.push_back(std::move(array));
	}

	const val options=request["options"];
	if(has(options,"candidates") && std::isfinite(options["candidates"].as<double>())){
		coreRequest.arguments.push_back("-k");
		coreRequest.arguments.push_back(std::to_string(options["candidates"].as<double>()));
	}else if(has(options,"maximumExplorationRatio") && std::isfinite(options["maximumExplorationRatio"].as<double>())){
		coreRequest.arguments.push_back("-f");
		coreRequest.arguments.push_back(std::to_string(options["maximumExplorationRatio"].as<double>()));
	}
	if(has(options,"neighbors")){
		const val neighbors=options["neighbors"];
		const unsigned count=neighbors["length"].as<unsigned>();
		if(count>0){
			coreRequest.arguments.push_back("-n");
			for(unsigned index=0;index<count;++index) coreRequest.arguments.push_back(std::to_string(neighbors[index].as<unsigned>()));
		}
	}
	if(has(options,"seed")){
		coreRequest.arguments.push_back("-s");
		coreRequest.arguments.push_back(std::to_string(options["seed"].as<unsigned>()));
	}
	if(has(options,"threads")){
		const double threads=options["threads"].as<double>();
		if(!std::isfinite(threads) || threads<1.0 || std::floor(threads)!=threads){
			throw std::invalid_argument("Effective browser thread count must be a positive integer");
		}
		coreRequest.arguments.push_back("-j");
		coreRequest.arguments.push_back(std::to_string(static_cast<unsigned>(threads)));
	}
	if(has(options,"mode") && options["mode"].as<std::string>()=="full") coreRequest.arguments.push_back("-fs");
	appendFlag(coreRequest.arguments,options,"forceSimulation","--forceSimulation");
	appendFlag(coreRequest.arguments,options,"circularTrainingImage","-cti");
	appendFlag(coreRequest.arguments,options,"circularSimulation","-csim");
	appendFlag(coreRequest.arguments,options,"noVerbatim","-nV");
	appendFlag(coreRequest.arguments,options,"fullStationary","-far");
	appendFlag(coreRequest.arguments,options,"pathOptimization","-wPO");
	appendFlag(coreRequest.arguments,options,"maximumNeighborhood","-maxNK");
	if(has(options,"distance")){
		const std::string distance=options["distance"].as<std::string>();
		if(distance=="kernel") coreRequest.arguments.push_back("-wd");
		else if(distance=="manhattan") coreRequest.arguments.push_back("-md");
		else if(distance=="euclidean") coreRequest.arguments.push_back("-ed");
	}
	if(has(options,"kernelSize") && std::isfinite(options["kernelSize"].as<double>())){
		coreRequest.arguments.push_back("-ks");
		coreRequest.arguments.push_back(std::to_string(options["kernelSize"].as<double>()));
	}
	if(has(options,"alpha") && std::isfinite(options["alpha"].as<double>())){
		coreRequest.arguments.push_back("-alpha");
		coreRequest.arguments.push_back(std::to_string(options["alpha"].as<double>()));
	}

	g2s::qs::Observer observer;
	observer.progress=[progressCallback](double percent,const std::string& detail){
		progressCallback(percent,detail);
	};
	g2s::qs::Result coreResult;
	std::string error;
	if(!g2s::qs::runInMemory(coreRequest,coreResult,error,&observer)){
		val output=val::object();
		output.set("error",error);
		return output;
	}

	val output=val::object();
	val arrays=val::array();
	for(size_t arrayIndex=0;arrayIndex<coreResult.arrays.size();++arrayIndex){
		const g2s::qs::Array& source=coreResult.arrays[arrayIndex];
		val destination=val::object();
		destination.set("name",source.name);
		destination.set("encoding",source.encoding);
		val shape=val::array();
		for(size_t index=0;index<source.dimensions.size();++index) shape.set(index,source.dimensions[index]);
		destination.set("shape",shape);
		val types=val::array();
		for(size_t index=0;index<source.variableTypes.size();++index) types.set(index,source.variableTypes[index]);
		destination.set("variableTypes",types);
		val data;
		if(source.encoding=="uint32"){
			data=val::global("Uint32Array").new_(source.values.size());
			const uint32_t* raw=reinterpret_cast<const uint32_t*>(source.values.data());
			for(size_t index=0;index<source.values.size();++index) data.set(index,raw[index]);
		}else if(source.encoding=="int32"){
			data=val::global("Int32Array").new_(source.values.size());
			const int32_t* raw=reinterpret_cast<const int32_t*>(source.values.data());
			for(size_t index=0;index<source.values.size();++index) data.set(index,raw[index]);
		}else{
			data=val::global("Float32Array").new_(source.values.size());
			for(size_t index=0;index<source.values.size();++index) data.set(index,source.values[index]);
		}
		destination.set("data",data);
		arrays.set(arrayIndex,destination);
	}
	output.set("arrays",arrays);
	output.set("durationMs",coreResult.durationMs);
	val metadata=val::object();
	for(const auto& entry:coreResult.metadata) metadata.set(entry.first,entry.second);
	output.set("metadata",metadata);
	return output;
}

EMSCRIPTEN_BINDINGS(g2s_qs_browser){
	emscripten::function("runQs",&runQs);
}

} // namespace

#endif
