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

#ifndef QS_DISTRIBUTED_UTILS_HPP
#define QS_DISTRIBUTED_UTILS_HPP

#include <map>
#include <string>
#include <vector>
#include <cstdio>

struct QsDistributedOptions{
	std::string jobGridPayload;
	std::string endpointGridPayload;
	std::string diGridPayload;
	size_t rowMajorJobPosition=0;
	size_t flattenedJobCount=0;
	std::vector<unsigned> gridDims;
	std::vector<unsigned> localJobGridCoordinate;
	std::vector<unsigned> flattenedJobIds;
	std::vector<std::string> flattenedEndpointNames;
	std::vector<std::string> flattenedDiNames;
};

inline void parseDistributedCliArgs(std::multimap<std::string, std::string>& arg, QsDistributedOptions& distributedOptions){
	if (arg.count("-jg") >= 1)
	{
		distributedOptions.jobGridPayload=arg.find("-jg")->second;
	}else if(arg.count("-job_grid_json") >= 1){
		distributedOptions.jobGridPayload=arg.find("-job_grid_json")->second;
	}else if(arg.count("-job_grid") >= 1){
		distributedOptions.jobGridPayload=arg.find("-job_grid")->second;
	}

	arg.erase("-jg");
	arg.erase("-job_grid_json");
	arg.erase("-job_grid");
	if (arg.count("-eg") >= 1)
	{
		distributedOptions.endpointGridPayload=arg.find("-eg")->second;
	}else if(arg.count("-endpoint_grid_json") >= 1){
		distributedOptions.endpointGridPayload=arg.find("-endpoint_grid_json")->second;
	}
	arg.erase("-eg");
	arg.erase("-endpoint_grid_json");
	if (arg.count("-di_grid_json") >= 1)
	{
		distributedOptions.diGridPayload=arg.find("-di_grid_json")->second;
	}
	arg.erase("-di_grid_json");
}

#ifdef G2S_QS_DISTRIBUTED

#include <fstream>
#include <iterator>
#include <cmath>
#include <limits>
#include <cctype>
#include <memory>
#include <exception>
#include <json/json.h>

namespace qs_distributed_utils {

inline std::string trimWhitespace(const std::string& value){
	size_t begin=0;
	size_t end=value.size();
	while(begin<end && std::isspace(static_cast<unsigned char>(value[begin]))){
		++begin;
	}
	while(end>begin && std::isspace(static_cast<unsigned char>(value[end-1]))){
		--end;
	}
	return value.substr(begin,end-begin);
}

inline bool parseJsonPayloadOrFile(const std::string& payload, Json::Value& out, std::string& error){
	auto parseString=[&out,&error](const std::string& content)->bool{
		Json::CharReaderBuilder builder;
		builder["collectComments"]=false;
		builder["allowTrailingCommas"]=true;
		std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
		std::string localError;
		const bool parsed=reader->parse(content.data(),content.data()+content.size(),&out,&localError);
		if(!parsed){
			error=localError;
		}
		return parsed;
	};

	std::ifstream maybeFile(payload.c_str(),std::ios::in|std::ios::binary);
	if(maybeFile.good()){
		std::string content((std::istreambuf_iterator<char>(maybeFile)),std::istreambuf_iterator<char>());
		if(content.empty()){
			error="empty json file";
			return false;
		}
		return parseString(content);
	}
	return parseString(payload);
}

inline bool decodeJobId(const Json::Value& value, unsigned& out, std::string& error){
	unsigned long long parsed=0;
	if(value.isString()){
		const std::string text=trimWhitespace(value.asString());
		if(text.empty()){
			error="empty string job id";
			return false;
		}
		size_t consumed=0;
		try{
			parsed=std::stoull(text,&consumed,10);
		}catch(const std::exception&){
			error="non integer string job id";
			return false;
		}
		if(consumed!=text.size()){
			error="job id string contains non-digit characters";
			return false;
		}
	}
#if defined(JSON_HAS_INT64)
	else if(value.isUInt64()){
		parsed=value.asUInt64();
	}
#endif
	else if(value.isUInt()){
		parsed=value.asUInt();
	}
#if defined(JSON_HAS_INT64)
	else if(value.isInt64()){
		const Json::Int64 local=value.asInt64();
		if(local<0){
			error="negative job id";
			return false;
		}
		parsed=static_cast<unsigned long long>(local);
	}
#endif
	else if(value.isInt()){
		const int local=value.asInt();
		if(local<0){
			error="negative job id";
			return false;
		}
		parsed=static_cast<unsigned long long>(local);
	}else if(value.isDouble()){
		const double local=value.asDouble();
		if(!std::isfinite(local) || local<0.0 || std::floor(local)!=local){
			error="non integer numeric job id";
			return false;
		}
		parsed=static_cast<unsigned long long>(local);
	}else{
		error="unsupported job id type";
		return false;
	}

	if(parsed>std::numeric_limits<unsigned>::max()){
		error="job id overflows unsigned";
		return false;
	}
	out=static_cast<unsigned>(parsed);
	return true;
}

inline bool decodeGridStringValue(const Json::Value& value, std::string& out, std::string& error){
	if(value.isString()){
		out=trimWhitespace(value.asString());
		if(out.empty()){
			error="empty string value in grid";
			return false;
		}
		return true;
	}
#if defined(JSON_HAS_INT64)
	if(value.isUInt64()){
		out=std::to_string(value.asUInt64());
		return true;
	}
#endif
	if(value.isUInt()){
		out=std::to_string(value.asUInt());
		return true;
	}
#if defined(JSON_HAS_INT64)
	if(value.isInt64()){
		const Json::Int64 local=value.asInt64();
		if(local<0){
			error="negative integer value in grid";
			return false;
		}
		out=std::to_string(local);
		return true;
	}
#endif
	if(value.isInt()){
		const int local=value.asInt();
		if(local<0){
			error="negative integer value in grid";
			return false;
		}
		out=std::to_string(local);
		return true;
	}
	if(value.isDouble()){
		const double local=value.asDouble();
		if(!std::isfinite(local) || std::floor(local)!=local || local<0.0){
			error="non integer numeric value in grid";
			return false;
		}
		out=std::to_string(static_cast<unsigned long long>(local));
		return true;
	}
	error="unsupported value type in grid";
	return false;
}

inline bool flattenGridLeavesRowMajor(const Json::Value& node,
	std::vector<unsigned>& gridDims,
	std::vector<unsigned>& coordinateBuffer,
	std::vector<std::vector<unsigned> >& coordinates,
	std::vector<Json::Value>& leaves,
	std::string& error){
	if(node.isArray()){
		if(node.empty()){
			error="empty dimension in grid";
			return false;
		}
		const size_t depth=coordinateBuffer.size();
		const unsigned sizeAtDepth=static_cast<unsigned>(node.size());
		if(gridDims.size()==depth){
			gridDims.push_back(sizeAtDepth);
		}else if(gridDims[depth]!=sizeAtDepth){
			error="ragged grid is not supported";
			return false;
		}
		for (Json::ArrayIndex i = 0; i < node.size(); ++i)
		{
			coordinateBuffer.push_back(static_cast<unsigned>(i));
			if(!flattenGridLeavesRowMajor(node[i],gridDims,coordinateBuffer,coordinates,leaves,error)){
				return false;
			}
			coordinateBuffer.pop_back();
		}
		return true;
	}
	if(coordinateBuffer.empty()){
		error="grid root must be an array";
		return false;
	}
	coordinates.push_back(coordinateBuffer);
	leaves.push_back(node);
	return true;
}

inline size_t rowMajorIndexFromCoordinate(const std::vector<unsigned>& coordinate, const std::vector<unsigned>& gridDims){
	size_t index=0;
	for (size_t i = 0; i < coordinate.size(); ++i)
	{
		index*=gridDims[i];
		index+=coordinate[i];
	}
	return index;
}

inline std::vector<unsigned> coordinateFromRowMajorIndex(size_t index, const std::vector<unsigned>& gridDims){
	std::vector<unsigned> coordinate(gridDims.size(),0);
	for (int i = int(gridDims.size())-1; i >= 0; --i)
	{
		if(gridDims[i]==0){
			continue;
		}
		coordinate[i]=static_cast<unsigned>(index%gridDims[i]);
		index/=gridDims[i];
	}
	return coordinate;
}

inline bool flattenRowMajorJobGrid(const Json::Value& root,
	std::vector<unsigned>& gridDims,
	std::vector<std::vector<unsigned> >& coordinates,
	std::vector<unsigned>& flattenedJobIds,
	std::string& error){
	std::vector<unsigned> coordinateBuffer;
	std::vector<Json::Value> leaves;
	gridDims.clear();
	coordinates.clear();
	flattenedJobIds.clear();
	if(!flattenGridLeavesRowMajor(root,gridDims,coordinateBuffer,coordinates,leaves,error)){
		return false;
	}
	if(leaves.empty()){
		error="empty grid";
		return false;
	}
	flattenedJobIds.resize(leaves.size(),0);
	for (size_t i = 0; i < leaves.size(); ++i)
	{
		if(!decodeJobId(leaves[i],flattenedJobIds[i],error)){
			return false;
		}
	}
	return true;
}

inline bool flattenRowMajorStringGrid(const Json::Value& root,
	std::vector<unsigned>& gridDims,
	std::vector<std::vector<unsigned> >& coordinates,
	std::vector<std::string>& flattenedValues,
	std::string& error){
	std::vector<unsigned> coordinateBuffer;
	std::vector<Json::Value> leaves;
	gridDims.clear();
	coordinates.clear();
	flattenedValues.clear();
	if(!flattenGridLeavesRowMajor(root,gridDims,coordinateBuffer,coordinates,leaves,error)){
		return false;
	}
	if(leaves.empty()){
		error="empty grid";
		return false;
	}
	flattenedValues.resize(leaves.size());
	for (size_t i = 0; i < leaves.size(); ++i)
	{
		if(!decodeGridStringValue(leaves[i],flattenedValues[i],error)){
			return false;
		}
	}
	return true;
}

inline bool locateUniqueJobPosition(const std::vector<unsigned>& flattenedJobIds, unsigned localJobId, size_t& position, std::string& error){
	size_t matchCount=0;
	for (size_t i = 0; i < flattenedJobIds.size(); ++i)
	{
		if(flattenedJobIds[i]==localJobId){
			position=i;
			++matchCount;
		}
	}
	if(matchCount==0){
		error="current job id not present in job grid";
		return false;
	}
	if(matchCount>1){
		error="current job id appears multiple times in job grid";
		return false;
	}
	return true;
}

} // namespace qs_distributed_utils

inline bool resolveDistributedJobPosition(FILE* reportFile, unsigned localJobId, QsDistributedOptions& distributedOptions){
	distributedOptions.rowMajorJobPosition=0;
	distributedOptions.flattenedJobCount=0;
	distributedOptions.gridDims.clear();
	distributedOptions.localJobGridCoordinate.clear();
	distributedOptions.flattenedJobIds.clear();
	distributedOptions.flattenedEndpointNames.clear();
	distributedOptions.flattenedDiNames.clear();

	if(distributedOptions.jobGridPayload.empty()){
		return true;
	}
	if(localJobId==unsigned(-1)){
		fprintf(reportFile, "distributed mode error: local job id is unknown (missing -id and non-standard -r)\n");
		return false;
	}

	Json::Value parsedJobGrid;
	std::string dmError;
	if(!qs_distributed_utils::parseJsonPayloadOrFile(distributedOptions.jobGridPayload,parsedJobGrid,dmError)){
		fprintf(reportFile, "distributed mode error: invalid -jg payload (%s)\n", dmError.c_str());
		return false;
	}

	std::vector<unsigned> gridDims;
	std::vector<std::vector<unsigned> > flattenedCoordinates;
	std::vector<unsigned> flattenedJobIds;
	if(!qs_distributed_utils::flattenRowMajorJobGrid(parsedJobGrid,gridDims,flattenedCoordinates,flattenedJobIds,dmError)){
		fprintf(reportFile, "distributed mode error: invalid job grid content (%s)\n", dmError.c_str());
		return false;
	}

	if(!qs_distributed_utils::locateUniqueJobPosition(flattenedJobIds,localJobId,distributedOptions.rowMajorJobPosition,dmError)){
		fprintf(reportFile, "distributed mode error: cannot locate job id %u in row-major grid (%s)\n", localJobId, dmError.c_str());
		return false;
	}

	distributedOptions.gridDims=gridDims;
	distributedOptions.localJobGridCoordinate=
		qs_distributed_utils::coordinateFromRowMajorIndex(distributedOptions.rowMajorJobPosition,distributedOptions.gridDims);
	distributedOptions.flattenedJobIds=flattenedJobIds;

	distributedOptions.flattenedDiNames.resize(flattenedJobIds.size());
	for (size_t i = 0; i < flattenedJobIds.size(); ++i)
	{
		distributedOptions.flattenedDiNames[i]=std::string("input_di_")+std::to_string(flattenedJobIds[i]);
	}

	distributedOptions.flattenedEndpointNames.resize(flattenedJobIds.size());
	if(!distributedOptions.endpointGridPayload.empty()){
		Json::Value parsedEndpointGrid;
		if(!qs_distributed_utils::parseJsonPayloadOrFile(distributedOptions.endpointGridPayload,parsedEndpointGrid,dmError)){
			fprintf(reportFile, "distributed mode error: invalid -eg payload (%s)\n", dmError.c_str());
			return false;
		}
		std::vector<unsigned> endpointGridDims;
		std::vector<std::vector<unsigned> > endpointCoordinates;
		std::vector<std::string> endpointNames;
		if(!qs_distributed_utils::flattenRowMajorStringGrid(parsedEndpointGrid,endpointGridDims,endpointCoordinates,endpointNames,dmError)){
			fprintf(reportFile, "distributed mode error: invalid -eg content (%s)\n", dmError.c_str());
			return false;
		}
		if(endpointGridDims!=distributedOptions.gridDims){
			fprintf(reportFile, "distributed mode error: -eg shape does not match -jg shape\n");
			return false;
		}
		if(endpointNames.size()!=distributedOptions.flattenedJobIds.size()){
			fprintf(reportFile, "distributed mode error: -eg size does not match -jg size\n");
			return false;
		}
		distributedOptions.flattenedEndpointNames=endpointNames;
	}else{
		for (size_t i = 0; i < distributedOptions.flattenedEndpointNames.size(); ++i)
		{
			distributedOptions.flattenedEndpointNames[i]=std::string("localhost:")+std::to_string(8130+int(i));
		}
	}

	if(!distributedOptions.diGridPayload.empty()){
		Json::Value parsedDiGrid;
		if(!qs_distributed_utils::parseJsonPayloadOrFile(distributedOptions.diGridPayload,parsedDiGrid,dmError)){
			fprintf(reportFile, "distributed mode error: invalid -di_grid_json payload (%s)\n", dmError.c_str());
			return false;
		}
		std::vector<unsigned> diGridDims;
		std::vector<std::vector<unsigned> > diCoordinates;
		std::vector<std::string> diNames;
		if(!qs_distributed_utils::flattenRowMajorStringGrid(parsedDiGrid,diGridDims,diCoordinates,diNames,dmError)){
			fprintf(reportFile, "distributed mode error: invalid -di_grid_json content (%s)\n", dmError.c_str());
			return false;
		}
		if(diGridDims!=distributedOptions.gridDims){
			fprintf(reportFile, "distributed mode error: -di_grid_json shape does not match -jg shape\n");
			return false;
		}
		if(diNames.size()!=distributedOptions.flattenedJobIds.size()){
			fprintf(reportFile, "distributed mode error: -di_grid_json size does not match -jg size\n");
			return false;
		}
		distributedOptions.flattenedDiNames=diNames;
	}

	distributedOptions.flattenedJobCount=flattenedJobIds.size();
	fprintf(reportFile, "distributed mode: job id %u is at row-major index %zu of %zu\n",
		localJobId,distributedOptions.rowMajorJobPosition,distributedOptions.flattenedJobCount);
	return true;
}

#endif // G2S_QS_DISTRIBUTED

#endif // QS_DISTRIBUTED_UTILS_HPP
