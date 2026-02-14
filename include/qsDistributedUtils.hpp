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
	size_t rowMajorJobPosition=0;
	size_t flattenedJobCount=0;
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
	arg.erase("-eg");
	arg.erase("-endpoint_grid_json");
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
#include <zmq.hpp>

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

inline bool flattenRowMajorJobGrid(const Json::Value& node, std::vector<unsigned>& out, std::string& error){
	if(node.isArray()){
		for (Json::ArrayIndex i = 0; i < node.size(); ++i)
		{
			if(!flattenRowMajorJobGrid(node[i],out,error)){
				return false;
			}
		}
		return true;
	}
	unsigned localId=0;
	if(!decodeJobId(node,localId,error)){
		return false;
	}
	out.push_back(localId);
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

	std::vector<unsigned> flattenedJobIds;
	if(!qs_distributed_utils::flattenRowMajorJobGrid(parsedJobGrid,flattenedJobIds,dmError)){
		fprintf(reportFile, "distributed mode error: invalid job grid content (%s)\n", dmError.c_str());
		return false;
	}

	if(!qs_distributed_utils::locateUniqueJobPosition(flattenedJobIds,localJobId,distributedOptions.rowMajorJobPosition,dmError)){
		fprintf(reportFile, "distributed mode error: cannot locate job id %u in row-major grid (%s)\n", localJobId, dmError.c_str());
		return false;
	}

	distributedOptions.flattenedJobCount=flattenedJobIds.size();
	fprintf(reportFile, "distributed mode: job id %u is at row-major index %zu of %zu\n",
		localJobId,distributedOptions.rowMajorJobPosition,distributedOptions.flattenedJobCount);
	return true;
}

#endif // G2S_QS_DISTRIBUTED

#endif // QS_DISTRIBUTED_UTILS_HPP
