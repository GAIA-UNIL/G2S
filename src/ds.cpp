/*
 * G2S
 * Copyright (C) 2018, Mathieu Gravey (gravey.mathieu@gmail.com) and UNIL
 *
 * Native Direct Sampling implementation.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>

#include "utils.hpp"
#include "DataImage.hpp"
#include "jobManager.hpp"
#include "jobReporting.hpp"
#include "simulation.hpp"
#include "directSampling.hpp"

void printHelp(){
	printf("Native Direct Sampling (DS)\n");
	printf("Required: -ti <training image> -di <destination image> -dt <types> -th <threshold> -f <exploration ratio> -n <neighbors>\n");
	printf("Optional: -wPO enables vector simulation path optimization, matching the QS flag behavior\n");
}

static bool parsePositiveFloatList(const std::string& rawValue, const char* flagName, FILE* reportFile, std::vector<float>& values){
	std::string normalized=rawValue;
	for (size_t i = 0; i < normalized.size(); ++i)
	{
		if(normalized[i]==',' || normalized[i]==';'){
			normalized[i]=' ';
		}
	}
	std::stringstream stream(normalized);
	std::string token;
	bool found=false;
	while(stream>>token){
		found=true;
		char* endPtr=nullptr;
		const float parsed=strtof(token.c_str(),&endPtr);
		if(endPtr==token.c_str() || *endPtr!='\0' || !std::isfinite(parsed) || parsed<=0.f){
			fprintf(reportFile,"error: %s expects strictly positive finite value(s), got '%s'\n",flagName,rawValue.c_str());
			return false;
		}
		values.push_back(parsed);
	}
	if(!found){
		fprintf(reportFile,"error: %s expects at least one value\n",flagName);
		return false;
	}
	return true;
}

static void normalizeLeadingVariableDim(g2s::DataImage& image, const std::vector<unsigned>& outputDims){
	if(!image.isEmpty() && image._nbVariable==1 && image._dims.size()==outputDims.size()+1){
		image.convertFirstDimInVariable();
	}
}

static bool validateFiniteMap(const g2s::DataImage& image, bool strictlyPositive, bool nonNegative){
	for (unsigned i = 0; i < const_cast<g2s::DataImage&>(image).dataSize(); ++i)
	{
		const float value=image._data[i];
		if(!std::isfinite(value)){
			return false;
		}
		if(strictlyPositive && value<=0.f){
			return false;
		}
		if(nonNegative && value<0.f){
			return false;
		}
	}
	return true;
}

static std::vector<unsigned char> protectedSamples(const g2s::DataImage& image, bool forceSimulation){
	std::vector<unsigned char> protectedMask(const_cast<g2s::DataImage&>(image).dataSize(),0);
	if(forceSimulation){
		return protectedMask;
	}
	for (unsigned i = 0; i < const_cast<g2s::DataImage&>(image).dataSize(); ++i)
	{
		protectedMask[i]=std::isfinite(image._data[i]);
	}
	return protectedMask;
}

static void cleanCategoricalSingletons(g2s::DataImage& image, const std::vector<unsigned char>& protectedMask){
	if(image._dims.empty()){
		return;
	}
	const unsigned cellCount=image.dataSize()/image._nbVariable;
	std::vector<float> cleaned(image._data,image._data+image.dataSize());
	std::vector<int> offset(image._dims.size(),0);
	const unsigned requiredMajority=std::max<unsigned>(2,image._dims.size());
	for (unsigned cell = 0; cell < cellCount; ++cell)
	{
		for (unsigned variable = 0; variable < image._nbVariable; ++variable)
		{
			if(image._types[variable]!=g2s::DataImage::Categorical){
				continue;
			}
			const unsigned flat=cell*image._nbVariable+variable;
			if(flat<protectedMask.size() && protectedMask[flat]){
				continue;
			}
			const float center=image._data[flat];
			if(!std::isfinite(center)){
				continue;
			}
			unsigned same=0;
			std::map<float,unsigned> neighborCounts;
			for (size_t dim = 0; dim < image._dims.size(); ++dim)
			{
				for (int direction = -1; direction <= 1; direction+=2)
				{
					std::fill(offset.begin(),offset.end(),0);
					offset[dim]=direction;
					unsigned neighborCell=0;
					if(!image.indexWithDelta(neighborCell,cell,offset)){
						continue;
					}
					const float value=image._data[neighborCell*image._nbVariable+variable];
					if(!std::isfinite(value)){
						continue;
					}
					if(value==center){
						same++;
					}else{
						neighborCounts[value]++;
					}
				}
			}
			if(same>0 || neighborCounts.empty()){
				continue;
			}
			float bestValue=center;
			unsigned bestCount=0;
			for (std::map<float,unsigned>::const_iterator it=neighborCounts.begin(); it!=neighborCounts.end(); ++it)
			{
				if(it->second>bestCount){
					bestValue=it->first;
					bestCount=it->second;
				}
			}
			if(bestCount>=requiredMajority){
				cleaned[flat]=bestValue;
			}
		}
	}
	std::copy(cleaned.begin(),cleaned.end(),image._data);
}

struct DsPathEntry{
	std::vector<int> offset;
	int flatIndex=-1;
	float priority=0.f;
};

static void buildPathForKernel(g2s::DataImage& kernel, g2s::DistanceType searchDistance, std::vector<std::vector<int> >& offsets, std::vector<int>& flatIndices){
	std::vector<std::vector<int> > localPath;
	localPath.push_back(std::vector<int>(0));
	for (size_t dim = 0; dim < kernel._dims.size(); ++dim)
	{
		const unsigned originalSize=localPath.size();
		const int sizeInThisDim=(kernel._dims[dim]+1)/2;
		localPath.resize(originalSize*(2*sizeInThisDim-1));
		for (unsigned k = 0; k < originalSize; ++k)
		{
			localPath[k].push_back(0);
		}
		for (int j = 1; j < sizeInThisDim; ++j)
		{
			std::copy(localPath.begin(),localPath.begin()+originalSize,localPath.begin()+originalSize*(-1+2*j+0));
			std::copy(localPath.begin(),localPath.begin()+originalSize,localPath.begin()+originalSize*(-1+2*j+1));
			for (unsigned k = originalSize*(-1+2*j+0); k < originalSize*(-1+2*j+1); ++k)
			{
				localPath[k][dim]=j;
			}
			for (unsigned k = originalSize*(-1+2*j+1); k < originalSize*(-1+2*j+2); ++k)
			{
				localPath[k][dim]=-j;
			}
		}
	}

	unsigned center=0;
	for (int dim = int(kernel._dims.size())-1; dim>=0; --dim)
	{
		center=center*kernel._dims[dim]+kernel._dims[dim]/2;
	}

	std::vector<DsPathEntry> entries;
	entries.reserve(localPath.size());
	for (size_t i = 0; i < localPath.size(); ++i)
	{
		unsigned flat=0;
		if(!kernel.indexWithDelta(flat,center,localPath[i])){
			continue;
		}
		if(std::isnan(kernel._data[flat*kernel._nbVariable])){
			continue;
		}
		float priority=0.f;
		if(searchDistance==g2s::KERNEL){
			for (unsigned variable = 0; variable < kernel._nbVariable; ++variable)
			{
				const float value=kernel._data[flat*kernel._nbVariable+variable];
				if(std::isfinite(value)){
					priority=std::max(priority,std::fabs(value));
				}
			}
		}else{
			float distance2=0.f;
			for (size_t dim = 0; dim < localPath[i].size(); ++dim)
			{
				distance2+=float(localPath[i][dim]*localPath[i][dim]);
			}
			priority=-distance2;
		}
		entries.push_back({localPath[i],static_cast<int>(flat),priority});
	}

	std::sort(entries.begin(),entries.end(),[](const DsPathEntry& lhs, const DsPathEntry& rhs){
		return lhs.priority>rhs.priority;
	});
	offsets.clear();
	flatIndices.clear();
	offsets.reserve(entries.size());
	flatIndices.reserve(entries.size());
	for (size_t i = 0; i < entries.size(); ++i)
	{
		offsets.push_back(entries[i].offset);
		flatIndices.push_back(entries[i].flatIndex);
	}
}

int main(int argc, char const *argv[]) {
	std::multimap<std::string, std::string> arg=g2s::argumentReader(argc,argv);

	char logFileName[2048]={0};
	std::vector<std::string> sourceFileNameVector;
	std::vector<std::string> kernelFileName;
	std::string targetFileName;
	std::string simulationPathFileName;
	std::string idImagePathFileName;
	std::string numberOfNeighboursFileName;
	std::string kernelIndexImageFileName;
	std::string kValueImageFileName;
	std::string rotationMapFileName;
	std::string rotationToleranceMapFileName;
	std::string scaleMapFileName;
	std::string scaleToleranceMapFileName;
	std::string outputFilename;
	std::string outputIndexFilename;

	jobIdType uniqueID=-1;
	bool run=true;

	FILE* reportFile=nullptr;
	if(arg.count("-r")>1){
		fprintf(stderr,"only one report file is possible\n");
		run=false;
	}else if(arg.count("-r")==1){
		strcpy(logFileName,(arg.find("-r")->second).c_str());
		reportFile=g2s::reporting::openReportFile((arg.find("-r")->second).c_str(),uniqueID);
		if(reportFile==nullptr){
			fprintf(stderr,"Impossible to open the report file\n");
			run=false;
		}
	}
	arg.erase("-r");
	if(reportFile==nullptr){
		reportFile=stderr;
	}

	if(arg.count("-id")==1){
		const long parsedId=atol((arg.find("-id")->second).c_str());
		if(parsedId>=0){
			uniqueID=static_cast<jobIdType>(parsedId);
		}
	}
	arg.erase("-id");
	g2s::reporting::markStarted(reportFile,"ds");

	if(arg.count("-h")==1 || arg.count("--help")==1){
		printHelp();
		return 0;
	}
	arg.erase("-h");
	arg.erase("--help");

	for (int i = 0; i < argc; ++i)
	{
		fprintf(reportFile,"%s ",argv[i]);
	}
	fprintf(reportFile,"\n");

	unsigned nbThreads=1;
	unsigned totalNumberOfThreadAvailable=1;
	#if _OPENMP
	totalNumberOfThreadAvailable=omp_get_max_threads();
	#endif
	if(arg.count("-j")>=1){
		auto jobsString=arg.lower_bound("-j");
		if(jobsString!=arg.upper_bound("-j")){
			float nbThreadsLoc=atof((jobsString->second).c_str());
			if(std::roundf(nbThreadsLoc)!=nbThreadsLoc){
				nbThreadsLoc=std::max(std::floor(nbThreadsLoc*totalNumberOfThreadAvailable),1.f);
			}
			nbThreads=static_cast<unsigned>(nbThreadsLoc);
		}
	}
	arg.erase("-j");
	if(arg.count("--jobs")>=1){
		auto jobsString=arg.lower_bound("--jobs");
		if(jobsString!=arg.upper_bound("--jobs")){
			float nbThreadsLoc=atof((jobsString->second).c_str());
			if(std::roundf(nbThreadsLoc)!=nbThreadsLoc){
				nbThreadsLoc=std::max(std::floor(nbThreadsLoc*totalNumberOfThreadAvailable),1.f);
			}
			nbThreads=static_cast<unsigned>(nbThreadsLoc);
		}
	}
	arg.erase("--jobs");
	if(nbThreads<1){
		nbThreads=1;
	}

	if(arg.count("-ti")>0){
		for(auto it=arg.equal_range("-ti").first; it!=arg.equal_range("-ti").second; ++it){
			sourceFileNameVector.push_back(it->second);
		}
	}else{
		fprintf(reportFile,"error: missing -ti\n");
		run=false;
	}
	arg.erase("-ti");

	if(arg.count("-di")==1){
		targetFileName=arg.find("-di")->second;
	}else{
		fprintf(reportFile,"error: missing -di\n");
		run=false;
	}
	arg.erase("-di");

	if(arg.count("-ki")>0){
		for(auto it=arg.equal_range("-ki").first; it!=arg.equal_range("-ki").second; ++it){
			kernelFileName.push_back(it->second);
		}
	}
	arg.erase("-ki");

	if(arg.count("-sp")==1) simulationPathFileName=arg.find("-sp")->second;
	arg.erase("-sp");
	if(arg.count("-ii")==1) idImagePathFileName=arg.find("-ii")->second;
	arg.erase("-ii");
	if(arg.count("-ni")==1) numberOfNeighboursFileName=arg.find("-ni")->second;
	arg.erase("-ni");
	if(arg.count("-kii")==1) kernelIndexImageFileName=arg.find("-kii")->second;
	arg.erase("-kii");
	if(arg.count("-kvi")==1) kValueImageFileName=arg.find("-kvi")->second;
	arg.erase("-kvi");
	if(arg.count("-rmi")==1) rotationMapFileName=arg.find("-rmi")->second;
	else if(arg.count("-rmi")>1){ fprintf(reportFile,"error: only one -rmi is possible\n"); run=false; }
	arg.erase("-rmi");
	if(arg.count("-rti")==1) rotationToleranceMapFileName=arg.find("-rti")->second;
	else if(arg.count("-rti")>1){ fprintf(reportFile,"error: only one -rti is possible\n"); run=false; }
	arg.erase("-rti");
	if(arg.count("-smi")==1) scaleMapFileName=arg.find("-smi")->second;
	else if(arg.count("-smi")>1){ fprintf(reportFile,"error: only one -smi is possible\n"); run=false; }
	arg.erase("-smi");
	if(arg.count("-sti")==1) scaleToleranceMapFileName=arg.find("-sti")->second;
	else if(arg.count("-sti")>1){ fprintf(reportFile,"error: only one -sti is possible\n"); run=false; }
	arg.erase("-sti");

	if(arg.count("-o")==1){
		outputFilename=arg.find("-o")->second;
	}else{
		outputFilename=std::to_string(uniqueID);
	}
	arg.erase("-o");
	if(arg.count("-oi")==1){
		outputIndexFilename=arg.find("-oi")->second;
	}else{
		outputIndexFilename=std::string("id_")+std::to_string(uniqueID);
	}
	arg.erase("-oi");

	std::vector<unsigned> nbNeighbors;
	float threshold=std::nanf("0");
	float mer=std::nanf("0");
	unsigned seed=std::chrono::high_resolution_clock::now().time_since_epoch().count();
	bool requestFullSimulation=false;
	bool considerTiAsCircular=false;
	bool circularSimulation=false;
	bool forceSimulation=false;
	bool withPathOptim=false;
	g2s::DistanceType searchDistance=g2s::EUCLIDIEN;
	std::vector<float> continuousNormInput;

	if(arg.count("-th")==1) threshold=atof((arg.find("-th")->second).c_str());
	arg.erase("-th");
	if(arg.count("-f")==1) mer=atof((arg.find("-f")->second).c_str());
	arg.erase("-f");
	if(arg.count("-mer")==1) mer=atof((arg.find("-mer")->second).c_str());
	arg.erase("-mer");
	if(arg.count("-n")>=1){
		for(auto it=arg.lower_bound("-n"); it!=arg.upper_bound("-n"); ++it){
			nbNeighbors.push_back(static_cast<unsigned>(std::max(0,atoi(it->second.c_str()))));
		}
	}
	arg.erase("-n");
	if(arg.count("-s")==1) seed=static_cast<unsigned>(atoi((arg.find("-s")->second).c_str()));
	arg.erase("-s");
	if(arg.count("-fs")==1) requestFullSimulation=true;
	arg.erase("-fs");
	if(arg.count("-cti")==1) considerTiAsCircular=true;
	arg.erase("-cti");
	if(arg.count("-csim")==1) circularSimulation=true;
	arg.erase("-csim");
	if(arg.count("--forceSimulation")==1) forceSimulation=true;
	arg.erase("--forceSimulation");
	if(arg.count("-wPO")==1) withPathOptim=true;
	arg.erase("-wPO");
	if(arg.count("-wd")==1) searchDistance=g2s::KERNEL;
	arg.erase("-wd");
	if(arg.count("-ed")==1) searchDistance=g2s::EUCLIDIEN;
	arg.erase("-ed");
	if(arg.count("-md")==1) searchDistance=g2s::MANAHTTAN;
	arg.erase("-md");
	if(arg.count("-cn")>=1){
		for(auto it=arg.lower_bound("-cn"); it!=arg.upper_bound("-cn"); ++it){
			run&=parsePositiveFloatList(it->second,"-cn",reportFile,continuousNormInput);
		}
	}
	arg.erase("-cn");
	if(arg.count("-cnorm")>=1){
		for(auto it=arg.lower_bound("-cnorm"); it!=arg.upper_bound("-cnorm"); ++it){
			run&=parsePositiveFloatList(it->second,"-cnorm",reportFile,continuousNormInput);
		}
	}
	arg.erase("-cnorm");
	if(arg.count("-W_GPU")>0){
		g2s::reporting::recordWarning(reportFile,"native ds is CPU-only; ignoring -W_GPU");
	}
	arg.erase("-W_GPU");
	if(arg.count("-W_CUDA")>0){
		g2s::reporting::recordWarning(reportFile,"native ds is CPU-only; ignoring -W_CUDA");
	}
	arg.erase("-W_CUDA");

	if(nbNeighbors.empty() && numberOfNeighboursFileName.empty()){
		fprintf(reportFile,"error: missing -n or -ni\n");
		run=false;
	}
	if(!std::isfinite(threshold) || threshold<0.f){
		fprintf(reportFile,"error: -th expects a finite non-negative threshold\n");
		run=false;
	}
	if(!std::isfinite(mer) || mer<=0.f){
		fprintf(reportFile,"error: -f/-mer expects a strictly positive finite exploration ratio\n");
		run=false;
	}

	if(!run){
		fprintf(reportFile,"simulation interrupted !!\n");
		return 0;
	}

	std::mt19937 randomGenerator(seed);

	std::vector<g2s::DataImage> TIs;
	for (size_t i = 0; i < sourceFileNameVector.size(); ++i)
	{
		TIs.push_back(g2s::DataImage::createFromFile(sourceFileNameVector[i]));
		g2s::reporting::logInput(reportFile,"-ti["+std::to_string(i)+"]",sourceFileNameVector[i],TIs.back());
	}
	g2s::DataImage DI=g2s::DataImage::createFromFile(targetFileName);
	g2s::reporting::logInput(reportFile,"-di",targetFileName,DI);
	if(DI.isEmpty() || TIs.empty() || TIs[0].isEmpty()){
		fprintf(reportFile,"error: failed to load -ti or -di\n");
		return 0;
	}
	std::vector<unsigned char> protectedMask=protectedSamples(DI,forceSimulation);

	if(arg.count("-dt")>=1){
		std::vector<g2s::DataImage::VariableType> types;
		for(auto it=arg.lower_bound("-dt"); it!=arg.upper_bound("-dt"); ++it){
			std::string normalized=it->second;
			for (size_t i = 0; i < normalized.size(); ++i)
			{
				if(normalized[i]==',' || normalized[i]==';') normalized[i]=' ';
			}
			std::stringstream stream(normalized);
			float value=0.f;
			while(stream>>value){
				types.push_back(value==0.f ? g2s::DataImage::Continuous : g2s::DataImage::Categorical);
			}
		}
		if(types.size()==DI._nbVariable){
			DI._types=types;
			for (size_t i = 0; i < TIs.size(); ++i)
			{
				TIs[i]._types=types;
			}
		}else{
			fprintf(reportFile,"error: -dt must provide one value per variable\n");
			return 0;
		}
	}
	arg.erase("-dt");

	for(auto it=arg.begin(); it!=arg.end(); ++it){
		fprintf(reportFile,"%s %s <== ignored !\n",it->first.c_str(),it->second.c_str());
	}

	for (size_t i = 0; i < TIs.size(); ++i)
	{
		if(TIs[i]._nbVariable!=DI._nbVariable || TIs[i]._types!=DI._types){
			fprintf(reportFile,"error: TI(s) and DI variable counts/types are incompatible\n");
			return 0;
		}
	}

	std::vector<unsigned> continuousVariableIndexes;
	for (unsigned variable = 0; variable < DI._nbVariable; ++variable)
	{
		if(DI._types[variable]==g2s::DataImage::Continuous){
			continuousVariableIndexes.push_back(variable);
		}
	}
	std::vector<float> continuousNormPowerByVariable(DI._nbVariable,2.f);
	if(!continuousNormInput.empty()){
		if(continuousNormInput.size()==1){
			for (size_t i = 0; i < continuousVariableIndexes.size(); ++i)
			{
				continuousNormPowerByVariable[continuousVariableIndexes[i]]=continuousNormInput[0];
			}
		}else if(continuousNormInput.size()==continuousVariableIndexes.size()){
			for (size_t i = 0; i < continuousVariableIndexes.size(); ++i)
			{
				continuousNormPowerByVariable[continuousVariableIndexes[i]]=continuousNormInput[i];
			}
		}else{
			fprintf(reportFile,"error: -cn/-cnorm value count must be one or match the number of continuous variables\n");
			return 0;
		}
	}

	std::vector<g2s::DataImage> kernels;
	for (size_t i = 0; i < kernelFileName.size(); ++i)
	{
		kernels.push_back(g2s::DataImage::createFromFile(kernelFileName[i]));
		if(kernels.back()._dims.size()-1==TIs[0]._dims.size()){
			kernels.back().convertFirstDimInVariable();
		}
		g2s::reporting::logInput(reportFile,"-ki["+std::to_string(i)+"]",kernelFileName[i],kernels.back());
	}
	if(kernels.empty()){
		std::vector<unsigned> maxSize=TIs[0]._dims;
		for (size_t ti = 0; ti < TIs.size(); ++ti)
		{
			for (size_t dim = 0; dim < maxSize.size(); ++dim)
			{
				maxSize[dim]=std::min(TIs[ti]._dims[dim]/2+1,maxSize[dim]);
			}
		}
		std::vector<float> variableWeight(DI._nbVariable,1.f);
		std::vector<float> alphas(DI._nbVariable,0.f);
		std::vector<g2s::KernelType> kernelTypes(DI._nbVariable,g2s::UNIFORM);
		kernels.push_back(g2s::DataImage::genearteKernel(kernelTypes,maxSize,variableWeight,alphas));
	}
	for (size_t i = 0; i < kernels.size(); ++i)
	{
		if(kernels[i]._dims.size()!=DI._dims.size()){
			fprintf(reportFile,"error: kernel rank must match DI rank\n");
			return 0;
		}
		if(!(kernels[i]._nbVariable==1 || kernels[i]._nbVariable==DI._nbVariable)){
			fprintf(reportFile,"error: kernel variable count must be 1 or match DI variables\n");
			return 0;
		}
	}

	std::vector<std::vector<std::vector<int> > > pathPositionArray;
	std::vector<std::vector<int> > kernelFlatIndexArray;
	for (size_t kernelIndex = 0; kernelIndex < kernels.size(); ++kernelIndex)
	{
		std::vector<std::vector<int> > offsets;
		std::vector<int> flatIndices;
		buildPathForKernel(kernels[kernelIndex],searchDistance,offsets,flatIndices);
		pathPositionArray.push_back(offsets);
		kernelFlatIndexArray.push_back(flatIndices);
	}

	g2s_path_index_t simulationPathSize=0;
	g2s_path_index_t* simulationPathIndex=nullptr;
	g2s_path_index_t beginPath=0;
	bool fullSimulation=false;
	g2s::DataImage simulationPath;
	if(simulationPathFileName.empty()){
		simulationPathSize=requestFullSimulation ? DI.dataSize() : DI.dataSize()/DI._nbVariable;
		fullSimulation=requestFullSimulation;
		simulationPathIndex=(g2s_path_index_t*)malloc(sizeof(g2s_path_index_t)*simulationPathSize);
		for (g2s_path_index_t i = 0; i < simulationPathSize; ++i) simulationPathIndex[i]=i;
		if(fullSimulation){
			for (g2s_path_index_t i = 0; i < simulationPathSize; ++i)
			{
				if(!std::isnan(DI._data[i]) && !forceSimulation){
					std::swap(simulationPathIndex[beginPath],simulationPathIndex[i]);
					beginPath++;
				}
			}
		}else{
			for (g2s_path_index_t i = 0; i < simulationPathSize; ++i)
			{
				bool valueSet=true;
				for (unsigned variable = 0; variable < DI._nbVariable; ++variable)
				{
					valueSet&=!std::isnan(DI._data[i*DI._nbVariable+variable]);
				}
				if(valueSet && !forceSimulation){
					std::swap(simulationPathIndex[beginPath],simulationPathIndex[i]);
					beginPath++;
				}
			}
		}
		std::shuffle(simulationPathIndex+beginPath,simulationPathIndex+simulationPathSize,randomGenerator);
	}else{
		simulationPath=g2s::DataImage::createFromFile(simulationPathFileName);
		g2s::reporting::logInput(reportFile,"-sp",simulationPathFileName,simulationPath);
		simulationPathSize=simulationPath.dataSize();
		fullSimulation=false;
		bool dimAgree=true;
		if(simulationPath._dims.size()!=DI._dims.size()){
			if(simulationPath._dims.size()-1==DI._dims.size()){
				simulationPath.convertFirstDimInVariable();
				fullSimulation=true;
			}else{
				dimAgree=false;
			}
		}
		for (size_t i = 0; i < simulationPath._dims.size(); ++i)
		{
			if(simulationPath._dims[i]!=DI._dims[i]) dimAgree=false;
		}
		if(!dimAgree){
			fprintf(reportFile,"dimension between simulation path and destination grid disagree\n");
			return 0;
		}
		simulationPathIndex=(g2s_path_index_t*)malloc(sizeof(g2s_path_index_t)*simulationPathSize);
		std::iota(simulationPathIndex,simulationPathIndex+simulationPathSize,0);
		float* simulationPathData=simulationPath._data;
		std::sort(simulationPathIndex,simulationPathIndex+simulationPathSize,[simulationPathData](g2s_path_index_t a, g2s_path_index_t b){
			return simulationPathData[a]<simulationPathData[b];
		});
		for (beginPath=0; beginPath<simulationPathSize; ++beginPath)
		{
			const float value=simulationPathData[simulationPathIndex[beginPath]];
			if((!std::isinf(value)) || (value>0)) break;
		}
	}

	g2s::DataImage id=DI.emptyCopy(!fullSimulation);
	id.setEncoding(g2s::DataImage::UInteger);
	memset(id._data,0,sizeof(unsigned)*id.dataSize());
	unsigned* importDataIndex=(unsigned*)id._data;

	float* seedForIndex=(float*)malloc(sizeof(float)*simulationPathSize);
	std::uniform_real_distribution<float> uniformDistribution(0.f,1.f);
	for (g2s_path_index_t i = 0; i < simulationPathSize; ++i)
	{
		seedForIndex[i]=uniformDistribution(randomGenerator);
		if(seedForIndex[i]==1.f) seedForIndex[i]=uniformDistribution(randomGenerator);
	}

	g2s::DataImage idImage;
	g2s::DataImage numberOfNeighboursImage;
	g2s::DataImage kernelIndexImage;
	g2s::DataImage kValueImage;
	if(!idImagePathFileName.empty()){
		idImage=g2s::DataImage::createFromFile(idImagePathFileName);
		g2s::reporting::logInput(reportFile,"-ii",idImagePathFileName,idImage);
	}
	if(!numberOfNeighboursFileName.empty()){
		numberOfNeighboursImage=g2s::DataImage::createFromFile(numberOfNeighboursFileName);
		g2s::reporting::logInput(reportFile,"-ni",numberOfNeighboursFileName,numberOfNeighboursImage);
	}
	if(!kernelIndexImageFileName.empty()){
		kernelIndexImage=g2s::DataImage::createFromFile(kernelIndexImageFileName);
		g2s::reporting::logInput(reportFile,"-kii",kernelIndexImageFileName,kernelIndexImage);
	}
	if(!kValueImageFileName.empty()){
		kValueImage=g2s::DataImage::createFromFile(kValueImageFileName);
		g2s::reporting::logInput(reportFile,"-kvi",kValueImageFileName,kValueImage);
	}

	g2s::DataImage rotationMapImage;
	g2s::DataImage rotationToleranceMapImage;
	g2s::DataImage scaleMapImage;
	g2s::DataImage scaleToleranceMapImage;
	const std::vector<unsigned> outputDims=DI._dims;
	if(!rotationMapFileName.empty()){
		rotationMapImage=g2s::DataImage::createFromFile(rotationMapFileName);
		normalizeLeadingVariableDim(rotationMapImage,outputDims);
		g2s::reporting::logInput(reportFile,"-rmi",rotationMapFileName,rotationMapImage);
	}
	if(!rotationToleranceMapFileName.empty()){
		rotationToleranceMapImage=g2s::DataImage::createFromFile(rotationToleranceMapFileName);
		normalizeLeadingVariableDim(rotationToleranceMapImage,outputDims);
		g2s::reporting::logInput(reportFile,"-rti",rotationToleranceMapFileName,rotationToleranceMapImage);
	}
	if(!scaleMapFileName.empty()){
		scaleMapImage=g2s::DataImage::createFromFile(scaleMapFileName);
		normalizeLeadingVariableDim(scaleMapImage,outputDims);
		g2s::reporting::logInput(reportFile,"-smi",scaleMapFileName,scaleMapImage);
	}
	if(!scaleToleranceMapFileName.empty()){
		scaleToleranceMapImage=g2s::DataImage::createFromFile(scaleToleranceMapFileName);
		normalizeLeadingVariableDim(scaleToleranceMapImage,outputDims);
		g2s::reporting::logInput(reportFile,"-sti",scaleToleranceMapFileName,scaleToleranceMapImage);
	}

	const bool useRotationMap=!rotationMapImage.isEmpty();
	const bool useRotationToleranceMap=!rotationToleranceMapImage.isEmpty();
	const bool useScaleMap=!scaleMapImage.isEmpty();
	const bool useScaleToleranceMap=!scaleToleranceMapImage.isEmpty();
	const bool useTransformMap=useRotationMap || useScaleMap;
	const size_t simulationRank=DI._dims.size();
	if((useRotationMap || useRotationToleranceMap || useScaleMap || useScaleToleranceMap) && simulationRank!=2 && simulationRank!=3){
		fprintf(reportFile,"transform map error: DS transforms support only 2D and 3D simulations\n");
		return 0;
	}
	auto validateMapDims=[&](const g2s::DataImage& image, const char* name)->bool{
		if(const_cast<g2s::DataImage&>(image).isEmpty()) return true;
		if(image._dims!=outputDims){
			fprintf(reportFile,"transform map error: %s dimensions must match -di dimensions\n",name);
			return false;
		}
		return true;
	};
	if(!validateMapDims(rotationMapImage,"-rmi") || !validateMapDims(rotationToleranceMapImage,"-rti") ||
		!validateMapDims(scaleMapImage,"-smi") || !validateMapDims(scaleToleranceMapImage,"-sti")){
		return 0;
	}
	if((useRotationMap || useRotationToleranceMap) && simulationRank==2){
		if((useRotationMap && rotationMapImage._nbVariable!=1) || (useRotationToleranceMap && rotationToleranceMapImage._nbVariable!=1)){
			fprintf(reportFile,"transform map error: 2D -rmi/-rti require exactly 1 channel\n");
			return 0;
		}
	}
	if((useRotationMap || useRotationToleranceMap) && simulationRank==3){
		if((useRotationMap && rotationMapImage._nbVariable!=4) || (useRotationToleranceMap && rotationToleranceMapImage._nbVariable!=4)){
			fprintf(reportFile,"transform map error: 3D -rmi/-rti require exactly 4 quaternion channels (qx,qy,qz,qw)\n");
			return 0;
		}
	}
	if(useScaleMap && (scaleMapImage._nbVariable!=1 || !validateFiniteMap(scaleMapImage,true,false))){
		fprintf(reportFile,"transform map error: -smi requires one strictly positive finite channel\n");
		return 0;
	}
	if(useScaleToleranceMap && (scaleToleranceMapImage._nbVariable!=1 || !validateFiniteMap(scaleToleranceMapImage,false,true))){
		fprintf(reportFile,"transform map error: -sti requires one finite non-negative channel\n");
		return 0;
	}
	if(useRotationToleranceMap && !validateFiniteMap(rotationToleranceMapImage,false,true)){
		fprintf(reportFile,"transform map error: -rti values must be finite and non-negative\n");
		return 0;
	}

	g2s::reporting::logParameter(reportFile,"threads",std::to_string(nbThreads));
	g2s::reporting::logParameter(reportFile,"seed",std::to_string(seed));
	g2s::reporting::logParameter(reportFile,"threshold",std::to_string(threshold));
	g2s::reporting::logParameter(reportFile,"max_exploration_ratio",std::to_string(mer));
	g2s::reporting::logParameter(reportFile,"simulation_mode",fullSimulation ? "full" : "vector");
	g2s::reporting::logParameter(reportFile,"circular_ti",g2s::reporting::boolString(considerTiAsCircular));
	g2s::reporting::logParameter(reportFile,"circular_simulation",g2s::reporting::boolString(circularSimulation));
	g2s::reporting::logParameter(reportFile,"force_simulation",g2s::reporting::boolString(forceSimulation));
	g2s::reporting::logParameter(reportFile,"path_optimization",g2s::reporting::boolString(withPathOptim));
	g2s::reporting::logParameter(reportFile,"rotation_map",g2s::reporting::boolString(useRotationMap));
	g2s::reporting::logParameter(reportFile,"rotation_tolerance_map",g2s::reporting::boolString(useRotationToleranceMap));
	g2s::reporting::logParameter(reportFile,"scale_map",g2s::reporting::boolString(useScaleMap));
	g2s::reporting::logParameter(reportFile,"scale_tolerance_map",g2s::reporting::boolString(useScaleToleranceMap));
	g2s::reporting::logParameter(reportFile,"simulation_path_size",std::to_string((unsigned long long)simulationPathSize));
	g2s::reporting::logParameter(reportFile,"output_image",outputFilename);
	g2s::reporting::logParameter(reportFile,"output_index",outputIndexFilename);

	DirectSamplingModule DSM(&TIs,&kernels[0],DI._types,continuousNormPowerByVariable,threshold,mer,considerTiAsCircular);

	std::vector<std::vector<float> > categoriesValues;
	qs_transform_utils::TransformContext transformContext;
	transformContext.rotationMap=useRotationMap ? &rotationMapImage : nullptr;
	transformContext.rotationToleranceMap=useRotationToleranceMap ? &rotationToleranceMapImage : nullptr;
	transformContext.scaleMap=useScaleMap ? &scaleMapImage : nullptr;
	transformContext.scaleToleranceMap=useScaleToleranceMap ? &scaleToleranceMapImage : nullptr;
	transformContext.rank=DI._dims.size();
	transformContext.globalSeed=seed;
	const qs_transform_utils::TransformContext* transformContextPtr=useTransformMap ? &transformContext : nullptr;

	std::vector<g2s_path_index_t> posteriorPath;
	const g2s_path_index_t maxPosteriorValue=std::numeric_limits<g2s_path_index_t>::max();
	const g2s_path_index_t numberOfPointToSimulate=simulationPathSize-beginPath;
	if(fullSimulation){
		posteriorPath.assign(DI.dataSize(),maxPosteriorValue);
		for (unsigned i = 0; i < DI.dataSize(); ++i)
		{
			if(!std::isnan(DI._data[i]) && !forceSimulation) posteriorPath[i]=0;
		}
	}else{
		const unsigned cellCount=DI.dataSize()/DI._nbVariable;
		posteriorPath.assign(cellCount,maxPosteriorValue);
		for (unsigned i = 0; i < cellCount; ++i)
		{
			bool withNan=false;
			for (unsigned variable = 0; variable < DI._nbVariable; ++variable)
			{
				withNan|=std::isnan(DI._data[i*DI._nbVariable+variable]);
			}
			if(!withNan && !forceSimulation) posteriorPath[i]=0;
		}
	}
	for (g2s_path_index_t i = 0; i < numberOfPointToSimulate; ++i)
	{
		posteriorPath[simulationPathIndex[beginPath+i]]=i;
	}

	auto begin=std::chrono::high_resolution_clock::now();
	if(fullSimulation){
		fprintf(reportFile,"full sim\n");
		simulationFull(reportFile,DI,TIs,kernels,DSM,pathPositionArray,simulationPathIndex+beginPath,numberOfPointToSimulate,
			(!idImage.isEmpty() ? &idImage : nullptr),(!kernelIndexImage.isEmpty() ? &kernelIndexImage : nullptr),
			seedForIndex,importDataIndex,nbNeighbors,(!numberOfNeighboursImage.isEmpty() ? &numberOfNeighboursImage : nullptr),
			(!kValueImage.isEmpty() ? &kValueImage : nullptr),categoriesValues,nbThreads,false,circularSimulation,forceSimulation,
			posteriorPath.data(),nullptr,nullptr,transformContextPtr,&kernelFlatIndexArray,seed);
	}else{
		fprintf(reportFile,"vector sim\n");
		simulation(reportFile,DI,TIs,kernels,DSM,pathPositionArray,simulationPathIndex+beginPath,numberOfPointToSimulate,
			(!idImage.isEmpty() ? &idImage : nullptr),(!kernelIndexImage.isEmpty() ? &kernelIndexImage : nullptr),
			seedForIndex,importDataIndex,nbNeighbors,(!numberOfNeighboursImage.isEmpty() ? &numberOfNeighboursImage : nullptr),
			(!kValueImage.isEmpty() ? &kValueImage : nullptr),categoriesValues,nbThreads,false,circularSimulation,forceSimulation,
			false,withPathOptim,posteriorPath.data(),nullptr,nullptr,transformContextPtr,&kernelFlatIndexArray,seed);
	}
	auto end=std::chrono::high_resolution_clock::now();
	cleanCategoricalSingletons(DI,protectedMask);
	const double time=1.0e-6*std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
	fprintf(reportFile,"compuattion time: %7.2f s\n",time/1000);
	fprintf(reportFile,"compuattion time: %.0f ms\n",time);
	g2s::reporting::recordMetric(reportFile,"duration_ms",std::to_string((long long)time));
	g2s::reporting::markFinished(reportFile,time);

	id.write(outputIndexFilename);
	DI.write(outputFilename);
	id.write(std::string("im_2_")+std::to_string(uniqueID));
	DI.write(std::string("im_1_")+std::to_string(uniqueID));
	g2s::reporting::logOutput(reportFile,"index_image",outputIndexFilename,id);
	g2s::reporting::logOutput(reportFile,"simulation_image",outputFilename,DI);
	g2s::reporting::logOutput(reportFile,"index_image_runtime",std::string("im_2_")+std::to_string(uniqueID),id);
	g2s::reporting::logOutput(reportFile,"simulation_image_runtime",std::string("im_1_")+std::to_string(uniqueID),DI);

	free(seedForIndex);
	free(simulationPathIndex);
	return 0;
}
