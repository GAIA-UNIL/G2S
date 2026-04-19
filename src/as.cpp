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
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <thread>

#include "utils.hpp"
#include "DataImage.hpp"
#include "jobManager.hpp"
#include "anchorSamplingModule.hpp"
#include "anchorSimulation.hpp"
#include "qsPaddingUtils.hpp"

enum simType
{
	vectorSim,
	fullSim
};

void printHelp(){
	printf("Anchor Sampling (AS)\n");
}

int main(int argc, char const *argv[]) {
	std::multimap<std::string, std::string> arg=g2s::argumentReader(argc,argv);

	char logFileName[2048]={0};
	std::vector<std::string> sourceFileNameVector;
	std::string targetFileName;
	std::vector<std::string> kernelFileName;
	std::string simulationPathFileName;
	std::string maskImageFileName;
	std::string numberOfNeighboursFileName;
	std::string kernelIndexImageFileName;
	std::string kValueImageFileName;
	std::string outputFilename;
	std::string outputIndexFilename;

	jobIdType uniqueID=-1;
	bool run=true;

	FILE *reportFile=nullptr;
	if (arg.count("-r") > 1)
	{
		fprintf(stderr,"only one rapport file is possible\n");
		run=false;
	}else if(arg.count("-r")==1){
		if(!strcmp((arg.find("-r")->second).c_str(),"stderr")){
			reportFile=stderr;
		}
		if(!strcmp((arg.find("-r")->second).c_str(),"stdout")){
			reportFile=stdout;
		}
		if(reportFile==nullptr){
			strcpy(logFileName,(arg.find("-r")->second).c_str());
			reportFile=fopen((arg.find("-r")->second).c_str(),"a");
			if(reportFile){
				setvbuf(reportFile,nullptr,_IOLBF,0);
				jobIdType logId;
				if(sscanf(logFileName,"/tmp/G2S/logs/%u.log",&logId)==1){
					uniqueID=logId;
				}
			}
		}
		if(reportFile==nullptr){
			fprintf(stderr,"Impossible to open the rapport file\n");
			run=false;
		}
	}
	arg.erase("-r");
	if(reportFile==nullptr){
		reportFile=stderr;
	}

	if (arg.count("-id") == 1)
	{
		const long parsedId=atol((arg.find("-id")->second).c_str());
		if(parsedId>=0){
			uniqueID=static_cast<jobIdType>(parsedId);
		}
	}
	arg.erase("-id");

	for (int i = 0; i < argc; ++i)
	{
		fprintf(reportFile,"%s ",argv[i]);
	}
	fprintf(reportFile,"\n");

	unsigned nbThreads=1;
	unsigned totalNumberOfThreadVailable=1;
	#if _OPENMP
	totalNumberOfThreadVailable=omp_get_max_threads();
	#endif

	if (arg.count("-j") >= 1)
	{
		auto jobsString=arg.lower_bound("-j");
		if(jobsString!=arg.upper_bound("-j")){
			float nbThreadsLoc=atof((jobsString->second).c_str());
			if(std::roundf(nbThreadsLoc)!=nbThreadsLoc){
				nbThreadsLoc=std::max(std::floor(nbThreadsLoc*totalNumberOfThreadVailable),1.f);
			}
			nbThreads=static_cast<unsigned>(nbThreadsLoc);
		}
	}
	arg.erase("-j");

	if (arg.count("--jobs") >= 1)
	{
		auto jobsString=arg.lower_bound("--jobs");
		if(jobsString!=arg.upper_bound("--jobs")){
			float nbThreadsLoc=atof((jobsString->second).c_str());
			if(std::roundf(nbThreadsLoc)!=nbThreadsLoc){
				nbThreadsLoc=std::max(std::floor(nbThreadsLoc*totalNumberOfThreadVailable),1.f);
			}
			nbThreads=static_cast<unsigned>(nbThreadsLoc);
		}
	}
	arg.erase("--jobs");

	if(nbThreads<1){
		#if _OPENMP
		nbThreads=totalNumberOfThreadVailable;
		#else
		nbThreads=1;
		#endif
	}

	if ((arg.count("-h") == 1)|| (arg.count("--help") == 1))
	{
		printHelp();
		return 0;
	}
	arg.erase("-h");
	arg.erase("--help");

	// data files
	if (arg.count("-ti") > 0)
	{
		for (auto it=arg.equal_range("-ti").first; it!=arg.equal_range("-ti").second; ++it)
		{
			sourceFileNameVector.push_back(it->second);
		}
	}else{
		fprintf(reportFile,"error source\n");
		run=false;
	}
	arg.erase("-ti");

	if (arg.count("-di") == 1)
	{
		targetFileName=arg.find("-di")->second;
	}else{
		fprintf(reportFile,"error target\n");
		run=false;
	}
	arg.erase("-di");

	if (arg.count("-ki") > 0)
	{
		for (auto it=arg.equal_range("-ki").first; it!=arg.equal_range("-ki").second; ++it)
		{
			kernelFileName.push_back(it->second);
		}
	}else{
		fprintf(reportFile,"non critical error : no kernel \n");
	}
	arg.erase("-ki");

	if (arg.count("-sp") == 1)
	{
		simulationPathFileName=arg.find("-sp")->second;
	}else{
		fprintf(reportFile,"non critical error : no simulation path\n");
	}
	arg.erase("-sp");

	if (arg.count("-ii") == 1)
	{
		fprintf(reportFile,"error: -ii is not supported by Anchor Sampling, use -mi instead\n");
		run=false;
	}
	arg.erase("-ii");

	if (arg.count("-mi") == 1)
	{
		maskImageFileName=arg.find("-mi")->second;
	}
	arg.erase("-mi");

	if (arg.count("-ni") == 1)
	{
		numberOfNeighboursFileName=arg.find("-ni")->second;
	}
	arg.erase("-ni");

	if (arg.count("-kii") == 1)
	{
		kernelIndexImageFileName=arg.find("-kii")->second;
	}
	arg.erase("-kii");

	if (arg.count("-kvi") == 1)
	{
		kValueImageFileName=arg.find("-kvi")->second;
	}
	arg.erase("-kvi");

	// output
	if (arg.count("-o") == 1)
	{
		outputFilename=arg.find("-o")->second;
	}else{
		outputFilename=std::to_string(uniqueID);
		outputIndexFilename=std::string("id_")+std::to_string(uniqueID);
	}
	arg.erase("-o");

	if (arg.count("-oi") == 1)
	{
		outputIndexFilename=arg.find("-oi")->second;
	}
	arg.erase("-oi");

	unsigned interval=0;
	jobIdType previousID=0;
	if (arg.count("-as") > 0)
	{
		auto autoSaveParam=arg.lower_bound("-as");
		if(autoSaveParam!=arg.upper_bound("-as")){
			interval=atoi((autoSaveParam->second).c_str());
			++autoSaveParam;
		}
		if(autoSaveParam!=arg.upper_bound("-as")){
			previousID=atol((autoSaveParam->second).c_str());
		}
	}
	arg.erase("-as");

	// settings
	bool noVerbatim=false;
	bool fullStationary=false;
	std::vector<unsigned> nbNeighbors;
	float mer=std::nanf("0");
	float nbCandidate=std::nanf("0");
	unsigned seed=std::chrono::high_resolution_clock::now().time_since_epoch().count();
	g2s::DistanceType searchDistance=g2s::EUCLIDIEN;
	bool requestFullSimulation=false;
	bool considerTiAsCircular=false;
	bool circularSimulation=false;
	bool forceSimulation=false;
	bool maxNK=false;
	std::vector<float> continuousNormPInput;

	if (arg.count("-nV") == 1)
	{
		noVerbatim=true;
		fprintf(reportFile,"warning: -nV is currently ignored by Anchor Sampling\n");
	}
	arg.erase("-nV");

	if (arg.count("-fs") == 1)
	{
		requestFullSimulation=true;
	}
	arg.erase("-fs");

	if (arg.count("-f") == 1)
	{
		mer=atof((arg.find("-f")->second).c_str());
	}
	arg.erase("-f");
	if (arg.count("-mer") == 1)
	{
		mer=atof((arg.find("-mer")->second).c_str());
	}
	arg.erase("-mer");

	if (arg.count("-k") == 1)
	{
		nbCandidate=atof((arg.find("-k")->second).c_str());
	}
	arg.erase("-k");

	if (arg.count("-n") >= 1)
	{
		for (auto val=arg.lower_bound("-n"); val!=arg.upper_bound("-n"); ++val)
		{
			nbNeighbors.push_back(atoi((val->second).c_str()));
		}
	}
	arg.erase("-n");

	if (arg.count("-s") == 1)
	{
		seed=atoi((arg.find("-s")->second).c_str());
	}
	arg.erase("-s");

	if (arg.count("--forceSimulation") == 1)
	{
		forceSimulation=true;
	}
	arg.erase("--forceSimulation");

	if (arg.count("-wd") == 1)
	{
		searchDistance=g2s::KERNEL;
	}
	arg.erase("-wd");
	if (arg.count("-ed") == 1)
	{
		searchDistance=g2s::EUCLIDIEN;
	}
	arg.erase("-ed");
	if (arg.count("-md") == 1)
	{
		searchDistance=g2s::MANAHTTAN;
	}
	arg.erase("-md");

	if (arg.count("-fastAndRisky") == 1 || arg.count("-far") == 1)
	{
		fullStationary=true;
	}
	arg.erase("-fastAndRisky");
	arg.erase("-far");

	if (arg.count("-cti") == 1)
	{
		considerTiAsCircular=true;
	}
	arg.erase("-cti");

	if (arg.count("-csim") == 1)
	{
		circularSimulation=true;
	}
	arg.erase("-csim");

	if (arg.count("-adsim") == 1)
	{
		fprintf(reportFile,"error: -adsim is not supported by Anchor Sampling\n");
		run=false;
	}
	arg.erase("-adsim");

	if (arg.count("-maxNK") == 1)
	{
		maxNK=true;
	}
	arg.erase("-maxNK");

	auto appendContinuousNormValues=[&](const std::string &rawValue, const char *flagName){
		std::string normalized=rawValue;
		for (size_t i = 0; i < normalized.size(); ++i)
		{
			if(normalized[i]==',' || normalized[i]==';'){
				normalized[i]=' ';
			}
		}

		std::stringstream stream(normalized);
		std::string token;
		bool foundToken=false;
		while(stream>>token){
			foundToken=true;
			char *endPtr=nullptr;
			const float parsedValue=strtof(token.c_str(),&endPtr);
			if(endPtr==token.c_str() || *endPtr!='\0' || !std::isfinite(parsedValue) || parsedValue<=0.f){
				fprintf(reportFile,"error: %s expects strictly positive finite value(s), got '%s'\n",flagName,rawValue.c_str());
				run=false;
				return;
			}
			continuousNormPInput.push_back(parsedValue);
		}
		if(!foundToken){
			fprintf(reportFile,"error: %s expects at least one value\n",flagName);
			run=false;
		}
	};

	if (arg.count("-cn") >= 1)
	{
		for (auto val=arg.lower_bound("-cn"); val!=arg.upper_bound("-cn"); ++val)
		{
			appendContinuousNormValues(val->second,"-cn");
		}
	}
	arg.erase("-cn");

	if (arg.count("-cnorm") >= 1)
	{
		for (auto val=arg.lower_bound("-cnorm"); val!=arg.upper_bound("-cnorm"); ++val)
		{
			appendContinuousNormValues(val->second,"-cnorm");
		}
	}
	arg.erase("-cnorm");

	float alpha=0.f;
	g2s::KernelType kernelTypeForGeneration=g2s::UNIFORM;
	int kernelSize=-1;
	arg.erase("-kernel");
	if (arg.count("-ks") == 1)
	{
		kernelSize=atof((arg.find("-ks")->second).c_str());
	}
	arg.erase("-ks");
	if (arg.count("-alpha") == 1)
	{
		alpha=atof((arg.find("-alpha")->second).c_str());
	}
	arg.erase("-alpha");

	if (arg.count("-W_GPU") == 1)
	{
		fprintf(reportFile,"warning: Anchor Sampling currently runs on CPU, ignoring -W_GPU\n");
	}
	arg.erase("-W_GPU");

	if (arg.count("-W_CUDA") >= 1)
	{
		fprintf(reportFile,"warning: Anchor Sampling currently runs on CPU, ignoring -W_CUDA\n");
	}
	arg.erase("-W_CUDA");

	if(nbNeighbors.empty() && numberOfNeighboursFileName.empty()){
		fprintf(reportFile,"%s\n","number of neighbor parameter not valide");
		run=false;
	}
	if(std::isnan(mer) && std::isnan(nbCandidate) && kValueImageFileName.empty()){
		fprintf(reportFile,"%s\n","maximum exploration ratio or numer of candidate need to be seted");
		run=false;
	}
	if(std::isnan(nbCandidate)){
		nbCandidate=1/mer;
	}

	for (auto it=arg.begin(); it!=arg.end(); ++it)
	{
		fprintf(reportFile,"%s %s <== ignored !\n",it->first.c_str(),it->second.c_str());
	}

	if(!run){
		fprintf(reportFile,"simulation interupted !!\n");
		return 0;
	}

	std::mt19937 randomGenerator(seed);

	std::vector<g2s::DataImage> TIs;
	for (size_t i = 0; i < sourceFileNameVector.size(); ++i)
	{
		TIs.push_back(g2s::DataImage::createFromFile(sourceFileNameVector[i]));
	}
	g2s::DataImage DI=g2s::DataImage::createFromFile(targetFileName);

	if(TIs.empty() || DI.isEmpty()){
		fprintf(reportFile,"error: missing TI or DI data\n");
		return 0;
	}

	for (size_t i = 0; i < TIs.size(); ++i)
	{
		if(TIs[i]._dims!=DI._dims){
			fprintf(reportFile,"error: Anchor Sampling requires all TI grids to match the destination grid geometry exactly\n");
			return 0;
		}
		if(TIs[i]._types.size()!=DI._types.size()){
			fprintf(reportFile,"error: TI(s) not compatible with the DI\n");
			return 0;
		}
		for (size_t j = 0; j < TIs[i]._types.size(); ++j)
		{
			if(TIs[i]._types[j]!=DI._types[j]){
				fprintf(reportFile,"error: TI(s) not compatible with the DI\n");
				return 0;
			}
		}
	}

	std::vector<unsigned> continuousVariableIndexes;
	for (unsigned variable = 0; variable < DI._nbVariable; ++variable)
	{
		if(DI._types[variable]==g2s::DataImage::VaraibleType::Continuous){
			continuousVariableIndexes.push_back(variable);
		}
	}

	std::vector<float> continuousNormPowerByVariable(DI._nbVariable,2.f);
	if(continuousVariableIndexes.empty() && !continuousNormPInput.empty()){
		fprintf(reportFile,"warning: -cnorm/-cn provided but no continuous variable is defined in -dt, option ignored\n");
	}
	if(!continuousVariableIndexes.empty()){
		std::vector<float> resolvedContinuousNormP;
		if(continuousNormPInput.empty()){
			resolvedContinuousNormP.assign(continuousVariableIndexes.size(),2.f);
		}else if(continuousNormPInput.size()==1){
			resolvedContinuousNormP.assign(continuousVariableIndexes.size(),continuousNormPInput[0]);
		}else if(continuousNormPInput.size()==continuousVariableIndexes.size()){
			resolvedContinuousNormP=continuousNormPInput;
		}else{
			fprintf(reportFile,
				"error: continuous norm expects either 1 value or one value per continuous variable (%zu expected, got %zu)\n",
				continuousVariableIndexes.size(),continuousNormPInput.size());
			return 0;
		}

		for (size_t i = 0; i < continuousVariableIndexes.size(); ++i)
		{
			continuousNormPowerByVariable[continuousVariableIndexes[i]]=resolvedContinuousNormP[i];
		}
		fprintf(reportFile,"AS continuous norm powers:");
		for (size_t i = 0; i < continuousVariableIndexes.size(); ++i)
		{
			fprintf(reportFile," v%u->%g",continuousVariableIndexes[i],continuousNormPowerByVariable[continuousVariableIndexes[i]]);
		}
		fprintf(reportFile,"\n");
	}

	const bool needCrossMeasurement=std::any_of(TIs.begin(),TIs.end(),[](const g2s::DataImage &ti){
		unsigned localSize=ti._nbVariable;
		for (size_t dim = 0; dim < ti._dims.size(); ++dim)
		{
			localSize*=ti._dims[dim];
		}
		for (unsigned i = 0; i < localSize; ++i)
		{
			if(std::isnan(ti._data[i])){
				return true;
			}
		}
		return false;
	});

	g2s::DataImage simulationPath;
	g2s::DataImage id;
	g2s::DataImage numberOfNeighboursImage;
	g2s::DataImage kernelIndexImage;
	g2s::DataImage kValueImage;
	g2s::DataImage maskImage;

	std::vector<g2s::DataImage> kernels;
	for (size_t i = 0; i < kernelFileName.size(); ++i)
	{
		kernels.push_back(g2s::DataImage::createFromFile(kernelFileName[i]));
		if(kernels[i]._dims.size()-1==TIs[0]._dims.size()){
			kernels[i].convertFirstDimInVariable();
		}
	}
	if(kernels.empty()){
		std::vector<unsigned> maxSize=TIs[0]._dims;
		if(kernelSize!=-1){
			for (size_t i = 0; i < maxSize.size(); ++i)
			{
				maxSize[i]=kernelSize;
			}
		}
		std::vector<float> variableWeight(TIs[0]._nbVariable,1.f);
		std::vector<float> alphas(TIs[0]._nbVariable,alpha);
		std::vector<g2s::KernelType> kernelsTypeFG(TIs[0]._nbVariable,kernelTypeForGeneration);
		kernels.push_back(g2s::DataImage::genearteKernel(kernelsTypeFG,maxSize,variableWeight,alphas));
	}

	if(!numberOfNeighboursFileName.empty()){
		numberOfNeighboursImage=g2s::DataImage::createFromFile(numberOfNeighboursFileName);
	}
	if(!kernelIndexImageFileName.empty()){
		kernelIndexImage=g2s::DataImage::createFromFile(kernelIndexImageFileName);
	}
	if(!kValueImageFileName.empty()){
		kValueImage=g2s::DataImage::createFromFile(kValueImageFileName);
		if(std::isnan(nbCandidate)){
			nbCandidate=1.f;
		}
		for (unsigned i = 0; i < kValueImage.dataSize(); ++i)
		{
			nbCandidate=std::max(nbCandidate,kValueImage._data[i]);
		}
	}
	if(!maskImageFileName.empty()){
		maskImage=g2s::DataImage::createFromFile(maskImageFileName);
		if(maskImage._nbVariable==1 && maskImage._dims.size()==TIs[0]._dims.size()+1){
			if(maskImage._dims.back()==TIs.size()){
				maskImage=maskImage.convertLastDimInVariable();
			}else if(maskImage._dims.front()==TIs.size()){
				maskImage.convertFirstDimInVariable();
			}
		}
		if(maskImage._dims!=DI._dims || maskImage._nbVariable!=TIs.size()){
			fprintf(reportFile,"error: -mi mask must have the same spatial geometry as -di and one value per TI\n");
			return 0;
		}
	}

	std::vector<unsigned> outputDims=DI._dims;
	std::vector<unsigned> spatialPadding(outputDims.size(),0);
	for (size_t kernelIndex = 0; kernelIndex < kernels.size(); ++kernelIndex)
	{
		for (size_t dim = 0; dim < std::min(kernels[kernelIndex]._dims.size(),spatialPadding.size()); ++dim)
		{
			spatialPadding[dim]=std::max(spatialPadding[dim],kernels[kernelIndex]._dims[dim]/2);
		}
	}
	const bool usePaddedDomain=qs_padding_utils::hasPadding(spatialPadding);

	std::vector<std::vector<std::vector<int> > > pathPositionArray;
	for (size_t kernelIndex = 0; kernelIndex < kernels.size(); ++kernelIndex)
	{
		std::vector<std::vector<int> > localPathPosition;
		localPathPosition.push_back(std::vector<int>(0));
		for (size_t i = 0; i < kernels[kernelIndex]._dims.size(); ++i)
		{
			const unsigned originalSize=localPathPosition.size();
			const int sizeInThisDim=(kernels[kernelIndex]._dims[i]+1)/2;
			localPathPosition.resize(originalSize*(2*sizeInThisDim-1));
			for (unsigned k = 0; k < originalSize; ++k)
			{
				localPathPosition[k].push_back(0);
			}
			for (int j = 1; j < sizeInThisDim; ++j)
			{
				std::copy(localPathPosition.begin(),localPathPosition.begin()+originalSize,localPathPosition.begin()+originalSize*(-1+2*j+0));
				std::copy(localPathPosition.begin(),localPathPosition.begin()+originalSize,localPathPosition.begin()+originalSize*(-1+2*j+1));
				for (unsigned k = originalSize*(-1+2*j+0); k < originalSize*(-1+2*j+1); ++k)
				{
					localPathPosition[k][i]=j;
				}
				for (unsigned k = originalSize*(-1+2*j+1); k < originalSize*(-1+2*j+2); ++k)
				{
					localPathPosition[k][i]=-j;
				}
			}
		}

		g2s::DataImage weightKernel=kernels[kernelIndex].emptyCopy(true);
		if(searchDistance==g2s::EUCLIDIEN){
			for (unsigned i = 0; i < weightKernel.dataSize(); ++i)
			{
				weightKernel._data[i]=-weightKernel.distance2ToCenter(i);
			}
		}
		if(searchDistance==g2s::KERNEL){
			const unsigned nbV=kernels[kernelIndex]._nbVariable;
			for (unsigned i = 0; i < weightKernel.dataSize(); ++i)
			{
				for (unsigned j = 0; j < nbV; ++j)
				{
					if(std::fabs(kernels[kernelIndex]._data[i*nbV+j])>weightKernel._data[i]){
						weightKernel._data[i]=std::fabs(kernels[kernelIndex]._data[i*nbV+j]);
					}
				}
			}
		}

		unsigned center=0;
		g2s::DataImage* weightKernelPtr=weightKernel.ptr();
		for (int i = int(weightKernelPtr->_dims.size())-1; i >=0 ; --i)
		{
			center=center*weightKernelPtr->_dims[i]+(weightKernelPtr->_dims[i]-1)/2;
		}

		auto localKernel=&kernels[kernelIndex];
		auto removeIt=std::remove_if(localPathPosition.begin(),localPathPosition.end(),[center, localKernel](std::vector<int> &v){
			unsigned l1=0;
			unsigned nbV=localKernel->_nbVariable;
			localKernel->indexWithDelta(l1,center,v);
			return std::isnan(localKernel->_data[l1*nbV+0]);
		});
		localPathPosition.erase(removeIt,localPathPosition.end());

		std::sort(localPathPosition.begin(),localPathPosition.end(),[weightKernelPtr, center](std::vector<int> &a, std::vector<int> &b){
			unsigned l1=0,l2=0;
			weightKernelPtr->indexWithDelta(l1,center,a);
			weightKernelPtr->indexWithDelta(l2,center,b);
			return weightKernelPtr->_data[l1] > weightKernelPtr->_data[l2];
		});
		pathPositionArray.push_back(localPathPosition);
	}

	g2s_path_index_t simulationPathSize=0;
	g2s_path_index_t* simulationPathIndex=nullptr;
	g2s_path_index_t beginPath=0;
	bool fullSimulation=false;

	if(simulationPathFileName.empty()){
		if(requestFullSimulation){
			simulationPathSize=DI.dataSize();
			fullSimulation=true;
		}else{
			simulationPathSize=DI.dataSize()/DI._nbVariable;
			fullSimulation=false;
		}
		simulationPathIndex=(g2s_path_index_t *)malloc(sizeof(g2s_path_index_t)*simulationPathSize);
		for (g2s_path_index_t i = 0; i < simulationPathSize; ++i)
		{
			simulationPathIndex[i]=i;
		}

		if(fullSimulation){
			for (g2s_path_index_t i = 0; i < simulationPathSize; ++i)
			{
				if(!std::isnan(DI._data[i])){
					std::swap(simulationPathIndex[beginPath],simulationPathIndex[i]);
					beginPath++;
				}
			}
		}else{
			for (g2s_path_index_t i = 0; i < simulationPathSize; ++i)
			{
				bool valueSet=true;
				for (unsigned j = 0; j < DI._nbVariable; ++j)
				{
					if(std::isnan(DI._data[i*DI._nbVariable+j])){
						valueSet=false;
					}
				}
				if(valueSet){
					std::swap(simulationPathIndex[beginPath],simulationPathIndex[i]);
					beginPath++;
				}
			}
		}
		std::shuffle(simulationPathIndex+beginPath,simulationPathIndex+simulationPathSize,randomGenerator);
	}else{
		simulationPath=g2s::DataImage::createFromFile(simulationPathFileName);
		simulationPathSize=simulationPath.dataSize();
		bool dimAgree=true;
		fullSimulation=false;
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
			if(simulationPath._dims[i]!=DI._dims[i]){
				dimAgree=false;
			}
		}
		if(!dimAgree){
			fprintf(reportFile,"%s\n","dimension between simulation path and destination grid disagree");
			return 0;
		}

		simulationPathIndex=(g2s_path_index_t *)malloc(sizeof(g2s_path_index_t)*simulationPathSize);
		std::iota(simulationPathIndex,simulationPathIndex+simulationPathSize,0);
		float* simulationPathData=simulationPath._data;
		std::sort(simulationPathIndex,simulationPathIndex+simulationPathSize,[simulationPathData](g2s_path_index_t i1, g2s_path_index_t i2){
			return simulationPathData[i1] < simulationPathData[i2];
		});
		for (beginPath=0; beginPath < simulationPathSize; ++beginPath)
		{
			float value=simulationPathData[simulationPathIndex[beginPath]];
			if((!std::isinf(value)) || (value>0)){
				break;
			}
		}
	}

	if(previousID>0){
		id=g2s::DataImage::createFromFile(std::string("im_2_")+std::to_string(previousID)+std::string(".auto_bk"));
		if(id._dims.size()>=1){
			DI=g2s::DataImage::createFromFile(std::string("im_1_")+std::to_string(previousID)+std::string(".auto_bk"));
		}
	}
	if(id._dims.size()<1){
		id=DI.emptyCopy(!fullSimulation);
		id.setEncoding(g2s::DataImage::EncodingType::UInteger);
		memset(id._data,0,sizeof(unsigned)*simulationPathSize);
	}

	unsigned* importDataIndex=nullptr;
	float* seedForIndex=(float*)malloc(sizeof(float)*simulationPathSize);
	std::uniform_real_distribution<float> uniformDistributionOverSource(0.f,1.f);
	for (g2s_path_index_t i = 0; i < simulationPathSize; ++i)
	{
		seedForIndex[i]=uniformDistributionOverSource(randomGenerator);
		if(seedForIndex[i]==1.f){
			seedForIndex[i]=uniformDistributionOverSource(randomGenerator);
		}
	}

	std::vector<std::vector<float> > categoriesValues;
	std::vector<unsigned> numberOfComputedVariableProVariable;
	for (size_t i = 0; i < DI._types.size(); ++i)
	{
		if(DI._types[i]==g2s::DataImage::VaraibleType::Continuous){
			numberOfComputedVariableProVariable.push_back(1);
		}
		if(DI._types[i]==g2s::DataImage::VaraibleType::Categorical){
			std::vector<float> currentVariable;
			for (size_t im = 0; im < TIs.size(); ++im)
			{
				for (unsigned j = i; j < TIs[im].dataSize(); j+=TIs[im]._nbVariable)
				{
					if(std::isnan(TIs[im]._data[j])){
						continue;
					}
					bool isPresent=false;
					for (size_t k = 0; k < currentVariable.size(); ++k)
					{
						isPresent|=(TIs[im]._data[j]==currentVariable[k]);
					}
					if(!isPresent){
						currentVariable.push_back(TIs[im]._data[j]);
					}
				}
			}
			categoriesValues.push_back(currentVariable);
			numberOfComputedVariableProVariable.push_back(currentVariable.size());
		}
	}

	for (size_t i = 0; i < kernels.size(); ++i)
	{
		kernels[i]=g2s::DataImage::offsetKernel4categories(kernels[i],numberOfComputedVariableProVariable,needCrossMeasurement);
	}

	outputDims=DI._dims;
	if(usePaddedDomain){
		fprintf(reportFile,"use padded simulation domain with halos:");
		for (size_t dim = 0; dim < spatialPadding.size(); ++dim)
		{
			fprintf(reportFile," %u",spatialPadding[dim]);
		}
		fprintf(reportFile,"\n");

		DI=qs_padding_utils::padDataImageWithValue(DI,spatialPadding,std::nanf("0"));
		id=qs_padding_utils::padDataImageWithValue(id,spatialPadding,0.f);
		for (size_t i = 0; i < TIs.size(); ++i)
		{
			TIs[i]=qs_padding_utils::padDataImageWithValue(TIs[i],spatialPadding,std::nanf("0"));
		}
		if(!numberOfNeighboursImage.isEmpty()){
			numberOfNeighboursImage=qs_padding_utils::padDataImageWithValue(numberOfNeighboursImage,spatialPadding,std::nanf("0"));
		}
		if(!kernelIndexImage.isEmpty()){
			kernelIndexImage=qs_padding_utils::padDataImageWithValue(kernelIndexImage,spatialPadding,std::nanf("0"));
		}
		if(!kValueImage.isEmpty()){
			kValueImage=qs_padding_utils::padDataImageWithValue(kValueImage,spatialPadding,std::nanf("0"));
		}
		if(!maskImage.isEmpty()){
			maskImage=qs_padding_utils::padDataImageWithValue(maskImage,spatialPadding,std::nanf("0"));
		}
		qs_padding_utils::mapSimulationPathToPadded(simulationPathIndex,simulationPathSize,fullSimulation,DI._nbVariable,outputDims,spatialPadding);
	}

	importDataIndex=(unsigned*)id._data;

	AnchorSamplingData anchorStack=AnchorSamplingData::build(TIs,categoriesValues,(maskImage.isEmpty() ? nullptr : &maskImage));
	AnchorSamplingModule ASM(&anchorStack,&TIs[0],(kernels.size()==1 ? &kernels[0] : nullptr),continuousNormPowerByVariable,nbCandidate,!needCrossMeasurement,considerTiAsCircular);

	std::vector<g2s_path_index_t> posteriorPath;
	const g2s_path_index_t maxPosteriorValue=std::numeric_limits<g2s_path_index_t>::max();
	const g2s_path_index_t numberOfPointToSimulate=simulationPathSize-beginPath;
	if(fullSimulation){
		posteriorPath.assign(DI.dataSize(),maxPosteriorValue);
		for (unsigned i = 0; i < DI.dataSize(); ++i)
		{
			if(!std::isnan(DI._data[i])){
				posteriorPath[i]=0;
			}
		}
	}else{
		const unsigned cellCount=DI.dataSize()/DI._nbVariable;
		posteriorPath.assign(cellCount,maxPosteriorValue);
		for (unsigned i = 0; i < cellCount; ++i)
		{
			bool withNan=false;
			for (unsigned j = 0; j < DI._nbVariable; ++j)
			{
				withNan|=std::isnan(DI._data[i*DI._nbVariable+j]);
			}
			if(!withNan){
				posteriorPath[i]=0;
			}
		}
	}
	for (g2s_path_index_t i = 0; i < numberOfPointToSimulate; ++i)
	{
		posteriorPath[simulationPathIndex[beginPath+i]]=i;
	}

	auto autoSaveFunction=[](g2s::DataImage &id, g2s::DataImage &DI, std::atomic<bool> &computationIsDone, unsigned interval, jobIdType uniqueID,
		bool usePaddedDomain, std::vector<unsigned> spatialPadding, std::vector<unsigned> outputDims){
		unsigned last=0;
		while(!computationIsDone){
			if(last>=interval){
				if(usePaddedDomain){
					g2s::DataImage croppedId=qs_padding_utils::cropDataImageCenter(id,spatialPadding,outputDims);
					g2s::DataImage croppedDI=qs_padding_utils::cropDataImageCenter(DI,spatialPadding,outputDims);
					croppedId.write(std::string("im_2_")+std::to_string(uniqueID)+std::string(".auto_bk"));
					croppedDI.write(std::string("im_1_")+std::to_string(uniqueID)+std::string(".auto_bk"));
				}else{
					id.write(std::string("im_2_")+std::to_string(uniqueID)+std::string(".auto_bk"));
					DI.write(std::string("im_1_")+std::to_string(uniqueID)+std::string(".auto_bk"));
				}
				last=0;
			}
			last++;
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		}
	};

	std::thread saveThread;
	std::atomic<bool> computationIsDone(false);
	if(interval>0){
		saveThread=std::thread(autoSaveFunction,std::ref(id),std::ref(DI),std::ref(computationIsDone),interval,uniqueID,usePaddedDomain,spatialPadding,outputDims);
	}

	auto begin=std::chrono::high_resolution_clock::now();
	if(fullSimulation){
		fprintf(reportFile,"%s\n","full sim");
		simulationFullAS(reportFile,DI,TIs,kernels,ASM,pathPositionArray,simulationPathIndex+beginPath,simulationPathSize-beginPath,
			(!kernelIndexImage.isEmpty() ? &kernelIndexImage : nullptr),seedForIndex,importDataIndex,nbNeighbors,
			(!numberOfNeighboursImage.isEmpty() ? &numberOfNeighboursImage : nullptr),(!kValueImage.isEmpty() ? &kValueImage : nullptr),
			categoriesValues,outputDims,spatialPadding,usePaddedDomain,nbThreads,fullStationary,circularSimulation,forceSimulation,posteriorPath.data());
	}else{
		fprintf(reportFile,"%s\n","vector sim");
		simulationAS(reportFile,DI,TIs,kernels,ASM,pathPositionArray,simulationPathIndex+beginPath,simulationPathSize-beginPath,
			(!kernelIndexImage.isEmpty() ? &kernelIndexImage : nullptr),seedForIndex,importDataIndex,nbNeighbors,
			(!numberOfNeighboursImage.isEmpty() ? &numberOfNeighboursImage : nullptr),(!kValueImage.isEmpty() ? &kValueImage : nullptr),
			categoriesValues,outputDims,spatialPadding,usePaddedDomain,nbThreads,fullStationary,circularSimulation,forceSimulation,maxNK,posteriorPath.data());
	}
	auto end=std::chrono::high_resolution_clock::now();
	computationIsDone=true;
	double time=1.0e-6*std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
	fprintf(reportFile,"compuattion time: %7.2f s\n",time/1000);
	fprintf(reportFile,"compuattion time: %.0f ms\n",time);

	if(saveThread.joinable()){
		saveThread.join();
	}

	if(usePaddedDomain){
		g2s::DataImage croppedId=qs_padding_utils::cropDataImageCenter(id,spatialPadding,outputDims);
		g2s::DataImage croppedDI=qs_padding_utils::cropDataImageCenter(DI,spatialPadding,outputDims);
		croppedId.write(outputIndexFilename);
		croppedDI.write(outputFilename);
		croppedId.write(std::string("im_2_")+std::to_string(uniqueID));
		croppedDI.write(std::string("im_1_")+std::to_string(uniqueID));
	}else{
		id.write(outputIndexFilename);
		DI.write(outputFilename);
		id.write(std::string("im_2_")+std::to_string(uniqueID));
		DI.write(std::string("im_1_")+std::to_string(uniqueID));
	}

	free(seedForIndex);
	free(simulationPathIndex);
	return 0;
}
