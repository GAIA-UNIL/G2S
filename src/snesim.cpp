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

#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>

#include <thread>
#include <atomic>

#include "utils.hpp"
#include "DataImage.hpp"
#include "jobManager.hpp"

#include "snesim.hpp"


enum simType
{
	vectorSim,
	fullSim,
	augmentedDimSim
};


void printHelp(){
	printf ("that is the help");
}

int main(int argc, char const *argv[]) {

	std::multimap<std::string, std::string> arg=g2s::argumentReader(argc,argv);

	char logFileName[2048]={0};
	std::vector<std::string> sourceFileNameVector;
	std::string targetFileName;
	std::vector<std::string> kernelFileName;
	std::string simuationPathFileName;
	std::string idImagePathFileName;

	std::string numberOfNeigboursFileName;
	std::string kernelIndexImageFileName;
	std::string kValueImageFileName;

	std::string outputFilename;
	std::string outputIndexFilename;


	jobIdType uniqueID=-1;
	bool run=true;


	// manage report file
	FILE *reportFile=NULL;
	if (arg.count("-r") > 1)
	{
		fprintf(reportFile,"only one rapport file is possible\n");
		run=false;
	}else{
		if(arg.count("-r") ==1){
			if(!strcmp((arg.find("-r")->second).c_str(),"stderr")){
				reportFile=stderr;
			}
			if(!strcmp((arg.find("-r")->second).c_str(),"stdout")){
				reportFile=stdout;
			}
			if (reportFile==NULL) {
				strcpy(logFileName,(arg.find("-r")->second).c_str());
				reportFile=fopen((arg.find("-r")->second).c_str(),"a");
				setvbuf ( reportFile , nullptr , _IOLBF , 0 ); // maybe  _IONBF


				jobIdType logId;
				if(sscanf(logFileName,"logs/%u.log",&logId)==1){
					std::to_string(logId);
					uniqueID=logId;
				}
			}
			if (reportFile==NULL){
				fprintf(stderr,"Impossible to open the rapport file\n");
				run=false;
			}
		}
	}
	arg.erase("-r");
	for (int i = 0; i < argc; ++i)
	{
		fprintf(reportFile,"%s ",argv[i]);
	}
	fprintf(reportFile,"\n");

	// LOOK FOR STANDARD PARAMETER

	unsigned nbThreads=1;
	unsigned nbThreadsOverTi=1;
	unsigned nbThreadsLastLevel=1;
	unsigned totalNumberOfThreadVailable=1;
	bool verbose=false;


	#if _OPENMP
		totalNumberOfThreadVailable=omp_get_max_threads();
	#endif	

	if (arg.count("-j") >= 1)
	{
		std::multimap<std::string, std::string>::iterator jobsString=arg.lower_bound("-j");

		if(jobsString!=arg.upper_bound("-j")){
			float nbThreadsLoc=atof((jobsString->second).c_str());
			if(std::roundf(nbThreadsLoc) != nbThreadsLoc){
				nbThreadsLoc=std::max(std::floor(nbThreadsLoc*totalNumberOfThreadVailable),1.f);
			}
			nbThreads=(int)(nbThreadsLoc);
			++jobsString;
		}
		if(jobsString!=arg.upper_bound("-j")){
			float nbThreadsOverTiLoc=atof((jobsString->second).c_str());
			if(std::roundf(nbThreadsOverTiLoc) != nbThreadsOverTiLoc){
				nbThreadsOverTiLoc=std::max(std::floor(nbThreadsOverTiLoc*totalNumberOfThreadVailable),1.f);
			}
			nbThreadsOverTi=(int)(nbThreadsOverTiLoc);
			++jobsString;
		}
		if(jobsString!=arg.upper_bound("-j")){
			float nbThreadsLastLevelLoc=atof((jobsString->second).c_str());
			if(std::roundf(nbThreadsLastLevelLoc) != nbThreadsLastLevelLoc){
				nbThreadsLastLevelLoc=std::max(std::floor(nbThreadsLastLevelLoc*totalNumberOfThreadVailable),1.f);
			}
			nbThreadsLastLevel=(int)(nbThreadsLastLevelLoc);
			++jobsString;
		}
	}
	arg.erase("-j");

	if (arg.count("--jobs") >= 1)
	{
		std::multimap<std::string, std::string>::iterator jobsString=arg.lower_bound("--jobs");
		if(jobsString!=arg.upper_bound("--jobs")){
			float nbThreadsLoc=atof((jobsString->second).c_str());
			if(std::roundf(nbThreadsLoc) != nbThreadsLoc){
				nbThreadsLoc=std::max(std::floor(nbThreadsLoc*totalNumberOfThreadVailable),1.f);
			}
			nbThreads=(int)(nbThreadsLoc);
			++jobsString;
		}
		if(jobsString!=arg.upper_bound("--jobs")){
			float nbThreadsOverTiLoc=atof((jobsString->second).c_str());
			if(std::roundf(nbThreadsOverTiLoc) != nbThreadsOverTiLoc){
				nbThreadsOverTiLoc=std::max(std::floor(nbThreadsOverTiLoc*totalNumberOfThreadVailable),1.f);
			}
			nbThreadsOverTi=(int)(nbThreadsOverTiLoc);
			++jobsString;
		}
		if(jobsString!=arg.upper_bound("--jobs")){
			float nbThreadsLastLevelLoc=atof((jobsString->second).c_str());
			if(std::roundf(nbThreadsLastLevelLoc) != nbThreadsLastLevelLoc){
				nbThreadsLastLevelLoc=std::max(std::floor(nbThreadsLastLevelLoc*totalNumberOfThreadVailable),1.f);
			}
			nbThreadsLastLevel=(int)(nbThreadsLastLevelLoc);
			++jobsString;
		}
	}
	arg.erase("--jobs");

	if(nbThreads<1)
	#if _OPENMP
		nbThreads=totalNumberOfThreadVailable;
	#else
		nbThreads=1;
	#endif

	if ((arg.count("-h") == 1)|| (arg.count("--help") == 1))
	{
		printHelp();
		return 0;
	}
	arg.erase("-h");

	if ((arg.count("-v") == 1) || (arg.count("--verbose") == 1))
	{
		verbose=true;
	}
	arg.erase("--verbose");






	// LOOK FOR DATA FILES
	//look for training images
	if (arg.count("-ti") > 0)
	{
		std::multimap<std::string, std::string>::iterator it;
	    for (it=arg.equal_range("-ti").first; it!=arg.equal_range("-ti").second; ++it)
	    {
	    	sourceFileNameVector.push_back(it->second);
	    }
	}else{	
		fprintf(reportFile,"error source\n");
		run=false;
	}
	arg.erase("-ti");

	//look for destination images (hard data)
	if (arg.count("-di") ==1)
	{
		targetFileName=arg.find("-di")->second;
	}else{	
		fprintf(reportFile,"error target\n");
		run=false;
	}
	arg.erase("-di");

	//look for -sp			: simulation path 
	if (arg.count("-sp") ==1)
	{
		simuationPathFileName=arg.find("-sp")->second;
	}else{	
		fprintf(reportFile,"non critical error : no simulation path\n");
	}
	arg.erase("-sp");

	bool useUniqueTI4Sampling=false;
	//look for -ii			: image of training index
	if (arg.count("-ii") ==1)
	{
		idImagePathFileName=arg.find("-ii")->second;
		useUniqueTI4Sampling=true;
	}
	arg.erase("-ii");

	//look for -ni			: image of neigbours
	if (arg.count("-ni") ==1)
	{
		numberOfNeigboursFileName=arg.find("-ni")->second;
	}
	arg.erase("-ni");


	// LOOK FOR OUTPUT
	if (arg.count("-o") ==1)
	{
		outputFilename=arg.find("-o")->second;
		run=false;
	}else{
		outputFilename=std::to_string(uniqueID);
		outputIndexFilename=std::string("id_")+std::to_string(uniqueID);
	}
	arg.erase("-o");

	if (arg.count("-oi") ==1)
	{
		outputIndexFilename=arg.find("-oi")->second;
	}
	arg.erase("-oi");

	// LOOK FOR SETINGS
	
	std::vector<unsigned> nbNeighbors;						// number of nighbors QS, DS ...
	unsigned seed=std::chrono::high_resolution_clock::now().time_since_epoch().count();
	bool requestFullSimulation=false;
	bool conciderTiAsCircular=false;
	bool circularSimulation=false;
	unsigned nbLevel=3;
	bool extendTree=false;

	if (arg.count("-n") >= 1)
	{
		for (auto val=arg.lower_bound("-n"); val!=arg.upper_bound("-n"); val++){
			nbNeighbors.push_back(atoi((val->second).c_str()));
		}
	}
	arg.erase("-n");

	if (arg.count("-s") == 1)
	{
		seed=atoi((arg.find("-s")->second).c_str());
	}
	arg.erase("-s");

	if (arg.count("-level") == 1)
	{
		nbLevel=atoi((arg.find("-level")->second).c_str());
	}
	arg.erase("-level");

	if (arg.count("-ET") == 1)
	{
		extendTree=atoi((arg.find("-ET")->second).c_str());
	}
	arg.erase("-ET");

	if (arg.count("-cti") == 1)
	{
		conciderTiAsCircular=true;
	}
	arg.erase("-cti");

	if (arg.count("-csim") == 1)
	{
		circularSimulation=true;
	}
	arg.erase("-csim");

	// if (arg.count("-adsim") == 1)
	// {
	// 	augmentedDimentionSimulation=true;
	// }
	// arg.erase("-adsim");


	// precheck | check what is mandatory

	if(nbNeighbors.size()<=0 && numberOfNeigboursFileName.empty()){
		run=false;
		fprintf(reportFile, "%s\n", "number of neighbor parameter not valide" );
	}

	// print all ignored parameters
	for (std::multimap<std::string, std::string>::iterator it=arg.begin(); it!=arg.end(); ++it){
		fprintf(reportFile, "%s %s <== ignored !\n", it->first.c_str(), it->second.c_str());
	}

	if(!run){
		fprintf(reportFile, "simulation interupted !!\n");
		return 0;
	}


	std::mt19937 randomGenerator(seed);

	std::vector<g2s::DataImage > TIs;

	for (size_t i = 0; i < sourceFileNameVector.size(); ++i)
	{
		TIs.push_back(g2s::DataImage::createFromFile(sourceFileNameVector[i]));
	}

	g2s::DataImage DI=g2s::DataImage::createFromFile(targetFileName);

	// if(DI._dims.size()<=TIs[0]._dims.size()) // auto desactivate of the dimention augmentation, if the dimention is not good
	// 	augmentedDimentionSimulation=false;
	
	g2s::DataImage simulationPath;
	g2s::DataImage idImage;
	g2s::DataImage numberOfNeigboursImage;


	unsigned simulationPathSize=0;
	unsigned* simulationPathIndex=nullptr;
	unsigned beginPath=0;
	bool fullSimulation=false;

	if(simuationPathFileName.empty()) {  //todo, need to be redsign to handle augmentedDimentionSimulation
		//fprintf(stderr, "generate simulation path\n");
		if (requestFullSimulation)
		{
			simulationPathSize=DI.dataSize();
			fullSimulation=true;
		}else{
			simulationPathSize=DI.dataSize()/DI._nbVariable;
			fullSimulation=false;
		}
		simulationPathIndex=(unsigned *)malloc(sizeof(unsigned)*simulationPathSize);
		for (unsigned i = 0; i < simulationPathSize; ++i)
		{
			simulationPathIndex[i]=i;
		}

		if (fullSimulation)
		{
			for (unsigned int i = 0; i < simulationPathSize; ++i)
			{
				if(!std::isnan(DI._data[i])){
					std::swap(simulationPathIndex[beginPath],simulationPathIndex[i]);
					beginPath++;
				}
			}

		}else{
			for (unsigned int i = 0; i < simulationPathSize; ++i)
			{
				bool valueSeted=true;
				for (unsigned int j = 0; j < DI._nbVariable; ++j)
				{
					if(std::isnan(DI._data[i*DI._nbVariable+j]))valueSeted=false;
				}
				if(valueSeted)
				{
					std::swap(simulationPathIndex[beginPath],simulationPathIndex[i]);
					beginPath++;
				}
			}
		}
		std::shuffle(simulationPathIndex+beginPath, simulationPathIndex + simulationPathSize, randomGenerator );
	}
	else {
		simulationPath=g2s::DataImage::createFromFile(simuationPathFileName);
		simulationPathSize=simulationPath.dataSize();
		bool dimAgree=true;
		fullSimulation=false;
		if(simulationPath._dims.size()!=DI._dims.size()){
			if(simulationPath._dims.size()-1==DI._dims.size()){
				simulationPath.convertFirstDimInVariable();
				fullSimulation=true;
			}else dimAgree=false;
		}
		for (size_t i = 0; i < simulationPath._dims.size(); ++i)
		{
			if(simulationPath._dims[i]!=DI._dims[i])dimAgree=false;
		}
		if(!dimAgree){
			fprintf(reportFile, "%s\n", "dimension between simulation path and destination grid disagree");
			return 0;
		}

		simulationPathIndex=(unsigned *)malloc(sizeof(unsigned)*simulationPathSize);
		std::iota(simulationPathIndex,simulationPathIndex+simulationPathSize,0);
		float* simulationPathData=simulationPath._data;
		std::sort(simulationPathIndex, simulationPathIndex+simulationPathSize,
			[simulationPathData](unsigned i1, unsigned i2) {return simulationPathData[i1] < simulationPathData[i2];});

		//Search begin path
		for ( beginPath=0 ; beginPath < simulationPathSize; ++beginPath)
		{
			float value=simulationPathData[simulationPathIndex[beginPath]];
			if((!std::isinf(value))||(value>0)) break;
		}

	}

	float* seedForIndex=( float* )malloc( sizeof(float) * simulationPathSize );
	std::uniform_real_distribution<float> uniformDitributionOverSource(0.f,1.f);

	for ( unsigned int i = 0; i < simulationPathSize; ++i)
	{
		seedForIndex[i]=uniformDitributionOverSource(randomGenerator);
		if(seedForIndex[i]==1.f)seedForIndex[i]=uniformDitributionOverSource(randomGenerator);
	}


	unsigned maxN=0;
	if (!numberOfNeigboursFileName.empty())
	{
		numberOfNeigboursImage=g2s::DataImage::createFromFile(numberOfNeigboursFileName);
		for (int i = 0; i < numberOfNeigboursImage.dataSize(); ++i)
		{
			if(maxN<numberOfNeigboursImage._data[i])
				maxN=numberOfNeigboursImage._data[i];
		}
	}else{
		for (int i = 0; i < nbNeighbors.size(); ++i)
		{
			if(maxN<nbNeighbors[i])
				maxN=nbNeighbors[i];
		}
	}	

	

	std::vector<std::vector<std::vector<int> > > pathPositionArray;
	for (int level = nbLevel-1; level>=0; level--)
	{
		unsigned factor=1<<level;
		std::vector<std::vector<int> > loaclPathPosition;
		loaclPathPosition.push_back(std::vector<int>(0));
		for (size_t i = 0; i < TIs[0]._dims.size(); ++i)
		{
			unsigned originalSize=loaclPathPosition.size();
			int sizeInThisDim=4; /// I can imagine any one want 
			loaclPathPosition.resize(originalSize*(2*sizeInThisDim-1));
			for (unsigned int k = 0; k < originalSize; ++k)
			{
				loaclPathPosition[k].push_back(0);
			}
			for (int j = 1; j < sizeInThisDim; ++j)
			{
				std::copy ( loaclPathPosition.begin(), loaclPathPosition.begin()+originalSize, loaclPathPosition.begin()+originalSize*(-1+2*j+0) );
				std::copy ( loaclPathPosition.begin(), loaclPathPosition.begin()+originalSize, loaclPathPosition.begin()+originalSize*(-1+2*j+1) );
				for (unsigned int k = originalSize*(-1+2*j+0); k < originalSize*(-1+2*j+1); ++k)
				{
					loaclPathPosition[k][i]=j*factor;
				}
				for (unsigned int k = originalSize*(-1+2*j+1); k < originalSize*(-1+2*j+2); ++k)
				{
					loaclPathPosition[k][i]=-j*factor;
				}
			}
		}

		std::sort(loaclPathPosition.begin(),loaclPathPosition.end(),[](std::vector<int> &a, std::vector<int> &b){
			unsigned int da=0;
			unsigned int db=0;
			for (int i = 0; i < 2; ++i)
			{
				da+=(a[i]*a[i]);
				db+=(b[i]*b[i]);
			}
			return da<db;
		});
		loaclPathPosition.resize(maxN);
		pathPositionArray.push_back(loaclPathPosition);
	}
	std::vector<snesimTreeElement<2>> trees;
	if(!loadTree(trees,pathPositionArray,sourceFileNameVector,extendTree)){
		trees=createTree<2>(TIs, pathPositionArray, nbThreads, extendTree);
		saveTree(trees,pathPositionArray,sourceFileNameVector,extendTree);
	}
	

	auto begin = std::chrono::high_resolution_clock::now();
	snesimSimulation<2>(reportFile,DI, trees, extendTree, pathPositionArray, simulationPath, simulationPathIndex+beginPath, simulationPathSize-beginPath, seedForIndex,
	 nbNeighbors, maxN, !numberOfNeigboursImage.isEmpty() ? &numberOfNeigboursImage : nullptr , nbThreads, circularSimulation);
	auto end = std::chrono::high_resolution_clock::now();
	double time = 1.0e-6 * std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	fprintf(reportFile,"compuattion time: %7.2f s\n", time/1000);
	fprintf(reportFile,"compuattion time: %.0f ms\n", time);


	// // new filename 
	DI.write(std::string("im_1_")+std::to_string(uniqueID));

	free(simulationPathIndex);
	simulationPathIndex=nullptr;

	return 0;
}