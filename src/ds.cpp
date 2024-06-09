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
#include <chrono>
#include "utils.hpp"
#include "DataImage.hpp"
#include "jobManager.hpp"

#include "simulation.hpp"

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
				if(sscanf(logFileName,"/tmp/G2S/logs/%d.log",&logId)==1){
					std::to_string(logId);
					//sprintf(outputFilename,"%d",logId);
					//symlink(outputName, fullFilename);
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
	bool verbose=false;

	if (arg.count("-j") == 1)
	{
		nbThreads=atoi((arg.find("-j")->second).c_str());
	}
	arg.erase("-j");

	if (arg.count("--jobs") == 1)
	{
		nbThreads=atoi((arg.find("--jobs")->second).c_str());
	}
	arg.erase("--jobs");	

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

	//look for -ki			: kernel image 
	if (arg.count("-ki") > 0)
	{
		std::multimap<std::string, std::string>::iterator it;
		for (it=arg.equal_range("-ki").first; it!=arg.equal_range("-ki").second; ++it)
		{
			kernelFileName.push_back(it->second);
		}
	}else{	
		fprintf(reportFile,"non critical error : no kernel \n");
	}
	arg.erase("-ki");

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

	//look for -ni			: image of neigbours
	if (arg.count("-kii") ==1)
	{
		kernelIndexImageFileName=arg.find("-kii")->second;
	}
	arg.erase("-kii");

	//look for -ni			: image of neigbours
	if (arg.count("-kvi") ==1)
	{
		kValueImageFileName=arg.find("-kvi")->second;
	}
	arg.erase("-kvi");


	// LOOK FOR OUTPUT
	if (arg.count("-o") ==1)
	{
		outputFilename=arg.find("-o")->second;
		run=false;
	}
	arg.erase("-o");

	if (arg.count("-oi") ==1)
	{
		outputIndexFilename=arg.find("-oi")->second;
	}
	arg.erase("-oi");


	// autoSave
	unsigned interval=0;
	jobIdType previousID=0;
	if (arg.count("-as") >0)
	{
		std::multimap<std::string, std::string>::iterator autoSaveParam=arg.lower_bound("-as");
		if(autoSaveParam!=arg.upper_bound("-as")){
			interval=atoi((autoSaveParam->second).c_str());
			++autoSaveParam;
		}
		if(autoSaveParam!=arg.upper_bound("-as")){
			previousID=atol((autoSaveParam->second).c_str());
			++autoSaveParam;
		}
	}
	arg.erase("-as");



	// LOOK FOR SETINGS

	float threshold=std::nanf("0");			// threshold for DS ...
	std::vector<unsigned> nbNeighbors;						// number of nighbors QS, DS ...
	float mer=std::nanf("0");				// maximum exploration ratio, called f in ds
	float nbCandidate=std::nanf("0");		// 1/f for QS
	float narrowness=std::nanf("0");		// narrowness for NDS
	unsigned seed=std::chrono::high_resolution_clock::now().time_since_epoch().count();
	g2s::DistanceType searchDistance=g2s::EUCLIDIEN;
	bool augmentedDimentionSimulation=false;
	bool requestFullSimulation=false;

	if (arg.count("-th") == 1)
	{
		threshold=atof((arg.find("-th")->second).c_str());
	}
	arg.erase("-th");

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

	if (arg.count("-nw") == 1)
	{
		narrowness=atof((arg.find("-nw")->second).c_str());
	}
	arg.erase("-nw");

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


	//add extra paremetre here
	//add extra paremetre here
	float alpha=0;
	g2s::KernelType kernelTypeForGeneration=g2s::UNIFORM;
	int kernelSize=-1;
	if (arg.count("-kernel") == 1)
	{
		//TODO implement the selecteur 
		// UNIFORM,
		// TRIANGULAR,
		// EXPONENTIAL,
		// EPANECHNIKOV,
		// QUARTIC,
		// TRIWEIGHT,
		// TRICUBE,
		// GAUSSIAN,
		// COSINE,
		// LOGISTIC,
		// SIGMOID,
		// SILVERMAN
	}
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


	// precheck | check what is mandatory

	if(nbNeighbors.size()<=0 && numberOfNeigboursFileName.empty()){
		run=false;
		fprintf(reportFile, "%s\n", "number of neighbor parameter not valide" );
	}
	/*if(std::isnan(threshold)){
		run=false;
		fprintf(reportFile, "%s\n", "threshold need to be seted" );
	}*/

	if(std::isnan(mer) && std::isnan(nbCandidate) && kValueImageFileName.empty()){
		run=false;
		fprintf(reportFile, "%s\n", "maximum exploration ratio or numer of candidate need to be seted" );
	}
	if(std::isnan(nbCandidate)){
		nbCandidate=1/mer;
	}

	// print all ignored parameters
	for (std::multimap<std::string, std::string>::iterator it=arg.begin(); it!=arg.end(); ++it){
		fprintf(reportFile, "%s %s <== ignored !\n", it->first.c_str(), it->second.c_str());
	}

	if(!run){
		fprintf(reportFile, "simulation interupted !!\n");
		return 0;
	}


	// print all ignored parameters
	for (std::multimap<std::string, std::string>::iterator it=arg.begin(); it!=arg.end(); ++it){
		fprintf(reportFile, "%s %s <== ignored !\n", it->first.c_str(), it->second.c_str());
	}

	if(!run) return 0;

#if _OPENMP
	omp_set_max_active_levels(3);
	#ifdef WITH_MKL
	mkl_set_num_threads(nbThreadsLastLevel);
	mkl_set_dynamic(false);
	#endif
#endif

	std::mt19937 randomGenerator(seed);

	std::vector<g2s::DataImage > TIs;

	for (size_t i = 0; i < sourceFileNameVector.size(); ++i)
	{
		TIs.push_back(g2s::DataImage::createFromFile(sourceFileNameVector[i]));
	}

	g2s::DataImage DI=g2s::DataImage::createFromFile(targetFileName);

	if(DI._dims.size()<=TIs[0]._dims.size()) // auto desactivate of the dimention augmentation, if the dimention is not good
		augmentedDimentionSimulation=false;
	
	g2s::DataImage simulationPath;
	g2s::DataImage idImage;
	g2s::DataImage numberOfNeigboursImage;
	g2s::DataImage kernelIndexImage;
	g2s::DataImage kValueImage;


	std::vector<g2s::DataImage > kernels;

	for (size_t i = 0; i < kernelFileName.size(); ++i)
	{
		kernels.push_back(g2s::DataImage::createFromFile(kernelFileName[i]));
		if(kernels[i]._dims.size()-1==TIs[0]._dims.size()){
			kernels[i].convertFirstDimInVariable();
		}
	}

	if(kernels.empty()) {
		std::vector<unsigned> maxSize=TIs[0]._dims;
		if(kernelSize!=-1){
			for (size_t i = 0; i < maxSize.size(); ++i)
			{
				maxSize[i]=kernelSize;
			}
		}else{
			for (size_t j = 0; j < TIs.size(); ++j)
			{
				for (size_t i = 0; i < maxSize.size(); ++i)
				{
					maxSize[i]=std::min(TIs[j]._dims[i]/2+1,maxSize[i]);
				}
			}
		}
		std::vector<float> variableWeight(TIs[0]._nbVariable);
		for (size_t i = 0; i < variableWeight.size(); ++i)
		{
			variableWeight[i]=1;
		}
		std::vector<float> alphas(TIs[0]._nbVariable);
		for (size_t i = 0; i < alphas.size(); ++i)
		{
			alphas[i]=alpha;
		}
		std::vector<g2s::KernelType> kernelsTypeFG(TIs[0]._nbVariable);
		for (size_t i = 0; i < kernelsTypeFG.size(); ++i)
		{
			kernelsTypeFG[i]=kernelTypeForGeneration;
		}
		kernels.push_back(g2s::DataImage::genearteKernel(kernelsTypeFG, maxSize, variableWeight, alphas));
	}

	std::vector<std::vector<std::vector<int> > > pathPositionArray;
	for (int kernelIndex = 0; kernelIndex < kernels.size(); ++kernelIndex)
	{
		std::vector<std::vector<int> > loaclPathPosition;
		loaclPathPosition.push_back(std::vector<int>(0));
		for (size_t i = 0; i < kernels[kernelIndex]._dims.size(); ++i)
		{
			unsigned originalSize=loaclPathPosition.size();
			int sizeInThisDim=(kernels[kernelIndex]._dims[i]+1)/2;
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
					loaclPathPosition[k][i]=j;
				}
				for (unsigned int k = originalSize*(-1+2*j+1); k < originalSize*(-1+2*j+2); ++k)
				{
					loaclPathPosition[k][i]=-j;
				}
			}
		}

		g2s::DataImage wieghtKernel=kernels[kernelIndex].emptyCopy(true);
		if(searchDistance==g2s::EUCLIDIEN){
			for (unsigned int i = 0; i < wieghtKernel.dataSize(); ++i)
			{
				wieghtKernel._data[i]=-wieghtKernel.distance2ToCenter(i);
			}
		}
		if(searchDistance==g2s::KERNEL){
			unsigned nbV=kernels[kernelIndex]._nbVariable;
			for (unsigned int i = 0; i < wieghtKernel.dataSize(); ++i)
			{
				for (unsigned int j = 0; j < nbV; ++j)
				{
					if(fabs(kernels[kernelIndex]._data[i*nbV+j])>wieghtKernel._data[i])wieghtKernel._data[i]=fabs(kernels[kernelIndex]._data[i*nbV+j]);
				}
			}
		}

		unsigned center=0;
		g2s::DataImage* wieghtKernelPtr=wieghtKernel.ptr();
		for (int i =  wieghtKernelPtr->_dims.size()-1; i>=0 ; i--)
		{
			center=center*wieghtKernelPtr->_dims[i]+(wieghtKernelPtr->_dims[i]-1)/2;
		}

		// kernel NaN cleaning, remove all non from neighours to explore.
		auto localKernel=&kernels[kernelIndex];
		#if __cplusplus >= 202002L
		std::erase_if(loaclPathPosition, [center, localKernel](std::vector<int> &v) {
			unsigned l1;
			unsigned nbV=localKernel->_nbVariable;
			localKernel->indexWithDelta(l1, center, v);
			return std::isnan(localKernel->_data[l1*nbV+0]);
		});
		#else

		auto it = std::remove_if(loaclPathPosition.begin(), loaclPathPosition.end(), [center, localKernel](std::vector<int> &v) {
			unsigned l1;
			unsigned nbV=localKernel->_nbVariable;
			localKernel->indexWithDelta(l1, center, v);
			return std::isnan(localKernel->_data[l1*nbV+0]);
		});
		loaclPathPosition.erase(it, loaclPathPosition.end());

		#endif

		std::sort(loaclPathPosition.begin(),loaclPathPosition.end(),[wieghtKernelPtr, center](std::vector<int> &a, std::vector<int> &b){
			unsigned l1,l2;
			wieghtKernelPtr->indexWithDelta(l1, center, a);
			wieghtKernelPtr->indexWithDelta(l2, center, b);
			return wieghtKernelPtr->_data[l1] > wieghtKernelPtr->_data[l2];
		});
		pathPositionArray.push_back(loaclPathPosition);
	}

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

	g2s::DataImage id=g2s::DataImage::createFromFile(std::string("im_2_")+std::to_string(previousID)+std::string(".auto_bk"));

	if(id._dims.size()<1){
		id=DI.emptyCopy(!fullSimulation);
		id.setEncoding(g2s::DataImage::EncodingType::UInteger);
		memset(id._data,0,sizeof(unsigned)*simulationPathSize);
	}else{
		DI=g2s::DataImage::createFromFile(std::string("im_1_")+std::to_string(previousID)+std::string(".auto_bk"));
	}
	
	unsigned* importDataIndex=(unsigned *)id._data;
	float* seedForIndex=( float* )malloc( sizeof(float) * simulationPathSize );
	std::uniform_real_distribution<float> uniformDitributionOverSource(0.f,1.f);

	for ( unsigned int i = 0; i < simulationPathSize; ++i)
	{
		seedForIndex[i]=uniformDitributionOverSource(randomGenerator);
		if(seedForIndex[i]==1.f)seedForIndex[i]=uniformDitributionOverSource(randomGenerator);
	}


	// id Image

	if (!idImagePathFileName.empty())
	{
		idImage=g2s::DataImage::createFromFile(idImagePathFileName);
	}

	if (!numberOfNeigboursFileName.empty())
	{
		numberOfNeigboursImage=g2s::DataImage::createFromFile(numberOfNeigboursFileName);
	}

	if (!kernelIndexImageFileName.empty())
	{
		kernelIndexImage=g2s::DataImage::createFromFile(kernelIndexImageFileName);
	}

	if (!kValueImageFileName.empty())
	{
		kValueImage=g2s::DataImage::createFromFile(kValueImageFileName);

		if(std::isnan(nbCandidate))
			nbCandidate=1.f;

		for (int i = 0; i < kValueImage.dataSize(); ++i)
		{
			nbCandidate=std::max(nbCandidate,kValueImage._data[i]);
		}
	}

	//init DS

	//DirectSamplingModule DSM(computeDeviceModuleArray, threshold, (kernels.size()==1 ? &kernels[0]:nullptr),nbCandidate,convertionTypeVectorMainVector, convertionTypeVectorConstVector, convertionCoefVectorConstVector, noVerbatim, !needCrossMesurement, nbThreads, nbThreadsOverTi, nbThreadsLastLevel, useUniqueTI4Sampling);

	// run QS

	auto begin = std::chrono::high_resolution_clock::now();

	// simType st=vectorSim;
	// if(fullSimulation) st=fullSim;
	// if(augmentedDimentionSimulation) st=augmentedDimSim;

	auto autoSaveFunction=[](g2s::DataImage &id, g2s::DataImage &DI, std::atomic<bool>  &computationIsDone, unsigned interval, jobIdType uniqueID){
		unsigned last=0;
		while (!computationIsDone)
		{
			if(last>=interval){
				id.write(std::string("im_2_")+std::to_string(uniqueID)+std::string(".auto_bk"));
				DI.write(std::string("im_1_")+std::to_string(uniqueID)+std::string(".auto_bk"));
				last=0;
			}
			last++;
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		}
	};


	std::thread saveThread;
	std::atomic<bool> computationIsDone(false);
	if(interval>0){
		saveThread=std::thread(autoSaveFunction, std::ref(id), std::ref(DI), std::ref(computationIsDone), interval, uniqueID);
	}

	// switch (st){
	// case fullSim:
	// 	fprintf(reportFile, "%s\n", "full sim");
	// 	simulationFull(reportFile, DI, TIs, kernels, QSM, pathPositionArray, simulationPathIndex+beginPath, simulationPathSize-beginPath, (useUniqueTI4Sampling ? &idImage : nullptr ),
	// 		(!kernelIndexImage.isEmpty() ? &kernelIndexImage : nullptr ), seedForIndex, importDataIndex, nbNeighbors,(!numberOfNeigboursImage.isEmpty() ? &numberOfNeigboursImage : nullptr ), (!kValueImage.isEmpty() ? &kValueImage : nullptr ), categoriesValues, nbThreads, fullStationary, circularSimulation, forceSimulation);
	// 	break;
	// case vectorSim:
	// 	fprintf(reportFile, "%s\n", "vector sim");
	// 	simulation(reportFile, DI, TIs, kernels, QSM, pathPositionArray, simulationPathIndex+beginPath, simulationPathSize-beginPath, (useUniqueTI4Sampling ? &idImage : nullptr ),
	// 		(!kernelIndexImage.isEmpty() ? &kernelIndexImage : nullptr ), seedForIndex, importDataIndex, nbNeighbors, (!numberOfNeigboursImage.isEmpty() ? &numberOfNeigboursImage : nullptr ), (!kValueImage.isEmpty() ? &kValueImage : nullptr ), categoriesValues, nbThreads, fullStationary, circularSimulation, forceSimulation,maxNK);
	// 	break;
	// case augmentedDimSim:
	// 	fprintf(reportFile, "%s\n", "augmented dimention sim");
	// 	simulationAD(reportFile, DI, TIs, kernels, QSM, pathPositionArray, simulationPathIndex+beginPath, simulationPathSize-beginPath, (useUniqueTI4Sampling ? &idImage : nullptr ),
	// 		(!kernelIndexImage.isEmpty() ? &kernelIndexImage : nullptr ), seedForIndex, importDataIndex, nbNeighbors, (!numberOfNeigboursImage.isEmpty() ? &numberOfNeigboursImage : nullptr ), (!kValueImage.isEmpty() ? &kValueImage : nullptr ), categoriesValues, nbThreads, nbThreadsOverTi, fullStationary, circularSimulation, forceSimulation);
	// 	break;
	// }

	auto end = std::chrono::high_resolution_clock::now();
	computationIsDone=true;
	double time = 1.0e-6 * std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	fprintf(reportFile,"compuattion time: %7.2f s\n", time/1000);
	fprintf(reportFile,"compuattion time: %.0f ms\n", time);

	// free memory



	// to remove later
	id.write(outputIndexFilename);
	DI.write(outputFilename);
	//end to remove

	// new filename 
	id.write(std::string("im_2_")+std::to_string(uniqueID));
	DI.write(std::string("im_1_")+std::to_string(uniqueID));

	if(saveThread.joinable()){
		saveThread.join();
	}

	free(simulationPathIndex);
	simulationPathIndex=nullptr;

	return 0;
}