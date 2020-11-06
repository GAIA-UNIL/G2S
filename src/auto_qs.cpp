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

#include "sharedMemoryManager.hpp"
#include "CPUThreadDevice.hpp"
#ifdef WITH_OPENCL
#include "OpenCLGPUDevice.hpp"
#endif
#ifdef WITH_CUDA
	#include <cuda_runtime.h>
	#include "NvidiaGPUAcceleratorDevice.hpp"
#endif // WITH_CUDA


#include "CPUThreadAcceleratorDevice.hpp"

#include "calibration.hpp"
#include "quantileSamplingModule.hpp"

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
	std::vector<std::string> kernelFileName;
	std::string simuationPathFileName;
	std::string idImagePathFileName;

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

	bool fullSimulation=false;
	bool conciderTiAsCircular=false;
	unsigned maxNbCandidate=1;
	std::vector<unsigned> maxNbNeihbours;

	unsigned maxNumberOfIteration=25000;
	unsigned minNumberOfIteration=1000;
	float metricPower=2;

	float maxt=INFINITY;
	std::vector<float> densityArray;

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

	//look for -ki			: kernel image 

	if (arg.count("-maxk") == 1)
	{
		maxNbCandidate=atof((arg.find("-maxk")->second).c_str());
	}
	arg.erase("-maxk");
	if (arg.count("-maxK") == 1)
	{
		maxNbCandidate=atof((arg.find("-maxK")->second).c_str());
	}
	arg.erase("-maxK");

	if (arg.count("-maxn") >= 1)
	{
		for (auto val=arg.lower_bound("-maxn"); val!=arg.upper_bound("-maxn"); val++){
			maxNbNeihbours.push_back(atoi((val->second).c_str()));
		}
	}
	arg.erase("-maxn");
	if (arg.count("-maxN") >= 1)
	{
		for (auto val=arg.lower_bound("-maxN"); val!=arg.upper_bound("-maxN"); val++){
			maxNbNeihbours.push_back(atoi((val->second).c_str()));
		}
	}
	arg.erase("-maxN");

	if (arg.count("-maxIter") == 1)
	{
		maxNumberOfIteration=atoi((arg.find("-maxIter")->second).c_str());
	}
	arg.erase("-maxIter");

	if (arg.count("-maxiter") == 1)
	{
		maxNumberOfIteration=atoi((arg.find("-maxiter")->second).c_str());
	}
	arg.erase("-maxiter");

	if (arg.count("-minIter") == 1)
	{
		minNumberOfIteration=atoi((arg.find("-minIter")->second).c_str());
	}
	arg.erase("-minIter");
	
	if (arg.count("-minIter") == 1)
	{
		minNumberOfIteration=atoi((arg.find("-minIter")->second).c_str());
	}
	arg.erase("-minIter");

	if (arg.count("-maxT") == 1)
	{
		maxt=atof((arg.find("-maxT")->second).c_str());
	}
	arg.erase("-maxT");

	if (arg.count("-maxt") == 1)
	{
		maxt=atof((arg.find("-maxt")->second).c_str());
	}
	arg.erase("-maxt");

	if (arg.count("-mpow") == 1)
	{
		metricPower=atof((arg.find("-mpow")->second).c_str());
	}
	arg.erase("-mpow");

	if (arg.count("-densities") > 0)
	{
		std::multimap<std::string, std::string>::iterator it;
		for (it=arg.equal_range("-densities").first; it!=arg.equal_range("-densities").second; ++it)
		{
			densityArray.push_back(atof(it->second.c_str()));
		}
	}
	arg.erase("-densities");
	if (arg.count("-density") > 0)
	{
		std::multimap<std::string, std::string>::iterator it;
		for (it=arg.equal_range("-density").first; it!=arg.equal_range("-density").second; ++it)
		{
			densityArray.push_back(atof(it->second.c_str()));
		}
	}
	arg.erase("-density");


	if (arg.count("-ki") > 0)
	{
		std::multimap<std::string, std::string>::iterator it;
		for (it=arg.equal_range("-ki").first; it!=arg.equal_range("-ki").second; ++it)
		{
			kernelFileName.push_back(it->second);
		}
	}else{	
		fprintf(reportFile,"error: no kernel\n");
		run=false;
	}
	arg.erase("-ki");

	if (arg.count("-cti") == 1)
	{
		conciderTiAsCircular=true;
	}
	arg.erase("-cti");

	// LOOK FOR SETINGS	
							// number of nighbors QS, DS ...
	unsigned seed=std::chrono::high_resolution_clock::now().time_since_epoch().count();
	g2s::DistanceType searchDistance=g2s::EUCLIDIEN;
	
	bool withGPU=false;
	if (arg.count("-W_GPU") == 1)
	{
		withGPU=true;//atof((arg.find("-W_GPU")->second).c_str());
	}
	arg.erase("-W_GPU");

	bool withCUDA=false;
	std::vector<int> cudaDeviceList;
	#ifdef WITH_CUDA
	if (arg.count("-W_CUDA") >= 1)
	{
		withCUDA=true;
		int cudaDeviceAvailable=0;
		cudaGetDeviceCount(&cudaDeviceAvailable);
		std::multimap<std::string, std::string>::iterator deviceString=arg.lower_bound("-W_CUDA");
		if(deviceString==arg.upper_bound("-W_CUDA")){
			for (int i = 0; i < cudaDeviceAvailable; ++i)
			{
				cudaDeviceList.push_back(i);
			}
		}
		while(deviceString!=arg.upper_bound("-W_CUDA")){
			int deviceId=atoi((deviceString->second).c_str());
			cudaDeviceList.push_back(deviceId);
			deviceString++;
		}
	}
	arg.erase("-W_CUDA");
	#endif

	// print all ignored parameters
	for (std::multimap<std::string, std::string>::iterator it=arg.begin(); it!=arg.end(); ++it){
		fprintf(reportFile, "%s %s <== ignored !\n", it->first.c_str(), it->second.c_str());
	}

	if(!run){
		fprintf(reportFile, "simulation interupted !!\n");
		return 0;
	}

	if(maxNbNeihbours.size()<1)
		maxNbNeihbours.push_back(150);

#if _OPENMP
	//omp_set_num_threads(nbThreads);
	//omp_set_nested(true);
	fftwf_init_threads();
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

	std::vector<g2s::DataImage > kernels;

	for (size_t i = 0; i < kernelFileName.size(); ++i)
	{
		kernels.push_back(g2s::DataImage::createFromFile(kernelFileName[i]));
		if(kernels[i]._dims.size()-1==TIs[0]._dims.size()){
			kernels[i].convertFirstDimInVariable();
		}
	}

	std::vector<std::vector<int> > pathPosition;
	pathPosition.push_back(std::vector<int>(0));
	for (size_t i = 0; i < kernels[0]._dims.size(); ++i)
	{
		unsigned originalSize=pathPosition.size();
		int sizeInThisDim=(kernels[0]._dims[i]+1)/2;
		pathPosition.resize(originalSize*(2*sizeInThisDim-1));
		for (unsigned int k = 0; k < originalSize; ++k)
		{
			pathPosition[k].push_back(0);
		}
		for (int j = 1; j < sizeInThisDim; ++j)
		{
			std::copy ( pathPosition.begin(), pathPosition.begin()+originalSize, pathPosition.begin()+originalSize*(-1+2*j+0) );
			std::copy ( pathPosition.begin(), pathPosition.begin()+originalSize, pathPosition.begin()+originalSize*(-1+2*j+1) );
			for (unsigned int k = originalSize*(-1+2*j+0); k < originalSize*(-1+2*j+1); ++k)
			{
				pathPosition[k][i]=j;
			}
			for (unsigned int k = originalSize*(-1+2*j+1); k < originalSize*(-1+2*j+2); ++k)
			{
				pathPosition[k][i]=-j;
			}
		}
	}

	g2s::DataImage wieghtKernel=kernels[0].emptyCopy(true);
	if(searchDistance==g2s::EUCLIDIEN){
		for (unsigned int i = 0; i < wieghtKernel.dataSize(); ++i)
		{
			wieghtKernel._data[i]=-wieghtKernel.distance2ToCenter(i);
		}
	}

	unsigned center=0;
	g2s::DataImage* wieghtKernelPtr=wieghtKernel.ptr();
	for (int i =  wieghtKernelPtr->_dims.size()-1; i>=0 ; i--)
	{
		center=center*wieghtKernelPtr->_dims[i]+(wieghtKernelPtr->_dims[i]-1)/2;
	}

	std::sort(pathPosition.begin(),pathPosition.end(),[wieghtKernelPtr, center](std::vector<int> &a, std::vector<int> &b){
		unsigned l1,l2;
		wieghtKernelPtr->indexWithDelta(l1, center, a);
		wieghtKernelPtr->indexWithDelta(l2, center, b);
		return wieghtKernelPtr->_data[l1] > wieghtKernelPtr->_data[l2];
	});

	// init QS
	std::vector<SharedMemoryManager*> sharedMemoryManagerVector;// a new shared memory manager for each TI
	std::vector<ComputeDeviceModule*> *computeDeviceModuleArray=new std::vector<ComputeDeviceModule*> [nbThreads];


	bool needCrossMesurement=false;

	for (size_t i = 0; i < TIs.size(); ++i)
	{
		#pragma omp simd reduction(|:needCrossMesurement)
		for (unsigned int j = 0; j < TIs[i].dataSize(); ++j)
		{
			needCrossMesurement|=std::isnan(TIs[i]._data[j]);
		}
	}

	if(needCrossMesurement && !fullSimulation)
	{
		for (size_t i = 0; i < TIs.size(); ++i)
		{
			int nbVariable=TIs[i]._types.size();
			for (unsigned int j = 0; j < TIs[i].dataSize()/nbVariable; ++j)
			{
				bool hasNan=false;
				for (int k = 0; k < nbVariable; ++k)
				{
					hasNan|=std::isnan(TIs[i]._data[nbVariable*j+k]);
				}
				if(hasNan){
					for (int k = 0; k < nbVariable; ++k)
					{
						TIs[i]._data[nbVariable*j+k]=std::nanf("0");
					}
				}
			}
		}
	}


	std::vector<std::vector<float> > categoriesValues;
	std::vector<unsigned> numberDeComputedVariableProVariable;
	for (size_t i = 0; i < TIs[0]._types.size(); ++i)
	{
		if(TIs[0]._types[i]==g2s::DataImage::VaraibleType::Continuous)
			numberDeComputedVariableProVariable.push_back(1);
		if(TIs[0]._types[i]==g2s::DataImage::VaraibleType::Categorical){
			std::vector<float> currentVariable;
			for (size_t im = 0; im < TIs.size(); ++im)
			{
				for (unsigned int j = i; j < TIs[im].dataSize(); j+=TIs[im]._nbVariable)
				{
					if(std::isnan(TIs[im]._data[j]))
						continue;
					bool isPresent=false;
					for (size_t k = 0; k < currentVariable.size(); ++k)
					{
						isPresent|=((TIs[im]._data[j])==(currentVariable[k]));
					}
					if(!isPresent){
						currentVariable.push_back(TIs[im]._data[j]);
					}
				}
			}
			categoriesValues.push_back(currentVariable);
			numberDeComputedVariableProVariable.push_back(currentVariable.size());
		}
	}

	// correct the kernel to take in account categories
	for (int i = 0; i < kernels.size(); ++i)
	{
		kernels[i]=g2s::DataImage::offsetKernel4categories(kernels[i],numberDeComputedVariableProVariable);
	}

	std::vector<std::vector<convertionType> > convertionTypeVectorMainVector;
	std::vector<g2s::OperationMatrix> coeficientMatrix;
	std::vector<std::vector<std::vector<convertionType> > > convertionTypeVectorConstVector;
	std::vector<std::vector<std::vector<float> > > convertionCoefVectorConstVector;
	TIs[0].generateCoefMatrix4Xcorr(coeficientMatrix, convertionTypeVectorMainVector, convertionTypeVectorConstVector, convertionCoefVectorConstVector, needCrossMesurement, categoriesValues);


	for (size_t i = 0; i < TIs.size(); ++i)
	{
		SharedMemoryManager* smm=new SharedMemoryManager(TIs[i]._dims);

		std::vector<std::vector<g2s::DataImage> > variablesImages=TIs[i].convertInput4Xcorr(smm->_fftSize, needCrossMesurement, categoriesValues);

		for (size_t j = 0; j < variablesImages.size(); ++j)
		{
			for (size_t k = 0; k < variablesImages[j].size(); ++k)
			{
				smm->addVaraible(variablesImages[j][k]._data);
			}
		}
		// alloc module
		std::vector<unsigned> gpuHostUnifiedMemory;
		#ifdef WITH_OPENCL
		gpuHostUnifiedMemory=OpenCLGPUDevice::DeviceWithHostUnifiedMemory(0);

		#endif

		int cudaDeviceNumber=cudaDeviceList.size();
		int cudaDeviceUsed=0;

		#pragma omp parallel for proc_bind(spread) num_threads(nbThreads) default(none) shared(cudaDeviceList, cudaDeviceUsed, computeDeviceModuleArray) firstprivate(gpuHostUnifiedMemory, withGPU, conciderTiAsCircular, nbThreadsLastLevel,coeficientMatrix, smm, nbThreads, needCrossMesurement, cudaDeviceNumber)
		for (unsigned int i = 0; i < nbThreads; ++i)
		{
			//#pragma omp critical (createDevices)
			{
				bool deviceCreated=false;
				#ifdef WITH_OPENCL
				if((!deviceCreated) && (i<gpuHostUnifiedMemory.size()) && withGPU){
					OpenCLGPUDevice* signleThread=new OpenCLGPUDevice(smm, coeficientMatrix, 0,gpuHostUnifiedMemory[i], needCrossMesurement, conciderTiAsCircular);
					signleThread->setTrueMismatch(false);
					computeDeviceModuleArray[i].push_back(signleThread);
					deviceCreated=true;
				}
				#endif
				#ifdef WITH_CUDA
				int localDeviceId=INT_MAX;
				#pragma omp atomic capture
				localDeviceId = cudaDeviceUsed++;
				if(!deviceCreated && localDeviceId<cudaDeviceNumber){
					NvidiaGPUAcceleratorDevice* signleCudaThread=new NvidiaGPUAcceleratorDevice(cudaDeviceList[localDeviceId], smm, coeficientMatrix, nbThreadsLastLevel, needCrossMesurement, conciderTiAsCircular);
					signleCudaThread->setTrueMismatch(true);
					computeDeviceModuleArray[i].push_back(signleCudaThread);
					deviceCreated=true;

				}
				#endif
				if(!deviceCreated){
					CPUThreadDevice* signleThread=new CPUThreadDevice(smm, coeficientMatrix, nbThreadsLastLevel, needCrossMesurement, conciderTiAsCircular);
					//CPUThreadAcceleratorDevice* signleThread=new CPUThreadAcceleratorDevice(smm, coeficientMatrix, nbThreadsLastLevel, needCrossMesurement, conciderTiAsCircular);
					signleThread->setTrueMismatch(false);
					computeDeviceModuleArray[i].push_back(signleThread);
					deviceCreated=true;
				}
			}
		}
		smm->allowNewModule(false);
		sharedMemoryManagerVector.push_back(smm);
	}

	QuantileSamplingModule QSM(computeDeviceModuleArray,nullptr,maxNbCandidate,convertionTypeVectorMainVector, convertionTypeVectorConstVector, convertionCoefVectorConstVector, true, !needCrossMesurement, nbThreads, nbThreadsOverTi, nbThreadsLastLevel, false);

	if(densityArray.size()<1){
		densityArray.push_back(0.0078);
		densityArray.push_back(0.0156);
		densityArray.push_back(0.0312);
		densityArray.push_back(0.0625);
		densityArray.push_back(0.1250);
		densityArray.push_back(0.2500);
		densityArray.push_back(0.5000);
	}

	// fprintf(stderr, "Density : " );
	// for (int i = 0; i < densityArray.size(); ++i)
	// {
	// 	fprintf(stderr, "%f, ", densityArray[i]);
	// }
	// fprintf(stderr, "\n");

	std::vector<unsigned> dims;
	dims.push_back(maxNbNeihbours[0]);
	dims.push_back(kernels.size());
	dims.push_back(densityArray.size());

	g2s::DataImage meanErrorimage(dims.size(),dims.data(),maxNbCandidate);
	g2s::DataImage devErrorimage(dims.size(),dims.data(),maxNbCandidate);
	g2s::DataImage numberOFsampleimage(dims.size(),dims.data(),maxNbCandidate);
	numberOFsampleimage.setEncoding(g2s::DataImage::EncodingType::UInteger);

	// run calib
	std::atomic<bool> computationIsDone(false);
	auto begin = std::chrono::high_resolution_clock::now();

	simType st=vectorSim;
	if(fullSimulation) st=fullSim;

	switch (st){
		// case fullSim:
		// 	fprintf(reportFile, "%s\n", "full calib");
		// 	calibrationFull(reportFile, TIs, kernels, QSM, pathPosition, maxNbNeihbours,  categoriesValues, nbThreads );
		// 	break;
		case vectorSim:
			fprintf(reportFile, "%s\n", "vector calib");
			calibration(reportFile, meanErrorimage, devErrorimage, numberOFsampleimage, TIs, kernels, QSM, pathPosition, 
					maxNbNeihbours, densityArray, categoriesValues, metricPower, nbThreads,maxNumberOfIteration, minNumberOfIteration, maxt);
			break;
	}

	auto end = std::chrono::high_resolution_clock::now();
	computationIsDone=true;
	double time = 1.0e-6 * std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	fprintf(reportFile,"compuattion time: %7.2f s\n", time/1000);
	fprintf(reportFile,"compuattion time: %.0f ms\n", time);

	// free memory

	for (unsigned int i = 0; i < nbThreads; ++i)
	{
		for (size_t j = 0; j < computeDeviceModuleArray[i].size(); ++j)
		{
			delete computeDeviceModuleArray[i][j];
			computeDeviceModuleArray[i][j]=nullptr;
		}
	}

	for (size_t i = 0; i < sharedMemoryManagerVector.size(); ++i)
	{
		delete sharedMemoryManagerVector[i];
		sharedMemoryManagerVector[i]=nullptr;
	}

	delete[] computeDeviceModuleArray;

	// new filename 
	numberOFsampleimage.write(std::string("im_3_")+std::to_string(uniqueID));
	devErrorimage.write(std::string("im_2_")+std::to_string(uniqueID));
	meanErrorimage.write(std::string("im_1_")+std::to_string(uniqueID));
	

	// if(saveThread.joinable()){
	// 	saveThread.join();
	// }

#if _OPENMP
	fftwf_cleanup_threads();
#endif

	return 0;
}