#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>



#include "utils.hpp"
#include "DataImage.hpp"
#include "jobManager.hpp"

#include "sharedMemoryManager.hpp"
#include "CPUThreadDevice.hpp"
#ifdef WITH_OPENCL
#include "OpenCLGPUDevice.hpp"
#endif

#include "kernelOptimization.hpp"
#include "quantileSamplingModule.hpp"

void printHelp(){
	printf ("that is the help");
}

int main(int argc, char const *argv[]) {

	std::multimap<std::string, std::string> arg=g2s::argumentReader(argc,argv);

	char logFileName[2048]={0};
	std::vector<std::string> sourceFileNameVector;
	std::string targetFileName;
	std::string kernelFileName;
	std::string simuationPathFileName;

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
	unsigned threadRatio=1;
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



	// LOOK FOR SETINGS

	float threshold=std::nanf("0");			// threshold for DS ...
	int nbNeighbors=-1;						// number of nighbors QS, DS ...
	float mer=std::nanf("0");				// maximum exploration ratio, called f in ds
	float nbCandidate=std::nanf("0");		// 1/f for QS
	float narrowness=std::nanf("0");		// narrowness for NDS
	unsigned seed=std::chrono::high_resolution_clock::now().time_since_epoch().count();
	g2s::DistanceType searchDistance=g2s::EUCLIDIEN;
	
	if (arg.count("-s") == 1)
	{
		seed=atoi((arg.find("-s")->second).c_str());
	}
	arg.erase("-s");

	if (arg.count("-k") == 1)
	{
		nbCandidate=atof((arg.find("-k")->second).c_str());
	}
	arg.erase("-k");

	unsigned kernelSize=-1;
	if (arg.count("-ks") == 1)
	{
		kernelSize=atoi((arg.find("-ks")->second).c_str());
	}
	arg.erase("-ks");

	bool withGPU=false;
	if (arg.count("-W_GPU") == 1)
	{
		withGPU=true;//atof((arg.find("-W_GPU")->second).c_str());
	}
	arg.erase("-W_GPU");


	std::string optimAlgo=std::string("genetic");

	//GeneAlgo configGA

	geneAlgoConfig configGA;
	greedyAlgoConfig configGreedy;


	if (arg.count("-oa") == 1)
	{
		optimAlgo=arg.find("-oa")->second;
	}
	arg.erase("-oa");

	if(!optimAlgo.compare("genetic"))
	{

		configGA.populationSize=200;
		configGA.survivingRate=0.3f;
		configGA.mutationRate=0.05f;
		configGA.heritageRate=0.5f;
		configGA.appearingRate=0.02f;
		configGA.randomPatternPoolSize=300;
		configGA.maxIteration=10000;

		if (arg.count("-ps") == 1)
		{
			configGA.populationSize=atoi((arg.find("-ps")->second).c_str());
		}
		arg.erase("-ps");

		if (arg.count("-sr") == 1)
		{
			configGA.survivingRate=atof((arg.find("-sr")->second).c_str());
		}
		arg.erase("-sr");

		if (arg.count("-mr") == 1)
		{
			configGA.mutationRate=atof((arg.find("-mr")->second).c_str());
		}
		arg.erase("-mr");

		if (arg.count("-hr") == 1)
		{
			configGA.heritageRate=atof((arg.find("-hr")->second).c_str());
		}
		arg.erase("-hr");

		if (arg.count("-ar") == 1)
		{
			configGA.appearingRate=atof((arg.find("-ar")->second).c_str());
		}
		arg.erase("-ar");

		if (arg.count("-pps") == 1)
		{
			configGA.randomPatternPoolSize=atoi((arg.find("-pps")->second).c_str());
		}
		arg.erase("-pps");

		if (arg.count("-maxi") == 1)
		{
			configGA.maxIteration=atoi((arg.find("-maxi")->second).c_str());
		}
		arg.erase("-maxi");
	}

	if(!optimAlgo.compare("greedy"))
	{

		configGreedy.initValue=0.001f;
		configGreedy.scalefactor=1.5;
		configGreedy.maxValue=100.f;
		configGreedy.randomPatternPoolSize=5000;

		if (arg.count("-iv") == 1)
		{
			configGreedy.initValue=atof((arg.find("-iv")->second).c_str());
		}
		arg.erase("-iv");

		if (arg.count("-sf") == 1)
		{
			configGreedy.scalefactor=atof((arg.find("-sf")->second).c_str());
		}
		arg.erase("-sf");

		if (arg.count("-mv") == 1)
		{
			configGreedy.maxValue=atof((arg.find("-mv")->second).c_str());
		}
		arg.erase("-mv");

		if (arg.count("-pps") == 1)
		{
			configGreedy.randomPatternPoolSize=atoi((arg.find("-pps")->second).c_str());
		}
		arg.erase("-pps");

	}


	// print all ignored parameters
	for (std::multimap<std::string, std::string>::iterator it=arg.begin(); it!=arg.end(); ++it){
		fprintf(reportFile, "%s %s <== ignored !\n", it->first.c_str(), it->second.c_str());
	}

	if(!run){
		fprintf(reportFile, "simulation interupted !!\n");
		return 0;
	}

#if _OPENMP
	omp_set_num_threads(nbThreads);
#endif
	std::mt19937 randomGenerator(seed);

	std::vector<g2s::DataImage > TIs;

	for (int i = 0; i < sourceFileNameVector.size(); ++i)
	{
		TIs.push_back(std::move(g2s::DataImage::createFromFile(sourceFileNameVector[i])));
	}

	// init QS
	std::vector<SharedMemoryManager*> sharedMemoryManagerVector;// a new shared memory manager for each TI
	std::vector<ComputeDeviceModule*> *computeDeviceModuleArray=new std::vector<ComputeDeviceModule*> [nbThreads];

	bool needCrossMesurement=false;

	for (int i = 0; i < TIs.size(); ++i)
	{
		#pragma omp simd reduction(|:needCrossMesurement)
		for (int j = 0; j < TIs[i].dataSize(); ++j)
		{
			needCrossMesurement|=std::isnan(TIs[i]._data[j]);
		}
	}

	std::vector<std::vector<float> > categoriesValues;

	for (int i = 0; i < TIs[0]._types.size(); ++i)
	{
		if(TIs[0]._types[i]!=g2s::DataImage::VaraibleType::Categorical) continue;
		std::vector<float> currentVariable;
		for (int im = 0; im < TIs.size(); ++im)
		{
			for (int j = i; j < TIs[im].dataSize(); j+=TIs[im]._nbVariable)
			{
				bool isPresent=false;
				for (int k = 0; k < currentVariable.size(); ++k)
				{
					isPresent|=((TIs[im]._data[j])==(currentVariable[k]));
				}
				if(!isPresent){
					currentVariable.push_back(TIs[im]._data[j]);
				}
			}
		}
		categoriesValues.push_back(currentVariable);
	}

	g2s::DataImage kernel;
	{
		std::vector<unsigned> maxSize=TIs[0]._dims;
		if(kernelSize!=-1){
			for (int i = 0; i < maxSize.size(); ++i)
			{
				maxSize[i]=kernelSize;
			}
		}else{
			for (int j = 0; j < TIs.size(); ++j)
			{
				for (int i = 0; i < maxSize.size(); ++i)
				{
					maxSize[i]=std::min(TIs[j]._dims[i]/2+1,maxSize[i]);
				}
			}
		}
		kernel=g2s::DataImage(maxSize.size(),maxSize.data(),TIs[0]._nbVariable);
	}


	std::vector<std::vector<int> > pathPosition;
	pathPosition.push_back(std::vector<int>(0));
	for (int i = 0; i < kernel._dims.size(); ++i)
	{
		unsigned originalSize=pathPosition.size();
		int sizeInThisDim=(kernel._dims[i]+1)/2;
		pathPosition.resize(originalSize*(2*sizeInThisDim-1));
		for (int k = 0; k < originalSize; ++k)
		{
			pathPosition[k].push_back(0);
		}
		for (int j = 1; j < sizeInThisDim; ++j)
		{
			std::copy ( pathPosition.begin(), pathPosition.begin()+originalSize, pathPosition.begin()+originalSize*(-1+2*j+0) );
			std::copy ( pathPosition.begin(), pathPosition.begin()+originalSize, pathPosition.begin()+originalSize*(-1+2*j+1) );
			for (int k = originalSize*(-1+2*j+0); k < originalSize*(-1+2*j+1); ++k)
			{
				pathPosition[k][i]=j;
			}
			for (int k = originalSize*(-1+2*j+1); k < originalSize*(-1+2*j+2); ++k)
			{
				pathPosition[k][i]=-j;
			}
		}
	}

	g2s::DataImage wieghtKernel=kernel.emptyCopy(true);
	if(searchDistance==g2s::EUCLIDIEN){ //only radius search
		for (int i = 0; i < wieghtKernel.dataSize(); ++i)
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

	for (int i = 0; i < TIs.size(); ++i)
	{
		SharedMemoryManager* smm=new SharedMemoryManager(TIs[i]._dims);

		std::vector<std::vector<g2s::DataImage> > variablesImages=TIs[i].convertInput4Xcorr(smm->_fftSize, needCrossMesurement, categoriesValues);

		for (int j = 0; j < variablesImages.size(); ++j)
		{
			for (int k = 0; k < variablesImages[j].size(); ++k)
			{
				smm->addVaraible(variablesImages[j][k]._data);
			}
		}
		// alloc module
		#ifdef WITH_OPENCL
		std::vector<unsigned> gpuHostUnifiedMemory=OpenCLGPUDevice::DeviceWithHostUnifiedMemory(0);

		#endif

		#pragma omp parallel for num_threads(nbThreads) default(none) shared(computeDeviceModuleArray) firstprivate(threadRatio, smm, nbThreads, needCrossMesurement)
		for (int i = 0; i < nbThreads; ++i)
		{
			#pragma omp critical (createDevices)
			{
				bool deviceCreated=false;

				#ifdef WITH_OPENCL
				if((!deviceCreated) && (i<gpuHostUnifiedMemory.size()) && withGPU){
					OpenCLGPUDevice* signleThread=new OpenCLGPUDevice(smm,0,gpuHostUnifiedMemory[i], needCrossMesurement);
					signleThread->setTrueMismatch(false);
					computeDeviceModuleArray[i].push_back(signleThread);
					deviceCreated=true;
				}
				#endif
				if(!deviceCreated){
					CPUThreadDevice* signleThread=new CPUThreadDevice(smm,threadRatio, needCrossMesurement);
					signleThread->setTrueMismatch(false);
					computeDeviceModuleArray[i].push_back(signleThread);
					deviceCreated=true;
				}
			}
		}
		smm->allowNewModule(false);
		sharedMemoryManagerVector.push_back(smm);
	}

	std::vector<std::vector<float> > variablesCoeficientMainVector;
	std::vector<std::vector<convertionType> > convertionTypeVectorMainVector;

	TIs[0].generateCoef4Xcorr(variablesCoeficientMainVector, convertionTypeVectorMainVector, needCrossMesurement, categoriesValues);

	QuantileSamplingModule QSM(computeDeviceModuleArray,&kernel,nbCandidate,convertionTypeVectorMainVector,variablesCoeficientMainVector, !needCrossMesurement, nbThreads);

	// run KO

	auto begin = std::chrono::high_resolution_clock::now();
	
	if(!optimAlgo.compare("genetic"))
		geneAlgo(reportFile, TIs, kernel, QSM, pathPosition, seed, nbCandidate, categoriesValues, configGA, 10, nbThreads);
	if(!optimAlgo.compare("greedy"))
		greedyAlgo(reportFile, TIs, kernel, QSM, pathPosition, seed, nbCandidate, categoriesValues,configGreedy, 10, nbThreads);


	auto end = std::chrono::high_resolution_clock::now();
	double time = 1.0e-6 * std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	fprintf(reportFile,"compuattion time: %7.2f s\n", time/1000);
	fprintf(reportFile,"compuattion time: %.0f ms\n", time);

	// free memory

	for (int i = 0; i < nbThreads; ++i)
	{
		for (int j = 0; j < computeDeviceModuleArray[i].size(); ++j)
		{
			delete computeDeviceModuleArray[i][j];
			computeDeviceModuleArray[i][j]=nullptr;
		}
	}

	for (int i = 0; i < sharedMemoryManagerVector.size(); ++i)
	{
		delete sharedMemoryManagerVector[i];
		sharedMemoryManagerVector[i]=nullptr;
	}

	delete[] computeDeviceModuleArray;

	kernel.setEncoding(g2s::DataImage::Float);
	kernel.write(outputFilename);

	return 0;
}