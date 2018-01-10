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

#include "simulation.hpp"
#include "thresholdSamplingModule.hpp"

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
	if (arg.count("-ki") ==1)
	{
		kernelFileName=arg.find("-ki")->second;
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

	float threshold=std::nanf("0");			// threshold for DS ...
	int nbNeighbors=-1;						// number of nighbors QS, DS ...
	float mer=std::nanf("0");				// maximum exploration ratio, called f in ds
	float nbCandidate=std::nanf("0");		// 1/f for QS
	float narrowness=std::nanf("0");		// narrowness for NDS
	unsigned seed=std::chrono::high_resolution_clock::now().time_since_epoch().count();
	g2s::DistanceType searchDistance=g2s::EUCLIDIEN;

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

	if (arg.count("-n") == 1)
	{
		nbNeighbors=atoi((arg.find("-n")->second).c_str());
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
	float alpha=0;
	g2s::KernelType kernelTypeForGeneration=g2s::UNIFORM;
	unsigned kernelSize=-1;
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


	arg.erase("-ks");
	bool withGPU=false;
	if (arg.count("-W_GPU") == 1)
	{
		withGPU=true;//atof((arg.find("-W_GPU")->second).c_str());
	}
	arg.erase("-W_GPU");


	// precheck | check what is mandatory

	if(nbNeighbors<0){
		run=false;
		fprintf(reportFile, "%s\n", "number of neighbor not valide" );
	}
	if(std::isnan(threshold)){
		run=false;
		fprintf(reportFile, "%s\n", "threshold need to be seted" );
	}
	if(std::isnan(mer) && std::isnan(nbCandidate)){
		run=false;
		fprintf(reportFile, "%s\n", "maximum exploration ratio or numer of candidate need to be seted" );
	}
	if(std::isnan(nbCandidate)){
		nbCandidate=1/mer;
	}
	/*if(std::isnan(narrowness)){
		run=false;
		fprintf(reportFile, "%s\n", "narrowness need to be seted" );
	}*/

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

	g2s::DataImage DI=g2s::DataImage::createFromFile(targetFileName);


	g2s::DataImage kernel;
	g2s::DataImage simulationPath;

	if(kernelFileName.empty()) {
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
		std::vector<float> variableWeight(TIs[0]._nbVariable);
		for (int i = 0; i < variableWeight.size(); ++i)
		{
			variableWeight[i]=1;
		}
		std::vector<float> alphas(TIs[0]._nbVariable);
		for (int i = 0; i < alphas.size(); ++i)
		{
			alphas[i]=alpha;
		}
		std::vector<g2s::KernelType> kernelsTypeFG(TIs[0]._nbVariable);
		for (int i = 0; i < kernelsTypeFG.size(); ++i)
		{
			kernelsTypeFG[i]=kernelTypeForGeneration;
		}
		kernel=g2s::DataImage::genearteKernel(kernelsTypeFG, maxSize, variableWeight, alphas);
	}
	else {
		kernel=g2s::DataImage::createFromFile(kernelFileName);
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
	if(searchDistance==g2s::EUCLIDIEN){
		for (int i = 0; i < wieghtKernel.dataSize(); ++i)
		{
			wieghtKernel._data[i]=-wieghtKernel.distance2ToCenter(i);
		}
	}
	if(searchDistance==g2s::KERNEL){
		unsigned nbV=kernel._nbVariable;
		for (int i = 0; i < wieghtKernel.dataSize(); ++i)
		{
			for (int j = 0; j < nbV; ++j)
			{
				if(fabs(kernel._data[i*nbV+j])>wieghtKernel._data[i])wieghtKernel._data[i]=fabs(kernel._data[i*nbV+j]);
			}
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


	/*for (int i = 0; i < 10; ++i)
	{
		for (int k = 0; k < pathPosition[i].size(); ++k)
		{
			fprintf(stderr, "%d ", pathPosition[i][k]);
		}
		unsigned position;
		wieghtKernelPtr->indexWithDelta(position, center, pathPosition[i]);
		fprintf(stderr, " ==> %f\n",wieghtKernelPtr->_data[position]);
	}
	fprintf(stderr, "\n\n" );*/

	if(simuationPathFileName.empty()) {
		//fprintf(stderr, "generate simulation path\n");
		simulationPath=DI.emptyCopy(true);
		for (int i = 0; i < simulationPath.dataSize(); ++i)
		{
			simulationPath._data[i]=i;
		}
		std::shuffle(simulationPath._data, simulationPath._data + simulationPath.dataSize(), randomGenerator );
		
		for (int i = 0; i < simulationPath.dataSize(); ++i)
		{
			bool valueSeted=true;
			for (int j = 0; j < DI._nbVariable; ++j)
			{
				if(std::isnan(DI._data[i*DI._nbVariable+j]))valueSeted=false;
			}
			if(valueSeted)
				simulationPath._data[i]=-INFINITY;
		}
	}
	else {
		simulationPath=g2s::DataImage::createFromFile(simuationPathFileName);
	}

	bool dimAgree=true;
	if(simulationPath._dims.size()!=DI._dims.size())dimAgree=false;
	for (int i = 0; i < simulationPath._dims.size(); ++i)
	{
		if(simulationPath._dims[i]!=DI._dims[i])dimAgree=false;
	}
	if(!dimAgree){
		fprintf(reportFile, "%s\n", "dimension bettwen simulation path and destination grid disagree");
		return 0;
	}

	unsigned simulationPathSize=0;
	unsigned* simulationPathIndex=(unsigned *)malloc(sizeof(unsigned)*simulationPath.dataSize());
	std::iota(simulationPathIndex,simulationPathIndex+simulationPath.dataSize(),0);
	float* simulationPathData=simulationPath._data;
	std::sort(simulationPathIndex, simulationPathIndex+simulationPath.dataSize(),
		[simulationPathData](unsigned i1, unsigned i2) {return simulationPathData[i1] < simulationPathData[i2];});

	//Search begin path

	unsigned beginPath=0;

	for ( beginPath=0 ; beginPath < simulationPath.dataSize(); ++beginPath)
	{
		float value=simulationPathData[simulationPathIndex[beginPath]];
		if((!std::isinf(value))||(value>0)) break;
	}

	unsigned* importDataIndex=(unsigned *)malloc(sizeof(unsigned)*simulationPath.dataSize());
	memset(importDataIndex,0,sizeof(unsigned)*simulationPath.dataSize());
	float* seedForIndex=( float* )malloc( sizeof(float) * DI.dataSize()/DI._nbVariable );
	
	std::uniform_real_distribution<float> uniformDitributionOverSource(0.f,1.f);

	for ( unsigned int i = 0; i < DI.dataSize()/DI._nbVariable; ++i)
	{
		seedForIndex[i]=uniformDitributionOverSource(randomGenerator);
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

	bool varaibleTypeAreCompatible=true;


	for (int i = 0; i < TIs.size(); ++i)
	{
		for (int j = 0; j < TIs[i]._types.size(); ++j)
		{
			varaibleTypeAreCompatible&=((TIs[i]._types[j])==(DI._types[j]));
		}
	}

	if(!varaibleTypeAreCompatible) {

		fprintf(reportFile, "TI(s) not compatible to gather or/and with the DI ==> simulation interupted !!\n");
		return 0;
	}

	for (int i = 0; i < TIs.size(); ++i)
	{
		if((TIs[i]._types.size())!=(DI._types.size())){
			varaibleTypeAreCompatible=false;
			break;
		}
		for (int j = 0; j < TIs[i]._types.size(); ++j)
		{
			varaibleTypeAreCompatible&=((TIs[i]._types[j])==(DI._types[j]));
		}
	}

	std::vector<std::vector<float> > categoriesValues;

	for (int i = 0; i < DI._types.size(); ++i)
	{
		if(DI._types[i]!=g2s::DataImage::VaraibleType::Categorical) continue;
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
					signleThread->setTrueMismatch(true);
					computeDeviceModuleArray[i].push_back(signleThread);
					deviceCreated=true;
				}
				#endif
				if(!deviceCreated){
					CPUThreadDevice* signleThread=new CPUThreadDevice(smm,threadRatio, needCrossMesurement);
					signleThread->setTrueMismatch(true);
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

	ThresholdSamplingModule TSM(computeDeviceModuleArray,&kernel, threshold*threshold, mer,convertionTypeVectorMainVector,variablesCoeficientMainVector,!needCrossMesurement);
	// run DS

	auto begin = std::chrono::high_resolution_clock::now();

	simulation(reportFile, DI, TIs, TSM, pathPosition, simulationPathIndex+beginPath, simulationPath.dataSize()-beginPath,
	 seedForIndex, importDataIndex, nbNeighbors, categoriesValues, nbThreads);
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

	
	g2s::DataImage id=DI.emptyCopy(true);
	id.setEncoding(g2s::DataImage::UInteger);
	memcpy(id._data,importDataIndex,id.dataSize()*sizeof(unsigned int));
	id.write(outputIndexFilename);
	DI.write(outputFilename);

	free(simulationPathIndex);
	simulationPathIndex=nullptr;
	free(importDataIndex);
	importDataIndex=nullptr;

	return 0;
}