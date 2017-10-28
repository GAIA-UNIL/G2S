#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>



#include "utils.hpp"
#include "dataManagement.hpp"
#include "jobManager.hpp"

#include "sharedMemoryManager.hpp"
#include "CPUThreadDevice.hpp"
#ifdef WITH_OPENCL
#include "OpenCLGPUDevice.hpp"
#endif

#include "simulation.hpp"
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
	std::string outputNarrownessFilename;
	std::string outputPathFilename;


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
		fprintf(reportFile,"error kernel\n");
	}
	arg.erase("-ki");




	// LOOK FOR OUTPUT
	if (arg.count("-o") ==1)
	{
		outputFilename=arg.find("-o")->second;
		run=false;
	}else{
		outputFilename=std::to_string(uniqueID);
		outputIndexFilename=std::string("id_")+std::to_string(uniqueID);
		outputNarrownessFilename=std::string("nw_")+std::to_string(uniqueID);
		outputPathFilename=std::string("path_")+std::to_string(uniqueID);
	}
	arg.erase("-o");

	if (arg.count("-oi") ==1)
	{
		outputIndexFilename=arg.find("-oi")->second;
	}
	arg.erase("-oi");

	if (arg.count("-op") ==1)
	{
		outputPathFilename=arg.find("-op")->second;
	}
	arg.erase("-op");

		if (arg.count("-onw") ==1)
	{
		outputNarrownessFilename=arg.find("-onw")->second;
	}
	arg.erase("-onw");





	// LOOK FOR SETINGS

	float mer=std::nanf("0");				// maximum exploration ratio, called f in ds
	float nbCandidate=std::nanf("0");		// 1/f for QS
	float narrownessRange=0.5;			 		// narrowness for NDS
	unsigned seed=std::chrono::high_resolution_clock::now().time_since_epoch().count();
	unsigned chunkSize=1;
	unsigned updateRadius=10;
	float maxProgression=1;
	g2s::DistanceType searchDistance=g2s::EUCLIDIEN;

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
		narrownessRange=atof((arg.find("-nw")->second).c_str());
	}
	arg.erase("-nw");

	if (arg.count("-cs") == 1)
	{
		chunkSize=atoi((arg.find("-cs")->second).c_str());
	}
	arg.erase("-cs");

	if (arg.count("-uds") == 1)
	{
		updateRadius=atoi((arg.find("-uds")->second).c_str());
	}
	arg.erase("-uds");

	if (arg.count("-mp") == 1)
	{
		maxProgression=atof((arg.find("-mp")->second).c_str());
	}
	arg.erase("-mp");

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
	if(std::isnan(mer) && std::isnan(nbCandidate)){
		run=false;
		fprintf(reportFile, "%s\n", "maximum exploration ratio or numer of candidate need to be seted" );
	}
	if(std::isnan(nbCandidate)){
		nbCandidate=1/mer;
	}
	if(std::isnan(narrownessRange)){
		run=false;
		fprintf(reportFile, "%s\n", "narrowness need to be seted" );
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
		TIs.push_back(std::move(g2s::DataImage(sourceFileNameVector[i])));
	}

	g2s::DataImage DI(targetFileName);

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
		kernel=g2s::DataImage(kernelFileName);
	}

	std::vector<std::vector<int> > pathPosition;
	pathPosition.push_back(std::vector<int>(0));
	for (int i = 0; i < kernel._dims.size(); ++i)
	{
		unsigned originalSize=pathPosition.size();
		int sizeInThisDim=(kernel._dims[i])/2+1;
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

	/*for (int i = 0; i < pathPosition.size(); ++i)
	{
		fprintf(stderr, "____\n");
		for (int j = 0; j < pathPosition[i].size(); ++j)
		{
			fprintf(stderr, "%d, ",pathPosition[i][j] );
		}
		fprintf(stderr, "\n");
	}*/

	

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
		center=center*wieghtKernelPtr->_dims[i]+wieghtKernelPtr->_dims[i]/2;
	}


	//TODO
	std::sort(pathPosition.begin(),pathPosition.end(),[wieghtKernelPtr, center](std::vector<int> &a, std::vector<int> &b){
		unsigned l1,l2;
		wieghtKernelPtr->indexWithDelta(l1, center, a);
		wieghtKernelPtr->indexWithDelta(l2, center, b);
		return wieghtKernelPtr->_data[l1] < wieghtKernelPtr->_data[l2];
	});

	/*std::sort(pathPosition.begin(),pathPosition.end(),[wieghtKernelPtr, center](std::vector<int> &a, std::vector<int> &b){
		unsigned l1,l2;
		wieghtKernelPtr->indexWithDelta(l1, center, a);
		wieghtKernelPtr->indexWithDelta(l2, center, b);
		return a[0]*a[0]+a[1]*a[1] < b[0]*b[0]+b[1]*b[1] ;
	});*/

	simulationPath=DI.emptyCopy(true);

	
	unsigned* importDataIndex=(unsigned *)malloc(sizeof(unsigned)*simulationPath.dataSize());
	memset(importDataIndex,0,sizeof(unsigned)*simulationPath.dataSize());
	float* seedForIndex=( float* )malloc( sizeof(float) * DI.dataSize()/DI._nbVariable );
	
	std::uniform_real_distribution<float> uniformDitributionOverSource(0.f,1.f);

	for ( unsigned int i = 0; i < DI.dataSize()/DI._nbVariable; ++i)
	{
		seedForIndex[i]=uniformDitributionOverSource(randomGenerator);
	}

	g2s::DataImage NI=DI.emptyCopy(true); //narrownessImage

	// init NDS
	std::vector<SharedMemoryManager*> sharedMemoryManagerVector;// a new shared memory manager for each TI
	std::vector<ComputeDeviceModule*> *computeDeviceModuleArray=new std::vector<ComputeDeviceModule*> [nbThreads];


	bool needCrossMesuremnt=false;

	for (int i = 0; i < TIs.size(); ++i)
	{
		#pragma omp simd reduction(|:needCrossMesuremnt)
		for (int j = 0; j < TIs[i].dataSize(); ++j)
		{
			needCrossMesuremnt|=std::isnan(TIs[i]._data[j]);
		}
	}


	for (int i = 0; i < TIs.size(); ++i)
	{
		std::vector<unsigned> srcSize;
		srcSize.push_back((TIs[i]._dims.size()>0) ?TIs[i]._dims[0]:1);
		if(TIs[i]._dims.size()>1) srcSize.push_back(TIs[i]._dims[1]);
		if(TIs[i]._dims.size()>2) srcSize.push_back(TIs[i]._dims[2]);
		unsigned srcSizeX=1;
		unsigned srcSizeY=1;
		unsigned srcSizeZ=1;
		if(srcSize.size()>0) srcSizeX=srcSize[0];
		if(srcSize.size()>1) srcSizeY=srcSize[1];
		if(srcSize.size()>2) srcSizeZ=srcSize[2];
		float* inputData=TIs[i]._data;
		SharedMemoryManager* smm=new SharedMemoryManager(srcSize);
		unsigned fftSizeX=1;
		unsigned fftSizeY=1;
		unsigned fftSizeZ=1;
		if(smm->_fftSize.size()>0)fftSizeX=smm->_fftSize[0];
		if(smm->_fftSize.size()>1)fftSizeY=smm->_fftSize[1];
		if(smm->_fftSize.size()>2)fftSizeZ=smm->_fftSize[2];
		unsigned srcVariable=TIs[i]._nbVariable;

		float** varaibleBands=(float**)malloc( 3 * srcVariable * sizeof(float*));
	
		// init Data
		for (int i = 0; i < srcVariable; ++i)
		{
			varaibleBands[3*i+0]=(float*)malloc(fftSizeX * fftSizeY * fftSizeZ * sizeof(float));
			varaibleBands[3*i+1]=(float*)malloc(fftSizeX * fftSizeY * fftSizeZ * sizeof(float));
			memset(varaibleBands[3*i+0],0,fftSizeX * fftSizeY * fftSizeZ * sizeof(float));
			memset(varaibleBands[3*i+1],0,fftSizeX * fftSizeY * fftSizeZ * sizeof(float));
			if(needCrossMesuremnt){
				varaibleBands[3*i+2]=(float*)malloc(fftSizeX * fftSizeY * fftSizeZ * sizeof(float));
				memset(varaibleBands[3*i+2],0,fftSizeX * fftSizeY * fftSizeZ * sizeof(float));
				std::fill(varaibleBands[3*i+1],varaibleBands[3*i+1]+fftSizeX * fftSizeY * fftSizeZ -1, std::nanf("0"));
			}
		}

		for (int v = 0; v < srcVariable; ++v)
		{
			float* A=varaibleBands[3*v+1];
			float* A0=varaibleBands[3*v+2];
			for (int l = 0; l < fftSizeZ; ++l)
			{
				if(l<(fftSizeZ-srcSizeZ))continue;
				for (int j = 0; j < fftSizeY; ++j)
				{
					if(j<(fftSizeY-srcSizeY))continue;
					for (int i = 0; i < fftSizeX; ++i)
					{
						if(i<(fftSizeX-srcSizeX))continue;
						A[i+j*fftSizeX+l*fftSizeY*fftSizeX]=((float*)inputData)[((fftSizeX-i-1)+(fftSizeY-j-1)*srcSizeX+(fftSizeZ-l-1)*srcSizeX*srcSizeY)*srcVariable+v];
					}
				}
			}
		}

		for (int i = 0; i < srcVariable; ++i)
		{
			float* A=varaibleBands[3*i+1];
			float* A2=varaibleBands[3*i+0];
			#pragma omp simd
			for (unsigned index = 0; index < fftSizeZ*fftSizeY*fftSizeX; ++index)
			{
				A2[index]=A[index]*A[index];
			}
		}

		if(needCrossMesuremnt){
			for (int i = 0; i < srcVariable; ++i)
			{
				float* A2=varaibleBands[3*i+0];
				float* A =varaibleBands[3*i+1];
				float* A0=varaibleBands[3*i+2];
				#pragma omp simd
				for (unsigned index = 0; index < fftSizeZ*fftSizeY*fftSizeX; ++index)
				{
					A0[index]=!std::isnan(A[index]);
				}
				#pragma omp simd
				for (unsigned index = 0; index < fftSizeZ*fftSizeY*fftSizeX; ++index)
				{
					if(std::isnan(A[index]))A[index]=0;
					if(std::isnan(A2[index]))A2[index]=0;
				}
			}
		}

		// add sources : inputData
		for (int i = 0; i < srcVariable; ++i)
		{
			smm->addVaraible(varaibleBands[3*i+0]);
			smm->addVaraible(varaibleBands[3*i+1]);
			if(needCrossMesuremnt){
				smm->addVaraible(varaibleBands[3*i+2]);
			}
		}
		// alloc module
		#ifdef WITH_OPENCL
		std::vector<unsigned> gpuHostUnifiedMemory=OpenCLGPUDevice::DeviceWithHostUnifiedMemory(0);

		#endif

		#pragma omp parallel for num_threads(nbThreads) default(none) shared(computeDeviceModuleArray) firstprivate(threadRatio, smm, nbThreads, needCrossMesuremnt)
		for (int i = 0; i < nbThreads; ++i)
		{
			#pragma omp critical (createDevices)
			{
				bool deviceCreated=false;

				#ifdef WITH_OPENCL
				if((!deviceCreated) && (i<gpuHostUnifiedMemory.size()) && withGPU){
					OpenCLGPUDevice* signleThread=new OpenCLGPUDevice(smm,0,gpuHostUnifiedMemory[i], needCrossMesuremnt);
					signleThread->setTrueMismatch(false);
					computeDeviceModuleArray[i].push_back(signleThread);
					deviceCreated=true;
				}
				#endif
				if(!deviceCreated){
					CPUThreadDevice* signleThread=new CPUThreadDevice(smm,threadRatio, needCrossMesuremnt);
					signleThread->setTrueMismatch(false);
					computeDeviceModuleArray[i].push_back(signleThread);
					deviceCreated=true;
				}
			}
		}

		smm->allowNewModule(false);

		sharedMemoryManagerVector.push_back(smm);

		for (int i = 0; i < srcVariable; ++i)
		{
			free(varaibleBands[3*i+0]);
			free(varaibleBands[3*i+1]);
			if(needCrossMesuremnt)free(varaibleBands[3*i+2]);
			varaibleBands[3*i+0]=nullptr;
			varaibleBands[3*i+1]=nullptr;
			varaibleBands[3*i+2]=nullptr;
		}

		free(varaibleBands);
		varaibleBands=nullptr;
	}

	std::vector<std::vector<float> > variablesCoeficientMainVector;
	std::vector<std::vector<SamplingModule::convertionType> > convertionTypeVectorMainVector;

	for (int i = 0; i < TIs[0]._nbVariable; ++i)
	{

		std::vector<float> variablesCoeficient;
		std::vector<SamplingModule::convertionType> convertionTypeVector;
		variablesCoeficient.push_back(1.0f);
		convertionTypeVector.push_back(SamplingModule::convertionType::P0);
		variablesCoeficient.push_back(-2.0f);
		convertionTypeVector.push_back(SamplingModule::convertionType::P1);
		if(needCrossMesuremnt){
			convertionTypeVector.push_back(SamplingModule::convertionType::P2);
			variablesCoeficient.push_back(1.0f);
		}else{
			// for delta
			variablesCoeficient.push_back(1.0f);
		}

		variablesCoeficientMainVector.push_back(variablesCoeficient);
		convertionTypeVectorMainVector.push_back(convertionTypeVector);
	}


	QuantileSamplingModule QSM(computeDeviceModuleArray,&kernel,nbCandidate,convertionTypeVectorMainVector,variablesCoeficientMainVector, !needCrossMesuremnt, nbThreads);
	QSM.setNarrownessFunction([&TIs,narrownessRange](float* errors, unsigned int *tiId, unsigned int *indexId , unsigned int nb){
		unsigned nbVariable=TIs[0]._nbVariable;
		float values[nb*nbVariable];
		float *values_ptr=values;;
		for (int i = 0; i < nb; ++i)
		{
			for (int j = 0; j < nbVariable; ++j)
			{
				values[i*nbVariable+j]=TIs[tiId[i]]._data[indexId[i]*TIs[tiId[i]]._nbVariable+j];
			}
		}
		unsigned localPosition[nb];
		std::iota(localPosition,localPosition+nb-1,0);
		std::sort(localPosition,localPosition+nb-1,[values_ptr,nbVariable](unsigned a, unsigned b){
			int valide=0;
			for (int j = 0; j < nbVariable; ++j)
			{
				valide+=(values_ptr[a*nbVariable+j]<values_ptr[b*nbVariable+j]);
			}
			return (valide<nbVariable/2);
		});

		float narrowness=0;

		for (int j = 0; j < nbVariable; ++j)
		{
			narrowness+=fabs(values[localPosition[int(ceil((0.5f-narrownessRange/2.f)*nb))]*nbVariable+j]-values[localPosition[int(floor((0.5f+narrownessRange/2.f)*nb))]*nbVariable+j]);
		}
		return narrowness/nbVariable;

	});
	// run QS

	auto begin = std::chrono::high_resolution_clock::now();

	narrowPathSimulation(reportFile, DI, NI, TIs, kernel, QSM, pathPosition, (unsigned*)simulationPath._data,
		seedForIndex, importDataIndex, chunkSize, updateRadius,maxProgression, nbThreads);
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

	DI.write(outputFilename);
	g2s::DataImage id=DI.emptyCopy(true);
	memcpy(id._data,importDataIndex,id.dataSize()*sizeof(unsigned int));
	id.write(outputIndexFilename);
	NI.write(outputNarrownessFilename);
	simulationPath.write(outputPathFilename);

	free(importDataIndex);
	importDataIndex=nullptr;

	return 0;
}