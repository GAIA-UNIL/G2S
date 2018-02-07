#ifndef KERNEL_OPTIMIZATION_HPP
#define KERNEL_OPTIMIZATION_HPP

#include "computeDeviceModule.hpp"
#include "quantileSamplingModule.hpp"
#include "fKst.hpp"
#include <thread>
#include <random>

template<class Engine>
void newKernel(g2s::DataImage *kernel, Engine &randomGenerator){
	std::uniform_real_distribution<float> uniformDitribution(0.f,1.f);
	float sum=0.f;
	for (int i = 0; i < kernel->dataSize(); ++i)
	{
		kernel->_data[i]=uniformDitribution(randomGenerator);
		sum+=kernel->_data[i];
	}
	for (int i = 0; i < kernel->dataSize(); ++i)
	{
		kernel->_data[i]/=sum;
	}
}

template<class Engine>
void fuseKernels(g2s::DataImage *kernelO, g2s::DataImage *kernelI1, g2s::DataImage *kernelI2, float alpha, Engine &randomGenerator){
	std::uniform_real_distribution<float> uniformDitribution(0.f,1.f);
	float sum=0.f;
	for (int i = 0; i < kernelO->dataSize(); ++i)
	{
		float val=uniformDitribution(randomGenerator);
		if(uniformDitribution(randomGenerator)>alpha) {
			kernelO->_data[i]=kernelI2->_data[i];
		}
		else {
			kernelO->_data[i]=kernelI1->_data[i];
		}
		sum+=kernelO->_data[i];
	}
	for (int i = 0; i < kernelO->dataSize(); ++i)
	{
		kernelO->_data[i]/=sum;
	}
}

template<class Engine>
void mutateKernels(g2s::DataImage *kernel, float alpha, Engine &randomGenerator){
	std::uniform_real_distribution<float> uniformDitribution(0.f,1.f);
	float sum=0.f;
	for (int i = 0; i < kernel->dataSize(); ++i)
	{
		float val=uniformDitribution(randomGenerator);
		if(uniformDitribution(randomGenerator)<alpha) {
			kernel->_data[i]=uniformDitribution(randomGenerator);
		}
		sum+=kernel->_data[i];
	}
	for (int i = 0; i < kernel->dataSize(); ++i)
	{
		kernel->_data[i]/=sum;
	}
}

void normalizeeKernels(g2s::DataImage *kernel){
	std::uniform_real_distribution<float> uniformDitribution(0.f,1.f);
	float sum=0.f;
	for (int i = 0; i < kernel->dataSize(); ++i)
	{
		sum+=kernel->_data[i];
	}
	if(sum==0.f)return;
	for (int i = 0; i < kernel->dataSize(); ++i)
	{
		kernel->_data[i]/=sum;
	}
}

struct geneAlgoConfig{
	unsigned populationSize;
	float survivingRate;
	float mutationRate;
	float heritageRate;
	float appearingRate;
	unsigned randomPatternPoolSize;
	unsigned maxIteration;
};


void geneAlgo(FILE *logFile, std::vector<g2s::DataImage> &TIs, g2s::DataImage &kernelSource, QuantileSamplingModule &samplingModule,
 std::vector<std::vector<int> > &pathPosition, unsigned seed, float numberOfCandidate, std::vector<std::vector<float> > categoriesValues, geneAlgoConfig config, const unsigned exclusionRadius, unsigned nbThreads=1){

	const unsigned populationSize=config.populationSize;
	const float survivingRate=config.survivingRate;
	const float mutationRate=config.mutationRate;
	const float heritageRate=config.heritageRate;
	const float appearingRate=config.appearingRate;
	const unsigned randomPatternPoolSize=config.randomPatternPoolSize;
	const unsigned maxIteration=config.maxIteration;

	std::mt19937 randomGenerator(seed);
	std::uniform_int_distribution<unsigned > uniformDitributionInPopulation(0,populationSize-1);
	std::uniform_int_distribution<unsigned > uniformDitributionInSurvivor(0,unsigned(ceil(populationSize*survivingRate))-1);
	std::uniform_real_distribution<float> uniformDitribution(0.f,1.f);

	g2s::DataImage* population=new g2s::DataImage[populationSize];
	//g2s::DataImage** population=(g2s::DataImage**)malloc(sizeof(g2s::DataImage*)*populationSize);
	float* kernelQuality=(float*)malloc(sizeof(float)*populationSize);
	unsigned* kernelsortedIndex=(unsigned*)malloc(sizeof(unsigned)*populationSize);

	// init population
	for (int i = 0; i < populationSize; ++i)
	{
		population[i]=g2s::DataImage(kernelSource.emptyCopy(false));
		newKernel(population+i, randomGenerator);
	}

	unsigned numberOfVariable=TIs[0]._nbVariable;
	for (int i = 0; i < categoriesValues.size(); ++i)
	{
		numberOfVariable+=categoriesValues[i].size()-1;
	}

	unsigned numberOfmodule=1;

	#if _OPENMP
		numberOfmodule=omp_get_max_threads();
	#endif

	std::mt19937* randomGenerators= new std::mt19937[numberOfmodule];
	for (int i = 0; i < numberOfmodule; ++i)
	{
		randomGenerators->seed(seed+i);
	}

	std::vector<std::vector<int> >* neighborArrayVectorList=new std::vector<std::vector<int> >[randomPatternPoolSize];
	std::vector<std::vector<float> >* neighborValueArrayVectorList=new std::vector<std::vector<float> >[randomPatternPoolSize];
	unsigned* positionList=new unsigned[randomPatternPoolSize];
	unsigned* TIList=new unsigned[randomPatternPoolSize];
	std::vector<unsigned>* exclusionList=new std::vector<unsigned>[randomPatternPoolSize];
	unsigned* sourceImage=new unsigned[randomPatternPoolSize];

	SamplingModule::matchLocation* result[nbThreads]; 


	for (int i = 0; i < nbThreads; ++i)
	{
		result[i]=(SamplingModule::matchLocation*)malloc(sizeof(SamplingModule::matchLocation)*int(ceil(numberOfCandidate)));
		for (int j = 0; j < int(ceil(numberOfCandidate)); ++j)
		{
			result[i][j].TI=0;
			result[i][j].index=0;
		}
	}

	std::vector<std::vector<float> >* categoriesValuesPtr=&categoriesValues;

	// run optim
	unsigned iteration=0;
	while(iteration<maxIteration){
		iteration++;
		if(iteration%10==1)
		{

			//generatePattern

			std::uniform_int_distribution<unsigned > uniformDitributionInImages(0,TIs.size()-1);

			#pragma omp parallel for num_threads(nbThreads) schedule(dynamic,1) default(none) firstprivate( \
			uniformDitributionInImages, uniformDitribution, pathPosition, numberOfVariable, categoriesValuesPtr, exclusionList,\
				sourceImage, randomGenerators, neighborValueArrayVectorList, neighborArrayVectorList, positionList, TIList) shared( TIs)
			for (int rpIndex = 0; rpIndex < randomPatternPoolSize; ++rpIndex)
			{
				unsigned moduleID=0;
				#if _OPENMP
					moduleID=omp_get_thread_num();
				#endif
				bool isOk=false;
				while (!isOk)
				{
					//fprintf(stderr, "randomPattern %d\n", rpIndex);
					std::vector<std::vector<int> > neighborArrayVector;
					std::vector<std::vector<float> > neighborValueArrayVector;

					unsigned randomImage=uniformDitributionInImages(randomGenerators[moduleID]);
					std::uniform_int_distribution<unsigned > uniformDitributionInTheImage(0,TIs[randomImage].dataSize()/TIs[randomImage]._nbVariable-1);
					unsigned randomPositionInside=uniformDitributionInTheImage(randomGenerators[moduleID]);

					unsigned bigestIndex=1;
					unsigned dataIndex;
					float proportion=uniformDitribution(randomGenerators[moduleID]);
					while((bigestIndex<pathPosition.size()) && (TIs[randomImage].indexWithDelta(dataIndex, randomPositionInside, pathPosition[bigestIndex])))
					{
						if(uniformDitribution(randomGenerators[moduleID])<proportion)
						{
							std::vector<float> data(numberOfVariable);
							unsigned numberOfNaN=0;
							float val;
							unsigned id=0;
							unsigned idCategorie=0;
							for (int i = 0; i < TIs[randomImage]._nbVariable; ++i)
							{
								if(TIs[randomImage]._types[i]==g2s::DataImage::Continuous){
									#pragma omp atomic read
									val=TIs[randomImage]._data[dataIndex*TIs[randomImage]._nbVariable+i];
									data[id]=val;
									id++;
								}
								if(TIs[randomImage]._types[i]==g2s::DataImage::Categorical){
									#pragma omp atomic read
									val=TIs[randomImage]._data[dataIndex*TIs[randomImage]._nbVariable+i];
									for (int k = 0; k < categoriesValuesPtr->at(idCategorie).size(); ++k)
									{
										if(val==categoriesValuesPtr->at(idCategorie)[k]){
											data[id]=1;
										}else{
											data[id]=0;
										}
										id++;
									}
									idCategorie++;
								}
							}
							neighborValueArrayVector.push_back(data);
							neighborArrayVector.push_back(pathPosition[bigestIndex]);
						}
						bigestIndex++;
					}

					if(neighborArrayVector.size()>1){
						//#pragma omp critical (updatdeRandomPattern)
						{
							neighborValueArrayVectorList[rpIndex]=(neighborValueArrayVector);
							neighborArrayVectorList[rpIndex]=(neighborArrayVector);
							positionList[rpIndex]=(randomPositionInside);
							TIList[rpIndex]=(randomImage);
							exclusionList[rpIndex].clear();

							for (int i = 0; i < int(floor(exclusionRadius*exclusionRadius*3.14159))-1 && i<pathPosition.size() ; ++i)
							{
								if(TIs[randomImage].indexWithDelta(dataIndex, randomPositionInside, pathPosition[i]))
									exclusionList[rpIndex].push_back(dataIndex);
							}
							sourceImage[rpIndex]=randomImage;
							isOk=true;
						}
					}
				}
			}
		}

		/////////// do measurement

		std::fill(kernelQuality,kernelQuality+populationSize,0.f);
		#pragma omp parallel num_threads(nbThreads) default(none) firstprivate( result, numberOfCandidate, neighborArrayVectorList, neighborValueArrayVectorList, population, exclusionList, sourceImage, kernelQuality) shared(TIs, samplingModule )
		for (int patternIndex = 0; patternIndex < randomPatternPoolSize; ++patternIndex)
		{
			//fprintf(stderr, "%d\n", sourceImage[patternIndex]);
			#pragma omp for schedule(dynamic,1) nowait
			for (int kernelIndex = 0; kernelIndex < populationSize; ++kernelIndex)
			{
				unsigned moduleID=0;
				#if _OPENMP
					moduleID=omp_get_thread_num();
				#endif

				float error=0.f;
				
				samplingModule.sample_complet(result[moduleID], neighborArrayVectorList[patternIndex], neighborValueArrayVectorList[patternIndex],
					moduleID, false, population+kernelIndex, exclusionList+patternIndex, sourceImage[patternIndex]);
		
				for (int i = 0; i < int(ceil(numberOfCandidate)); ++i)
				{
					float localError=0.f;
					for (int j = 0; j < TIs[result[moduleID][i].TI]._nbVariable; ++j)
					{
						float val=TIs[result[moduleID][i].TI]._data[TIs[result[moduleID][i].TI]._nbVariable*result[moduleID][i].index+j]-
								TIs[sourceImage[patternIndex]]._data[TIs[sourceImage[patternIndex]]._nbVariable*exclusionList[patternIndex][0]+j];
						if(TIs[0]._types[i]==0)
							localError+=val*val;
						if((TIs[0]._types[i]==1) && (val!=0))
							localError+=1.f;
					}
					error+=(1.f+std::min(0.f, numberOfCandidate-i)) * sqrt(localError);
				}

				#pragma omp atomic update
				kernelQuality[kernelIndex]+=error;
			}
		}

		////////// modifiy population
		// sort kernels
		std::iota(kernelsortedIndex, kernelsortedIndex+populationSize, 0);
		std::sort(kernelsortedIndex, kernelsortedIndex+populationSize,
			[kernelQuality](unsigned i1, unsigned i2) {return kernelQuality[i1] < kernelQuality[i2];});
		
		fprintf(logFile, "best %f \n",kernelQuality[kernelsortedIndex[0]] );

		memcpy(kernelSource._data,(population+kernelsortedIndex[0])->_data,kernelSource.dataSize()*sizeof(float));

		#pragma omp parallel for num_threads(nbThreads) schedule(dynamic,1) default(none) firstprivate( population, kernelsortedIndex, uniformDitributionInSurvivor,randomGenerators)
		for ( unsigned indexNew = unsigned(ceil(populationSize*survivingRate)); indexNew < populationSize-unsigned(ceil(populationSize*appearingRate)); ++indexNew)
		{
			unsigned moduleID=0;
			#if _OPENMP
				moduleID=omp_get_thread_num();
			#endif
			fuseKernels(population+kernelsortedIndex[indexNew], population+kernelsortedIndex[uniformDitributionInSurvivor(randomGenerators[moduleID])], population+kernelsortedIndex[uniformDitributionInSurvivor(randomGenerators[moduleID])], heritageRate, randomGenerators[moduleID]);
		}

		#pragma omp parallel for num_threads(nbThreads) schedule(dynamic,1) default(none) firstprivate(population, kernelsortedIndex,randomGenerators)
		for (unsigned indexNew= populationSize-unsigned(ceil(populationSize*appearingRate)); indexNew < populationSize; ++indexNew)
		{
			unsigned moduleID=0;
			#if _OPENMP
				moduleID=omp_get_thread_num();
			#endif
			newKernel(population+kernelsortedIndex[indexNew], randomGenerators[moduleID]);
		}

		#pragma omp parallel for num_threads(nbThreads) schedule(dynamic,1) default(none) firstprivate(population, uniformDitribution,randomGenerators)
		for (int i = 0; i < populationSize; ++i)
		{
			unsigned moduleID=0;
			#if _OPENMP
				moduleID=omp_get_thread_num();
			#endif
			if(uniformDitribution(randomGenerators[moduleID])<appearingRate){
				mutateKernels(population+i, mutationRate, randomGenerators[moduleID]);
			}
		}
		
		if(iteration%(maxIteration/100)==0)fprintf(logFile, "progress : %.2f%%\n",float(iteration)/maxIteration*100);
	}

	for (int i = 0; i < nbThreads; ++i)
	{
		free(result[i]);
	}
	
	delete[] population;
	free(kernelQuality);
	free(kernelsortedIndex);
	delete[] neighborArrayVectorList;
	delete[] neighborValueArrayVectorList;
	delete[] positionList;
	delete[] TIList;
	delete[] exclusionList;
	delete[] sourceImage;
	delete[] randomGenerators;
}


struct greedyAlgoConfig{
	float precision;
	unsigned randomPatternPoolSize;
	float initValue;
	float scalefactor;
	float maxValue;
};


void greedyAlgo(FILE *logFile, std::vector<g2s::DataImage> &TIs, g2s::DataImage &kernelSource, QuantileSamplingModule &samplingModule,
 std::vector<std::vector<int> > &pathPosition, unsigned seed, float numberOfCandidate, std::vector<std::vector<float> > categoriesValues, greedyAlgoConfig config, const unsigned exclusionRadius, unsigned nbThreads=1){

	const unsigned randomPatternPoolSize=config.randomPatternPoolSize;

	std::mt19937 randomGenerator(seed);
	std::uniform_real_distribution<float> uniformDitribution(0.f,1.f);

	unsigned nbFreedom=kernelSource.dataSize();

	unsigned* indexPathPtr=(unsigned *)malloc(nbFreedom*sizeof(unsigned));

	std::iota(indexPathPtr,indexPathPtr+nbFreedom,0);
	std::shuffle(indexPathPtr,indexPathPtr+nbFreedom, randomGenerator );

	unsigned numberOfVariable=TIs[0]._nbVariable;
	for (int i = 0; i < categoriesValues.size(); ++i)
	{
		numberOfVariable+=categoriesValues[i].size()-1;
	}

	unsigned numberOfmodule=1;
	#if _OPENMP
		numberOfmodule=omp_get_max_threads();
	#endif

	std::mt19937* randomGenerators= new std::mt19937[numberOfmodule];
	for (int i = 0; i < numberOfmodule; ++i)
	{
		randomGenerators->seed(seed+i);
	}

	std::vector<std::vector<int> >* neighborArrayVectorList=new std::vector<std::vector<int> >[randomPatternPoolSize];
	std::vector<std::vector<float> >* neighborValueArrayVectorList=new std::vector<std::vector<float> >[randomPatternPoolSize];
	unsigned* positionList=new unsigned[randomPatternPoolSize];
	unsigned* TIList=new unsigned[randomPatternPoolSize];
	std::vector<unsigned>* exclusionList=new std::vector<unsigned>[randomPatternPoolSize];
	unsigned* sourceImage=new unsigned[randomPatternPoolSize];

	std::vector<std::vector<float> >* categoriesValuesPtr=&categoriesValues;

	{
		//generatePattern

		std::uniform_int_distribution<unsigned > uniformDitributionInImages(0,TIs.size()-1);

		#pragma omp parallel for num_threads(nbThreads) schedule(dynamic,1) default(none) firstprivate( \
		uniformDitributionInImages, uniformDitribution, pathPosition, numberOfVariable, categoriesValuesPtr, exclusionList,\
			sourceImage, randomGenerators, neighborValueArrayVectorList, neighborArrayVectorList, positionList, TIList) shared( TIs)
		for (int rpIndex = 0; rpIndex < randomPatternPoolSize; ++rpIndex)
		{
			unsigned moduleID=0;
			#if _OPENMP
				moduleID=omp_get_thread_num();
			#endif
			bool isOk=false;
			while (!isOk)
			{
				//fprintf(stderr, "randomPattern %d\n", rpIndex);
				std::vector<std::vector<int> > neighborArrayVector;
				std::vector<std::vector<float> > neighborValueArrayVector;

				unsigned randomImage=uniformDitributionInImages(randomGenerators[moduleID]);
				std::uniform_int_distribution<unsigned > uniformDitributionInTheImage(0,TIs[randomImage].dataSize()/TIs[randomImage]._nbVariable-1);
				unsigned randomPositionInside=uniformDitributionInTheImage(randomGenerators[moduleID]);

				unsigned bigestIndex=1;
				unsigned dataIndex;
				float proportion=uniformDitribution(randomGenerators[moduleID]);
				while((bigestIndex<pathPosition.size()) && (TIs[randomImage].indexWithDelta(dataIndex, randomPositionInside, pathPosition[bigestIndex])))
				{
					if(uniformDitribution(randomGenerators[moduleID])<proportion)
					{
						std::vector<float> data(numberOfVariable);
						unsigned numberOfNaN=0;
						float val;
						unsigned id=0;
						unsigned idCategorie=0;
						for (int i = 0; i < TIs[randomImage]._nbVariable; ++i)
						{
							if(TIs[randomImage]._types[i]==g2s::DataImage::Continuous){
								#pragma omp atomic read
								val=TIs[randomImage]._data[dataIndex*TIs[randomImage]._nbVariable+i];
								data[id]=val;
								id++;
							}
							if(TIs[randomImage]._types[i]==g2s::DataImage::Categorical){
								#pragma omp atomic read
								val=TIs[randomImage]._data[dataIndex*TIs[randomImage]._nbVariable+i];
								for (int k = 0; k < categoriesValuesPtr->at(idCategorie).size(); ++k)
								{
									if(val==categoriesValuesPtr->at(idCategorie)[k]){
										data[id]=1;
									}else{
										data[id]=0;
									}
									id++;
								}
								idCategorie++;
							}
						}
						neighborValueArrayVector.push_back(data);
						neighborArrayVector.push_back(pathPosition[bigestIndex]);
					}
					bigestIndex++;
				}

				if(neighborArrayVector.size()>1){
					//#pragma omp critical (updatdeRandomPattern)
					{
						neighborValueArrayVectorList[rpIndex]=(neighborValueArrayVector);
						neighborArrayVectorList[rpIndex]=(neighborArrayVector);
						positionList[rpIndex]=(randomPositionInside);
						TIList[rpIndex]=(randomImage);
						exclusionList[rpIndex].clear();

						for (int i = 0; i < int(floor(exclusionRadius*exclusionRadius*3.14159))-1 && i<pathPosition.size() ; ++i)
						{
							if(TIs[randomImage].indexWithDelta(dataIndex, randomPositionInside, pathPosition[i]))
								exclusionList[rpIndex].push_back(dataIndex);
						}
						sourceImage[rpIndex]=randomImage;
						isOk=true;
					}
				}
			}
		}
	}

	//genearte array of possibel value

	float initValue=config.initValue;
	float scalefactor=config.scalefactor;
	float maxValue=config.maxValue;

	unsigned numberOfposibility=unsigned(floor(log(maxValue/initValue)/log(scalefactor))+1);
	float valueArray[numberOfposibility];
	valueArray[0]=0.f;
	int pos=1;
	valueArray[pos]=initValue;
	while(valueArray[pos]*scalefactor<maxValue){
		valueArray[pos+1]=valueArray[pos]*scalefactor;
		pos++;
	}

	const int populationSize=numberOfposibility;

	g2s::DataImage* population=new g2s::DataImage[populationSize];
	// init population
	for (int i = 0; i < populationSize; ++i)
	{
		population[i]=g2s::DataImage(kernelSource.emptyCopy(false));
	}

	SamplingModule::matchLocation* result[nbThreads]; 


	for (int i = 0; i < nbThreads; ++i)
	{
		result[i]=(SamplingModule::matchLocation*)malloc(sizeof(SamplingModule::matchLocation)*int(ceil(numberOfCandidate)));
		for (int j = 0; j < int(ceil(numberOfCandidate)); ++j)
		{
			result[i][j].TI=0;
			result[i][j].index=0;
		}
	}


	float* kernelQuality=(float*)malloc(sizeof(float)*populationSize);
	unsigned* kernelsortedIndex=(unsigned*)malloc(sizeof(unsigned)*populationSize);
	for (int indexInPath = 0; indexInPath < nbFreedom; ++indexInPath)
	{
		unsigned index=indexPathPtr[indexInPath];
		for (int modifIndex = 0; modifIndex < populationSize; ++modifIndex)
		{
			population[modifIndex]._data[index]=valueArray[modifIndex];
		}

		std::fill(kernelQuality,kernelQuality+populationSize,0.f);
		#pragma omp parallel num_threads(nbThreads) default(none) firstprivate( result, numberOfCandidate, neighborArrayVectorList, neighborValueArrayVectorList, population, exclusionList, sourceImage, kernelQuality) shared(TIs, samplingModule )
		for (int patternIndex = 0; patternIndex < randomPatternPoolSize; ++patternIndex)
		{
			//fprintf(stderr, "%d\n", sourceImage[patternIndex]);
			#pragma omp for schedule(dynamic,1) nowait
			for (int kernelIndex = 0; kernelIndex < populationSize; ++kernelIndex)
			{
				unsigned moduleID=0;
				#if _OPENMP
					moduleID=omp_get_thread_num();
				#endif

				float error=0.f;
				
				samplingModule.sample_complet(result[moduleID], neighborArrayVectorList[patternIndex], neighborValueArrayVectorList[patternIndex],
					moduleID, false, population+kernelIndex, exclusionList+patternIndex, sourceImage[patternIndex]);
		
				for (int i = 0; i < int(ceil(numberOfCandidate)); ++i)
				{
					float localError=0.f;
					for (int j = 0; j < TIs[result[moduleID][i].TI]._nbVariable; ++j)
					{
						float val=TIs[result[moduleID][i].TI]._data[TIs[result[moduleID][i].TI]._nbVariable*result[moduleID][i].index+j]-
								TIs[sourceImage[patternIndex]]._data[TIs[sourceImage[patternIndex]]._nbVariable*exclusionList[patternIndex][0]+j];
						if(TIs[0]._types[i]==0)
							localError+=val*val;
						if((TIs[0]._types[i]==1) && (val!=0))
							localError+=1.f;
					}
					error+=(1.f+std::min(0.f, numberOfCandidate-i)) * sqrt(localError);
				}

				#pragma omp atomic update
				kernelQuality[kernelIndex]+=error;
			}
		}

		//take the best smallest value

		int bigestIndex=0;
		float bestQuality=kernelQuality[0];

		for (int i = 1; i < populationSize; ++i)
		{
			if(bestQuality>kernelQuality[i]){
				bestQuality=kernelQuality[i];
				bigestIndex=i;
			}
		}

		for (int modifIndex = 0; modifIndex < populationSize; ++modifIndex)
		{
			population[modifIndex]._data[index]=valueArray[bigestIndex];
			normalizeeKernels(population+modifIndex);
		}

		if(indexInPath%(nbFreedom/100)==0)fprintf(logFile, "progress : %.2f%%\n",float(indexInPath)/nbFreedom*100);

	}

	memcpy(kernelSource._data,population[0]._data,kernelSource.dataSize()*sizeof(float));

	for (int i = 0; i < nbThreads; ++i)
	{
		free(result[i]);
	}

	delete[] population;
	delete[] neighborArrayVectorList;
	delete[] neighborValueArrayVectorList;
	delete[] positionList;
	delete[] TIList;
	delete[] exclusionList;
	delete[] sourceImage;
	delete[] randomGenerators;
	free(indexPathPtr);
	free(kernelQuality);
	free(kernelsortedIndex);
	
}

#endif // KERNEL_OPTIMIZATION_HPP
