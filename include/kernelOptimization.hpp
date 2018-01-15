#ifndef KERNEL_OPTIMIZATION_HPP
#define KERNEL_OPTIMIZATION_HPP

#include "computeDeviceModule.hpp"
#include "QuantileSamplingModule.hpp"
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


void optimization(FILE *logFile, std::vector<g2s::DataImage> &TIs, g2s::DataImage &kernelSource, QuantileSamplingModule &samplingModule,
 std::vector<std::vector<int> > &pathPosition, unsigned seed, float numberOfCandidate, std::vector<std::vector<float> > categoriesValues, unsigned nbThreads=1){

	const unsigned populationSize=200;
	const float survivingRate=0.5f;
	const float mutationRate=0.1f;
	const float heritageRate=0.5f;
	const float appearingRate=0.02f;
	const unsigned randomPatternPoolSize=500;
	const float k=2.f;
	const unsigned exclusionRadius=10;
	const unsigned maxIteration=100;

	std::mt19937 randomGenerator(seed);
	std::uniform_int_distribution<unsigned > uniformDitributionInPopulation(0,populationSize-1);
	std::uniform_int_distribution<unsigned > uniformDitributionInSurvivor(0,unsigned(ceil(populationSize*survivingRate))-1);
	std::uniform_real_distribution<float> uniformDitribution(0.f,1.f);

	g2s::DataImage** population=(g2s::DataImage**)malloc(sizeof(g2s::DataImage*)*populationSize);
	float* kernelQuality=(float*)malloc(sizeof(float)*populationSize);
	unsigned* kernelsortedIndex=(unsigned*)malloc(sizeof(unsigned)*populationSize);

	// init population
	for (int i = 0; i < populationSize; ++i)
	{
		population[i]= new g2s::DataImage(kernelSource.emptyCopy(false));
	}

	unsigned numberOfVariable=TIs[0]._nbVariable;
	for (int i = 0; i < categoriesValues.size(); ++i)
	{
		numberOfVariable+=categoriesValues[i].size()-1;
	}


	std::vector<std::vector<std::vector<int> > > neighborArrayVectorList;
	std::vector<std::vector<std::vector<float> > > neighborValueArrayVectorList;
	std::vector<unsigned> positionList;
	std::vector<unsigned> TIList;
	std::vector<unsigned> exclusionList[randomPatternPoolSize];
	unsigned sourceImage[randomPatternPoolSize];

	// run optim
	unsigned iteration=0;
	while(iteration<maxIteration){
		iteration++;
		if(iteration%100==1)
		{
			neighborArrayVectorList.clear();
			neighborArrayVectorList.resize(randomPatternPoolSize);
			neighborValueArrayVectorList.clear();
			neighborValueArrayVectorList.resize(randomPatternPoolSize);
			positionList.clear();
			positionList.resize(randomPatternPoolSize);
			TIList.clear();
			TIList.resize(randomPatternPoolSize);

			//generatePattern

			std::uniform_int_distribution<unsigned > uniformDitributionInImages(0,TIs.size()-1);

			#pragma omp parallel for num_threads(nbThreads) schedule(dynamic,1) default(none) firstprivate(__stderrp, uniformDitributionInImages,\
			 uniformDitribution, pathPosition, numberOfVariable, categoriesValues,neighborValueArrayVectorList, neighborArrayVectorList, positionList, TIList ) shared(exclusionList, sourceImage, TIs, randomGenerator)
			for (int rpIndex = 0; rpIndex < randomPatternPoolSize; ++rpIndex)
			{
				//fprintf(stderr, "randomPattern %d\n", rpIndex);
				std::vector<std::vector<int> > neighborArrayVector;
				std::vector<std::vector<float> > neighborValueArrayVector;

				unsigned randomImage=uniformDitributionInImages(randomGenerator);
				std::uniform_int_distribution<unsigned > uniformDitributionInTheImage(0,TIs[randomImage].dataSize()/TIs[randomImage]._nbVariable-1);
				unsigned randomPositionInside=uniformDitributionInTheImage(randomGenerator);

				unsigned bigestIndex=1;
				unsigned dataIndex;
				float proportion=uniformDitribution(randomGenerator);
				while((bigestIndex<pathPosition.size()) && (TIs[randomImage].indexWithDelta(dataIndex, randomPositionInside, pathPosition[bigestIndex])))
				{
					if(uniformDitribution(randomGenerator)<proportion)
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
								for (int k = 0; k < categoriesValues[idCategorie].size(); ++k)
								{
									if(val==categoriesValues[idCategorie][k]){
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

						for (int i = 0; i < int(floor(exclusionRadius*exclusionRadius*3.14159))-1; ++i)
						{
							if(TIs[randomImage].indexWithDelta(dataIndex, randomPositionInside, pathPosition[i]))
								exclusionList[rpIndex].push_back(dataIndex);
						}
						sourceImage[rpIndex]=randomImage;
					}
				}else{
					rpIndex--;
				}
			}
		}

		/////////// do measurement

		std::fill(kernelQuality,kernelQuality+populationSize,0.f);
		#pragma omp parallel num_threads(nbThreads) default(none) firstprivate( __stderrp,numberOfCandidate, neighborArrayVectorList, neighborValueArrayVectorList, population, exclusionList, sourceImage, kernelQuality) shared(TIs, samplingModule)
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
				SamplingModule::matchLocation result[int(ceil(numberOfCandidate))];
				samplingModule.sample_complet(result, neighborArrayVectorList[patternIndex], neighborValueArrayVectorList[patternIndex],
					moduleID=0, false, population[kernelIndex], exclusionList+patternIndex, sourceImage[patternIndex]);

				for (int i = 0; i < int(ceil(numberOfCandidate)); ++i)
				{
					float localError=0.f;
					for (int j = 0; j < TIs[result[i].TI]._nbVariable; ++j)
					{
						float val=TIs[result[i].TI]._data[TIs[result[i].TI]._nbVariable*result[i].index+j]-
								TIs[sourceImage[patternIndex]]._data[TIs[sourceImage[patternIndex]]._nbVariable*exclusionList[patternIndex][0]+j];
						localError+=val*val;
					}
					error+=(i-numberOfCandidate) * sqrt(localError);
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
		
		memcpy(kernelSource._data,population[kernelsortedIndex[0]]->_data,kernelSource.dataSize()*sizeof(float));

		unsigned indexNew=0;
		for ( indexNew = unsigned(ceil(populationSize*survivingRate)); indexNew < populationSize-unsigned(ceil(populationSize*appearingRate)); ++indexNew)
		{
			fuseKernels(population[kernelsortedIndex[indexNew]], population[kernelsortedIndex[uniformDitributionInSurvivor(randomGenerator)]], population[kernelsortedIndex[uniformDitributionInSurvivor(randomGenerator)]], heritageRate, randomGenerator);
		}

		for (; indexNew < populationSize; ++indexNew)
		{
			newKernel(population[kernelsortedIndex[indexNew]], randomGenerator);
		}

		for (int i = 0; i < populationSize; ++i)
		{
			if(uniformDitribution(randomGenerator)<appearingRate){
				mutateKernels(population[i], mutationRate, randomGenerator);
			}
		}
		
		if(iteration%(maxIteration/100)==0)fprintf(logFile, "progress : %.2f%%\n",float(iteration)/maxIteration*100);
	}

	for (int i = 0; i < populationSize; ++i)
	{
		delete population[i];
	}
	free(population);
	free(kernelQuality);
}

#endif // KERNEL_OPTIMIZATION_HPP