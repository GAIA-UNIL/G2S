#ifndef ANCHOR_SIMULATION_HPP
#define ANCHOR_SIMULATION_HPP

#include <thread>

#include "anchorSamplingModule.hpp"
#include "pathIndexType.hpp"
#include "qsPaddingUtils.hpp"
#include "simulationUpdateCallback.hpp"

inline unsigned anchorSimulationExpandedVariableCount(g2s::DataImage &di, std::vector<std::vector<float> > categoriesValues){
	unsigned numberOfVariable=di._nbVariable;
	for (unsigned int i = 0; i < categoriesValues.size(); ++i)
	{
		numberOfVariable+=categoriesValues[i].size()-1;
	}
	return numberOfVariable;
}

inline void anchorSimulationConvertNeighbors(
	g2s::DataImage &di,
	std::vector<std::vector<float> > &neighborValueArrayVector,
	std::vector<std::vector<float> > categoriesValues,
	unsigned numberOfVariable)
{
	for (size_t j = 0; j < neighborValueArrayVector.size(); ++j)
	{
		std::vector<float> data(numberOfVariable);
		unsigned id=0;
		unsigned idCategorie=0;
		for (unsigned int i = 0; i < di._nbVariable; ++i)
		{
			if(di._types[i]==g2s::DataImage::Continuous){
				data[id]=neighborValueArrayVector[j][i];
				id++;
			}
			if(di._types[i]==g2s::DataImage::Categorical){
				for (size_t k = 0; k < categoriesValues[idCategorie].size(); ++k)
				{
					data[id]=(neighborValueArrayVector[j][i]==categoriesValues[idCategorie][k]);
					id++;
				}
				idCategorie++;
			}
		}
		neighborValueArrayVector[j]=data;
	}
}

inline unsigned anchorSimulationOriginalIndex(
	unsigned paddedCell,
	const std::vector<unsigned> &paddedDims,
	const std::vector<unsigned> &padding,
	const std::vector<unsigned> &outputDims)
{
	std::vector<unsigned> coord=qs_padding_utils::indexToCoord(paddedCell,paddedDims);
	for (size_t i = 0; i < coord.size(); ++i)
	{
		coord[i]-=padding[i];
	}
	return qs_padding_utils::coordToIndex(coord,outputDims);
}

inline void simulationAS(FILE *logFile,g2s::DataImage &di, std::vector<g2s::DataImage> &TIs, std::vector<g2s::DataImage> &kernels, AnchorSamplingModule &samplingModule,
 std::vector<std::vector<std::vector<int> > > &pathPositionArray, g2s_path_index_t* solvingPath, g2s_path_index_t numberOfPointToSimulate, g2s::DataImage *kii, float* seedAray, unsigned* importDataIndex, std::vector<unsigned> numberNeighbor, g2s::DataImage *nii, g2s::DataImage *kvi,
  std::vector<std::vector<float> > categoriesValues, std::vector<unsigned> outputDims, std::vector<unsigned> spatialPadding, bool usePaddedDomain, unsigned nbThreads=1, bool fullStationary=false, bool circularSim=false, bool forceSimulation=false, bool kernelAutoSelection=false, g2s_path_index_t* inputPosterioryPath=nullptr,
  g2s_simulation_update_callback_t updateCallback=nullptr, void* updateCallbackUserData=nullptr){

	g2s_path_index_t displayRatio=std::max(numberOfPointToSimulate/g2s_path_index_t(100),g2s_path_index_t(1));
	bool localPosterioryPathAllocated=false;
	g2s_path_index_t* posterioryPath=inputPosterioryPath;
	if(posterioryPath==nullptr){
		localPosterioryPathAllocated=true;
		posterioryPath=(g2s_path_index_t*)malloc(sizeof(g2s_path_index_t)*di.dataSize()/di._nbVariable);
		memset(posterioryPath,255,sizeof(g2s_path_index_t)*di.dataSize()/di._nbVariable);
		for (unsigned int i = 0; i < di.dataSize()/di._nbVariable; ++i)
		{
			bool withNan=false;
			for (unsigned int j = 0; j < di._nbVariable; ++j)
			{
				withNan|=std::isnan(di._data[i*di._nbVariable+j]);
			}
			if(!withNan){
				posterioryPath[i]=0;
			}
		}
		for (g2s_path_index_t i = 0; i < numberOfPointToSimulate; ++i)
		{
			posterioryPath[solvingPath[i]]=i;
		}
	}
	const bool useExternalPosteriorPath=!localPosterioryPathAllocated;
	const unsigned numberOfVariable=anchorSimulationExpandedVariableCount(di,categoriesValues);

	int* externalMemory4IndexComputation[nbThreads];
	for (unsigned i = 0; i < nbThreads; ++i)
	{
		externalMemory4IndexComputation[i]=new int[di._dims.size()];
	}

	#pragma omp parallel for num_threads(nbThreads) schedule(monotonic:dynamic,1) default(none) firstprivate(kernelAutoSelection,forceSimulation, kvi, nii, kii, displayRatio, circularSim, fullStationary, numberOfVariable, categoriesValues, numberOfPointToSimulate, posterioryPath, solvingPath, seedAray, numberNeighbor, importDataIndex, logFile, externalMemory4IndexComputation, useExternalPosteriorPath, updateCallback, updateCallbackUserData, outputDims, spatialPadding, usePaddedDomain) shared(pathPositionArray, di, samplingModule, TIs, kernels)
	for (g2s_path_index_t indexPath = 0; indexPath < numberOfPointToSimulate; ++indexPath)
	{
		unsigned moduleID=0;
		#if _OPENMP
			moduleID=omp_get_thread_num();
		#endif

		int* localExternalMemory4IndexComputation=externalMemory4IndexComputation[moduleID];
		const unsigned currentCell=solvingPath[indexPath];
		const float localSeed=seedAray[indexPath];
		const g2s_path_index_t currentPathOrder=useExternalPosteriorPath ? posterioryPath[currentCell] : indexPath;

		bool withOnlyData=true;
		for (unsigned int i = 0; i < di._nbVariable; ++i)
		{
			withOnlyData&=!std::isnan(di._data[currentCell*di._nbVariable+i]);
		}
		if(withOnlyData && !forceSimulation){
			continue;
		}

		if(nii){
			numberNeighbor.clear();
			for (int i = 0; i < nii->_nbVariable; ++i)
			{
				numberNeighbor.push_back(int(nii->_data[currentCell*nii->_nbVariable+i]));
			}
		}

		int kernelImageIndex=-1;
		std::vector<std::vector<int> > *pathPosition=&pathPositionArray[0];
		if(kii){
			kernelImageIndex=int(kii->_data[currentCell*kii->_nbVariable+0]);
			pathPosition=&pathPositionArray[kernelImageIndex];
		}

		float localk=0.f;
		if(kvi){
			localk=kvi->_data[currentCell*kvi->_nbVariable+0];
		}

		std::vector<unsigned> numberOfNeighborsProVariable(di._nbVariable);
		std::vector<std::vector<int> > neighborArrayVector;
		neighborArrayVector.reserve(numberNeighbor.empty() ? 0 : numberNeighbor[0]);
		std::vector<std::vector<float> > neighborValueArrayVector;
		neighborValueArrayVector.reserve(numberNeighbor.empty() ? 0 : numberNeighbor[0]);

		bool needMoreNeighbours=false;
		for (unsigned l = 0; l < di._nbVariable; ++l)
		{
			needMoreNeighbours|=numberOfNeighborsProVariable[l]<numberNeighbor[l%numberNeighbor.size()];
		}

		if(kernelAutoSelection && pathPositionArray.size()>1){
			unsigned bestKi=0;
			unsigned bestPositionSearch=pathPositionArray[0].size()+1;
			unsigned bestAmount=0;
			unsigned sameQuality=0;

			std::mt19937 generator;
			generator.seed(localSeed*2);

			for (unsigned kernelIndex = 0; kernelIndex < pathPositionArray.size(); ++kernelIndex)
			{
				unsigned numberOfNeighboursForThisKernel=0;
				unsigned positionSearch=0;
				bool localNeedMoreNeighbours=needMoreNeighbours;
				std::vector<unsigned> numberOfNeighborsProVariableLocal(numberOfNeighborsProVariable);
				while((numberNeighbor.size()>1 || localNeedMoreNeighbours) && (positionSearch<pathPositionArray[kernelIndex].size())){
					unsigned dataIndex=0;
					std::vector<int> *vectorInDi=&pathPositionArray[kernelIndex][positionSearch];
					vectorInDi->resize(di._dims.size(),0);
					if(di.indexWithDelta(dataIndex,currentCell,*vectorInDi,localExternalMemory4IndexComputation) || circularSim)
					{
						if(posterioryPath[dataIndex]<=currentPathOrder){
							numberOfNeighboursForThisKernel++;
							unsigned cpt=0;
							for (unsigned i = 0; i < di._nbVariable; ++i)
							{
								if(numberOfNeighborsProVariableLocal[i]<numberNeighbor[i%numberNeighbor.size()]){
									cpt++;
									numberOfNeighborsProVariableLocal[i]++;
									localNeedMoreNeighbours=false;
									for (unsigned l = 0; l < di._nbVariable; ++l)
									{
										localNeedMoreNeighbours|=numberOfNeighborsProVariableLocal[l]<numberNeighbor[l%numberNeighbor.size()];
									}
								}
							}
							if(cpt==0){
								break;
							}
						}
					}
					positionSearch++;
				}

				int needToChange=-1;
				if(bestPositionSearch>=positionSearch){
					if(bestPositionSearch>positionSearch){
						needToChange=1;
					}else if(numberOfNeighboursForThisKernel>=bestAmount){
						needToChange=(numberOfNeighboursForThisKernel>bestAmount);
					}
				}

				if(needToChange==1){
					bestKi=kernelIndex;
					sameQuality=1;
					bestPositionSearch=positionSearch;
					bestAmount=numberOfNeighboursForThisKernel;
				}
				if(needToChange==0){
					if(std::uniform_real_distribution<float>(0.0,1.0)(generator)<(1.f/++sameQuality)){
						bestKi=kernelIndex;
					}
				}
			}
			kernelImageIndex=int(bestKi);
			pathPosition=&pathPositionArray[kernelImageIndex];
		}

		unsigned positionSearch=0;
		while((numberNeighbor.size()>1 || needMoreNeighbours) && (positionSearch<pathPosition->size())){
			unsigned dataIndex=0;
			std::vector<int> *vectorInDi=&(*pathPosition)[positionSearch];
			vectorInDi->resize(di._dims.size(),0);
			if(di.indexWithDelta(dataIndex,currentCell,*vectorInDi,localExternalMemory4IndexComputation) || circularSim)
			{
				if(posterioryPath[dataIndex]<=currentPathOrder){
					unsigned numberOfNaN=0;
					float val=std::nanf("0");
					while(true){
						numberOfNaN=0;
						for (unsigned i = 0; i < di._nbVariable; ++i)
						{
							#pragma omp atomic read
							val=di._data[dataIndex*di._nbVariable+i];
							numberOfNaN+=(numberOfNeighborsProVariable[i]<numberNeighbor[i%numberNeighbor.size()]) && std::isnan(val);
						}
						if(numberOfNaN==0 || posterioryPath[dataIndex]==currentPathOrder){
							break;
						}
						std::this_thread::sleep_for(std::chrono::microseconds(250));
					}

					std::vector<float> data(di._nbVariable);
					unsigned cpt=0;
					for (unsigned i = 0; i < di._nbVariable; ++i)
					{
						if(numberOfNeighborsProVariable[i]<numberNeighbor[i%numberNeighbor.size()]){
							#pragma omp atomic read
							val=di._data[dataIndex*di._nbVariable+i];
							data[i]=val;
							cpt++;
							numberOfNeighborsProVariable[i]++;
							needMoreNeighbours=false;
							for (unsigned l = 0; l < di._nbVariable; ++l)
							{
								needMoreNeighbours|=numberOfNeighborsProVariable[l]<numberNeighbor[l%numberNeighbor.size()];
							}
						}else{
							data[i]=std::nanf("0");
						}
					}
					neighborValueArrayVector.push_back(data);
					neighborArrayVector.push_back((*pathPosition)[positionSearch]);
					if(cpt==0){
						break;
					}
				}
			}
			positionSearch++;
		}

		anchorSimulationConvertNeighbors(di,neighborValueArrayVector,categoriesValues,numberOfVariable);
		SamplingModule::matchLocation importIndex=samplingModule.sample(currentCell,neighborArrayVector,neighborValueArrayVector,localSeed,fullStationary,UINT_MAX,localk,(kernelImageIndex>-1 ? &(kernels[kernelImageIndex]) : nullptr));

		const unsigned originalIndex=usePaddedDomain ? anchorSimulationOriginalIndex(importIndex.index,di._dims,spatialPadding,outputDims) : importIndex.index;
		importDataIndex[currentCell]=originalIndex*TIs.size()+importIndex.TI;
		for (unsigned int j = 0; j < TIs[importIndex.TI]._nbVariable; ++j)
		{
			if(std::isnan(di._data[currentCell*di._nbVariable+j]) || forceSimulation){
				#pragma omp atomic write
				di._data[currentCell*di._nbVariable+j]=TIs[importIndex.TI]._data[importIndex.index*TIs[importIndex.TI]._nbVariable+j];
			}
		}

		if(updateCallback){
			updateCallback(g2s_simulation_update_kind::Vector, static_cast<g2s_path_index_t>(currentCell), 0, updateCallbackUserData);
		}
		if(indexPath%displayRatio==0){
			fprintf(logFile,"progress : %.2f%%\n",float(indexPath)/numberOfPointToSimulate*100);
		}
	}

	for (unsigned i = 0; i < nbThreads; ++i)
	{
		delete externalMemory4IndexComputation[i];
	}
	if(localPosterioryPathAllocated){
		free(posterioryPath);
	}
}

inline void simulationFullAS(FILE *logFile,g2s::DataImage &di, std::vector<g2s::DataImage> &TIs, std::vector<g2s::DataImage> &kernels, AnchorSamplingModule &samplingModule,
 std::vector<std::vector<std::vector<int> > > &pathPositionArray, g2s_path_index_t* solvingPath, g2s_path_index_t numberOfPointToSimulate, g2s::DataImage *kii, float* seedAray, unsigned* importDataIndex, std::vector<unsigned> numberNeighbor, g2s::DataImage *nii, g2s::DataImage *kvi,
  std::vector<std::vector<float> > categoriesValues, std::vector<unsigned> outputDims, std::vector<unsigned> spatialPadding, bool usePaddedDomain, unsigned nbThreads=1, bool fullStationary=false, bool circularSim=false, bool forceSimulation=false, g2s_path_index_t* inputPosterioryPath=nullptr,
  g2s_simulation_update_callback_t updateCallback=nullptr, void* updateCallbackUserData=nullptr){

	g2s_path_index_t displayRatio=std::max(numberOfPointToSimulate/g2s_path_index_t(100),g2s_path_index_t(1));
	bool localPosterioryPathAllocated=false;
	g2s_path_index_t* posterioryPath=inputPosterioryPath;
	if(posterioryPath==nullptr){
		localPosterioryPathAllocated=true;
		posterioryPath=(g2s_path_index_t*)malloc(sizeof(g2s_path_index_t)*di.dataSize());
		memset(posterioryPath,255,sizeof(g2s_path_index_t)*di.dataSize());
		for (unsigned int i = 0; i < di.dataSize(); ++i)
		{
			if(!std::isnan(di._data[i])){
				posterioryPath[i]=0;
			}
		}
		for (g2s_path_index_t i = 0; i < numberOfPointToSimulate; ++i)
		{
			posterioryPath[solvingPath[i]]=i;
		}
	}
	const bool useExternalPosteriorPath=!localPosterioryPathAllocated;
	const unsigned numberOfVariable=anchorSimulationExpandedVariableCount(di,categoriesValues);

	int* externalMemory4IndexComputation[nbThreads];
	for (unsigned i = 0; i < nbThreads; ++i)
	{
		externalMemory4IndexComputation[i]=new int[di._dims.size()];
	}

	#pragma omp parallel for num_threads(nbThreads) schedule(monotonic:dynamic,1) default(none) firstprivate(forceSimulation, kvi, nii, kii, displayRatio, circularSim, fullStationary, numberOfVariable, categoriesValues, numberOfPointToSimulate, posterioryPath, solvingPath, seedAray, numberNeighbor, importDataIndex, logFile, externalMemory4IndexComputation, useExternalPosteriorPath, updateCallback, updateCallbackUserData, outputDims, spatialPadding, usePaddedDomain) shared(pathPositionArray, di, samplingModule, TIs, kernels)
	for (g2s_path_index_t indexPath = 0; indexPath < numberOfPointToSimulate; ++indexPath)
	{
		unsigned moduleID=0;
		#if _OPENMP
			moduleID=omp_get_thread_num();
		#endif

		int* localExternalMemory4IndexComputation=externalMemory4IndexComputation[moduleID];
		const unsigned currentCell=solvingPath[indexPath];
		if(!std::isnan(di._data[currentCell]) && !forceSimulation){
			continue;
		}
		const float localSeed=seedAray[indexPath];
		const g2s_path_index_t currentPathOrder=useExternalPosteriorPath ? posterioryPath[currentCell] : indexPath;
		const unsigned currentVariable=currentCell%di._nbVariable;
		const unsigned currentPosition=currentCell/di._nbVariable;

		if(nii){
			numberNeighbor.clear();
			for (int i = 0; i < nii->_nbVariable; ++i)
			{
				numberNeighbor.push_back(int(nii->_data[currentPosition*nii->_nbVariable+i]));
			}
		}

		int kernelImageIndex=-1;
		std::vector<std::vector<int> > *pathPosition=&pathPositionArray[0];
		if(kii){
			kernelImageIndex=int(kii->_data[currentCell*kii->_nbVariable+0]);
			pathPosition=&pathPositionArray[kernelImageIndex];
		}

		float localk=0.f;
		if(kvi){
			localk=kvi->_data[currentCell*kvi->_nbVariable+currentVariable];
		}

		std::vector<unsigned> numberOfNeighborsProVariable(di._nbVariable);
		std::vector<std::vector<int> > neighborArrayVector;
		neighborArrayVector.reserve(numberNeighbor.empty() ? 0 : numberNeighbor[0]);
		std::vector<std::vector<float> > neighborValueArrayVector;
		neighborValueArrayVector.reserve(numberNeighbor.empty() ? 0 : numberNeighbor[0]);

		bool needMoreNeighbours=false;
		for (unsigned l = 0; l < di._nbVariable; ++l)
		{
			needMoreNeighbours|=numberOfNeighborsProVariable[l]<numberNeighbor[l%numberNeighbor.size()];
		}

		unsigned positionSearch=0;
		while((numberNeighbor.size()>1 || needMoreNeighbours) && (positionSearch<pathPosition->size())){
			unsigned dataIndex=0;
			std::vector<int> *vectorInDi=&(*pathPosition)[positionSearch];
			vectorInDi->resize(di._dims.size(),0);
			if(di.indexWithDelta(dataIndex,currentPosition,*vectorInDi,localExternalMemory4IndexComputation) || circularSim)
			{
				bool needToBeAdded=false;
				for (unsigned i = 0; i < di._nbVariable; ++i)
				{
					needToBeAdded|=(numberOfNeighborsProVariable[i]<numberNeighbor[i%numberNeighbor.size()]) && (posterioryPath[dataIndex*di._nbVariable+i]<currentPathOrder);
				}
				if(needToBeAdded){
					unsigned numberOfNaN=0;
					float val=std::nanf("0");
					while(true){
						numberOfNaN=0;
						for (unsigned i = 0; i < di._nbVariable; ++i)
						{
							#pragma omp atomic read
							val=di._data[dataIndex*di._nbVariable+i];
							numberOfNaN+=(numberOfNeighborsProVariable[i]<numberNeighbor[i%numberNeighbor.size()]) && (posterioryPath[dataIndex*di._nbVariable+i]<currentPathOrder) && std::isnan(val);
						}
						if(numberOfNaN==0){
							break;
						}
						std::this_thread::sleep_for(std::chrono::microseconds(250));
					}

					std::vector<float> data(di._nbVariable);
					unsigned cpt=0;
					for (unsigned i = 0; i < di._nbVariable; ++i)
					{
						if((numberOfNeighborsProVariable[i]<numberNeighbor[i%numberNeighbor.size()]) && (posterioryPath[dataIndex*di._nbVariable+i]<currentPathOrder)){
							#pragma omp atomic read
							val=di._data[dataIndex*di._nbVariable+i];
							data[i]=val;
							cpt++;
							numberOfNeighborsProVariable[i]++;
							needMoreNeighbours=false;
							for (unsigned l = 0; l < di._nbVariable; ++l)
							{
								needMoreNeighbours|=numberOfNeighborsProVariable[l]<numberNeighbor[l%numberNeighbor.size()];
							}
						}else{
							data[i]=std::nanf("0");
						}
					}
					neighborValueArrayVector.push_back(data);
					neighborArrayVector.push_back((*pathPosition)[positionSearch]);
					if(cpt==0){
						break;
					}
				}
			}
			positionSearch++;
		}

		anchorSimulationConvertNeighbors(di,neighborValueArrayVector,categoriesValues,numberOfVariable);
		SamplingModule::matchLocation importIndex=samplingModule.sample(currentPosition,neighborArrayVector,neighborValueArrayVector,localSeed,fullStationary,currentVariable,localk,(kernelImageIndex>-1 ? &(kernels[kernelImageIndex]) : nullptr));

		const unsigned originalIndex=usePaddedDomain ? anchorSimulationOriginalIndex(importIndex.index,di._dims,spatialPadding,outputDims) : importIndex.index;
		importDataIndex[currentCell]=originalIndex*TIs.size()+importIndex.TI;
		if(std::isnan(di._data[currentCell]) || forceSimulation){
			#pragma omp atomic write
			di._data[currentCell]=TIs[importIndex.TI]._data[importIndex.index*TIs[importIndex.TI]._nbVariable+currentVariable];
			if(updateCallback){
				updateCallback(g2s_simulation_update_kind::Full, static_cast<g2s_path_index_t>(currentCell), currentVariable, updateCallbackUserData);
			}
		}

		if(indexPath%displayRatio==0){
			fprintf(logFile,"progress : %.2f%%\n",float(indexPath)/numberOfPointToSimulate*100);
		}
	}

	for (unsigned i = 0; i < nbThreads; ++i)
	{
		delete externalMemory4IndexComputation[i];
	}
	if(localPosterioryPathAllocated){
		free(posterioryPath);
	}
}

#endif // ANCHOR_SIMULATION_HPP
