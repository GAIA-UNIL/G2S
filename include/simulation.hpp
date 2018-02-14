#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "computeDeviceModule.hpp"
#include "samplingModule.hpp"
#include "fKst.hpp"
#include <thread>


void simulation(FILE *logFile,g2s::DataImage &di, std::vector<g2s::DataImage> &TIs, SamplingModule &samplingModule,
 std::vector<std::vector<int> > &pathPosition, unsigned* solvingPath, unsigned numberOfPointToSimulate, float* seedAray, unsigned* importDataIndex, unsigned numberNeighbor,
  std::vector<std::vector<float> > categoriesValues, unsigned nbThreads=1){

	unsigned* posterioryPath=(unsigned*)malloc( sizeof(unsigned) * di.dataSize()/di._nbVariable);
	memset(posterioryPath,255,sizeof(unsigned) * di.dataSize()/di._nbVariable);
	for (int i = 0; i < di.dataSize()/di._nbVariable; ++i)
	{
		bool withNan=false;
		for (int j = 0; j < di._nbVariable; ++j)
		{
			withNan|=std::isnan(di._data[i*di._nbVariable+j]);
		}
		if(!withNan)
			posterioryPath[i]=0;
	}
	for (int i = 0; i < numberOfPointToSimulate; ++i)
	{
		posterioryPath[solvingPath[i]]=i;
	}
	
	unsigned numberOfVariable=di._nbVariable;
	for (int i = 0; i < categoriesValues.size(); ++i)
	{
		numberOfVariable+=categoriesValues[i].size()-1;
	}
	#pragma omp parallel for num_threads(nbThreads) schedule(dynamic,1) default(none) firstprivate(numberOfVariable,categoriesValues,numberOfPointToSimulate,posterioryPath, solvingPath, seedAray, numberNeighbor, importDataIndex, logFile) shared( pathPosition, di, samplingModule, TIs)
	for (int indexPath = 0; indexPath < numberOfPointToSimulate; ++indexPath){
		
		/*if(indexPath<TIs[0].dataSize()/TIs[0]._nbVariable-1000){
			unsigned currentCell=solvingPath[indexPath];
			memcpy(di._data+currentCell*di._nbVariable,TIs[0]._data+currentCell*TIs[0]._nbVariable,TIs[0]._nbVariable*sizeof(float));
			continue;
		}*/

		unsigned moduleID=0;
		#if _OPENMP
			moduleID=omp_get_thread_num();
		#endif
		unsigned currentCell=solvingPath[indexPath];
		float localSeed=seedAray[indexPath];

		std::vector<std::vector<int> > neighborArrayVector;
		std::vector<std::vector<float> > neighborValueArrayVector;
		{
			unsigned positionSearch=0;
			while((neighborArrayVector.size()<numberNeighbor)&&(positionSearch<pathPosition.size())){
				unsigned dataIndex;
				if(di.indexWithDelta(dataIndex, currentCell, pathPosition[positionSearch]))
				{
					if(posterioryPath[dataIndex]<=indexPath){
						std::vector<float> data(numberOfVariable);
						unsigned numberOfNaN=0;
						float val;
						while(true) {
							numberOfNaN=0;
							unsigned id=0;
							unsigned idCategorie=0;
							for (int i = 0; i < di._nbVariable; ++i)
							{
								if(di._types[i]==g2s::DataImage::Continuous){
									#pragma omp atomic read
									val=di._data[dataIndex*di._nbVariable+i];
									numberOfNaN+=std::isnan(val);
									data[id]=val;
									id++;
								}
								if(di._types[i]==g2s::DataImage::Categorical){
									#pragma omp atomic read
									val=di._data[dataIndex*di._nbVariable+i];
									numberOfNaN+=std::isnan(val);
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
							if((numberOfNaN==0)||(posterioryPath[dataIndex]==indexPath))break;
							std::this_thread::sleep_for(std::chrono::microseconds(250));
						}
						neighborValueArrayVector.push_back(data);
						neighborArrayVector.push_back(pathPosition[positionSearch]);
					}
				}
				positionSearch++;
			}
		}

		/*for (int i = 0; i < neighborValueArrayVector.size(); ++i)
		{
			for (int j = 0; j < neighborValueArrayVector[i].size(); ++j)
			{
				fprintf(stderr, "%f\n", neighborValueArrayVector[i][j]);
			}
		}*/

		SamplingModule::matchLocation importIndex;

		if(neighborArrayVector.size()>0){
			importIndex=samplingModule.sample(neighborArrayVector,neighborValueArrayVector,localSeed,moduleID);
		}else{
			// sample from the marginal
			unsigned cumulated=0;
			for (int i = 0; i < TIs.size(); ++i)
			{
				cumulated+=TIs[i].dataSize();
			}
			
			unsigned position=int(floor(localSeed*(cumulated/TIs[0]._nbVariable)))*TIs[0]._nbVariable;

			cumulated=0;
			for (int i = 0; i < TIs.size(); ++i)
			{
				if(position<cumulated+TIs[i].dataSize()){
					importIndex.TI=i;
					importIndex.index=TIs[i]._data[position-cumulated];
					break;
				}else{
					cumulated+=TIs[i].dataSize();
				}
			}

			bool hasNaN=false;

			for (int j = 0; j < TIs[importIndex.TI]._nbVariable; ++j)
			{
				if(std::isnan(TIs[importIndex.TI]._data[importIndex.index*TIs[importIndex.TI]._nbVariable+j])){
					hasNaN=true;
				}
			}
		
			if(hasNaN){ // nan safe, much slower
				unsigned cumulated=0;
				for (int i = 0; i < TIs.size(); ++i)
				{
					for (int k = 0; k < TIs[i].dataSize()/TIs[i]._nbVariable; ++k)
					{
						bool locHasNan=false;
						for (int j = 0; j < TIs[i]._nbVariable; ++j)
						{
							locHasNan|=std::isnan(TIs[i]._data[k*TIs[i]._nbVariable+j]);
						}
						cumulated+=!locHasNan;
					}
				}
				unsigned position=int(floor(localSeed*(cumulated/TIs[0]._nbVariable)))*TIs[0]._nbVariable;

				cumulated=0;

				for (int i = 0; i < TIs.size(); ++i)
				{
					for (int k = 0; k < TIs[i].dataSize()/TIs[i]._nbVariable; ++k)
					{
						bool locHasNan=false;
						for (int j = 0; j < TIs[i]._nbVariable; ++j)
						{
							locHasNan|=std::isnan(TIs[i]._data[k*TIs[i]._nbVariable+j]);
						}
						cumulated+=!locHasNan;
						if(position>=cumulated){
							importIndex.TI=i;
							importIndex.index=k;
							break;
						}
					}
					if(position>=cumulated)break;
				}
			}
		}
		// import data
		//memcpy(di._data+currentCell*di._nbVariable,TIs[importIndex.TI]._data+importIndex.index*TIs[importIndex.TI]._nbVariable,TIs[importIndex.TI]._nbVariable*sizeof(float));
		for (int j = 0; j < TIs[importIndex.TI]._nbVariable; ++j)
		{
			if(std::isnan(di._data[currentCell*di._nbVariable+j])){
				#pragma omp atomic write
				di._data[currentCell*di._nbVariable+j]=TIs[importIndex.TI]._data[importIndex.index*TIs[importIndex.TI]._nbVariable+j];
			}
		}
		importDataIndex[currentCell]=importIndex.index*TIs.size()+importIndex.TI;
		if(indexPath%(numberOfPointToSimulate/100)==0)fprintf(logFile, "progress : %.2f%%\n",float(indexPath)/numberOfPointToSimulate*100);
	}

	free(posterioryPath);
}





void narrowPathSimulation(FILE *logFile,g2s::DataImage &di, g2s::DataImage &ni, std::vector<g2s::DataImage> &TIs, g2s::DataImage &kernel, SamplingModule &samplingModule,
 std::vector<std::vector<int> > &pathPosition, unsigned* solvingPath, float* seedAray, unsigned* importDataIndex, unsigned chunkSize, unsigned maxUpdate,
   float maxProgression=1.f, unsigned nbThreads=1){

	SamplingModule::matchLocation* candidates=(SamplingModule::matchLocation*) malloc( sizeof(SamplingModule::matchLocation) * ni.dataSize()/ni._nbVariable);
	float* narrownessArray=(float*) malloc( sizeof(float) * ni.dataSize()/ni._nbVariable);
	std::fill(narrownessArray,narrownessArray+ni.dataSize()/ni._nbVariable-1,std::nanf("0"));
	std::fill(solvingPath,solvingPath+di.dataSize()/di._nbVariable-1,UINT_MAX);

	std::vector<unsigned> placeToUpdate;
	placeToUpdate.reserve(di.dataSize()/di._nbVariable);

	for (unsigned i = 0; i < di.dataSize()/di._nbVariable; ++i)
	{
		bool hasNaN=false;
		for (int j = 0; j < di._nbVariable; ++j)
		{
			hasNaN|=std::isnan(di._data[i*di._nbVariable+j]);
		}
		if(!hasNaN){
			solvingPath[i]=0;
			ni._data[i]=0.f;
			narrownessArray[i]=0.f;
		}else{
			placeToUpdate.push_back(i);
		}
	}
	unsigned sizeSimulation=placeToUpdate.size();
	unsigned fullSize=sizeSimulation;
	unsigned indicationSize=100*nbThreads;

	unsigned solvingPathIndex=0;
	while((sizeSimulation>0) && ((float(sizeSimulation)/fullSize)>(1.f-maxProgression))){
	
		unsigned bunchSize=ceil(std::min(indicationSize,unsigned(placeToUpdate.size()))/float(nbThreads));
		//update all needed place to //
		#pragma omp parallel for schedule(dynamic,bunchSize) num_threads(nbThreads) default(none) firstprivate(logFile, placeToUpdate, bunchSize, seedAray, pathPosition, candidates, fullSize, narrownessArray) shared(di, samplingModule)
		for (int i = 0; i < placeToUpdate.size(); ++i)
		{
			unsigned moduleID=0;
			#if _OPENMP
				moduleID=omp_get_thread_num();
			#endif

			unsigned currentCell=placeToUpdate[i];
			float localSeed=seedAray[currentCell];

			std::vector<std::vector<int> > neighborArrayVector;
			std::vector<std::vector<float> > neighborValueArrayVector;
			{
				unsigned positionSearch=0;
				while((positionSearch<pathPosition.size())){
					unsigned dataIndex;
					if(di.indexWithDelta(dataIndex, currentCell, pathPosition[positionSearch]))
					{
						std::vector<float> data(di._nbVariable);
						for (int i = 0; i < di._nbVariable; ++i)
						{
							data[i]=di._data[dataIndex*di._nbVariable+i];
						}
						neighborValueArrayVector.push_back(data);
						neighborArrayVector.push_back(pathPosition[positionSearch]);
						
					}
					positionSearch++;
				}
			}
			if(neighborValueArrayVector.empty()) continue;

			SamplingModule::narrownessMeasurment currentNarrowness;
			currentNarrowness=samplingModule.narrowness(neighborArrayVector,neighborValueArrayVector,localSeed,moduleID);
			narrownessArray[currentCell]=currentNarrowness.narrowness;
			candidates[currentCell]=currentNarrowness.candidate;
			if((i+1)%(fullSize/100)==0)fprintf(logFile, "progress init: %.2f%%\n",float(i)/fullSize*100);
		
		}
		placeToUpdate.clear();

		unsigned bestPlaces[chunkSize];
		float usedNarrowness[chunkSize];

		unsigned maxAutorisedChunkSize=std::min(chunkSize,sizeSimulation);

		fKst::findKSmallest<float>(narrownessArray, ni.dataSize()/ni._nbVariable, maxAutorisedChunkSize, usedNarrowness, bestPlaces);
		solvingPathIndex++;
		for (int i = 0; i < maxAutorisedChunkSize; ++i)
		{
			unsigned simulatedPlace=bestPlaces[i];
			if(/*simulatedPlace<0 ||*/ simulatedPlace>(ni.dataSize()/ni._nbVariable) || std::isnan(usedNarrowness[i])) continue;

			SamplingModule::matchLocation importIndex=candidates[simulatedPlace];
			solvingPath[simulatedPlace]=solvingPathIndex;
			ni._data[simulatedPlace]=narrownessArray[simulatedPlace];
			narrownessArray[simulatedPlace]=INFINITY;
			for (int j = 0; j < TIs[importIndex.TI]._nbVariable; ++j)
			{
				if(std::isnan(di._data[simulatedPlace*di._nbVariable+j])){
					di._data[simulatedPlace*di._nbVariable+j]=TIs[importIndex.TI]._data[importIndex.index*TIs[importIndex.TI]._nbVariable+j];
				}
			}
			importDataIndex[simulatedPlace]=importIndex.index*TIs.size()+importIndex.TI;
		}

		for (int i = 0; i < maxAutorisedChunkSize; ++i)
		{
			unsigned simulatingPlace=bestPlaces[i];
			unsigned dataIndex;
			for (int j = 0; j < std::min(unsigned(pathPosition.size()),maxUpdate); ++j)
			{
				if(di.indexWithDelta(dataIndex, simulatingPlace, pathPosition[j])){
					if(solvingPath[dataIndex]>solvingPathIndex) placeToUpdate.push_back(dataIndex);
				}
			}
		}
		sizeSimulation-=maxAutorisedChunkSize;
		std::sort(placeToUpdate.begin(), placeToUpdate.end());
		auto last = std::unique(placeToUpdate.begin(), placeToUpdate.end());
		placeToUpdate.erase(last, placeToUpdate.end()); 
		if((sizeSimulation)%(fullSize/100)==0)fprintf(logFile, "progress : %.2f%%\n",float(fullSize-sizeSimulation)/fullSize*100);

	}

	for (int i = 0; i < di.dataSize()/di._nbVariable; ++i)
	{
		if(!std::isnan(narrownessArray[i]) && (solvingPath[i]>solvingPathIndex)){

			unsigned simulatedPlace=i;
			if(/*simulatedPlace<0 || */simulatedPlace>(ni.dataSize()/ni._nbVariable) || std::isnan(narrownessArray[i])) continue;
			solvingPath[simulatedPlace]=solvingPathIndex;

			SamplingModule::matchLocation importIndex=candidates[simulatedPlace];
			ni._data[simulatedPlace]=narrownessArray[simulatedPlace];
			narrownessArray[simulatedPlace]=INFINITY;
			for (int j = 0; j < TIs[importIndex.TI]._nbVariable; ++j)
			{
				if(std::isnan(di._data[simulatedPlace*di._nbVariable+j])){
					di._data[simulatedPlace*di._nbVariable+j]=TIs[importIndex.TI]._data[importIndex.index*TIs[importIndex.TI]._nbVariable+j];
				}
			}
			importDataIndex[simulatedPlace]=importIndex.index*TIs.size()+importIndex.TI;
		}
	}

	free(candidates);
	free(narrownessArray);
}

#endif // SIMULATION_HPP