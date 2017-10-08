#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "samplingModule.hpp"

//template<class randGen>
void simulation(FILE *logFile,g2s::DataImage &di, std::vector<g2s::DataImage> &TIs, SamplingModule &samplingModule,
 std::vector<std::vector<int> > &pathPosition, unsigned* solvingPath, unsigned numberOfPointToSimulate, float* seedAray, unsigned* importDataIndex, unsigned numberNeighbor,
   unsigned nbThreads=1){

   	unsigned* posterioryPath=(unsigned*)malloc( sizeof(unsigned) * di.dataSize()/di._nbVariable);
   	memset(posterioryPath,0,sizeof(unsigned) * di.dataSize()/di._nbVariable);
   	for (int i = 0; i < numberOfPointToSimulate; ++i)
   	{
   		posterioryPath[solvingPath[i]]=i;
   	}
	
   	unsigned numberOfVariable=di._nbVariable;
	#pragma omp parallel for num_threads(nbThreads) schedule(dynamic,1) default(none)
	for (int indexPath = 0; indexPath < numberOfPointToSimulate; ++indexPath){
		
		/*if(indexPath<TIs[0].dataSize()/TIs[0]._nbVariable-300){
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
					std::vector<float> data(di._nbVariable);
					memcpy(data.data(),di._data+dataIndex*di._nbVariable,di._nbVariable * sizeof(float));
					unsigned numberOfNaN=0;
					for (int i = 0; i < di._nbVariable; ++i)
					{
						numberOfNaN+=std::isnan(data[i]);
					}
					if(numberOfNaN==0){
						neighborValueArrayVector.push_back(data);
						neighborArrayVector.push_back(pathPosition[positionSearch]);
					}
				}
				positionSearch++;
			}
		}

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
		}
		// import data
		memcpy(di._data+currentCell*di._nbVariable,TIs[importIndex.TI]._data+importIndex.index*TIs[importIndex.TI]._nbVariable,TIs[importIndex.TI]._nbVariable*sizeof(float));
		importDataIndex[currentCell]=importIndex.index*TIs.size()+importIndex.TI;
		if(indexPath%(numberOfPointToSimulate/100)==0)fprintf(logFile, "progress : %.2f%%\n",float(indexPath)/numberOfPointToSimulate*100);
	}
}

#endif // SIMULATION_HPP