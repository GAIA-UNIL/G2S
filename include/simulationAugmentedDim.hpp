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

#ifndef SIMULATION_AUGMENTED_DIMENTIONALITY_HPP
#define SIMULATION_AUGMENTED_DIMENTIONALITY_HPP

#include "computeDeviceModule.hpp"
#include "samplingModule.hpp"
#include "quantileSamplingModule.hpp"
#include "fKst.hpp"
#include <thread>

unsigned nChoosek( unsigned n, unsigned k )
{
	if (k > n) return 0;
	if (k * 2 > n) /*return*/ k = n-k;  //remove the commented section
	if (k == 0) return 1;
 
	int result = n;
	for( int i = 2; i <= k; ++i ) {
		result *= (n-i+1);
		result /= i;
	}
	return result;
}


void simulationAD(FILE *logFile,g2s::DataImage &di, std::vector<g2s::DataImage> &TIs, QuantileSamplingModule &samplingModule,
 std::vector<std::vector<int> > &pathPosition, unsigned* solvingPath, unsigned numberOfPointToSimulate, g2s::DataImage *ii, float* seedAray, unsigned* importDataIndex, std::vector<unsigned> numberNeighbor,
  std::vector<std::vector<float> > categoriesValues, unsigned nbThreads=1, bool fullStationary=false, bool circularSim=false){

	std::vector<std::vector<std::vector<unsigned> > > marginals;
	for (size_t i = 0; i < TIs.size(); ++i)
	{
		marginals.push_back(TIs[i].computeMagninals(categoriesValues));
	}


	unsigned* posterioryPath=(unsigned*)malloc( sizeof(unsigned) * di.dataSize()/di._nbVariable);
	memset(posterioryPath,255,sizeof(unsigned) * di.dataSize()/di._nbVariable);
	for (unsigned int i = 0; i < di.dataSize()/di._nbVariable; ++i)
	{
		bool withNan=false;
		for (unsigned int j = 0; j < di._nbVariable; ++j)
		{
			withNan|=std::isnan(di._data[i*di._nbVariable+j]);
		}
		if(!withNan)
			posterioryPath[i]=0;
	}
	for (unsigned int i = 0; i < numberOfPointToSimulate; ++i)
	{
		posterioryPath[solvingPath[i]]=i;
	}
	
	unsigned numberOfVariable=di._nbVariable;
	for (unsigned int i = 0; i < categoriesValues.size(); ++i)
	{
		numberOfVariable+=categoriesValues[i].size()-1;
	}
	#pragma omp parallel for num_threads(nbThreads) schedule(dynamic,1) default(none) firstprivate(circularSim, fullStationary, numberOfVariable,categoriesValues,numberOfPointToSimulate,posterioryPath, solvingPath, seedAray, numberNeighbor, importDataIndex, logFile, ii) shared( pathPosition, di, samplingModule, TIs)
	for (unsigned int indexPath = 0; indexPath < numberOfPointToSimulate; ++indexPath){
		
		unsigned moduleID=0;
		#if _OPENMP
			moduleID=omp_get_thread_num();
		#endif
		unsigned currentCell=solvingPath[indexPath];
		float localSeed=seedAray[indexPath];

		bool withDataInCenter=false;
		bool withOnlyData=true;

		for (unsigned int i = 0; i < di._nbVariable; ++i)
		{
			withDataInCenter|=!std::isnan(di._data[currentCell*di._nbVariable+i]);
			withOnlyData&=!std::isnan(di._data[currentCell*di._nbVariable+i]);
		}

		if(withOnlyData) continue;

		std::vector<int> combi(TIs[0]._dims.size(),1);
		combi.resize(di._dims.size());

		std::vector<std::vector<SamplingModule::matchLocation> > importIndexs4EachDim;

		for (int combinatoryIdx = 0; combinatoryIdx < nChoosek(di._dims.size(),TIs[0]._dims.size()); ++combinatoryIdx)
		{
			std::next_permutation(combi.begin(),combi.end(),[](const int &a, const int &b){return (a!=0)<(b!=0);});
			std::vector<unsigned> numberOfNeighborsProVariable(di._nbVariable);
			std::vector<std::vector<int> > neighborArrayVector;
			std::vector<std::vector<float> > neighborValueArrayVector;

			{
				unsigned positionSearch=0;
				while((numberNeighbor.size()>1||(neighborArrayVector.size()<numberNeighbor[0]))&&(positionSearch<pathPosition.size())){
					unsigned dataIndex;
					std::vector<int> vectorInDi=pathPosition[positionSearch];
					vectorInDi.resize(di._dims.size(),0);
					if(di.indexWithDelta(dataIndex, currentCell, vectorInDi, combi) || circularSim)
					{
						//add for
						if(posterioryPath[dataIndex]<=indexPath){
							unsigned numberOfNaN=0;
							float val;
							while(true) {
								numberOfNaN=0;
								for (unsigned int i = 0; i < di._nbVariable; ++i)
								{
									#pragma omp atomic read
									val=di._data[dataIndex*di._nbVariable+i];
									numberOfNaN+=(numberOfNeighborsProVariable[i]<numberNeighbor[i%numberNeighbor.size()]) && std::isnan(val);
								}
								if((numberOfNaN==0)||(posterioryPath[dataIndex]==indexPath))break;
								std::this_thread::sleep_for(std::chrono::microseconds(250));
							}

							std::vector<float> data(di._nbVariable);
							unsigned cpt=0;
							for (unsigned int i = 0; i < di._nbVariable; ++i)
							{
								if((numberOfNeighborsProVariable[i]<numberNeighbor[i%numberNeighbor.size()]))
								{
									#pragma omp atomic read
									val=di._data[dataIndex*di._nbVariable+i];
									data[i]=val;
									cpt++;
									numberOfNeighborsProVariable[i]++;
								}else{
									data[i]=std::nanf("0");
								}
							}
							neighborValueArrayVector.push_back(data);
							neighborArrayVector.push_back(pathPosition[positionSearch]);
							if(cpt==0) break;
						}
					}
					positionSearch++;
				}
			}
			// conversion from one variable to many
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
							data[id] = (neighborValueArrayVector[j][i] == categoriesValues[idCategorie][k]);
							id++;
						}
						idCategorie++;
					}
				}
				neighborValueArrayVector[j]=data;
			}

			std::vector<SamplingModule::matchLocation> importIndexs;

			if(neighborArrayVector.size()>1){
				unsigned dataIndex;
				std::vector<int> vectorInDi=neighborArrayVector[1];
				vectorInDi.resize(di._dims.size(),0);
				di.indexWithDelta(dataIndex, currentCell, vectorInDi);
				unsigned verbatimIndex=importDataIndex[dataIndex];
				SamplingModule::matchLocation verbatimRecord;
				verbatimRecord.TI=verbatimIndex%TIs.size();
				std::vector<int> reverseVector=neighborArrayVector[1];
				for (size_t i = 0; i < reverseVector.size(); ++i)
				{
					reverseVector[i]*=-1;
				}
				TIs[verbatimRecord.TI].indexWithDelta(verbatimRecord.index, verbatimIndex/TIs.size(), reverseVector);
				/*for (int i = 0; i < 10; ++i)
				{
					SamplingModule::matchLocation importIndex;
					importIndex.TI=combinatoryIdx % TIs[0]._dims.size();
					importIndex.index=25;
					importIndexs.push_back(importIndex);
				}*/
				importIndexs=samplingModule.distribution(neighborArrayVector,neighborValueArrayVector,localSeed,verbatimRecord,moduleID,fullStationary,0,combinatoryIdx % TIs[0]._dims.size());
			}else if(withDataInCenter){
				SamplingModule::matchLocation verbatimRecord;
				verbatimRecord.TI=TIs.size();
				importIndexs=samplingModule.distribution(neighborArrayVector,neighborValueArrayVector,localSeed,verbatimRecord,moduleID,fullStationary,0,combinatoryIdx % TIs[0]._dims.size());
			}
			importIndexs4EachDim.push_back(importIndexs);
		}

		// average over the dimentions

		//int sumElement=0;
		std::vector<float> avergaeOverDim(categoriesValues[0].size(),0);
		int numberOfDim=0;
		for (int i = 0; i <importIndexs4EachDim.size() ; ++i)
		{
			numberOfDim+=importIndexs4EachDim[i].size()>0;
			for (int j = 0; j < importIndexs4EachDim[i].size(); ++j)
			{
				SamplingModule::matchLocation importIndex=importIndexs4EachDim[i][j];
				float val=TIs[importIndex.TI]._data[importIndex.index*TIs[importIndex.TI]._nbVariable+j];
				
				for (int k = 0; k < categoriesValues[0].size(); ++k)
				{
					avergaeOverDim[k]+=(val==categoriesValues[0][k]);
				}
				//sumElement+=1;
			}
		}

		int classIndex=0;
		if(numberOfDim>0){
			std::transform(avergaeOverDim.begin(), avergaeOverDim.end(), avergaeOverDim.begin(),
				std::bind2nd(std::divides<float>(), std::accumulate(avergaeOverDim.begin(), avergaeOverDim.end(),0.f)));

			/*for (int j = 0; j < avergaeOverDim.size(); ++j)
			{
				avergaeOverDim[j]/=sumElement;
			}*/

			//Sample

			float cumulate=0.f;
			for (int i = 0; i < avergaeOverDim.size(); ++i)
			{
				cumulate+=avergaeOverDim[i];
				classIndex+=(cumulate<localSeed);
			}

		}else{
			classIndex=0;
		}

		for (unsigned int j = 0; j < 1 /*TIs[importIndex.TI]._nbVariable*/; ++j)
		{
			if(std::isnan(di._data[currentCell*di._nbVariable+j])){
				#pragma omp atomic write
				di._data[currentCell*di._nbVariable+j]=categoriesValues[0][classIndex];
			}
		}

		//sqrt(prod(x/marginalLocal))*Globalmarginal

		/*// import data
		//memcpy(di._data+currentCell*di._nbVariable,TIs[importIndex.TI]._data+importIndex.index*TIs[importIndex.TI]._nbVariable,TIs[importIndex.TI]._nbVariable*sizeof(float));
		importDataIndex[currentCell]=importIndex.index*TIs.size()+importIndex.TI;
		//fprintf(stderr, "write %d\n", importDataIndex[currentCell]);
		for (unsigned int j = 0; j < TIs[importIndex.TI]._nbVariable; ++j)
		{
			if(std::isnan(di._data[currentCell*di._nbVariable+j])){
				#pragma omp atomic write
				di._data[currentCell*di._nbVariable+j]=TIs[importIndex.TI]._data[importIndex.index*TIs[importIndex.TI]._nbVariable+j];
			}
		}//*/
		if(indexPath%(numberOfPointToSimulate/100)==0)fprintf(logFile, "progress : %.2f%%\n",float(indexPath)/numberOfPointToSimulate*100);
	}

	free(posterioryPath);
}


#endif // SIMULATION_AUGMENTED_DIMENTIONALITY_HPP