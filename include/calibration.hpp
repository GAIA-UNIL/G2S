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

#ifndef CALIBARTION_HPP
#define CALIBARTION_HPP

#include "computeDeviceModule.hpp"
#include "samplingModule.hpp"
#include "quantileSamplingModule.hpp"
#include "fKst.hpp"
#include <thread>
#include <random>


void calibration(FILE *logFile, g2s::DataImage &MeanErrorimage, g2s::DataImage &DevErrorimage, g2s::DataImage &NumberOFsampleimage, std::vector<g2s::DataImage> &TIs, std::vector<g2s::DataImage> &kernels,
 	QuantileSamplingModule &samplingModule, std::vector<std::vector<int> > &pathPosition, std::vector<std::vector<unsigned> > listNbNeihbours, std::vector<float> &densityArray, std::vector<std::vector<float> > categoriesValues, 
 	float power, unsigned nbThreads=1,unsigned maxNumberOfIteration=25000,unsigned minNumberOfIteration=1000, float maxT=INFINITY){

	int maxK=MeanErrorimage._types.size();
	float radius=20;
	
	bool circularSim=false;

	unsigned numberOfVariable=TIs[0]._nbVariable;
	for (unsigned int i = 0; i < categoriesValues.size(); ++i)
	{
		numberOfVariable+=categoriesValues[i].size()-1;
	}

	float* bestProDensity=(float*)malloc(densityArray.size()*sizeof(float));
	float* devBestProDensity=(float*)malloc(densityArray.size()*sizeof(float));
	for (int i = 0; i < densityArray.size(); ++i)
	{
		bestProDensity[i]=INFINITY;
		devBestProDensity[i]=0;
	}

	size_t computeArraySize=densityArray.size()*listNbNeihbours.size()*kernels.size();
	size_t arraySize=maxK*computeArraySize;
	double *cumulattedError=(double*)malloc(sizeof(double)*arraySize);
	memset(cumulattedError,0,sizeof(double)*arraySize);
	double *cumulattedSquaredError=(double*)malloc(sizeof(double)*arraySize);
	memset(cumulattedSquaredError,0,sizeof(double)*arraySize);
	uint16_t  *numberOfSampling=(uint16_t*)malloc(sizeof(uint16_t)*arraySize);
	memset(numberOfSampling,0,sizeof(uint16_t)*arraySize);

	unsigned seed=std::chrono::high_resolution_clock::now().time_since_epoch().count();
	std::mt19937 randomGenerator(seed);
	std::uniform_real_distribution<float> uniformDitributionOverSource(0.f,1.f);

	auto startTime = std::chrono::high_resolution_clock::now();

	std::atomic<bool> stop(false);
	#pragma omp parallel num_threads(nbThreads)  default(none) firstprivate(power, minNumberOfIteration, maxT, startTime, maxNumberOfIteration, logFile, seed, computeArraySize,densityArray,\
	 	circularSim, numberOfVariable,categoriesValues, radius, cumulattedError, maxK, cumulattedSquaredError, numberOfSampling, bestProDensity, devBestProDensity) \
		shared(listNbNeihbours, samplingModule, pathPosition, TIs , kernels,stop)
	{


		unsigned moduleID=0;
		#if _OPENMP
			moduleID=omp_get_thread_num();
		#endif

		FILE *fp;
		char fileName[2048];
		sprintf(fileName,"measure_%d.csv",moduleID);
		fp=fopen(fileName, "w");
		
		std::mt19937 randomGenerator(seed+moduleID);
		std::uniform_real_distribution<float> uniformDitributionOverSource(0.f,1.f);

		unsigned iteration=0;
		while (!stop && (iteration< maxNumberOfIteration)){
			// #pragma omp single
			// {
			// 	fprintf(logFile, "iteration %d\n", iteration);
			// }
			
			
			#pragma omp for schedule(dynamic,1)
			for (int setupIndex = 0; setupIndex < computeArraySize; ++setupIndex)
			{

				unsigned val=setupIndex;

				unsigned numberOfneihbours=val % listNbNeihbours.size();
				val/=listNbNeihbours.size();
				unsigned kernelIndex=val % kernels.size();
				val/=kernels.size();
				unsigned densityIndex=val % densityArray.size();
				val/=densityArray.size();			


				float localErrorBefore=std::pow(cumulattedError[setupIndex*maxK+0]/numberOfSampling[setupIndex*maxK+0],1/power);
				float localDevBefore=std::pow(cumulattedSquaredError[setupIndex*maxK+0]/numberOfSampling[setupIndex*maxK+0]-localErrorBefore*localErrorBefore,0.5/power);

				float nbSigmas=1;

				if( (numberOfSampling[setupIndex*maxK+0]>minNumberOfIteration) && ((localErrorBefore-nbSigmas*localDevBefore) > (bestProDensity[densityIndex]+nbSigmas*devBestProDensity[densityIndex]) )){
					continue;
				}

				// fprintf(logFile, "%d, ",densityIndex );
				float density=densityArray[densityIndex];
				std::vector<unsigned> numberNeighbor=listNbNeihbours[numberOfneihbours];

				int tiIndex=int(floor(uniformDitributionOverSource(randomGenerator)*TIs.size()));
				auto ti=&TIs[tiIndex];
				unsigned currentCell=unsigned(floor( uniformDitributionOverSource(randomGenerator) * ti->dataSize() / ti->_nbVariable ));

				std::vector<unsigned> numberOfNeighborsProVariable(TIs[0]._nbVariable);
				std::vector<std::vector<int> > neighborArrayVector;
				std::vector<std::vector<float> > neighborValueArrayVector;
				{
					

					unsigned positionSearch=0;
					while((numberNeighbor.size()>1||(neighborArrayVector.size()<numberNeighbor[0]))&&(positionSearch<pathPosition.size())){
						unsigned dataIndex;
						std::vector<int> vectorInDi=pathPosition[positionSearch];
						vectorInDi.resize(ti->_dims.size(),0);
						if(uniformDitributionOverSource(randomGenerator)<density && (ti->indexWithDelta(dataIndex, currentCell, vectorInDi) || circularSim))
						{
							std::vector<float> data(ti->_nbVariable);
							unsigned cpt=0;
							for (unsigned int i = 0; i < ti->_nbVariable; ++i)
							{
								if((numberOfNeighborsProVariable[i]<numberNeighbor[i%numberNeighbor.size()]))
								{
									float val=ti->_data[dataIndex*ti->_nbVariable+i];
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
						positionSearch++;
					}
				}
				// conversion from one variable to many
				for (size_t j = 0; j < neighborValueArrayVector.size(); ++j)
				{
					std::vector<float> data(numberOfVariable);
					unsigned id=0;
					unsigned idCategorie=0;
					for (unsigned int i = 0; i < ti->_nbVariable; ++i)
					{
						if(ti->_types[i]==g2s::DataImage::Continuous){
							
							data[id]=neighborValueArrayVector[j][i];
							id++;
						}
						if(ti->_types[i]==g2s::DataImage::Categorical){
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

				std::vector<SamplingModule::matchLocation> importIndex;

				if(neighborArrayVector.size()>1){
					SamplingModule::matchLocation origin;
					origin.TI=tiIndex;
					origin.index=currentCell;
					importIndex= samplingModule.distribution(neighborArrayVector,neighborValueArrayVector,uniformDitributionOverSource(randomGenerator),
						 origin, radius, moduleID, false, 0, 0.f, -1,&(kernels[kernelIndex]));
					
					//fprintf(stderr, "local %d, best %d \n",currentCell ,importIndex[0].index);
					// fprintf(fp, "%d,%d,%d", densityIndex, kernelIndex, numberOfneihbours);
					for (int i = 0; i < importIndex.size(); ++i)
					{
						double error=std::pow(std::fabs(TIs[importIndex[i].TI]._data[importIndex[i].index]-TIs[tiIndex]._data[currentCell]),power);
						//fprintf(stderr, "%d\n", importIndex[i].index);
						// fprintf(fp, ",%f", error);
						cumulattedError[setupIndex*maxK+i]+=error;
						cumulattedSquaredError[setupIndex*maxK+i]+=error*error;
						numberOfSampling[setupIndex*maxK+i]++;
					}

					// fprintf(fp, "\n");

					float localErrorAfter=cumulattedError[setupIndex*maxK+0]/numberOfSampling[setupIndex*maxK+0];
					if(localErrorBefore>localErrorAfter){
						float bestInDensity;
						#pragma omp atomic read
						bestInDensity=bestProDensity[densityIndex];
						if(bestInDensity>localErrorAfter){
							#pragma omp critical (updateBestProDensity)
							{
								bestProDensity[densityIndex]=std::pow(localErrorAfter,1/power);
								devBestProDensity[densityIndex]=std::pow(cumulattedSquaredError[setupIndex*maxK+0]/numberOfSampling[setupIndex*maxK+0]-localErrorAfter*localErrorAfter,0.5/power);
							}

						}
						
					}
				}
			}
			#pragma omp single
			{
				if(iteration%(maxNumberOfIteration/100)==0)
					fprintf(logFile, "progress : %.2f%%\n",float(iteration)/maxNumberOfIteration*100);
			
				if(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - startTime).count()>maxT)
				{
					stop=true;
				}
			}

			iteration++;
		}
		fclose(fp);
	}

	// #pragma omp parallel for num_threads(nbThreads)  default(none)
	for (int i = 0; i < arraySize; ++i)
	{
		float localMeanValue=cumulattedError[i]/numberOfSampling[i];
		MeanErrorimage._data[i]=std::pow(localMeanValue,1/power);
		DevErrorimage._data[i]=std::pow(cumulattedSquaredError[i]/numberOfSampling[i]-localMeanValue*localMeanValue,0.5/power);
		((unsigned*) (NumberOFsampleimage._data))[i]=numberOfSampling[i];
	}

	free(cumulattedError);
	free(cumulattedSquaredError);
	free(numberOfSampling);
	free(bestProDensity);
}


void calibrationFull(FILE *logFile, std::vector<g2s::DataImage> &TIs, std::vector<g2s::DataImage> &kernels, SamplingModule &samplingModule,
 std::vector<std::vector<int> > &pathPosition, std::vector<unsigned> numberNeighbor, std::vector<std::vector<float> > categoriesValues, unsigned nbThreads=1 ){

	// int displayRatio=std::max(numberOfPointToSimulate/100,1u);
	// unsigned* posterioryPath=(unsigned*)malloc( sizeof(unsigned) * di.dataSize());
	// memset(posterioryPath,255,sizeof(unsigned) * di.dataSize());
	// for (unsigned int i = 0; i < di.dataSize(); ++i)
	// {
	// 	bool withNan=false;
	// 	for (unsigned int j = 0; j < di._nbVariable; ++j)
	// 	{
	// 		withNan|=std::isnan(di._data[i]);
	// 	}
	// 	if(!withNan)
	// 		posterioryPath[i]=0;
	// }
	// for (unsigned int i = 0; i < numberOfPointToSimulate; ++i)
	// {
	// 	posterioryPath[solvingPath[i]]=i;
	// }
	
	// unsigned numberOfVariable=di._nbVariable;
	// for (size_t i = 0; i < categoriesValues.size(); ++i)
	// {
	// 	numberOfVariable+=categoriesValues[i].size()-1;
	// }
	// #pragma omp parallel for num_threads(nbThreads) schedule(dynamic,1) default(none) firstprivate(displayRatio,circularSim, fullStationary, numberOfVariable, categoriesValues, numberOfPointToSimulate, \
	// 	posterioryPath, solvingPath, seedAray, numberNeighbor, importDataIndex, logFile, ii) shared( pathPosition, di, samplingModule, TIs)
	// for (unsigned int indexPath = 0; indexPath < numberOfPointToSimulate; ++indexPath){
		

	// 	unsigned moduleID=0;
	// 	#if _OPENMP
	// 		moduleID=omp_get_thread_num();
	// 	#endif
	// 	unsigned currentCell=solvingPath[indexPath];
	// 	if(!std::isnan(di._data[currentCell])) continue;
	// 	float localSeed=seedAray[indexPath];

	// 	unsigned currentVariable=currentCell%di._nbVariable;
	// 	unsigned currentPosition=currentCell/di._nbVariable;

	// 	std::vector<unsigned> numberOfNeighborsProVariable(di._nbVariable);
	// 	std::vector<std::vector<int> > neighborArrayVector;
	// 	std::vector<std::vector<float> > neighborValueArrayVector;
	// 	{
	// 		unsigned positionSearch=0;
	// 		while((numberNeighbor.size()>1||(neighborArrayVector.size()<numberNeighbor[0]))&&(positionSearch<pathPosition.size())){
	// 			unsigned dataIndex;
	// 			std::vector<int> vectorInDi=pathPosition[positionSearch];
	// 			vectorInDi.resize(di._dims.size(),0);
	// 			if(di.indexWithDelta(dataIndex, currentPosition, vectorInDi) || circularSim)
	// 			{
	// 				bool needToBeadd=false;
	// 				for (unsigned int i = 0; i < di._nbVariable; ++i)
	// 				{
	// 					needToBeadd|=(numberOfNeighborsProVariable[i]<numberNeighbor[i%numberNeighbor.size()])&&(posterioryPath[dataIndex*di._nbVariable+i]<indexPath) ;
	// 				}
	// 				//add for
	// 				if(needToBeadd){
	// 					unsigned numberOfNaN=0;
	// 					float val;
	// 					while(true) {
	// 						numberOfNaN=0;
	// 						for (unsigned int i = 0; i < di._nbVariable; ++i)
	// 						{
	// 							#pragma omp atomic read
	// 							val=di._data[dataIndex*di._nbVariable+i];
	// 							numberOfNaN+=(numberOfNeighborsProVariable[i]<numberNeighbor[i%numberNeighbor.size()])&&(posterioryPath[dataIndex*di._nbVariable+i]<indexPath) && std::isnan(val);
	// 						}
	// 						if(numberOfNaN==0)break;
	// 						std::this_thread::sleep_for(std::chrono::microseconds(250));
	// 					}

	// 					std::vector<float> data(di._nbVariable);
	// 					unsigned cpt=0;
	// 					for (unsigned int i = 0; i < di._nbVariable; ++i)
	// 					{
	// 						if((numberOfNeighborsProVariable[i]<numberNeighbor[i%numberNeighbor.size()])&&(posterioryPath[dataIndex*di._nbVariable+i]<indexPath))
	// 						{
	// 							#pragma omp atomic read
	// 							val=di._data[dataIndex*di._nbVariable+i];
	// 							data[i]=val;
	// 							cpt++;
	// 							numberOfNeighborsProVariable[i]++;
	// 						}else{
	// 							data[i]=std::nanf("0");
	// 						}
	// 					}
	// 					neighborValueArrayVector.push_back(data);
	// 					neighborArrayVector.push_back(pathPosition[positionSearch]);
	// 					if(cpt==0) break;
	// 				}
	// 			}
	// 			positionSearch++;
	// 		}
	// 	}
	// 	// conversion from one variable to many
	// 	for (size_t j = 0; j < neighborValueArrayVector.size(); ++j)
	// 	{
	// 		std::vector<float> data(numberOfVariable);
	// 		unsigned id=0;
	// 		unsigned idCategorie=0;
	// 		for (unsigned int i = 0; i < di._nbVariable; ++i)
	// 		{
	// 			if(di._types[i]==g2s::DataImage::Continuous){
					
	// 				data[id]=neighborValueArrayVector[j][i];
	// 				id++;
	// 			}
	// 			if(di._types[i]==g2s::DataImage::Categorical){
	// 				for (size_t k = 0; k < categoriesValues[idCategorie].size(); ++k)
	// 				{
	// 					data[id] = (neighborValueArrayVector[j][i] == categoriesValues[idCategorie][k]);
	// 					id++;
	// 				}
	// 				idCategorie++;
	// 			}
	// 		}
	// 		neighborValueArrayVector[j]=data;
	// 	}


	// 	SamplingModule::matchLocation importIndex;

	// 	importIndex.TI=0;
	// 	importIndex.index=INT_MAX;

	// 	if(neighborArrayVector.size()>1){
	// 		SamplingModule::matchLocation verbatimRecord;
	// 		verbatimRecord.TI=TIs.size();
	// 		importIndex=samplingModule.sample(neighborArrayVector,neighborValueArrayVector,localSeed,verbatimRecord,moduleID, fullStationary, currentVariable ,(ii!=nullptr ? int(ii->_data[currentCell]):-1));
	// 	}else{

	// 		// sample from the marginal
	// 		unsigned cumulated=0;
	// 		for (size_t i = 0; i < TIs.size(); ++i)
	// 		{
	// 			cumulated+=TIs[i].dataSize();
	// 		}
			
	// 		unsigned position=int(floor(localSeed*(cumulated/TIs[0]._nbVariable)));

	// 		cumulated=0;
	// 		for (size_t i = 0; i < TIs.size(); ++i)
	// 		{
	// 			if(position*TIs[0]._nbVariable<cumulated+TIs[i].dataSize()){
	// 				importIndex.TI=i;
	// 				importIndex.index=position-cumulated/TIs[0]._nbVariable;
	// 				break;
	// 			}else{
	// 				cumulated+=TIs[i].dataSize();
	// 			}
	// 		}

	// 		bool hasNaN=std::isnan(TIs[importIndex.TI]._data[importIndex.index*TIs[importIndex.TI]._nbVariable+currentVariable]); 
		
	// 		if(hasNaN){ // nan safe, much slower
	// 			unsigned cumulated=0;
	// 			for (size_t i = 0; i < TIs.size(); ++i)
	// 			{
	// 				for (unsigned int k = 0; k < TIs[i].dataSize()/TIs[i]._nbVariable; ++k)
	// 				{
	// 					bool locHasNan=false;
	// 					int j=currentVariable;
	// 					{
	// 						locHasNan|=std::isnan(TIs[i]._data[k*TIs[i]._nbVariable+j]);
	// 					}
	// 					cumulated+=!locHasNan;
	// 				}
	// 			}
	// 			if(cumulated==0)fprintf(logFile, "error - no available data for variable: %d", currentVariable);
	// 			unsigned position=int(floor(localSeed*(cumulated/TIs[0]._nbVariable)))*TIs[0]._nbVariable;

	// 			cumulated=0;

	// 			for (size_t i = 0; i < TIs.size(); ++i)
	// 			{
	// 				for (unsigned int k = 0; k < TIs[i].dataSize()/TIs[i]._nbVariable; ++k)
	// 				{
	// 					bool locHasNan=false;
	// 					int j=currentVariable;
	// 					{
	// 						locHasNan|=std::isnan(TIs[i]._data[k*TIs[i]._nbVariable+j]);
	// 					}
	// 					cumulated+=!locHasNan;
	// 					if(position<=cumulated){
	// 						importIndex.TI=i;
	// 						importIndex.index=k;
	// 						break;
	// 					}
	// 				}
	// 				if(position<=cumulated)break;
	// 			}
	// 		}
	// 	}
	// 	// import data
	// 	//memcpy(di._data+currentCell*di._nbVariable,TIs[importIndex.TI]._data+importIndex.index*TIs[importIndex.TI]._nbVariable,TIs[importIndex.TI]._nbVariable*sizeof(float));
	// 	importDataIndex[currentCell]=importIndex.index*TIs.size()+importIndex.TI;
	// 	//fprintf(stderr, "write %d\n", importDataIndex[currentCell]);
		
	// 	if(std::isnan(di._data[currentCell])){
	// 		#pragma omp atomic write
	// 		di._data[currentCell]=TIs[importIndex.TI]._data[importIndex.index*TIs[importIndex.TI]._nbVariable+currentVariable];
	// 	}
	// 	if(indexPath%(displayRatio)==0)fprintf(logFile, "progress : %.2f%%\n",float(indexPath)/numberOfPointToSimulate*100);
	// }
	// free(posterioryPath);
}

#endif // CALIBARTION_HPP