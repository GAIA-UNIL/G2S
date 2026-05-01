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

#ifndef QUANTILE_SAMPLING_MODULE_HPP
#define QUANTILE_SAMPLING_MODULE_HPP

#include <cmath>
#include <limits>
#include "samplingModule.hpp"
#include "AcceleratorDevice.hpp"
#include "fKst.hpp"



class QuantileSamplingModule: public SamplingModule {

private:
	float _k;
	bool _completeTIs=true;
	unsigned _nbThreadOverTI=1;
	unsigned _threadRatio=1;
	bool _noVerbatim=false;
	std::vector<std::vector<conversionType> > _conversionTypeVector;
	std::vector<std::vector<std::vector<conversionType> > > _conversionTypeVectorConstVector;
	std::vector<std::vector<std::vector<float> > > _conversionCoefVectorConstVector;
	bool _useUniqueTI=false;

	std::vector<float*> _errors;
	std::vector<unsigned*> _encodedPosition;

	inline unsigned seedFromUnitFloat(float seed) const{
		if(!std::isfinite(seed) || seed<=0.f) return 0;
		if(seed>=1.f) return std::numeric_limits<unsigned>::max();
		return static_cast<unsigned>(std::floor(static_cast<double>(seed)*static_cast<double>(std::numeric_limits<unsigned>::max())));
	}
public:
	
	QuantileSamplingModule(std::vector<ComputeDeviceModule *> *cdmV, g2s::DataImage* kernel, float k,  std::vector<std::vector<conversionType> > conversionTypeVector,
		std::vector<std::vector<std::vector<conversionType> > > conversionTypeVectorConstVector, std::vector<std::vector<std::vector<float> > > conversionCoefVectorConstVector,
		bool noVerbatim, bool completeTIs, unsigned nbThread, unsigned nbThreadOverTI=1, unsigned threadRatio=1, bool useUniqueTI=false):SamplingModule(cdmV,kernel)
	{
		_k=k;
		_conversionTypeVector=conversionTypeVector;
		_errors.resize(nbThread,nullptr);
		_encodedPosition.resize(nbThread,nullptr);
		_completeTIs=completeTIs;

		_useUniqueTI=useUniqueTI;
		for (unsigned i = 0; i < nbThread; ++i)
		{
			_errors[i]=(float*)malloc(_cdmV[0].size() * int(ceil(_k)) * sizeof(float));
			_encodedPosition[i]=(unsigned*)malloc(_cdmV[0].size() * int(ceil(_k)) * sizeof(unsigned));
		}
		_nbThreadOverTI=nbThreadOverTI;
		_threadRatio=threadRatio;
		_noVerbatim=noVerbatim;

		_conversionTypeVectorConstVector=conversionTypeVectorConstVector;
		_conversionCoefVectorConstVector=conversionCoefVectorConstVector;
	}

	~QuantileSamplingModule(){
		for (size_t i = 0; i < _errors.size(); ++i)
		{
			free(_errors[i]);
			_errors[i]=nullptr;
		}
		
		for (size_t i = 0; i < _encodedPosition.size(); ++i)
		{
			free(_encodedPosition[i]);
			_encodedPosition[i]=nullptr;
		}
	};

	inline matchLocation sample(std::vector<std::vector<int>> neighborArrayVector, std::vector<std::vector<float> > neighborValueArrayVector,float seed, matchLocation verbatimRecord, unsigned moduleID=0, bool fullStationary=false, unsigned variableOfInterest=0, float localk=0.f, int idTI4Sampling=-1, g2s::DataImage* localKernel=nullptr){

		if(localk<=0.f)
			localk=_k;

		unsigned vectorSize=_cdmV[moduleID].size();
		float *errors=_errors[moduleID];
		unsigned* encodedPosition=_encodedPosition[moduleID];
		int extendK=int(ceil(localk));
		std::fill(errors,errors+vectorSize*extendK,-INFINITY);
		determineDistribution(errors, encodedPosition, neighborArrayVector,neighborValueArrayVector, seed, verbatimRecord, 0, moduleID, fullStationary, variableOfInterest, localk, idTI4Sampling, localKernel);
		unsigned localPosition[extendK*vectorSize];
		std::iota(localPosition,localPosition+extendK*vectorSize,0);
		
		//printf("%d %d %d %d\n", updated[0],updated[1],updated[2],updated[3]);

		std::sort(localPosition,localPosition+extendK*vectorSize,[errors](unsigned a, unsigned b){
			return errors[a] > errors[b];
		});
		//printf("%f %f %f %f\n", errors[localPosition[0]],errors[localPosition[1]],errors[localPosition[2]],errors[localPosition[3]]);
		//fprintf(stderr, "%f\n", errors[0]);
		//fKst::findKSmallest(_errors,3,extendK, localErrors, localPosition);

		unsigned selectedIndex=int(floor(seed*localk*(ceil(vectorSize/localk)/vectorSize)));
		unsigned selectedTI=localPosition[selectedIndex]/extendK;
		//fprintf(stderr, "mask : %f\n", (_cdmV[moduleID][selectedTI]->getCrossErrorArray())[encodedPosition[localPosition[selectedIndex]]]);
		//fprintf(stderr, "position %d \n",encodedPosition[localPosition[selectedIndex]] );
		//fprintf(stderr, "position %d \n",_cdmV[moduleID][selectedTI]->getErrorsArraySize() - encodedPosition[localPosition[selectedIndex]] );
		unsigned indexInTI=_cdmV[moduleID][selectedTI]->cvtIndexToPosition(encodedPosition[localPosition[selectedIndex]]);
		//fprintf(stderr, "%d %d %d %d %d %f\n", selectedTI, indexInTI, localPosition[selectedIndex],selectedIndex,extendK, errors[localPosition[selectedIndex]]);

		matchLocation result;
		result.TI=selectedTI;
		result.index=indexInTI;

		return result;
	}

	std::vector<matchLocation> distribution(std::vector<std::vector<int>> neighborArrayVector, std::vector<std::vector<float> > neighborValueArrayVector,float seed, matchLocation verbatimRecord, float verbatimRadius=0, unsigned moduleID=0, bool fullStationary=false, unsigned variableOfInterest=0, float localk=0.f, int idTI4Sampling=-1, g2s::DataImage* localKernel=nullptr){

		if(localk<=0.f)
			localk=_k;

		unsigned vectorSize=_cdmV[moduleID].size();
		int extendK=int(ceil(localk));
		//std::vector<float> errorsArray(extendK*3);
		//float *errors=(float*)errorsArray.data();
		float *errors=_errors[moduleID];
		std::fill(errors,errors+vectorSize*extendK,-INFINITY);
		unsigned* encodedPosition=_encodedPosition[moduleID];

		determineDistribution(errors, encodedPosition, neighborArrayVector,neighborValueArrayVector, seed, verbatimRecord, verbatimRadius, moduleID, fullStationary, variableOfInterest, localk, idTI4Sampling, localKernel);

		unsigned localPosition[extendK*vectorSize];
		// std::iota(localPosition+idTI4Sampling*extendK,localPosition+(idTI4Sampling+1)*extendK,idTI4Sampling*extendK);

		// std::sort(localPosition+idTI4Sampling*extendK,localPosition+(idTI4Sampling+1)*extendK,[errors](unsigned a, unsigned b){
		// 	return errors[a] > errors[b];
		// });

		std::iota(localPosition,localPosition+extendK*vectorSize,0);
		std::sort(localPosition,localPosition+extendK*vectorSize,[errors](unsigned a, unsigned b){
			return errors[a] > errors[b];
		});

		std::vector<matchLocation> results;

		for (int i = 0; i < extendK; ++i)
		{
			// unsigned selectedIndex=idTI4Sampling*extendK+i;//int(floor(seed*_k*(ceil(vectorSize/_k)/vectorSize)));
			unsigned selectedTI=localPosition[i]/extendK;
			unsigned indexInTI=_cdmV[moduleID][selectedTI]->cvtIndexToPosition(encodedPosition[localPosition[i]]);
			matchLocation result;
			result.TI=selectedTI;
			result.index=indexInTI;
			results.push_back(result);
		}

		return results;
	}

	narrownessMeasurment narrowness(std::vector<std::vector<int>> neighborArrayVector, std::vector<std::vector<float> > neighborValueArrayVector,float seed, unsigned moduleID=0, bool fullStationary=false){
		unsigned localNbThreadOverTI=_nbThreadOverTI;
		unsigned vectorSize=_cdmV[moduleID].size();
		float *errors=_errors[moduleID];
		unsigned* encodedPosition=_encodedPosition[moduleID];
		bool updated[vectorSize];
		memset(updated,false,vectorSize);

		std::vector<std::vector<float> > convertedNeighborValueArrayVector(neighborArrayVector.size());
		std::vector<float> cumulatedVariablesCoefficient;

		//if(_conversionTypeVector[0].size()!=neighborValueArrayVector[0].size()) //to redo
		//	fprintf(stderr, "%s %d vs %d\n", "failure",_conversionTypeVector[0].size(),neighborValueArrayVector[0].size());

		unsigned indexCenter=0;
		for (int i =  int(_kernel->_dims.size()-1); i>=0 ; i--)
		{
			indexCenter=indexCenter*_kernel->_dims[i]+_kernel->_dims[i]/2;
		}

		for (size_t i = 0; i < _conversionTypeVector.size(); ++i)
		{

			for (size_t j = 0; j < _conversionTypeVector[i].size(); ++j)
			{
				switch(_conversionTypeVector[i][j]){
					case conversionType::P0:
						for (size_t k = 0; k < neighborArrayVector.size(); ++k)
						{
							unsigned indexInKernel=indexCenter;
							if(_kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]) && !std::isnan(neighborValueArrayVector[k][i]))
								convertedNeighborValueArrayVector[k].push_back(_kernel->_data[indexInKernel*_kernel->_nbVariable+(i%_kernel->_nbVariable)]*1.f);
							else
								convertedNeighborValueArrayVector[k].push_back(0.f);
						}						
					break;
					case conversionType::P1:
						for (size_t k = 0; k < neighborArrayVector.size(); ++k)
						{
							unsigned indexInKernel=indexCenter;
							if(_kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]) && !std::isnan(neighborValueArrayVector[k][i]))
								convertedNeighborValueArrayVector[k].push_back(_kernel->_data[indexInKernel*_kernel->_nbVariable+(i%_kernel->_nbVariable)]*neighborValueArrayVector[k][i]);
							else
								convertedNeighborValueArrayVector[k].push_back(0.f);
						}						
					break;
					case conversionType::P2:
						for (size_t k = 0; k < neighborArrayVector.size(); ++k)
						{
							unsigned indexInKernel=indexCenter;
							if(_kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]) && !std::isnan(neighborValueArrayVector[k][i]))
								convertedNeighborValueArrayVector[k].push_back(_kernel->_data[indexInKernel*_kernel->_nbVariable+(i%_kernel->_nbVariable)]*neighborValueArrayVector[k][i]*neighborValueArrayVector[k][i]);
							else
								convertedNeighborValueArrayVector[k].push_back(0.f);
						}						
					break;
					case conversionType::MinMinus1:
					case conversionType::MaxPlus1:
					break;
				}	
			}
		}

		std::vector<float> delta;
		//if(_completeTIs)
		{
			for (size_t p = 0; p < _conversionTypeVectorConstVector.size(); ++p)
			{
				float sum=0;
				for (size_t i = 0; i < _conversionTypeVectorConstVector[p].size(); ++i)
				{
					for (size_t j = 0; j < _conversionTypeVectorConstVector[p][i].size(); ++j)
					{
						switch(_conversionTypeVectorConstVector[p][i][j]){
							case conversionType::P0:
								for (size_t k = 0; k < neighborArrayVector.size(); ++k)
								{
									unsigned indexInKernel=indexCenter;
									if(_kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]) && !std::isnan(neighborValueArrayVector[k][i]))
										sum+=_conversionCoefVectorConstVector[p][i][j]*(_kernel->_data[indexInKernel*_kernel->_nbVariable+(i%_kernel->_nbVariable)]*1.f);
								}
							break;
							case conversionType::P1:
								for (size_t k = 0; k < neighborArrayVector.size(); ++k)
								{
									unsigned indexInKernel=indexCenter;
									//fprintf(stderr, "%d ==> %f\n", _kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]) && !std::isnan(neighborValueArrayVector[k][i]),neighborValueArrayVector[k][i]);
									if(_kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]) && !std::isnan(neighborValueArrayVector[k][i]))
										sum+=_conversionCoefVectorConstVector[p][i][j]*(_kernel->_data[indexInKernel*_kernel->_nbVariable+(i%_kernel->_nbVariable)]*neighborValueArrayVector[k][i]);
								}
							break;
							case conversionType::P2:
								for (size_t k = 0; k < neighborArrayVector.size(); ++k)
								{
									unsigned indexInKernel=indexCenter;
									if(_kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]) && !std::isnan(neighborValueArrayVector[k][i]))
										sum+=_conversionCoefVectorConstVector[p][i][j]*(_kernel->_data[indexInKernel*_kernel->_nbVariable+(i%_kernel->_nbVariable)]*neighborValueArrayVector[k][i]*neighborValueArrayVector[k][i]);
								}
							break;
							case conversionType::MinMinus1:
							case conversionType::MaxPlus1:
							break;
						}	
					}
				}
				//fprintf(stderr, "%d ==> %f\n",delta.size(), sum);
				delta.push_back(sum);
			}
		}

		#pragma omp parallel for default(none) num_threads(localNbThreadOverTI) firstprivate(vectorSize,delta,moduleID) shared(updated, neighborArrayVector, convertedNeighborValueArrayVector, cumulatedVariablesCoefficient) 
		for (unsigned int i = 0; i < vectorSize; ++i)
		{
			updated[i]=_cdmV[moduleID][i]->candidateForPattern(neighborArrayVector, convertedNeighborValueArrayVector, cumulatedVariablesCoefficient,delta);
		}

		int extendK=int(ceil(_k));
		std::fill(errors,errors+vectorSize*extendK,-INFINITY);

		#pragma omp parallel for default(none) num_threads(localNbThreadOverTI) firstprivate(extendK,errors,encodedPosition,vectorSize,delta,moduleID) shared(updated, neighborArrayVector, convertedNeighborValueArrayVector, cumulatedVariablesCoefficient) 
		for (unsigned int i = 0; i < vectorSize; ++i)
		{
			if(updated[i])
			{
				float* errorsArray=_cdmV[moduleID][i]->getErrorsArray();
				float* crossErrorsArray=_cdmV[moduleID][i]->getCrossErrorArray();

				if(!_completeTIs)
				{
					_cdmV[moduleID][i]->maskCroossError();
					#pragma omp simd
					for (unsigned int j = 0; j < _cdmV[moduleID][i]->getErrorsArraySize(); ++j)
					{
						errorsArray[j]=-std::fabs(errorsArray[j]/(crossErrorsArray[j]*crossErrorsArray[j]*crossErrorsArray[j]*crossErrorsArray[j]));
						if(crossErrorsArray[j]==0.0f) errorsArray[j]=-INFINITY;
					}
				}
				fKst::findKBiggest(errorsArray,_cdmV[moduleID][i]->getErrorsArraySize(),extendK, errors+i*extendK, encodedPosition+i*extendK);
			}
		}

		unsigned localPosition[extendK*vectorSize];
		std::iota(localPosition,localPosition+extendK*vectorSize,0);
		std::sort(localPosition,localPosition+extendK*vectorSize,[errors](unsigned a, unsigned b){
			return errors[a] > errors[b];
		});
		//fKst::findKSmallest(_errors,3,extendK, localErrors, localPosition);

		unsigned selectedIndex=int(floor(seed*_k));
		unsigned selectedTI=localPosition[selectedIndex]/extendK;
		unsigned indexInTI=_cdmV[moduleID][selectedTI]->cvtIndexToPosition(encodedPosition[localPosition[selectedIndex]]);

		float errorsForNS[extendK];
		unsigned int index[extendK];
		unsigned int tiId[extendK];

		for (int i = 0; i < extendK; ++i)
		{
			errorsForNS[i]=errors[localPosition[i]];
			tiId[i]=localPosition[i]/extendK;
			index[i]=_cdmV[moduleID][tiId[i]]->cvtIndexToPosition(encodedPosition[localPosition[i]]);
		}

		narrownessMeasurment result;
		result.narrowness=_narrownessFunction(errorsForNS,tiId,index,extendK);
		result.candidate.TI=selectedTI;
		result.candidate.index=indexInTI;
		return result;
	}

private:
	inline void determineDistribution(float *errors, unsigned* encodedPosition, std::vector<std::vector<int>> neighborArrayVector, std::vector<std::vector<float> > neighborValueArrayVector,float seed, matchLocation verbatimRecord, float verbatimRadius=0.f, unsigned moduleID=0, bool fullStationary=false, unsigned variableOfInterest=0, float localk=0.f, int idTI4Sampling=-1, g2s::DataImage* localKernel=nullptr){
		
		if(localk<=0.f)
			localk=_k;

		unsigned localNbThreadOverTI=_nbThreadOverTI;
		unsigned vectorSize=_cdmV[moduleID].size();;
		bool updated[vectorSize];
		memset(updated,false,vectorSize);

		std::vector<std::vector<float> > convertedNeighborValueArrayVector(neighborArrayVector.size());
		std::vector<float> cumulatedVariablesCoefficient;

		//if(_conversionTypeVector[0].size()!=neighborValueArrayVector[0].size()) //to redo
		//	fprintf(stderr, "%s %d vs %d\n", "failure",_conversionTypeVector[0].size(),neighborValueArrayVector[0].size());

		g2s::DataImage* kernel=_kernel;
		if(localKernel!=nullptr)
			kernel=localKernel;

		unsigned indexCenter=0;
		for (int i =  int(kernel->_dims.size()-1); i>=0 ; i--)
		{
			indexCenter=indexCenter*kernel->_dims[i]+kernel->_dims[i]/2;
		}

		for (size_t i = 0; i < _conversionTypeVector.size(); ++i)
		{

			for (size_t j = 0; j < _conversionTypeVector[i].size(); ++j)
			{
				switch(_conversionTypeVector[i][j]){
					case conversionType::P0:
						for (size_t k = 0; k < neighborArrayVector.size(); ++k)
						{
							unsigned indexInKernel=indexCenter;
							if(kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]) && !std::isnan(neighborValueArrayVector[k][i]))
								convertedNeighborValueArrayVector[k].push_back(kernel->_data[indexInKernel*kernel->_nbVariable+(i%kernel->_nbVariable)]*1.f);
							else
								convertedNeighborValueArrayVector[k].push_back(0.f);
						}
					break;
					case conversionType::P1:
						for (size_t k = 0; k < neighborArrayVector.size(); ++k)
						{
							unsigned indexInKernel=indexCenter;
							//fprintf(stderr, "%d ==> %f\n", kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]) && !std::isnan(neighborValueArrayVector[k][i]),neighborValueArrayVector[k][i]);
							if(kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]) && !std::isnan(neighborValueArrayVector[k][i]))
								convertedNeighborValueArrayVector[k].push_back(kernel->_data[indexInKernel*kernel->_nbVariable+(i%kernel->_nbVariable)]*neighborValueArrayVector[k][i]);
							else
								convertedNeighborValueArrayVector[k].push_back(0.f);
						}
					break;
					case conversionType::P2:
						for (size_t k = 0; k < neighborArrayVector.size(); ++k)
						{
							unsigned indexInKernel=indexCenter;
							if(kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]) && !std::isnan(neighborValueArrayVector[k][i]))
								convertedNeighborValueArrayVector[k].push_back(kernel->_data[indexInKernel*kernel->_nbVariable+(i%kernel->_nbVariable)]*neighborValueArrayVector[k][i]*neighborValueArrayVector[k][i]);
							else
								convertedNeighborValueArrayVector[k].push_back(0.f);
						}
					break;
					case conversionType::MinMinus1:
					case conversionType::MaxPlus1:
					break;
				}	
			}
		}

		std::vector<float> delta;
		//if(_completeTIs)
		{
			for (size_t p = 0; p < _conversionTypeVectorConstVector.size(); ++p)
			{
				float sum=0;
				for (size_t i = 0; i < _conversionTypeVectorConstVector[p].size(); ++i)
				{
					for (size_t j = 0; j < _conversionTypeVectorConstVector[p][i].size(); ++j)
					{
						switch(_conversionTypeVectorConstVector[p][i][j]){
							case conversionType::P0:
								for (size_t k = 0; k < neighborArrayVector.size(); ++k)
								{
									unsigned indexInKernel=indexCenter;
									if(kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]) && !std::isnan(neighborValueArrayVector[k][i]))
										sum+=_conversionCoefVectorConstVector[p][i][j]*(kernel->_data[indexInKernel*kernel->_nbVariable+(i%kernel->_nbVariable)]*1.f);
								}
							break;
							case conversionType::P1:
								for (size_t k = 0; k < neighborArrayVector.size(); ++k)
								{
									unsigned indexInKernel=indexCenter;
									//fprintf(stderr, "%d ==> %f\n", kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]) && !std::isnan(neighborValueArrayVector[k][i]),neighborValueArrayVector[k][i]);
									if(kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]) && !std::isnan(neighborValueArrayVector[k][i]))
										sum+=_conversionCoefVectorConstVector[p][i][j]*(kernel->_data[indexInKernel*kernel->_nbVariable+(i%kernel->_nbVariable)]*neighborValueArrayVector[k][i]);
								}
							break;
							case conversionType::P2:
								for (size_t k = 0; k < neighborArrayVector.size(); ++k)
								{
									unsigned indexInKernel=indexCenter;
									if(kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]) && !std::isnan(neighborValueArrayVector[k][i]))
										sum+=_conversionCoefVectorConstVector[p][i][j]*(kernel->_data[indexInKernel*kernel->_nbVariable+(i%kernel->_nbVariable)]*neighborValueArrayVector[k][i]*neighborValueArrayVector[k][i]);
								}
							break;
							case conversionType::MinMinus1:
							case conversionType::MaxPlus1:
							break;
						}	
					}
				}
				//fprintf(stderr, "%d ==> %f\n",delta.size(), sum);
				delta.push_back(sum);
			}
		}

		bool toUpdate[vectorSize];
		std::fill(toUpdate,toUpdate+vectorSize,true);

		if(fullStationary){
			std::mt19937 generator;
			unsigned baseSeed=seedFromUnitFloat(seed);
			generator.seed(baseSeed>0 ? baseSeed-1 : 0);
			for (int i = 0; i < vectorSize-ceil(vectorSize/localk); ++i)
			{
				toUpdate[i]=false;
			}
			std::shuffle(toUpdate,toUpdate+vectorSize,generator);
		}
		if(idTI4Sampling>=0){
			std::fill(toUpdate,toUpdate+vectorSize,false);
			toUpdate[idTI4Sampling]=true;
			localNbThreadOverTI=1;
		}

		#pragma omp parallel for default(none) num_threads(localNbThreadOverTI) firstprivate(toUpdate,vectorSize,delta,moduleID) shared(updated, neighborArrayVector, convertedNeighborValueArrayVector, cumulatedVariablesCoefficient) 
		for (unsigned int i = 0; i < vectorSize; ++i)
		{
			if(toUpdate[i]){
				updated[i]=_cdmV[moduleID][i]->candidateForPattern(neighborArrayVector, convertedNeighborValueArrayVector, cumulatedVariablesCoefficient,delta);
			}
		}

		int extendK=int(ceil(localk));
		
		//#pragma omp parallel for default(none) num_threads(localNbThreadOverTI) /*proc_bind(close)*/ firstprivate(seed, extendK,errors,encodedPosition,vectorSize,delta,moduleID,verbatimRecord,variableOfInterest) shared(updated, neighborArrayVector, convertedNeighborValueArrayVector, cumulatedVariablesCoefficient) 
		for (unsigned int i = 0; i < vectorSize; ++i)
		{
			
			if(updated[i])
			{
				AcceleratorDevice* localCdmV=dynamic_cast<AcceleratorDevice*>(_cdmV[moduleID][i]);

				float localError[extendK*_threadRatio];
				float* localErrorPtr=localError;
				unsigned localEncodedPosition[extendK*_threadRatio];
				unsigned* localEncodedPositionPtr=localEncodedPosition;

				if(localCdmV==nullptr){ // is not AcceleratorDevice

					float* errorsArray=_cdmV[moduleID][i]->getErrorsArray();
					float* crossErrorsArray=_cdmV[moduleID][i]->getCrossErrorArray();
					unsigned sizeArray=_cdmV[moduleID][i]->getErrorsArraySize();

					if(!_completeTIs)
					{
						_cdmV[moduleID][i]->maskCroossErrorWithVariable(variableOfInterest);
						#pragma omp simd
						for (unsigned int j = 0; j < _cdmV[moduleID][i]->getErrorsArraySize(); ++j)
						{
							errorsArray[j]=-std::fabs(errorsArray[j]/(crossErrorsArray[j]*crossErrorsArray[j]*crossErrorsArray[j]*crossErrorsArray[j]));
							if(crossErrorsArray[j]==0.0f) errorsArray[j]=-INFINITY;
						}
					}

					if(_noVerbatim && (verbatimRecord.TI==i)){
						if(verbatimRadius<1)
							errorsArray[_cdmV[moduleID][i]->cvtPositionToIndex(verbatimRecord.index)]=-INFINITY;
						else
							_cdmV[moduleID][i]->setValueInErrorArrayWithRadius(_cdmV[moduleID][i]->cvtPositionToIndex(verbatimRecord.index), -INFINITY, verbatimRadius);
					}
					
			#if !defined( __GNUC__) || defined(__INTEL_COMPILER)  // remove OpenMP in this section for GCC compiler, some GCC compiler produce a code that crash without any reasons
					#pragma omp parallel default(none) num_threads(_threadRatio) /*proc_bind(close)*/ firstprivate(seed, sizeArray, errorsArray, extendK, localErrorPtr, localEncodedPositionPtr)
			#endif
					{
						unsigned k=0;
						#if _OPENMP && (!defined( __GNUC__) || defined(__INTEL_COMPILER))
						k=omp_get_thread_num();
						#endif
						std::mt19937 generator;// can be inprouved by only resetting the seed each time
						generator.seed(seedFromUnitFloat(seed)+k);
						std::uniform_real_distribution<float> distribution(0.0,1.0);

						auto rng = std::bind(distribution, std::ref(generator));
						unsigned chunkSize=unsigned(ceil(sizeArray/float(_threadRatio)));
						fKst::findKBiggest(errorsArray+k*chunkSize,chunkSize,extendK, localErrorPtr+k*extendK, localEncodedPositionPtr+k*extendK, rng);
						for (int j = 0; j < extendK; ++j)
						{
							localEncodedPositionPtr[k*extendK+j]+=k*chunkSize;
						}
					}

					for (int j = 0; j <extendK ; ++j)
					{
						unsigned bestIndex=0;
						for (unsigned int l = 1; l < _threadRatio*extendK; ++l)
						{
							if(localError[l] > localError[bestIndex]) bestIndex=l;
						}

						errors[i*extendK+j]=localError[bestIndex];
						encodedPosition[i*extendK+j]=localEncodedPosition[bestIndex];
						localError[bestIndex]=-INFINITY;
					}
				}else{	//is AcceleratorDevice
					if(!_completeTIs)
					{

						localCdmV->maskCroossErrorWithVariable(variableOfInterest);
						localCdmV->compensateMissingData();
					}

					if(_noVerbatim && (verbatimRecord.TI==i)){
						if(verbatimRadius<1)
							localCdmV->setValueInErrorArray(localCdmV->cvtPositionToIndex(verbatimRecord.index),-INFINITY);
						else
							localCdmV->setValueInErrorArrayWithRadius(localCdmV->cvtPositionToIndex(verbatimRecord.index), -INFINITY, verbatimRadius);
					}

					localCdmV->searchKBiggest(errors+i*extendK,encodedPosition+i*extendK,extendK,seed);
				}
			}
		}
	}
};

#endif // QUANTILE_SAMPLING_MODULE_HPP
