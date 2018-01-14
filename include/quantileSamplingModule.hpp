#ifndef QUANTILE_SAMPLING_MODULE_HPP
#define QUANTILE_SAMPLING_MODULE_HPP

#include "samplingModule.hpp"
#include "fKst.hpp"



class QuantileSamplingModule: public SamplingModule {

private:
	float _k;
	bool _completeTIs=true;
	std::vector<std::vector<convertionType> > _convertionTypeVector;
	std::vector<std::vector<float> > _variablesCoeficient;

	std::vector<float*> _errors;
	std::vector<unsigned*> _encodedPosition;
public:
	QuantileSamplingModule(std::vector<ComputeDeviceModule *> *cdmV, g2s::DataImage* kernel, float k,  std::vector<std::vector<convertionType> > convertionTypeVector, std::vector<std::vector<float> > variablesCoeficient, bool completeTIs, unsigned nbThread):SamplingModule(cdmV,kernel)
	{
		_k=k;
		_convertionTypeVector=convertionTypeVector;
		_variablesCoeficient=variablesCoeficient;
		_errors.resize(nbThread,nullptr);
		_encodedPosition.resize(nbThread,nullptr);
		_completeTIs=completeTIs;
		for (int i = 0; i < nbThread; ++i)
		{
			_errors[i]=(float*)malloc(_cdmV[0].size() * int(ceil(_k)) * sizeof(float));
			_encodedPosition[i]=(unsigned*)malloc(_cdmV[0].size() * int(ceil(_k)) * sizeof(unsigned));
		}
		
	}
	~QuantileSamplingModule(){
		for (int i = 0; i < _errors.size(); ++i)
		{
			free(_errors[i]);
			_errors[i]=nullptr;
		}
		
		for (int i = 0; i < _encodedPosition.size(); ++i)
		{
			free(_encodedPosition[i]);
			_encodedPosition[i]=nullptr;
		}
	};

	inline void sample_complet(matchLocation *result, std::vector<std::vector<int>> neighborArrayVector, std::vector<std::vector<float> > neighborValueArrayVector, unsigned moduleID=0, bool fullStationary=false,
		g2s::DataImage* withSpecificKernel=nullptr, std::vector<unsigned>* exclusionList=nullptr, unsigned sourceImage=-1){
		unsigned vectorSize=_cdmV[moduleID].size();
		float *errors=_errors[moduleID];
		unsigned* encodedPosition=_encodedPosition[moduleID];
		bool updated[vectorSize];
		memset(updated,false,vectorSize);

		std::vector<std::vector<float> > convertedNeighborValueArrayVector(neighborArrayVector.size());
		std::vector<float> cummulatedVariablesCoeficient;

		//if(_convertionTypeVector[0].size()!=neighborValueArrayVector[0].size()) //to redo
		//	fprintf(stderr, "%s %d vs %d\n", "failure",_convertionTypeVector[0].size(),neighborValueArrayVector[0].size());
		if(withSpecificKernel==nullptr)withSpecificKernel=withSpecificKernel;
		unsigned sizeDimsKernel=withSpecificKernel->dataSize()/withSpecificKernel->_nbVariable;
		unsigned indexCenter=0;
		for (int i =  withSpecificKernel->_dims.size()-1; i>=0 ; i--)
		{
			indexCenter=indexCenter*withSpecificKernel->_dims[i]+withSpecificKernel->_dims[i]/2;
		}

		for (int i = 0; i < _convertionTypeVector.size(); ++i)
		{

			for (int j = 0; j < _convertionTypeVector[i].size(); ++j)
			{
				switch(_convertionTypeVector[i][j]){
					case convertionType::P0:
						for (int k = 0; k < neighborArrayVector.size(); ++k)
						{
							unsigned indexInKernel=indexCenter;
							if(withSpecificKernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]) && !std::isnan(neighborValueArrayVector[k][i]))
								convertedNeighborValueArrayVector[k].push_back(withSpecificKernel->_data[indexInKernel*withSpecificKernel->_nbVariable+i]*1.f);
							else
								convertedNeighborValueArrayVector[k].push_back(0.f);
						}
						cummulatedVariablesCoeficient.push_back(_variablesCoeficient[i][j]);
					break;
					case convertionType::P1:
						for (int k = 0; k < neighborArrayVector.size(); ++k)
						{
							unsigned indexInKernel=indexCenter;
							//fprintf(stderr, "%d ==> %f\n", _kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]) && !std::isnan(neighborValueArrayVector[k][i]),neighborValueArrayVector[k][i]);
							if(withSpecificKernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]) && !std::isnan(neighborValueArrayVector[k][i]))
								convertedNeighborValueArrayVector[k].push_back(withSpecificKernel->_data[indexInKernel*withSpecificKernel->_nbVariable+i]*neighborValueArrayVector[k][i]);
							else
								convertedNeighborValueArrayVector[k].push_back(0.f);
						}
						cummulatedVariablesCoeficient.push_back(_variablesCoeficient[i][j]);
					break;
					case convertionType::P2:
						for (int k = 0; k < neighborArrayVector.size(); ++k)
						{
							unsigned indexInKernel=indexCenter;
							if(withSpecificKernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]) && !std::isnan(neighborValueArrayVector[k][i]))
								convertedNeighborValueArrayVector[k].push_back(withSpecificKernel->_data[indexInKernel*withSpecificKernel->_nbVariable+i]*neighborValueArrayVector[k][i]*neighborValueArrayVector[k][i]);
							else
								convertedNeighborValueArrayVector[k].push_back(0.f);
						}
						cummulatedVariablesCoeficient.push_back(_variablesCoeficient[i][j]);
					break;
				}	
			}
		}

		float delta=0;
		if(_completeTIs)
		{
			for (int i = 0; i < _variablesCoeficient.size(); ++i)
			{
				if(_convertionTypeVector[i].size()<_variablesCoeficient[i].size()){
					float coef=_variablesCoeficient[i].back();
					for (int k = 0; k < neighborArrayVector.size(); ++k)
					{
						unsigned indexInKernel;
						if(withSpecificKernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]) && !std::isnan(neighborValueArrayVector[k][i]))
							delta+=coef*withSpecificKernel->_data[indexInKernel*withSpecificKernel->_nbVariable+i]*neighborValueArrayVector[k][i]*neighborValueArrayVector[k][i];
					}
				}
			}
		}

		for (int i = 0; i < vectorSize; ++i)
		{
			updated[i]=_cdmV[moduleID][i]->candidateForPatern(neighborArrayVector, convertedNeighborValueArrayVector, cummulatedVariablesCoeficient,delta);
		}

		int extendK=int(ceil(_k));
		std::fill(errors,errors+vectorSize*extendK,-INFINITY);

		for (int i = 0; i < vectorSize; ++i)
		{
			if(sourceImage==i)
			{
				int delta=int(_cdmV[moduleID][i]->cvtIndexToPosition(_cdmV[moduleID][i]->getErrorsArraySize()))-int(_cdmV[moduleID][i]->getErrorsArraySize());
				/*for (int k = 0; k < exclusionList->size(); ++k)
				{
					_cdmV[moduleID][i]->getErrorsArraySize()[exclusionList->at(k)+delta]=-INFINITY;
				}*/
			}
			if(updated[i])
			{
				float* errosArray=_cdmV[moduleID][i]->getErrorsArray();
				float* crossErrosArray=_cdmV[moduleID][i]->getCossErrorArray();

				if(!_completeTIs)
				{
					#pragma omp simd
					for (int j = 0; j < _cdmV[moduleID][i]->getErrorsArraySize(); ++j)
					{
						errosArray[j]=std::fabs(errosArray[j]/crossErrosArray[j]);
					}
				}
				fKst::findKBigest(errosArray,_cdmV[moduleID][i]->getErrorsArraySize(),extendK, errors+i*extendK, encodedPosition+i*extendK);
			}
		}

		unsigned localPosition[extendK*vectorSize];
		std::iota(localPosition,localPosition+extendK*vectorSize,0);
		
		std::sort(localPosition,localPosition+extendK*vectorSize,[errors](unsigned a, unsigned b){
			return errors[a] > errors[b];
		});

		for (int i = 0; i < int(ceil(_k)); ++i)
		{
			result[i].TI=localPosition[i]/extendK;
			result[i].index=_cdmV[moduleID][result[i].TI]->cvtIndexToPosition(encodedPosition[localPosition[i]]);
		}
	}

	inline matchLocation sample(std::vector<std::vector<int>> neighborArrayVector, std::vector<std::vector<float> > neighborValueArrayVector,float seed, unsigned moduleID=0, bool fullStationary=false){
		unsigned numberOfCandidate=int(ceil(_k));
		matchLocation result[numberOfCandidate];

		sample_complet(result, neighborArrayVector, neighborValueArrayVector, moduleID, fullStationary);
		return result[int(floor(seed*_k))];
	}

	narrownessMeasurment narrowness(std::vector<std::vector<int>> neighborArrayVector, std::vector<std::vector<float> > neighborValueArrayVector,float seed, unsigned moduleID=0, bool fullStationary=false){
		unsigned vectorSize=_cdmV[moduleID].size();
		float *errors=_errors[moduleID];
		unsigned* encodedPosition=_encodedPosition[moduleID];
		bool updated[vectorSize];
		memset(updated,false,vectorSize);

		std::vector<std::vector<float> > convertedNeighborValueArrayVector(neighborArrayVector.size());
		std::vector<float> cummulatedVariablesCoeficient;

		//if(_convertionTypeVector[0].size()!=neighborValueArrayVector[0].size()) //to redo
		//	fprintf(stderr, "%s %d vs %d\n", "failure",_convertionTypeVector[0].size(),neighborValueArrayVector[0].size());

		unsigned sizeDimsKernel=_kernel->dataSize()/_kernel->_nbVariable;
		unsigned indexCenter=0;
		for (int i =  _kernel->_dims.size()-1; i>=0 ; i--)
		{
			indexCenter=indexCenter*_kernel->_dims[i]+_kernel->_dims[i]/2;
		}

		for (int i = 0; i < _convertionTypeVector.size(); ++i)
		{

			for (int j = 0; j < _convertionTypeVector[i].size(); ++j)
			{
				switch(_convertionTypeVector[i][j]){
					case convertionType::P0:
						for (int k = 0; k < neighborArrayVector.size(); ++k)
						{
							unsigned indexInKernel=indexCenter;
							if(_kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]) && !std::isnan(neighborValueArrayVector[k][i]))
								convertedNeighborValueArrayVector[k].push_back(_kernel->_data[indexInKernel*_kernel->_nbVariable+i]*1.f);
							else
								convertedNeighborValueArrayVector[k].push_back(0.f);
						}
						cummulatedVariablesCoeficient.push_back(_variablesCoeficient[i][j]);
					break;
					case convertionType::P1:
						for (int k = 0; k < neighborArrayVector.size(); ++k)
						{
							unsigned indexInKernel=indexCenter;
							if(_kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]) && !std::isnan(neighborValueArrayVector[k][i]))
								convertedNeighborValueArrayVector[k].push_back(_kernel->_data[indexInKernel*_kernel->_nbVariable+i]*neighborValueArrayVector[k][i]);
							else
								convertedNeighborValueArrayVector[k].push_back(0.f);
						}
						cummulatedVariablesCoeficient.push_back(_variablesCoeficient[i][j]);
					break;
					case convertionType::P2:
						for (int k = 0; k < neighborArrayVector.size(); ++k)
						{
							unsigned indexInKernel=indexCenter;
							if(_kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]) && !std::isnan(neighborValueArrayVector[k][i]))
								convertedNeighborValueArrayVector[k].push_back(_kernel->_data[indexInKernel*_kernel->_nbVariable+i]*neighborValueArrayVector[k][i]*neighborValueArrayVector[k][i]);
							else
								convertedNeighborValueArrayVector[k].push_back(0.f);
						}
						cummulatedVariablesCoeficient.push_back(_variablesCoeficient[i][j]);
					break;
				}	
			}
		}

		float delta=0;
		if(_completeTIs)
		{
			for (int i = 0; i < _variablesCoeficient.size(); ++i)
			{
				if(_convertionTypeVector[i].size()<_variablesCoeficient[i].size()){
					float coef=_variablesCoeficient[i].back();
					for (int k = 0; k < neighborArrayVector.size(); ++k)
					{
						unsigned indexInKernel;
						if(_kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]) && !std::isnan(neighborValueArrayVector[k][i]))
							delta+=coef*_kernel->_data[indexInKernel*_kernel->_nbVariable+i]*neighborValueArrayVector[k][i]*neighborValueArrayVector[k][i];
					}
				}
			}
		}

		for (int i = 0; i < vectorSize; ++i)
		{
			updated[i]=_cdmV[moduleID][i]->candidateForPatern(neighborArrayVector, convertedNeighborValueArrayVector, cummulatedVariablesCoeficient,delta);
		}

		int extendK=int(ceil(_k));
		std::fill(errors,errors+vectorSize*extendK,-INFINITY);

		for (int i = 0; i < vectorSize; ++i)
		{
			if(updated[i])
			{
				float* errosArray=_cdmV[moduleID][i]->getErrorsArray();
				float* crossErrosArray=_cdmV[moduleID][i]->getCossErrorArray();

				if(!_completeTIs)
				{
					#pragma omp simd
					for (int j = 0; j < _cdmV[moduleID][i]->getErrorsArraySize(); ++j)
					{
						errosArray[j]=std::fabs(errosArray[j]/crossErrosArray[j]);
					}
				}
				fKst::findKBigest(errosArray,_cdmV[moduleID][i]->getErrorsArraySize(),extendK, errors+i*extendK, encodedPosition+i*extendK);
			}
		}

		unsigned localPosition[extendK*vectorSize];
		std::iota(localPosition,localPosition+extendK*vectorSize,0);
		std::sort(localPosition,localPosition+extendK*vectorSize,[errors](unsigned a, unsigned b){
			return errors[a] > errors[b];
		});
		//fKst::findKSmallest(_errors,3,extendK, localErrors, localPosition);

		unsigned slectedIndex=int(floor(seed*_k));
		unsigned selectedTI=localPosition[slectedIndex]/extendK;
		unsigned indexInTI=_cdmV[moduleID][selectedTI]->cvtIndexToPosition(encodedPosition[localPosition[slectedIndex]]);

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
};

#endif // QUANTILE_SAMPLING_MODULE_HPP