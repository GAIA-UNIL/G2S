#ifndef QUANTILE_SAMPLING_MODULE_HPP
#define QUANTILE_SAMPLING_MODULE_HPP

#include "samplingModule.hpp"
#include "fKst.hpp"




class QuantileSamplingModule: public SamplingModule {
public:
	enum convertionType{
		P0=0,
		P1=1,
		P2=2
	};
private:
	float _k;
	std::vector<std::vector<convertionType> > _convertionTypeVector;
	std::vector<std::vector<float> > _variablesCoeficient;

	float* _errors=nullptr;
	unsigned* _encodedPosition=nullptr;
public:
	QuantileSamplingModule(std::vector<ComputeDeviceModule *> *cdmV, g2s::DataImage* kernel, float k,  std::vector<std::vector<convertionType> > convertionTypeVector, std::vector<std::vector<float> > variablesCoeficient):SamplingModule(cdmV,kernel)
	{
		_k=k;
		_convertionTypeVector=convertionTypeVector;
		_variablesCoeficient=variablesCoeficient;
		_errors=(float*)malloc(_cdmV[0].size() * int(ceil(_k)) * sizeof(float));
		_encodedPosition=(unsigned*)malloc(_cdmV[0].size() * int(ceil(_k)) * sizeof(unsigned));
	}
	~QuantileSamplingModule(){
		free(_errors);
		free(_encodedPosition);
	};

	inline matchLocation sample(std::vector<std::vector<int>> neighborArrayVector, std::vector<std::vector<float> > neighborValueArrayVector,float seed, unsigned moduleID=0, bool fullStationary=false){
		unsigned vectorSize=_cdmV[moduleID].size();
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
							if(_kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]))
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
							if(_kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]))
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
							if(_kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]))
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
		for (int i = 0; i < _variablesCoeficient.size(); ++i)
		{
			if(_convertionTypeVector[i].size()<_variablesCoeficient[i].size()){
				float coef=_variablesCoeficient[i].back();
				for (int k = 0; k < neighborArrayVector.size(); ++k)
				{
					unsigned indexInKernel;
					if(_kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k]))
						delta+=coef*_kernel->_data[indexInKernel*_kernel->_nbVariable+i]*neighborValueArrayVector[k][i]*neighborValueArrayVector[k][i];
				}
			}
		}

		for (int i = 0; i < vectorSize; ++i)
		{
			_cdmV[moduleID][i]->candidateForPatern(neighborArrayVector, convertedNeighborValueArrayVector, cummulatedVariablesCoeficient,delta);
			updated[i]=true;
		}

		int extendK=int(ceil(_k));
		std::fill(_errors,_errors+vectorSize*extendK,INFINITY);

		for (int i = 0; i < vectorSize; ++i)
		{
			if(updated[i])
			{
				fKst::findKSmallest(_cdmV[moduleID][i]->getErrorsArray(),_cdmV[moduleID][i]->getErrorsArraySize(),extendK, _errors+i*extendK, _encodedPosition+i*extendK);
			}
		}

		unsigned localPosition[extendK*vectorSize];
		std::iota(localPosition,localPosition+extendK*vectorSize,0);
		float *errors=_errors;
		std::sort(localPosition,localPosition+extendK*vectorSize,[&errors](unsigned a, unsigned b){
			return errors[a] < errors[b];
		});
		//fKst::findKSmallest(_errors,3,extendK, localErrors, localPosition);

		unsigned slectedIndex=int(floor(seed*_k));
		unsigned selectedTI=localPosition[slectedIndex]/extendK;
		//fprintf(stderr, "%d %d %d %d\n", selectedTI,localPosition[slectedIndex],slectedIndex,extendK);
		unsigned indexInTI=_cdmV[moduleID][selectedTI]->cvtIndexToPosition(_encodedPosition[localPosition[slectedIndex]]);

		matchLocation result;
		result.TI=selectedTI;
		result.index=indexInTI;
		return result;
	}
};

#endif // QUANTILE_SAMPLING_MODULE_HPP