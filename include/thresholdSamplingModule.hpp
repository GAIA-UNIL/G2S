#ifndef QUANTILE_SAMPLING_MODULE_HPP
#define QUANTILE_SAMPLING_MODULE_HPP

#include "samplingModule.hpp"
#include "fKst.hpp"




class ThresholdSamplingModule: public SamplingModule {

private:
	float _threshold;
	float _f;
	std::vector<std::vector<convertionType> > _convertionTypeVector;
	std::vector<std::vector<float> > _variablesCoeficient;

	std::mt19937 _randgen;
	std::vector<unsigned> _maxNumberOfElement;
public:
	ThresholdSamplingModule(std::vector<ComputeDeviceModule *> *cdmV, g2s::DataImage* kernel, float threshold, float f, std::vector<std::vector<convertionType> > convertionTypeVector, std::vector<std::vector<float> > variablesCoeficient):SamplingModule(cdmV,kernel)
	{
		_threshold=threshold;
		_f=f;
		_convertionTypeVector=convertionTypeVector;
		_variablesCoeficient=variablesCoeficient;

	}
	~ThresholdSamplingModule(){

	};

	inline matchLocation sample(std::vector<std::vector<int>> neighborArrayVector, std::vector<std::vector<float> > neighborValueArrayVector,float seed, unsigned moduleID=0, bool fullStationary=false){
		
		if(moduleID>=_maxNumberOfElement.size()){
			_maxNumberOfElement.resize(moduleID+1,0);
		}
		if(_maxNumberOfElement[moduleID]==0){
			unsigned numberElement=0;
			for (int i = 0; i < _cdmV[moduleID].size(); ++i)
			{
				
				numberElement+=_cdmV[moduleID][i]->getErrorsArraySize();
				
			}
			_maxNumberOfElement[moduleID]=numberElement;
		}



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
		float deltaKernel=0.f;
		for (int i = 0; i < _variablesCoeficient.size(); ++i)
		{
			if(_convertionTypeVector[i].size()<_variablesCoeficient[i].size()){
				float coef=_variablesCoeficient[i].back();
				for (int k = 0; k < neighborArrayVector.size(); ++k)
				{
					unsigned indexInKernel;
					if(_kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[k])){
						delta+=coef*_kernel->_data[indexInKernel*_kernel->_nbVariable+i]*neighborValueArrayVector[k][i]*neighborValueArrayVector[k][i];
						deltaKernel+=coef*_kernel->_data[indexInKernel*_kernel->_nbVariable+i];
					}
				}
			}
		}

		//_randgen.seed(unsigned(seed*(2<<24)));

		for (int i = 0; i < vectorSize; ++i)
		{
			_cdmV[moduleID][i]->candidateForPatern(neighborArrayVector, convertedNeighborValueArrayVector, cummulatedVariablesCoeficient,delta);
			updated[i]=true;
		}

		unsigned numberElement=0;

		for (int i = 0; i < vectorSize; ++i)
		{
			if(updated[i])
			{
				numberElement+=_cdmV[moduleID][i]->getErrorsArraySize();
			}
		}

		std::uniform_int_distribution<unsigned> uniformDis(0, numberElement-1);

		unsigned cpt=0;

		float bestValue=INFINITY;
		unsigned bestPosition=0;
		unsigned bestImage=0;

		while((bestValue>_threshold) && (cpt<=_maxNumberOfElement[moduleID]*_f)){
			unsigned index=uniformDis(_randgen);
			unsigned imageID=0;
			unsigned numberElementCumul=0;
			for (imageID = 0; imageID < vectorSize; ++imageID)
			{
				numberElementCumul+=_cdmV[moduleID][imageID]->getErrorsArraySize();
				if(index<numberElementCumul) break;
			}

			unsigned positionInImage=index-(numberElementCumul-_cdmV[moduleID][imageID]->getErrorsArraySize());
			float loaclError=_cdmV[moduleID][imageID]->gerErrorAtPosition(positionInImage)/deltaKernel;

			if(loaclError<bestValue){
				bestValue=loaclError;
				bestPosition=positionInImage;
				bestImage=imageID;
			}
			cpt++;
		}

		matchLocation result;
		result.TI=bestImage;
		result.index=_cdmV[moduleID][bestImage]->cvtIndexToPosition(bestPosition);
		return result;
	}
};

#endif // QUANTILE_SAMPLING_MODULE_HPP