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

#ifndef DIRECT_SAMPLING_MODULE_HPP
#define DIRECT_SAMPLING_MODULE_HPP

#include "samplingModule.hpp"

class DirectSamplingModule: public SamplingModule {

private:
	float _k;
	float _threshold;
	bool _completeTIs=true;
	unsigned _nbThreadOverTI=1;
	unsigned _threadRatio=1;
	bool _noVerbatim=false;
	bool _useUniqueTI=false;
	unsigned _nbThread=1;
	std::vector<int* > _externalMemory;
	std::vector<g2s::DataImage> *_TIs;
public:
	
	DirectSamplingModule(std::vector<g2s::DataImage> &TIs, g2s::DataImage* kernel, float threshold, float k,  bool noVerbatim, bool completeTIs, unsigned nbThread, unsigned nbThreadOverTI=1, unsigned threadRatio=1, bool useUniqueTI=false):SamplingModule(nullptr,kernel)
	{
		_TIs=&TIs;
		_k=k;
		_threshold=threshold;
		_completeTIs=completeTIs;
		_useUniqueTI=useUniqueTI;

		_nbThreadOverTI=nbThreadOverTI;
		_threadRatio=threadRatio;
		_noVerbatim=noVerbatim;
		_nbThread=nbThread;

		_externalMemory.resize(_nbThread);
		
		for (int i = 0; i < _nbThread; ++i)
		{
			_externalMemory[i]=(int*)malloc(sizeof(int)*TIs[0]._dims.size());
		}
	}

	~DirectSamplingModule(){
		for (int i = 0; i < _nbThread; ++i)
		{
			free(_externalMemory[i]);
			_externalMemory[i]=nullptr;
		}
	};

	inline matchLocation sample(std::vector<std::vector<int>> neighborArrayVector, std::vector<std::vector<float> > neighborValueArrayVector,float seed, matchLocation verbatimRecord, unsigned moduleID=0, bool fullStationary=false, unsigned variableOfInterest=0, float localk=0.f, int idTI4Sampling=-1, g2s::DataImage* localKernel=nullptr){

		if(localk<=0.f)
			localk=_k;

		std::mt19937 generator;// can be inprouved by only resetting the seed each time
		generator.seed(floor(UINT_MAX*seed));
		std::uniform_real_distribution<float> distribution(0.0,1.0);

		//auto rng = std::bind(distribution, std::ref(generator));

		unsigned numberPixel=(*_TIs)[0].dataSize();

		std::uniform_int_distribution<unsigned> distInImage(0,numberPixel);

		g2s::DataImage* kernel=_kernel;
		if(localKernel!=nullptr)
			kernel=localKernel;

		unsigned indexCenter=0;
		for (int i =  int(kernel->_dims.size()-1); i>=0 ; i--)
		{
			indexCenter=indexCenter*kernel->_dims[i]+kernel->_dims[i]/2;
		}

		float sumError=0;
		for (unsigned l = 0; l < neighborArrayVector.size(); ++l)
		{
			unsigned indexInKernel=indexCenter;
			float kernelValue=1.f;
			if(kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[l],_externalMemory[moduleID]) && !std::isnan(neighborValueArrayVector[l][0]))
				kernelValue=kernel->_data[indexInKernel*kernel->_nbVariable]*1.f;
			sumError+=kernelValue;
		}

		unsigned bestIndex=-1;
		float bestLocalError=INFINITY;
		for (unsigned i = 0; i < unsigned(numberPixel/localk); ++i)
		{
			unsigned index =  distInImage(generator);
			float localSumError=0.f;
			float localSumKernel=1.f;
			for (unsigned l = 0; l < neighborArrayVector.size(); ++l)
			{
				unsigned location=-1;
				float missMatch=1.f;
				unsigned indexInKernel=indexCenter;
				float kernelValue=1.f;
				if(kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[l],_externalMemory[moduleID]))
					kernelValue=kernel->_data[indexInKernel];
				if((*_TIs)[0].indexWithDelta(location, index, neighborArrayVector[l], _externalMemory[moduleID]) && !std::isnan(neighborValueArrayVector[l][0]))
				{
					missMatch=std::fabs((*_TIs)[0]._data[location]-neighborValueArrayVector[l][0]);
	
					localSumError+=kernelValue*missMatch;
					localSumKernel+=kernelValue;
				}
				// else{
				// 	localSumError+=kernelValue*_threshold*_threshold;
				// 	localSumKernel+=kernelValue;
				// }
			}

			if((localSumKernel>0.75*sumError) && (bestLocalError>(localSumError/localSumKernel))){
				bestLocalError=localSumError/localSumKernel;
				bestIndex=index;
			}

			if(bestLocalError<_threshold*_threshold*sumError){
				break;
			}
		}

		matchLocation result;
		result.TI=0;
		result.index=bestIndex;

		return result;
	}


	// not implemented and probably would be never
	narrownessMeasurment narrowness(std::vector<std::vector<int>> neighborArrayVector, std::vector<std::vector<float> > neighborValueArrayVector,float seed, unsigned moduleID=0, bool fullStationary=false){

		narrownessMeasurment result;
		result.narrowness=0.2;
		result.candidate.TI=0;
		result.candidate.index=0;
		return result;
	}
};

#endif // DIRECT_SAMPLING_MODULE_HPP
