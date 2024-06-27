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

#ifndef FAST_DIRECT_SAMPLING_MODULE_HPP
#define FAST_DIRECT_SAMPLING_MODULE_HPP

#include "samplingModule.hpp"
#include <deque>

// Structure to hold value and its original position
struct ValuePosition {
    float value;
    unsigned position;

    // Comparator for sorting based on the value
    bool operator<(const ValuePosition& other) const {
        return value < other.value;
    }
};

class FastDirectSamplingModule: public SamplingModule {

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

	std::vector<std::vector<unsigned> > _categoricalBuffer;
	std::vector<std::vector<unsigned> > _categoricalPositionBuffer;

	std::vector<float> _continuousValueBuffer;
	std::vector<unsigned> _continuousPositionBuffer;



private:
	void initalizeImageBuffer(std::vector<g2s::DataImage> &TIs){
		if(TIs[0]._types[0]==g2s::DataImage::VaraibleType::Categorical){
			// values are sorted by construction

			for (unsigned i = 0; i < TIs[0].dataSize(); ++i)
			{
				unsigned val=unsigned(TIs[0]._data[i]);
				if(_categoricalBuffer.size()<=(val+1))
				{
					_categoricalBuffer.resize(val+1);
					_categoricalPositionBuffer.resize(val+1);
					_categoricalPositionBuffer[val].resize(TIs[0].dataSize(),0);
				}
				for (int j = 0; j < _categoricalPositionBuffer.size(); ++j)
				{
					_categoricalPositionBuffer[j][i]=_categoricalBuffer[j].size();
				}
				_categoricalBuffer[val].push_back(i);

			}


		}
		if(TIs[0]._types[0]==g2s::DataImage::VaraibleType::Continuous){
			std::vector<ValuePosition> combinedBuffer;
			combinedBuffer.reserve(TIs[0].dataSize());
			// can be optimised but for the moment we keep like this maybe in the future it would be worse it to change to manage multiple variable
			for (unsigned i = 0; i < TIs[0].dataSize(); ++i)
			{
				combinedBuffer.push_back({TIs[0]._data[i], i});
			}
			
			// Sort the combined buffer based on values
			std::sort(combinedBuffer.begin(), combinedBuffer.end());

			_continuousValueBuffer.resize(TIs[0].dataSize());
			_continuousPositionBuffer.resize(TIs[0].dataSize());
			// Extract the sorted values and positions back into the original buffers
			for (size_t i = 0; i < combinedBuffer.size(); ++i) {
				_continuousValueBuffer[i] = combinedBuffer[i].value;
				_continuousPositionBuffer[i] = combinedBuffer[i].position;
			}

		}
	}

public:
	
	FastDirectSamplingModule(std::vector<g2s::DataImage> &TIs, g2s::DataImage* kernel, float threshold, float k,  bool noVerbatim, bool completeTIs, unsigned nbThread, unsigned nbThreadOverTI=1, unsigned threadRatio=1, bool useUniqueTI=false):SamplingModule(nullptr,kernel)
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

		initalizeImageBuffer(TIs);

	}

	~FastDirectSamplingModule(){
		for (int i = 0; i < _nbThread; ++i)
		{
			free(_externalMemory[i]);
			_externalMemory[i]=nullptr;
		}
	};

	inline matchLocation sample(std::vector<std::vector<int>> neighborArrayVector, std::vector<std::vector<float> > neighborValueArrayVector,float seed, matchLocation verbatimRecord, unsigned moduleID=0, bool fullStationary=false, unsigned variableOfInterest=0, float localk=0.f, int idTI4Sampling=-1, g2s::DataImage* localKernel=nullptr){
		//fprintf(stderr, "sample\n");
		if(localk<=0.f)
			localk=_k;

		std::mt19937 generator;// can be inprouved by only resetting the seed each time
		generator.seed(floor(UINT_MAX*seed));
		
		unsigned numberPixel=_TIs->at(0).dataSize();

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
			if(kernel->indexWithDelta(indexInKernel, indexCenter, neighborArrayVector[l],_externalMemory[moduleID]))
				kernelValue=kernel->_data[indexInKernel*kernel->_nbVariable]*1.f;
			sumError+=kernelValue;
		}

		int bestIndex = -1;

		//fprintf(stderr, "%d\n", 0);

		if(_TIs->at(0)._types[0]==g2s::DataImage::VaraibleType::Categorical){
			//to optimize variable selection
			//fprintf(stderr, "%d\n", 1);
			std::vector<ValuePosition> importanceVariable;
			importanceVariable.reserve(neighborValueArrayVector.size());
			for (unsigned l = 0; l < neighborValueArrayVector.size(); ++l)
			{
				int value=-1;
				for (int j = 0; j < neighborValueArrayVector[l].size(); ++j)
				{
					if(neighborValueArrayVector[l][j]==1.0f)
					value=j;
				}
				if(value<0)
					continue;

				float dist=0;
				for (int i = 0; i < neighborArrayVector[l].size(); ++i)
				{
					dist+=neighborArrayVector[l][i]*neighborArrayVector[l][i];
				}
				
				float val=-(float(numberPixel)/(_categoricalBuffer[value].size())-std::sqrt(dist)/10.);
				importanceVariable.push_back({val,l});  // +1 is to stybilize
			}
			//std::sort(importanceVariable.begin(), importanceVariable.end());

			std::deque<unsigned> bufferSource;
			std::deque<unsigned> bufferDestination;

			for (unsigned i = 0; i < numberPixel; ++i)
			{
				bufferSource.push_back(i);
			}


			for (unsigned l = 0; l < neighborArrayVector.size(); ++l)
			{
				if(bufferSource.empty())
					break;

				std::uniform_int_distribution<unsigned> distInImage(0,bufferSource.size()-1);
				bestIndex =  bufferSource.at(distInImage(generator));

				int value=-1;
				for (int i = 0; i < neighborValueArrayVector[importanceVariable[l].position].size(); ++i)
				{
					if(neighborValueArrayVector[importanceVariable[l].position][i]==1.0f)
					value=i;
				}
				if(value<0)
					continue;

				int posInBuffer=bufferSource.front();
				bufferSource.pop_front();
				int offset=-_TIs->at(0).offset(neighborArrayVector[importanceVariable[l].position]);
				//_categoricalPositionBuffer[value][std::max(posInBuffer-offset-1,0)]
				for (int indexInCat=_categoricalPositionBuffer[value][std::max(posInBuffer-offset,0)]; indexInCat<_categoricalBuffer[value].size(); indexInCat++){
					int catPos=_categoricalBuffer[value][indexInCat];
					//fprintf(stderr, "%d VS  %d ; %d, offset %d\n", indexInCat, _categoricalPositionBuffer[value][catPos]);
					if(catPos+offset>posInBuffer){
						while(catPos+offset>posInBuffer){
							if(bufferSource.empty()){
								indexInCat=INT_MAX;
								break;
							}
							posInBuffer=bufferSource.front();
							bufferSource.pop_front();
						}
						if(indexInCat==INT_MAX)
						 	break;
						//fprintf(stderr, "%d VS %d \n",indexInCat,  _categoricalPositionBuffer[value][posInBuffer-offset] );
						indexInCat=_categoricalPositionBuffer[value][posInBuffer-offset];
						catPos=_categoricalBuffer[value][indexInCat];
					}
					

					//if(catPos+offset<posInBuffer) continue;
					if(catPos+offset==posInBuffer) {
						unsigned index;
						if(_TIs->at(0).indexWithDelta(index, posInBuffer, neighborArrayVector[importanceVariable[l].position], _externalMemory[moduleID]))
						{
							bufferDestination.push_back(posInBuffer);
						}
					}
				}
				bufferSource.clear();
				bufferDestination.swap(bufferSource);
				//fprintf(stderr, "size %d\n", bufferSource.size());
				if(bufferSource.size()==1)
				{
					bestIndex=bufferSource.front();
					bufferSource.pop_front();
				}
			}
		}

		if(_TIs->at(0)._types[0]==g2s::DataImage::VaraibleType::Continuous){
			float bestLocalError=INFINITY;
			float factor=5.f;
			unsigned lower=0;
			unsigned upper=0;

			unsigned refValueIndex=generator()%std::min(neighborValueArrayVector.size()-1,size_t(4))+1;
			// for (int j = 0; j < neighborValueArrayVector.size(); ++j)
			// {
			// 	if(!std::isnan(neighborValueArrayVector[j][0]))
			// 	{
			// 		refValueIndex=j;
			// 		break;
			// 	}
			// }

			//fprintf(stderr, "%f\n", neighborValueArrayVector[refValueIndex][0]);
			for (int i = 0; i < _continuousValueBuffer.size(); ++i)
			{
				if(_continuousValueBuffer[i]<=(neighborValueArrayVector[refValueIndex][0]-_threshold*factor))
				{
					lower=i;
				}
				if(_continuousValueBuffer[i]<=(neighborValueArrayVector[refValueIndex][0]+_threshold*factor))
				{
					upper=i;
				}
			}
			//fprintf(stderr, "[%d, %d]\n", lower, upper);
    		std::uniform_int_distribution<unsigned> distInImage(lower,upper);

    		auto lagVector=neighborArrayVector[refValueIndex];
    		for (int i = 0; i < lagVector.size(); ++i)
    		{
    			lagVector[i]*=-1;
    		}
    		
			for (unsigned i = 0; i < unsigned((upper-lower)/localk); ++i)
			{
				unsigned index;

				if(!_TIs->at(0).indexWithDelta(index, _continuousPositionBuffer[distInImage(generator)], lagVector, _externalMemory[moduleID])){
					continue;
				}

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
					if(_TIs->at(0).indexWithDelta(location, index, neighborArrayVector[l], _externalMemory[moduleID]) && !std::isnan(neighborValueArrayVector[l][0]))
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
		}

		//fprintf(stderr, "bestIndex %d\n", bestIndex);

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

#endif // FAST_DIRECT_SAMPLING_MODULE_HPP
