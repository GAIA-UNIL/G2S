/*
 * G2S (c) by Mathieu Gravey (gravey.mathieu@gmail.com)
 * 
 * G2S is licensed under a
 * Creative Commons Attribution-NonCommercial 4.0 International License.
 * 
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
 */

#include <cstring>
#include "FullMeasureCPUThreadDevice.hpp"
#include "sharedMemoryManager.hpp"
#include "utils.hpp"
#include "complexMulti.hpp"

#define PARTIAL_FFT

#define FFTW_PLAN_OPTION FFTW_PATIENT

#if __cilk
	#define fillVectorized(name, begin, amount, value) name[begin:amount]=value;
#else
	#define fillVectorized(name, begin, amount, value) std::fill(name+begin,name+begin+amount-1,value);
#endif


FullMeasureCPUThreadDevice::FullMeasureCPUThreadDevice(SharedMemoryManager* sharedMemoryManager, unsigned int threadRatio, bool withCrossMesurement){
	_deviceType=DT_cpuThreads;
	int chip,core;
	g2s::rdtscp(&chip, &core);
	_crossMesurement=withCrossMesurement;
	//printf("core %d, chip %d\n",core, chip );
	_deviceID=chip;
	_sharedMemoryManager=sharedMemoryManager;
	sharedMemoryManager->addDevice(this);

	_fftSize=_sharedMemoryManager->_fftSize;
	_srcSize=sharedMemoryManager->_srcSize;

	_srcCplx=_sharedMemoryManager->adressSharedMemory(_memoryID);

	// alloc memory
	_realSpaceSize=1;

	_min=std::vector<int>(_fftSize.size());
	_max=std::vector<int>(_fftSize.size());

	for (int i = 0; i < _fftSize.size()-1; ++i)
	{
		_realSpaceSize*=_fftSize[i];
	}

	_realSpaceSize*=_fftSize.back();

	_realSpace=(dataType*)malloc(_realSpaceSize* sizeof(dataType));
	if(_crossMesurement){
		_realCrossSpace=(dataType*)malloc(_realSpaceSize* sizeof(dataType));
	}
}

FullMeasureCPUThreadDevice::~FullMeasureCPUThreadDevice(){
	_sharedMemoryManager->removeDevice(this);
	free(_realSpace);
	if(_crossMesurement){
		free(_realCrossSpace);
	}
}

std::vector<g2s::spaceFrequenceMemoryAddress> FullMeasureCPUThreadDevice::allocAndInitSharedMemory(std::vector<void* > srcMemoryAdress, std::vector<unsigned> srcSize, std::vector<unsigned> fftSize){
	
	unsigned realSpaceSize=1;

	for (int i = 0; i < fftSize.size()-1; ++i)
	{
		realSpaceSize*=fftSize[i];
	}

	realSpaceSize*=fftSize.back();

	std::vector<g2s::spaceFrequenceMemoryAddress> sharedMemory;
	for (int i = 0; i < srcMemoryAdress.size(); ++i)
	{
		g2s::spaceFrequenceMemoryAddress sharedMemoryAdress;
		sharedMemoryAdress.space=malloc(realSpaceSize * sizeof(dataType));
		memcpy(sharedMemoryAdress.space,srcMemoryAdress[i], realSpaceSize * sizeof(dataType));
		sharedMemoryAdress.fft=nullptr;
	
		sharedMemory.push_back(sharedMemoryAdress);
	}
	return sharedMemory;

}

std::vector<g2s::spaceFrequenceMemoryAddress> FullMeasureCPUThreadDevice::freeSharedMemory(std::vector<g2s::spaceFrequenceMemoryAddress> sharedMemoryAdress){
	for (int i = 0; i < sharedMemoryAdress.size(); ++i)
	{
		free(sharedMemoryAdress[i].space);
	}
	sharedMemoryAdress.clear();
	return sharedMemoryAdress;
}

//compute function

dataType* FullMeasureCPUThreadDevice::getErrorsArray(){
	return _realSpace;
}

float FullMeasureCPUThreadDevice::getErrorAtPosition(unsigned index){
	return _realSpace[index];
}

dataType* FullMeasureCPUThreadDevice::getCossErrorArray(){
	return _realCrossSpace;
}
float FullMeasureCPUThreadDevice::getCroossErrorAtPosition(unsigned index){
	if(_realCrossSpace==nullptr) return std::nanf("0");
	return _realCrossSpace[index];
}

unsigned FullMeasureCPUThreadDevice::getErrorsArraySize(){
	return _realSpaceSize;
}

unsigned FullMeasureCPUThreadDevice::cvtIndexToPosition(unsigned index){
	
	unsigned position=0;
	unsigned divFactor=_realSpaceSize;
	for (int i = _fftSize.size()-1; i>=0; --i)
	{
		divFactor/=_fftSize[i];
		position=position*_srcSize[i] + (_fftSize[i]-(index/(divFactor))%_fftSize[i]-_min[i]-1);
	}

	return position;
}

void FullMeasureCPUThreadDevice::setTrueMismatch(bool value){
	_trueMismatch=value;
}

bool  FullMeasureCPUThreadDevice::candidateForPatern(std::vector<std::vector<int> > &neighborArrayVector, std::vector<std::vector<float> >  &neighborValueArrayVector, std::vector<float> &variablesCoeficient, float delta0){
	
	for (int i = 0; i < _min.size(); ++i)
	{
		_min[i]=0;
		_max[i]=0;
	}
	
	for (int i = 0; i < neighborArrayVector.size(); ++i)
	{
		for (int j = 0; j < neighborArrayVector[i].size(); ++j)
		{
			if(_min[j]>neighborArrayVector[i][j])_min[j]=neighborArrayVector[i][j];
			if(_max[j]<neighborArrayVector[i][j])_max[j]=neighborArrayVector[i][j];
		}
	}

	bool valideData=false;
	for (int i = 0; i < _min.size(); ++i)
	{
		if(_min[i]!=0) valideData=true;
		if(_max[i]!=0) valideData=true;
	}
	if(!valideData){
		return false;
	}
	else
	{
		memset(_realSpace,0,sizeof(dataType) * _realSpaceSize );
		_valueForPositions=neighborValueArrayVector;
		_encodedDeltaPosition.clear();

		for (int i = 0; i < neighborArrayVector.size(); ++i)
		{
			int encoded=0;
			for (int j = neighborArrayVector[i].size()-1; j >=0; ++j)
			{
				encoded+=encoded*_fftSize[j]+neighborArrayVector[i][j];
			}
			_encodedDeltaPosition.push_back(encoded);
		}

		bool lines[_fftSize.back()];
		memset(_frenquencySpaceOutput, 0, _fftSpaceSize * sizeof(FFTW_PRECISION(complex)) );
		std::vector<std::vector<int> > neighborArray=neighborArrayVector;

		
		int begin;
		int end;

		for (int j = neighborArrayVector[i].size()-1; j >=0; ++j)
		{
			begin+=begin*_fftSize[j]+(_min[j]);
			end+=end*_fftSize[j]+_max[j];
		}

		for


		std::vector<unsigned> position(_fftSize.size(),0);

		for (int i = 0; i < _encodedDeltaPosition.size(); ++i)
		{
			for (int j = 0; j < _srcCplx.size()/2; ++j)
			{
				for (int k = -begin; k < _realSpaceSize-end; ++k)
				{
					_realSpace[k]+=(_srcCplx[j].space[k+_encodedDeltaPosition[i]]-_valueForPositions[2*j+1]/_valueForPositions[2*j+0])*(_srcCplx[j].space[k+_encodedDeltaPosition[i]]-_valueForPositions[2*j+1]/_valueForPositions[2*j+0])*_valueForPositions[2*j+0];
				}
			}
		}


		//Remove fobidden/wrong value
		for (int i = 0; i < _fftSize.size(); ++i)
		{
			unsigned blockSize=1;
			for (int j = 0; j < i; ++j)
			{
				blockSize*=_fftSize[j];
			}
			blockSize*=_fftSize[i]-(_srcSize[i]-(_max[i]-_min[i]));

			unsigned delta=1;
			for (int j = 0; j <= i; ++j)
			{
				delta*=_fftSize[j];
			}

			for (int j = 0; j < _realSpaceSize; j+=delta)
			{
				fillVectorized(_realSpace,j,blockSize,INFINITY);
			}
		}

		if(_trueMismatch) // correct value needed
		{
		#if __cilk
			_realSpace[0:_realSpaceSize]=_realSpace[0:_realSpaceSize]/(_realSpaceSize)+delta0;
		#else
			#pragma omp simd
			for (int i = 0; i < _realSpaceSize; ++i)
			{
				_realSpace[i]=_realSpace[i]/(_realSpaceSize)+delta0;
			}

		#endif	
		}


		// cross Mesuremnt 

		if(_crossMesurement){
			FFTW_PRECISION(execute)(_pInvCross);
			//Remove fobidden/wrong value
			for (int i = _fftSize.size()-1; i>=0; --i)
			{
				unsigned blockSize=1;
				for (int j = 0; j < i-1; ++j)
				{
					blockSize*=_fftSize[j];
				}
				blockSize*=_max[i]-_min[i];

				unsigned delta=1;
				for (int j = 0; j <= i; ++j)
				{
					delta*=_fftSize[j];
				}

				for (int j = 0; j < _realSpaceSize; j+=delta)
				{
					fillVectorized(_realCrossSpace,j,blockSize,0.0f);
				}
			}
		
		#if __cilk
			_realCrossSpace[0:_realSpaceSize]/=(_realSpaceSize);
		#else
			#pragma omp simd
			for (int i = 0; i < _realSpaceSize; ++i)
			{
				_realCrossSpace[i]/=(_realSpaceSize);
			}

		#endif	
		}
	}
	return true;
}

