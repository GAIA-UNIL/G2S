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
#include "DirectMeasureCPUThreadDevice.hpp"
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


DirectMeasureCPUThreadDevice::DirectMeasureCPUThreadDevice(SharedMemoryManager* sharedMemoryManager, unsigned int threadRatio, bool withCrossMesurement){
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
}

DirectMeasureCPUThreadDevice::~DirectMeasureCPUThreadDevice(){
	_sharedMemoryManager->removeDevice(this);
}

std::vector<g2s::spaceFrequenceMemoryAddress> DirectMeasureCPUThreadDevice::allocAndInitSharedMemory(std::vector<void* > srcMemoryAdress, std::vector<unsigned> srcSize, std::vector<unsigned> fftSize){
	
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

std::vector<g2s::spaceFrequenceMemoryAddress> DirectMeasureCPUThreadDevice::freeSharedMemory(std::vector<g2s::spaceFrequenceMemoryAddress> sharedMemoryAdress){
	for (int i = 0; i < sharedMemoryAdress.size(); ++i)
	{
		free(sharedMemoryAdress[i].space);
	}
	sharedMemoryAdress.clear();
	return sharedMemoryAdress;
}

//compute function

dataType* DirectMeasureCPUThreadDevice::getErrorsArray(){
	return nullptr;
}

float DirectMeasureCPUThreadDevice::getErrorAtPosition(unsigned index){

	std::vector<unsigned> position(_fftSize.size(),0);

	unsigned val=index;
	for (int i = 0; i < _fftSize.size(); ++i)
	{
		position[i]=val%_fftSize[i];
		val/=_fftSize[i];
	}

	bool isOkForMesure=true

	for (int i = 0; i < _fftSize.size(); ++i)
	{
		isOkForMesure &= (position[i]>=_min[i]);
		isOkForMesure &= (position[i]<_fftSize[i]-_max[i]);
	}
	if(!isOkForMesure) return std::inf("0");

	float error;

	for (int i = 0; i < _encodedDeltaPosition.size(); ++i)
	{
		for (int j = 0; j < _srcCplx.size()/2; ++j)
		{
			float missmatch=_srcCplx[j].space[index+_encodedDeltaPosition[i]]-_valueForPositions[2*j+1]/_valueForPositions[2*j+0];
			error+=missmatch*missmatch*_valueForPositions[2*j+0];
		}
	}

	return error;
}

dataType* DirectMeasureCPUThreadDevice::getCossErrorArray(){
	return nullptr;
}
float DirectMeasureCPUThreadDevice::getCroossErrorAtPosition(unsigned index){
	std::vector<unsigned> position(_fftSize.size(),0);

	unsigned val=index;
	for (int i = 0; i < _fftSize.size(); ++i)
	{
		position[i]=val%_fftSize[i];
		val/=_fftSize[i];
	}

	bool isOkForMesure=true

	for (int i = 0; i < _fftSize.size(); ++i)
	{
		isOkForMesure &= (position[i]>=_min[i]);
		isOkForMesure &= (position[i]<_fftSize[i]-_max[i]);
	}
	if(!isOkForMesure) return 0.f;

	float error;

	for (int i = 0; i < _encodedDeltaPosition.size(); ++i)
	{
		for (int j = 0; j < _srcCplx.size()/2; ++j)
		{
			error+=_valueForPositions[2*j+0];
		}
	}

	return error;
}

unsigned DirectMeasureCPUThreadDevice::getErrorsArraySize(){
	return _realSpaceSize;
}

unsigned DirectMeasureCPUThreadDevice::cvtIndexToPosition(unsigned index){

	return index;
}

void DirectMeasureCPUThreadDevice::setTrueMismatch(bool value){
	_trueMismatch=value;
}

bool  DirectMeasureCPUThreadDevice::candidateForPatern(std::vector<std::vector<int> > &neighborArrayVector, std::vector<std::vector<float> >  &neighborValueArrayVector, std::vector<float> &variablesCoeficient, float delta0){
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

	}
	return true;
}

