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

#include <cstring>
#include <limits>
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


DirectMeasureCPUThreadDevice::DirectMeasureCPUThreadDevice(SharedMemoryManager* sharedMemoryManager, unsigned int threadRatio, bool withCrossMeasurement){
	_deviceType=DT_cpuThreads;
	int chip,core;
	g2s::rdtscp(&chip, &core);
	_crossMeasurement=withCrossMeasurement;
	//printf("core %d, chip %d\n",core, chip );
	_deviceID=chip;
	_sharedMemoryManager=sharedMemoryManager;
	sharedMemoryManager->addDevice(this);

	_fftSize=_sharedMemoryManager->_fftSize;
	_srcSize=sharedMemoryManager->_srcSize;

	_srcCplx=_sharedMemoryManager->addressSharedMemory(_memoryID);

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

std::vector<g2s::spaceFrequencyMemoryAddress> DirectMeasureCPUThreadDevice::allocAndInitSharedMemory(std::vector<void* > srcMemoryAddress, std::vector<unsigned> srcSize, std::vector<unsigned> fftSize){
	
	unsigned realSpaceSize=1;

	for (int i = 0; i < fftSize.size()-1; ++i)
	{
		realSpaceSize*=fftSize[i];
	}

	realSpaceSize*=fftSize.back();

	std::vector<g2s::spaceFrequencyMemoryAddress> sharedMemory;
	for (int i = 0; i < srcMemoryAddress.size(); ++i)
	{
		g2s::spaceFrequencyMemoryAddress sharedMemoryAddress;
		sharedMemoryAddress.space=malloc(realSpaceSize * sizeof(dataType_g2s));
		memcpy(sharedMemoryAddress.space,srcMemoryAddress[i], realSpaceSize * sizeof(dataType_g2s));
		sharedMemoryAddress.fft=nullptr;
	
		sharedMemory.push_back(sharedMemoryAddress);
	}
	return sharedMemory;

}

std::vector<g2s::spaceFrequencyMemoryAddress> DirectMeasureCPUThreadDevice::freeSharedMemory(std::vector<g2s::spaceFrequencyMemoryAddress> sharedMemoryAddress){
	for (int i = 0; i < sharedMemoryAddress.size(); ++i)
	{
		free(sharedMemoryAddress[i].space);
	}
	sharedMemoryAddress.clear();
	return sharedMemoryAddress;
}

//compute function

dataType_g2s* DirectMeasureCPUThreadDevice::getErrorsArray(){
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

	bool isOkForMesure=true;

	for (int i = 0; i < _fftSize.size(); ++i)
	{
		isOkForMesure &= (position[i]>=_min[i]);
		isOkForMesure &= (position[i]<_fftSize[i]-_max[i]);
	}
	if(!isOkForMesure) return std::numeric_limits<float>::infinity();

	float error=0.f;

	for (int i = 0; i < _encodedDeltaPosition.size(); ++i)
	{
		for (int j = 0; j < _srcCplx.size()/2; ++j)
		{
			float missmatch=((dataType_g2s*)_srcCplx[j].space)[index+_encodedDeltaPosition[i]]-_valueForPositions[i][2*j+1]/_valueForPositions[i][2*j+0];
			error+=missmatch*missmatch*_valueForPositions[i][2*j+0];
		}
	}

	return error;
}

dataType_g2s* DirectMeasureCPUThreadDevice::getCrossErrorArray(){
	return nullptr;
}
float DirectMeasureCPUThreadDevice::getCrossErrorAtPosition(unsigned index){
	std::vector<unsigned> position(_fftSize.size(),0);

	unsigned val=index;
	for (int i = 0; i < _fftSize.size(); ++i)
	{
		position[i]=val%_fftSize[i];
		val/=_fftSize[i];
	}

	bool isOkForMesure=true;

	for (int i = 0; i < _fftSize.size(); ++i)
	{
		isOkForMesure &= (position[i]>=_min[i]);
		isOkForMesure &= (position[i]<_fftSize[i]-_max[i]);
	}
	if(!isOkForMesure) return 0.f;

	float error=0.f;

	for (int i = 0; i < _encodedDeltaPosition.size(); ++i)
	{
		for (int j = 0; j < _srcCplx.size()/2; ++j)
		{
			error+=_valueForPositions[i][2*j+0];
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

bool  DirectMeasureCPUThreadDevice::candidateForPattern(std::vector<std::vector<int> > &neighborArrayVector, std::vector<std::vector<float> >  &neighborValueArrayVector, std::vector<float> &variablesCoefficient, float delta0){
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

	bool validData=false;
	for (int i = 0; i < _min.size(); ++i)
	{
		if(_min[i]!=0) validData=true;
		if(_max[i]!=0) validData=true;
	}
	if(!validData){
		return false;
	}
	else
	{
		_valueForPositions=neighborValueArrayVector;
		_encodedDeltaPosition.clear();

		for (int i = 0; i < neighborArrayVector.size(); ++i)
		{
			int encoded=0;
			for (int j = neighborArrayVector[i].size()-1; j >=0; --j)
			{
				encoded+=encoded*_fftSize[j]+neighborArrayVector[i][j];
			}
			_encodedDeltaPosition.push_back(encoded);
		}
	}
	return true;
}
