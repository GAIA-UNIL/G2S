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
#include "AcceleratorDevice.hpp"
#include "sharedMemoryManager.hpp"
#include "utils.hpp"
#include "complexMulti.hpp"

#define PARTIAL_FFT

#ifndef FFTW_PLAN_OPTION
	//FFTW_PATIENT
	#define FFTW_PLAN_OPTION FFTW_ESTIMATE
#endif



AcceleratorDevice::AcceleratorDevice(SharedMemoryManager* sharedMemoryManager,std::vector<g2s::OperationMatrix> coeficientMatrix, unsigned int threadRatio, bool withCrossMesurement, bool circularTI){
	_coeficientMatrix=coeficientMatrix;
	_threadRatio=threadRatio;

	_crossMesurement=withCrossMesurement;
	_circularTI=circularTI;
	
	_sharedMemoryManager=sharedMemoryManager;
}

AcceleratorDevice::~AcceleratorDevice(){
	
}

void AcceleratorDevice::initDim(){
	_fftSize=_sharedMemoryManager->_fftSize;
	_srcSize=_sharedMemoryManager->_srcSize;

	_srcCplx=_sharedMemoryManager->adressSharedMemory(_memoryID);

	// alloc memory

	_fftSpaceSize=1;
	_realSpaceSize=1;

	_min=std::vector<int>(_fftSize.size());
	_max=std::vector<int>(_fftSize.size());

	_fftSpaceSize*=_fftSize.front()/2+1;
	_realSpaceSize*=_fftSize.front();

	for (size_t i = 1; i < _fftSize.size(); ++i)
	{
		_fftSpaceSize*=_fftSize[i];
		_realSpaceSize*=_fftSize[i];
	}
}


//compute function

dataType* AcceleratorDevice::getArray(unsigned arrayIndex){
	return nullptr;
}


dataType* AcceleratorDevice::getErrorsArray(){
	return nullptr;
}

dataType* AcceleratorDevice::getCossErrorArray(){
	return nullptr;
}


void AcceleratorDevice::setTrueMismatch(bool value){
	_trueMismatch=value;
}

bool  AcceleratorDevice::candidateForPatern(std::vector<std::vector<int> > &neighborArrayVector, std::vector<std::vector<float> >  &neighborValueArrayVector, std::vector<float> &variablesCoeficient, std::vector<float> delta0){
	if(neighborValueArrayVector.size()==0)return false;

	for (size_t i = 0; i < _min.size(); ++i)
	{
		_min[i]=0;
		_max[i]=0;
	}
	
	for (size_t i = 0; i < neighborArrayVector.size(); ++i)
	{
		for (size_t j = 0; j < neighborArrayVector[i].size(); ++j)
		{
			if(_min[j]>neighborArrayVector[i][j])_min[j]=neighborArrayVector[i][j];
			if(_max[j]<neighborArrayVector[i][j])_max[j]=neighborArrayVector[i][j];
		}
	}
	
	{
		
		for (size_t dataArrayIndex = 0; dataArrayIndex < _coeficientMatrix.size(); ++dataArrayIndex)
		{
			zerosFrenquencySpaceOutputArray(dataArrayIndex);
		}
		
		std::vector<std::vector<int> > neighborArray=neighborArrayVector;

		//update coordonate
		for (size_t j = 0; j < neighborArray.size(); ++j)
		{
			for (size_t i = 0; i < _min.size(); ++i)
			{
				neighborArray[j][i]-=_min[i];
			}
		}

		computeFreqMismatchMap( neighborArray, neighborValueArrayVector);

		// add
		
		computeRealMissmatchAndRemoveWrongPattern(delta0.data());

	}
	return true;
}


void AcceleratorDevice::maskCroossError(){
	maskCroossErrorWithVariable(0);
}

void AcceleratorDevice::maskCroossErrorWithVariable(unsigned variable){
	maskLayerWithVariable(_realSpaceArray.size()-1,variable);
}

