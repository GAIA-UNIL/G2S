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

#ifndef ACCELERATOR_DEVICE_HPP
#define ACCELERATOR_DEVICE_HPP

#include "computeDeviceModule.hpp"
#ifdef WITH_MKL
	#include <fftw3_mkl.h>
#else
	#include <fftw3.h>
#endif

//#if PRECISION == 1
 #define FFTW_PRECISION(Name) fftwf_##Name
 #define dataType_g2s float
/*#else
 #define FFTW_PRECISION(Name) fftw_##Name
 #define dataType_g2s double
#endif*/


class AcceleratorDevice : public ComputeDeviceModule
{
public:
	AcceleratorDevice(SharedMemoryManager* sharedMemoryManager,std::vector<g2s::OperationMatrix> coeficientMatrix, unsigned int threadRatio=1, bool withCrossMesurement=false, bool circularTI=false);
	~AcceleratorDevice();

	void initDim();

	virtual bool candidateForPatern(std::vector<std::vector<int> > &neighborArrayVector, std::vector<std::vector<float> >  &neighborValueArrayVector, std::vector<float> &variablesCoeficient, std::vector<float> delta0);
	virtual std::vector<g2s::spaceFrequenceMemoryAddress> allocAndInitSharedMemory(std::vector<void* > srcMemoryAdress, std::vector<unsigned> srcSize, std::vector<unsigned> fftSize)=0;
	virtual std::vector<g2s::spaceFrequenceMemoryAddress> freeSharedMemory(std::vector<g2s::spaceFrequenceMemoryAddress> sharedMemoryAdress)=0;

	virtual dataType_g2s* getErrorsArray();
	virtual dataType_g2s* getArray(unsigned);
	virtual unsigned getErrorsArraySize()=0;
	virtual unsigned getArraySize()=0;
	virtual dataType_g2s* getCossErrorArray();
	virtual float getErrorAtPosition(unsigned)=0;
	virtual float getValueAtPosition(unsigned, unsigned)=0;
	virtual float getCroossErrorAtPosition(unsigned)=0;
	virtual unsigned cvtIndexToPosition(unsigned)=0;
	virtual unsigned cvtPositionToIndex(unsigned)=0;
	virtual void setTrueMismatch(bool value);

	virtual void maskCroossError();
	virtual void maskCroossErrorWithVariable(unsigned );
	virtual void maskLayerWithVariable(unsigned, unsigned )=0;

	virtual void setValueInErrorArray(unsigned position, float value)=0;
	virtual void compensateMissingData()=0;
	virtual void searchKBigest(float* errors,unsigned *encodedPosition, unsigned extendK, float seed)=0;

	virtual void zerosFrenquencySpaceOutputArray(unsigned layer)=0;
	virtual void computeFreqMismatchMap(std::vector<std::vector<int> > neighborArray, std::vector<std::vector<float> >  &neighborValueArrayVector)=0;
	virtual void computeRealMissmatchAndRemoveWrongPattern(float* delta0)=0;

protected:
	unsigned _threadRatio=1;

	std::vector<g2s::spaceFrequenceMemoryAddress> _srcCplx;

	std::vector<dataType_g2s*> _realSpaceArray;

	std::vector<int> _min,_max;// to init
	std::vector<unsigned> _fftSize;// to init
	std::vector<unsigned> _srcSize;// to init
	unsigned _fftSpaceSize;// to init
	unsigned _realSpaceSize;

	bool _trueMismatch=true;
	bool _crossMesurement=false;
	bool _circularTI=false;
};

#endif