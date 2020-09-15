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

//To use only for debugging purpose

#ifndef CPU_THREAD_ACCELERATOR_DEVICE_DEVICE_HPP
#define CPU_THREAD_ACCELERATOR_DEVICE_DEVICE_HPP

#include "AcceleratorDevice.hpp"

//#if PRECISION == 1
 #define FFTW_PRECISION(Name) fftwf_##Name
 #define dataType_g2s float
/*#else
 #define FFTW_PRECISION(Name) fftw_##Name
 #define dataType_g2s double
#endif*/


class CPUThreadAcceleratorDevice : public AcceleratorDevice
{
public:
	CPUThreadAcceleratorDevice(SharedMemoryManager* sharedMemoryManager,std::vector<g2s::OperationMatrix> coeficientMatrix, unsigned int threadRatio=1, bool withCrossMesurement=false, bool circularTI=false);
	~CPUThreadAcceleratorDevice();

	std::vector<g2s::spaceFrequenceMemoryAddress> allocAndInitSharedMemory(std::vector<void* > srcMemoryAdress, std::vector<unsigned> srcSize, std::vector<unsigned> fftSize);
	std::vector<g2s::spaceFrequenceMemoryAddress> freeSharedMemory(std::vector<g2s::spaceFrequenceMemoryAddress> sharedMemoryAdress);

	unsigned getErrorsArraySize();
	unsigned getArraySize();
	float getErrorAtPosition(unsigned);
	float getValueAtPosition(unsigned, unsigned);
	float getCroossErrorAtPosition(unsigned);
	unsigned cvtIndexToPosition(unsigned);
	unsigned cvtPositionToIndex(unsigned);

	void maskLayerWithVariable(unsigned, unsigned );

	void setValueInErrorArray(unsigned position, float value);
	void compensateMissingData();
	void searchKBigest(float* errors,unsigned *encodedPosition, unsigned extendK, float seed);

	void zerosFrenquencySpaceOutputArray(unsigned layer);
	void computeFreqMismatchMap(std::vector<std::vector<int> > neighborArray, std::vector<std::vector<float> >  &neighborValueArrayVector);
	void computeRealMissmatchAndRemoveWrongPattern(float* delta0);

private:

	FFTW_PRECISION(complex)* _frenquencySpaceInput=nullptr;
	std::vector<FFTW_PRECISION(complex)*> _frenquencySpaceOutputArray;

	FFTW_PRECISION(plan) _pInv;
	FFTW_PRECISION(plan) _p;

	FFTW_PRECISION(plan) _pPatchM;
	FFTW_PRECISION(plan)* _pPatchL;

	inline unsigned index( std::vector<int> &deltaVect){
		unsigned finalValue=deltaVect.back();
		for (int i = deltaVect.size()-2; i >=0; i--)
		{
			finalValue*=_fftSize[i];
			finalValue+=deltaVect[i];
		}
		return finalValue;
	}
};

#endif