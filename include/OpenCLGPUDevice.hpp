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

#ifndef OPENCL_GPU_DEVICE_HPP
#define OPENCL_GPU_DEVICE_HPP

#include "computeDeviceModule.hpp"
#ifdef WITH_MKL
	#include <fftw3_mkl.h>
#else
	#include <fftw3.h>
#endif

#include <clFFT.h>

//#if PRECISION == 1
 #define FFTW_PRECISION(Name) fftwf_##Name
 #define dataType_g2s float
/*#else
 #define FFTW_PRECISION(Name) fftw_##Name
 #define dataType_g2s double
#endif*/


class OpenCLGPUDevice : public ComputeDeviceModule
{
public:
	static std::vector<unsigned> DeviceWithHostUnifiedMemory(unsigned platform_id);


	OpenCLGPUDevice(SharedMemoryManager* sharedMemoryManager, std::vector<g2s::OperationMatrix> coeficientMatrix, unsigned int platform, unsigned int device, bool withCrossMesurement=false, bool circularTI=false);
	~OpenCLGPUDevice();

	bool candidateForPatern(std::vector<std::vector<int> > &neighborArrayVector, std::vector<std::vector<float> >  &neighborValueArrayVector, std::vector<float> &variablesCoeficient, std::vector<float> delta0);
	std::vector<g2s::spaceFrequenceMemoryAddress> allocAndInitSharedMemory(std::vector<void* > srcMemoryAdress, std::vector<unsigned> srcSize, std::vector<unsigned> fftSize);
	std::vector<g2s::spaceFrequenceMemoryAddress> freeSharedMemory(std::vector<g2s::spaceFrequenceMemoryAddress> sharedMemoryAdress);

	dataType_g2s* getErrorsArray();
	dataType_g2s* getArray(unsigned);
	unsigned getErrorsArraySize();
	unsigned getArraySize();
	dataType_g2s* getCossErrorArray();
	float getErrorAtPosition(unsigned);
	float getValueAtPosition(unsigned, unsigned);
	float getCroossErrorAtPosition(unsigned);
	unsigned cvtIndexToPosition(unsigned);
	void setTrueMismatch(bool value);
	unsigned cvtPositionToIndex(unsigned position);

	void maskCroossError();
	void maskCroossErrorWithVariable(unsigned );
	void maskLayerWithVariable(unsigned, unsigned );

private:

	std::vector<g2s::spaceFrequenceMemoryAddress> _srcCplx;

	FFTW_PRECISION(complex)* _frenquencySpaceInput=nullptr;
	std::vector<FFTW_PRECISION(complex)*> _frenquencySpaceOutputArray;
	std::vector<dataType_g2s*> _realSpaceArray;

	FFTW_PRECISION(plan) _pInv;
	FFTW_PRECISION(plan) _pInvCross;
	FFTW_PRECISION(plan) _p;

	FFTW_PRECISION(plan) _pPatchM;
	FFTW_PRECISION(plan)* _pPatchL;

	std::vector<int> _min,_max;// to init
	std::vector<unsigned> _fftSize;// to init
	std::vector<unsigned> _srcSize;// to init
	unsigned _fftSpaceSize;// to init
	unsigned _realSpaceSize;

	// inline unsigned index( std::vector<int> &deltaVect){
	// 	unsigned finalValue=0;
	// 	for (int i = 0; i < deltaVect.size(); ++i)
	// 	{
	// 		unsigned val=deltaVect[i];
	// 		for (int j = 0; j < i; ++j)
	// 		{
	// 			val*=_fftSize[j];
	// 		}
	// 		finalValue+=val;
	// 	}
	// 	return finalValue;
	// }

	inline unsigned index( std::vector<int> &deltaVect){
		unsigned finalValue=deltaVect.back();
		for (int i = deltaVect.size()-2; i >=0; i--)
		{
			finalValue*=_fftSize[i];
			finalValue+=deltaVect[i];
		}
		return finalValue;
	}

	bool _trueMismatch=true;
	bool _crossMesurement=false;
	bool _circularTI=false;

	std::vector<cl_mem> frenquencySpaceOutputArray_d;
	std::vector<cl_mem> realSpaceArray_d;
	cl_mem tmpBuffer_d;

	cl_context ctx = 0;
	cl_command_queue queue = 0;
	clfftPlanHandle planHandle;
};

#endif