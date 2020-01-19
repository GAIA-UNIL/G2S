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

#include <cstring>
#include "NvidiaGPUAcceleratorDevice.hpp"
#include "sharedMemoryManager.hpp"
#include "utils.hpp"
#include <algorithm>

#define PARTIAL_FFT

#ifndef FFTW_PLAN_OPTION
	//FFTW_PATIENT
	#define FFTW_PLAN_OPTION FFTW_ESTIMATE
#endif
// #if __cilk
// 	#define fillVectorized(name, begin, amount, value) name[begin:amount]=value;
// #else
	#define fillVectorized(name, begin, amount, value) std::fill(name+begin,name+begin+amount,value);
// #endif


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
 #define gpuErrchk(ans) { ans; }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

inline void gpuAssert(cublasStatus_t code, const char *file, int line, bool abort=true)
{
	if (code != CUBLAS_STATUS_SUCCESS) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", _cudaGetErrorEnum(code), file, line);
		if (abort) exit(code);
	}
}

static const char *_cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}

inline void gpuAssert(cufftResult code, const char *file, int line, bool abort=true)
{
	if (code != CUFFT_SUCCESS) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", _cudaGetErrorEnum(code), file, line);
		if (abort) exit(code);
	}
}


/* ------------- start kernels -----------------*/

__host__ __device__ static __inline__  cuComplex cuCfmafAlpha( cuComplex x, cuComplex y, cuComplex d, float alpha)
{
	float real_res;
	float imag_res;
	
	real_res = (cuCrealf(x) *  cuCrealf(y))*alpha + cuCrealf(d);
	imag_res = (cuCrealf(x) *  cuCimagf(y))*alpha + cuCimagf(d);
			
	real_res = -(cuCimagf(x) * cuCimagf(y))*alpha  + real_res;  
	imag_res =  (cuCimagf(x) *  cuCrealf(y))*alpha + imag_res;          
	 
	return make_cuComplex(real_res, imag_res);
}

__global__ void complexAddAlphaxCxD(cuFloatComplex* dst, const cuFloatComplex* C, const cuFloatComplex* D, const float alpha, const unsigned int size){

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < size) dst[i]=cuCfmafAlpha(C[i],D[i],dst[i],alpha);
}

__global__ void removeBorder(float* dst, const unsigned int size, const unsigned int delta, const unsigned int blockSize, const float remplace){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int blockId = blockIdx.y * blockDim.y + threadIdx.y;
	int pos = i * delta + blockId;
	// if (pos < size && blockId < blockSize){
	// 	dst[pos]=remplace;
	// }
}

__global__ void fma(float* realSpace, const unsigned int size,  const float alpha,  const float delta){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < size)
		realSpace[i]=fmaf(realSpace[i],alpha,delta);
}

__global__ void compensateMissingDatakernel(float* errosArray, float* crossErrosArray, const unsigned int size, float val){
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	if (j < size)
	{
		errosArray[j]=-std::fabs(errosArray[j]/(crossErrosArray[j]*crossErrosArray[j]*crossErrosArray[j]*crossErrosArray[j]));
		if(crossErrosArray[j]==0.0f) errosArray[j]=val;
	}
}

__global__ void copyAndRemove(float* errosArray, unsigned int*  _encodedPosition_d,float* _mismatch_d,const unsigned int i, const float val){

	_encodedPosition_d[i]-=1;
	if(_mismatch_d){
		_mismatch_d[i]=errosArray[_encodedPosition_d[i]];
	}
	errosArray[_encodedPosition_d[i]]=val;
}	

/* ------------- end kernels -----------------*/


#define cufftPlan(p, rank, n, type) cufftPlanMany(p, rank, n, 0, 1, 1,  0, 1, 1, type, 1)

NvidiaGPUAcceleratorDevice::NvidiaGPUAcceleratorDevice(int deviceId, SharedMemoryManager* sharedMemoryManager,std::vector<g2s::OperationMatrix> coeficientMatrix,
	unsigned int threadRatio, bool withCrossMesurement, bool circularTI)
	:AcceleratorDevice( sharedMemoryManager, coeficientMatrix, threadRatio, withCrossMesurement, circularTI)
{
	_deviceType=DT_gpuCuda;
	// int chip,core;
	// g2s::rdtscp(&chip, &core);
	int numberOfDevice;
	gpuErrchk(cudaGetDeviceCount(&numberOfDevice));
	_deviceID=deviceId%numberOfDevice;
	cudaSetDevice(_deviceID);
	
	cudaStreamCreate(&_cudaLocalStream);
	//_cudaLocalStream=0;
	gpuErrchk(cudaEventCreate(&_cudaEventFinal));
	#pragma omp critical (createDevices)
	{
		_sharedMemoryManager->addDevice(this);
	}
	initDim();
	gpuErrchk(cublasCreate(&_cublasHandle));
	gpuErrchk(cublasSetStream(_cublasHandle, _cudaLocalStream));
	gpuErrchk(cublasSetPointerMode(_cublasHandle, CUBLAS_POINTER_MODE_DEVICE));

	gpuErrchk( cudaMalloc(&_frenquencySpaceInput,_fftSpaceSize * sizeof(cufftComplex)));

	for (size_t i = 0; i < _coeficientMatrix.size(); ++i)
	{
		cufftComplex* ptrCplx;
		gpuErrchk( cudaMalloc(&ptrCplx,_fftSpaceSize * sizeof(cufftComplex)));
		_frenquencySpaceOutputArray.push_back(ptrCplx);
		float* ptrReal;
		gpuErrchk( cudaMalloc(&ptrReal,_realSpaceSize* sizeof(dataType)));
		_realSpaceArray.push_back(ptrReal);
	}

	std::vector<int> reverseFftSize(_fftSize.begin(),_fftSize.end());
	std::reverse(reverseFftSize.begin(),reverseFftSize.end());
	{
		gpuErrchk(cufftPlan(&_pInv,reverseFftSize.size(), reverseFftSize.data(),CUFFT_C2R));
		gpuErrchk(cufftSetStream(_pInv,_cudaLocalStream));

		gpuErrchk(cufftPlan(&_p,reverseFftSize.size(), reverseFftSize.data(),CUFFT_R2C));
		gpuErrchk(cufftSetStream(_p,_cudaLocalStream));

		if(_fftSize.size()>1){
			unsigned reducedSize=1;

			for (size_t i = 1; i < _fftSize.size()-1; ++i)
			{
				reducedSize*=_fftSize[i];
			}

			_reducedFftSize=reducedSize*(_fftSize.front()/2+1);
			_reducedRealSize=reducedSize*(_fftSize.front());

			gpuErrchk(cufftPlan(&_pPatchL,reverseFftSize.size()-1, reverseFftSize.data()+1,CUFFT_R2C));
			gpuErrchk(cufftSetStream(_pPatchL,_cudaLocalStream));

			gpuErrchk(cufftPlanMany(&_pPatchM, 1, reverseFftSize.data(), reverseFftSize.data(),
				_reducedFftSize, 1, reverseFftSize.data(),
				_reducedFftSize, 1, CUFFT_R2C, _reducedFftSize));
			gpuErrchk(cufftSetStream(_pPatchM,_cudaLocalStream));

		}

	}
	cudaEventRecord(_cudaEventFinal,_cudaLocalStream);
	cudaEventSynchronize(_cudaEventFinal);
}

NvidiaGPUAcceleratorDevice::~NvidiaGPUAcceleratorDevice(){
	_sharedMemoryManager->removeDevice(this);
	gpuErrchk(cufftDestroy(_pInv));
	gpuErrchk(cufftDestroy(_p));

	if(_fftSize.size()>1){
		gpuErrchk(cufftDestroy(_pPatchM));
		gpuErrchk(cufftDestroy(_pPatchL));
	}

	gpuErrchk(cudaFree(_frenquencySpaceInput));
	for (size_t i = 0; i < _frenquencySpaceOutputArray.size(); ++i)
	{
		gpuErrchk(cudaFree(_frenquencySpaceOutputArray[i]));
		_frenquencySpaceOutputArray[i]=nullptr;
	}
	for (size_t i = 0; i < _realSpaceArray.size(); ++i)
	{
		gpuErrchk(cudaFree(_realSpaceArray[i]));
		_realSpaceArray[i]=nullptr;
	}

	if(_encodedPosition_d){
		gpuErrchk(cudaFree(_encodedPosition_d));
		gpuErrchk(cudaFree(_mismatch_d));
		_encodedPosition_d=nullptr;
		_mismatch_d=nullptr;
		_encodedPosition_dSize=0;	
	}

	gpuErrchk(cudaEventDestroy(_cudaEventFinal));

	gpuErrchk(cublasDestroy(_cublasHandle));
	if(_cudaLocalStream!=0)
		gpuErrchk(cudaStreamDestroy(_cudaLocalStream));
}

std::vector<g2s::spaceFrequenceMemoryAddress> NvidiaGPUAcceleratorDevice::allocAndInitSharedMemory(std::vector<void* > srcMemoryAdress, std::vector<unsigned> srcSize, std::vector<unsigned> fftSize){
	
	//fprintf(stderr, "alloc shared memory CPU\n");
	cudaError_t cudaError;
	unsigned fftSpaceSize=1;
	unsigned realSpaceSize=1;

	fftSpaceSize*=fftSize.front()/2+1;
	realSpaceSize*=fftSize.front();

	for (size_t i = 1; i < fftSize.size(); ++i)
	{
		fftSpaceSize*=fftSize[i];
		realSpaceSize*=fftSize[i];
	}

	std::vector<int> reverseFftSize(fftSize.begin(),fftSize.end());
	std::reverse(reverseFftSize.begin(),reverseFftSize.end());

	std::vector<g2s::spaceFrequenceMemoryAddress> sharedMemory;
	for (size_t i = 0; i < srcMemoryAdress.size(); ++i)
	{
		g2s::spaceFrequenceMemoryAddress sharedMemoryAdress;
		gpuErrchk(cudaMalloc(&sharedMemoryAdress.space, realSpaceSize * sizeof(dataType)));
		gpuErrchk(cudaMemcpyAsync(sharedMemoryAdress.space, srcMemoryAdress[i], realSpaceSize * sizeof(dataType), cudaMemcpyHostToDevice, _cudaLocalStream));
		gpuErrchk(cudaMalloc(&sharedMemoryAdress.fft, fftSpaceSize * sizeof(cufftComplex)));
		
		sharedMemory.push_back(sharedMemoryAdress);

		cufftHandle p;
			
		cufftResult cufftError;
		gpuErrchk(cufftPlan(&p,reverseFftSize.size(), reverseFftSize.data(),CUFFT_R2C));
		gpuErrchk(cufftSetStream(p,_cudaLocalStream));
		//FFTW_PRECISION(plan_dft_r2c)(reverseFftSize.size(), reverseFftSize.data(), (dataType*)sharedMemoryAdress.space, (cufftComplex*)sharedMemoryAdress.fft, FFTW_ESTIMATE);
		gpuErrchk(cufftExecR2C(p, (dataType*)sharedMemoryAdress.space, (cufftComplex*)sharedMemoryAdress.fft));
		gpuErrchk(cufftDestroy(p));
	}
	return sharedMemory;

}

std::vector<g2s::spaceFrequenceMemoryAddress> NvidiaGPUAcceleratorDevice::freeSharedMemory(std::vector<g2s::spaceFrequenceMemoryAddress> sharedMemoryAdress){
	for (size_t i = 0; i < sharedMemoryAdress.size(); ++i)
	{
		gpuErrchk(cudaFree(sharedMemoryAdress[i].space));
		gpuErrchk(cudaFree(sharedMemoryAdress[i].fft));
	}
	sharedMemoryAdress.clear();
	return sharedMemoryAdress;
}

//compute function

unsigned NvidiaGPUAcceleratorDevice::getArraySize(){
	return _realSpaceSize;
}

float NvidiaGPUAcceleratorDevice::getValueAtPosition(unsigned arrayIndex, unsigned index){
	float tmp;
	gpuErrchk(cudaMemcpyAsync(_realSpaceArray[arrayIndex]+index, &tmp, sizeof(float), cudaMemcpyDeviceToHost, _cudaLocalStream));
	gpuErrchk(cudaStreamSynchronize(_cudaLocalStream));
	return tmp;
}

float NvidiaGPUAcceleratorDevice::getErrorAtPosition(unsigned index){	
	float tmp;
	gpuErrchk(cudaMemcpyAsync(_realSpaceArray.front()+index, &tmp, sizeof(float), cudaMemcpyDeviceToHost, _cudaLocalStream));
	gpuErrchk(cudaStreamSynchronize(_cudaLocalStream));
	return tmp;
}

float NvidiaGPUAcceleratorDevice::getCroossErrorAtPosition(unsigned index){	
	float tmp;
	gpuErrchk(cudaMemcpyAsync(_realSpaceArray.back()+index, &tmp, sizeof(float), cudaMemcpyDeviceToHost, _cudaLocalStream));
	gpuErrchk(cudaStreamSynchronize(_cudaLocalStream));
	return tmp;
}

unsigned NvidiaGPUAcceleratorDevice::getErrorsArraySize(){
	return _realSpaceSize;
}

unsigned NvidiaGPUAcceleratorDevice::cvtIndexToPosition(unsigned index){
	
	unsigned position=0;
	unsigned divFactor=_realSpaceSize;
	for (int i = int(_fftSize.size()-1); i>=0; --i)
	{
		divFactor/=_fftSize[i];
		position=position*_srcSize[i] + ((_fftSize[i]-(index/(divFactor))%_fftSize[i]-_min[i]-1)+_srcSize[i])%_srcSize[i];
	}

	return position;
}

unsigned NvidiaGPUAcceleratorDevice::cvtPositionToIndex(unsigned position){

	unsigned index=0;
	unsigned divFactor=1;
	for (int i = int(_fftSize.size()-1); i>=0; --i)
	{
		divFactor*=_srcSize[i];
	}
	for (int i = int(_fftSize.size()-1); i>=0; --i)
	{
		divFactor/=_srcSize[i];
		index=index*_fftSize[i] + (_fftSize[i]-(position/(divFactor)+_min[i])%_srcSize[i]-1);
	}
	return index;
}

void NvidiaGPUAcceleratorDevice::zerosFrenquencySpaceOutputArray(unsigned layer){
	gpuErrchk(cudaMemsetAsync(_frenquencySpaceOutputArray[layer], 0, _fftSpaceSize * sizeof(cufftComplex), _cudaLocalStream ));
}

void NvidiaGPUAcceleratorDevice::computeFreqMismatchMap(std::vector<std::vector<int> > neighborArray, std::vector<std::vector<float> >  &neighborValueArrayVector){
	
	for (unsigned int var = 0; var <_coeficientMatrix[0].getNumberOfVariable() ; ++var)
	{
		bool lines[_fftSize.back()];

		bool needTobeComputed=false;
		for (size_t dataArrayIndex = 0; dataArrayIndex < _coeficientMatrix.size(); ++dataArrayIndex)
		{
			needTobeComputed|=_coeficientMatrix[dataArrayIndex].needVariableAlongB(var);
		}
		if(!needTobeComputed) return;

		gpuErrchk(cudaMemsetAsync(_realSpaceArray[0],0,sizeof(dataType) * _realSpaceSize, _cudaLocalStream ));
		gpuErrchk(cudaMemsetAsync(_frenquencySpaceInput,0,_fftSpaceSize * sizeof(cufftComplex), _cudaLocalStream ));

		for (size_t i = 0; i < neighborArray.size(); ++i)
		{
			gpuErrchk(cudaMemcpyAsync(_realSpaceArray[0] + index(neighborArray[i]), &neighborValueArrayVector[i][var], sizeof(float), cudaMemcpyHostToDevice, _cudaLocalStream));
			lines[neighborArray[i].back()]=true;
		}

		bool patialFFT=false;

		#ifdef PARTIAL_FFT
		//patialFFT=true;
		#endif

		cufftResult cufftError;
		if(patialFFT && (_fftSize.size()>1)){
			
			for (unsigned int i = 0; i < _fftSize.back(); ++i)
			{
				if(lines[i]){
					gpuErrchk(cufftExecR2C(_pPatchL, _realSpaceArray[0]+i*_reducedRealSize, _frenquencySpaceInput+i*_reducedFftSize));
				}
			}
			gpuErrchk(cufftExecC2C(_p, _frenquencySpaceInput, _frenquencySpaceInput, CUFFT_FORWARD));
		}else{
			gpuErrchk(cufftExecR2C(_p, _realSpaceArray[0], _frenquencySpaceInput));
		}


		for (size_t dataArrayIndex = 0; dataArrayIndex < _coeficientMatrix.size(); ++dataArrayIndex)
		{
			for (unsigned int varA = 0; varA < _coeficientMatrix[dataArrayIndex].getNumberOfVariable(); ++varA)
			{
				float localCoef=_coeficientMatrix[dataArrayIndex].getVariableAt(varA,var);
				if (localCoef!=0.f)
				{
					complexAddAlphaxCxD<<<(_fftSpaceSize+255)/256, 256,0, _cudaLocalStream >>>(_frenquencySpaceOutputArray[dataArrayIndex],(const cuFloatComplex *)_srcCplx[varA].fft, (const cuFloatComplex *)_frenquencySpaceInput, localCoef, _fftSpaceSize);
					gpuErrchk(cudaPeekAtLastError());
				}
			}
		}
	}
}


void NvidiaGPUAcceleratorDevice::computeRealMissmatchAndRemoveWrongPattern(float* delta0)
{
	const short chunk=32;
	for (size_t dataArrayIndex = 0; dataArrayIndex < _coeficientMatrix.size(); ++dataArrayIndex)
	{
		
		gpuErrchk(cufftExecC2R(_pInv, _frenquencySpaceOutputArray[dataArrayIndex], _realSpaceArray[dataArrayIndex]));

		dataType* realSpace= _realSpaceArray[dataArrayIndex];

		//Remove fobidden/wrong value
		if (!_circularTI)
		{
			for (size_t i = 0; i < _fftSize.size(); ++i)
			{

				unsigned blockSize=1;
				for (size_t j = 0; j < i; ++j)
				{
					blockSize*=_fftSize[j];
				}
				blockSize*=_fftSize[i]-(_srcSize[i]-(_max[i]-_min[i]));

				unsigned delta=1;
				for (size_t j = 0; j <= i; ++j)
				{
					delta*=_fftSize[j];
				}

				dim3 grid((_realSpaceSize/delta+(chunk-1))/chunk,(blockSize+(chunk-1))/chunk,1);
				dim3 block(chunk,chunk,1);
				
				//fprintf(stderr, "blockSize ==> %d, %d ==> %d, %d \n", (_realSpaceSize/delta+(chunk-1))/chunk, (blockSize+(chunk-1))/chunk, delta, blockSize );

				if(blockSize!=0)
					removeBorder<<<grid, block, 0, _cudaLocalStream >>>(realSpace,_realSpaceSize,delta,blockSize,-INFINITY);
				gpuErrchk(cudaPeekAtLastError());
			}
		}

		if(_trueMismatch && !_crossMesurement) // correct value needed
		{
			fma<<<(_realSpaceSize+255)/256, 256, 0, _cudaLocalStream >>>(realSpace, _realSpaceSize, 1./_realSpaceSize, delta0[dataArrayIndex]);
			gpuErrchk(cudaPeekAtLastError());
		}
	}
}


void NvidiaGPUAcceleratorDevice::maskLayerWithVariable(unsigned layer, unsigned variable){

	int deltaCross=0;
	for (int k = int(_min.size())-1; k >=0; k--)
	{
		deltaCross=deltaCross*_fftSize[k]+_min[k];
	}
	int convertedVariable=0;
	int tmp=variable;
	for (unsigned int var = 0; var <_coeficientMatrix[layer].getNumberOfVariable() ; ++var)
	{
		tmp-=_coeficientMatrix[layer].needVariableAlongA(var);
		if(tmp<0)
		{
			convertedVariable=var;
			break;
		}
	}
	//TODO : to remove missing data point
	/*for (unsigned int i = 0; i < _realSpaceSize; ++i){
		_realSpaceArray[layer][i]*=((dataType*)_srcCplx[convertedVariable].space)[(i+deltaCross)%_realSpaceSize];

		//-((1.f-[j])*1.1f)*FLT_MAX);
	}*/
}

void NvidiaGPUAcceleratorDevice::setValueInErrorArray(unsigned position, float value){
	
	float* errosArray=_realSpaceArray.front();
	errosArray[position]=value;
}

void NvidiaGPUAcceleratorDevice::compensateMissingData(){
	
	float* errosArray=_realSpaceArray.front();
	float* crossErrosArray=_realSpaceArray.back();
	unsigned sizeArray=getErrorsArraySize();

	compensateMissingDatakernel<<<(_realSpaceSize+255)/256, 256, 0, _cudaLocalStream >>>(errosArray, crossErrosArray, _realSpaceSize,-INFINITY);
	gpuErrchk(cudaPeekAtLastError());
}

void NvidiaGPUAcceleratorDevice::searchKBigest(float* errors,unsigned *encodedPosition, unsigned extendK, float seed){
	
	float* errosArray=_realSpaceArray.front();
	unsigned sizeArray=getErrorsArraySize();

	if(_encodedPosition_dSize<extendK)
	{
		if(_encodedPosition_d!=nullptr)
		{
			gpuErrchk(cudaFree(_encodedPosition_d));
			gpuErrchk(cudaFree(_mismatch_d));
			_encodedPosition_d=nullptr;
			_encodedPosition_dSize=0;
		}
		gpuErrchk(cudaMalloc(&_encodedPosition_d,extendK*sizeof(unsigned int)));
		gpuErrchk(cudaMalloc(&_mismatch_d,extendK*sizeof(unsigned int)));
		_encodedPosition_dSize=extendK;
	};

	for (int i = 0; i < extendK; ++i)
	{

		gpuErrchk(cublasIsamin(_cublasHandle, sizeArray, errosArray, 1, ((int*)_encodedPosition_d)+i));
		copyAndRemove<<<1,1,0,_cudaLocalStream>>>(errosArray,_encodedPosition_d,_mismatch_d,i,-INFINITY);
		gpuErrchk(cudaPeekAtLastError());
	}

	gpuErrchk(cudaMemcpyAsync(encodedPosition, _encodedPosition_d, extendK*sizeof(unsigned int), cudaMemcpyDeviceToHost, _cudaLocalStream));
	gpuErrchk(cudaMemcpyAsync(errors, _mismatch_d, extendK*sizeof(float), cudaMemcpyDeviceToHost, _cudaLocalStream));
	gpuErrchk(cudaEventRecord(_cudaEventFinal,_cudaLocalStream));
	gpuErrchk(cudaEventSynchronize(_cudaEventFinal));
	// gpuErrchk(cudaDeviceSynchronize());
	// for (int i = 0; i < extendK; ++i)
	// {
	// 	encodedPosition[i]=1;
	// 	errors[i]=0;
	// }
	

}
