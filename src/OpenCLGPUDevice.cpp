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

#ifdef WITH_OPENCL
#include <cstring>
#include "OpenCLGPUDevice.hpp"
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

std::vector<unsigned> OpenCLGPUDevice::DeviceWithHostUnifiedMemory(unsigned platform_id){

	std::vector<unsigned> result;

	cl_int err;
	cl_uint deviceCount;
	cl_uint platformCount;

	cl_platform_id selectedPlatform;
	cl_device_id selectedDevice;

	/* Setup OpenCL environment. */
	clGetPlatformIDs(0, NULL, &platformCount);

	//fprintf(stderr, "we have %d platforms\n", platformCount);
	cl_platform_id* platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
	clGetPlatformIDs(platformCount, platforms, NULL);
	//for (int i = 0; i < platformCount; ++i)
	
	int i=platform_id;
	{
		cl_device_id* devices;
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);
		devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, deviceCount, devices, NULL);

		for (int j = 0; j < deviceCount; j++) {
			cl_bool value=false;
			clGetDeviceInfo(devices[j], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(value), &value, NULL);
			if(value){
				result.push_back(j);
			}
		}
		free(devices);
	}
	free(platforms);

	return result;
}


OpenCLGPUDevice::OpenCLGPUDevice(SharedMemoryManager* sharedMemoryManager, unsigned int platformIndex, unsigned int deviceIndex, bool withCrossMesurement){
	_deviceType=DT_gpuOpenCL;
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

	_fftSpaceSize=1;
	_realSpaceSize=1;

	_min=std::vector<int>(_fftSize.size());
	_max=std::vector<int>(_fftSize.size());

	_fftSpaceSize*=_fftSize.front()/2+1;
	_realSpaceSize*=_fftSize.front();

	for (int i = 1; i < _fftSize.size(); ++i)
	{
		_fftSpaceSize*=_fftSize[i];
		_realSpaceSize*=_fftSize[i];
	}

	_frenquencySpaceInput=(FFTW_PRECISION(complex)*)malloc(_fftSpaceSize * sizeof(FFTW_PRECISION(complex)));
	_frenquencySpaceOutput=(FFTW_PRECISION(complex)*)malloc(_fftSpaceSize * sizeof(FFTW_PRECISION(complex)));
	_realSpace=(dataType*)malloc(_realSpaceSize* sizeof(dataType));

	if(_crossMesurement){
		_frenquencySpaceCrossOutput=(FFTW_PRECISION(complex)*)malloc(_fftSpaceSize * sizeof(FFTW_PRECISION(complex)));
		_realCrossSpace=(dataType*)malloc(_realSpaceSize* sizeof(dataType));
	}

	_pPatchL=(FFTW_PRECISION(plan)*)malloc(sizeof(FFTW_PRECISION(plan)) * _fftSize.back());


	std::vector<int> reverseFftSize(_fftSize.begin(),_fftSize.end());
	std::reverse(reverseFftSize.begin(),reverseFftSize.end());
	#pragma omp critical (initPlan)
	{
		_pInv=FFTW_PRECISION(plan_dft_c2r)(reverseFftSize.size(), reverseFftSize.data(),  _frenquencySpaceOutput, _realSpace, FFTW_PLAN_OPTION);
		if(_crossMesurement){
			_pInvCross=FFTW_PRECISION(plan_dft_c2r)(reverseFftSize.size(), reverseFftSize.data(),  _frenquencySpaceCrossOutput, _realCrossSpace, FFTW_PLAN_OPTION);
		}

		_p=FFTW_PRECISION(plan_dft_r2c)( reverseFftSize.size(), reverseFftSize.data(), _realSpace, _frenquencySpaceInput, FFTW_PLAN_OPTION);

		if(_fftSize.size()>1){
			unsigned reducedSize=1;

			unsigned reducedFftSize=reducedSize*(_fftSize.front()/2+1);
			unsigned reducedRealSize=reducedSize*(_fftSize.front());

			for (int i = 1; i < _fftSize.size()-1; ++i)
			{
				reducedSize*=_fftSize[i];
			}

			for (int i = 0; i < _fftSize.back(); ++i)
			{
				_pPatchL[i]=FFTW_PRECISION(plan_dft_r2c)(reverseFftSize.size()-1, reverseFftSize.data()+1, _realSpace+i*reducedRealSize, _frenquencySpaceInput+i*reducedFftSize, FFTW_PLAN_OPTION);
			}

			FFTW_PRECISION(plan_with_nthreads)(threadRatio);
			_pPatchM=FFTW_PRECISION(plan_many_dft)(1, reverseFftSize.data(),reducedFftSize,
				_frenquencySpaceInput, reverseFftSize.data(),
				reducedFftSize, 1,
				_frenquencySpaceInput, reverseFftSize.data(),
				reducedFftSize, 1,
				FFTW_FORWARD, FFTW_PLAN_OPTION);
		}

		//OpenCl part
		cl_int err;
		cl_uint deviceCount;
		cl_uint platformCount;

		cl_platform_id selectedPlatform;
		cl_device_id selectedDevice;

		/* Setup OpenCL environment. */
		clGetPlatformIDs(0, NULL, &platformCount);

		fprintf(stderr, "we have %d platforms\n", platformCount);
		cl_platform_id* platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
		clGetPlatformIDs(platformCount, platforms, NULL);
		
		int i=platformIndex;
		{
			cl_device_id* devices;
			clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);
			devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
			clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, deviceCount, devices, NULL);

			selectedPlatform=platforms[platformIndex];
			selectedDevice=devices[deviceIndex];
			free(devices);
		}
		free(platforms);

		cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };

		props[1] = (cl_context_properties)selectedPlatform;
		ctx = clCreateContext( props, 1, &selectedDevice, NULL, NULL, &err );
		queue = clCreateCommandQueue( ctx, selectedDevice, 0, &err );

		unsigned erroid=0;
		// Setup clFFT.
		clfftSetupData fftSetup;
		err = clfftInitSetupData(&fftSetup);
		if(err != CLFFT_SUCCESS) fprintf(stderr, "error %d\n", erroid++);
		err = clfftSetup(&fftSetup);
		if(err != CLFFT_SUCCESS) fprintf(stderr, "error %d\n", erroid++);

		frenquencySpaceOutput_d = clCreateBuffer( ctx, CL_MEM_USE_HOST_PTR , _fftSpaceSize * sizeof(FFTW_PRECISION(complex)), _frenquencySpaceOutput, &err );
		realSpace_d = clCreateBuffer( ctx, CL_MEM_USE_HOST_PTR , _realSpaceSize*sizeof(float), _realSpace, &err );


		/* FFT library realted declarations */
		clfftDim dim;
		size_t clLengths[_fftSize.size()];
		for (int i = 0; i < _fftSize.size(); ++i)
		{
			clLengths[i]=_fftSize[i];
			if(i==0)dim=CLFFT_1D;
			if(i==1)dim=CLFFT_2D;
			if(i==2)dim=CLFFT_3D;
		}

		err = clfftCreateDefaultPlan(&planHandle, ctx, dim, clLengths);
		if(err != CLFFT_SUCCESS) fprintf(stderr, "error %d\n", erroid++);

		err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE_FAST); //CLFFT_SINGLE , Is not a big improuvement.
		if(err != CLFFT_SUCCESS) fprintf(stderr, "error %d\n", erroid++);
		err = clfftSetLayout(planHandle, CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL);
		if(err != CLFFT_SUCCESS) fprintf(stderr, "error %d\n", erroid++);
		err = clfftSetResultLocation(planHandle, CLFFT_OUTOFPLACE);
		if(err != CLFFT_SUCCESS) fprintf(stderr, "error %d\n", erroid++);

		size_t clInStride[3]={1,101,1};
		size_t clOutStride[3]={1,200,1};

		for (int i = 1; i < _fftSize.size()-1; ++i)
		{
			for (int j = i; j < _fftSize.size(); ++j)
			{
				clInStride[j]=_fftSize[i-1];
				clOutStride[j]=_fftSize[i-1];
			}
		}

		clInStride[_fftSize.size()-1]=_fftSize[_fftSize.size()-2]/2+1;
		clOutStride[_fftSize.size()-1]=_fftSize[_fftSize.size()-2];

		err = clfftSetPlanInStride(planHandle, dim, clInStride);
 		err = clfftSetPlanOutStride(planHandle, dim, clOutStride);


		//Bake the plan.
		err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);
		if(err != CLFFT_SUCCESS) fprintf(stderr, "error %d\n", erroid++);

		int status = 0;
		size_t tmpBufferSize = 0;
 		status = clfftGetTmpBufSize(planHandle, &tmpBufferSize);
 		if ((status == 0) && (tmpBufferSize > 0)) {
  			tmpBuffer_d = clCreateBuffer(ctx, CL_MEM_READ_WRITE, tmpBufferSize, 0, &err);
  		}

	}
}

OpenCLGPUDevice::~OpenCLGPUDevice(){
	_sharedMemoryManager->removeDevice(this);
	cl_int err;
	clReleaseMemObject(frenquencySpaceOutput_d);
	clReleaseMemObject(realSpace_d);
	clReleaseMemObject(tmpBuffer_d);

	//Release the plan.
	err = clfftDestroyPlan( &planHandle );

	//Release clFFT library.
	clfftTeardown( );

	//Release OpenCL working objects.
	clReleaseCommandQueue( queue );
	clReleaseContext( ctx );

	FFTW_PRECISION(destroy_plan)(_pInv);
	FFTW_PRECISION(destroy_plan)(_p);
	
	if(_fftSize.size()>1){
		FFTW_PRECISION(destroy_plan)(_pPatchM);
		for (int i = 0; i < _fftSize.back(); ++i)
		{
			FFTW_PRECISION(destroy_plan)(_pPatchL[i]);
		}
	}
	if(_crossMesurement){
		FFTW_PRECISION(destroy_plan)(_pInvCross);
		free(_frenquencySpaceCrossOutput);
		free(_realCrossSpace);
	}
	free(_frenquencySpaceInput);
	free(_frenquencySpaceOutput);
	free(_realSpace);
}

std::vector<g2s::spaceFrequenceMemoryAddress> OpenCLGPUDevice::allocAndInitSharedMemory(std::vector<void* > srcMemoryAdress, std::vector<unsigned> srcSize, std::vector<unsigned> fftSize){
	
	//fprintf(stderr, "alloc shared memory CPU\n");

	unsigned fftSpaceSize=1;
	unsigned realSpaceSize=1;

	fftSpaceSize*=fftSize.front()/2+1;
	realSpaceSize*=fftSize.front();

	for (int i = 1; i < fftSize.size(); ++i)
	{
		fftSpaceSize*=fftSize[i];
		realSpaceSize*=fftSize[i];
	}

	std::vector<int> reverseFftSize(fftSize.begin(),fftSize.end());
	std::reverse(reverseFftSize.begin(),reverseFftSize.end());

	std::vector<g2s::spaceFrequenceMemoryAddress> sharedMemory;
	for (int i = 0; i < srcMemoryAdress.size(); ++i)
	{
		g2s::spaceFrequenceMemoryAddress sharedMemoryAdress;
		sharedMemoryAdress.space=malloc(realSpaceSize * sizeof(dataType));
		memcpy(sharedMemoryAdress.space,srcMemoryAdress[i], realSpaceSize * sizeof(dataType));
		sharedMemoryAdress.fft=malloc( fftSpaceSize * sizeof(FFTW_PRECISION(complex)));
	
		sharedMemory.push_back(sharedMemoryAdress);

		FFTW_PRECISION(plan) p;
		#pragma omp critical (initPlan)
		{
			
			p=FFTW_PRECISION(plan_dft_r2c)(reverseFftSize.size(), reverseFftSize.data(), (dataType*)sharedMemoryAdress.space, (FFTW_PRECISION(complex)*)sharedMemoryAdress.fft, FFTW_ESTIMATE);
		}
		FFTW_PRECISION(execute)(p);
		FFTW_PRECISION(destroy_plan)(p);
	}
	return sharedMemory;

}

std::vector<g2s::spaceFrequenceMemoryAddress> OpenCLGPUDevice::freeSharedMemory(std::vector<g2s::spaceFrequenceMemoryAddress> sharedMemoryAdress){
	for (int i = 0; i < sharedMemoryAdress.size(); ++i)
	{
		free(sharedMemoryAdress[i].space);
		free(sharedMemoryAdress[i].fft);
	}
	sharedMemoryAdress.clear();
	return sharedMemoryAdress;
}

//compute function

dataType* OpenCLGPUDevice::getErrorsArray(){
	return _realSpace;
}

float OpenCLGPUDevice::getErrorAtPosition(unsigned index){
	return _realSpace[index];
}

dataType* OpenCLGPUDevice::getCossErrorArray(){
	return _realCrossSpace;
}
float OpenCLGPUDevice::getCroossErrorAtPosition(unsigned index){
	if(_realCrossSpace==nullptr) return std::nanf("0");
	return _realCrossSpace[index];
}

unsigned OpenCLGPUDevice::getErrorsArraySize(){
	return _realSpaceSize;
}

unsigned OpenCLGPUDevice::cvtIndexToPosition(unsigned index){
	
	unsigned position=0;
	unsigned divFactor=_realSpaceSize;
	for (int i = _fftSize.size()-1; i>=0; --i)
	{
		divFactor/=_fftSize[i];
		position=position*_srcSize[i] + (_fftSize[i]-(index/(divFactor))%_fftSize[i]-_min[i]-1);
	}

	return position;
}

void OpenCLGPUDevice::setTrueMismatch(bool value){
	_trueMismatch=value;
}

bool  OpenCLGPUDevice::candidateForPatern(std::vector<std::vector<int> > &neighborArrayVector, std::vector<std::vector<float> >  &neighborValueArrayVector, std::vector<float> &variablesCoeficient, float delta0){
	if(neighborValueArrayVector.size()==0)return false;

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

	{
		bool lines[_fftSize.back()];
		memset(_frenquencySpaceOutput, 0, _fftSpaceSize * sizeof(FFTW_PRECISION(complex)) );
		std::vector<std::vector<int> > neighborArray=neighborArrayVector;

		//update coordonate
		for (int j = 0; j < neighborArray.size(); ++j)
		{
			for (int i = 0; i < _min.size(); ++i)
			{
				neighborArray[j][i]-=_min[i];
			}
		}

		for (int var = 0; var <neighborValueArrayVector[0].size() ; ++var)
		{
			
			memset(_realSpace,0,sizeof(dataType) * _realSpaceSize );
			memset(_frenquencySpaceInput,0,_fftSpaceSize * sizeof(FFTW_PRECISION(complex)) );

			for (int i = 0; i < neighborArray.size(); ++i)
			{
				_realSpace[ index(neighborArray[i]) ] =  neighborValueArrayVector[i][var];
				lines[neighborArray[i].back()]=true;
			}
		#ifdef PARTIAL_FFT
			for (int i = 0; i < _fftSize.back(); ++i)
			{
				if(lines[i]){
					FFTW_PRECISION(execute)(_pPatchL[i]);
				}
			}
			FFTW_PRECISION(execute)(_pPatchM);
		#else
			FFTW_PRECISION(execute)(_p);
		#endif
			g2s::complexAddAlphaxCxD((dataType*)_frenquencySpaceOutput, (dataType*)_srcCplx[var].fft, (dataType*)_frenquencySpaceInput, variablesCoeficient[var],_fftSpaceSize);
			if(_crossMesurement && var==0){
				g2s::complexAddAlphaxCxD((dataType*)_frenquencySpaceCrossOutput, (dataType*)_srcCplx[variablesCoeficient.size()-1].fft, (dataType*)_frenquencySpaceInput, variablesCoeficient[var],_fftSpaceSize);
			}
		}

		//FFTW_PRECISION(execute)(_pInv);
		cl_int err;
		//Execute the plan.
		err = clfftEnqueueTransform(planHandle, CLFFT_BACKWARD, 1, &queue, 0, NULL, NULL, &frenquencySpaceOutput_d, &realSpace_d, tmpBuffer_d);
		if(err != CLFFT_SUCCESS) fprintf(stderr, "error %d\n", 1001);
		//Wait for calculations to be finished.
		err = clFinish(queue);
		if(err != CLFFT_SUCCESS) fprintf(stderr, "error %d\n", 1002);


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

		if(_trueMismatch && !_crossMesurement) // correct value needed
		{
		#if __cilk
			_realSpace[0:_realSpaceSize]=_realSpace[0:_realSpaceSize]/(_realSpaceSize)+delta0;
		#else
			#pragma omp simd
			for (int i = 0; i < _realSpaceSize; ++i)
			{
				_realSpace[i]=_realSpace[i]+delta0;
				//fprintf(stderr, "%f\n",_realSpace[i]);
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

			#pragma omp simd
			for (int i = 0; i < _realSpaceSize; ++i)
			{
				_realCrossSpace[i]+=(std::isnan(((dataType*)_srcCplx[var].src)[i])+(_realSpaceSize[i]==0))*INFINITY;
			}
		}
	}
	return true;
}

#endif
