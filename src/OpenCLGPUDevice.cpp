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

#define fillVectorized(name, begin, amount, value) std::fill(name+begin,name+begin+amount-1,value);


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

OpenCLGPUDevice::OpenCLGPUDevice(SharedMemoryManager* sharedMemoryManager,std::vector<g2s::OperationMatrix> coeficientMatrix, unsigned int platformIndex, unsigned int deviceIndex, bool withCrossMesurement, bool circularTI){
	_coeficientMatrix=coeficientMatrix;
	_deviceType=DT_gpuOpenCL;
	int chip,core;
	g2s::rdtscp(&chip, &core);
	_crossMesurement=withCrossMesurement;
	_circularTI=circularTI;
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
	for (int i = 0; i < _coeficientMatrix.size(); ++i)
	{
		FFTW_PRECISION(complex)* ptrCplx=(FFTW_PRECISION(complex)*)malloc(_fftSpaceSize * sizeof(FFTW_PRECISION(complex)));
		_frenquencySpaceOutputArray.push_back(ptrCplx);
		float* ptrReal=(dataType_g2s*)malloc(_realSpaceSize* sizeof(dataType_g2s));
		_realSpaceArray.push_back(ptrReal);
	}

	_pPatchL=(FFTW_PRECISION(plan)*)malloc(sizeof(FFTW_PRECISION(plan)) * _fftSize.back());


	std::vector<int> reverseFftSize(_fftSize.begin(),_fftSize.end());
	std::reverse(reverseFftSize.begin(),reverseFftSize.end());
	#pragma omp critical (initPlan)
	{
		_pInv=FFTW_PRECISION(plan_dft_c2r)(reverseFftSize.size(), reverseFftSize.data(),  _frenquencySpaceOutputArray[0], _realSpaceArray[0], FFTW_PLAN_OPTION);
		_p=FFTW_PRECISION(plan_dft_r2c)( reverseFftSize.size(), reverseFftSize.data(), _realSpaceArray[0], _frenquencySpaceInput, FFTW_PLAN_OPTION);

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
				_pPatchL[i]=FFTW_PRECISION(plan_dft_r2c)(reverseFftSize.size()-1, reverseFftSize.data()+1, _realSpaceArray[0]+i*reducedRealSize, _frenquencySpaceInput+i*reducedFftSize, FFTW_PLAN_OPTION);
			}

			
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

		//fprintf(stderr, "we have %d platforms\n", platformCount);
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

		for (int i = 0; i < _coeficientMatrix.size(); ++i)
		{
			frenquencySpaceOutputArray_d.push_back(clCreateBuffer( ctx, CL_MEM_USE_HOST_PTR , _fftSpaceSize * sizeof(FFTW_PRECISION(complex)), _frenquencySpaceOutputArray[i], &err ));
			realSpaceArray_d.push_back(clCreateBuffer( ctx, CL_MEM_USE_HOST_PTR , _realSpaceSize*sizeof(float), _realSpaceArray[i], &err ));
		}


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
	for (int i = 0; i < frenquencySpaceOutputArray_d.size(); ++i)
	{
			clReleaseMemObject(frenquencySpaceOutputArray_d[i]);
	}
	for (int i = 0; i < realSpaceArray_d.size(); ++i)
	{
			clReleaseMemObject(realSpaceArray_d[i]);
	}

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

	free(_frenquencySpaceInput);
	for (int i = 0; i < _frenquencySpaceOutputArray.size(); ++i)
	{
		free(_frenquencySpaceOutputArray[i]);
		_frenquencySpaceOutputArray[i]=nullptr;
	}
	for (int i = 0; i < _realSpaceArray.size(); ++i)
	{
		free(_realSpaceArray[i]);
		_realSpaceArray[i]=nullptr;
	}
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
		sharedMemoryAdress.space=malloc(realSpaceSize * sizeof(dataType_g2s));
		memcpy(sharedMemoryAdress.space,srcMemoryAdress[i], realSpaceSize * sizeof(dataType_g2s));
		sharedMemoryAdress.fft=malloc( fftSpaceSize * sizeof(FFTW_PRECISION(complex)));
	
		sharedMemory.push_back(sharedMemoryAdress);

		FFTW_PRECISION(plan) p;
		#pragma omp critical (initPlan)
		{
			
			p=FFTW_PRECISION(plan_dft_r2c)(reverseFftSize.size(), reverseFftSize.data(), (dataType_g2s*)sharedMemoryAdress.space, (FFTW_PRECISION(complex)*)sharedMemoryAdress.fft, FFTW_ESTIMATE);
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

dataType_g2s* OpenCLGPUDevice::getArray(unsigned arrayIndex){
	return _realSpaceArray[arrayIndex];
}

unsigned OpenCLGPUDevice::getArraySize(){
	return _realSpaceSize;
}

float OpenCLGPUDevice::getValueAtPosition(unsigned arrayIndex, unsigned index){
	return _realSpaceArray[arrayIndex][index];
}


dataType_g2s* OpenCLGPUDevice::getErrorsArray(){
	return _realSpaceArray.front();
}

float OpenCLGPUDevice::getErrorAtPosition(unsigned index){
	return _realSpaceArray.front()[index];
}

dataType_g2s* OpenCLGPUDevice::getCossErrorArray(){
	return _realSpaceArray.back();
}
float OpenCLGPUDevice::getCroossErrorAtPosition(unsigned index){
	return _realSpaceArray.back()[index];
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
		position=position*_srcSize[i] + ((_fftSize[i]-(index/(divFactor))%_fftSize[i]-_min[i]-1)+_srcSize[i])%_srcSize[i];
	}

	return position;
}

void OpenCLGPUDevice::setTrueMismatch(bool value){
	_trueMismatch=value;
}

unsigned OpenCLGPUDevice::cvtPositionToIndex(unsigned position){

	unsigned index=0;
	unsigned divFactor=1;
	for (int i = _fftSize.size()-1; i>=0; --i)
	{
	    divFactor*=_srcSize[i];
	}
	for (int i = _fftSize.size()-1; i>=0; --i)
	{
		divFactor/=_srcSize[i];
		index=index*_fftSize[i] + (_fftSize[i]-(position/(divFactor)+_min[i])%_srcSize[i]-1);
	}
	return index;
}


bool  OpenCLGPUDevice::candidateForPatern(std::vector<std::vector<int> > &neighborArrayVector, std::vector<std::vector<float> >  &neighborValueArrayVector, std::vector<float> &variablesCoeficient, std::vector<float> delta0){
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
		for (int dataArrayIndex = 0; dataArrayIndex < _coeficientMatrix.size(); ++dataArrayIndex)
		{
			memset(_frenquencySpaceOutputArray[dataArrayIndex], 0, _fftSpaceSize * sizeof(FFTW_PRECISION(complex)) );
		}
		
		std::vector<std::vector<int> > neighborArray=neighborArrayVector;

		//update coordonate
		for (int j = 0; j < neighborArray.size(); ++j)
		{
			for (int i = 0; i < _min.size(); ++i)
			{
				neighborArray[j][i]-=_min[i];
			}
		}

		for (int var = 0; var <_coeficientMatrix[0].getNumberOfVariable() ; ++var)
		{
			bool needTobeComputed=false;
			for (int dataArrayIndex = 0; dataArrayIndex < _coeficientMatrix.size(); ++dataArrayIndex)
			{
				needTobeComputed|=_coeficientMatrix[dataArrayIndex].needVariableAlongB(var);
			}
			if(!needTobeComputed) continue;

			memset(_realSpaceArray[0],0,sizeof(dataType_g2s) * _realSpaceSize );
			memset(_frenquencySpaceInput,0,_fftSpaceSize * sizeof(FFTW_PRECISION(complex)) );

			for (int i = 0; i < neighborArray.size(); ++i)
			{
				_realSpaceArray[0][ index(neighborArray[i]) ] =  neighborValueArrayVector[i][var];
				lines[neighborArray[i].back()]=true;
			}

			bool patialFFT=false;

			#ifdef PARTIAL_FFT
			patialFFT=true;
			#endif

			if(patialFFT && (_fftSize.size()>1)){
				
				//#pragma omp parallel for default(none) num_threads(_threadRatio) schedule(dynamic,1) shared(lines)
				for (int i = 0; i < _fftSize.back(); ++i)
				{
					if(lines[i]){
						FFTW_PRECISION(execute)(_pPatchL[i]);
					}
				}
				FFTW_PRECISION(execute)(_pPatchM);
			}else{
				FFTW_PRECISION(execute)(_p);
			}


			for (int dataArrayIndex = 0; dataArrayIndex < _coeficientMatrix.size(); ++dataArrayIndex)
			{
				for (int varA = 0; varA < _coeficientMatrix[dataArrayIndex].getNumberOfVariable(); ++varA)
				{
					float localCoef=_coeficientMatrix[dataArrayIndex].getVariableAt(varA,var);
					if (localCoef!=0.f)
					{
						unsigned shift=0;
						g2s::complexAddAlphaxCxD(((dataType_g2s*)_frenquencySpaceOutputArray[dataArrayIndex])+shift, ((dataType_g2s*)_srcCplx[varA].fft)+shift, ((dataType_g2s*)_frenquencySpaceInput)+shift, localCoef, std::min(_fftSpaceSize,_fftSpaceSize-shift));
					}
				}
			}

		}

		// add //isation
		for (int dataArrayIndex = 0; dataArrayIndex < _coeficientMatrix.size(); ++dataArrayIndex)
		{

			//FFTW_PRECISION(execute)(_pInv);
			cl_int err;
			//Execute the plan.
			err = clfftEnqueueTransform(planHandle, CLFFT_BACKWARD, 1, &queue, 0, NULL, NULL, &frenquencySpaceOutputArray_d[dataArrayIndex], &realSpaceArray_d[dataArrayIndex], tmpBuffer_d);
			if(err != CLFFT_SUCCESS) fprintf(stderr, "error %d\n", 1001);
			//Wait for calculations to be finished.
			err = clFinish(queue);
			if(err != CLFFT_SUCCESS) fprintf(stderr, "error %d\n", 1002);

			dataType_g2s* realSpace= _realSpaceArray[dataArrayIndex];
			//Remove fobidden/wrong value
			if (!_circularTI)
			{
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

					//#pragma omp parallel for default(none) num_threads(_threadRatio) /*proc_bind(close)*/ firstprivate(delta,blockSize,realSpace)
					for (int j = 0; j < _realSpaceSize; j+=delta)
					{
						fillVectorized(realSpace,j,blockSize,-INFINITY);
					}
				}
			}

			if(_trueMismatch && !_crossMesurement) // correct value needed
			{
				//#pragma omp parallel for simd default(none) num_threads(_threadRatio) /*proc_bind(close)*/ firstprivate(delta0,realSpace,dataArrayIndex)
				for (int i = 0; i < _realSpaceSize; ++i)
				{
					realSpace[i]=realSpace[i]/(_realSpaceSize)+delta0[dataArrayIndex];
				}
			}
		}
	}
	return true;
}

void OpenCLGPUDevice::maskCroossError(){
	maskCroossErrorWithVariable(0);
}

void OpenCLGPUDevice::maskCroossErrorWithVariable(unsigned variable){
	maskLayerWithVariable(_realSpaceArray.size()-1,variable);
}

void OpenCLGPUDevice::maskLayerWithVariable(unsigned layer, unsigned variable){
	int deltaCross=0;
	for (int k = int(_min.size())-1; k >=0; k--)
	{
		deltaCross=deltaCross*_fftSize[k]+_min[k];
	}
	int convertedVariable=0;
	int tmp=variable;
	for (unsigned int var = 0; var <_coeficientMatrix[1].getNumberOfVariable() ; ++var)
	{
		tmp-=_coeficientMatrix[1].needVariableAlongA(var);
		if(tmp<0)
		{
			convertedVariable=var;
			break;
		}
	}

	for (unsigned int i = 0; i < _realSpaceSize; ++i){
		_realSpaceArray[layer][i]*=((dataType_g2s*)_srcCplx[convertedVariable].space)[(i+deltaCross)%_realSpaceSize];

		//-((1.f-[j])*1.1f)*FLT_MAX);
	}
}

#endif
