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
#include "CPUThreadAcceleratorDevice.hpp"
#include "sharedMemoryManager.hpp"
#include "utils.hpp"
#include "complexMulti.hpp"
#include <random>
#include "fKst.hpp"

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


CPUThreadAcceleratorDevice::CPUThreadAcceleratorDevice(SharedMemoryManager* sharedMemoryManager,std::vector<g2s::OperationMatrix> coeficientMatrix, unsigned int threadRatio, bool withCrossMesurement, bool circularTI)
	:AcceleratorDevice( sharedMemoryManager, coeficientMatrix, threadRatio, withCrossMesurement, circularTI){

	_deviceType=DT_cpuThreads;
	int chip,core;
	g2s::rdtscp(&chip, &core);	
	_deviceID=chip;
	
	#pragma omp critical (createDevices)
	{
		_sharedMemoryManager->addDevice(this);
	}

	initDim();

	_frenquencySpaceInput=(FFTW_PRECISION(complex)*)malloc(_fftSpaceSize * sizeof(FFTW_PRECISION(complex)));

	for (size_t i = 0; i < _coeficientMatrix.size(); ++i)
	{
		FFTW_PRECISION(complex)* ptrCplx=(FFTW_PRECISION(complex)*)malloc(_fftSpaceSize * sizeof(FFTW_PRECISION(complex)));
		_frenquencySpaceOutputArray.push_back(ptrCplx);
		float* ptrReal=(dataType*)malloc(_realSpaceSize* sizeof(dataType));
		_realSpaceArray.push_back(ptrReal);
	}

	_pPatchL=(FFTW_PRECISION(plan)*)malloc(sizeof(FFTW_PRECISION(plan)) * _fftSize.back());


	std::vector<int> reverseFftSize(_fftSize.begin(),_fftSize.end());
	std::reverse(reverseFftSize.begin(),reverseFftSize.end());
	#pragma omp critical (initPlan)
	{
		FFTW_PRECISION(plan_with_nthreads)(_threadRatio);
		_pInv=FFTW_PRECISION(plan_dft_c2r)(reverseFftSize.size(), reverseFftSize.data(),  _frenquencySpaceOutputArray[0], _realSpaceArray[0], FFTW_PLAN_OPTION);

		FFTW_PRECISION(plan_with_nthreads)(_threadRatio);
		_p=FFTW_PRECISION(plan_dft_r2c)( reverseFftSize.size(), reverseFftSize.data(), _realSpaceArray[0], _frenquencySpaceInput, FFTW_PLAN_OPTION);

		if(_fftSize.size()>1){
			unsigned reducedSize=1;

			for (size_t i = 1; i < _fftSize.size()-1; ++i)
			{
				reducedSize*=_fftSize[i];
			}

			unsigned reducedFftSize=reducedSize*(_fftSize.front()/2+1);
			unsigned reducedRealSize=reducedSize*(_fftSize.front());

			for (unsigned int i = 0; i < _fftSize.back(); ++i)
			{
				FFTW_PRECISION(plan_with_nthreads)(1);
				_pPatchL[i]=FFTW_PRECISION(plan_dft_r2c)(reverseFftSize.size()-1, reverseFftSize.data()+1, _realSpaceArray[0]+i*reducedRealSize, _frenquencySpaceInput+i*reducedFftSize, FFTW_PLAN_OPTION);
			}

			FFTW_PRECISION(plan_with_nthreads)(_threadRatio);
			_pPatchM=FFTW_PRECISION(plan_many_dft)(1, reverseFftSize.data(),reducedFftSize,
				_frenquencySpaceInput, reverseFftSize.data(),
				reducedFftSize, 1,
				_frenquencySpaceInput, reverseFftSize.data(),
				reducedFftSize, 1,
				FFTW_FORWARD, FFTW_PLAN_OPTION);
		}

	}
}

CPUThreadAcceleratorDevice::~CPUThreadAcceleratorDevice(){
	_sharedMemoryManager->removeDevice(this);
	FFTW_PRECISION(destroy_plan)(_pInv);
	FFTW_PRECISION(destroy_plan)(_p);

	if(_fftSize.size()>1){
		FFTW_PRECISION(destroy_plan)(_pPatchM);
		for (unsigned int i = 0; i < _fftSize.back(); ++i)
		{
			FFTW_PRECISION(destroy_plan)(_pPatchL[i]);
		}
	}

	free(_pPatchL);
	free(_frenquencySpaceInput);
	for (size_t i = 0; i < _frenquencySpaceOutputArray.size(); ++i)
	{
		free(_frenquencySpaceOutputArray[i]);
		_frenquencySpaceOutputArray[i]=nullptr;
	}
	for (size_t i = 0; i < _realSpaceArray.size(); ++i)
	{
		free(_realSpaceArray[i]);
		_realSpaceArray[i]=nullptr;
	}
}

std::vector<g2s::spaceFrequenceMemoryAddress> CPUThreadAcceleratorDevice::allocAndInitSharedMemory(std::vector<void* > srcMemoryAdress, std::vector<unsigned> srcSize, std::vector<unsigned> fftSize){
	
	//fprintf(stderr, "alloc shared memory CPU\n");

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

std::vector<g2s::spaceFrequenceMemoryAddress> CPUThreadAcceleratorDevice::freeSharedMemory(std::vector<g2s::spaceFrequenceMemoryAddress> sharedMemoryAdress){
	for (size_t i = 0; i < sharedMemoryAdress.size(); ++i)
	{
		free(sharedMemoryAdress[i].space);
		free(sharedMemoryAdress[i].fft);
	}
	sharedMemoryAdress.clear();
	return sharedMemoryAdress;
}

//compute function

unsigned CPUThreadAcceleratorDevice::getArraySize(){
	return _realSpaceSize;
}

float CPUThreadAcceleratorDevice::getValueAtPosition(unsigned arrayIndex, unsigned index){
	return _realSpaceArray[arrayIndex][index];
}

float CPUThreadAcceleratorDevice::getErrorAtPosition(unsigned index){
	return _realSpaceArray.front()[index];
}

float CPUThreadAcceleratorDevice::getCroossErrorAtPosition(unsigned index){
	return _realSpaceArray.back()[index];
}

unsigned CPUThreadAcceleratorDevice::getErrorsArraySize(){
	return _realSpaceSize;
}

unsigned CPUThreadAcceleratorDevice::cvtIndexToPosition(unsigned index){
	
	unsigned position=0;
	unsigned divFactor=_realSpaceSize;
	for (int i = int(_fftSize.size()-1); i>=0; --i)
	{
		divFactor/=_fftSize[i];
		position=position*_srcSize[i] + ((_fftSize[i]-(index/(divFactor))%_fftSize[i]-_min[i]-1)+_srcSize[i])%_srcSize[i];
	}

	return position;
}

unsigned CPUThreadAcceleratorDevice::cvtPositionToIndex(unsigned position){

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

void CPUThreadAcceleratorDevice::zerosFrenquencySpaceOutputArray(unsigned layer){
	memset(_frenquencySpaceOutputArray[layer], 0, _fftSpaceSize * sizeof(FFTW_PRECISION(complex)) );
}


void CPUThreadAcceleratorDevice::computeFreqMismatchMap(std::vector<std::vector<int> > neighborArray, std::vector<std::vector<float> >  &neighborValueArrayVector){
	for (unsigned int var = 0; var <_coeficientMatrix[0].getNumberOfVariable() ; ++var)
	{
		bool lines[_fftSize.back()];

		bool needTobeComputed=false;
		for (size_t dataArrayIndex = 0; dataArrayIndex < _coeficientMatrix.size(); ++dataArrayIndex)
		{
			needTobeComputed|=_coeficientMatrix[dataArrayIndex].needVariableAlongB(var);
		}
		if(!needTobeComputed) return;

		memset(_realSpaceArray[0],0,sizeof(dataType) * _realSpaceSize );
		memset(_frenquencySpaceInput,0,_fftSpaceSize * sizeof(FFTW_PRECISION(complex)) );

		for (size_t i = 0; i < neighborArray.size(); ++i)
		{
			_realSpaceArray[0][ index(neighborArray[i]) ] =  neighborValueArrayVector[i][var];
			lines[neighborArray[i].back()]=true;
		}

		bool patialFFT=false;

		#ifdef PARTIAL_FFT
		patialFFT=true;
		#endif

		if(patialFFT && (_fftSize.size()>1)){
			
			#pragma omp parallel for default(none) num_threads(_threadRatio) schedule(dynamic,1) firstprivate(lines)
			for (unsigned int i = 0; i < _fftSize.back(); ++i)
			{
				if(lines[i]){
					FFTW_PRECISION(execute)(_pPatchL[i]);
				}
			}
			FFTW_PRECISION(execute)(_pPatchM);
		}else{
			FFTW_PRECISION(execute)(_p);
		}


		for (size_t dataArrayIndex = 0; dataArrayIndex < _coeficientMatrix.size(); ++dataArrayIndex)
		{
			for (unsigned int varA = 0; varA < _coeficientMatrix[dataArrayIndex].getNumberOfVariable(); ++varA)
			{
				float localCoef=_coeficientMatrix[dataArrayIndex].getVariableAt(varA,var);
				if (localCoef!=0.f)
				{
					//#pragma omp parallel default(none) num_threads(_threadRatio) firstprivate(variablesCoeficient,var)
					{
						unsigned k=0;
						#if _OPENMP
						//k=omp_get_thread_num();
						#endif
						unsigned shift=k*unsigned(ceil(_fftSpaceSize/float(_threadRatio)));
						g2s::complexAddAlphaxCxD(((dataType*)_frenquencySpaceOutputArray[dataArrayIndex])+shift, ((dataType*)_srcCplx[varA].fft)+shift, ((dataType*)_frenquencySpaceInput)+shift, localCoef, std::min(_fftSpaceSize,_fftSpaceSize-shift));
					}
				}
			}
		}
	}
}


void CPUThreadAcceleratorDevice::computeRealMissmatchAndRemoveWrongPattern(float* delta0)
{
	for (size_t dataArrayIndex = 0; dataArrayIndex < _coeficientMatrix.size(); ++dataArrayIndex)
	{
		FFTW_PRECISION(execute_dft_c2r)(_pInv, _frenquencySpaceOutputArray[dataArrayIndex], _realSpaceArray[dataArrayIndex]);
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

				#pragma omp parallel for default(none) num_threads(_threadRatio) firstprivate(delta,blockSize,realSpace)
				for (unsigned int j = 0; j < _realSpaceSize; j+=delta)
				{
					fillVectorized(realSpace,j,blockSize,-INFINITY);
				}
			}
		}

		if(_trueMismatch && !_crossMesurement) // correct value needed
		{
			#pragma omp parallel for simd default(none) num_threads(_threadRatio)  firstprivate(delta0,realSpace,dataArrayIndex)
			for (unsigned int i = 0; i < _realSpaceSize; ++i)
			{
				realSpace[i]=realSpace[i]/(_realSpaceSize)+delta0[dataArrayIndex];
			}
		}
	}
}


void CPUThreadAcceleratorDevice::maskLayerWithVariable(unsigned layer, unsigned variable){
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

	for (unsigned int i = 0; i < _realSpaceSize; ++i){
		_realSpaceArray[layer][i]*=((dataType*)_srcCplx[convertedVariable].space)[(i+deltaCross)%_realSpaceSize];

		//-((1.f-[j])*1.1f)*FLT_MAX);
	}
}



void CPUThreadAcceleratorDevice::setValueInErrorArray(unsigned position, float value){
	float* errosArray=getErrorsArray();
	errosArray[position]=value;
}

void CPUThreadAcceleratorDevice::compensateMissingData(){
	float* errosArray=_realSpaceArray.front();
	float* crossErrosArray=_realSpaceArray.back();
	unsigned sizeArray=getErrorsArraySize();

	#pragma omp simd
	for (unsigned int j = 0; j < getErrorsArraySize(); ++j)
	{
		errosArray[j]=-std::fabs(errosArray[j]/(crossErrosArray[j]*crossErrosArray[j]*crossErrosArray[j]*crossErrosArray[j]));
		if(crossErrosArray[j]==0.0f) errosArray[j]=-INFINITY;
	}
}

void CPUThreadAcceleratorDevice::searchKBigest(float* errors,unsigned *encodedPosition, unsigned extendK, float seed){
	float localError[extendK*_threadRatio];
	float* localErrorPtr=localError;
	unsigned localEncodedPosition[extendK*_threadRatio];
	unsigned* localEncodedPositionPtr=localEncodedPosition;

	float* errosArray=_realSpaceArray.front();
	unsigned sizeArray=getErrorsArraySize();

	#pragma omp parallel default(none) num_threads(_threadRatio) /*proc_bind(close)*/ firstprivate(seed, sizeArray, errosArray, extendK, localErrorPtr, localEncodedPositionPtr)
	{
		unsigned k=0;
		#if _OPENMP
		k=omp_get_thread_num();
		#endif
		std::mt19937 generator;// can be inprouved by only resetting the seed each time
		generator.seed(floor(UINT_MAX*seed)+k);
		std::uniform_real_distribution<float> distribution(0.0,1.0);

		auto rng = std::bind(distribution, std::ref(generator));
		unsigned chunkSize=unsigned(ceil(sizeArray/float(_threadRatio)));
		fKst::findKBigest(errosArray+k*chunkSize,chunkSize,extendK, localErrorPtr+k*extendK, localEncodedPositionPtr+k*extendK, rng);
		for (int j = 0; j < extendK; ++j)
		{
			localEncodedPositionPtr[k*extendK+j]+=k*chunkSize;
		}
	}

	for (int j = 0; j <extendK ; ++j)
	{
		unsigned bestIndex=0;
		for (unsigned int l = 1; l < _threadRatio*extendK; ++l)
		{
			if(localError[l] > localError[bestIndex]) bestIndex=l;
		}

		errors[j]=localError[bestIndex];
		encodedPosition[j]=localEncodedPosition[bestIndex];
		localError[bestIndex]=-INFINITY;
	}

}
