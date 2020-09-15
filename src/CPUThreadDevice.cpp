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
#include "CPUThreadDevice.hpp"
#include "sharedMemoryManager.hpp"
#include "utils.hpp"
#include "complexMulti.hpp"
#ifdef HBW_MALLOC
	#include <hbwmalloc.h>
	#define mem_malloc hbw_malloc
	#define mem_free hbw_free
#else
	#define mem_malloc malloc
	#define mem_free free
#endif

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


CPUThreadDevice::CPUThreadDevice(SharedMemoryManager* sharedMemoryManager,std::vector<g2s::OperationMatrix> coeficientMatrix, unsigned int threadRatio, bool withCrossMesurement, bool circularTI){
	_coeficientMatrix=coeficientMatrix;

	_deviceType=DT_cpuThreads;
	_threadRatio=threadRatio;
	int chip,core;
	g2s::rdtscp(&chip, &core);
	_crossMesurement=withCrossMesurement;
	_circularTI=circularTI;
	//printf("core %d, chip %d\n",core, chip );
	_deviceID=chip;
	_sharedMemoryManager=sharedMemoryManager;
	#pragma omp critical (createDevices)
	{
		_sharedMemoryManager->addDevice(this);
	}

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

	for (size_t i = 1; i < _fftSize.size(); ++i)
	{
		_fftSpaceSize*=_fftSize[i];
		_realSpaceSize*=_fftSize[i];
	}

	_frenquencySpaceInput=(FFTW_PRECISION(complex)*)mem_malloc(_fftSpaceSize * sizeof(FFTW_PRECISION(complex)));
	
	for (size_t i = 0; i < _coeficientMatrix.size(); ++i)
	{
		FFTW_PRECISION(complex)* ptrCplx=(FFTW_PRECISION(complex)*)mem_malloc(_fftSpaceSize * sizeof(FFTW_PRECISION(complex)));
		_frenquencySpaceOutputArray.push_back(ptrCplx);
		float* ptrReal=(dataType_g2s*)mem_malloc(_realSpaceSize* sizeof(dataType_g2s));
		_realSpaceArray.push_back(ptrReal);
	}

	_pPatchL=(FFTW_PRECISION(plan)*)mem_malloc(sizeof(FFTW_PRECISION(plan)) * _fftSize.back());


	std::vector<int> reverseFftSize(_fftSize.begin(),_fftSize.end());
	std::reverse(reverseFftSize.begin(),reverseFftSize.end());
	#pragma omp critical (initPlan)
	{
		FFTW_PRECISION(plan_with_nthreads)(threadRatio);
		_pInv=FFTW_PRECISION(plan_dft_c2r)(reverseFftSize.size(), reverseFftSize.data(),  _frenquencySpaceOutputArray[0], _realSpaceArray[0], FFTW_PLAN_OPTION);

		FFTW_PRECISION(plan_with_nthreads)(threadRatio);
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

			FFTW_PRECISION(plan_with_nthreads)(threadRatio);
			_pPatchM=FFTW_PRECISION(plan_many_dft)(1, reverseFftSize.data(),reducedFftSize,
				_frenquencySpaceInput, reverseFftSize.data(),
				reducedFftSize, 1,
				_frenquencySpaceInput, reverseFftSize.data(),
				reducedFftSize, 1,
				FFTW_FORWARD, FFTW_PLAN_OPTION);
		}

	}
}

CPUThreadDevice::~CPUThreadDevice(){
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

	mem_free(_pPatchL);
	mem_free(_frenquencySpaceInput);
	for (size_t i = 0; i < _frenquencySpaceOutputArray.size(); ++i)
	{
		mem_free(_frenquencySpaceOutputArray[i]);
		_frenquencySpaceOutputArray[i]=nullptr;
	}
	for (size_t i = 0; i < _realSpaceArray.size(); ++i)
	{
		mem_free(_realSpaceArray[i]);
		_realSpaceArray[i]=nullptr;
	}
}

std::vector<g2s::spaceFrequenceMemoryAddress> CPUThreadDevice::allocAndInitSharedMemory(std::vector<void* > srcMemoryAdress, std::vector<unsigned> srcSize, std::vector<unsigned> fftSize){
	
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
		sharedMemoryAdress.space=mem_malloc(realSpaceSize * sizeof(dataType_g2s));
		memcpy(sharedMemoryAdress.space,srcMemoryAdress[i], realSpaceSize * sizeof(dataType_g2s));
		sharedMemoryAdress.fft=mem_malloc( fftSpaceSize * sizeof(FFTW_PRECISION(complex)));
		
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

std::vector<g2s::spaceFrequenceMemoryAddress> CPUThreadDevice::freeSharedMemory(std::vector<g2s::spaceFrequenceMemoryAddress> sharedMemoryAdress){
	for (size_t i = 0; i < sharedMemoryAdress.size(); ++i)
	{
		mem_free(sharedMemoryAdress[i].space);
		mem_free(sharedMemoryAdress[i].fft);
	}
	sharedMemoryAdress.clear();
	return sharedMemoryAdress;
}

//compute function

dataType_g2s* CPUThreadDevice::getArray(unsigned arrayIndex){
	return _realSpaceArray[arrayIndex];
}

unsigned CPUThreadDevice::getArraySize(){
	return _realSpaceSize;
}

float CPUThreadDevice::getValueAtPosition(unsigned arrayIndex, unsigned index){
	return _realSpaceArray[arrayIndex][index];
}


dataType_g2s* CPUThreadDevice::getErrorsArray(){
	return _realSpaceArray.front();
}

float CPUThreadDevice::getErrorAtPosition(unsigned index){
	return _realSpaceArray.front()[index];
}

dataType_g2s* CPUThreadDevice::getCossErrorArray(){
	return _realSpaceArray.back();
}
float CPUThreadDevice::getCroossErrorAtPosition(unsigned index){
	return _realSpaceArray.back()[index];
}

unsigned CPUThreadDevice::getErrorsArraySize(){
	return _realSpaceSize;
}

unsigned CPUThreadDevice::cvtIndexToPosition(unsigned index){
	
	unsigned position=0;
	unsigned divFactor=_realSpaceSize;
	for (int i = int(_fftSize.size()-1); i>=0; --i)
	{
		divFactor/=_fftSize[i];
		position=position*_srcSize[i] + ((_fftSize[i]-(index/(divFactor))%_fftSize[i]-_min[i]-1)+_srcSize[i])%_srcSize[i];
	}

	return position;
}

unsigned CPUThreadDevice::cvtPositionToIndex(unsigned position){

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


void CPUThreadDevice::setTrueMismatch(bool value){
	_trueMismatch=value;
}

bool  CPUThreadDevice::candidateForPatern(std::vector<std::vector<int> > &neighborArrayVector, std::vector<std::vector<float> >  &neighborValueArrayVector, std::vector<float> &variablesCoeficient, std::vector<float> delta0){
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
		bool lines[_fftSize.back()];
		for (size_t dataArrayIndex = 0; dataArrayIndex < _coeficientMatrix.size(); ++dataArrayIndex)
		{
			memset(_frenquencySpaceOutputArray[dataArrayIndex], 0, _fftSpaceSize * sizeof(FFTW_PRECISION(complex)) );
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

		for (unsigned int var = 0; var <_coeficientMatrix[0].getNumberOfVariable() ; ++var)
		{
			bool needTobeComputed=false;
			for (size_t dataArrayIndex = 0; dataArrayIndex < _coeficientMatrix.size(); ++dataArrayIndex)
			{
				needTobeComputed|=_coeficientMatrix[dataArrayIndex].needVariableAlongB(var);
			}
			if(!needTobeComputed) continue;

			memset(_realSpaceArray[0],0,sizeof(dataType_g2s) * _realSpaceSize );
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
						//#pragma omp parallel default(none) num_threads(_threadRatio) /*proc_bind(close)*/ firstprivate(variablesCoeficient,var)
						{
							unsigned k=0;
							#if _OPENMP
							//k=omp_get_thread_num();
							#endif
							unsigned shift=k*unsigned(ceil(_fftSpaceSize/float(_threadRatio)));
							g2s::complexAddAlphaxCxD(((dataType_g2s*)_frenquencySpaceOutputArray[dataArrayIndex])+shift, ((dataType_g2s*)_srcCplx[varA].fft)+shift, ((dataType_g2s*)_frenquencySpaceInput)+shift, localCoef, std::min(_fftSpaceSize,_fftSpaceSize-shift));
						}
					}
				}
			}

		}

		// add //isation
		for (size_t dataArrayIndex = 0; dataArrayIndex < _coeficientMatrix.size(); ++dataArrayIndex)
		{
			FFTW_PRECISION(execute_dft_c2r)(_pInv, _frenquencySpaceOutputArray[dataArrayIndex], _realSpaceArray[dataArrayIndex]);
			dataType_g2s* realSpace= _realSpaceArray[dataArrayIndex];
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

					#pragma omp parallel for default(none) num_threads(_threadRatio) /*proc_bind(close)*/ firstprivate(delta,blockSize,realSpace)
					for (unsigned int j = 0; j < _realSpaceSize; j+=delta)
					{
						fillVectorized(realSpace,j,blockSize,-INFINITY);
					}
				}
			}

			if(_trueMismatch && !_crossMesurement) // correct value needed
			{
				#pragma omp parallel for simd default(none) num_threads(_threadRatio) /*proc_bind(close)*/ firstprivate(delta0,realSpace,dataArrayIndex)
				for (unsigned int i = 0; i < _realSpaceSize; ++i)
				{
					realSpace[i]=realSpace[i]/(_realSpaceSize)+delta0[dataArrayIndex];
				}
			}
		}
	}
	return true;
}


void CPUThreadDevice::maskCroossError(){
	maskCroossErrorWithVariable(0);
}

void CPUThreadDevice::maskCroossErrorWithVariable(unsigned variable){
	maskLayerWithVariable(_realSpaceArray.size()-1,variable);
}

void CPUThreadDevice::maskLayerWithVariable(unsigned layer, unsigned variable){
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

