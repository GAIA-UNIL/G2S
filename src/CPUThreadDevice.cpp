/*
 * Mathieu Gravey
 * Copyright (C) 2017 Mathieu Gravey
 * 
 * This program is protected software: you can not redistribute, use, and/or modify it
 * without the explicit accord from the author : Mathieu Gravey, gravey.mathieu@gmail.com
 *
 */
#include <cstring>
#include "CPUThreadDevice.hpp"
#include "sharedMemoryManager.hpp"
#include "utils.hpp"
#include "complexMulti.hpp"

#define PARTIAL_FFT

#define FFTW_PLAN_OPTION FFTW_PATIENT

// #if __cilk
// 	#define fillVectorized(name, begin, amount, value) name[begin:amount]=value;
// #else
	#define fillVectorized(name, begin, amount, value) std::fill(name+begin,name+begin+amount,value);
// #endif


CPUThreadDevice::CPUThreadDevice(SharedMemoryManager* sharedMemoryManager, unsigned int threadRatio, bool withCrossMesurement){
	_deviceType=DT_cpuThreads;
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
		FFTW_PRECISION(plan_with_nthreads)(threadRatio);
		_pInv=FFTW_PRECISION(plan_dft_c2r)(reverseFftSize.size(), reverseFftSize.data(),  _frenquencySpaceOutput, _realSpace, FFTW_PLAN_OPTION);
		if(_crossMesurement){
			FFTW_PRECISION(plan_with_nthreads)(threadRatio);
			_pInvCross=FFTW_PRECISION(plan_dft_c2r)(reverseFftSize.size(), reverseFftSize.data(),  _frenquencySpaceCrossOutput, _realCrossSpace, FFTW_PLAN_OPTION);
		}

		FFTW_PRECISION(plan_with_nthreads)(threadRatio);
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

	}
}

CPUThreadDevice::~CPUThreadDevice(){
	_sharedMemoryManager->removeDevice(this);
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

	free(_pPatchL);
	free(_frenquencySpaceInput);
	free(_frenquencySpaceOutput);
	free(_realSpace);
}

std::vector<g2s::spaceFrequenceMemoryAddress> CPUThreadDevice::allocAndInitSharedMemory(std::vector<void* > srcMemoryAdress, std::vector<unsigned> srcSize, std::vector<unsigned> fftSize){
	
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

std::vector<g2s::spaceFrequenceMemoryAddress> CPUThreadDevice::freeSharedMemory(std::vector<g2s::spaceFrequenceMemoryAddress> sharedMemoryAdress){
	for (int i = 0; i < sharedMemoryAdress.size(); ++i)
	{
		free(sharedMemoryAdress[i].space);
		free(sharedMemoryAdress[i].fft);
	}
	sharedMemoryAdress.clear();
	return sharedMemoryAdress;
}

//compute function

dataType* CPUThreadDevice::getErrorsArray(){
	return _realSpace;
}

float CPUThreadDevice::getErrorAtPosition(unsigned index){
	return _realSpace[index];
}

dataType* CPUThreadDevice::getCossErrorArray(){
	return _realCrossSpace;
}
float CPUThreadDevice::getCroossErrorAtPosition(unsigned index){
	if(_realCrossSpace==nullptr) return 0;
	return _realCrossSpace[index];
}

unsigned CPUThreadDevice::getErrorsArraySize(){
	return _realSpaceSize;
}

unsigned CPUThreadDevice::cvtIndexToPosition(unsigned index){
	
	unsigned position=0;
	unsigned divFactor=_realSpaceSize;
	for (int i = _fftSize.size()-1; i>=0; --i)
	{
		divFactor/=_fftSize[i];
		position=position*_srcSize[i] + (_fftSize[i]-(index/(divFactor))%_fftSize[i]-_min[i]-1);
	}

	return position+1; //TODO check the origine of this 1
}

void CPUThreadDevice::setTrueMismatch(bool value){
	_trueMismatch=value;
}

bool  CPUThreadDevice::candidateForPatern(std::vector<std::vector<int> > &neighborArrayVector, std::vector<std::vector<float> >  &neighborValueArrayVector, std::vector<float> &variablesCoeficient, float delta0){
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
		if(_crossMesurement)memset(_frenquencySpaceCrossOutput, 0, _fftSpaceSize * sizeof(FFTW_PRECISION(complex)) );

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
				//fprintf(stderr, "%f\n", neighborValueArrayVector[i][var]);
				lines[neighborArray[i].back()]=true;
			}

			bool patialFFT=false;

			#ifdef PARTIAL_FFT
			patialFFT=true;
			#endif

			if(patialFFT && (_fftSize.size()>1)){
				
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
			g2s::complexAddAlphaxCxD((dataType*)_frenquencySpaceOutput, (dataType*)_srcCplx[var].fft, (dataType*)_frenquencySpaceInput, variablesCoeficient[var],_fftSpaceSize);
			if(_crossMesurement && var==0){
				g2s::complexAddAlphaxCxD((dataType*)_frenquencySpaceCrossOutput, (dataType*)_srcCplx[variablesCoeficient.size()-1].fft, (dataType*)_frenquencySpaceInput, variablesCoeficient[var],_fftSpaceSize);
			}
		}

		FFTW_PRECISION(execute)(_pInv);
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
				fillVectorized(_realSpace,j,blockSize,-INFINITY);
			}
		}

		if(_trueMismatch && !_crossMesurement) // correct value needed
		{
			#pragma omp simd
			for (int i = 0; i < _realSpaceSize; ++i)
			{
				_realSpace[i]=_realSpace[i]/(_realSpaceSize)+delta0;
			}
		}


		// cross Mesuremnt 

		if(_crossMesurement){
			FFTW_PRECISION(execute)(_pInvCross);
			int deltaCross=0;
			for (int k = _min.size()-1; k >=0; k--)
			{
				deltaCross=deltaCross*_fftSize[k]+_min[k];
			}
			//fprintf(stderr, "delta --> %d \n", deltaCross);
			//Remove fobidden/wrong value
			/*for (int i = _fftSize.size()-1; i>=0; --i)
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
			}*/
			
		/*#if __cilk
			_realCrossSpace[0:_realSpaceSize]/=(_realSpaceSize);
		#else
			#pragma omp simd
			for (int i = 0; i < _realSpaceSize; ++i)
			{
				_realCrossSpace[i]/=(_realSpaceSize);
			}

		#endif*/
			unsigned nbVariable=neighborValueArrayVector[0].size();
			#pragma omp simd
			for (int i = 0; i < _realSpaceSize; ++i)
			{
				_realCrossSpace[i]*=((dataType*)_srcCplx[nbVariable-1].space)[(i+deltaCross)%_realSpaceSize];
			}
		}
	}
	return true;
}

