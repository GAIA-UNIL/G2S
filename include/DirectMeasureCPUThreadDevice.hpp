/*
 * Mathieu Gravey
 * Copyright (C) 2017 Mathieu Gravey
 * 
 * This program is protected software: you can not redistribute, use, and/or modify it
 * without the explicit accord from the author : Mathieu Gravey, gravey.mathieu@gmail.com
 *
 */
#ifndef CPU_THREAD_DEVICE_HPP
#define CPU_THREAD_DEVICE_HPP

#include "computeDeviceModule.hpp"
#ifdef WITH_MKL
	#include <fftw3_mkl.h>
#else
	#include <fftw3.h>
#endif

//#if PRECISION == 1
 #define dataType float
/*#else
 #define FFTW_PRECISION(Name) fftw_##Name
 #define dataType double
#endif*/


class DirectMeasureCPUThreadDevice : public ComputeDeviceModule
{
public:
	DirectMeasureCPUThreadDevice(SharedMemoryManager* sharedMemoryManager, unsigned int threadRatio=1, bool withCrossMesurement=false);
	~DirectMeasureCPUThreadDevice();

	bool candidateForPatern(std::vector<std::vector<int> > &neighborArrayVector, std::vector<std::vector<float> >  &neighborValueArrayVector, std::vector<float> &variablesCoeficient, float delta0);
	std::vector<g2s::spaceFrequenceMemoryAddress> allocAndInitSharedMemory(std::vector<void* > srcMemoryAdress, std::vector<unsigned> srcSize, std::vector<unsigned> fftSize);
	std::vector<g2s::spaceFrequenceMemoryAddress> freeSharedMemory(std::vector<g2s::spaceFrequenceMemoryAddress> sharedMemoryAdress);

	dataType* getErrorsArray();
	unsigned getErrorsArraySize();
	dataType* getCossErrorArray();
	float getErrorAtPosition(unsigned);
	float getCroossErrorAtPosition(unsigned);
	unsigned cvtIndexToPosition(unsigned);
	void setTrueMismatch(bool value);

private:

	std::vector<g2s::spaceFrequenceMemoryAddress> _srcCplx;

	dataType* _realSpace=nullptr;
	dataType* _realCrossSpace=nullptr;


	std::vector<int> _min,_max;// to init
	std::vector<unsigned> _fftSize;// to init
	std::vector<unsigned> _srcSize;// to init
	unsigned _realSpaceSize;

	std::vector<int> _encodedDeltaPosition;
	std::vector<std::vector<float> >  _valueForPositions;

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
};

#endif