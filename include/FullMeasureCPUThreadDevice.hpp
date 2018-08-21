/*
 * G2S (c) by Mathieu Gravey (gravey.mathieu@gmail.com)
 * 
 * G2S is licensed under a
 * Creative Commons Attribution-NonCommercial 4.0 International License.
 * 
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
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


class FullMeasureCPUThreadDevice : public ComputeDeviceModule
{
public:
	FullMeasureCPUThreadDevice(SharedMemoryManager* sharedMemoryManager, unsigned int threadRatio=1, bool withCrossMesurement=false);
	~FullMeasureCPUThreadDevice();

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
};

#endif