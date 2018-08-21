/*
 * G2S (c) by Mathieu Gravey (gravey.mathieu@gmail.com)
 * 
 * G2S is licensed under a
 * Creative Commons Attribution-NonCommercial 4.0 International License.
 * 
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
 */

#include <iostream>
#include <climits>
#include <vector>
 

#ifndef ComputeDeviceModule_HPP
#define ComputeDeviceModule_HPP

#include "utils.hpp"
class SharedMemoryManager;

enum DeviceType
{
	DT_cpuThreads=0,
	DT_gpuCuda,
	DT_gpuOpenCL,
	/*DT_cpuNode,
	DT_gpu,
	DT_fpga,*/
	DT_none=INT_MAX
};

//#if PRECISION == 1
	//#define FFTW_PRECISION(Name) fftwf_##Name
	#define dataType float
/*#else
	//#define FFTW_PRECISION(Name) fftw_##Name
	#define dataType double
#endif*/

class ComputeDeviceModule
{
public:
	ComputeDeviceModule(){}
	virtual ~ComputeDeviceModule(){}

	virtual bool candidateForPatern(std::vector<std::vector<int> > &neighborArrayVector, std::vector<std::vector<float> >  &neighborValueArrayVector, std::vector<float> &variablesCoeficient, float delta0)=0;
	virtual std::vector<g2s::spaceFrequenceMemoryAddress > allocAndInitSharedMemory(std::vector<void* > srcMemoryAdress, std::vector<unsigned> srcSize, std::vector<unsigned> fftSize)=0;
	virtual std::vector<g2s::spaceFrequenceMemoryAddress > freeSharedMemory(std::vector<g2s::spaceFrequenceMemoryAddress > sharedMemoryAdress)=0;

	virtual dataType* getErrorsArray()=0;
	virtual unsigned getErrorsArraySize()=0;
	virtual dataType* getCossErrorArray()=0;
	virtual float getErrorAtPosition(unsigned)=0;
	virtual float getCroossErrorAtPosition(unsigned)=0;
	virtual unsigned cvtIndexToPosition(unsigned)=0;
	virtual unsigned cvtPositionToIndex(unsigned)=0;

	unsigned _memoryID=UINT_MAX;
	DeviceType _deviceType=DT_none;
	unsigned _deviceID;

	SharedMemoryManager* _sharedMemoryManager=nullptr;
	
};


#endif