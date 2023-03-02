/*
 * G2S
 * Copyright (C) 2018, Mathieu Gravey (gravey.mathieu@gmail.com) and UNIL (University of Lausanne)
 * 
 * This program is free software: you can retistribute it and/or motify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License.
 * 
 * This program is tistributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>



#include "utils.hpp"
#include "DataImage.hpp"
#include "jobManager.hpp"

#include "sharedMemoryManager.hpp"
#include "CPUThreadDevice.hpp"
#ifdef WITH_OPENCL
#include "OpenCLGPUDevice.hpp"
#endif

#include "simulation.hpp"
#include "quantileSamplingModule.hpp"

void printHelp(){
	printf ("that is the help");
}

int main(int argc, char const *argv[]) {

	bool allRight=true;
	std::multimap<std::string, std::string> arg=g2s::argumentReader(argc,argv);

	char logFileName[2048]={0};
	jobIdType uniqueID=-1;
	bool run=true;


	// manage report file
	FILE *reportFile=stderr;
	if (arg.count("-r") > 1)
	{
		fprintf(reportFile,"only one rapport file is possible\n");
		run=false;
	}else{
		if(arg.count("-r") ==1){
			if(!strcmp((arg.find("-r")->second).c_str(),"stderr")){
				reportFile=stderr;
			}
			if(!strcmp((arg.find("-r")->second).c_str(),"stdout")){
				reportFile=stdout;
			}
			if (reportFile==NULL) {
				strcpy(logFileName,(arg.find("-r")->second).c_str());
				reportFile=fopen((arg.find("-r")->second).c_str(),"a");
				setvbuf ( reportFile , nullptr , _IOLBF , 0 ); // maybe  _IONBF


				jobIdType logId;
				if(sscanf(logFileName,"logs/%u.log",&logId)==1){
					std::to_string(logId);
					uniqueID=logId;
				}
			}
			if (reportFile==NULL){
				fprintf(stderr,"Impossible to open the rapport file\n");
				run=false;
			}
		}
	}
	arg.erase("-r");
	for (int i = 0; i < argc; ++i)
	{
		fprintf(reportFile,"%s ",argv[i]);
	}
	fprintf(reportFile,"\n");

	unsigned nbThreads=1;
	unsigned nbThreadsOverTi=1;
	unsigned nbThreadsLastLevel=1;
	unsigned totalNumberOfThreadVailable=1;
	bool verbose=false;


	#if _OPENMP
		totalNumberOfThreadVailable=omp_get_max_threads();
	#endif	

	if (arg.count("-j") >= 1)
	{
		std::multimap<std::string, std::string>::iterator jobsString=arg.lower_bound("-j");

		if(jobsString!=arg.upper_bound("-j")){
			float nbThreadsLoc=atof((jobsString->second).c_str());
			if(std::roundf(nbThreadsLoc) != nbThreadsLoc){
				nbThreadsLoc=std::max(std::floor(nbThreadsLoc*totalNumberOfThreadVailable),1.f);
			}
			nbThreads=(int)(nbThreadsLoc);
			++jobsString;
		}
		if(jobsString!=arg.upper_bound("-j")){
			float nbThreadsOverTiLoc=atof((jobsString->second).c_str());
			if(std::roundf(nbThreadsOverTiLoc) != nbThreadsOverTiLoc){
				nbThreadsOverTiLoc=std::max(std::floor(nbThreadsOverTiLoc*totalNumberOfThreadVailable),1.f);
			}
			nbThreadsOverTi=(int)(nbThreadsOverTiLoc);
			++jobsString;
		}
		if(jobsString!=arg.upper_bound("-j")){
			float nbThreadsLastLevelLoc=atof((jobsString->second).c_str());
			if(std::roundf(nbThreadsLastLevelLoc) != nbThreadsLastLevelLoc){
				nbThreadsLastLevelLoc=std::max(std::floor(nbThreadsLastLevelLoc*totalNumberOfThreadVailable),1.f);
			}
			nbThreadsLastLevel=(int)(nbThreadsLastLevelLoc);
			++jobsString;
		}
	}
	arg.erase("-j");

	if (arg.count("--jobs") >= 1)
	{
		std::multimap<std::string, std::string>::iterator jobsString=arg.lower_bound("--jobs");
		if(jobsString!=arg.upper_bound("--jobs")){
			float nbThreadsLoc=atof((jobsString->second).c_str());
			if(std::roundf(nbThreadsLoc) != nbThreadsLoc){
				nbThreadsLoc=std::max(std::floor(nbThreadsLoc*totalNumberOfThreadVailable),1.f);
			}
			nbThreads=(int)(nbThreadsLoc);
			++jobsString;
		}
		if(jobsString!=arg.upper_bound("--jobs")){
			float nbThreadsOverTiLoc=atof((jobsString->second).c_str());
			if(std::roundf(nbThreadsOverTiLoc) != nbThreadsOverTiLoc){
				nbThreadsOverTiLoc=std::max(std::floor(nbThreadsOverTiLoc*totalNumberOfThreadVailable),1.f);
			}
			nbThreadsOverTi=(int)(nbThreadsOverTiLoc);
			++jobsString;
		}
		if(jobsString!=arg.upper_bound("--jobs")){
			float nbThreadsLastLevelLoc=atof((jobsString->second).c_str());
			if(std::roundf(nbThreadsLastLevelLoc) != nbThreadsLastLevelLoc){
				nbThreadsLastLevelLoc=std::max(std::floor(nbThreadsLastLevelLoc*totalNumberOfThreadVailable),1.f);
			}
			nbThreadsLastLevel=(int)(nbThreadsLastLevelLoc);
			++jobsString;
		}
	}
	arg.erase("--jobs");

	if(nbThreads<1)
	#if _OPENMP
		nbThreads=totalNumberOfThreadVailable;
	#else
		nbThreads=1;
	#endif

	if ((arg.count("-h") == 1)|| (arg.count("--help") == 1))
	{
		printHelp();
		return 0;
	}
	arg.erase("-h");

	if ((arg.count("-v") == 1) || (arg.count("--verbose") == 1))
	{
		verbose=true;
	}
	arg.erase("--verbose");







	// test samnpling

	if(arg.count("-sampling") >= 1){
		std::multimap<std::string, std::string>::iterator jobsString=arg.lower_bound("-sampling");
		
		while(jobsString!=arg.upper_bound("-sampling")){
			unsigned nbdim=-1;
			unsigned sizes[3];
			if((jobsString->second).compare("1D")==0){
				nbdim=1;
				sizes[0]=1500;
			}
			if((jobsString->second).compare("2D")==0){
				nbdim=2;
				sizes[0]=256;
				sizes[1]=128;
			}
			if((jobsString->second).compare("3D")==0){
				nbdim=3;
				sizes[0]=64;
				sizes[1]=128;
				sizes[2]=256;
			}
			if(nbdim>0)
			{
				unsigned nbVariable=1;
				fprintf(reportFile, "sampling using %dD\n", nbdim);
				std::vector<SharedMemoryManager*> sharedMemoryManagerVector;
				std::vector<ComputeDeviceModule*> *computeDeviceModuleArray=new std::vector<ComputeDeviceModule*> [nbThreads];

				g2s::DataImage ti=g2s::DataImage( nbdim, sizes, 1);
				
				// fill with random

				unsigned seed1 =1;// std::chrono::system_clock::now().time_since_epoch().count();   
				std::mt19937 generator(seed1);
				std::uniform_real_distribution<float> distribution(0.0,1.0);

				for (int i = 0; i < ti.dataSize(); ++i)
				{
					ti._data[i]=distribution(generator);
				}
			
				std::vector<std::vector<float> > categoriesValues;

				bool needCrossMesurement=false;

				bool withGPU=false;
				bool conciderTiAsCircular=false;
				
				float nbCantidate=2.f;
				bool noVerbatim=false;
				bool useUniqueTI4Sampling=false;

				g2s::DataImage kernel;

				std::vector<float> variableWeight(nbVariable,1);
				std::vector<float> alphas(nbVariable,0);

				unsigned kernelSize=21;
				std::vector<g2s::KernelType> kernelsTypeFG(nbVariable,g2s::UNIFORM);
				std::vector<unsigned> maxSize(ti._dims.size(),kernelSize);
				kernel=g2s::DataImage::genearteKernel(kernelsTypeFG, maxSize, variableWeight, alphas);

				std::vector<std::vector<int> > pathPosition;
				pathPosition.push_back(std::vector<int>(0));
				for (size_t i = 0; i < kernel._dims.size(); ++i)
				{
					unsigned originalSize=pathPosition.size();
					int sizeInThistim=(kernel._dims[i]+1)/2;
					pathPosition.resize(originalSize*(2*sizeInThistim-1));
					for (unsigned int k = 0; k < originalSize; ++k)
					{
						pathPosition[k].push_back(0);
					}
					for (int j = 1; j < sizeInThistim; ++j)
					{
						std::copy ( pathPosition.begin(), pathPosition.begin()+originalSize, pathPosition.begin()+originalSize*(-1+2*j+0) );
						std::copy ( pathPosition.begin(), pathPosition.begin()+originalSize, pathPosition.begin()+originalSize*(-1+2*j+1) );
						for (unsigned int k = originalSize*(-1+2*j+0); k < originalSize*(-1+2*j+1); ++k)
						{
							pathPosition[k][i]=j;
						}
						for (unsigned int k = originalSize*(-1+2*j+1); k < originalSize*(-1+2*j+2); ++k)
						{
							pathPosition[k][i]=-j;
						}
					}
				}

				g2s::DataImage wieghtKernel=kernel.emptyCopy(true);
				for (unsigned int i = 0; i < wieghtKernel.dataSize(); ++i)
				{
					wieghtKernel._data[i]=-wieghtKernel.distance2ToCenter(i);
				}

				unsigned center=0;
				g2s::DataImage* wieghtKernelPtr=wieghtKernel.ptr();
				for (int i =  wieghtKernelPtr->_dims.size()-1; i>=0 ; i--)
				{
					center=center*wieghtKernelPtr->_dims[i]+(wieghtKernelPtr->_dims[i]-1)/2;
				}




				std::sort(pathPosition.begin(),pathPosition.end(),[wieghtKernelPtr, center](std::vector<int> &a, std::vector<int> &b){
					unsigned l1,l2;
					wieghtKernelPtr->indexWithDelta(l1, center, a);
					wieghtKernelPtr->indexWithDelta(l2, center, b);
					return wieghtKernelPtr->_data[l1] > wieghtKernelPtr->_data[l2];
				});
				

				std::vector<unsigned> numberDeComputedVariableProVariable(nbVariable,1);

				std::vector<std::vector<convertionType> > convertionTypeVectorMainVector;
				std::vector<g2s::OperationMatrix> coeficientMatrix;
				std::vector<std::vector<std::vector<convertionType> > > convertionTypeVectorConstVector;
				std::vector<std::vector<std::vector<float> > > convertionCoefVectorConstVector;
				ti.generateCoefMatrix4Xcorr(coeficientMatrix, convertionTypeVectorMainVector, convertionTypeVectorConstVector, convertionCoefVectorConstVector, needCrossMesurement, categoriesValues);

				SharedMemoryManager* smm=new SharedMemoryManager(ti._dims);

				std::vector<std::vector<g2s::DataImage> > variablesImages=ti.convertInput4Xcorr(smm->_fftSize, needCrossMesurement, categoriesValues);

				for (size_t j = 0; j < variablesImages.size(); ++j)
				{
					for (size_t k = 0; k < variablesImages[j].size(); ++k)
					{
						smm->addVaraible(variablesImages[j][k]._data);
					}
				}
				// alloc module
				#ifdef WITH_OPENCL
				std::vector<unsigned> gpuHostUnifiedMemory=OpenCLGPUDevice::DeviceWithHostUnifiedMemory(0);

				#endif

				#pragma omp parallel for proc_bind(spread) num_threads(nbThreads) default(none) shared(computeDeviceModuleArray) firstprivate(withGPU, gpuHostUnifiedMemory, conciderTiAsCircular, nbThreadsLastLevel,coeficientMatrix, smm, nbThreads, needCrossMesurement)
				for (unsigned int i = 0; i < nbThreads; ++i)
				{
					//#pragma omp critical (createDevices)
					{
						bool deviceCreated=false;
						#ifdef WITH_OPENCL
						if((!deviceCreated) && (i<gpuHostUnifiedMemory.size()) && withGPU){
							OpenCLGPUDevice* signleThread=new OpenCLGPUDevice(smm, coeficientMatrix, 0,gpuHostUnifiedMemory[i], needCrossMesurement, conciderTiAsCircular);
							signleThread->setTrueMismatch(false);
							computeDeviceModuleArray[i].push_back(signleThread);
							deviceCreated=true;
						}
						#endif
						if(!deviceCreated){
							CPUThreadDevice* signleThread=new CPUThreadDevice(smm, coeficientMatrix, nbThreadsLastLevel, needCrossMesurement, conciderTiAsCircular);
							signleThread->setTrueMismatch(false);
							computeDeviceModuleArray[i].push_back(signleThread);
							deviceCreated=true;
						}
					}
				}
				smm->allowNewModule(false);
				sharedMemoryManagerVector.push_back(smm);
				
				QuantileSamplingModule QSM(computeDeviceModuleArray,&kernel,nbCantidate,convertionTypeVectorMainVector, convertionTypeVectorConstVector, convertionCoefVectorConstVector, noVerbatim, !needCrossMesurement, nbThreads, nbThreadsOverTi, nbThreadsLastLevel, useUniqueTI4Sampling);

				bool circularSim=false;
				for (int ljhvhkj = 0; ljhvhkj < 30; ++ljhvhkj)
				{
					std::vector<std::vector<int> > neighborArrayVector;
					std::vector<std::vector<float> > neighborValueArrayVector;
					
					std::uniform_int_distribution<unsigned> distribution(0,ti.dataSize()/ti._nbVariable);
					{
						unsigned currentCell=distribution(generator);
						
						std::vector<unsigned> numberOfNeighborsProVariable(ti._nbVariable);
						std::vector<std::vector<int> > neighborArrayVector;
						std::vector<std::vector<float> > neighborValueArrayVector;
						
						{
							unsigned positionSearch=1;
							while((positionSearch<pathPosition.size())){
								unsigned dataIndex;
								std::vector<int> vectorInti=pathPosition[positionSearch];
								vectorInti.resize(ti._dims.size(),0);
								if(ti.indexWithDelta(dataIndex, currentCell, vectorInti) || circularSim)
								{
									std::vector<float> data(ti._nbVariable);
									unsigned cpt=0;
									float val;
									for (unsigned int i = 0; i < ti._nbVariable; ++i)
									{
										
										#pragma omp atomic read
										val=ti._data[dataIndex*ti._nbVariable+i];
										data[i]=val;
										cpt++;
										numberOfNeighborsProVariable[i]++;
										
									}
									neighborValueArrayVector.push_back(data);
									neighborArrayVector.push_back(pathPosition[positionSearch]);
									if(cpt==0) break;
								}
								positionSearch++;
							}
						}
						// conversion from one variable to many
						for (size_t j = 0; j < neighborValueArrayVector.size(); ++j)
						{
							std::vector<float> data(ti._nbVariable);
							unsigned id=0;
							unsigned idCategorie=0;
							for (unsigned int i = 0; i < ti._nbVariable; ++i)
							{
								if(ti._types[i]==g2s::DataImage::Continuous){
									
									data[id]=neighborValueArrayVector[j][i];
									id++;
								}
								if(ti._types[i]==g2s::DataImage::Categorical){
									for (size_t k = 0; k < categoriesValues[idCategorie].size(); ++k)
									{
										data[id] = (neighborValueArrayVector[j][i] == categoriesValues[idCategorie][k]);
										id++;
									}
									idCategorie++;
								}
							}
							neighborValueArrayVector[j]=data;
						}

						SamplingModule::matchLocation verbatimRecord;
						verbatimRecord.TI=2;

						auto importIndex=QSM.sample(neighborArrayVector,neighborValueArrayVector,0.0,verbatimRecord,0,false,0,0);

						fprintf(reportFile, "is %d and should be %d%s\n", importIndex.index,currentCell,importIndex.index==currentCell ? "": ", is wrong");
						allRight&=importIndex.index==currentCell;
					}

				}


				for (unsigned int i = 0; i < nbThreads; ++i)
				{
					for (size_t j = 0; j < computeDeviceModuleArray[i].size(); ++j)
					{
						delete computeDeviceModuleArray[i][j];
						computeDeviceModuleArray[i][j]=nullptr;
					}
				}

				for (size_t i = 0; i < sharedMemoryManagerVector.size(); ++i)
				{
					delete sharedMemoryManagerVector[i];
					sharedMemoryManagerVector[i]=nullptr;
				}

				delete[] computeDeviceModuleArray;

			}
			jobsString++;
		}
	}

	return (allRight ? 0 : -1);
}