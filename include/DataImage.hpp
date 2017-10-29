#ifndef DATA_IMAGE_HPP
#define DATA_IMAGE_HPP

#include <cmath>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <vector>
#include "utils.hpp"

float* loadData(const char * hash, int &sizeX, int &sizeY, int &sizeZ, int &dim, int &nbVariable);
char* writeData(float* data, int sizeX, int sizeY, int sizeZ, int dim, int nbVariable);
namespace g2s{

class DataImage{
	
	public:
	float* _data=nullptr;
	std::vector<unsigned> _dims;
	unsigned _nbVariable;

	inline DataImage()
	{
		_nbVariable=0;
		_data=nullptr;
	}

	inline DataImage(unsigned nbDim,unsigned sizes[],unsigned nbVariable)
	{
		_nbVariable=nbVariable;
		unsigned arraySize=nbVariable;
		for (int i = 0; i < nbDim; ++i)
		{
			_dims.push_back(sizes[i]);
			arraySize*=sizes[i];
		}
		_data=(float*)malloc(sizeof(float)*arraySize);
		memset(_data,0,sizeof(float)*arraySize);
	}

	inline DataImage(std::string filename)
	{
		int srcSizeX;
		int srcSizeY;
		int srcSizeZ;
		int srcDims;
		int srcVariable;
		_data=loadData((const char *) filename.c_str(),srcSizeX, srcSizeY, srcSizeZ, srcDims, srcVariable);
		_nbVariable=srcVariable;
		if(srcDims>0)_dims.push_back(srcSizeX);
		if(srcDims>1)_dims.push_back(srcSizeY);
		if(srcDims>2)_dims.push_back(srcSizeZ);
	}

	/*inline DataImage(const DataImage& o){
		_dims=o._dims;
		_nbVariable=o._nbVariable;
		_data=(float*)malloc(sizeof(float)*dataSize());
		memcpy(_data,o._data,sizeof(float)*dataSize());

		fprintf(stderr, "%s\n", "use copy constructor");
	}*/

	inline DataImage(DataImage&& o)
	{
		_dims=o._dims;
		_nbVariable=o._nbVariable;
		_data=o._data;
		o._data=nullptr;
		o._nbVariable=0;
		o._dims.clear();
		//fprintf(stderr, "%s\n", "use  move constructor ");
	}

	/*DataImage& operator=(const DataImage& other)
    {
    	*this=std::move(other);
        return *this;
    }*/

	inline DataImage& operator=(DataImage&& o)
	{
		_dims=o._dims;
		_nbVariable=o._nbVariable;
		_data=o._data;
		o._data=nullptr;
		o._nbVariable=0;
		o._dims.clear();
		//fprintf(stderr, "%s\n", "use  move assigne ");
		return *this;
	}

	inline DataImage* ptr(){
		return this;
	}

	inline ~DataImage(){
		if(_data)free(_data);
		_data=nullptr;		
	}

	inline void write(std::string filename){
		char* outputName=writeData(_data,(_dims.size()>0)?_dims[0]:1, (_dims.size()>1)?_dims[1]:1, (_dims.size()>2)?_dims[2]:1, _dims.size(), _nbVariable);
		//fprintf(stderr, "save as %s\n",outputName );
		//fprintf(stderr, "save as %s\n",filename.c_str() );
		char fullFilename[2048];
		char outputFullFilename[2048];
		sprintf(fullFilename,"./data/%s.bgrid",filename.c_str());
		sprintf(outputFullFilename,"./%s.bgrid",outputName);
		symlink(outputFullFilename, fullFilename);
		free(outputName);
	}

	inline DataImage emptyCopy(bool singleVariableOnly=false){
		return DataImage(_dims.size(),_dims.data(),( singleVariableOnly ? 1 : _nbVariable));
	}

	inline unsigned dataSize(){
		unsigned result= _nbVariable;
		for (int i = 0; i < _dims.size(); ++i)
		{
			result*=_dims[i];
		}
		return result;
	}

	inline bool indexWithDelta(unsigned &location, unsigned position, std::vector<int> &deltaVect){
		unsigned finalValue=0;
		std::vector<int> dists(_dims.size());

		bool isOk=true;
		for (int i = 0; i < _dims.size(); ++i)
		{
			int val=position;
			for (int j = 0; j < i; ++j)
			{
				val/=_dims[j];
			}
			val%=_dims[i];
			val+=deltaVect[i];
			if((val<0)|| (val>= _dims[i])){
				isOk=false;
			}
			for (int j = 0; j < i; ++j)
			{
				val*=_dims[j];
			}
			finalValue+=val;
		}
		location=finalValue;

		return isOk;
	}

	static inline DataImage genearteKernel(std::vector<g2s::KernelType> kernelsTypeForGeneration,std::vector<unsigned> maxSize, std::vector<float> variableWeight, std::vector<float> alphas){

		DataImage kernel(maxSize.size(), maxSize.data(), variableWeight.size());
		
		for (int j = 0; j < variableWeight.size(); ++j)
		{
			switch(kernelsTypeForGeneration[j]) {
				case  UNIFORM:
					for (int i = 0; i < kernel.dataSize()/kernel._nbVariable; ++i)
					{
						kernel._data[i*kernel._nbVariable+j]=variableWeight[j];
					}
					break;
				case  EXPONENTIAL:
					for (int i = 0; i < kernel.dataSize()/kernel._nbVariable; ++i)
					{
						float dist=sqrt(kernel.distance2ToCenter(i));
						kernel._data[i*kernel._nbVariable+j]=exp(-alphas[j]*dist);
					}
					break;
				case  GAUSSIAN:
					for (int i = 0; i < kernel.dataSize()/kernel._nbVariable; ++i)
					{
						float dist2=kernel.distance2ToCenter(i)*alphas[j]*alphas[j];
						kernel._data[i*kernel._nbVariable+j]=1/sqrt(2*M_PI)*exp(-dist2/2.f);
					}
					break;
			}
		}
		/*
	enum KernelType{
		UNIFORM,
		TRIANGULAR,
		EXPONENTIAL,
		EPANECHNIKOV,
		QUARTIC,
		TRIWEIGHT,
		TRICUBE,
		GAUSSIAN,
		COSINE,
		LOGISTIC,
		SIGMOID,
		SILVERMAN
	};
	*/
		return kernel;
	};

	public:
		float distance2ToCenter(unsigned index){
			std::vector<int> dists(_dims.size());
			for (int i = 0; i < _dims.size(); ++i)
			{
				int val=index;
				for (int j = 0; j < i; ++j)
				{
					val/=_dims[j];
				}
				val%=_dims[i];
				dists[i]=abs(val-(int(_dims[i])-1)/2);
			}

			float value=0.f;
			for (int i = 0; i < _dims.size(); ++i)
			{
				value+=dists[i]*dists[i];
			}
			return value;
		}
};
}

#endif // DATA_IMAGE_HPP