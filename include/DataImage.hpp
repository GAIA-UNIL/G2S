#ifndef DATA_IMAGE_HPP
#define DATA_IMAGE_HPP

#include <cmath>
#include <iostream>
#include <cstring>
#include <vector>
#include "utils.hpp"
#include "typeDefine.hpp"

char* loadRawData(const char * hash);
char* writeRawData(char* data, bool compresed=false);
void createLink(char* outputFullFilename, char* fullFilename);

namespace g2s{

class DataImage{
	public:
	enum VaraibleType{
		Continuous,
		Categorical
	};

	enum EncodingType{
		Float,
		Integer,
		UInteger
	};

	public:
	float* _data=nullptr;
	std::vector<unsigned> _dims;
	unsigned _nbVariable;
	std::vector<VaraibleType> _types;
	EncodingType _encodingType=EncodingType::Float;

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
		_types.resize(nbVariable,VaraibleType::Continuous);
	}

	static DataImage createFromFile(std::string filename)
	{
		char *raw=loadRawData((const char *) filename.c_str());
		DataImage toReturn(raw);
		free(raw);
		return toReturn; 
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
		_types=o._types;
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
	
	/**
	raw:
		- full size (full size included)
		- number of dim
		- size of each dim
		- number of variable
		- type of variable
		- data
	**/

	char* serialize(){

		size_t fullSize=sizeof(fullSize)+sizeof(unsigned)*(1+_dims.size()+1+_types.size()+1)+dataSize()*sizeof(float);
		char* raw=(char*)malloc(fullSize);
		size_t index=0;
		*((size_t*)(raw+4*index))=fullSize;
		index+=sizeof(fullSize)/4;
		*((unsigned*)(raw+4*index))=_dims.size();
		index++;
		for (int i = 0; i < _dims.size(); ++i)
		{
			*((unsigned*)(raw+4*index))=_dims[i];
			index++;
		}
		*((unsigned*)(raw+4*index))=_types.size();
		index++;
		for (int i = 0; i < _types.size(); ++i)
		{
			*((VaraibleType*)(raw+4*index))=_types[i];
			index++;
		}
		*((EncodingType*)(raw+4*index))=_encodingType;
		index++;
		memcpy(raw+4*index,_data,fullSize-4*index);

		return raw;

	}

	inline DataImage(char* raw){
		size_t index=0;
		size_t fullSize=*((size_t*)(raw+4*index));
		index+=sizeof(fullSize)/4;
		_dims.resize(*((unsigned*)(raw+4*index)));
		index++;
		for (int i = 0; i < _dims.size(); ++i)
		{
			_dims[i]=*((unsigned*)(raw+4*index));
			index++;
		}
		_types.resize(*((unsigned*)(raw+4*index)));
		index++;
		for (int i = 0; i < _types.size(); ++i)
		{
			_types[i]=*((VaraibleType*)(raw+4*index));
			index++;
		}
		_nbVariable=_types.size();
		_encodingType=*((EncodingType*)(raw+4*index));
		index++;
		_data=(float*)malloc(fullSize-4*index);
		memcpy(_data,raw+4*index,fullSize-4*index);
		
	}

	inline DataImage& operator=(DataImage&& o)
	{
		_dims=o._dims;
		_nbVariable=o._nbVariable;
		_data=o._data;
		_types=o._types;
		o._data=nullptr;
		o._nbVariable=0;
		o._dims.clear();
		o._types.clear();
		return *this;
	}

	inline DataImage* ptr(){
		return this;
	}

	inline ~DataImage(){
		if(_data)free(_data);
		_data=nullptr;		
	}

	inline void write(std::string filename, bool compresed=true){
		char* raw=serialize();
		char* outputName=writeRawData(raw,compresed);
		free(raw);
		//fprintf(stderr, "save as %s\n",outputName );
		//fprintf(stderr, "save as %s\n",filename.c_str() );
		char fullFilename[2048];
		char outputFullFilename[2048];
		char extra[16]={0};
		if(compresed) strcpy(extra,".gz");
		sprintf(fullFilename,"./data/%s.bgrid%s",filename.c_str(),extra);
		sprintf(outputFullFilename,"./%s.bgrid%s",outputName,extra);
		createLink(outputFullFilename, fullFilename);
		free(outputName);
	}

	inline DataImage emptyCopy(bool singleVariableOnly=false){
		return DataImage(_dims.size(),_dims.data(),( singleVariableOnly ? 1 : _nbVariable));
	}

	inline void setEncoding(EncodingType enc){
		_encodingType=enc;
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

		std::vector<int> val(_dims.size());

		for (int i = 0; i < _dims.size(); ++i)
		{
			finalValue*=_dims[i];
			val[i]=position%_dims[i]+deltaVect[i];
			if((val[i]<0)|| (val[i] >= _dims[i])){
				isOk=false;
			}
			position/=_dims[i];
		}

		for (int i = _dims.size()-1; i >= 0; i--)
		{
			finalValue*=_dims[i];
			finalValue+=val[i];
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
			int val=index;
			for (int i = 0; i < _dims.size(); ++i)
			{
				dists[i]=abs(int(val%_dims[i])-(int(_dims[i])-1)/2);
				val/=_dims[i];
			}

			float value=0.f;
			for (int i = 0; i < _dims.size(); ++i)
			{
				value+=dists[i]*dists[i];
			}
			return value;
		}

		std::vector<std::vector<g2s::DataImage> > convertInput4Xcorr( std::vector<unsigned> fftSize, bool needCrossMesurement, std::vector<std::vector<float> > categoriesValues){
			std::vector<std::vector<g2s::DataImage> > output;
			unsigned categoriesValuesIndex=0;
			for (int i = 0; i < _nbVariable; ++i)
			{
				if(_types[i]==Continuous){
					output.push_back(std::vector<g2s::DataImage>());
					output[i].push_back(g2s::DataImage(fftSize.size(),fftSize.data(),1));
					output[i].push_back(g2s::DataImage(fftSize.size(),fftSize.data(),1));
					//output[i].push_back(g2s::DataImage(fftSize.size(),fftSize.data(),1));
					memset(output[i][0]._data,0,sizeof(float) * output[i][0].dataSize());
					memset(output[i][1]._data,0,sizeof(float) * output[i][1].dataSize());
					//memset(output[i][2]._data,0,sizeof(float) * output[i][2].dataSize());

					for (int j = i; j < dataSize(); j+=_nbVariable)
					{
						unsigned newIndex=output[i][0].dataSize()-output[i][0].corrd2Index(index2Corrd(j/_nbVariable));
						output[i][0]._data[newIndex] = (!std::isnan( _data[j] )) * _data[j] * _data[j];
						output[i][1]._data[newIndex] = (!std::isnan( _data[j] )) * _data[j];
						//output[i][2]._data[newIndex] = (!std::isnan( _data[j] ));
					}	
				}

				if(_types[i]==Categorical){
					output.push_back(std::vector<g2s::DataImage>());
					unsigned numberOfCategorie=categoriesValues[categoriesValuesIndex].size();
					for (int k = 0; k < numberOfCategorie; ++k)
					{
						output[i].push_back(g2s::DataImage(fftSize.size(),fftSize.data(),1));
						memset(output[i][k]._data,0,sizeof(float) * output[i][k].dataSize());
					}

					for (int j = i; j < dataSize(); j+=_nbVariable)
					{
						unsigned newIndex=output[i][0].dataSize()-output[i][0].corrd2Index(index2Corrd(j/_nbVariable));
						for (int k = 0; k < numberOfCategorie; ++k)
						{
							//output[i][k]._data[newIndex] = (_data[j] != categoriesValues[categoriesValuesIndex][k])/float(numberOfCategorie);
							output[i][k]._data[newIndex] = (_data[j] == categoriesValues[categoriesValuesIndex][k]);
						}
					}
				}
			}

			if(needCrossMesurement){
				output.push_back(std::vector<g2s::DataImage>());
				output[_nbVariable].push_back(g2s::DataImage(fftSize.size(),fftSize.data(),1));
				memset(output[_nbVariable][0]._data,0,sizeof(float) * output[_nbVariable][0].dataSize());

				for (int j = 0; j < dataSize(); j++)
				{
					unsigned newIndex=output[_nbVariable][0].dataSize()-output[_nbVariable][0].corrd2Index(index2Corrd(j/_nbVariable));
					if(std::isnan(_data[j])){
						output[_nbVariable][0]._data[newIndex] = 0.f;
					}else{
						output[_nbVariable][0]._data[newIndex] = 1.f;
					}
				}
			}
			return output;
		}

		void generateCoef4Xcorr(std::vector<std::vector<float> > &variablesCoeficientMainVector, std::vector<std::vector<convertionType> > &convertionTypeVectorMainVector, bool needCrossMesurement, std::vector<std::vector<float> > categoriesValues){
			unsigned categoriesValuesIndex=0;

			for (int i = 0; i < _nbVariable; ++i)
			{

				if(_types[i]==Continuous){
					std::vector<float> coef;
					std::vector<convertionType> convType;
					coef.push_back(-1.f);
					convType.push_back(P0);
					coef.push_back(2.f);
					convType.push_back(P1);
					coef.push_back(-1.f);
					if(needCrossMesurement)convType.push_back(P2);
					variablesCoeficientMainVector.push_back(coef);
					convertionTypeVectorMainVector.push_back(convType);
				}

				if(_types[i]==Categorical){
					unsigned numberOfCategorie=categoriesValues[categoriesValuesIndex].size();
					for (int k = 0; k < numberOfCategorie; ++k)
					{
						std::vector<float> coef;
						std::vector<convertionType> convType;
						coef.push_back(1);
						convType.push_back(P1);
						variablesCoeficientMainVector.push_back(coef);
						convertionTypeVectorMainVector.push_back(convType);
					}
				}
				
			}
		}

	protected:
		std::vector<unsigned> index2Corrd(unsigned index){
			std::vector<unsigned> result(_dims.size());
			for (int i = 0; i < _dims.size(); ++i)
			{
				result[i]=index%_dims[i];
				index/=_dims[i];
			}
			return result;
		}

		unsigned corrd2Index(std::vector<unsigned> coord){
			unsigned result=0;

			for (int i = std::min(_dims.size(),coord.size())-1; i >=0 ; i--)
			{
				result*=_dims[i];
				result+=coord[i];
			}
			return result;
		}
};


}

#endif // DATA_IMAGE_HPP