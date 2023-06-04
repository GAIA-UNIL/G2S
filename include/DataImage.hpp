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

#ifndef DATA_IMAGE_HPP
#define DATA_IMAGE_HPP

#include <cmath>
#include <iostream>
#include <cstring>
#include <vector>
#include "utils.hpp"
#include "typeDefine.hpp"
#include "OperationMatrix.hpp"

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

	inline bool isEmpty(){
		return _types.size()==0;
	}

	inline DataImage(unsigned nbDim,unsigned sizes[],unsigned nbVariable)
	{
		_nbVariable=nbVariable;
		unsigned arraySize=nbVariable;
		for (unsigned int i = 0; i < nbDim; ++i)
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
		if(!raw) return DataImage();
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
		for (size_t i = 0; i < _dims.size(); ++i)
		{
			*((unsigned*)(raw+4*index))=_dims[i];
			index++;
		}
		*((unsigned*)(raw+4*index))=_types.size();
		index++;
		for (size_t i = 0; i < _types.size(); ++i)
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
		for (size_t i = 0; i < _dims.size(); ++i)
		{
			_dims[i]=*((unsigned*)(raw+4*index));
			index++;
		}
		_types.resize(*((unsigned*)(raw+4*index)));
		index++;
		for (size_t i = 0; i < _types.size(); ++i)
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
		snprintf(fullFilename,2048,"/tmp/G2S/data/%s.bgrid%s",filename.c_str(),extra);
		snprintf(outputFullFilename,2048,"/tmp/G2S/data/%s.bgrid%s",outputName,extra);
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
		for (size_t i = 0; i < _dims.size(); ++i)
		{
			result*=_dims[i];
		}
		return result;
	}

	inline bool indexWithDelta(unsigned &location, unsigned position, std::vector<int> &deltaVect, int* externalMemory=nullptr){
		unsigned finalValue=0;

		bool isOk=true;
		int *val=nullptr;
		if(!externalMemory){
			val=new int[_dims.size()];
		}else{
			val=externalMemory;
		}

		for (size_t i = 0; i < _dims.size(); ++i)
		{
			finalValue*=_dims[i];
			val[i]=position%_dims[i]+deltaVect[i];
			isOk &= (val[i]>=0) && (val[i] < int(_dims[i]));
			//val[i] = ((val[i] % int(_dims[i])) + int(_dims[i])) % int(_dims[i]);
			val[i] = (val[i] + int(_dims[i])) % int(_dims[i]);
			position/=_dims[i];
		}

		for (int i = int(_dims.size()-1); i >= 0; i--)
		{
			finalValue*=_dims[i];
			finalValue+=val[i];
		}

		location=finalValue;
		if(!externalMemory){
			delete[] val;
		}

		return isOk;
	}

	inline bool indexWithDelta(unsigned &location, unsigned position, std::vector<int> &deltaVect, std::vector<int> combi){
		unsigned finalValue=0;

		bool isOk=true;

		std::vector<int> val(_dims.size());
		int deltaVectoridx=0;
		for (size_t i = 0; i < _dims.size(); ++i)
		{
			finalValue*=_dims[i];
			val[i]=position%_dims[i]+( combi[i] ? deltaVect[deltaVectoridx++] : 0);
			isOk &= (val[i]>=0) && (val[i] < int(_dims[i]));
			//val[i] = ((val[i] % int(_dims[i])) + int(_dims[i])) % int(_dims[i]);
			val[i] = (val[i] + int(_dims[i])) % int(_dims[i]);
			position/=_dims[i];
		}

		for (int i = int(_dims.size()-1); i >= 0; i--)
		{
			finalValue*=_dims[i];
			finalValue+=val[i];
		}

		location=finalValue;

		return isOk;
	}

	inline DataImage convertLastDimInVariable(){
		DataImage result(_dims.size()-1,_dims.data(),_dims.back());
		unsigned newVariableSize=_dims.back();
		unsigned newDataSizeProLayer=dataSize()/newVariableSize;
		for (unsigned int i = 0; i < dataSize(); ++i)
		{
			result._data[(i%newDataSizeProLayer)*newVariableSize+i/newDataSizeProLayer]=_data[i];
		}
		return result;
	}

	inline void convertFirstDimInVariable(){
		_nbVariable=_dims.front();
		_dims.erase(_dims.begin());
		_types=std::vector<VaraibleType>(_nbVariable, _types[0]);
	}


	inline DataImage flipDataDimensions(){
		DataImage result(_dims.size(),_dims.data(),_dims.back());
		for (unsigned int i = 0; i < dataSize(); ++i)
		{
			result._data[i]=_data[flippedCoordinates(i)];
		}
		return result;
	}

	static inline DataImage genearteKernel(std::vector<g2s::KernelType> kernelsTypeForGeneration,std::vector<unsigned> maxSize, std::vector<float> variableWeight, std::vector<float> alphas){

		DataImage kernel(maxSize.size(), maxSize.data(), variableWeight.size());
		
		for (size_t j = 0; j < variableWeight.size(); ++j)
		{
			switch(kernelsTypeForGeneration[j]) {
				case  UNIFORM:
					for (unsigned int i = 0; i < kernel.dataSize()/kernel._nbVariable; ++i)
					{
						kernel._data[i*kernel._nbVariable+j]=variableWeight[j];
					}
					break;
				case  EXPONENTIAL:
					for (unsigned int i = 0; i < kernel.dataSize()/kernel._nbVariable; ++i)
					{
						float dist=sqrt(kernel.distance2ToCenter(i));
						kernel._data[i*kernel._nbVariable+j]=exp(-alphas[j]*dist);
					}
					break;
				case  GAUSSIAN:
					for (unsigned int i = 0; i < kernel.dataSize()/kernel._nbVariable; ++i)
					{
						float dist2=kernel.distance2ToCenter(i)*alphas[j]*alphas[j];
						kernel._data[i*kernel._nbVariable+j]=1/sqrt(2*M_PI)*exp(-dist2/2.f);
					}
					break;
						
				case TRIANGULAR :
				case EPANECHNIKOV :
				case QUARTIC :
				case TRIWEIGHT :
				case TRICUBE :
				case COSINE :
				case LOGISTIC :
				case SIGMOID :
				case SILVERMAN :
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

	static inline DataImage offsetKernel4categories(DataImage &currentKernel, std::vector<unsigned> factor){

		unsigned cumulated=0;
		for (size_t i = 0; i < factor.size(); ++i)
		{
			cumulated+=factor[i];
		}


		DataImage kernel=DataImage(currentKernel._dims.size(),currentKernel._dims.data(),cumulated);

		unsigned currentPosition=0;
		for (size_t i = 0; i < factor.size(); ++i)
		{
			for (unsigned int j = 0; j < factor[i]; ++j)
			{
				for (unsigned int k = 0; k < currentKernel.dataSize()/currentKernel._nbVariable; ++k)
				{
					kernel._data[k*cumulated+currentPosition]=currentKernel._data[k*currentKernel._nbVariable+i];
				}
				currentPosition++;
			}
		}
		
		return kernel;
	};

	public:
		float distance2ToCenter(unsigned index){
			std::vector<int> dists(_dims.size());
			int val=index;
			for (size_t i = 0; i < _dims.size(); ++i)
			{
				dists[i]=abs(int(val%_dims[i])-(int(_dims[i])-1)/2);
				val/=_dims[i];
			}

			float value=0.f;
			for (size_t i = 0; i < _dims.size(); ++i)
			{
				value+=dists[i]*dists[i];
			}
			return value;
		}

		std::vector<std::vector<g2s::DataImage> > convertInput4Xcorr( std::vector<unsigned> fftSize, bool needCrossMesurement, std::vector<std::vector<float> > categoriesValues){
			std::vector<std::vector<g2s::DataImage> > output;
			unsigned categoriesValuesIndex=0;
			for (unsigned int i = 0; i < _nbVariable; ++i)
			{
				if(_types[i]==Continuous){
					output.push_back(std::vector<g2s::DataImage>());
					output[i].push_back(g2s::DataImage(fftSize.size(),fftSize.data(),1));
					output[i].push_back(g2s::DataImage(fftSize.size(),fftSize.data(),1));
					//output[i].push_back(g2s::DataImage(fftSize.size(),fftSize.data(),1));
					memset(output[i][0]._data,0,sizeof(float) * output[i][0].dataSize());
					memset(output[i][1]._data,0,sizeof(float) * output[i][1].dataSize());
					//memset(output[i][2]._data,0,sizeof(float) * output[i][2].dataSize());

					for (unsigned int j = i; j < dataSize(); j+=_nbVariable)
					{
						unsigned newIndex=output[i][0].dataSize()-1-output[i][0].corrd2Index(index2Corrd(j/_nbVariable));
						output[i][0]._data[newIndex] = _data[j] * _data[j];
						if(std::isnan( _data[j] ))output[i][0]._data[newIndex] = 0.f;
						output[i][1]._data[newIndex] = _data[j];
						if(std::isnan( _data[j] ))output[i][1]._data[newIndex] = 0.f;
						//output[i][2]._data[newIndex] = (!std::isnan( _data[j] ));
					}	
				}

				if(_types[i]==Categorical){
					output.push_back(std::vector<g2s::DataImage>());
					unsigned numberOfCategorie=categoriesValues[categoriesValuesIndex].size();
					for (unsigned int k = 0; k < numberOfCategorie; ++k)
					{
						output[i].push_back(g2s::DataImage(fftSize.size(),fftSize.data(),1));
						memset(output[i][k]._data,0,sizeof(float) * output[i][k].dataSize());
					}

					for (unsigned int j = i; j < dataSize(); j+=_nbVariable)
					{
						unsigned newIndex=output[i][0].dataSize()-1-output[i][0].corrd2Index(index2Corrd(j/_nbVariable));
						for (unsigned int k = 0; k < numberOfCategorie; ++k)
						{
							//output[i][k]._data[newIndex] = (_data[j] != categoriesValues[categoriesValuesIndex][k])/float(numberOfCategorie);
							output[i][k]._data[newIndex] = (_data[j] == categoriesValues[categoriesValuesIndex][k]);
						}
					}
				}
				if(needCrossMesurement){
					output[i].push_back(g2s::DataImage(fftSize.size(),fftSize.data(),1));
					int lastPosition=output[i].size()-1;
					//std::fill(output[i][lastPosition]._data, output[i][lastPosition]._data+output[i][lastPosition].dataSize(),1.f);
					for (unsigned int j = i; j < dataSize(); j+=_nbVariable)
					{
						unsigned newIndex=output[i][0].dataSize()-1-output[i][0].corrd2Index(index2Corrd(j/_nbVariable));
						if(std::isnan(_data[j])){
							output[i][lastPosition]._data[newIndex] = 0.f;
						}else{
							output[i][lastPosition]._data[newIndex] = 1.f;
						}
					}
				}
			}

			return output;
		}

		std::vector<std::vector<unsigned> > computeMagninals(  std::vector<std::vector<float> > categoriesValues){
			std::vector<std::vector<unsigned> > marginal;
			unsigned categoriesValuesIndex=0;
			for (unsigned int i = 0; i < _nbVariable; ++i)
			{
				if(_types[i]==Continuous){
					marginal.push_back(std::vector<unsigned>());
				}

				if(_types[i]==Categorical){
					unsigned numberOfCategorie=categoriesValues[categoriesValuesIndex].size();
					marginal.push_back(std::vector<unsigned>(numberOfCategorie,0));
					for (unsigned int j = i; j < dataSize(); j+=_nbVariable)
					{
						for (unsigned int k = 0; k < numberOfCategorie; ++k)
						{
							marginal[i][k]+=(_data[j] == categoriesValues[categoriesValuesIndex][k]);
						}
					}
				}
			}

			return marginal;
		}

		void generateCoefMatrix4Xcorr(std::vector<g2s::OperationMatrix> &coeficientMatrix, std::vector<std::vector<convertionType> > &convertionTypeVectorMainVector,
				 std::vector<std::vector<std::vector<convertionType> > > &convertionTypeVectorConstVector, std::vector<std::vector<std::vector<float> > > &convertionCoefVectorConstVector,
				  bool forXMesurement, std::vector<std::vector<float> > categoriesValues){
			unsigned categoriesValuesIndex=0;
			int numberOfSubVariable=0;

			for (unsigned int i = 0; i < _nbVariable; ++i)
			{
				if(_types[i]==Continuous){
					numberOfSubVariable+=2;
				}

				if(_types[i]==Categorical){
					numberOfSubVariable+=categoriesValues[categoriesValuesIndex].size();
					categoriesValuesIndex++;
				}
				numberOfSubVariable+=forXMesurement;
			}

			g2s::OperationMatrix regular(numberOfSubVariable);
			g2s::OperationMatrix Xmeassurement(numberOfSubVariable);

			std::vector<std::vector<convertionType> > convertionTypeVectorConstRegular(_nbVariable);
			std::vector<std::vector<float> > convertionCoefVectorConstRegular(_nbVariable);
			std::vector<std::vector<convertionType> > convertionTypeVectorConstXmeassurement(_nbVariable);
			std::vector<std::vector<float> > convertionCoefVectorConstXmeassurement(_nbVariable);

			categoriesValuesIndex=0;
			int subVariablePosition=0;
			for (unsigned int i = 0; i < _nbVariable; ++i)
			{
				//convertionTypeVectorConstXmeassurement[i].push_back(P0);
				//convertionCoefVectorConstXmeassurement[i].push_back(1.f);

				if(_types[i]==Continuous){
					std::vector<convertionType> convType;
					convType.push_back(P0);
					regular.setVariableAt(subVariablePosition,subVariablePosition,-1.f);
					subVariablePosition+=1;
					convType.push_back(P1);
					regular.setVariableAt(subVariablePosition,subVariablePosition,2.f);
					subVariablePosition+=1;
					if(forXMesurement){
						convType.push_back(P2);
						regular.setVariableAt(subVariablePosition,subVariablePosition,-1.f);
						Xmeassurement.setVariableAt(subVariablePosition,subVariablePosition-2,1.f);
						subVariablePosition+=1;
					}else{
						convertionTypeVectorConstRegular[i].push_back(P2);
						convertionCoefVectorConstRegular[i].push_back(-1.f);
					}
					convertionTypeVectorMainVector.push_back(convType);
				}

				if(_types[i]==Categorical){
					unsigned numberOfCategorie=categoriesValues[categoriesValuesIndex].size();
					for (unsigned int k = 0; k < numberOfCategorie; ++k)
					{
						std::vector<convertionType> convType;
						convType.push_back(P1);
						regular.setVariableAt(subVariablePosition,subVariablePosition,1.f);
						subVariablePosition+=1;
						convertionTypeVectorMainVector.push_back(convType);
					}
					if(forXMesurement){
						std::vector<convertionType> convType;
						convType.push_back(P0);
						regular.setVariableAt(subVariablePosition,subVariablePosition,-1.f);
						Xmeassurement.setVariableAt(subVariablePosition,subVariablePosition,1.f);
						subVariablePosition+=1;
						convertionTypeVectorMainVector.push_back(convType);
					}
					categoriesValuesIndex++;
				}
			}
			coeficientMatrix.push_back(regular);
			if(forXMesurement)coeficientMatrix.push_back(Xmeassurement);

			convertionTypeVectorConstVector.push_back(convertionTypeVectorConstRegular);
			//convertionTypeVectorConstVector.push_back(std::vector<std::vector<convertionType> >(_nbVariable));
			//convertionTypeVectorConstVector.push_back(convertionTypeVectorConstXmeassurement);
			convertionCoefVectorConstVector.push_back(convertionCoefVectorConstRegular);
			//convertionCoefVectorConstVector.push_back(std::vector<std::vector<float> >(_nbVariable));
			//convertionCoefVectorConstVector.push_back(convertionCoefVectorConstXmeassurement);
		}

		/*void generateCoef4Xcorr(std::vector<std::vector<float> > &variablesCoeficientMainVector, std::vector<std::vector<convertionType> > &convertionTypeVectorMainVector, bool needCrossMesurement, std::vector<std::vector<float> > categoriesValues){
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
					categoriesValuesIndex++;
				}
				
			}
		}*/

	protected:
		std::vector<unsigned> index2Corrd(unsigned index){
			std::vector<unsigned> result(_dims.size());
			for (size_t i = 0; i < _dims.size(); ++i)
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

	public:
		unsigned flippedCoordinates(unsigned index){
			unsigned result=0;

			std::vector<unsigned> localDim=_dims;
			localDim.insert(localDim.begin(),_nbVariable);
			for (size_t i = 0; i < localDim.size(); ++i)
			{
				result*=localDim[i];
				result+=(index%localDim[i]);
				index/=localDim[i];
			}
			return result;
		}

	public:
		inline static DataImage readSGEMS(const char * fileName){

			FILE *file;
			file = fopen(fileName, "r");
			if (!file) DataImage();

			unsigned sizes[3];
			float offset;
			fscanf(file,"%d %d %d",sizes+0, sizes+1, sizes+2);
			fscanf(file,"%f %f %f",&offset, &offset, &offset);
			fscanf(file,"%f %f %f",&offset, &offset, &offset);
			unsigned nbDim=(sizes[0]>1)+(sizes[1]>1)+(sizes[2]>1);
			unsigned nbVariable;
			fscanf(file,"%d",&nbVariable);
			DataImage result=g2s::DataImage(nbDim, sizes, nbVariable);

			for (unsigned int i = 0; i < nbVariable; ++i)
			{
				int id;
				char variableName[1024];
				fscanf(file, "%s_%d", variableName, &id);
				if(strcmp("Continuous",variableName)==0){
					result._types[i]=Continuous;
				}
				if(strcmp("Categorical",variableName)==0){
					result._types[i]=Categorical;
				}
			}
			float* dataPtr=result._data;
			for (unsigned int i = 0; i < result.dataSize(); ++i)
			{
				fscanf(file,"%f", dataPtr+i);
			}
			fclose(file);
			return result;
		}

		inline void writeSGEMS(const char * fileName){
			FILE *file;
			file = fopen(fileName, "w");
			if (file) {
				fprintf(file,"%d %d %d 1.0 1.0 1.0 0.0 0.0 0.0\n",_dims[0], _dims.size()>1 ? _dims[1] : 1, _dims.size()>2 ? _dims[2] : 1);
				fprintf(file,"%lu\n",_types.size());
				for (size_t i = 0; i < _types.size(); ++i)
				{
					std::string startName;
					switch(_types[i]){
						case Continuous:
							startName="Continuous";
							break;
						case Categorical:
							startName="Categorical";
							break;
						default :
							startName="unkown";
					}
					fprintf(file,"%s_%zu ",startName.c_str(), i);
				}
				int nbVariable=_types.size();
				for (unsigned int i = 0; i < dataSize(); ++i)
				{
					if( i%(nbVariable)==0 ) fprintf(file,"\n");  //start new line for each position
					fprintf(file,"%f ", _data[i]);
					
				}
				fclose(file);
			}
		}

};


}

#endif // DATA_IMAGE_HPP