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

#ifndef MATLAB_INTERFACE_HPP
#define MATLAB_INTERFACE_HPP

#define MATLAB

#include <mex.h>
#include "mexInterrupt.hpp"
#include "matrix.h"
#include "inerfaceTemplate.hpp"

#ifndef MATLAB_VERSION
#define MATLAB_VERSION 0
#endif

class InerfaceTemplateMatlab: public InerfaceTemplate
{

std::future<void> InterruptCheck;
std::atomic<bool> _done=false;

public:

	InerfaceTemplateMatlab(){
		InterruptCheck = mexInterrupt::startInterruptCheck(_done);
	}

	~InerfaceTemplateMatlab(){
		_done=true;
		InterruptCheck.wait();
	}


	void updateDisplay(){
		mexEvalString("drawnow");
	}

	bool userRequestInteruption(){
		return _done;
	}

	bool isDataMatrix(std::any val){
		return mxIsNumeric(std::any_cast<mxArray const*>(val)) || mxIsLogical(std::any_cast<mxArray const*>(val));
	}

	std::string nativeToStandardString(std::any val){
		mxArray const* matlabArray=std::any_cast<mxArray const*>(val);
		if(mxIsChar(matlabArray))
			return std::string(mxArrayToString(matlabArray)); //mxGetString
		if(mxIsScalar(matlabArray))
			return std::to_string(float(mxGetScalar((matlabArray))));
		return "";
	}

	double nativeToScalar(std::any val){
		return mxGetScalar(std::any_cast<mxArray const*>(val));
	}

	unsigned nativeToUint32(std::any val){
		return *(unsigned*)mxGetPr(std::any_cast<mxArray const*>(val));
	};

	template <typename type>
	type anyNativeTo(std::any val){
		type result;
		mxArray const* prh=std::any_cast<mxArray const*>(val);
		//manage data
		if(mxIsDouble(prh)){
			double *matrixData=(double *)mxGetPr(prh);
			result=matrixData[0];
		}
		if(mxIsSingle(prh)){
			float *matrixData=(float *)mxGetPr(prh);
			result=matrixData[0];
		}
		if(mxGetClassID(prh)==mxUINT8_CLASS){
			uint8_t *matrixData=(uint8_t *)mxGetPr(prh);
			result=matrixData[0];
		}
		if(mxGetClassID(prh)==mxUINT16_CLASS){
			uint16_t *matrixData=(uint16_t *)mxGetPr(prh);
			result=matrixData[0];
		}
		if(mxGetClassID(prh)==mxUINT32_CLASS){
			uint32_t *matrixData=(uint32_t *)mxGetPr(prh);
			result=matrixData[0];
		}
		if(mxGetClassID(prh)==mxUINT64_CLASS){
			uint64_t *matrixData=(uint64_t *)mxGetPr(prh);
			result=matrixData[0];
		}
		if(mxGetClassID(prh)==mxINT8_CLASS){
			int8_t *matrixData=(int8_t *)mxGetPr(prh);
			result=matrixData[0];
		}
		if(mxGetClassID(prh)==mxINT16_CLASS){
			int16_t *matrixData=(int16_t *)mxGetPr(prh);
			result=matrixData[0];
		}
		if(mxGetClassID(prh)==mxINT32_CLASS){
			int32_t *matrixData=(int32_t *)mxGetPr(prh);
			result=matrixData[0];
		}
		if(mxGetClassID(prh)==mxINT64_CLASS){
			int64_t *matrixData=(int64_t *)mxGetPr(prh);
			result=matrixData[0];
		}
		if(mxGetClassID(prh)==mxLOGICAL_CLASS){
			bool *matrixData=(bool *)mxGetPr(prh);
			result=matrixData[0];
		}
		return result;
	};

	unsigned anyNativeToUnsigned(std::any val){
		return anyNativeTo<unsigned>(val);
	}
	float anyNativeToFloat(std::any val){
		return anyNativeTo<float>(val);
	}
	double anyNativeToDouble(std::any val){
		return anyNativeTo<double>(val);
	}
	long unsigned anyNativeToLongUnsigned(std::any val){
		return anyNativeTo<long unsigned>(val);
	}

	std::any ScalarToNative(double val){
		size_t one=1;
		mxArray *output=mxCreateNumericArray(1, &one, mxDOUBLE_CLASS, mxREAL);
		*(double*)mxGetPr(output)=val;
		return std::any(output);
	}

	std::any Uint32ToNative(unsigned val){
		size_t one=1;
		mxArray *output=mxCreateNumericArray(1, &one, mxUINT32_CLASS, mxREAL);
		*(unsigned*)mxGetPr(output)=val;
		return std::any(output);
	};

	void sendError(std::string val){
		mexErrMsgIdAndTxt("g2s:error", val.c_str());
	}

	void sendWarning(std::string val){
		mexErrMsgIdAndTxt("g2s:error", val.c_str());
	}

	void eraseAndPrint(std::string val){
		mxArray *arg[1];
		arg[0]=mxCreateString(val.c_str());
		mexCallMATLAB(0,NULL,1,arg,"disp");
	}

	std::any convert2NativeMatrix(g2s::DataImage &image){
		size_t *dimsArray=(size_t *)malloc(sizeof(size_t)*(image._dims.size()+1));
		for (size_t i = 0; i < image._dims.size(); ++i)
		{
			dimsArray[i]=image._dims[i];
		}
		std::reverse(dimsArray,dimsArray+image._dims.size());
		dimsArray[image._dims.size()]=image._nbVariable;
		mxArray *array=nullptr;
		if(image._encodingType==g2s::DataImage::Float)
			array=mxCreateNumericArray(image._dims.size()+1, dimsArray, mxSINGLE_CLASS , mxREAL);
		if(image._encodingType==g2s::DataImage::Integer)
			array=mxCreateNumericArray(image._dims.size()+1, dimsArray, mxINT32_CLASS , mxREAL);
		if(image._encodingType==g2s::DataImage::UInteger)
			array=mxCreateNumericArray(image._dims.size()+1, dimsArray, mxUINT32_CLASS , mxREAL);
		free(dimsArray);

		float* data=(float*)mxGetPr(array);
		unsigned nbOfVariable=image._nbVariable;
		unsigned dataSize=image.dataSize();
		
		#pragma omp parallel for simd
		for (int i = 0; i < dataSize; ++i)
		{
			data[image.flippedCoordinates(i)]=image._data[i];
		}
		
		return std::any(array);
	}


	g2s::DataImage convertNativeMatrix2DataImage(std::any matrix, std::any dataTypeVariable=nullptr){
		
		mxArray const* prh=std::any_cast<mxArray const*>(matrix);
		mxArray const* variableTypeArray=nullptr;
		if(dataTypeVariable.type()==typeid(mxArray const*)) variableTypeArray=std::any_cast<mxArray const*>(dataTypeVariable);

		int dataSize=mxGetNumberOfElements(prh);
		int nbOfVariable=1;
		if(variableTypeArray)nbOfVariable=mxGetNumberOfElements(variableTypeArray);
		int dimData = mxGetNumberOfDimensions(prh)-(nbOfVariable>1);
		const size_t * dim_array = mxGetDimensions(prh);
		if(nbOfVariable>1 && dim_array[dimData]!=nbOfVariable)
			sendError("Last dimension of the inputed matrix do not fit -dt parameter size");
		unsigned *dimArray=(unsigned *)malloc(sizeof(unsigned)*dimData);
		for (int i = 0; i < dimData; ++i)
		{
			dimArray[i]=dim_array[i];
		}

		std::reverse(dimArray,dimArray+dimData);		

		g2s::DataImage image(dimData,dimArray,nbOfVariable);
		free(dimArray);
		float *data=image._data;
		
		
		if (variableTypeArray && mxIsSingle(variableTypeArray))
		{
			float* ptrVarType=(float *)mxGetPr(variableTypeArray);
			for (int i = 0; i < nbOfVariable; ++i)
			{
				if(ptrVarType[i]==0.f)image._types[i]=g2s::DataImage::VaraibleType::Continuous;
				if(ptrVarType[i]==1.f)image._types[i]=g2s::DataImage::VaraibleType::Categorical;
			}
		}

		if (variableTypeArray && mxIsDouble(variableTypeArray))
		{
			double* ptrVarType=(double *)mxGetPr(variableTypeArray);
			for (int i = 0; i < nbOfVariable; ++i)
			{
				if(ptrVarType[i]==0.)image._types[i]=g2s::DataImage::VaraibleType::Continuous;
				if(ptrVarType[i]==1.)image._types[i]=g2s::DataImage::VaraibleType::Categorical;
			}
		}
		
		memset(data,0,sizeof(float)*dataSize);

		//manage data
		if(mxIsDouble(prh)){
			double *matrixData=(double *)mxGetPr(prh);
			#pragma omp parallel for simd
			for (int i = 0; i < dataSize; ++i)
			{
				data[i]=matrixData[image.flippedCoordinates(i)];
			}
		}
		if(mxIsSingle(prh)){
			float *matrixData=(float *)mxGetPr(prh);
			#pragma omp parallel for simd
			for (int i = 0; i < dataSize; ++i)
			{
				data[i]=matrixData[image.flippedCoordinates(i)];
			}
		}
		if(mxGetClassID(prh)==mxUINT8_CLASS){
			uint8_t *matrixData=(uint8_t *)mxGetPr(prh);
			#pragma omp parallel for simd
			for (int i = 0; i < dataSize; ++i)
			{
				data[i]=matrixData[image.flippedCoordinates(i)];
			}
		}
		if(mxGetClassID(prh)==mxUINT16_CLASS){
			uint16_t *matrixData=(uint16_t *)mxGetPr(prh);
			#pragma omp parallel for simd
			for (int i = 0; i < dataSize; ++i)
			{
				data[i]=matrixData[image.flippedCoordinates(i)];
			}
		}
		if(mxGetClassID(prh)==mxUINT32_CLASS){
			uint32_t *matrixData=(uint32_t *)mxGetPr(prh);
			#pragma omp parallel for simd
			for (int i = 0; i < dataSize; ++i)
			{
				data[i]=matrixData[image.flippedCoordinates(i)];
			}
		}
		if(mxGetClassID(prh)==mxUINT64_CLASS){
			uint64_t *matrixData=(uint64_t *)mxGetPr(prh);
			#pragma omp parallel for simd
			for (int i = 0; i < dataSize; ++i)
			{
				data[i]=matrixData[image.flippedCoordinates(i)];
			}
		}
		if(mxGetClassID(prh)==mxINT8_CLASS){
			int8_t *matrixData=(int8_t *)mxGetPr(prh);
			#pragma omp parallel for simd
			for (int i = 0; i < dataSize; ++i)
			{
				data[i]=matrixData[image.flippedCoordinates(i)];
			}
		}
		if(mxGetClassID(prh)==mxINT16_CLASS){
			int16_t *matrixData=(int16_t *)mxGetPr(prh);
			#pragma omp parallel for simd
			for (int i = 0; i < dataSize; ++i)
			{
				data[i]=matrixData[image.flippedCoordinates(i)];
			}
		}
		if(mxGetClassID(prh)==mxINT32_CLASS){
			int32_t *matrixData=(int32_t *)mxGetPr(prh);
			#pragma omp parallel for simd
			for (int i = 0; i < dataSize; ++i)
			{
				data[i]=matrixData[image.flippedCoordinates(i)];
			}
		}
		if(mxGetClassID(prh)==mxINT64_CLASS){
			int64_t *matrixData=(int64_t *)mxGetPr(prh);
			#pragma omp parallel for simd
			for (int i = 0; i < dataSize; ++i)
			{
				data[i]=matrixData[image.flippedCoordinates(i)];
			}
		}

		if(mxGetClassID(prh)==mxLOGICAL_CLASS){
			bool *matrixData=(bool *)mxGetPr(prh);
			#pragma omp parallel for simd
			for (int i = 0; i < dataSize; ++i)
			{
				data[i]=matrixData[image.flippedCoordinates(i)];
			}
		}

		return image;
	}


	void runStandardCommunicationMatlab(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
		
		if(nrhs>0 && mxIsChar(prhs[0])){
			std::string str=mxArrayToString(prhs[0]);
			if(str=="--version"){ 
				if(nlhs>0 && nrhs==1){
					for (int i = 0; i < nlhs; ++i)
					{
						if(i==0) plhs[0]=mxCreateString(VERSION);
						if(i==1) plhs[1]=mxCreateString(__DATE__);
						if(i==2) plhs[2]=mxCreateString(__TIME__);
					}
				}else{
					char buff[1000];
					snprintf(buff, sizeof(buff), "G2S version %s, compiled the %s %s with R%x",VERSION,__DATE__,__TIME__,MATLAB_VERSION);
					std::string buffAsStdStr = buff;
					eraseAndPrint(buffAsStdStr);
				}
				return;
			}
		}

		std::multimap<std::string, std::any> inputs;
		std::multimap<std::string, std::any> outputs;

		std::vector<int> listOfIndex;
		
		for (int i = 0; i < nrhs; ++i)
		{
			if(mxIsChar(prhs[i])){
				std::string str=mxArrayToString(prhs[i]);
				if( str.size()>1 && str.at(0)=='-' ) listOfIndex.push_back(i);
			}
		}
		listOfIndex.push_back(nrhs);

		for (int j = 0; j < int(listOfIndex.size())-1; ++j)
		{
			if(listOfIndex[j]+1==listOfIndex[j+1]){
				inputs.insert(std::pair<std::string, std::any>(mxArrayToString(prhs[listOfIndex[j]]),nullptr));
			}else{
				for (int i = listOfIndex[j]+1; i < listOfIndex[j+1]; ++i)
				{
					if(mxIsCell(prhs[i])){
						for (size_t p = 0; p < mxGetNumberOfElements(prhs[i]); ++p)
						{
							inputs.insert(std::pair<std::string, std::any>(mxArrayToString(prhs[listOfIndex[j]]),std::any((const mxArray *)mxGetCell(prhs[i],p))));
						}
					}
					else
						inputs.insert(std::pair<std::string, std::any>(mxArrayToString(prhs[listOfIndex[j]]),std::any(prhs[i])));
				}
			}
		}

		runStandardCommunication(inputs, outputs, nlhs);

		int position=0;
		for (int i=0; i < nlhs; ++i)
		{
			auto iter=outputs.find(std::to_string(i+1));
			if(iter!=outputs.end() && position<std::max(nlhs-1,1))
			{
				plhs[position]=std::any_cast<mxArray *>(iter->second);
				position++;
			}
		}

		if(position<nlhs){
			auto iter=outputs.find("t");
			if(iter!=outputs.end())
			{
				plhs[position]=std::any_cast<mxArray *>(iter->second);
				position++;
			}
		}


		if(position<nlhs){
			auto iter=outputs.find("progression");
			if(iter!=outputs.end())
			{
				plhs[position]=std::any_cast<mxArray *>(ScalarToNative(std::any_cast<float>(iter->second)));
				position++;
			}
		}

		if(position<nlhs){
			auto iter=outputs.find("id");
			if(iter!=outputs.end())
			{
				plhs[position]=std::any_cast<mxArray *>(iter->second);
				position++;
			}
		}
	}
};


#endif // MATLAB_INTERFACE_HPP
