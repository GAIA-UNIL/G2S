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

#ifndef PYTHON_3_INTERFACE_HPP
#define PYTHON_3_INTERFACE_HPP

#include <Rcpp.h>
#include "interfaceTemplate.hpp"

/* Check for interrupt without long jumping */
void check_interrupt_fn(void *dummy) {
  R_CheckUserInterrupt();
}

int pending_interrupt() {
  return !(R_ToplevelExec(check_interrupt_fn, NULL));
}

class InterfaceTemplateR: public InterfaceTemplate
{
private:

public:

unsigned anyNativeToUnsigned(std::any val){
		Rcpp::RObject obj=std::any_cast<Rcpp::RObject>(val);
		return Rcpp::as<unsigned>(obj);
	}

	float anyNativeToFloat(std::any val){
		Rcpp::RObject obj=std::any_cast<Rcpp::RObject>(val);
		return Rcpp::as<float>(obj);
	}

	double anyNativeToDouble(std::any val){
		Rcpp::RObject obj=std::any_cast<Rcpp::RObject>(val);
		return Rcpp::as<double>(obj);
	}

	long unsigned anyNativeToLongUnsigned(std::any val){
		Rcpp::RObject obj=std::any_cast<Rcpp::RObject>(val);
		return Rcpp::as<long unsigned>(obj);
	}



	bool userRequestInteruption(){
		return pending_interrupt();
	}

	bool isDataMatrix(std::any val){
		Rcpp::RObject obj=std::any_cast<Rcpp::RObject>(val);
		if(!Rcpp::is<Rcpp::NumericVector>(obj)) return false;
		Rcpp::NumericVector vec=Rcpp::as<Rcpp::NumericVector>(obj);
		return vec.hasAttribute("dim");
	}

	std::string nativeToStandardString(std::any val){
		Rcpp::RObject obj=std::any_cast<Rcpp::RObject>(val);
		if(Rcpp::is<Rcpp::CharacterVector>(obj)) return Rcpp::as<std::string>(obj);
		if(Rcpp::is<Rcpp::StringVector>(obj)) return Rcpp::as<std::string>(obj);
		if(Rcpp::is<Rcpp::IntegerVector>(obj)) return std::to_string(Rcpp::as<int>(obj));
		if(Rcpp::is<Rcpp::NumericVector>(obj)) return std::to_string(Rcpp::as<double>(obj));
	}

	double nativeToScalar(std::any val){
		return Rcpp::as<float>(std::any_cast<Rcpp::RObject>(val));
	}

	unsigned nativeToUint32(std::any val){
		return Rcpp::as<unsigned>(std::any_cast<Rcpp::RObject>(val));
	};

	std::any ScalarToNative(double val){
		return std::any(Rcpp::RObject(Rcpp::NumericVector::create(val)));
	}

	std::any Uint32ToNative(unsigned val){
		return std::any(Rcpp::RObject(Rcpp::IntegerVector::create(val)));
	};

	void sendError(std::string val){
		Rcpp::stop("%s ==> %s","g2s:error", val.c_str());
	}

	void sendWarning(std::string val){
		printf("%s ==> %s","g2s:warning", val.c_str());
	}

	void eraseAndPrint(std::string val){
		printf("\r%s\n",val.c_str());
	}

	std::any convert2NativeMatrix(g2s::DataImage &image){
		std::vector<int> dimsArray(image._dims.size()+1);
		for (int i = 0; i < image._dims.size(); ++i)
		{
			dimsArray[i]=image._dims[i];
		}
		std::reverse(dimsArray.begin(),dimsArray.begin()+image._dims.size());
		dimsArray[image._dims.size()]=image._nbVariable;
		int nbOfVariable=image._nbVariable;
		Rcpp::RObject array;
		int completeSize=image.dataSize();
		if(dimsArray.back()==1)dimsArray.pop_back();
		if(image._encodingType==g2s::DataImage::Float){
			Rcpp::NumericVector localArray=Rcpp::NumericVector(completeSize);
			localArray.attr("dim")=Rcpp::as<Rcpp::IntegerVector>(Rcpp::wrap(dimsArray));
			//copy memory
			for (int i = 0; i < completeSize/nbOfVariable; ++i)
			{
				for (int j = 0; j < nbOfVariable; ++j)
				{
					localArray[i+j*(completeSize/nbOfVariable)]=image._data[i*nbOfVariable+j];
				}
			}
			array=localArray;
		}
		if(image._encodingType==g2s::DataImage::Integer|| image._encodingType==g2s::DataImage::UInteger){
			Rcpp::IntegerVector localArray=Rcpp::IntegerVector(completeSize);
			localArray.attr("dim")=Rcpp::as<Rcpp::IntegerVector>(Rcpp::wrap(dimsArray));
			//copy memory
			for (int i = 0; i < completeSize/nbOfVariable; ++i)
			{
				for (int j = 0; j < nbOfVariable; ++j)
				{
					localArray[i+j*(completeSize/nbOfVariable)]=image._data[i*nbOfVariable+j];
				}
			}
			array=localArray;
		}

		return array;
	}


	g2s::DataImage convertNativeMatrix2DataImage(std::any matrix, std::any dataTypeVariable=nullptr){
		Rcpp::NumericVector arrayRcpp=Rcpp::as<Rcpp::NumericVector>(std::any_cast<Rcpp::RObject>(matrix));
	
		int dataSize=arrayRcpp.size();
		int nbOfVariable=1;
		if(dataTypeVariable.type()==typeid(Rcpp::RObject))
		{
			Rcpp::NumericVector variableTypeArrayRcpp=Rcpp::as<Rcpp::NumericVector>(std::any_cast<Rcpp::RObject>(dataTypeVariable));
			nbOfVariable=variableTypeArrayRcpp.size();
		}
		Rcpp::IntegerVector dim_array = Rcpp::as<Rcpp::IntegerVector>(arrayRcpp.attr("dim"));
		int dimData = dim_array.size()-(nbOfVariable>1);
		unsigned dimArray[dimData];
		for (int i = 0; i < dimData; ++i)
		{
			dimArray[i]=dim_array[i];
		}

		std::reverse(dimArray,dimArray+dimData);


		g2s::DataImage image(dimData,dimArray,nbOfVariable);
		float *data=image._data;
		
		if (dataTypeVariable.type()==typeid(Rcpp::RObject))
		{
			Rcpp::NumericVector variableTypeArrayRcpp=Rcpp::as<Rcpp::NumericVector>(std::any_cast<Rcpp::RObject>(dataTypeVariable));
			for (int i = 0; i < nbOfVariable; ++i)
			{
				if(variableTypeArrayRcpp[i]==0.f)image._types[i]=g2s::DataImage::VaraibleType::Continuous;
				if(variableTypeArrayRcpp[i]==1.f)image._types[i]=g2s::DataImage::VaraibleType::Categorical;
			}
		}
		
		memset(data,0,sizeof(float)*dataSize);
		//manage data
		for (int i = 0; i < dataSize/nbOfVariable; ++i)
		{
			for (int j = 0; j < nbOfVariable; ++j)
			{
				data[i*nbOfVariable+j]=arrayRcpp[i+j*(dataSize/nbOfVariable)];
			}
		}
		return image;
	}


	void addInputsToInputsMap(std::multimap<std::string, std::any> &inputs, std::string key, Rcpp::RObject value){
		
		/*if(Rcpp::is<Rcpp::NumericVector>(value)){
			auto localValue=Rcpp::as<Rcpp::NumericVector>(value);
			for (int i = 0; i < localValue.size(); ++i)
			{
				inputs.insert(std::pair<std::string, std::any>(key,Rcpp::RObject(localValue[i])));
			}
			return;
		}*/
		if(Rcpp::is<Rcpp::List>(value)){
			auto localValue=Rcpp::as<Rcpp::List>(value);
			for (int i = 0; i < localValue.size(); ++i)
			{
				inputs.insert(std::pair<std::string, std::any>(key,Rcpp::RObject(localValue[i])));
			}
			return;
		}
		if(Rcpp::is<Rcpp::DataFrame>(value)){
			auto localValue=Rcpp::as<Rcpp::DataFrame>(value);
			for (int i = 0; i < localValue.size(); ++i)
			{
				inputs.insert(std::pair<std::string, std::any>(key,Rcpp::RObject(localValue[i])));
			}
			return;
		}

		inputs.insert(std::pair<std::string, std::any>(key,std::any(Rcpp::RObject(value))));	
	}



	Rcpp::RObject runStandardCommunicationR(Rcpp::List args, int numberOfOutput=INT_MAX){
		
		std::multimap<std::string, std::any> inputs;
		std::multimap<std::string, std::any> outputs;
		
		Rcpp::CharacterVector listNames=args.names();

		std::vector<Rcpp::RObject> withoutKey;

		for (int i = 0; i < args.size(); ++i)
		{
			std::string key=Rcpp::as<std::string>(listNames[i]);
			if(key.compare("") == 0)
			{
				withoutKey.push_back(args[i]);
			}else{
				addInputsToInputsMap(inputs,std::string("-")+key,args[i]);
			}
		}

		std::vector<int> listOfIndex;
		for (int i = 0; i < withoutKey.size(); ++i)
		{
			if(Rcpp::is<Rcpp::CharacterVector>(withoutKey[i])){
				std::string str=Rcpp::as<std::string>(withoutKey[i]);
				if( str.size()>1 && str.at(0)=='-' ) listOfIndex.push_back(i);
			}
		}
		listOfIndex.push_back(withoutKey.size());

		for (int j = 0; j < listOfIndex.size()-1; ++j)
		{
			if(listOfIndex[j]+1==listOfIndex[j+1]){
				inputs.insert(std::pair<std::string, std::any>(Rcpp::as<std::string>(withoutKey[listOfIndex[j]]),nullptr));
			}else{
				for (int i = listOfIndex[j]+1; i < listOfIndex[j+1]; ++i)
				{
					addInputsToInputsMap(inputs,Rcpp::as<std::string>(withoutKey[listOfIndex[j]]),withoutKey[i]);
				}
			}
		}

		runStandardCommunication(inputs, outputs, numberOfOutput);

		int nlhs=std::min(numberOfOutput,std::max(int(outputs.size())-1,1));

		std::vector<Rcpp::RObject> result;
		int position=0;
		for (int i=0; i < nlhs; ++i)
		{
			auto iter=outputs.find(std::to_string(i+1));
			if(iter!=outputs.end() && position<std::max(nlhs-1,1))
			{
				result.push_back(std::any_cast<Rcpp::RObject>(iter->second));
				position++;
			}else break;
		}

		if(position<nlhs){
			auto iter=outputs.find("t");
			if(iter!=outputs.end())
			{
				result.push_back(std::any_cast<Rcpp::RObject>(iter->second));
				position++;
			}
		}

		if(position<nlhs){
			auto iter=outputs.find("progression");
			if(iter!=outputs.end())
			{
				result.push_back(std::any_cast<Rcpp::RObject>(ScalarToNative(std::any_cast<float>(iter->second))));
				position++;
			}
		}

		if(position<nlhs){
			auto iter=outputs.find("id");
			if(iter!=outputs.end())
			{
				result.push_back(std::any_cast<Rcpp::RObject>(iter->second));
				position++;
			}
		}

		if(nlhs==1) return result[0];

		return Rcpp::as<Rcpp::List>(Rcpp::wrap(result));
	}
};


#endif // PYTHON_3_INTERFACE_HPP
