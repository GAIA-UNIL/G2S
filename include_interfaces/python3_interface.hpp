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

#ifndef PYTHON_VERSION
#define PYTHON_VERSION "unknown"
#endif

#include <Python.h>
#include <numpy/arrayobject.h>
#include "inerfaceTemplate.hpp"

class InerfaceTemplatePython3: public InerfaceTemplate
{
private:
	PyThreadState *_save=nullptr;
public:

	void unlockThread(){
		_save = PyEval_SaveThread();
	}
	void lockThread(){
		PyEval_RestoreThread(_save);
		_save=nullptr;
	}

	bool userRequestInteruption(){
		bool status=false;
		if(_save){
			lockThread();
			status=PyErr_CheckSignals();
			unlockThread();
		}else{
			status=PyErr_CheckSignals();
		}
		return status;
	}

	bool isDataMatrix(std::any val){
		return PyArray_Check(std::any_cast<PyObject *>(val));
	}

	std::string nativeToStandardString(std::any val){
		PyObject * pyObj=std::any_cast<PyObject *>(val);
		if(PyUnicode_Check(pyObj))
			return std::string(PyUnicode_AsUTF8(pyObj)); //mxGetString
		if(PyFloat_Check(pyObj))
			return std::to_string(float(PyFloat_AsDouble(pyObj)));
		if(PyLong_Check(pyObj))
			return std::to_string(unsigned(PyLong_AsLong(pyObj)));
		return "";
	}

	double nativeToScalar(std::any val){
		return PyFloat_AsDouble(std::any_cast<PyObject *>(val));
	}

	unsigned nativeToUint32(std::any val){
		return PyLong_AsLong(std::any_cast<PyObject *>(val));
	};

	unsigned anyNativeToUnsigned(std::any val){
		PyObject * prh=std::any_cast<PyObject *>(val);
		unsigned result;
		//manage data
		if(PyLong_Check(prh)){
			result= PyLong_AsUnsignedLong(prh);
		}
		if(PyFloat_Check(prh)){
			result= PyFloat_AsDouble(prh);
		}
		return result;
	}

	float anyNativeToFloat(std::any val){
		PyObject * prh=std::any_cast<PyObject *>(val);
		float result;
		//manage data
		if(PyLong_Check(prh)){
			result= PyLong_AsDouble(prh);
		}
		if(PyFloat_Check(prh)){
			result= PyFloat_AsDouble(prh);
		}
		return result;
	}

	double anyNativeToDouble(std::any val){
		PyObject * prh=std::any_cast<PyObject *>(val);
		double result;
		//manage data
		if(PyLong_Check(prh)){
			result= PyLong_AsDouble(prh);
		}
		if(PyFloat_Check(prh)){
			result= PyFloat_AsDouble(prh);
		}
		return result;
	}

	long unsigned anyNativeToLongUnsigned(std::any val){
		PyObject * prh=std::any_cast<PyObject *>(val);
		long unsigned result;
		//manage data
		if(PyLong_Check(prh)){
			result= PyLong_AsUnsignedLong(prh);
		}
		if(PyFloat_Check(prh)){
			result= PyFloat_AsDouble(prh);
		}
		return result;
	}

	std::any ScalarToNative(double val){
		return std::any(PyFloat_FromDouble(val));
	}

	std::any Uint32ToNative(unsigned val){
		return std::any(PyLong_FromUnsignedLong(val));
	};

	void sendError(std::string val){
		if(_save){
			lockThread();
			PyErr_Format(PyExc_Exception,"%s ==> %s","g2s:error", val.c_str()); //PyExc_Exception
			throw "G2S interrupt";
			//unlockThread();
		}else{
			PyErr_Format(PyExc_Exception,"%s ==> %s","g2s:error", val.c_str()); //PyExc_Exception
			throw "G2S interrupt";
		}

	}

	void sendWarning(std::string val){
		if(_save){
			lockThread();
			PyErr_WarnFormat(PyExc_Warning,2,"%s ==> %s","g2s:warning", val.c_str());
			unlockThread();
		}else{
			PyErr_WarnFormat(PyExc_Warning,2,"%s ==> %s","g2s:warning", val.c_str());
		}
	}

	void eraseAndPrint(std::string val){
		printf("\r%s        ",val.c_str());
	}

	std::any convert2NativeMatrix(g2s::DataImage &image){
		npy_intp* dimsArray=new npy_intp[image._dims.size()+1];
		for (size_t i = 0; i < image._dims.size(); ++i)
		{
			dimsArray[i]=image._dims[i];
		}
		std::reverse(dimsArray,dimsArray+image._dims.size());
		dimsArray[image._dims.size()]=image._nbVariable;
		
		PyObject *array=nullptr;
		if(image._encodingType==g2s::DataImage::Float)
			array=PyArray_SimpleNew(image._dims.size()+(image._nbVariable>1), dimsArray,NPY_FLOAT);
		if(image._encodingType==g2s::DataImage::Integer)
			array=PyArray_SimpleNew(image._dims.size()+(image._nbVariable>1), dimsArray,NPY_INT32);
		if(image._encodingType==g2s::DataImage::UInteger)
			array=PyArray_SimpleNew(image._dims.size()+(image._nbVariable>1), dimsArray,NPY_UINT32);
		delete[] dimsArray;
		float* data=(float*)PyArray_DATA(array);
		//unsigned nbOfVariable=image._nbVariable;
		unsigned dataSize=image.dataSize();
		/*for (int i = 0; i < dataSize/nbOfVariable; ++i)
		{
			for (int j = 0; j < nbOfVariable; ++j)
			{
				data[i+j*(dataSize/nbOfVariable)]=image._data[i*nbOfVariable+j];
			}
		}*/

		memcpy(data,image._data,dataSize*sizeof(float));

		return array;
	}


	g2s::DataImage convertNativeMatrix2DataImage(std::any matrix, std::any dataTypeVariable=nullptr){
		PyObject * prh=std::any_cast<PyObject *>(matrix);
		PyObject * variableTypeArray=nullptr;
		if(dataTypeVariable.type()==typeid(PyObject *))
			variableTypeArray=std::any_cast<PyObject *>(dataTypeVariable);

		prh=PyArray_ContiguousFromAny(prh,PyArray_TYPE(prh),0,0);
		int dataSize=PyArray_SIZE(prh);
		int nbOfVariable=1;
		if(variableTypeArray)nbOfVariable=PyArray_SIZE(variableTypeArray);
		int dimData = PyArray_NDIM(prh)-(nbOfVariable>1);
		const npy_intp * dim_array = PyArray_DIMS(prh);
		if(nbOfVariable>1 && dim_array[dimData]!=nbOfVariable)
			sendError("Last dimension of the inputed matrix do not fit -dt parameter size");
		unsigned *dimArray=new unsigned[dimData];
		for (int i = 0; i < dimData; ++i)
		{
			dimArray[i]=dim_array[i];
		}

		std::reverse(dimArray,dimArray+dimData);


		g2s::DataImage image(dimData,dimArray,nbOfVariable);
		delete[] dimArray;
		float *data=image._data;
		
		
		if (variableTypeArray && PyArray_TYPE(variableTypeArray)==NPY_FLOAT)
		{
			float* ptrVarType=(float *)PyArray_DATA(variableTypeArray);
			for (int i = 0; i < nbOfVariable; ++i)
			{
				if(ptrVarType[i]==0.f)image._types[i]=g2s::DataImage::VaraibleType::Continuous;
				if(ptrVarType[i]==1.f)image._types[i]=g2s::DataImage::VaraibleType::Categorical;
			}
		}

		if (variableTypeArray && PyArray_TYPE(variableTypeArray)==NPY_DOUBLE)
		{
			double* ptrVarType=(double *)PyArray_DATA(variableTypeArray);
			for (int i = 0; i < nbOfVariable; ++i)
			{
				if(ptrVarType[i]==0.)image._types[i]=g2s::DataImage::VaraibleType::Continuous;
				if(ptrVarType[i]==1.)image._types[i]=g2s::DataImage::VaraibleType::Categorical;
			}
		}
		
		memset(data,0,sizeof(float)*dataSize);
		//manage data
		if(PyArray_TYPE(prh)==NPY_DOUBLE){
			double *matrixData=(double *)PyArray_DATA(prh);
			#pragma omp parallel for simd
			for (int i = 0; i < dataSize; ++i)
			{
				data[i]=matrixData[i];
			}
		}
		if(PyArray_TYPE(prh)==NPY_FLOAT){
			float *matrixData=(float *)PyArray_DATA(prh);
			#pragma omp parallel for simd
			for (int i = 0; i < dataSize; ++i)
			{
				data[i]=matrixData[i];
			}
		}
		if(PyArray_TYPE(prh)==NPY_UINT8){
			uint8_t *matrixData=(uint8_t *)PyArray_DATA(prh);
			#pragma omp parallel for simd
			for (int i = 0; i < dataSize; ++i)
			{
				data[i]=matrixData[i];
			}
		}
		if(PyArray_TYPE(prh)==NPY_UINT16){
			uint16_t *matrixData=(uint16_t *)PyArray_DATA(prh);
			#pragma omp parallel for simd
			for (int i = 0; i < dataSize; ++i)
			{
				data[i]=matrixData[i];
			}
		}
		if(PyArray_TYPE(prh)==NPY_UINT32){
			uint32_t *matrixData=(uint32_t *)PyArray_DATA(prh);
			#pragma omp parallel for simd
			for (int i = 0; i < dataSize; ++i)
			{
				data[i]=matrixData[i];
			}
		}
		if(PyArray_TYPE(prh)==NPY_UINT64){
			uint64_t *matrixData=(uint64_t *)PyArray_DATA(prh);
			#pragma omp parallel for simd
			for (int i = 0; i < dataSize; ++i)
			{
				data[i]=matrixData[i];
			}
		}
		if(PyArray_TYPE(prh)==NPY_INT8){
			int8_t *matrixData=(int8_t *)PyArray_DATA(prh);
			#pragma omp parallel for simd
			for (int i = 0; i < dataSize; ++i)
			{
				data[i]=matrixData[i];
			}
		}
		if(PyArray_TYPE(prh)==NPY_INT16){
			int16_t *matrixData=(int16_t *)PyArray_DATA(prh);
			#pragma omp parallel for simd
			for (int i = 0; i < dataSize; ++i)
			{
				data[i]=matrixData[i];
			}
		}
		if(PyArray_TYPE(prh)==NPY_INT32){
			int32_t *matrixData=(int32_t *)PyArray_DATA(prh);
			#pragma omp parallel for simd
			for (int i = 0; i < dataSize; ++i)
			{
				data[i]=matrixData[i];
			}
		}
		if(PyArray_TYPE(prh)==NPY_INT64){
			int64_t *matrixData=(int64_t *)PyArray_DATA(prh);
			#pragma omp parallel for simd
			for (int i = 0; i < dataSize; ++i)
			{
				data[i]=matrixData[i];
			}
		}

		if(PyArray_TYPE(prh)==NPY_BOOL){
			bool *matrixData=(bool *)PyArray_DATA(prh);
			#pragma omp parallel for simd
			for (int i = 0; i < dataSize; ++i)
			{
				data[i]=matrixData[i];
			}
		}
		return image;
	}

	void addInputsToInputsMap(std::multimap<std::string, std::any> &inputs, std::string key, PyObject* value){
		if(PyTuple_Check(value)){
			for (int p = 0; p < PyTuple_Size(value) ; ++p)
			{
				inputs.insert(std::pair<std::string, std::any>(key,std::any(PyTuple_GetItem(value,p))));
			}
			return;
		}
		if(PyList_Check(value)){
			for (int p = 0; p < PyList_Size(value) ; ++p)
			{
				inputs.insert(std::pair<std::string, std::any>(key,std::any(PyList_GetItem(value,p))));
			}
			return;
		}
		if(PySet_Check(value)){
			auto it = PyObject_GetIter(value);
		    if (it == NULL)
		        return;
		    PyObject* obj;
		    while ((obj = PyIter_Next(it)) != NULL) {
				inputs.insert(std::pair<std::string, std::any>(key,std::any(obj)));
		        Py_DECREF(obj);
		    }
		    Py_DECREF(it);
			return;
		}
		if(PyArray_Check(value) && PyArray_ISOBJECT(value)){
			for (int p = 0; p < PyArray_SIZE(value) ; ++p)
			{
				inputs.insert(std::pair<std::string, std::any>(key,std::any(PyArray_GETITEM(value,PyArray_GETPTR1(value,p)))));
			}
			return;
		}
		
		inputs.insert(std::pair<std::string, std::any>(key,std::any(value)));
	}


	PyObject * runStandardCommunicationPython(PyObject *self, PyObject *args, PyObject *keywds, int numberOfOutput=INT_MAX){
		setbuf(stdout, NULL);
		std::multimap<std::string, std::any> inputs;
		std::multimap<std::string, std::any> outputs;

		// the tuple
		if(args && PyTuple_Check(args)){
			if(PyTuple_Size(args)>0 && PyUnicode_Check(PyTuple_GetItem(args,0))){
				std::string str=std::string(PyUnicode_AsUTF8(PyTuple_GetItem(args,0)));
				if(str=="--version"){ 
					if(numberOfOutput!=INT_MAX){
						PyObject* pyResult=PyTuple_New(3);
						PyTuple_SetItem(pyResult,0,PyUnicode_FromString(VERSION));
						PyTuple_SetItem(pyResult,1,PyUnicode_FromString(__DATE__));
						PyTuple_SetItem(pyResult,2,PyUnicode_FromString(__TIME__));
						return pyResult;
					}else{
						char buff[1000];
						snprintf(buff, sizeof(buff), "G2S version %s, compiled the %s %s with Python %s",VERSION,__DATE__,__TIME__,PYTHON_VERSION);
						std::string buffAsStdStr = buff;
						printf("%s\n",buffAsStdStr.c_str());
					}
					return Py_None;
				}
			}
			std::vector<int> listOfIndex;
			for (int i = 0; i < PyTuple_Size(args); ++i)
			{
				if(PyUnicode_Check(PyTuple_GetItem(args,i))){
					std::string str=std::string(PyUnicode_AsUTF8(PyTuple_GetItem(args,i)));
					if( str.size()>1 && str.at(0)=='-' ) listOfIndex.push_back(i);
				}
			}
			listOfIndex.push_back(PyTuple_Size(args));

			for (size_t j = 0; j < listOfIndex.size()-1; ++j)
			{
				if(listOfIndex[j]+1==listOfIndex[j+1]){
					inputs.insert(std::pair<std::string, std::any>(std::string(PyUnicode_AsUTF8(PyTuple_GetItem(args,listOfIndex[j]))),nullptr));
				}else{
					//("%s\n",std::string(PyUnicode_AsUTF8(PyTuple_GetItem(args,listOfIndex[j]))).c_str());
					for (int i = listOfIndex[j]+1; i < listOfIndex[j+1]; ++i)
					{
						addInputsToInputsMap(inputs,std::string(PyUnicode_AsUTF8(PyTuple_GetItem(args,listOfIndex[j]))),PyTuple_GetItem(args,i));
					}
				}
			}
		}

		// keywords

		if(keywds && PyDict_CheckExact(keywds)){
		#ifndef PYPY_VERSION
			PyArg_ValidateKeywordArguments(keywds);
		#endif
			PyObject *key, *value;
			Py_ssize_t pos = 0;

			while (PyDict_Next(keywds, &pos, &key, &value)) {
				addInputsToInputsMap(inputs,std::string("-")+std::string(PyUnicode_AsUTF8(key)),value);
			}
		}

		// manage '-dt'
		if(inputs.count("-dt")>0){
			if(!PyArray_Check(std::any_cast<PyObject*>(inputs.find("-dt")->second))){
				PyObject* list=PyList_New(inputs.count("-dt"));
				int i=0;
				for (auto it=inputs.equal_range("-dt").first; it!=inputs.equal_range("-dt").second; ++it)
				{
					PyList_SetItem(list,i,std::any_cast<PyObject*>(it->second));
					i++;
				}
				inputs.erase("-dt");
				addInputsToInputsMap(inputs,"-dt",PyArray_FromAny(list,PyArray_DescrFromType(NPY_FLOAT), 1,1, 0, NULL));
			}else{ // it's a numpy array
				PyArrayObject* dt=(PyArrayObject*)std::any_cast<PyObject*>(inputs.find("-dt")->second);
				inputs.erase("-dt");
				addInputsToInputsMap(inputs,"-dt",PyArray_CastToType(dt,PyArray_DescrFromType(NPY_FLOAT),0));
			}
		}
		// fprintf(stderr, "%d\n", inputs.count("-dt"));
		// if(inputs.count("-dt")==1){
		// 	fprintf(stderr, "%s\n", inputs.find("-dt")->second.type().name());
		// 	PyObject* dt=std::any_cast<PyObject*>(inputs.find("-dt")->second);
		// 	if(!PyArray_Check(dt)){
		// 		PyObject* newDt=PyArray_FromAny(dt,NULL, 1,1, 0, NULL);
		// 		inputs.find("-dt")->second=newDt;
		// 		PyObject* objectsRepresentation = PyObject_Repr(newDt);
		// 		fprintf(stderr, "%s\n", PyUnicode_AsUTF8(objectsRepresentation));
		// 	}
		// }else{
		// 	fprintf(stderr, "%s\n", inputs.find("-dt")->second.type().name());
		// 	PyObject* dt=std::any_cast<PyObject*>(inputs.find("-dt")->second);
			
		// 	PyObject* objectsRepresentation = PyObject_Repr(dt);
		// 	fprintf(stderr, "%s\n", PyUnicode_AsUTF8(objectsRepresentation));

		// 	// if(!PyArray_Check(dt)){
		// 	// 	PyObject* newDt=PyArray_FromAny(&dt,NULL, 1,1, 0, NULL);
				
		// 	// 	PyObject* objectsRepresentation = PyObject_Repr(newDt);
		// 	// 	fprintf(stderr, "%s\n", PyUnicode_AsUTF8(&dt));

		// 	// 	// fprintf(stderr, "%d\n",PyArray_NDIM(newDt) );
		// 	// 	// for (int i = 0; i < PyArray_NDIM(newDt); ++i)
		// 	// 	// {
		// 	// 	// 	fprintf(stderr, "%d\n", PyArray_DIMS(newDt)[i]);
		// 	// 	// }
		// 	// 	inputs.find("-dt")->second=newDt;
		// 	// }
		// }


		try{
			runStandardCommunication(inputs, outputs, numberOfOutput);
		}catch(const char* msg){
			return Py_None;
		}

		if(outputs.size()==0){
			return Py_None;
		}

		int nlhs=std::min(numberOfOutput,std::max(int(outputs.size()),1));
		// printf("requested output %d\n",nlhs);
		PyObject* pyResult=PyTuple_New(nlhs);
		int position=0;
		for (int i=0; i < nlhs; ++i)
		{
			auto iter=outputs.find(std::to_string(i+1));
			if(iter!=outputs.end() && position<std::max(nlhs-1,1))
			{
				PyTuple_SetItem(pyResult,position,std::any_cast<PyObject*>(iter->second));
				Py_INCREF(PyTuple_GetItem(pyResult,position));
				position++;
			}
		}

		if(position<nlhs){
			auto iter=outputs.find("t");
			if(iter!=outputs.end())
			{
				PyTuple_SetItem(pyResult,position,std::any_cast<PyObject*>(iter->second));
				Py_INCREF(PyTuple_GetItem(pyResult,position));
				position++;
			}
		}

		if(position<nlhs){
			auto iter=outputs.find("progression");
			if(iter!=outputs.end())
			{
				PyTuple_SetItem(pyResult,position,std::any_cast<PyObject*>(ScalarToNative(std::any_cast<float>(iter->second))));
				position++;
			}
		}


		if(position<nlhs){
			auto iter=outputs.find("id");
			if(iter!=outputs.end())
			{
				PyTuple_SetItem(pyResult,position,std::any_cast<PyObject*>(iter->second));
				Py_INCREF(PyTuple_GetItem(pyResult,position));
				position++;
			}
		}

		for (auto it=outputs.begin(); it!=outputs.end(); ++it){
			if(it->second.type()==typeid(PyObject *)){
				Py_DECREF(std::any_cast<PyObject*>(it->second));
			}
		}

		if((nlhs==1) && (numberOfOutput==INT_MAX) && (PyTuple_Size(pyResult)==1 )){
			PyObject* pyResultUnique=PyTuple_GetItem(pyResult,0);
			Py_INCREF(pyResultUnique);
			Py_DECREF(pyResult);
			return pyResultUnique;
		}
		if(nlhs==0){
			Py_DECREF(pyResult);
			return Py_None;
		}


		return pyResult;
	}
};


#endif // PYTHON_3_INTERFACE_HPP

