/*
 * G2S - Python 3 Interface
 * Copyright (C) 2018-2025
 *   Mathieu Gravey (gravey.mathieu@gmail.com) and UNIL (University of Lausanne)
 *
 * SPDX-License-Identifier: LGPL-3.0-only
 */

#ifndef PYTHON_3_INTERFACE_HPP
#define PYTHON_3_INTERFACE_HPP

#ifndef PYTHON_VERSION
#define PYTHON_VERSION "unknown"
#endif

#include <Python.h>
#include <numpy/arrayobject.h>
#include "interfaceTemplate.hpp"
#include <algorithm>
#include <any>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>

/* ------------------------------------------------------------------
 * NumPy ≥ 2.0 compatibility wrappers (strict PyArrayObject typing)
 * ------------------------------------------------------------------ */
#ifndef G2S_NUMPY_COMPAT_H
#define G2S_NUMPY_COMPAT_H
#define NPY_OBJ(o) reinterpret_cast<const PyArrayObject*>(o)
#define PyArray_DATA_SAFE(o)    PyArray_DATA(NPY_OBJ(o))
#define PyArray_TYPE_SAFE(o)    PyArray_TYPE(NPY_OBJ(o))
#define PyArray_DIMS_SAFE(o)    PyArray_DIMS(NPY_OBJ(o))
#define PyArray_NDIM_SAFE(o)    PyArray_NDIM(NPY_OBJ(o))
#define PyArray_SIZE_SAFE(o)    PyArray_SIZE(NPY_OBJ(o))
#define PyArray_BYTES_SAFE(o)   PyArray_BYTES(NPY_OBJ(o))
#define PyArray_STRIDES_SAFE(o) PyArray_STRIDES(NPY_OBJ(o))
#define PyArray_ISOBJECT_SAFE(o) (PyTypeNum_ISOBJECT(PyArray_TYPE_SAFE(o)))
#define PyArray_GETPTR1_SAFE(o,i) ((void*)(PyArray_BYTES_SAFE(o) + (i)*PyArray_STRIDES_SAFE(o)[0]))
#define PyArray_GETITEM_SAFE(o,p) PyArray_GETITEM(NPY_OBJ(o),(const char*)(p))
#endif

/* ------------------------------------------------------------------
 * Python3 interface
 * ------------------------------------------------------------------ */
class InterfaceTemplatePython3 : public InterfaceTemplate {
private:
    PyThreadState* _save = nullptr;

public:
    /* ------------ threading ------------ */
    void unlockThread() override { _save = PyEval_SaveThread(); }
    void lockThread()override  {
        if (_save) { PyEval_RestoreThread(_save); _save = nullptr; }
    }

    bool userRequestInteruption() override{
        if (_save) { lockThread(); bool s = PyErr_CheckSignals(); unlockThread(); return s; }
        return PyErr_CheckSignals();
    }

    /* ------------ conversions ------------ */
    bool isDataMatrix (std::any v) override {
        return PyArray_Check(std::any_cast<PyObject*>(v));
    }

    std::string nativeToStandardString(std::any v) override {
        PyObject* o = std::any_cast<PyObject*>(v);
        if (PyUnicode_Check(o)) return PyUnicode_AsUTF8(o);
        if (PyFloat_Check(o))   return std::to_string(PyFloat_AsDouble(o));
        if (PyLong_Check(o))    return std::to_string(PyLong_AsLong(o));
        return {};
    }

    double nativeToScalar(std::any v) override {
        return PyFloat_AsDouble(std::any_cast<PyObject*>(v));
    }

    unsigned nativeToUint32(std::any v) override {
        return PyLong_AsUnsignedLong(std::any_cast<PyObject*>(v));
    }


    unsigned anyNativeToUnsigned(std::any val) override {
        PyObject* obj = std::any_cast<PyObject*>(val);
        if (PyLong_Check(obj)) return PyLong_AsUnsignedLong(obj);
        if (PyFloat_Check(obj)) return static_cast<unsigned>(PyFloat_AsDouble(obj));
        return 0;
    }

    float anyNativeToFloat(std::any val) override {
        PyObject* obj = std::any_cast<PyObject*>(val);
        if (PyLong_Check(obj)) return static_cast<float>(PyLong_AsDouble(obj));
        if (PyFloat_Check(obj)) return static_cast<float>(PyFloat_AsDouble(obj));
        return 0.f;
    }

    double anyNativeToDouble(std::any val) override {
        PyObject* obj = std::any_cast<PyObject*>(val);
        if (PyLong_Check(obj)) return PyLong_AsDouble(obj);
        if (PyFloat_Check(obj)) return PyFloat_AsDouble(obj);
        return 0.;
    }

    unsigned long anyNativeToLongUnsigned(std::any val) override {
        PyObject* obj = std::any_cast<PyObject*>(val);
        if (PyLong_Check(obj)) return PyLong_AsUnsignedLong(obj);
        if (PyFloat_Check(obj)) return static_cast<unsigned long>(PyFloat_AsDouble(obj));
        return 0;
    }

    std::any ScalarToNative(double val) override {
        return std::any(PyFloat_FromDouble(val));
    }

    std::any Uint32ToNative(unsigned val) override {
        return std::any(PyLong_FromUnsignedLong(val));
    }


    /* ------------ interface-required overrides ------------ */
    void sendError(std::string val) override {
        if (_save) lockThread();
        PyErr_Format(PyExc_Exception, "g2s:error ==> %s", val.c_str());
        throw std::runtime_error("G2S interrupt");
    }

    void sendWarning(std::string val) override {
        if (_save) lockThread();
        PyErr_WarnFormat(PyExc_Warning, 2, "g2s:warning ==> %s", val.c_str());
        if (_save) unlockThread();
    }

    void eraseAndPrint(std::string val) override {
        printf("\r%s        ", val.c_str());
        fflush(stdout);
    }

    /* ------------ numpy conversion helpers ------------ */
    std::any convert2NativeMatrix(g2s::DataImage& img) override {
        const size_t ndim = img._dims.size();
        std::vector<npy_intp> dims(ndim + 1);
        for (size_t i = 0; i < ndim; ++i) dims[i] = img._dims[i];
        std::reverse(dims.begin(), dims.begin() + ndim);
        dims[ndim] = img._nbVariable;

        int typ = NPY_FLOAT;
        if (img._encodingType == g2s::DataImage::Integer)  typ = NPY_INT32;
        if (img._encodingType == g2s::DataImage::UInteger) typ = NPY_UINT32;

        PyObject* arr = PyArray_SimpleNew((int)(ndim + (img._nbVariable > 1)),
                                          dims.data(), typ);
        float* data = (float*)PyArray_DATA_SAFE(arr);
        std::memcpy(data, img._data, img.dataSize() * sizeof(float));
        return arr;
    }

    g2s::DataImage convertNativeMatrix2DataImage(std::any matrix,
                                                 std::any dataTypeVariable = nullptr) override {
        PyObject* prh = std::any_cast<PyObject*>(matrix);
        PyObject* variableTypeArray = nullptr;
        if (dataTypeVariable.type() == typeid(PyObject*))
            variableTypeArray = std::any_cast<PyObject*>(dataTypeVariable);

        prh = PyArray_ContiguousFromAny(prh, PyArray_TYPE_SAFE(prh), 0, 0);
        int dataSize = PyArray_SIZE_SAFE(prh);
        int nbVar = 1;
        if (variableTypeArray) nbVar = PyArray_SIZE_SAFE(variableTypeArray);
        int dimData = PyArray_NDIM_SAFE(prh) - (nbVar > 1);
        const npy_intp* dims = PyArray_DIMS_SAFE(prh);

        if (nbVar > 1 && dims[dimData] != nbVar)
            sendError("Last dimension of input matrix does not match -dt size");

        std::vector<unsigned> dimArray(dimData);
        for (int i = 0; i < dimData; ++i) dimArray[i] = dims[i];
        std::reverse(dimArray.begin(), dimArray.end());

        g2s::DataImage image(dimData, dimArray.data(), nbVar);
        float* data = image._data;
        std::memset(data, 0, sizeof(float) * dataSize);

        if (variableTypeArray && PyArray_TYPE_SAFE(variableTypeArray) == NPY_FLOAT) {
            auto* ptr = (float*)PyArray_DATA_SAFE(variableTypeArray);
            for (int i = 0; i < nbVar; ++i)
                image._types[i] = ptr[i] == 0.f
                                      ? g2s::DataImage::VaraibleType::Continuous
                                      : g2s::DataImage::VaraibleType::Categorical;
        } else if (variableTypeArray && PyArray_TYPE_SAFE(variableTypeArray) == NPY_DOUBLE) {
            auto* ptr = (double*)PyArray_DATA_SAFE(variableTypeArray);
            for (int i = 0; i < nbVar; ++i)
                image._types[i] = ptr[i] == 0.
                                      ? g2s::DataImage::VaraibleType::Continuous
                                      : g2s::DataImage::VaraibleType::Categorical;
        }

#define COPY_TYPED(TYPE, ENUM) \
    if (PyArray_TYPE_SAFE(prh) == ENUM) { \
        auto* src = (TYPE*)PyArray_DATA_SAFE(prh); \
        _Pragma("omp parallel for simd") \
        for (int i = 0; i < dataSize; ++i) data[i] = (float)src[i]; \
    }
        COPY_TYPED(double, NPY_DOUBLE)
        COPY_TYPED(float, NPY_FLOAT)
        COPY_TYPED(uint8_t, NPY_UINT8)
        COPY_TYPED(uint16_t, NPY_UINT16)
        COPY_TYPED(uint32_t, NPY_UINT32)
        COPY_TYPED(uint64_t, NPY_UINT64)
        COPY_TYPED(int8_t, NPY_INT8)
        COPY_TYPED(int16_t, NPY_INT16)
        COPY_TYPED(int32_t, NPY_INT32)
        COPY_TYPED(int64_t, NPY_INT64)
        COPY_TYPED(bool, NPY_BOOL)
#undef COPY_TYPED
        return image;
    }

    /* ------------ utility ------------ */
    void addInputsToInputsMap(std::multimap<std::string, std::any>& inputs,
                              const std::string& key, PyObject* value) {
        if (PyTuple_Check(value)) {
            for (Py_ssize_t i = 0; i < PyTuple_Size(value); ++i)
                inputs.emplace(key, PyTuple_GetItem(value, i));
            return;
        }
        if (PyList_Check(value)) {
            for (Py_ssize_t i = 0; i < PyList_Size(value); ++i)
                inputs.emplace(key, PyList_GetItem(value, i));
            return;
        }
        if (PySet_Check(value)) {
            if (PyObject* it = PyObject_GetIter(value)) {
                while (PyObject* obj = PyIter_Next(it)) {
                    inputs.emplace(key, obj);
                    Py_DECREF(obj);
                }
                Py_DECREF(it);
            }
            return;
        }
        if (PyArray_Check(value) && PyArray_ISOBJECT_SAFE(value)) {
            for (Py_ssize_t p = 0; p < PyArray_SIZE_SAFE(value); ++p) {
                auto ptr = PyArray_GETPTR1_SAFE(value, p);
                inputs.emplace(key, PyArray_GETITEM_SAFE(value, ptr));
            }
            return;
        }
        inputs.emplace(key, value);
    }

    inline PyObject * runStandardCommunicationPython(PyObject *self, PyObject *args, PyObject *keywds, int numberOfOutput=INT_MAX){
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

        // Remove all entries where value is None
        for (auto it = inputs.begin(); it != inputs.end(); ) {
            if (it->second.type() == typeid(PyObject*)) {
                PyObject* val = std::any_cast<PyObject*>(it->second);
                if (val == Py_None) {
                    it = inputs.erase(it);
                    continue;
                }
            }
            ++it;
        }
        // fprintf(stderr, "%d\n", inputs.count("-dt"));
        // if(inputs.count("-dt")==1){
        //  fprintf(stderr, "%s\n", inputs.find("-dt")->second.type().name());
        //  PyObject* dt=std::any_cast<PyObject*>(inputs.find("-dt")->second);
        //  if(!PyArray_Check(dt)){
        //      PyObject* newDt=PyArray_FromAny(dt,NULL, 1,1, 0, NULL);
        //      inputs.find("-dt")->second=newDt;
        //      PyObject* objectsRepresentation = PyObject_Repr(newDt);
        //      fprintf(stderr, "%s\n", PyUnicode_AsUTF8(objectsRepresentation));
        //  }
        // }else{
        //  fprintf(stderr, "%s\n", inputs.find("-dt")->second.type().name());
        //  PyObject* dt=std::any_cast<PyObject*>(inputs.find("-dt")->second);
            
        //  PyObject* objectsRepresentation = PyObject_Repr(dt);
        //  fprintf(stderr, "%s\n", PyUnicode_AsUTF8(objectsRepresentation));

        //  // if(!PyArray_Check(dt)){
        //  //  PyObject* newDt=PyArray_FromAny(&dt,NULL, 1,1, 0, NULL);
                
        //  //  PyObject* objectsRepresentation = PyObject_Repr(newDt);
        //  //  fprintf(stderr, "%s\n", PyUnicode_AsUTF8(&dt));

        //  //  // fprintf(stderr, "%d\n",PyArray_NDIM(newDt) );
        //  //  // for (int i = 0; i < PyArray_NDIM(newDt); ++i)
        //  //  // {
        //  //  //  fprintf(stderr, "%d\n", PyArray_DIMS(newDt)[i]);
        //  //  // }
        //  //  inputs.find("-dt")->second=newDt;
        //  // }
        // }


        try{
            runStandardCommunication(inputs, outputs, numberOfOutput);
        }catch(const std::exception& e){
            return nullptr;
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

#endif
