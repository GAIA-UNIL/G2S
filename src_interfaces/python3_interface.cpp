#include "python3_interface.hpp"
#include <frameobject.h>
// C++20 is enabled via build flags; no changes to the interface template header.

typedef unsigned jobIdType;

static char module_docstring[] =
"This module provides an interface for computing geostatistical simulation remotly using G2S";
static char run_docstring[] = "Executing a simulation";
static char load_docstring[] = "load a dataImage from G2S";
static char write_docstring[] = "write a dataImage from G2S";
static PyObject *g2s_run(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *g2s_loadData(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *g2s_writeData(PyObject *self, PyObject *args, PyObject *keywds);

static PyMethodDef module_methods[] = {
    {"run", (PyCFunction)g2s_run, METH_VARARGS|METH_KEYWORDS, run_docstring},
    {"loadData", (PyCFunction)g2s_loadData, METH_VARARGS|METH_KEYWORDS, load_docstring},
    {"writeData", (PyCFunction)g2s_writeData, METH_VARARGS|METH_KEYWORDS, write_docstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC PyInit_g2s(void)
{
#ifndef PYPY_VERSION
    Py_Initialize();
#endif
    PyObject *module;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "g2s",
        module_docstring,
        -1,
        module_methods,
        NULL,
        NULL,
        NULL,
        NULL
    };
    module = PyModule_Create(&moduledef);
    if (!module) return NULL;
    /* Load `numpy` functionality. */
    import_array();
    return module;
}

// --- unchanged logic below; kept for PyPy best-effort and CPython 3.12/3.13 ---
// (We rely on your existing bytecode inspection with version guards.)

int pythonNumberOfOutputParameterDecoder(const char* s){
    switch(s[0]){
        case 1: return 0;             // any output
        case 90: return -1;           // single output
        case 92: return int(s[1]);    // tuple output
        case 4:  return std::max(pythonNumberOfOutputParameterDecoder(s+2),
                                 pythonNumberOfOutputParameterDecoder(s+4)); // many output
        case 94: return INT_MAX;      // output with star: a,b,*c
        default: return 0;
    }
}

#ifdef PYPY_VERSION
int pythonNumberOfOutputParameter(){
    return INT_MAX;
}
#else
int pythonNumberOfOutputParameter(){
    PyFrameObject* frame = PyEval_GetFrame();
#if PY_VERSION_HEX < 0x030B0000
    const char* s  = PyBytes_AS_STRING(frame->f_code->co_code);
    int start = frame->f_lasti;
#else
    const char* s  = PyBytes_AS_STRING(PyCode_GetCode(PyFrame_GetCode(frame)));
    int start = PyFrame_GetLasti(frame);
#endif
    start+=2;
    int expected=pythonNumberOfOutputParameterDecoder(s+start);
    return (expected>0 ? expected : INT_MAX);
}
#endif

static PyObject *g2s_run(PyObject *self, PyObject *args, PyObject *keywds)
{
    InterfaceTemplatePython3 interfaceTemplatePython3;
    return interfaceTemplatePython3.runStandardCommunicationPython(self, args, keywds, pythonNumberOfOutputParameter() );
}

static PyObject *g2s_loadData(PyObject *self, PyObject *args, PyObject *keywds)
{
#ifndef _WIN32
    static char *kwlist[] = {(char*)"filename", NULL};
    PyObject* filename;
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "U", kwlist, &filename)) {
        return NULL;
    }
    InterfaceTemplatePython3 interfaceTemplatePython3;
    std::string filenameStr=interfaceTemplatePython3.nativeToStandardString(std::any(filename));
    g2s::DataImage loadedImage=g2s::DataImage::createFromFile(filenameStr);
    PyObject* imageNumpy=std::any_cast<PyObject*>(interfaceTemplatePython3.convert2NativeMatrix(loadedImage));
    long size=(loadedImage._types.size());
    PyObject* type=PyArray_SimpleNew(1,&size,NPY_INT32);
    int32_t *typePtr=(int32_t*)PyArray_DATA_SAFE(type);
    for (size_t i = 0; i < loadedImage._types.size(); ++i)
    {
        switch(loadedImage._types[i]){
            case g2s::DataImage::VaraibleType::Continuous:  typePtr[i]=0; break;
            case g2s::DataImage::VaraibleType::Categorical: typePtr[i]=1; break;
        }
    }
    PyObject* pyResult=PyTuple_New(2);
    PyTuple_SetItem(pyResult,0,imageNumpy);
    PyTuple_SetItem(pyResult,1,type);
    return pyResult;
#else
    Py_RETURN_NONE;
#endif
}

static PyObject *g2s_writeData(PyObject *self, PyObject *args, PyObject *keywds)
{
#ifndef _WIN32
    static char *kwlist[] = {(char*)"image",(char*)"dataType",(char*)"filename", NULL};
    PyObject* pyImage;
    PyObject* pyDataType;
    PyObject* filename;
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOU", kwlist, &pyImage, &pyDataType, &filename)) {
        return NULL;
    }
    InterfaceTemplatePython3 interfaceTemplatePython3;
    g2s::DataImage image=interfaceTemplatePython3.convertNativeMatrix2DataImage(std::any(pyImage),std::any(pyDataType));
    std::string filenameStr=interfaceTemplatePython3.nativeToStandardString(std::any(filename));
    image.write(filenameStr);
#endif
    Py_RETURN_NONE;
}
