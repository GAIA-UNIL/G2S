#include "python3_interface.hpp"
#include <frameobject.h>

typedef unsigned jobIdType;

static char module_docstring[] =
"This module provides an interface for computing geostatistical simulation remotly using G2S";
static char run_docstring[] =
"Executing a simulation";
static PyObject *g2s_run(PyObject *self, PyObject *args, PyObject *keywds);
static PyMethodDef module_methods[] = {
	{"run", (PyCFunction)g2s_run, METH_VARARGS|METH_KEYWORDS, run_docstring},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC PyInit_g2s(void)
{
	Py_Initialize();
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

int pythonNumberOfOutputParameterDecoder(const char* s){
	switch(s[0]){
		case 1: //any output
		return 0;
		case 90: // single output
		return -1;
		case 92: // tuple output
		return int(s[1]);
		case 4: // many output
		return std::max(pythonNumberOfOutputParameterDecoder(s+2),pythonNumberOfOutputParameterDecoder(s+4));
		case 94: //output with star : a,b,*c
		return INT_MAX;
		default:
		return 0;
	}
}

int pythonNumberOfOutputParameter(){
	PyFrameObject* frame = PyEval_GetFrame();
	const char* s  = PyBytes_AS_STRING(frame->f_code->co_code);
	
	// for (int i = 0; i < 50; ++i)
	// {
	// 	printf("%d ==> %d\n",i,s[i]);
	// }


	int start = frame->f_lasti;
	//const unsigned char* s = (unsigned char*)PyUnicode_DATA(filename);
	start+=2;
	// printf("start %d, end %d\n", start,frame->f_lasti);
	int expected=pythonNumberOfOutputParameterDecoder(s+start);
	return (expected>0 ? expected : INT_MAX);
}


static PyObject *g2s_run(PyObject *self, PyObject *args, PyObject *keywds)
{
	InerfaceTemplatePython3 inerfaceTemplatePython3;
	return inerfaceTemplatePython3.runStandardCommunicationPython(self, args, keywds, pythonNumberOfOutputParameter() );

}