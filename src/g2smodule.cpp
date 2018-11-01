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
#define NOMINMAX
#define _USE_MATH_DEFINES
#include <cmath>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <zmq.hpp>
#include <vector>
#include "protocol.hpp"
#include "picosha2.h"
#include <json/json.h>
#include <numeric>
#include <chrono>
#include <thread>
#include <atomic>
#include <future>
#include <algorithm>
#include "DataImage.hpp"

#ifdef WITH_WEB_SUPPORT
#include "cvtZMQ2WS.hpp"
#endif 

typedef unsigned jobIdType;

static char module_docstring[] =
"This module provides an interface for computing geostatistical simulation remotly using G2S";
static char run_docstring[] =
"Executing a simulation";
static PyObject *g2s_run(PyObject *self, PyObject *args);
static PyMethodDef module_methods[] = {
	{"run", g2s_run, METH_VARARGS, run_docstring},
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




inline PyObject* convert2NDArray(g2s::DataImage &image){

	npy_intp* dimsArray=new npy_intp[image._dims.size()+1];
	for (int i = 0; i < image._dims.size(); ++i)
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
	unsigned nbOfVariable=image._nbVariable;
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


inline void sendKill(zmq::socket_t &socket, jobIdType id){
	infoContainer task;
	task.version=1;
	task.task=KILL;
		
	zmq::message_t request (sizeof(infoContainer)+sizeof(jobIdType));
	memcpy(request.data (), &task, sizeof(infoContainer));
	memcpy((char*)request.data()+sizeof(infoContainer),&id,sizeof(jobIdType));
	socket.send (request);
	PyErr_Format(PyExc_ValueError,
				"%s : %s", "gss:error", "Ctrl C, user interupted");
}


inline std::string uploadData(zmq::socket_t &socket, PyObject* prh, PyObject* variableTypeArray=nullptr){

	bool withTimeout=false;
	char sourceName[65]={0};
	int dataSize=PyArray_SIZE(prh);
	int nbOfVariable=1;
	if(variableTypeArray)nbOfVariable=PyArray_SIZE(variableTypeArray);
	int dimData = PyArray_NDIM(prh)-(nbOfVariable>1);
	const npy_intp * dim_array = PyArray_DIMS(prh);
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
		for (int i = 0; i < dataSize; ++i)
		{
			data[i]=matrixData[i];
		}
	}
	if(PyArray_TYPE(prh)==NPY_FLOAT){
		float *matrixData=(float *)PyArray_DATA(prh);
		for (int i = 0; i < dataSize; ++i)
		{
			data[i]=matrixData[i];
		}
	}
	if(PyArray_TYPE(prh)==NPY_UINT8){
		uint8_t *matrixData=(uint8_t *)PyArray_DATA(prh);
		for (int i = 0; i < dataSize; ++i)
		{
			data[i]=matrixData[i];
		}
	}
	if(PyArray_TYPE(prh)==NPY_UINT16){
		uint16_t *matrixData=(uint16_t *)PyArray_DATA(prh);
		for (int i = 0; i < dataSize; ++i)
		{
			data[i]=matrixData[i];
		}
	}
	if(PyArray_TYPE(prh)==NPY_UINT32){
		uint32_t *matrixData=(uint32_t *)PyArray_DATA(prh);
		for (int i = 0; i < dataSize; ++i)
		{
			data[i]=matrixData[i];
		}
	}
	if(PyArray_TYPE(prh)==NPY_UINT64){
		uint64_t *matrixData=(uint64_t *)PyArray_DATA(prh);
		for (int i = 0; i < dataSize; ++i)
		{
			data[i]=matrixData[i];
		}
	}
	if(PyArray_TYPE(prh)==NPY_INT8){
		int8_t *matrixData=(int8_t *)PyArray_DATA(prh);
		for (int i = 0; i < dataSize; ++i)
		{
			data[i]=matrixData[i];
		}
	}
	if(PyArray_TYPE(prh)==NPY_INT16){
		int16_t *matrixData=(int16_t *)PyArray_DATA(prh);
		for (int i = 0; i < dataSize; ++i)
		{
			data[i]=matrixData[i];
		}
	}
	if(PyArray_TYPE(prh)==NPY_INT32){
		int32_t *matrixData=(int32_t *)PyArray_DATA(prh);
		for (int i = 0; i < dataSize; ++i)
		{
			data[i]=matrixData[i];
		}
	}
	if(PyArray_TYPE(prh)==NPY_INT64){
		int64_t *matrixData=(int64_t *)PyArray_DATA(prh);
		for (int i = 0; i < dataSize; ++i)
		{
			data[i]=matrixData[i];
		}
	}

	if(PyArray_TYPE(prh)==NPY_BOOL){
		bool *matrixData=(bool *)PyArray_DATA(prh);
		for (int i = 0; i < dataSize; ++i)
		{
			data[i]=matrixData[i];
		}
	}
	//compute hash

	char* rawData=image.serialize();
	size_t fullsize=*((size_t*)rawData);
	std::vector<unsigned char> hash(32);
	picosha2::hash256((unsigned char*)rawData, ((unsigned char*)rawData)+fullsize-1, hash.begin(), hash.end());

	//check existance
	infoContainer task;
	task.version=1;
	task.task=EXIST;

	zmq::message_t request( sizeof(infoContainer) + 64 * sizeof(unsigned char));
	char * positionInTheStream=(char*)request.data();

	char hashInHexa[65]={0};
	for (int i = 0; i < 32; ++i)
	{
		sprintf(hashInHexa+2*i,"%02x",hash.data()[i]);
	}
	memcpy(sourceName,hashInHexa,65*sizeof(char));

	memcpy (positionInTheStream, &task, sizeof(infoContainer));
	positionInTheStream+=sizeof(infoContainer);
	memcpy (positionInTheStream, hashInHexa, 64 * sizeof(unsigned char));
	positionInTheStream+=64 * sizeof(unsigned char);

	if(!socket.send (request) && withTimeout){
		PyErr_Format(PyExc_ValueError,
				"%s : %s", "gss:error", "timeout sending data");
	}

	zmq::message_t reply;
	if(!socket.recv (&reply) && withTimeout){
		PyErr_Format(PyExc_ValueError,
				"%s : %s", "gss:error", "timeout receive data");
	}
	if(reply.size()!=sizeof(int))
		PyErr_Format(PyExc_ValueError,
				"%s : %s", "gss:error", "wrong answer if data exist!");
	int isPresent=*((int*)reply.data());

	//upload if needed
	if(!isPresent){
		infoContainer task;
		task.version=1;
		task.task=UPLOAD;

		zmq::message_t request( sizeof(infoContainer) + 64 * sizeof(unsigned char) + fullsize);
		char * positionInTheStream=(char*)request.data();

		memcpy (positionInTheStream, &task, sizeof(infoContainer));
		positionInTheStream+=sizeof(infoContainer);
		memcpy (positionInTheStream, hashInHexa, 64 * sizeof(unsigned char));
		positionInTheStream+=64 * sizeof(unsigned char);
		memcpy (positionInTheStream, rawData, fullsize);
		positionInTheStream+=fullsize;
		

		if(!socket.send (request) && withTimeout){
			PyErr_Format(PyExc_ValueError,
				"%s : %s", "gss:error", "timeout sending data");
		}

		zmq::message_t reply;
		if(!socket.recv (&reply) && withTimeout){
			PyErr_Format(PyExc_ValueError,
				"%s : %s", "gss:error", "timeout receive data");
		}
		if(reply.size()!=sizeof(int)) printf( "%s\n", "wrong answer in upload!");
		if(*((int*)reply.data())!=0){
			if(*((int*)reply.data())==1) {
				printf( "%s\n", "data present before upload!");
			}
			else {
				printf( "%s code : %d\n", "error in upload data!",*((int*)reply.data()));
			}
		}
	}
	//clean
	free(rawData);

	return std::string(sourceName);
}

inline std::vector<std::vector<std::string> > lookForUpload(zmq::socket_t &socket, PyObject* args){
	std::vector<std::vector<std::string> > result;
	Py_ssize_t nrhs = PyTuple_Size(args);
	int dataTypeIndex=-1;
	for (int i = 0; i < nrhs; ++i)
	{
		if(PyUnicode_Check(PyTuple_GetItem(args, i)) && (0==strcmp(PyUnicode_AsUTF8(PyTuple_GetItem(args, i)),"-dt")))dataTypeIndex=i+1;
	}

	for (int i = 0; i < nrhs; ++i)
	{
		if(PyUnicode_Check(PyTuple_GetItem(args,i))&& 0==strcmp(PyUnicode_AsUTF8(PyTuple_GetItem(args, i)),"-ti")){
			std::vector<std::string> localVector;
			localVector.push_back("-ti");
			for (int j=i+1; j < nrhs; ++j)
			{
				if(PyUnicode_Check(PyTuple_GetItem(args,j)) && PyUnicode_AsUTF8(PyTuple_GetItem(args, j))[0]=='-'){
					break;
				}
				if(PyUnicode_Check(PyTuple_GetItem(args,j))){
					localVector.push_back(std::string(PyUnicode_AsUTF8(PyTuple_GetItem(args, j))));
				}else{
					if(dataTypeIndex<0)PyErr_Format(PyExc_ValueError,
						 "%s : %s", "gss:error", "-dt wasn't specified");
					localVector.push_back(uploadData(socket, PyTuple_GetItem(args, j) ,PyTuple_GetItem(args,dataTypeIndex)));
				}
			}
			result.push_back(localVector);
		}
		if(PyUnicode_Check(PyTuple_GetItem(args,i))&& 0==strcmp(PyUnicode_AsUTF8(PyTuple_GetItem(args, i)),"-di")){
			std::vector<std::string> localVector;
			localVector.push_back("-di");
			for (int j=i+1; j < nrhs; ++j)
			{
				if(PyUnicode_Check(PyTuple_GetItem(args,j)) && PyUnicode_AsUTF8(PyTuple_GetItem(args, j))[0]=='-'){
					break;
				}
				if(PyUnicode_Check(PyTuple_GetItem(args,j))){
					localVector.push_back(std::string(PyUnicode_AsUTF8(PyTuple_GetItem(args, j))));
				}else{
					if(dataTypeIndex<0)PyErr_Format(PyExc_ValueError,
						 "%s : %s", "gss:error", "-dt wasn't specified");
					localVector.push_back(uploadData(socket, PyTuple_GetItem(args, j) ,PyTuple_GetItem(args,dataTypeIndex)));
				}
			}
			result.push_back(localVector);
		}
		if(PyUnicode_Check(PyTuple_GetItem(args,i))&& 0==strcmp(PyUnicode_AsUTF8(PyTuple_GetItem(args, i)),"-ki")){
			std::vector<std::string> localVector;
			localVector.push_back("-ki");
			for (int j=i+1; j < nrhs; ++j)
			{
				if(PyUnicode_Check(PyTuple_GetItem(args,j)) && PyUnicode_AsUTF8(PyTuple_GetItem(args, j))[0]=='-'){
					break;
				}
				if(PyUnicode_Check(PyTuple_GetItem(args,j))){
					localVector.push_back(std::string(PyUnicode_AsUTF8(PyTuple_GetItem(args, j))));
				}else{
					/*PyObject* variable=PyArray_SimpleNew(PyArray_NDIM(PyTuple_GetItem(args, dataTypeIndex)),
														 PyArray_DIMS(PyTuple_GetItem(args, dataTypeIndex)),NPY_FLOAT);
					for (int i = 0; i < PyArray_SIZE(variable); ++i)
					{
						((float*)PyArray_DATA(variable))[i]=0.f;
					}*/
					localVector.push_back(uploadData(socket, PyTuple_GetItem(args, j)));
				}
			}
			result.push_back(localVector);
		}
		if(PyUnicode_Check(PyTuple_GetItem(args,i))&& 0==strcmp(PyUnicode_AsUTF8(PyTuple_GetItem(args, i)),"-sp")){
			std::vector<std::string> localVector;
			localVector.push_back("-sp");
			for (int j=i+1; j < nrhs; ++j)
			{
				if(PyUnicode_Check(PyTuple_GetItem(args,j)) && PyUnicode_AsUTF8(PyTuple_GetItem(args, j))[0]=='-'){
					break;
				}
				if(PyUnicode_Check(PyTuple_GetItem(args,j))){
					localVector.push_back(std::string(PyUnicode_AsUTF8(PyTuple_GetItem(args, j))));
				}else{
					localVector.push_back(uploadData(socket, PyTuple_GetItem(args, j)));
				}
			}
			result.push_back(localVector);
		}
	}

	return result;
}
#ifdef WITH_WEB_SUPPORT
void cvtServerCall(std::string from, std::string to, std::atomic<bool> &serverRun, std::atomic<bool> &done){
	cvtServer((char*)from.c_str(),(char*)to.c_str(),serverRun, done);
}

#endif 


void pyFunctionWork(PyObject *self, PyObject *args, std::atomic<bool> &done, std::vector<PyObject*> &plhs){

	jobIdType id=0;
	bool stop=false;
	bool withTimeout=true;
	bool noOutput=false;

	std::vector<std::string> inputArray;
	std::vector<int> inputArrayIndex;

	Py_ssize_t nrhs = PyTuple_Size(args);

	for (int i = 0; i < nrhs; ++i)
	{
		if(PyUnicode_Check(PyTuple_GetItem(args,i))){
			char charTemp[2048];
			strcpy(charTemp,PyUnicode_AsUTF8(PyTuple_GetItem(args,i)));
			inputArray.push_back(std::string(charTemp));
			inputArrayIndex.push_back(i);
		}

		if(PyLong_Check(PyTuple_GetItem(args,i))){
			inputArray.push_back(std::to_string(float(PyLong_AsLong(PyTuple_GetItem(args,i)))));
			inputArrayIndex.push_back(i);
		}
		if(PyFloat_Check(PyTuple_GetItem(args,i))){
			inputArray.push_back(std::to_string(float(PyFloat_AsDouble(PyTuple_GetItem(args,i)))));
			inputArrayIndex.push_back(i);
		}
	}

	int saP1_Index=-1;
	int dimP1_Index=-1;
	int aP1_Index=-1;
	int pP1_Index=-1;
	int id_index=-1;

	bool submit=true;
	bool statusOnly=false;
	bool waitAndDownload=true;
	bool kill=false;
	bool serverShutdown=false;
	bool silentMode=false;

	for (int i = 0; i < inputArray.size(); ++i)
	{
		if(!inputArray[i].compare("-sa")){
			saP1_Index=inputArrayIndex[i]+1;
		}
		if(!inputArray[i].compare("-a")){
			aP1_Index=inputArrayIndex[i]+1;
		}
		if(!inputArray[i].compare("-p")){
			pP1_Index=inputArrayIndex[i]+1;
		}
		if(!inputArray[i].compare("-dim")){
			dimP1_Index=inputArrayIndex[i]+1;
		}
		if(!inputArray[i].compare("-noTO")){
			withTimeout=false;
		}
		if(!inputArray[i].compare("-submitOnly")){
			waitAndDownload=false;
			/*if(nlhs>1){
				mexErrMsgIdAndTxt("gss:error", "require maximum one output");
				stop=true;
			}*/
		}
		if(!inputArray[i].compare("-statusOnly")){
			statusOnly=true;
			waitAndDownload=false;
			submit=false;
			id_index=inputArrayIndex[i]+1;
			/*if(nlhs>1){
				mexErrMsgIdAndTxt("gss:error", "require maximum one output");
				stop=true;
			}*/
		}
		if(!inputArray[i].compare("-waitAndDownload")){
			submit=false;
			id_index=inputArrayIndex[i]+1;
		}
		if(!inputArray[i].compare("-kill")){
			kill=true;
			id_index=inputArrayIndex[i]+1;
		}
		if(!inputArray[i].compare("-shutdown")){
			serverShutdown=true;
			waitAndDownload=false;
			noOutput=true;
		}
		if(!inputArray[i].compare("-silent")){
			silentMode=true;
		}
	}

	std::atomic<bool> serverRun;
	serverRun=true;

	if((saP1_Index!=-1)){
		std::string adress=std::string(PyUnicode_AsUTF8(PyTuple_GetItem(args,saP1_Index)));
		std::transform(adress.begin(), adress.end(), adress.begin(),::tolower);
		#ifdef WITH_WEB_SUPPORT
		if(!adress.compare("web")||!adress.compare("browser")){
			//printf("use browser server\n");
			serverRun=false;
			std::string from="tcp://*:8128";
			std::string to="ws://localhsot:8129";

			std::thread serverThread(cvtServerCall,from,to,std::ref(serverRun),std::ref(done));
			serverThread.detach();
			saP1_Index=-1;
			while (!serverRun){
				std::this_thread::sleep_for(std::chrono::milliseconds(300));
			}

			printf("server run now\n");
		}
		#endif

	}


	zmq::context_t context (1);
	zmq::socket_t socket (context, ZMQ_REQ);
	int timeout=30000;
	socket.setsockopt(ZMQ_LINGER, timeout);
	if(withTimeout){
		socket.setsockopt(ZMQ_RCVTIMEO, timeout);
		socket.setsockopt(ZMQ_SNDTIMEO, timeout);
	}

	short port=8128;
	std::string serverAddress=std::string("localhost");
	if(pP1_Index!=-1)
		port=PyLong_Check(PyTuple_GetItem(args,pP1_Index));
	if(saP1_Index!=-1)
		serverAddress=PyUnicode_AsUTF8(PyTuple_GetItem(args,saP1_Index));

	char address[4096];
	sprintf(address,"tcp://%s:%d",serverAddress.c_str(),port);
	socket.connect (address);

	if(serverShutdown){
		infoContainer task;
		task.version=1;
		task.task=SHUTDOWN;

		zmq::message_t request (sizeof(infoContainer));
		memcpy(request.data (), &task, sizeof(infoContainer));
		if(!socket.send (request) && withTimeout ){
			PyErr_Format(PyExc_ValueError,
				"%s : %s", "gss:error", "fail to shutdown the server");
		}
		stop=true;
		done=true;
	}

	std::vector<std::vector<std::string> > dataString=lookForUpload(socket, args);

	Py_BEGIN_ALLOW_THREADS

	if(done) {
		stop=true;
	}

	if(!stop && submit){
		infoContainer task;
		task.version=1;
		char algo[2048];
		strcpy(algo,PyUnicode_AsUTF8(PyTuple_GetItem(args,aP1_Index)));

		Json::Value object(Json::objectValue);
		object["Algorithm"]=algo;
		{
			object["Priority"]="1";
			
			Json::Value parameter(Json::objectValue);
			
			Json::Value jsonDi(Json::arrayValue);
			Json::Value jsonKi(Json::arrayValue);
			Json::Value jsonSP(Json::arrayValue);

			for (int i = 0; i < dataString.size(); ++i)
			{
				Json::Value jsonArray(Json::arrayValue);
				for (int j = 1; j <  dataString[i].size(); ++j)
				{
					jsonArray.append(dataString[i][j]);
				}
				parameter[dataString[i][0]]=jsonArray;
			}
			
			for (int i = 0; i < inputArray.size(); ++i)
			{
				bool managed=false;
				if(inputArray[i].at(0)!='-') continue;
				if(!inputArray[i].compare("-sa") ||
				   !inputArray[i].compare("-ti") ||
				   !inputArray[i].compare("-di") ||
				   !inputArray[i].compare("-sp") ||
				   !inputArray[i].compare("-ki") ||
				   !inputArray[i].compare("-a")){
					managed=true;
				}

				if (!managed)
				{
					if((inputArray.size()>i+1) && (inputArray[i+1].at(0)!='-') ){
						parameter[inputArray[i]]=inputArray[i+1];
					}else{
						parameter[inputArray[i]]="";//Json::Value::null;
					}
					managed=true;
				}
			}

			object["Parameter"]=parameter;

		}

		Json::FastWriter fast;
		std::string jsonJob = fast.write(object);
		const char* jsonJob_c=jsonJob.c_str();

		//printf( "%s\n", jsonJob_c);

		task.task=JOB;

		zmq::message_t request (sizeof(infoContainer)+strlen(jsonJob_c));
		memcpy(request.data (), &task, sizeof(infoContainer));
		memcpy((char*)request.data()+sizeof(infoContainer),jsonJob_c,strlen(jsonJob_c));
		if(!socket.send (request) && withTimeout ){
			PyErr_Format(PyExc_ValueError,
				"%s : %s", "gss:error", "timeout sending job");
		}

		zmq::message_t reply;
		if(!socket.recv (&reply) && withTimeout){
			PyErr_Format(PyExc_ValueError,
				"%s : %s", "gss:error", "timeout starting job, maybe job run on server !!");
		}
		if((reply.size()!=sizeof(jobIdType))&& !silentMode){
			Py_BLOCK_THREADS
			printf( "%s\n", "wrong answer !");
			Py_UNBLOCK_THREADS
		}
		id=*((jobIdType*)reply.data());
		if(id<0 && !silentMode)
		{
			Py_BLOCK_THREADS
			printf( "%s\n", "error in job distribution!");
			Py_UNBLOCK_THREADS
		}
		if(!silentMode){
			Py_BLOCK_THREADS
			printf("job Id is: %u\n",id );
			Py_UNBLOCK_THREADS
		}

		if(done) {
			sendKill(socket, id);
			stop=true;
		}

	}else if (!submit) {
		id=PyLong_AsUnsignedLong(PyTuple_GetItem(args,id_index));
	}


	if(submit && !waitAndDownload){
		
		{
			size_t one=1;
			plhs.push_back(PyLong_FromUnsignedLong(id));
		}
		noOutput=true;
		stop=true;
	}

	float lastProgression=-1.f;

	char nameFile[65]={0};
	sprintf(nameFile,"%u",id);
	setbuf(stdout, NULL);

	if(kill) done=true;
	
	if(!stop && !done && !silentMode) {
		Py_BLOCK_THREADS
		printf("progres %.3f%%",0/1000. );
		Py_UNBLOCK_THREADS
	}
	while(!stop) {
		std::this_thread::sleep_for(std::chrono::milliseconds(600));
		if(done) {
			sendKill(socket, id);
			stop=true;
			continue;
		}
		// status
		{
			infoContainer task;
			task.version=1;
			task.task=STATUS;
			
			zmq::message_t request (sizeof(infoContainer)+sizeof( jobIdType ));
			memcpy(request.data (), &task, sizeof(infoContainer));
			memcpy((char*)request.data()+sizeof(infoContainer),&id,sizeof( jobIdType ));
			if(!socket.send (request)){
				continue;
			}
			if(done) {
				sendKill(socket, id);
				stop=true;
				continue;
			}
			zmq::message_t reply;
			if(!socket.recv (&reply) && withTimeout){
				continue;
			}
			if(done) {
				sendKill(socket, id);
				stop=true;
				continue;
			}
			if(reply.size()!=sizeof(int) && !silentMode)
				{
					Py_BLOCK_THREADS
					printf( "%s\n", "wrong answer !");
					Py_UNBLOCK_THREADS
				}
			else{
				int progress=*((int*)reply.data());
				if(!silentMode && (progress>=0) && fabs(lastProgression-progress/1000.)>0.0001){
					Py_BLOCK_THREADS
					printf("\rprogres %.3f%%",progress/1000. );
					Py_UNBLOCK_THREADS
					lastProgression=progress/1000.;
				}
				if(!waitAndDownload)
				{
					size_t one=1;
					plhs.push_back(PyFloat_FromDouble(progress/1000.));
				}
			}   
		}

		if(!waitAndDownload){
			stop=true;
			noOutput=true;
			continue;
		}

		if(done) {
			sendKill(socket, id);
			stop=true;
			continue;
		}

		{
			infoContainer task;
			task.version=1;
			task.task=EXIST;

			zmq::message_t request( sizeof(infoContainer) + 64 * sizeof(unsigned char));
			char * positionInTheStream=(char*)request.data();

			memcpy (positionInTheStream, &task, sizeof(infoContainer));
			positionInTheStream+=sizeof(infoContainer);
			memcpy (positionInTheStream, nameFile, 64 * sizeof(unsigned char));
			positionInTheStream+=64 * sizeof(unsigned char);

			if(!socket.send (request) && withTimeout){
				PyErr_Format(PyExc_ValueError,
				"%s : %s", "gss:error", "timeout sending data");
			}

			zmq::message_t reply;
			if(!socket.recv (&reply) && withTimeout){
				PyErr_Format(PyExc_ValueError,
				"%s : %s", "gss:error", "timeout receive data");
			}
			if(reply.size()!=sizeof(int))
				PyErr_Format(PyExc_ValueError,
				"%s : %s", "gss:error", "wrong answer if data exist!");
			int isPresent=*((int*)reply.data());
			if (isPresent) break;
		}

	}
	
	
	if(!noOutput){
		// download data
		infoContainer task;
		task.version=1;
		task.task=DOWNLOAD;

		//printf("%s\n", nameFile);
		zmq::message_t request (sizeof(infoContainer)+64);
		memcpy (request.data(), &task, sizeof(infoContainer));
		memcpy ((char*)request.data()+sizeof(infoContainer), nameFile, 64);
		//std::cout << "ask for data" << std::endl;
		if(!socket.send (request) && withTimeout ){
			PyErr_Format(PyExc_ValueError,
				"%s : %s", "gss:error", "timeout asking for data");
		}
		zmq::message_t reply;
		if(!socket.recv (&reply) && withTimeout){
			PyErr_Format(PyExc_ValueError,
				"%s : %s", "gss:error", "timeout : get data dont answer");
		}
		if(reply.size()!=0){
			if(!silentMode){
				Py_BLOCK_THREADS
				printf("\rprogres %.3f%%\n",100.0f );
				Py_UNBLOCK_THREADS
			}
			g2s::DataImage image((char*)reply.data());
			Py_BLOCK_THREADS
			plhs.push_back(convert2NDArray(image));
			Py_UNBLOCK_THREADS
			stop=true;
		}
	}

	char nameFile_local[65]={0};
	std::vector<std::string> prefix;
	std::vector<float> isInteger;
	prefix.push_back("id_");
	isInteger.push_back(true);
	prefix.push_back("nw_");
	isInteger.push_back(false);
	prefix.push_back("path_");
	isInteger.push_back(true);


	int downloadInformation=1;
	if(!noOutput){
		int positionInOutput=0;
		while (positionInOutput<prefix.size()+1){
			// download data
			sprintf(nameFile_local,"%s%u",prefix[positionInOutput].c_str(),id);

			{
				infoContainer task;
				task.version=1;
				task.task=EXIST;

				zmq::message_t request( sizeof(infoContainer) + 64 * sizeof(unsigned char));
				char * positionInTheStream=(char*)request.data();

				memcpy (positionInTheStream, &task, sizeof(infoContainer));
				positionInTheStream+=sizeof(infoContainer);
				memcpy (positionInTheStream, nameFile_local, 64 * sizeof(unsigned char));
				positionInTheStream+=64 * sizeof(unsigned char);

				if(!socket.send (request) && withTimeout){
					PyErr_Format(PyExc_ValueError,
				"%s : %s", "gss:error", "timeout sending data");
				}

				zmq::message_t reply;
				if(!socket.recv (&reply) && withTimeout){
					PyErr_Format(PyExc_ValueError,
				"%s : %s", "gss:error", "gss:error", "timeout receive data");
				}
				if(reply.size()!=sizeof(int))
					PyErr_Format(PyExc_ValueError,
				"%s : %s", "gss:error", "wrong answer if data exist!");
				int isPresent=*((int*)reply.data());
				if (!isPresent) break;
			}

			infoContainer task;
			task.version=1;
			task.task=DOWNLOAD;

			//printf("%s\n", nameFile);
			zmq::message_t request (sizeof(infoContainer)+64);
			memcpy (request.data(), &task, sizeof(infoContainer));
			memcpy ((char*)request.data()+sizeof(infoContainer), nameFile_local, 64);
			//std::cout << "ask for data" << std::endl;
			if(!socket.send (request) && withTimeout ){
				PyErr_Format(PyExc_ValueError,
				"%s : %s", "gss:error", "timeout asking for data");
			}
			zmq::message_t reply;
			if(!socket.recv (&reply) && withTimeout){
				PyErr_Format(PyExc_ValueError,
				"%s : %s", "gss:error", "timeout : get data dont answer");
			}
			
			if(reply.size()!=0){
				g2s::DataImage image((char*)reply.data());
				Py_BLOCK_THREADS
				plhs.push_back(convert2NDArray(image));
				Py_UNBLOCK_THREADS
				stop=true;
				downloadInformation++;
			}
			positionInOutput++;
		}
	}
	if(!noOutput){
		// download data

		{
			infoContainer task;
			task.version=1;
			task.task=DURATION;
			
			zmq::message_t request (sizeof(infoContainer)+sizeof( jobIdType ));
			memcpy(request.data (), &task, sizeof(infoContainer));
			memcpy((char*)request.data()+sizeof(infoContainer),&id,sizeof( jobIdType ));
			if(!socket.send (request)){
				PyErr_Format(PyExc_ValueError,
				"%s : %s", "gss:error", "timeout asking for data");
			}
			zmq::message_t reply;
			if(!socket.recv (&reply) && withTimeout){
				PyErr_Format(PyExc_ValueError,
				"%s : %s", "gss:error", "timeout : get data dont answer");
			}
			if(reply.size()!=sizeof(int)) printf( "%s\n", "wrong answer !");
			else{
				int duration=*((int*)reply.data());
				size_t one=1;
				Py_BLOCK_THREADS
				plhs.push_back(PyFloat_FromDouble(duration/(1000.f)));
				Py_UNBLOCK_THREADS
			}   
		}
	}
	done=true;
	Py_END_ALLOW_THREADS
};

void testIfInterupted(std::atomic<bool> &done){
	while (!done){
		std::this_thread::sleep_for(std::chrono::milliseconds(300));
		if(PyErr_CheckSignals()){
			done=true;
		}
	}
}

static PyObject *g2s_run(PyObject *self, PyObject *args)
{
	
	std::atomic<bool> done(false);
	std::vector<PyObject*> result;
	auto myFuture = std::async(std::launch::async, pyFunctionWork,self, args,std::ref(done),std::ref(result));
	testIfInterupted(std::ref(done));
	myFuture.wait();

	PyObject* pyResult;
	if(result.size()==1){
		pyResult=result[0];
	}else{
		pyResult=PyList_New(0);
		for (int i = 0; i < result.size(); ++i)
		{
			PyList_Append(pyResult,result[i]);
		}
	}

	return pyResult;
}