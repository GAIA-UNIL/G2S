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

#define MATLAB
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <mex.h>
#include <zmq.hpp>
#include <vector>
#include "protocol.hpp"
#include "picosha2.h"
#include <json/json.h>
#include <numeric>
#include <chrono>
#include <atomic>
#include <algorithm>
#include "DataImage.hpp"
#include "mexInterrupt.hpp"

#ifdef WITH_WEB_SUPPORT
#include "cvtZMQ2WS.hpp"
#endif 

#include "matrix.h"

typedef unsigned jobIdType;


inline mxArray* convert2MxArray(g2s::DataImage &image){

	size_t *dimsArray=(size_t *)malloc(sizeof(size_t)*(image._dims.size()+1));
	for (int i = 0; i < image._dims.size(); ++i)
	{
		dimsArray[i]=image._dims[i];
	}
	dimsArray[image._dims.size()]=image._nbVariable;
	mxArray *array=nullptr;
	if(image._encodingType==g2s::DataImage::Float)
		array=mxCreateNumericArray(image._dims.size()+1, dimsArray, mxSINGLE_CLASS , mxREAL);
	if(image._encodingType==g2s::DataImage::Integer)
		array=mxCreateNumericArray(image._dims.size()+1, dimsArray, mxINT32_CLASS , mxREAL);
	if(image._encodingType==g2s::DataImage::UInteger)
		array=mxCreateNumericArray(image._dims.size()+1, dimsArray, mxUINT32_CLASS , mxREAL);

	float* data=(float*)mxGetPr(array);
	unsigned nbOfVariable=image._nbVariable;
	unsigned dataSize=image.dataSize();
	for (int i = 0; i < dataSize/nbOfVariable; ++i)
	{
		for (int j = 0; j < nbOfVariable; ++j)
		{
			data[i+j*(dataSize/nbOfVariable)]=image._data[i*nbOfVariable+j];
		}
	}
	free(dimsArray);
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
	mexErrMsgIdAndTxt("gss:error", "Ctrl C, user interupted");
}

std::string mxGetString(const mxArray *pm){
	char result[2048]={0};
	mxGetString(pm, result, 2048);
	return std::string(result);
}

inline std::string uploadData(zmq::socket_t &socket, const mxArray* prh, const mxArray* variableTypeArray=nullptr){

	bool withTimeout=false;
	char sourceName[65]={0};
	int dataSize=mxGetNumberOfElements(prh);
	int nbOfVariable=1;
	if(variableTypeArray)nbOfVariable=mxGetNumberOfElements(variableTypeArray);
	int dimData = mxGetNumberOfDimensions(prh)-(nbOfVariable>1);
	const size_t * dim_array = mxGetDimensions(prh);
	unsigned *dimArray=(unsigned *)malloc(sizeof(unsigned)*dimData);
	for (int i = 0; i < dimData; ++i)
	{
		dimArray[i]=dim_array[i];
	}

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
		for (int i = 0; i < dataSize/nbOfVariable; ++i)
		{
			for (int j = 0; j < nbOfVariable; ++j)
			{
				data[i*nbOfVariable+j]=matrixData[i+j*(dataSize/nbOfVariable)];
			}
		}
	}
	if(mxIsSingle(prh)){
		float *matrixData=(float *)mxGetPr(prh);
		for (int i = 0; i < dataSize/nbOfVariable; ++i)
		{
			for (int j = 0; j < nbOfVariable; ++j)
			{
				data[i*nbOfVariable+j]=matrixData[i+j*(dataSize/nbOfVariable)];
			}
		}
	}
	if(mxGetClassID(prh)==mxUINT8_CLASS){
		uint8_t *matrixData=(uint8_t *)mxGetPr(prh);
		for (int i = 0; i < dataSize/nbOfVariable; ++i)
		{
			for (int j = 0; j < nbOfVariable; ++j)
			{
				data[i*nbOfVariable+j]=matrixData[i+j*(dataSize/nbOfVariable)];
			}
		}
	}
	if(mxGetClassID(prh)==mxUINT16_CLASS){
		uint16_t *matrixData=(uint16_t *)mxGetPr(prh);
		for (int i = 0; i < dataSize/nbOfVariable; ++i)
		{
			for (int j = 0; j < nbOfVariable; ++j)
			{
				data[i*nbOfVariable+j]=matrixData[i+j*(dataSize/nbOfVariable)];
			}
		}
	}
	if(mxGetClassID(prh)==mxUINT32_CLASS){
		uint32_t *matrixData=(uint32_t *)mxGetPr(prh);
		for (int i = 0; i < dataSize/nbOfVariable; ++i)
		{
			for (int j = 0; j < nbOfVariable; ++j)
			{
				data[i*nbOfVariable+j]=matrixData[i+j*(dataSize/nbOfVariable)];
			}
		}
	}
	if(mxGetClassID(prh)==mxUINT64_CLASS){
		uint64_t *matrixData=(uint64_t *)mxGetPr(prh);
		for (int i = 0; i < dataSize/nbOfVariable; ++i)
		{
			for (int j = 0; j < nbOfVariable; ++j)
			{
				data[i*nbOfVariable+j]=matrixData[i+j*(dataSize/nbOfVariable)];
			}
		}
	}
	if(mxGetClassID(prh)==mxINT8_CLASS){
		int8_t *matrixData=(int8_t *)mxGetPr(prh);
		for (int i = 0; i < dataSize/nbOfVariable; ++i)
		{
			for (int j = 0; j < nbOfVariable; ++j)
			{
				data[i*nbOfVariable+j]=matrixData[i+j*(dataSize/nbOfVariable)];
			}
		}
	}
	if(mxGetClassID(prh)==mxINT16_CLASS){
		int16_t *matrixData=(int16_t *)mxGetPr(prh);
		for (int i = 0; i < dataSize/nbOfVariable; ++i)
		{
			for (int j = 0; j < nbOfVariable; ++j)
			{
				data[i*nbOfVariable+j]=matrixData[i+j*(dataSize/nbOfVariable)];
			}
		}
	}
	if(mxGetClassID(prh)==mxINT32_CLASS){
		int32_t *matrixData=(int32_t *)mxGetPr(prh);
		for (int i = 0; i < dataSize/nbOfVariable; ++i)
		{
			for (int j = 0; j < nbOfVariable; ++j)
			{
				data[i*nbOfVariable+j]=matrixData[i+j*(dataSize/nbOfVariable)];
			}
		}
	}
	if(mxGetClassID(prh)==mxINT64_CLASS){
		int64_t *matrixData=(int64_t *)mxGetPr(prh);
		for (int i = 0; i < dataSize/nbOfVariable; ++i)
		{
			for (int j = 0; j < nbOfVariable; ++j)
			{
				data[i*nbOfVariable+j]=matrixData[i+j*(dataSize/nbOfVariable)];
			}
		}
	}

	if(mxGetClassID(prh)==mxLOGICAL_CLASS){
		bool *matrixData=(bool *)mxGetPr(prh);
		for (int i = 0; i < dataSize/nbOfVariable; ++i)
		{
			for (int j = 0; j < nbOfVariable; ++j)
			{
				data[i*nbOfVariable+j]=matrixData[i+j*(dataSize/nbOfVariable)];
			}
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
		snprintf(hashInHexa+2*i,65-2*i,"%02x",hash.data()[i]);
	}
	memcpy(sourceName,hashInHexa,65*sizeof(char));

	memcpy (positionInTheStream, &task, sizeof(infoContainer));
	positionInTheStream+=sizeof(infoContainer);
	memcpy (positionInTheStream, hashInHexa, 64 * sizeof(unsigned char));
	positionInTheStream+=64 * sizeof(unsigned char);

	if(!socket.send (request) && withTimeout){
		mexErrMsgIdAndTxt("gss:error", "timeout sending data");
	}

	zmq::message_t reply;
	if(!socket.recv (&reply) && withTimeout){
		mexErrMsgIdAndTxt("gss:error", "timeout receive data");
	}
	if(reply.size()!=sizeof(int)) mexErrMsgIdAndTxt("gss:error", "wrong answer if data exist!");
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
			mexErrMsgIdAndTxt("gss:error", "timeout sending data");
		}

		zmq::message_t reply;
		if(!socket.recv (&reply) && withTimeout){
			mexErrMsgIdAndTxt("gss:error", "timeout receive data");
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

inline std::vector<std::vector<std::string> > lookForUpload(zmq::socket_t &socket, int nrhs, const mxArray *prhs[]){
	std::vector<std::vector<std::string> > result;
	int dataTypeIndex=-1;
	for (int i = 0; i < nrhs; ++i)
	{
		if(0==strcmp(mxGetString(prhs[i]).c_str(),"-dt"))dataTypeIndex=i+1;
	}

	for (int i = 0; i < nrhs; ++i)
	{
		if(mxIsChar(prhs[i])&& 0==strcmp(mxGetString(prhs[i]).c_str(),"-ti")){
			std::vector<std::string> localVector;
			localVector.push_back("-ti");
			for (int j=i+1; j < nrhs; ++j)
			{
				if(mxIsChar(prhs[j]) && mxGetString(prhs[j]).c_str()[0]=='-'){
					break;
				}
				if(mxIsChar(prhs[j])){
					localVector.push_back(mxGetString(prhs[j]));
				}else{
					if(dataTypeIndex<0)mexErrMsgIdAndTxt("gss:error", "-dt wasn't specified");
					localVector.push_back(uploadData(socket, prhs[j] ,prhs[dataTypeIndex]));
				}
			}
			result.push_back(localVector);
		}
		if(mxIsChar(prhs[i])&& 0==strcmp(mxGetString(prhs[i]).c_str(),"-di")){
			std::vector<std::string> localVector;
			localVector.push_back("-di");
			for (int j=i+1; j < nrhs; ++j)
			{
				if(mxIsChar(prhs[j]) && mxGetString(prhs[j]).c_str()[0]=='-'){
					break;
				}
				if(mxIsChar(prhs[j])){
					localVector.push_back(mxGetString(prhs[j]));
				}else{
					if(dataTypeIndex<0)mexErrMsgIdAndTxt("gss:error", "-dt wasn't specified");
					localVector.push_back(uploadData(socket, prhs[j], prhs[dataTypeIndex]));
				}
			}
			result.push_back(localVector);
		}
		if(mxIsChar(prhs[i])&& 0==strcmp(mxGetString(prhs[i]).c_str(),"-ki")){
			std::vector<std::string> localVector;
			localVector.push_back("-ki");
			for (int j=i+1; j < nrhs; ++j)
			{
				if(mxIsChar(prhs[j]) && mxGetString(prhs[j]).c_str()[0]=='-'){
					break;
				}
				if(mxIsChar(prhs[j])){
					localVector.push_back(mxGetString(prhs[j]));
				}else{
					/*mxArray *variable=mxCreateNumericArray(mxGetNumberOfDimensions(prhs[dataTypeIndex]), mxGetDimensions(prhs[dataTypeIndex]), mxSINGLE_CLASS , mxREAL);
					for (int i = 0; i < mxGetNumberOfElements(variable); ++i)
					{
						((float*)mxGetPr(variable))[i]=0.f;
					}*/
					localVector.push_back(uploadData(socket, prhs[j]));
				}
			}
			result.push_back(localVector);
		}
		if(mxIsChar(prhs[i])&& 0==strcmp(mxGetString(prhs[i]).c_str(),"-sp")){
			std::vector<std::string> localVector;
			localVector.push_back("-sp");
			for (int j=i+1; j < nrhs; ++j)
			{
				if(mxIsChar(prhs[j]) && mxGetString(prhs[j]).c_str()[0]=='-'){
					break;
				}
				if(mxIsChar(prhs[j])){
					localVector.push_back(mxGetString(prhs[j]));
				}else{
					localVector.push_back(uploadData(socket, prhs[j]));
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

void mexFunctionWork(int nlhs, mxArray *plhs[],
				 int nrhs, const mxArray *prhs[],std::atomic<bool> &done){

	jobIdType id;
	bool stop=false;
	bool withTimeout=true;
	bool noOutput=false;

	std::vector<std::string> inputArray;
	std::vector<int> inputArrayIndex;

	for (int i = 0; i < nrhs; ++i)
	{
		if(mxIsChar(prhs[i])){
			char charTemp[2048];
			mxGetString(prhs[i], charTemp, 2048);
			inputArray.push_back(charTemp);
			inputArrayIndex.push_back(i);
		}
		if(mxIsScalar(prhs[i])){
			if(mxIsNumeric(prhs[i]))
				inputArray.push_back(std::to_string(float(mxGetScalar(prhs[i]))));
			else
				inputArray.push_back(std::to_string((mxGetScalar(prhs[i]))));
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
			if(nlhs>1){
				mexErrMsgIdAndTxt("gss:error", "require maximum one output");
				stop=true;
			}
		}
		if(!inputArray[i].compare("-statusOnly")){
			statusOnly=true;
			waitAndDownload=false;
			submit=false;
			id_index=inputArrayIndex[i]+1;
			if(nlhs>1){
				mexErrMsgIdAndTxt("gss:error", "require maximum one output");
				stop=true;
			}
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
		std::string adress=mxArrayToString(prhs[saP1_Index]);
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
			mexEvalString("drawnow");

		}
		#endif

	}

	int dim=2;

	if(dimP1_Index!=-1){
		if(mxIsChar(prhs[dimP1_Index])){
			dim=atoi(mxGetString(prhs[dimP1_Index]).c_str());
		}else{
			dim=mxGetScalar(prhs[dimP1_Index]);
		}
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
		port=mxGetScalar(prhs[pP1_Index]);
	if(saP1_Index!=-1)
		serverAddress=mxArrayToString(prhs[saP1_Index]);

	char address[4096];
	snprintf(address,4096,"tcp://%s:%d",serverAddress.c_str(),port);
	socket.connect (address);

	if(serverShutdown){
		infoContainer task;
		task.version=1;
		task.task=SHUTDOWN;

		zmq::message_t request (sizeof(infoContainer));
		memcpy(request.data (), &task, sizeof(infoContainer));
		if(!socket.send (request) && withTimeout ){
			{
				done=true;
				mexErrMsgIdAndTxt("g2s:error", "fail to shutdown the server");
			}
		}
		stop=true;
		done=true;
	}

	std::vector<std::vector<std::string> > dataString;
	if (submit) dataString=lookForUpload(socket, nrhs, prhs);


	/*for (int i = 0; i < dataString.size(); ++i)
	{
		printf("%s\n", dataString[i].c_str());
	}*/
	
	// start the process

	//std::this_thread::sleep_for(std::chrono::seconds(10));
	if(done) {
		stop=true;
	}

	if(!stop && submit){
		infoContainer task;
		task.version=1;
		char algo[2048];
		mxGetString(prhs[aP1_Index], algo, 2048);

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
				   !inputArray[i].compare("-dt") ||
				   !inputArray[i].compare("-sp") ||
				   !inputArray[i].compare("-ki") ||
				   !inputArray[i].compare("-a")){
					managed=true;
				}

				if (!managed)
				{
					unsigned i_source=i;
					Json::Value jsonArray(Json::arrayValue);
					while((inputArray.size()>i+1) && (inputArray[i+1].at(0)!='-'))
					{
						jsonArray.append(inputArray[i+1]);
						i++;
					}
					parameter[inputArray[i_source]]=jsonArray;
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
		if(!silentMode) std::cout << "Send job" << std::endl;
		if(!socket.send (request) && withTimeout ){
			mexErrMsgIdAndTxt("gss:error", "timeout sending job");
		}

		zmq::message_t reply;
		if(!socket.recv (&reply) && withTimeout){
			mexErrMsgIdAndTxt("gss:error", "timeout starting job, maybe job run on server !!");
		}
		if(reply.size()!=sizeof(jobIdType) && !silentMode) printf( "%s\n", "wrong answer !");
		id=*((jobIdType*)reply.data());
		if(id<0 && !silentMode) printf( "%s\n", "error in job distribution!");
		if(!silentMode)printf("job Id is: %u\n",id );
		mexEvalString("drawnow");

		if(done) {
			sendKill(socket, id);
			stop=true;
		}

	}else if (!submit) {
		id=*(unsigned*)mxGetPr(prhs[id_index]);
	}

	if(submit && !waitAndDownload){
		if(nlhs>0)
		{
			size_t one=1;
			plhs[0]=mxCreateNumericArray(1, &one, mxUINT32_CLASS, mxREAL);
			*(unsigned*)mxGetPr(plhs[0])=id;
		}
		noOutput=true;
		stop=true;
	}

	float lastProgression=-1.f;

	char nameFile[65]={0};
	snprintf(nameFile,65,"%u",id);

	if(kill) done=true;

	
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
			task.task=PROGESSION;
			
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
			if(reply.size()!=sizeof(int) && !silentMode) printf( "%s\n", "wrong answer !");
			else{
				int progress=*((int*)reply.data());
				if(!silentMode && (progress>=0) && fabs(lastProgression-progress/1000.)>0.0001){
					printf("progres %.3f%%\n",progress/1000. );
					lastProgression=progress/1000.;
					mexEvalString("drawnow");
				}

				if(!waitAndDownload && nlhs>0)
				{
					size_t one=1;
					plhs[0]=mxCreateNumericArray(1, &one, mxSINGLE_CLASS, mxREAL);
					*(float*)mxGetPr(plhs[0])=progress/1000;
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
				mexErrMsgIdAndTxt("gss:error", "timeout sending data");
			}

			zmq::message_t reply;
			if(!socket.recv (&reply) && withTimeout){
				mexErrMsgIdAndTxt("gss:error", "timeout receive data");
			}
			if(reply.size()!=sizeof(int)) mexErrMsgIdAndTxt("gss:error", "wrong answer if data exist!");
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
			mexErrMsgIdAndTxt("gss:error", "timeout asking for data");
		}
		zmq::message_t reply;
		if(!socket.recv (&reply) && withTimeout){
			mexErrMsgIdAndTxt("gss:error", "timeout : get data dont answer");
		}
		if(reply.size()!=0){
			if(!silentMode)printf("progres %.3f%%\n",100.0f );
			mexEvalString("drawnow");
			//printf( "recive\n");

			g2s::DataImage image((char*)reply.data());
			plhs[0]=convert2MxArray(image);
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



	if(!noOutput && nlhs>1){
		int downloadInformation=1;
		int positionInOutput=0;
		while (nlhs-1>downloadInformation){
			// download data
			snprintf(nameFile_local,65,"%s%u",prefix[positionInOutput].c_str(),id);

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
					mexErrMsgIdAndTxt("gss:error", "timeout sending data");
				}

				zmq::message_t reply;
				if(!socket.recv (&reply) && withTimeout){
					mexErrMsgIdAndTxt("gss:error", "timeout receive data");
				}
				if(reply.size()!=sizeof(int)) mexErrMsgIdAndTxt("gss:error", "wrong answer if data exist!");
				int isPresent=*((int*)reply.data());
				if (!isPresent) return;
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
				mexErrMsgIdAndTxt("gss:error", "timeout asking for data");
			}
			zmq::message_t reply;
			if(!socket.recv (&reply) && withTimeout){
				mexErrMsgIdAndTxt("gss:error", "timeout : get data dont answer");
			}
			
			if(reply.size()!=0){
				g2s::DataImage image((char*)reply.data());
				plhs[downloadInformation]=convert2MxArray(image);
				stop=true;
				downloadInformation++;
			}
			positionInOutput++;
		}
	}
	if(!noOutput && nlhs>1){
		// download data

		{
			infoContainer task;
			task.version=1;
			task.task=DURATION;
			
			zmq::message_t request (sizeof(infoContainer)+sizeof( jobIdType ));
			memcpy(request.data (), &task, sizeof(infoContainer));
			memcpy((char*)request.data()+sizeof(infoContainer),&id,sizeof( jobIdType ));
			if(!socket.send (request)){
				mexErrMsgIdAndTxt("gss:error", "timeout asking for data");
			}
			zmq::message_t reply;
			if(!socket.recv (&reply) && withTimeout){
				mexErrMsgIdAndTxt("gss:error", "timeout : get data dont answer");
			}
			if(reply.size()!=sizeof(int)) printf( "%s\n", "wrong answer !");
			else{
				int duration=*((int*)reply.data());
				size_t one=1;
				plhs[nlhs-1]=mxCreateNumericArray(1, &one, mxSINGLE_CLASS, mxREAL);
				*(float*)mxGetPr(plhs[nlhs-1])=duration/(1000.f);
			}	
		}
	}
	done=true;
};


void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	std::atomic<bool> done(false);
	auto myFuture = mexInterrupt::startInterruptCheck(done);
	mexFunctionWork(nlhs, plhs,  nrhs, prhs,done); // MATLAB APIs need to be exactuted in the main thread
	myFuture.wait();
	mexEvalString("drawnow");
}