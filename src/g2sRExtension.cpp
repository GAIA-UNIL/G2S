#include <cmath>
#include <iostream>
#include <zmq.hpp>
#include <vector>
#include <json/json.h>
#include <numeric>
#include <chrono>
#include <thread>
#include <atomic>
#include <future>
#include <algorithm>
#include <Rcpp.h>

#include "protocol.hpp"
#include "picosha2.h"
#include "DataImage.hpp"
#ifdef WITH_WEB_SUPPORT
#include "cvtZMQ2WS.hpp"
#endif 

typedef unsigned jobIdType;

/* Check for interrupt without long jumping */
void check_interrupt_fn(void *dummy) {
  R_CheckUserInterrupt();
}

int pending_interrupt() {
  return !(R_ToplevelExec(check_interrupt_fn, NULL));
}

/*
				if(vec.hasAttribute("dim"))	//vec.attributeNames();
			{	
				int numberOfDim=Rcpp::as<Rcpp::IntegerVector>(vec.attr("dim")).size();
				fprintf(stderr, "%d\n", numberOfDim);
			}else
*/

inline Rcpp::RObject convert2NDArray(g2s::DataImage &image){

	std::vector<int> dimsArray(image._dims.size()+1);
	for (int i = 0; i < image._dims.size(); ++i)
	{
		dimsArray[i]=image._dims[i];
	}
	std::reverse(dimsArray.begin(),dimsArray.begin()+image._dims.size());
	dimsArray[image._dims.size()]=image._nbVariable;
	Rcpp::RObject array;
	int completeSize=image.dataSize();
	if(dimsArray.back()==1)dimsArray.pop_back();
	if(image._encodingType==g2s::DataImage::Float){
		Rcpp::NumericVector localArray=Rcpp::NumericVector(completeSize);
		localArray.attr("dim")=Rcpp::as<Rcpp::IntegerVector>(Rcpp::wrap(dimsArray));
		//copy memory
		for (int i = 0; i < completeSize; ++i)
		{
			localArray[i]=image._data[i];
		}
		array=localArray;
	}
	if(image._encodingType==g2s::DataImage::Integer|| image._encodingType==g2s::DataImage::UInteger){
		Rcpp::IntegerVector localArray=Rcpp::IntegerVector(completeSize);
		localArray.attr("dim")=Rcpp::as<Rcpp::IntegerVector>(Rcpp::wrap(dimsArray));
		//copy memory
		for (int i = 0; i < completeSize; ++i)
		{
			localArray[i]=((int*)image._data)[i];
		}
		array=localArray;
	}

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
	Rcpp::stop(
				"%s : %s", "gss:error", "Ctrl C, user interupted");
}


inline std::string uploadData(zmq::socket_t &socket, Rcpp::RObject* prh, Rcpp::RObject* variableTypeArray=nullptr){

	bool withTimeout=false;
	char sourceName[65]={0};
	Rcpp::NumericVector arrayRcpp=Rcpp::as<Rcpp::NumericVector>(*prh);
	
	int dataSize=arrayRcpp.size();
	int nbOfVariable=1;
	if(variableTypeArray)
	{
		Rcpp::NumericVector variableTypeArrayRcpp=Rcpp::as<Rcpp::NumericVector>(*variableTypeArray);
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
	
	if (variableTypeArray)
	{
		Rcpp::NumericVector variableTypeArrayRcpp=Rcpp::as<Rcpp::NumericVector>(*variableTypeArray);
		for (int i = 0; i < nbOfVariable; ++i)
		{
			if(variableTypeArrayRcpp[i]==0.f)image._types[i]=g2s::DataImage::VaraibleType::Continuous;
			if(variableTypeArrayRcpp[i]==1.f)image._types[i]=g2s::DataImage::VaraibleType::Categorical;
		}
	}
	
	memset(data,0,sizeof(float)*dataSize);
	//manage data

	for (int i = 0; i < dataSize; ++i)
	{
		data[i]=arrayRcpp[i];
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
		Rcpp::stop(
				"%s : %s", "gss:error", "timeout sending data");
	}

	zmq::message_t reply;
	if(!socket.recv (&reply) && withTimeout){
		Rcpp::stop(
				"%s : %s", "gss:error", "timeout receive data");
	}
	if(reply.size()!=sizeof(int))
		Rcpp::stop(
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
			Rcpp::stop(
				"%s : %s", "gss:error", "timeout sending data");
		}

		zmq::message_t reply;
		if(!socket.recv (&reply) && withTimeout){
			Rcpp::stop(
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

inline std::vector<std::vector<std::string> > lookForUpload(zmq::socket_t &socket, Rcpp::List args){
	std::vector<std::vector<std::string> > result;
	int nrhs = args.size();
	int dataTypeIndex=-1;
	for (int i = 0; i < nrhs; ++i)
	{
		if(Rcpp::is<Rcpp::CharacterVector>(args[i]) && (0==strcmp(Rcpp::as<std::string>(args[i]).c_str(),"-dt")) ) dataTypeIndex=i+1;
	}

	for (int i = 0; i < nrhs; ++i)
	{
		if(Rcpp::is<Rcpp::CharacterVector>(args[i]) && 0==strcmp(Rcpp::as<std::string>(args[i]).c_str(),"-ti")){
			std::vector<std::string> localVector;
			localVector.push_back("-ti");
			for (int j=i+1; j < nrhs; ++j)
			{
				if(Rcpp::is<Rcpp::CharacterVector>(args[j]) && Rcpp::as<std::string>(args[j]).c_str()[0]=='-'){
					break;
				}
				if(Rcpp::is<Rcpp::CharacterVector>(args[j])){
					localVector.push_back(Rcpp::as<std::string>(args[j]));
				}else{
					if(dataTypeIndex<0)Rcpp::stop(
						 "%s : %s", "gss:error", "-dt wasn't specified");
					Rcpp::RObject obj1=Rcpp::as<Rcpp::RObject>(args[j]);
					Rcpp::RObject obj2=Rcpp::as<Rcpp::RObject>(args[dataTypeIndex]);
					localVector.push_back(uploadData(socket, &obj1, &obj2));
				}
			}
			result.push_back(localVector);
		}
		if(Rcpp::is<Rcpp::CharacterVector>(args[i]) && 0==strcmp(Rcpp::as<std::string>(args[i]).c_str(),"-di")){
			std::vector<std::string> localVector;
			localVector.push_back("-di");
			for (int j=i+1; j < nrhs; ++j)
			{
								if(Rcpp::is<Rcpp::CharacterVector>(args[j]) && Rcpp::as<std::string>(args[j]).c_str()[0]=='-'){
					break;
				}
				if(Rcpp::is<Rcpp::CharacterVector>(args[j])){
					localVector.push_back(Rcpp::as<std::string>(args[j]));
				}else{
					if(dataTypeIndex<0)Rcpp::stop(
						 "%s : %s", "gss:error", "-dt wasn't specified");
					Rcpp::RObject obj1=Rcpp::as<Rcpp::RObject>(args[j]);
					Rcpp::RObject obj2=Rcpp::as<Rcpp::RObject>(args[dataTypeIndex]);
					localVector.push_back(uploadData(socket, &obj1, &obj2));
				}
			}
			result.push_back(localVector);
		}
		if(Rcpp::is<Rcpp::CharacterVector>(args[i]) && 0==strcmp(Rcpp::as<std::string>(args[i]).c_str(),"-ki")){
			std::vector<std::string> localVector;
			localVector.push_back("-ki");
			for (int j=i+1; j < nrhs; ++j)
			{
				if(Rcpp::is<Rcpp::CharacterVector>(args[j]) && Rcpp::as<std::string>(args[j]).c_str()[0]=='-'){
					break;
				}
				if(Rcpp::is<Rcpp::CharacterVector>(args[j])){
					localVector.push_back(Rcpp::as<std::string>(args[j]));
				}else{
					if(dataTypeIndex<0)Rcpp::stop(
						 "%s : %s", "gss:error", "-dt wasn't specified");
					Rcpp::RObject obj1=Rcpp::as<Rcpp::RObject>(args[j]);
					Rcpp::IntegerVector dim=Rcpp::as<Rcpp::NumericVector>(args[dataTypeIndex]).attr("dim");
					Rcpp::RObject obj2=Rcpp::NumericVector(dim[0],0.f);
					localVector.push_back(uploadData(socket, &obj1, &obj2));
				}
			}
			result.push_back(localVector);
		}
		if(Rcpp::is<Rcpp::CharacterVector>(args[i]) && 0==strcmp(Rcpp::as<std::string>(args[i]).c_str(),"-sp")){
			std::vector<std::string> localVector;
			localVector.push_back("-sp");
			for (int j=i+1; j < nrhs; ++j)
			{
				if(Rcpp::is<Rcpp::CharacterVector>(args[j]) && Rcpp::as<std::string>(args[j]).c_str()[0]=='-'){
					break;
				}
				if(Rcpp::is<Rcpp::CharacterVector>(args[j])){
					localVector.push_back(Rcpp::as<std::string>(args[j]));
				}else{
					Rcpp::RObject obj1=Rcpp::as<Rcpp::RObject>(args[j]);
					localVector.push_back(uploadData(socket, &obj1));
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

void RFunctionWork( Rcpp::List args, std::atomic<bool> &done, std::vector<Rcpp::RObject> &plhs){

	jobIdType id=0;
	bool stop=false;
	bool withTimeout=true;
	bool noOutput=false;

	std::vector<std::string> inputArray;
	std::vector<int> inputArrayIndex;

	int nrhs = args.size();

	for (int i = 0; i < nrhs; ++i)
	{
		if(Rcpp::is<Rcpp::CharacterVector>(args[i])){
			inputArray.push_back(Rcpp::as<std::string>(args[i]));
			inputArrayIndex.push_back(i);
		}
		if(Rcpp::is<Rcpp::NumericVector>(args[i])){
			Rcpp::NumericVector vec=Rcpp::as<Rcpp::NumericVector>(args[i]);
			if(!vec.hasAttribute("dim")){
				inputArray.push_back(std::to_string(Rcpp::as<float>(vec)));
				inputArrayIndex.push_back(i);
			}
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

		}
		if(!inputArray[i].compare("-statusOnly")){
			statusOnly=true;
			waitAndDownload=false;
			submit=false;
			id_index=inputArrayIndex[i]+1;

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
		std::string adress=Rcpp::as<std::string>(args[saP1_Index]);
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
		port=Rcpp::as<short>(args[pP1_Index]);
	if(saP1_Index!=-1)
		serverAddress=Rcpp::as<std::string>(args[saP1_Index]);

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
			Rcpp::stop(
				"%s : %s", "gss:error", "fail to shutdown the server");
		}
		stop=true;
		done=true;
	}

	std::vector<std::vector<std::string> > dataString=lookForUpload(socket, args);

	

	if(done) {
		stop=true;
	}

	if(!stop && submit){
		infoContainer task;
		task.version=1;
		char algo[2048];
		strcpy(algo,Rcpp::as<std::string>(args[aP1_Index]).c_str());

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
			Rcpp::stop(
				"%s : %s", "gss:error", "timeout sending job");
		}

		zmq::message_t reply;
		if(!socket.recv (&reply) && withTimeout){
			Rcpp::stop(
				"%s : %s", "gss:error", "timeout starting job, maybe job run on server !!");
		}
		if((reply.size()!=sizeof(jobIdType))&& !silentMode){
			printf( "%s\n", "wrong answer !");
		}
		id=*((jobIdType*)reply.data());
		if(id<0 && !silentMode)
		{
			printf( "%s\n", "error in job distribution!");
		}
		if(!silentMode){
			printf("job Id is: %u\n",id );
		}

		if(done) {
			sendKill(socket, id);
			stop=true;
		}

	}else if (!submit) {
		id=Rcpp::as<jobIdType>(args[id_index]);
	}


	if(submit && !waitAndDownload){
		
		{
			size_t one=1;
			plhs.push_back(Rcpp::IntegerVector::create(id));
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
		printf("progres %.3f%%",0/1000. );
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
					printf( "%s\n", "wrong answer !");
				}
			else{
				int progress=*((int*)reply.data());
				if(!silentMode && (progress>=0) && fabs(lastProgression-progress/1000.)>0.0001){
					printf("\rprogres %.3f%%",progress/1000. );
					lastProgression=progress/1000.;
				}
				if(!waitAndDownload)
				{
					size_t one=1;
					plhs.push_back(Rcpp::NumericVector(progress/1000.));
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
				Rcpp::stop(
				"%s : %s", "gss:error", "timeout sending data");
			}

			zmq::message_t reply;
			if(!socket.recv (&reply) && withTimeout){
				Rcpp::stop(
				"%s : %s", "gss:error", "timeout receive data");
			}
			if(reply.size()!=sizeof(int))
				Rcpp::stop(
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
			Rcpp::stop(
				"%s : %s", "gss:error", "timeout asking for data");
		}
		zmq::message_t reply;
		if(!socket.recv (&reply) && withTimeout){
			Rcpp::stop(
				"%s : %s", "gss:error", "timeout : get data dont answer");
		}
		if(reply.size()!=0){
			if(!silentMode){
				printf("\rprogres %.3f%%\n",100.0f );
			}
			g2s::DataImage image((char*)reply.data());
			plhs.push_back(convert2NDArray(image));
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
					Rcpp::stop(
				"%s : %s", "gss:error", "timeout sending data");
				}

				zmq::message_t reply;
				if(!socket.recv (&reply) && withTimeout){
					Rcpp::stop(
				"%s : %s", "gss:error", "gss:error", "timeout receive data");
				}
				if(reply.size()!=sizeof(int))
					Rcpp::stop(
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
				Rcpp::stop(
				"%s : %s", "gss:error", "timeout asking for data");
			}
			zmq::message_t reply;
			if(!socket.recv (&reply) && withTimeout){
				Rcpp::stop(
				"%s : %s", "gss:error", "timeout : get data dont answer");
			}
			
			if(reply.size()!=0){
				g2s::DataImage image((char*)reply.data());
				plhs.push_back(convert2NDArray(image));
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
				Rcpp::stop(
				"%s : %s", "gss:error", "timeout asking for data");
			}
			zmq::message_t reply;
			if(!socket.recv (&reply) && withTimeout){
				Rcpp::stop(
				"%s : %s", "gss:error", "timeout : get data dont answer");
			}
			if(reply.size()!=sizeof(int)) printf( "%s\n", "wrong answer !");
			else{
				int duration=*((int*)reply.data());
				size_t one=1;
				plhs.push_back(Rcpp::NumericVector::create(duration/(1000.f)));
			}   
		}
	}
	done=true;
}

void testIfInterupted(std::atomic<bool> &done){
	while (!done){
		std::this_thread::sleep_for(std::chrono::milliseconds(300));
		if(pending_interrupt()){
			done=true;
		}
	}
}

// [[Rcpp::export]]
Rcpp::RObject g2sInterface(Rcpp::List args)
{
	
	std::atomic<bool> done(false);
	std::vector<Rcpp::RObject> result;

	//RFunctionWork( args, done, result);

	auto myFuture = std::async(std::launch::async, RFunctionWork, args,std::ref(done),std::ref(result));
	testIfInterupted(std::ref(done));
	myFuture.wait();

	return Rcpp::as<Rcpp::List>(Rcpp::wrap(result));
}