#ifndef INERFACE_TEMPLATE_HPP
#define INERFACE_TEMPLATE_HPP

#ifndef VERSION
	#define VERSION "unknown"
#endif



#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <zmq.hpp>
#include <vector>
#include "protocol.hpp"
#include "picosha2.h"
#include <json/json.h>
#include <numeric>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <map>
#include <any>
#include <set>
#include <thread>
#include "DataImage.hpp"

#ifdef WITH_WEB_SUPPORT
#include "cvtZMQ2WS.hpp"
#endif 

typedef unsigned jobIdType;

#ifdef WITH_WEB_SUPPORT
void cvtServerCall(std::string from, std::string to, std::atomic<bool> &serverRun, std::atomic<bool> &done){
	cvtServer((char*)from.c_str(),(char*)to.c_str(),serverRun, done);
}

#endif

//template <class T>
class InerfaceTemplate
{
public:

	virtual unsigned anyNativeToUnsigned(std::any val)=0;
	virtual float anyNativeToFloat(std::any val)=0;
	virtual double anyNativeToDouble(std::any val)=0;
	virtual long unsigned anyNativeToLongUnsigned(std::any val)=0;

	virtual void unlockThread(){}
	virtual void lockThread(){}
	virtual void updateDisplay(){};
	virtual bool userRequestInteruption(){return false;}

	virtual bool isDataMatrix(std::any val)=0;

	virtual std::string nativeToStandardString(std::any val)=0;
	virtual double nativeToScalar(std::any val)=0;
	virtual unsigned nativeToUint32(std::any val)=0;

	virtual std::any ScalarToNative(double val)=0;
	virtual std::any Uint32ToNative(unsigned val)=0;
	virtual void sendError(std::string val)=0;
	virtual void sendWarning(std::string val)=0;
	virtual void eraseAndPrint(std::string val)=0;

	std::string toString(std::any val){
		if(val.type()==typeid(nullptr))
			return "";
		if(val.type()==typeid(std::string))
			return std::any_cast<std::string>(val);
		return nativeToStandardString(val);
	}

//


	virtual std::any convert2NativeMatrix(g2s::DataImage &im)=0;
	virtual g2s::DataImage convertNativeMatrix2DataImage(std::any matrix, std::any dataTypeVariable=nullptr)=0;

		//to improuve
	std::string uploadData(zmq::socket_t &socket, std::any matrix, std::any dataTypeVariable=nullptr){
		bool withTimeout=false;
		char sourceName[65]={0};

		char* rawData=convertNativeMatrix2DataImage(matrix, dataTypeVariable).serialize();
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

		if(!socket.send (request,zmq::send_flags::none) && withTimeout){
			sendError("timeout sending data");
		}

		zmq::message_t reply;
		if(!socket.recv (reply) && withTimeout){
			sendError("timeout receive data");
		}
		if(reply.size()!=sizeof(int)) sendError("wrong answer if data exist!");
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
			

			if(!socket.send (request,zmq::send_flags::none) && withTimeout){
				sendError("timeout sending data");
			}

			zmq::message_t reply;
			if(!socket.recv (reply) && withTimeout){
				sendError("timeout receive data");
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

	void lookForUpload(zmq::socket_t &socket, std::multimap<std::string, std::any> &input){

		auto dataTypeVariable=input.find("-dt");
		std::set<std::string> listOfParameterToUploadIfNeededWithdataTypeVariable= {"-ti","-di","-nl"};
		std::set<std::string> listOfParameterToUploadIfNeededWithoutdataTypeVariable= {"-ki","-sp","-ii","-ni","-kii","-kvi"};
		for (auto it=input.begin(); it!=input.end(); ++it){
			if(listOfParameterToUploadIfNeededWithdataTypeVariable.find(it->first) != listOfParameterToUploadIfNeededWithdataTypeVariable.end())
				if(isDataMatrix(it->second)){
					if(dataTypeVariable!=input.end())
						it->second=std::any(uploadData(socket, it->second,dataTypeVariable->second));
					else
						sendError("-dt is missing, impossible to uplaod a matrix without data type");
					}
			if(listOfParameterToUploadIfNeededWithoutdataTypeVariable.find(it->first) != listOfParameterToUploadIfNeededWithoutdataTypeVariable.end()){
				if(isDataMatrix(it->second))
					it->second=std::any(uploadData(socket, it->second));
			}
		}
		input.erase("-dt");
	}

	inline void sendKill(zmq::socket_t &socket, jobIdType id, bool silent=false){
		infoContainer task;
		task.version=1;
		task.task=KILL;
			
		zmq::message_t request (sizeof(infoContainer)+sizeof(jobIdType));
		memcpy(request.data (), &task, sizeof(infoContainer));
		memcpy((char*)request.data()+sizeof(infoContainer),&id,sizeof(jobIdType));
		socket.send (request,zmq::send_flags::none);
		zmq::message_t reply;
		socket.recv (reply);
		
		//if(!silent)sendError("Ctrl C, user interupted");
	}

	void runStandardCommunication(std::multimap<std::string, std::any> input, std::multimap<std::string, std::any> &outputs, int maxOutput=INT_MAX){

		std::atomic<bool> done(false);

		jobIdType id=0;
		bool stop=false;
		bool withTimeout=true;
		int timeout=60000;
		bool noOutput=false;
		bool spectifiedTimeout=false;

		bool submit=true;
		bool statusOnly=false;
		bool waitAndDownload=true;
		bool kill=false;
		bool serverShutdown=false;
		bool silentMode=false;
		bool requestServerStatus=false; // to check programmatically if the server is running

		if(input.count("-noTO")>0)
		{
			withTimeout=false;
		}
		if(input.count("-TO")>0)
		{
			timeout=anyNativeToUnsigned(input.find("-TO")->second);
			spectifiedTimeout=true;
		}
		if(input.count("-submitOnly")>0)
		{
			waitAndDownload=false;
		}
		if(input.count("-shutdown")>0)
		{
			serverShutdown=true;
			waitAndDownload=false;
			noOutput=true;
		}
		if(input.count("-silent")>0)
		{
			silentMode=true;
		}

		if(input.count("-statusOnly")>0){
			statusOnly=true;
			waitAndDownload=false;
			submit=false;
			input.insert(std::pair<std::string, std::any>("-id",input.find("-statusOnly")->second));					
		}
		if(input.count("-waitAndDownload")>0){
			submit=false;
			input.insert(std::pair<std::string, std::any>("-id",input.find("-waitAndDownload")->second));
		}
		if(input.count("-kill")>0){
			kill=true;
			submit=false;
			input.insert(std::pair<std::string, std::any>("-id",input.find("-kill")->second));
		}

		if(input.count("-serverStatus")>0){
			requestServerStatus=true;
		}

		std::atomic<bool> serverRun;
		serverRun=true;

		std::string serverAddress="localhost";
		if(input.count("-sa")>0)serverAddress=nativeToStandardString(input.find("-sa")->second);
		input.erase("-sa");
		std::transform(serverAddress.begin(), serverAddress.end(), serverAddress.begin(),::tolower);
		#ifdef WITH_WEB_SUPPORT
		if(!serverAddress.compare("web")||!serverAddress.compare("browser")){
			//printf("use browser server\n");
			serverRun=false;
			std::string from="tcp://*:8128";
			std::string to="ws://localhsot:8129";

			std::thread serverThread(cvtServerCall,from,to,std::ref(serverRun),std::ref(done));
			serverThread.detach();
			while (!serverRun){
				std::this_thread::sleep_for(std::chrono::milliseconds(300));
			}

			printf("server run now\n");
			updateDisplay();

		}
		#endif

		zmq::context_t context (1);
		zmq::socket_t socket (context, ZMQ_REQ);
		
		socket.setsockopt(ZMQ_LINGER, 500);
		if(withTimeout){
			socket.setsockopt(ZMQ_RCVTIMEO, 500);
			socket.setsockopt(ZMQ_SNDTIMEO, 500);
		}

		if(spectifiedTimeout){
			printf( "%d\n", timeout);
			socket.setsockopt(ZMQ_LINGER, timeout);
			socket.setsockopt(ZMQ_RCVTIMEO, timeout);
			socket.setsockopt(ZMQ_SNDTIMEO, timeout);
		}

		short port=8128;
		if(input.count("-p")>0)port=nativeToScalar(input.find("-p")->second);

		char address[4096];
		sprintf(address,"tcp://%s:%d",serverAddress.c_str(),port);
		socket.connect (address);

		if (requestServerStatus){

			int serverStatus=0;

			infoContainer task;
			task.version=1;
			task.task=SERVER_STATUS;

			zmq::message_t request (sizeof(infoContainer));
			memcpy(request.data (), &task, sizeof(infoContainer));
			if(!socket.send (request,zmq::send_flags::none) ){
				serverStatus=-1;
			}
			zmq::message_t reply;
			if(serverStatus==0 && !socket.recv (reply) ){
				serverStatus=-2;
			}
			if(serverStatus==0 && reply.size()==sizeof(int))
				serverStatus=*((int*)reply.data());
			outputs.insert({"1",ScalarToNative(double(serverStatus))});
			done=true;
			return;
		}


		{// check if server is available
			infoContainer task;
			task.version=1;
			task.task=SERVER_STATUS;

			zmq::message_t request (sizeof(infoContainer));
			memcpy(request.data (), &task, sizeof(infoContainer));
			if(!socket.send (request,zmq::send_flags::none) && withTimeout ){
				done=true;
				sendError("The server is probably off-line, please execute first ./server. If you try to connect to a remote server maybe the network connection  ");
			}
			zmq::message_t reply;
			if(!socket.recv (reply) && withTimeout){
				done=true;
				sendError("The server is probably off-line, please execute first ./server. If you try to connect to a remote server maybe the network connection  ");
			}
			if(reply.size()!=sizeof(int)) sendError("wrong answer!");
			int serverStatus=*((int*)reply.data());
			if(serverStatus<1){
				sendError("something wrong on the server !");
			}
		}

		socket.setsockopt(ZMQ_LINGER, timeout);
		socket.setsockopt(ZMQ_RCVTIMEO, timeout);
		socket.setsockopt(ZMQ_SNDTIMEO, timeout);

		if(serverShutdown){
			infoContainer task;
			task.version=1;
			task.task=SHUTDOWN;

			zmq::message_t request (sizeof(infoContainer));
			memcpy(request.data (), &task, sizeof(infoContainer));
			if(!socket.send (request,zmq::send_flags::none) && withTimeout ){
				{
					done=true;
					sendError("fail to shutdown the server");
				}
			}
			done=true;
			return;
		}


		if (submit) lookForUpload(socket, input);

		// start the process
		unlockThread();
		//std::this_thread::sleep_for(std::chrono::seconds(10));

		if(!done && submit){
			infoContainer task;
			task.version=1;

			Json::Value object(Json::objectValue);
			std::string test;
			if(input.count("-a")>0) object["Algorithm"]=nativeToStandardString(input.find("-a")->second);
			else if(!silentMode) sendWarning("Upload only! You maybe want to use '-a' to specify an algorithm.");
			input.erase("-a");
			{
				object["Priority"]="1";
				{
					Json::Value jsonArray(Json::arrayValue);
					for (auto it=input.equal_range("-after").first; it!=input.equal_range("-after").second; ++it)
					{
						jsonArray.append(anyNativeToUnsigned(it->second));
					}
					input.erase("-after");
					object["Dependency"]=jsonArray;
				}
				
				Json::Value parameter(Json::objectValue);

				std::set<std::string> listOfAllKeys;
				for (auto it = input.begin(); it != input.end(); ++it)
				{
					listOfAllKeys.insert(it->first);					
				}

				for (auto itKey = listOfAllKeys.begin(); itKey != listOfAllKeys.end(); ++itKey)
				{	
					Json::Value jsonArray(Json::arrayValue);	
					for (auto it=input.equal_range(*itKey).first; it!=input.equal_range(*itKey).second; ++it)
					{
						jsonArray.append(toString(it->second));
					}
					parameter[*itKey]=jsonArray;
				}

				object["Parameter"]=parameter;
			}
			/*
			Json::FastWriter fast;
		    std::string jsonJob = fast.write(object);
			*/
			Json::StreamWriterBuilder builder;
			builder["commentStyle"] = "None";
			builder["indentation"] = "";  // or whatever you like
			std::string jsonJob = Json::writeString(builder, object);
			const char* jsonJob_c=jsonJob.c_str();

		    //printf( "%s\n", jsonJob_c);

			task.task=JOB;

			zmq::message_t request (sizeof(infoContainer)+strlen(jsonJob_c));
			memcpy(request.data (), &task, sizeof(infoContainer));
			memcpy((char*)request.data()+sizeof(infoContainer),jsonJob_c,strlen(jsonJob_c));
			if(!socket.send (request,zmq::send_flags::none) && withTimeout ){
				sendError("timeout sending job");
			}

			zmq::message_t reply;
			if(!socket.recv (reply) && withTimeout){
				sendError("timeout starting job, maybe job run on server !!");
			}
			if(reply.size()!=sizeof(jobIdType) && !silentMode) printf( "%s\n", "wrong answer !");
			id=*((jobIdType*)reply.data());
			if(id<0 && !silentMode) printf( "%s\n", "error in job distribution!");
			if(!silentMode)printf("job Id is: %u\n",id );
			updateDisplay();

			if(done) {
				sendKill(socket, id);
				stop=true;
			}
		
		}else if (!submit) {
			id=anyNativeToUnsigned(input.find("-id")->second);
		}

		outputs.insert(std::pair<std::string, std::any>("id",Uint32ToNative(id)));
		if(submit && !waitAndDownload){
			noOutput=true;
			done=true;
		}

		float lastProgression=-1.f;

		char nameFile[65]={0};
		sprintf(nameFile,"%u",id);

		if(kill){
			sendKill(socket, id, true);
			done=true;
		}

		if(waitAndDownload || statusOnly) outputs.insert({"progression",0.f});

		while(!done && (waitAndDownload || statusOnly)) {

			// status
			if(!silentMode || statusOnly){
				infoContainer task;
				task.version=1;
				task.task=PROGESSION;
				
				zmq::message_t request (sizeof(infoContainer)+sizeof( jobIdType ));
				memcpy(request.data (), &task, sizeof(infoContainer));
				memcpy((char*)request.data()+sizeof(infoContainer),&id,sizeof( jobIdType ));
				if(!socket.send (request,zmq::send_flags::none)){
					continue;
				}
				zmq::message_t reply;
				if(!socket.recv (reply) && withTimeout){
					continue;
				}

				if(reply.size()!=sizeof(int)) sendWarning( "wrong answer !");
				else{
					int progress=*((int*)reply.data());
					if((progress>=0) && fabs(lastProgression-progress/1000.)>0.0001){
						char buff[100];
						snprintf(buff, sizeof(buff), "progress %.3f%%",progress/1000.);
						std::string buffAsStdStr = buff;
						if(!silentMode)eraseAndPrint(buffAsStdStr);
						lastProgression=progress/1000.;
						if(!silentMode)updateDisplay();
					}
					outputs.find("progression")->second=lastProgression;
				}
				if(statusOnly) break;
			}

			if(userRequestInteruption()){
				kill=true;
				break;
			}

			{
				infoContainer task;
				task.version=1;
				task.task=JOB_STATUS;

				zmq::message_t request(sizeof(infoContainer)+sizeof( jobIdType ));
				memcpy(request.data(), &task, sizeof(infoContainer));
				memcpy((char*)request.data()+sizeof(infoContainer),&id,sizeof( jobIdType ));
				if(!socket.send (request,zmq::send_flags::none)){
					continue;
				}
				zmq::message_t reply;
				if(!socket.recv (reply) && withTimeout){
					continue;
				}

				if(reply.size()!=sizeof(int)){
					sendWarning( "handel error !");
				}else{

					int statusCode=*((int*)reply.data());
					if(statusCode==0){
						char buff[100];
						snprintf(buff, sizeof(buff), "progress %.3f%%\n",100.);
						std::string buffAsStdStr = buff;
						if(!silentMode)eraseAndPrint(buffAsStdStr);
						lastProgression=100.;
						outputs.find("progression")->second=lastProgression;
						break;
					}
					if(statusCode>0){
						// error occurred
						// check for error
						char nameFile_local[65]={0};
						sprintf(nameFile_local,"error_%u",id);
					
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

							if(!socket.send (request,zmq::send_flags::none) && withTimeout){
								sendError("timeout sending data");
							}

							zmq::message_t reply;
							if(!socket.recv (reply) && withTimeout){
								sendError("timeout receive data");
							}
							if(reply.size()!=sizeof(int)) sendError("wrong answer if data exist!");
							int isPresent=*((int*)reply.data());
							if (!isPresent) break;
						}

						{
							infoContainer task;
							task.version=1;
							task.task=DOWNLOAD_TEXT;

							//printf("%s\n", nameFile);
							zmq::message_t request (sizeof(infoContainer)+64);
							memcpy (request.data(), &task, sizeof(infoContainer));
							memcpy ((char*)request.data()+sizeof(infoContainer), nameFile_local, 64);
							//std::cout << "ask for data" << std::endl;
							if(!socket.send (request,zmq::send_flags::none) && withTimeout ){
								sendError("timeout asking for data");
							}
							zmq::message_t reply;
							if(!socket.recv (reply) && withTimeout){
								sendError("timeout : get data dont answer");
							}
							
							if(reply.size()!=0){
								std::string errorText=std::string((char*)reply.data());
								sendError(errorText);
							}
						}
						break;
					}
				}

			}

			if(userRequestInteruption()){
				kill=true;
				break;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(600));
		}

		if(kill && !done) {
			sendKill(socket, id);
			done=true;
			sendError("canceled job");
		}


		lockThread();
		//id.write(std::string("im_2_")+std::to_string(uniqueID));

		//look for imageData result
		int dataIndex=1;
		int nbElement=0;
		while(waitAndDownload){
			char nameFile_local[65]={0};
			sprintf(nameFile_local,"im_%d_%u",dataIndex,id);

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

				if(!socket.send (request,zmq::send_flags::none) && withTimeout){
					sendError("timeout sending data");
				}

				zmq::message_t reply;
				if(!socket.recv (reply) && withTimeout){
					sendError("timeout receive data");
				}
				if(reply.size()!=sizeof(int)) sendError("wrong answer if data exist!");
				int isPresent=*((int*)reply.data());
				if (!isPresent) break;
			}

			{
				infoContainer task;
				task.version=1;
				task.task=DOWNLOAD;

				//printf("%s\n", nameFile);
				zmq::message_t request (sizeof(infoContainer)+64);
				memcpy (request.data(), &task, sizeof(infoContainer));
				memcpy ((char*)request.data()+sizeof(infoContainer), nameFile_local, 64);
				//std::cout << "ask for data" << std::endl;
				if(!socket.send (request,zmq::send_flags::none) && withTimeout ){
					sendError("timeout asking for data");
				}
				zmq::message_t reply;
				if(!socket.recv (reply) && withTimeout){
					sendError("timeout : get data dont answer");
				}
				
				if(reply.size()!=0){
					g2s::DataImage image((char*)reply.data());
					outputs.insert({std::to_string(dataIndex),convert2NativeMatrix(image)});
				}
				dataIndex++;
				nbElement++;
			}
			if(maxOutput-1<nbElement) break;
		}

		
		if(waitAndDownload)
		{
			infoContainer task;
			task.version=1;
			task.task=DURATION;
			
			zmq::message_t request (sizeof(infoContainer)+sizeof( jobIdType ));
			memcpy(request.data (), &task, sizeof(infoContainer));
			memcpy((char*)request.data()+sizeof(infoContainer),&id,sizeof( jobIdType ));
			if(!socket.send (request,zmq::send_flags::none)){
				sendError("timeout asking for data");
			}
			zmq::message_t reply;
			if(!socket.recv (reply) && withTimeout){
				sendError("timeout : get data dont answer");
			}

			if(reply.size()!=sizeof(int))
				printf( "%s\n", "wrong answer !");
			else{
				outputs.insert({"t",ScalarToNative(double(*((int*)reply.data()))/1000.)});				
			}	
			
		}
		done=true;
		
	};
};

#endif // INERFACE_TEMPLATE_HPP
