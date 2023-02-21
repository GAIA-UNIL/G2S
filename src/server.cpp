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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> /* for fork */
#include <sys/types.h> /* for pid_t */
#include <sys/wait.h> /* for wait */
#include <sys/stat.h>
#include <thread>

#include <spawn.h>

#ifdef WITH_VERSION_CONTROL
#include <curl/curl.h>
#endif


#include <zmq.hpp>
#if __EMSCRIPTEN__
#include "wsZmq.hpp"
#endif

#include "protocol.hpp"

#include "jobTasking.hpp"
#include "jobManager.hpp"
#include "dataManagement.hpp"
#include "status.hpp"

#include <stdio.h>
#include <dirent.h>

#ifndef SERVER_TYPE
#define SERVER_TYPE 0
#endif

 
void removeAllFile(char* dir, double olderThan)
{
	struct dirent *de;

	DIR *dr = opendir(dir);

	if (dr == NULL) 
	{
		printf("Could not open current directory" );
		return;
	}

	time_t now;
	time(&now);

	while ((de = readdir(dr)) != NULL){
		char completeName[2048];
		sprintf(completeName,"%s/%s",dir,de->d_name);
		struct stat info;

		if(0==lstat(completeName,&info) && (S_ISREG(info.st_mode) || S_ISLNK(info.st_mode))){
			//printf("%s\n", de->d_name);
			double difInSeconds=difftime(now,info.st_atime);
			//printf("%f s\n", difInSeconds);
			if(difInSeconds>olderThan)
				unlink(completeName);
		}
	}
	closedir(dr);
}

		
int main(int argc, char const *argv[]) {
#ifdef WITH_VERSION_CONTROL
	std::string currentVersion=CURRENT_VERSION;
#endif
	bool runAsDaemon=false;
	bool singleTask=false;
	bool functionMode=false;
	bool keepOldData=false;
	float timeoutDuration=std::nanf("0");
	double maxFileAge=24*3600.;
	short port=8128;
	unsigned maxNumberOfConcurrentJob=500;

	
	for (int i = 1; i < argc; ++i)
	{
		if(0==strcmp(argv[i], "-d")) runAsDaemon=true;
		if((0==strcmp(argv[i], "-To")) && (i+1 < argc))
		{
			timeoutDuration=atof(argv[i+1]);
		}
		if(0==strcmp(argv[i], "-mT")) singleTask=true;
		if(0==strcmp(argv[i], "-fM")) functionMode=true;
		if(0==strcmp(argv[i], "-kod")) keepOldData=true;
		if(0==strcmp(argv[i], "-maxCJ")) maxNumberOfConcurrentJob=atoi(argv[i+1]);
		if((0==strcmp(argv[i], "-age")) && (i+1 < argc))
		{
			maxFileAge=atof(argv[i+1]);
		}
		if(0==strcmp(argv[i], "-p")) port=atoi(argv[i+1]);
	}

#ifdef WITH_VERSION_CONTROL
	
	std::string gitAdress=GIT_URL;
	gitAdress=gitAdress.substr(0, gitAdress.size()-4);

	CURL *curl;
	CURLcode res;
	char url[2048];
	sprintf(url,"%s/raw/master/version",gitAdress.c_str());
	curl = curl_easy_init();                                                                                                                                                                                                                                                           
	if (curl)
	{
		curl_easy_setopt(curl, CURLOPT_URL, url);
		std::string resultBody{ };
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resultBody);
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, static_cast<size_t ( *)(char*, size_t, size_t, void*)>(
			[](char* ptr, size_t size, size_t nmemb, void* resultBody){
				*(static_cast<std::string*>(resultBody)) += std::string {ptr, size * nmemb};
				return size * nmemb;
			}
		));
		res = curl_easy_perform(curl);
		if(res == CURLE_OK)
		{
			if(resultBody.size()<20 && currentVersion.compare(resultBody)<0){
				fprintf(stdout, "The new version %s is avialable on GitHub: %s \n",resultBody.c_str(), gitAdress.c_str() );
				fprintf(stdout, "The version %s is currently installed\n",currentVersion.c_str() );
			}
		}
		curl_easy_cleanup(curl);
	}


#endif

	//run daemon
	if(runAsDaemon ){
		fprintf(stderr, "start daemon\n" );
		bool stayInCurrentDirectory=true;
		bool dontRedirectStdIO=false;
		int value=daemon(stayInCurrentDirectory,dontRedirectStdIO);
		if(value==-1) fprintf(stderr, "fail to run daemon\n" );
		else fprintf(stderr, "Daemon started\n" );
	}
	
	// start soket reading
	bool needToStop=false;
	jobArray jobIds;
	// fill error array
	for (int i = 0; i < 1000; ++i)
	{
		jobIds.errorsByJobId.push_back({0,0});
		jobIds.errorsByPid.push_back({0,0});
	}

	mkdir("/tmp/G2S", 0777);

	mkdir("/tmp/G2S/data", 0777);
	mkdir("/tmp/G2S/logs", 0777);

	std::thread fileCleaningThread([&] {
		time_t last;
		time(&last);
		removeAllFile((char*)"/tmp/G2S/data",( keepOldData ? maxFileAge : 0));
		removeAllFile((char*)"/tmp/G2S/logs",( keepOldData ? maxFileAge : 0));
		while(!needToStop) {
			time_t now;
			time(&now);
		    if(difftime(now,last)>std::max(maxFileAge/100,10.)){
		    	removeAllFile((char*)"/tmp/G2S/data",maxFileAge);
				removeAllFile((char*)"/tmp/G2S/logs",maxFileAge);
				last=now;
		    }else{
		    	std::this_thread::sleep_for(std::chrono::seconds(1));
		    }
		}
	});

	//  Prepare our context and socket
#if __EMSCRIPTEN__
	wsZmq::socket_t receiver(ZMQ_REP);
#else
	zmq::context_t context (1);
	zmq::socket_t receiver(context,ZMQ_REP);
#endif
	
	char address[1024];
	sprintf(address,"tcp://*:%d",port);

	try {
		receiver.bind(address);
	}
	catch(const std::exception& e) {
		std::cerr << e.what() << '\n';
		needToStop=true;
	}

#ifndef __EMSCRIPTEN__
	if(!std::isnan(timeoutDuration)) {
		int timeout=int(timeoutDuration*1000);
		receiver.set(zmq::sockopt::linger, timeout);
		receiver.set(zmq::sockopt::rcvtimeo, timeout);
		receiver.set(zmq::sockopt::sndtimeo, timeout);
	}
#endif

	jobQueue jobQueue;

	while (!needToStop) {
		zmq::message_t request;
		bool newRequest=false;
		//  Wait for next request from client
		auto reciveMessage=receiver.recv(request,zmq::recv_flags::dontwait);
		if( reciveMessage )
		{
			newRequest=true;
			size_t requesSize=request.size();
			if(requesSize>=sizeof(infoContainer)){
				infoContainer infoRequest;
				memcpy(&infoRequest,request.data(),sizeof(infoContainer));
				if(infoRequest.version<=0) continue;
				switch(infoRequest.task)
				{
					case EXIST :
						{
							int type=dataIsPresent((char*)request.data()+sizeof(infoContainer));
							zmq::message_t reply(sizeof(type));
							memcpy (reply.data (), &type, sizeof(type));
							receiver.send(reply,zmq::send_flags::none);
							break;
						}
					case UPLOAD :
						{
							int error=storeData((char*)request.data()+sizeof(infoContainer), requesSize-sizeof(infoContainer), infoRequest.task != UPLOAD, true);
							zmq::message_t reply(sizeof(error));
							memcpy (reply.data (), &error, sizeof(error));
							receiver.send(reply,zmq::send_flags::none);
							break;
						}
					case DOWNLOAD :
						{
							zmq::message_t answer=sendData((char*)request.data()+sizeof(infoContainer));
							receiver.send(answer,zmq::send_flags::none);
							break;
						}
					case JOB :
						{
							int id=recieveJob(jobQueue,(char*)request.data()+sizeof(infoContainer), requesSize-sizeof(infoContainer));
							zmq::message_t reply(sizeof(id));
							memcpy (reply.data (), &id, sizeof(id));
							receiver.send(reply,zmq::send_flags::none);
							break;
						}
					case PROGESSION :
						{
							int progess=lookForStatus((char*)request.data()+sizeof(infoContainer),requesSize-sizeof(infoContainer));
							zmq::message_t reply(sizeof(progess));
							memcpy (reply.data (), &progess, sizeof(progess));
							receiver.send(reply,zmq::send_flags::none);
							break;
						}
					case JOB_STATUS :
						{

							jobIdType jobId;
							memcpy(&jobId,(char*)request.data()+sizeof(infoContainer),sizeof(jobId));
							int error=statusJobs(jobIds,jobQueue,jobId);
							zmq::message_t reply(sizeof(error));
							memcpy (reply.data (), &error, sizeof(error));
							receiver.send(reply,zmq::send_flags::none);
							break;
						}
					case DURATION :
						{
							int progess=lookForDuration((char*)request.data()+sizeof(infoContainer),requesSize-sizeof(infoContainer));
							zmq::message_t reply(sizeof(progess));
							memcpy (reply.data (), &progess, sizeof(progess));
							receiver.send(reply,zmq::send_flags::none);
							break;
						}
					case KILL :
						{
							fprintf(stderr, "%s\n", "recieve KILL");
							jobIdType jobId;
							memcpy(&jobId,(char*)request.data()+sizeof(infoContainer),sizeof(jobId));
							recieveKill(jobIds,jobQueue,jobId);
							int error=0;
							zmq::message_t reply(sizeof(error));
							memcpy (reply.data (), &error, sizeof(error));
							receiver.send(reply,zmq::send_flags::none);
							break;
						}
					case UPLOAD_JSON :
						{
							int error=storeJson((char*)request.data()+sizeof(infoContainer), requesSize-sizeof(infoContainer), infoRequest.task != UPLOAD, true);
							zmq::message_t reply(sizeof(error));
							memcpy (reply.data (), &error, sizeof(error));
							receiver.send(reply,zmq::send_flags::none);
							break;
						}
					case DOWNLOAD_JSON :
						{
							zmq::message_t answer=sendJson((char*)request.data()+sizeof(infoContainer));
							receiver.send(answer,zmq::send_flags::none);
							break;
						}
					case DOWNLOAD_TEXT :
						{
							zmq::message_t answer=sendText((char*)request.data()+sizeof(infoContainer));
							receiver.send(answer,zmq::send_flags::none);
							break;
						}
					case SHUTDOWN :
						{
							needToStop=true;
							int error=0;
							zmq::message_t reply(sizeof(error));
							memcpy (reply.data (), &error, sizeof(error));
							receiver.send(reply,zmq::send_flags::none);
							break;
						}
					case SERVER_STATUS :
						{
							int status=SERVER_TYPE+1; // (1 everything is ok)
							zmq::message_t reply(sizeof(status));
							memcpy (reply.data (), &status, sizeof(status));
							receiver.send(reply,zmq::send_flags::none);
							break;
						}
				}
			}
		}
		if(cleanJobs(jobIds) || newRequest){
			runJobInQueue(jobQueue, jobIds, singleTask, functionMode, maxNumberOfConcurrentJob);
		}else{
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
	}
	fileCleaningThread.join();

	return 0;
 }