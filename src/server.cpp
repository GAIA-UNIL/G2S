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
#include <cerrno>
#include <limits>
#include <unistd.h> /* for fork */
#include <sys/types.h> /* for pid_t */
#include <sys/wait.h> /* for wait */
#include <sys/stat.h>
#include <thread>
#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
  #define WITH_FILESYSTEM_INCLUDE 1
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem> 
  namespace fs = std::experimental::filesystem;
  #define WITH_FILESYSTEM_INCLUDE 1
#endif

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

bool ensureRuntimeDirectory(const char* path)
{
	const mode_t runtimeMode=0770;
	if(mkdir(path, runtimeMode) != 0 && errno != EEXIST) {
		fprintf(stderr, "Could not create runtime directory %s: %s\n", path, strerror(errno));
		return false;
	}

	struct stat info;
	if(lstat(path, &info) != 0) {
		fprintf(stderr, "Could not inspect runtime directory %s: %s\n", path, strerror(errno));
		return false;
	}

	if(!S_ISDIR(info.st_mode) || S_ISLNK(info.st_mode)) {
		fprintf(stderr, "Runtime path %s exists but is not a directory\n", path);
		return false;
	}

	if(chmod(path, runtimeMode) != 0) {
		fprintf(stderr, "Could not set permissions on runtime directory %s: %s\n", path, strerror(errno));
		return false;
	}

	return true;
}

 
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
		snprintf(completeName,2048,"%s/%s",dir,de->d_name);
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
	unsigned port=8128;
	unsigned maxNumberOfConcurrentJob=500;
	bool moveToServerFolder=true;
	bool allowUnregisteredAlgorithm=false;

	auto parseUnsignedOption = [&](const char* option, int &i, unsigned minimum, unsigned maximum, unsigned &destination) {
		if(i+1 >= argc) {
			fprintf(stderr, "Missing value for %s\n", option);
			return false;
		}

		const char* value=argv[i+1];
		if(value[0] == '\0' || value[0] == '-') {
			fprintf(stderr, "Invalid value for %s: %s\n", option, value);
			return false;
		}

		char* end=nullptr;
		errno=0;
		unsigned long parsed=strtoul(value, &end, 10);
		if(errno == ERANGE || end == value || *end != '\0' || parsed < minimum || parsed > maximum) {
			fprintf(stderr, "Invalid value for %s: %s (expected %u-%u)\n", option, value, minimum, maximum);
			return false;
		}

		destination=static_cast<unsigned>(parsed);
		++i;
		return true;
	};

	
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
		if(0==strcmp(argv[i], "-maxCJ") && !parseUnsignedOption("-maxCJ", i, 1, std::numeric_limits<unsigned>::max(), maxNumberOfConcurrentJob)) return EXIT_FAILURE;
		if((0==strcmp(argv[i], "-age")) && (i+1 < argc))
		{
			maxFileAge=atof(argv[i+1]);
		}
		if(0==strcmp(argv[i], "-p") && !parseUnsignedOption("-p", i, 1, 65535, port)) return EXIT_FAILURE;
		if(0==strcmp(argv[i], "-kcwd")) moveToServerFolder=false;
		if((0==strcmp(argv[i], "--allow-unregistered-algorithms")) || (0==strcmp(argv[i], "-allowUnregisteredAlgorithm"))) allowUnregisteredAlgorithm=true;
	}
	#ifdef WITH_FILESYSTEM_INCLUDE
	if (moveToServerFolder)
	{
		fs::path exe_path(argv[0]);
		std::string exe_dir = exe_path.parent_path().string();
		fs::current_path(exe_dir);
	}
	#else
	fprintf(stderr, "This assumes you run the server from the server folder using: ./server ... ");
	#endif



#ifdef WITH_VERSION_CONTROL
	
	std::string gitAdress=GIT_URL;
	gitAdress=gitAdress.substr(0, gitAdress.size()-4);

	CURL *curl;
	CURLcode res;
	char url[2048];
	snprintf(url,2048,"%s/raw/master/version",gitAdress.c_str());
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

	if(!ensureRuntimeDirectory("/tmp/G2S") ||
	   !ensureRuntimeDirectory("/tmp/G2S/data") ||
	   !ensureRuntimeDirectory("/tmp/G2S/logs")) {
		return 1;
	}

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
	snprintf(address,1024,"tcp://*:%d",port);

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

	auto sendIntReply = [&](int value) {
		zmq::message_t reply(sizeof(value));
		memcpy(reply.data(), &value, sizeof(value));
		receiver.send(reply,zmq::send_flags::none);
	};

	auto payloadSizeIs = [&](size_t requestSize, size_t expected) {
		return requestSize == sizeof(infoContainer)+expected;
	};

	auto payloadSizeBetween = [&](size_t requestSize, size_t minimum, size_t maximum) {
		if(requestSize < sizeof(infoContainer)) return false;
		size_t payloadSize=requestSize-sizeof(infoContainer);
		return payloadSize >= minimum && payloadSize <= maximum;
	};

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
				if(infoRequest.version<=0) {
					sendIntReply(-1);
					continue;
				}
				switch(infoRequest.task)
				{
					case EXIST :
						{
							if(!payloadSizeIs(requesSize, 64)) {
								sendIntReply(-1);
								break;
							}
							int type=dataIsPresent((char*)request.data()+sizeof(infoContainer));
							sendIntReply(type);
							break;
						}
					case UPLOAD :
						{
							if(!payloadSizeBetween(requesSize, 64+sizeof(size_t)+sizeof(unsigned)*3, 64+g2s::DataImage::MaxSerializedBytes)) {
								sendIntReply(-1);
								break;
							}
							int error=storeData((char*)request.data()+sizeof(infoContainer), requesSize-sizeof(infoContainer), infoRequest.task != UPLOAD, true);
							sendIntReply(error);
							break;
						}
					case DOWNLOAD :
						{
							if(!payloadSizeIs(requesSize, 64)) {
								receiver.send(zmq::message_t(0),zmq::send_flags::none);
								break;
							}
							zmq::message_t answer=sendData((char*)request.data()+sizeof(infoContainer));
							receiver.send(answer,zmq::send_flags::none);
							break;
						}
					case JOB :
						{
							int id=recieveJob(jobQueue,(char*)request.data()+sizeof(infoContainer), requesSize-sizeof(infoContainer), allowUnregisteredAlgorithm);
							zmq::message_t reply(sizeof(id));
							memcpy (reply.data (), &id, sizeof(id));
							receiver.send(reply,zmq::send_flags::none);
							break;
						}
					case PROGESSION :
						{
							if(!payloadSizeIs(requesSize, sizeof(jobIdType))) {
								sendIntReply(-1);
								break;
							}
							int progess=lookForStatus((char*)request.data()+sizeof(infoContainer),requesSize-sizeof(infoContainer));
							sendIntReply(progess);
							break;
						}
					case JOB_STATUS :
						{
							if(!payloadSizeIs(requesSize, sizeof(jobIdType))) {
								sendIntReply(-1);
								break;
							}

							jobIdType jobId;
							memcpy(&jobId,(char*)request.data()+sizeof(infoContainer),sizeof(jobId));
							int error=statusJobs(jobIds,jobQueue,jobId);
							sendIntReply(error);
							break;
						}
					case DURATION :
						{
							if(!payloadSizeIs(requesSize, sizeof(jobIdType))) {
								sendIntReply(-1);
								break;
							}
							int progess=lookForDuration((char*)request.data()+sizeof(infoContainer),requesSize-sizeof(infoContainer));
							sendIntReply(progess);
							break;
						}
					case KILL :
						{
							fprintf(stderr, "%s\n", "recieve KILL");
							int error=-1;
							if(payloadSizeIs(requesSize, sizeof(jobIdType))){
								jobIdType jobId;
								memcpy(&jobId,(char*)request.data()+sizeof(infoContainer),sizeof(jobId));
								error=recieveKill(jobIds,jobQueue,jobId);
							}
							sendIntReply(error);
							break;
						}
					case UPLOAD_JSON :
						{
							if(!payloadSizeBetween(requesSize, 65, 64+g2s::DataImage::MaxSerializedBytes)) {
								sendIntReply(-1);
								break;
							}
							int error=storeJson((char*)request.data()+sizeof(infoContainer), requesSize-sizeof(infoContainer), infoRequest.task != UPLOAD, false);
							sendIntReply(error);
							break;
						}
					case DOWNLOAD_JSON :
						{
							if(!payloadSizeIs(requesSize, 64)) {
								receiver.send(zmq::message_t(0),zmq::send_flags::none);
								break;
							}
							zmq::message_t answer=sendJson((char*)request.data()+sizeof(infoContainer));
							receiver.send(answer,zmq::send_flags::none);
							break;
						}
					case DOWNLOAD_TEXT :
						{
							if(!payloadSizeIs(requesSize, 64)) {
								receiver.send(zmq::message_t(0),zmq::send_flags::none);
								break;
							}
							zmq::message_t answer=sendText((char*)request.data()+sizeof(infoContainer));
							receiver.send(answer,zmq::send_flags::none);
							break;
						}
					case SHUTDOWN :
						{
							needToStop=true;
							int error=0;
							sendIntReply(error);
							break;
						}
					case SERVER_STATUS :
						{
							int status=SERVER_TYPE+1; // (1 everything is ok)
							sendIntReply(status);
							break;
						}
					default:
						{
							sendIntReply(-1);
							break;
						}
				}
			}else{
				sendIntReply(-1);
			}
		}
		if(cleanJobs(jobIds) || newRequest){
			runJobInQueue(jobQueue, jobIds, singleTask, functionMode, maxNumberOfConcurrentJob, allowUnregisteredAlgorithm);
		}else{
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
	}
	fileCleaningThread.join();

	return 0;
 }
