#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> /* for fork */
#include <sys/types.h> /* for pid_t */
#include <sys/wait.h> /* for wait */
#include <sys/stat.h>

#include <zmq.hpp>
#if __EMSCRIPTEN__
#include "wsZmq.hpp"
#endif

#include "protocol.hpp"

#include "jobTasking.hpp"
#include "jobManager.hpp"
#include "dataManagement.hpp"
#include "status.hpp"

		
int main(int argc, char const *argv[]) {

	bool runAsDaemon=false;
	bool singleTask=false;
	bool functionMode=false;

	
	for (int i = 1; i < argc; ++i)
	{
		if(0==strcmp(argv[i], "-d")) runAsDaemon=true;
		if(0==strcmp(argv[i], "-mT")) singleTask=true;
		if(0==strcmp(argv[i], "-fM")) functionMode=true;
	}

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
	mkdir("./data", 0777);
	mkdir("./logs", 0777);

	//  Prepare our context and socket
#if __EMSCRIPTEN__
	wsZmq::socket_t receiver(ZMQ_REP);
#else
	zmq::context_t context (1);
	zmq::socket_t receiver(context,ZMQ_REP);
#endif
	receiver.bind ("tcp://*:8128");

	while (!needToStop) {
		zmq::message_t request;

        //  Wait for next request from client
		receiver.recv (&request);
		size_t requesSize=request.size();
		if(requesSize>=sizeof(infoContainer)){
			infoContainer infoRequest;
			memcpy(&infoRequest,request.data(),sizeof(infoContainer));
			fprintf(stderr, "receive %d\n",infoRequest.task);
			if(infoRequest.version<=0) continue;
			switch(infoRequest.task)
			{
				case EXIST :
					{
						int error=dataIsPresent((char*)request.data()+sizeof(infoContainer));
						zmq::message_t reply(sizeof(error));
						memcpy (reply.data (), &error, sizeof(error));
						receiver.send(reply);
						break;
					}
				case UPLOAD :
					{
						int error=storeData((char*)request.data()+sizeof(infoContainer), requesSize-sizeof(infoContainer), infoRequest.task != UPLOAD, false);
						zmq::message_t reply(sizeof(error));
						memcpy (reply.data (), &error, sizeof(error));
						receiver.send(reply);
					break;
					}
				case DOWNLOAD :
					{
						zmq::message_t answer=sendData((char*)request.data()+sizeof(infoContainer));
						receiver.send(answer);
						break;
					}
				case JOB :
					{
						int id=recieveJob(jobIds,(char*)request.data()+sizeof(infoContainer), requesSize-sizeof(infoContainer),singleTask, functionMode);
						//fprintf(stderr, "send answer job\n");
						zmq::message_t reply(sizeof(id));
						memcpy (reply.data (), &id, sizeof(id));
						receiver.send(reply);
						//fprintf(stderr, "answer job sended\n");
						break;
					}
				case STATUS :
					{
						int progess=lookForStatus((char*)request.data()+sizeof(infoContainer),requesSize-sizeof(infoContainer));
						zmq::message_t reply(sizeof(progess));
						memcpy (reply.data (), &progess, sizeof(progess));
						receiver.send(reply);
						break;
					}
				case KILL :
					{
						fprintf(stderr, "%s\n", "recieve KILL");
						jobIdType jobId;
						memcpy(&jobId,(char*)request.data()+sizeof(infoContainer),sizeof(jobId));
						recieveKill(jobIds,jobId);
						int error=0;
						zmq::message_t reply(sizeof(error));
						memcpy (reply.data (), &error, sizeof(error));
						receiver.send(reply);
						break;
					}
				case SHUTDOWN :
					{
						needToStop=true;
						int error=0;
						zmq::message_t reply(sizeof(error));
						memcpy (reply.data (), &error, sizeof(error));
						receiver.send(reply);
						break;
					}
			}
		}
    }

    return 0;
 }