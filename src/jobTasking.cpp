#include "jobTasking.hpp"
#include <cstring>
#include <thread>
#include <dlfcn.h>

void call_functionMode(jobArray &jobIds, bool singleTask, jobIdType uniqueId, const char* functionName, int index, char** argv){
#ifndef EMSCRIPTEN
	try{
#endif
		std::vector<std::string> argvV(argv, argv + index);
		std::string functionNameStr(functionName);
#ifndef EMSCRIPTEN
		std::thread myThread([argvV,functionNameStr,uniqueId]()
#endif
		{
			int index=argvV.size();
			
			std::vector<char*> cstrings;
		    for(size_t i = 0; i < argvV.size(); ++i)
				cstrings.push_back(const_cast<char*>(argvV[i].c_str()));

			char** argv=cstrings.data();

			void *handle=nullptr;
			char *error=nullptr;
			bool isOk=true;
			if(isOk)
				handle = dlopen((std::string("./")+functionNameStr+std::string(".so")).c_str(), RTLD_LAZY);
			if (!handle) {
				fputs (dlerror(), stderr);
				//exit(1);
				isOk=false;
			}else{
				//fprintf(stderr, "before creating main\n");
				int (*main_lib)(int , char **) = (int (*)(int , char **))dlsym(handle, "main");
				if ((error = dlerror()) != NULL)  {
					fputs(error, stderr);
					//exit(1);
					isOk=false;
				}
				//fprintf(stderr, "before calling main\n");
				if(isOk)main_lib(index,(char**)argv);
				dlclose(handle);
			}
		}
#ifndef EMSCRIPTEN
		);

		if(singleTask) {
			myThread.join();
		}
		else {
			myThread.detach();
		}
#endif

		//jobIds.threads[uniqueId]=&myThread;
		
		//jobIds.threads[uniqueId];
		//std::swap(myThread,jobIds.threads[uniqueId]);
		
#ifndef EMSCRIPTEN
	} catch (const std::exception& e) {
		fprintf(stderr, "qs crash: %s\n", e.what());
	}
#endif
}

jobIdType echo_call(jobArray &jobIds, Json::Value job, bool singleTask, bool functionMode)
{
	jobIdType uniqueId=std::chrono::high_resolution_clock::now().time_since_epoch().count();
	cleanId(jobIds, uniqueId);
	if(job.isMember("Parameter")){
		Json::Value param=job["Parameter"];
		if(param.isMember("-ti")){
			
			char* argv[100];
			argv[0]=(char*)"./echo";
			for (int i = 1; i < 100; ++i)
			{
				argv[i]=nullptr;
			}
			//init defualt
			Json::Value::Members member=param.getMemberNames();
			int index=1;
			for (int i = 0; i < member.size(); ++i)
			{
				argv[index]=(char *)member[i].c_str();
				//fprintf(stderr, "%s\n", argv[index]);
				index++;
				if(param[member[i]].isArray()){
					for (int j = 0; j < param[member[i]].size(); ++j)
					{
						argv[index]=(char *)param[member[i]][j].asCString();
						index++;
					}
				}else{
					argv[index]=(char *)param[member[i]].asCString();
					//fprintf(stderr, "%s\n", argv[index]);
					index++;
				}
			}
			{
				argv[index]=(char*)"-r";
				//fprintf(stderr, "%s\n", argv[index]);
				index++;
				char buffer [128];
				snprintf(buffer, sizeof(buffer), "logs/%u.log", uniqueId);
				argv[index]=buffer;
				//fprintf(stderr, "%s\n", argv[index]);
				index++;
			}

			/*// add specific
			for (int i = 0; i < index; ++i)
			{
				fprintf(stderr, "%s ", argv[i]);
			}
			fprintf(stderr, "\n" );*/

			//Execute
			if(functionMode)
			{
				call_functionMode(jobIds, singleTask,  uniqueId, "echo",index, argv);
				
			}else{
				pid_t pid=fork(); // fork to be crash resistant !!
				if (pid==0) { // child process //
					
					execv("./echo", argv);
					exit(127); // only if execv fails //
				}
				else { // pid!=0; parent process //
					jobIds.look4pid.insert ( std::pair<jobIdType, pid_t >(uniqueId,pid) );
					jobIds.look4jobId.insert ( std::pair<pid_t, jobIdType >(pid,uniqueId) );
					if(singleTask)waitpid(pid,0,0); // wait for child to exit //
				}
			}
		}else{
			fprintf(stderr, "-ti needed\n");
		}
	}else{
		fprintf(stderr, "Parameter is mandatory for QS\n");
	}
	return uniqueId;
}

jobIdType test_call(jobArray &jobIds, Json::Value job, bool singleTask, bool functionMode)
{
	jobIdType uniqueId=std::chrono::high_resolution_clock::now().time_since_epoch().count();
	cleanId(jobIds, uniqueId);
	if(job.isMember("Parameter")){
		Json::Value param=job["Parameter"];
		if(param.isMember("-ti")){
			
			char* argv[100];
			argv[0]=(char*)"./test";
			for (int i = 1; i < 100; ++i)
			{
				argv[i]=nullptr;
			}
			//init defualt
			Json::Value::Members member=param.getMemberNames();
			int index=1;
			for (int i = 0; i < member.size(); ++i)
			{
				argv[index]=(char *)member[i].c_str();
				//fprintf(stderr, "%s\n", argv[index]);
				index++;
				argv[index]=(char *)param[member[i]].asCString();
				//fprintf(stderr, "%s\n", argv[index]);
				index++;
			}
			{
				argv[index]=(char*)"-r";
				//fprintf(stderr, "%s\n", argv[index]);
				index++;
				char buffer [128];
				snprintf(buffer, sizeof(buffer), "logs/%u.log", uniqueId);
				argv[index]=buffer;
				//fprintf(stderr, "%s\n", argv[index]);
				index++;
			}

			// add specific

			//Execute
			if(functionMode)
			{
				call_functionMode(jobIds, singleTask,  uniqueId, "test",index, argv);
				
			}else{
				pid_t pid=fork(); // fork to be crash resistant !!
				if (pid==0) { // child process //
					
					execv("./test", argv);
					exit(127); // only if execv fails //
				}
				else { // pid!=0; parent process //
					jobIds.look4pid.insert ( std::pair<jobIdType, pid_t >(uniqueId,pid) );
					jobIds.look4jobId.insert ( std::pair<pid_t, jobIdType >(pid,uniqueId) );
					if(singleTask)waitpid(pid,0,0); // wait for child to exit //
				}
			}
		}else{
			fprintf(stderr, "-ti needed\n");
		}
	}else{
		fprintf(stderr, "Parameter is mandatory for QS\n");
	}
	return uniqueId;
}

jobIdType qs_call(jobArray &jobIds, Json::Value job, bool singleTask, bool functionMode)
{
	jobIdType uniqueId=std::chrono::high_resolution_clock::now().time_since_epoch().count();
	cleanId(jobIds, uniqueId);
	if(job.isMember("Parameter")){
		Json::Value param=job["Parameter"];
		if(param.isMember("-ti") && (param.isMember("-di") || param.isMember("-ds") )){
			
			char* argv[100];
			char tempMemory[100][100];
			unsigned tempMemIndex=0;
			memset(tempMemory,0,100*100);
			argv[0]=(char*)"./qs";
			for (int i = 1; i < 100; ++i)
			{
				argv[i]=nullptr;
			}
			//init defualt
			Json::Value::Members member=param.getMemberNames();
			int index=1;
			for (int i = 0; i < member.size(); ++i)
			{
				argv[index]=(char *)member[i].c_str();
				//fprintf(stderr, "%s\n", argv[index]);
				index++;

				if(param[member[i]].isString())
				{
					strcpy(tempMemory[tempMemIndex], (char *)param[member[i]].asString().c_str());
					argv[index]=tempMemory[tempMemIndex];
					tempMemIndex++;
					//fprintf(stderr, "%s\n", argv[index]);
					index++;
				}
				if(param[member[i]].isArray())
				{
					//fprintf(stderr, "%s\n", "is array");
					Json::Value arrayData=param[member[i]];
					for (int j = 0; j < arrayData.size(); ++j)
					{
						if(arrayData[j].isString()){
							strcpy(tempMemory[tempMemIndex], arrayData[j].asCString());
							argv[index]=tempMemory[tempMemIndex];
							tempMemIndex++;
							//fprintf(stderr, "%s\n", argv[index]);
							index++;
						}
					}
				}
			}
			{
				argv[index]=(char*)"-r";
				//fprintf(stderr, "%s\n", argv[index]);
				index++;
				char buffer [128];
				snprintf(buffer, sizeof(buffer), "logs/%u.log", uniqueId);
				argv[index]=buffer;
				//fprintf(stderr, "%s\n", argv[index]);
				index++;
			}

			// add specific

			/*for (int i = 0; i < index; ++i)
			{
				fprintf(stderr, "%s ",argv[i] );
			}*/

			//Execute
			if(functionMode)
			{
				call_functionMode(jobIds, singleTask,  uniqueId, "qs",index, argv);
				
			}else{
				pid_t pid=fork(); // fork, to be crash resistant !!
				if (pid==0) { // child process //
					
					execv("./qs", argv);
					exit(127); // only if execv fails //
				}
				else { // pid!=0; parent process //
					jobIds.look4pid.insert ( std::pair<jobIdType, pid_t >(uniqueId,pid) );
					jobIds.look4jobId.insert ( std::pair<pid_t, jobIdType >(pid,uniqueId) );
					if(singleTask)waitpid(pid,0,0); // wait for child to exit //
				}
			}
		}else{
			fprintf(stderr, "-ti and (-di or -ds) is mandatory for QS\n");
		}
	}else{
		fprintf(stderr, "Parameter is mandatory for QS\n");
	}
	return uniqueId;
}

jobIdType ds_call(jobArray &jobIds, Json::Value job, bool singleTask, bool functionMode)
{
	jobIdType uniqueId=std::chrono::high_resolution_clock::now().time_since_epoch().count();
	cleanId(jobIds, uniqueId);
	if(job.isMember("Parameter")){
		Json::Value param=job["Parameter"];
		if(param.isMember("-ti") && (param.isMember("-di") || param.isMember("-ds") )){
			
			char* argv[100];
			argv[0]=(char*)"./ds";
			for (int i = 1; i < 100; ++i)
			{
				argv[i]=nullptr;
			}
			//init defualt
			Json::Value::Members member=param.getMemberNames();
			int index=1;
			for (int i = 0; i < member.size(); ++i)
			{
				argv[index]=(char *)member[i].c_str();
				//fprintf(stderr, "%s\n", argv[index]);
				index++;
				argv[index]=(char *)param[member[i]].asCString();
				//fprintf(stderr, "%s\n", argv[index]);
				index++;
			}
			{
				argv[index]=(char*)"-r";
				//fprintf(stderr, "%s\n", argv[index]);
				index++;
				char buffer [128];
				snprintf(buffer, sizeof(buffer), "logs/%u.log", uniqueId);
				argv[index]=buffer;
				//fprintf(stderr, "%s\n", argv[index]);
				index++;
			}

			// add specific

			//Execute
			if(functionMode)
			{
				call_functionMode(jobIds, singleTask,  uniqueId, "ds",index, argv);
				
			}else{
				pid_t pid=fork(); // fork to be crash resistant !!
				if (pid==0) { // child process //
					
					execv("./ds", argv);
					exit(127); // only if execv fails //
				}
				else { // pid!=0; parent process //
					jobIds.look4pid.insert ( std::pair<jobIdType, pid_t >(uniqueId,pid) );
					jobIds.look4jobId.insert ( std::pair<pid_t, jobIdType >(pid,uniqueId) );
					if(singleTask)waitpid(pid,0,0); // wait for child to exit //
				}
			}
		}else{
			fprintf(stderr, "-ti and (-di or -ds) is mandatory for DS\n");
		}
	}else{
		fprintf(stderr, "Parameter is mandatory for DS\n");
	}
	return uniqueId;
}


jobIdType dsl_call(jobArray &jobIds, Json::Value job, bool singleTask, bool functionMode)
{
	jobIdType uniqueId=std::chrono::high_resolution_clock::now().time_since_epoch().count();
	cleanId(jobIds, uniqueId);
	if(job.isMember("Parameter")){
		Json::Value param=job["Parameter"];
		if(param.isMember("-ti") && (param.isMember("-di") || param.isMember("-ds") )){
			
			char* argv[100];
			char tempMemory[100][100];
			unsigned tempMemIndex=0;
			memset(tempMemory,0,100*100);
			argv[0]=(char*)"./ds-l";
			for (int i = 1; i < 100; ++i)
			{
				argv[i]=nullptr;
			}
			//init defualt
			Json::Value::Members member=param.getMemberNames();
			int index=1;
			for (int i = 0; i < member.size(); ++i)
			{
				argv[index]=(char *)member[i].c_str();
				//fprintf(stderr, "%s\n", argv[index]);
				index++;

				if(param[member[i]].isString())
				{
					strcpy(tempMemory[tempMemIndex], (char *)param[member[i]].asString().c_str());
					argv[index]=tempMemory[tempMemIndex];
					tempMemIndex++;
					//fprintf(stderr, "%s\n", argv[index]);
					index++;
				}
				if(param[member[i]].isArray())
				{
					//fprintf(stderr, "%s\n", "is array");
					Json::Value arrayData=param[member[i]];
					for (int j = 0; j < arrayData.size(); ++j)
					{
						if(arrayData[j].isString()){
							strcpy(tempMemory[tempMemIndex], arrayData[j].asCString());
							argv[index]=tempMemory[tempMemIndex];
							tempMemIndex++;
							//fprintf(stderr, "%s\n", argv[index]);
							index++;
						}
					}
				}
			}
			{
				argv[index]=(char*)"-r";
				//fprintf(stderr, "%s\n", argv[index]);
				index++;
				char buffer [128];
				snprintf(buffer, sizeof(buffer), "logs/%u.log", uniqueId);
				argv[index]=buffer;
				//fprintf(stderr, "%s\n", argv[index]);
				index++;
			}

			// add specific

			/*for (int i = 0; i < index; ++i)
			{
				fprintf(stderr, "%s ",argv[i] );
			}*/

			//Execute
			if(functionMode)
			{
				call_functionMode(jobIds, singleTask,  uniqueId, "ds-l",index, argv);
				
			}else{
				pid_t pid=fork(); // fork, to be crash resistant !!
				if (pid==0) { // child process //
					
					execv("./ds-l", argv);
					exit(127); // only if execv fails //
				}
				else { // pid!=0; parent process //
					jobIds.look4pid.insert ( std::pair<jobIdType, pid_t >(uniqueId,pid) );
					jobIds.look4jobId.insert ( std::pair<pid_t, jobIdType >(pid,uniqueId) );
					if(singleTask)waitpid(pid,0,0); // wait for child to exit //
				}
			}
		}else{
			fprintf(stderr, "-ti and (-di or -ds) is mandatory for DS\n");
		}
	}else{
		fprintf(stderr, "Parameter is mandatory for DS\n");
	}
	return uniqueId;
}

jobIdType nds_call(jobArray &jobIds, Json::Value job, bool singleTask, bool functionMode)
{
	jobIdType uniqueId=std::chrono::high_resolution_clock::now().time_since_epoch().count();
	cleanId(jobIds, uniqueId);
	if(job.isMember("Parameter")){
		Json::Value param=job["Parameter"];
		if(param.isMember("-ti") && (param.isMember("-di") || param.isMember("-ds") )){
			
			char* argv[100];
			char tempMemory[100][100];
			unsigned tempMemIndex=0;
			memset(tempMemory,0,100*100);
			argv[0]=(char*)"./nds";
			for (int i = 1; i < 100; ++i)
			{
				argv[i]=nullptr;
			}
			//init defualt
			Json::Value::Members member=param.getMemberNames();
			int index=1;
			for (int i = 0; i < member.size(); ++i)
			{
				argv[index]=(char *)member[i].c_str();
				//fprintf(stderr, "%s\n", argv[index]);
				index++;

				if(param[member[i]].isString())
				{
					strcpy(tempMemory[tempMemIndex], (char *)param[member[i]].asString().c_str());
					argv[index]=tempMemory[tempMemIndex];
					tempMemIndex++;
					//fprintf(stderr, "%s\n", argv[index]);
					index++;
				}
				if(param[member[i]].isArray())
				{
					//fprintf(stderr, "%s\n", "is array");
					Json::Value arrayData=param[member[i]];
					for (int j = 0; j < arrayData.size(); ++j)
					{
						if(arrayData[j].isString()){
							strcpy(tempMemory[tempMemIndex], arrayData[j].asCString());
							argv[index]=tempMemory[tempMemIndex];
							tempMemIndex++;
							//fprintf(stderr, "%s\n", argv[index]);
							index++;
						}
					}
				}
			}
			{
				argv[index]=(char*)"-r";
				//fprintf(stderr, "%s\n", argv[index]);
				index++;
				char buffer [128];
				snprintf(buffer, sizeof(buffer), "logs/%u.log", uniqueId);
				argv[index]=buffer;
				//fprintf(stderr, "%s\n", argv[index]);
				index++;
			}

			// add specific

			/*for (int i = 0; i < index; ++i)
			{
				fprintf(stderr, "%s ",argv[i] );
			}*/

			//Execute
			if(functionMode)
			{
				call_functionMode(jobIds, singleTask,  uniqueId, "nds",index, argv);
				
			}else{
				pid_t pid=fork(); // fork, to be crash resistant !!
				if (pid==0) { // child process //
					
					execv("./nds", argv);
					exit(127); // only if execv fails //
				}
				else { // pid!=0; parent process //
					jobIds.look4pid.insert ( std::pair<jobIdType, pid_t >(uniqueId,pid) );
					jobIds.look4jobId.insert ( std::pair<pid_t, jobIdType >(pid,uniqueId) );
					if(singleTask)waitpid(pid,0,0); // wait for child to exit //
				}
			}
		}else{
			fprintf(stderr, "-ti and (-di or -ds) is mandatory for NDS\n");
		}
	}else{
		fprintf(stderr, "Parameter is mandatory for NDS\n");
	}
	return uniqueId;
}

jobIdType recieveJob(jobArray &jobIds,void* data, size_t sizeBuffer, bool singleTask, bool functionMode)
{
	jobIdType id=-1;
	Json::CharReaderBuilder builder;
  	Json::CharReader * reader = builder.newCharReader();
	Json::Value job;
	std::string errors;
	if(!reader->parse((const char*)data,(const char*)data+sizeBuffer,&job,&errors))
		fprintf(stderr,"%s\n", errors.c_str());
	//Json::Value::Members member=job.getMemberNames();

	/*for (int i = 0; i < member.size(); ++i)
	{
		fprintf(stderr, "%s\n", member[i].c_str());
	}*/

	if(job.isMember("Algorithm"))
	{	
		bool algoFounded=false;
		if(!strcmp(job["Algorithm"].asCString(),"Echo")) //echo
		{
			algoFounded=true;
			id=echo_call(jobIds, job,singleTask,functionMode);
		}
		if(!strcmp(job["Algorithm"].asCString(),"Test")) //test
		{
			algoFounded=true;
			id=test_call(jobIds, job,singleTask,functionMode);
		}
		if(!strcmp(job["Algorithm"].asCString(),"QucikSampling")) //QuickSampling
		{
			algoFounded=true;
			id=qs_call(jobIds, job,singleTask,functionMode);
		}
		if(!strcmp(job["Algorithm"].asCString(),"DirectSampling")) //DirectSampling
		{
			algoFounded=true;
			id=ds_call(jobIds, job,singleTask,functionMode);
		}
		if(!strcmp(job["Algorithm"].asCString(),"DirectSamplingLike")) //DirectSampling Like
		{
			algoFounded=true;
			id=dsl_call(jobIds, job,singleTask,functionMode);
		}
		if(!strcmp(job["Algorithm"].asCString(),"NarrawDistributionSelection")) // NDS
		{
			algoFounded=true;
			id=nds_call(jobIds, job,singleTask,functionMode);
		}
		if(!algoFounded)
			fprintf(stderr, "unknown aglorithm : %s\n", job["Algorithm"].asCString());
	}else{
		fprintf(stderr, "%s\n", "No Algorithm");
	}
	return id;
}