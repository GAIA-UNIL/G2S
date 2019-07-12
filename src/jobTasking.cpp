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

#include "jobTasking.hpp"
#include <cstring>
#include <thread>
#include <dlfcn.h>
#include <numeric>


void call_functionMode(jobArray &jobIds, bool singleTask, jobIdType uniqueId, const char* functionName, int index, char** argv){
#ifndef EMSCRIPTEN
	try{
#endif
		std::vector<std::string> argvV(argv, argv + index);
		std::string functionNameStr(functionName);
#ifndef EMSCRIPTEN
		std::thread myThread([argvV,functionNameStr]()
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
				handle = dlopen((functionNameStr+std::string(".so")).c_str(), RTLD_LAZY);
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

jobIdType general_call(jobTask theJobTask, jobArray &jobIds, bool singleTask, bool functionMode)
{
	jobIdType uniqueId=std::get<0>(theJobTask);
	Json::Value job=std::get<1>(theJobTask);

	std::string algoStr=job["Algorithm"].asString();
	char algo[2048];
	strcpy(algo,algoStr.c_str());
	if(job.isMember("Parameter")){
		Json::Value param=job["Parameter"];
		if(param.isMember("-id")) {
			uniqueId=atoll(param["-id"][0].asCString());
		}

		char exeName[1024];
		sprintf(exeName,"./%s",algo);
		FILE *fp = fopen("./algosName.config", "r");
		std::vector<std::string>  listOfMendatory;
		if (fp){
			size_t sizeBuffer=2048;
			char* line=(char*)malloc(sizeBuffer);
			size_t readedSize;
			char sourceName[1024];
			char tagetName[1024];
			char requested[2048];
			char toAdd[2048];
			char extra[2048];
			char comment[2048];
			while((readedSize=getline(&line, &sizeBuffer, fp))){
				memset(sourceName,0,1024);memset(tagetName,0,1024);memset(requested,0,2048);memset(toAdd,0,2048);
				memset(requested,0,2048);memset(toAdd,0,2048);memset(extra,0,2048);memset(comment,0,2048);
				if((readedSize>1) && ((line[0]!='/') || (line[0]!='#'))){
					if(sscanf(line, "%s\t%s\t%s\t%s\t%s\t%s",sourceName,tagetName,requested,toAdd,extra,comment)>=2){
						if (strcmp(sourceName,algo)==0){
							strcpy(exeName,tagetName);
							char * pch;
							pch = strtok (requested,",");
							while (pch != NULL)
							{
								listOfMendatory.push_back(std::string(pch));
								pch = strtok (NULL, ",");
							}
							break;
						}
					}
				}
			}
			free(line);
			fclose(fp);
		}

		std::vector<std::string> missing;


		for (size_t i = 0; i < listOfMendatory.size(); ++i)
		{
			if(! param.isMember(listOfMendatory[i])){
				missing.push_back(listOfMendatory[i]);
			}
		}

		if(missing.size()==0){
			//init defualt
			Json::Value::Members member=param.getMemberNames();

			int sizeParamArray=50;

			for (size_t i = 0; i < member.size(); ++i)
			{
				sizeParamArray++;

				if(param[member[i]].isString())
				{
					sizeParamArray++;
				}
				if(param[member[i]].isArray())
				{
					Json::Value arrayData=param[member[i]];
					for (int j = 0; j < int(arrayData.size()); ++j)
					{
						if(arrayData[j].isString()){
							sizeParamArray++;
						}
					}
				}
			}

			int index=1;
			char* argv[sizeParamArray];
			for (int i = 0; i < sizeParamArray; ++i)
			{
				argv[i]=nullptr;
			}
			char tempMemory[sizeParamArray][100];
			unsigned tempMemIndex=0;
			memset(tempMemory,0,sizeParamArray*100);
			//fprintf(stderr, "%s\n", exeName);
			argv[index]=exeName;
			index++;
			
			for (size_t i = 0; i < member.size(); ++i)
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
					for (int j = 0; j < int(arrayData.size()); ++j)
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
				call_functionMode(jobIds, singleTask,  uniqueId, exeName, index, argv);
				
			}else{
				pid_t pid=fork(); // fork, to be crash resistant !!
				if (pid==0) { // child process //
					char *runVariable=argv[1];
					char *dot = strrchr(runVariable, '.');
					if (dot && !strcmp(dot, ".py"))
						argv[0]=(char *)"python3";
					if (dot && !strcmp(dot, ".sh"))
						argv[0]=(char *)"bash";
					if(argv[0])
					{
						execv(argv[0], argv);
					}
					else
					{
						execv(argv[1], argv+1);
					}
					exit(127); // only if execv fails //
				}
				else { // pid!=0; parent process //
					jobIds.look4pid.insert ( std::pair<jobIdType, pid_t >(uniqueId,pid) );
					jobIds.look4jobId.insert ( std::pair<pid_t, jobIdType >(pid,uniqueId) );
					if(singleTask)waitpid(pid,0,0); // wait for child to exit //
				}
			}
		}else{
			std::string s;
			if(missing.size()>0) s.append(missing[0]);
			fprintf(stderr, "%lu\n", missing.size());
			for (int i = 1; i < int(missing.size())-2; ++i)
			{
				s.append(std::string(", "));
				s.append(missing[i]);
			}
			if(missing.size()>1){
				s.append(std::string(" and "));
				s.append(missing[missing.size()-1]);
			}
			fprintf(stderr, "%s %s mandatory for %s\n",s.c_str(),(missing.size()>1 ? std::string("is") : std::string("are")).c_str(),algo);
			uniqueId=-1;
		}
	}else{
		fprintf(stderr, "Parameter is mandatory for QS\n");
		uniqueId=-1;
	}
	return uniqueId;
}

bool runJobInQueue(jobQueue &queue, jobArray &jobIds, bool singleTask, bool functionMode){
	bool runNewJob=false;
	if(queue.empty()) return false;

	if(jobIds.look4pid.empty()){
		general_call(queue.front(), jobIds, singleTask, functionMode);
		queue.pop_front();
		runNewJob=true;
	}else{
		for (size_t i = 0; i < queue.size(); ++i)
		{
			std::vector<jobIdType> jobDependency=std::get<2>(queue[i]);
			if(jobDependency.empty()){
				break;//make sens to me to se an unseted as a barrier
			}else{
				bool toRun=true;
				for (size_t j = 0; (j < jobDependency.size()) && toRun; ++j)
				{
					if(jobIds.look4pid.count(jobDependency[j])>0)toRun=false;
					for (size_t k = 0; k < i; ++k)
					{
						if(std::get<0>(queue[k])==jobDependency[j])toRun=false;
					}
				}
				if (toRun)
				{
					general_call(queue[i], jobIds, singleTask, functionMode);
					queue.erase(queue.begin()+i);
					runNewJob=true;
				}
			}
		}
	}
	return runNewJob;
}


jobIdType stackJob(Json::Value job,jobQueue &queue){
	jobIdType uniqueId=std::chrono::high_resolution_clock::now().time_since_epoch().count();
	if(job.isMember("Parameter")){
		Json::Value param=job["Parameter"];
		if(param.isMember("-id")) {
			uniqueId=atoll(param["-id"][0].asCString());
		}
		std::vector<jobIdType> jobDependency;
		if(job.isMember("Dependency") && job["Dependency"].isArray()){
			Json::Value dep=job["Dependency"];
			for (int i = 0; i < int(dep.size()); ++i)
			{
				if(dep[i].isUInt())
				jobDependency.push_back(dep[i].asUInt());
			}
		}
		queue.push_back({uniqueId, job, jobDependency});
	}else{
		fprintf(stderr, "Parameter is mandatory for QS\n");
		uniqueId=-1;
	}
	return uniqueId;	
}

jobIdType recieveJob(jobQueue &queue,void* data, size_t sizeBuffer)
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
		id=stackJob( job, queue);	
	}else{
		fprintf(stderr, "%s\n", "No Algorithm");
	}
	return id;
}