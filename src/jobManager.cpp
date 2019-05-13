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

#include "jobManager.hpp"


void recieveKill(jobArray &jobIds, jobIdType jobId ){
	pid_t pid=jobIds.look4pid[jobId];
	kill(pid, SIGTERM);

	bool died = false;
	for (int loop=0; !died && loop < 6 ; ++loop)
	{
		int status;
		usleep(300);
		if (waitpid(pid, &status, WNOHANG) == pid) died = true;
	}

	if (!died) kill(pid, SIGKILL);
	jobIds.look4jobId.erase(pid);
	jobIds.look4pid.erase(jobId);
}

bool cleanJobs(jobArray &jobIds){
	bool cleanSomething=false;
	pid_t pid;
	int status=0;
	pid = waitpid(-1,  &status, WNOHANG);
	while ( pid > 0 ){
		jobIdType localJobId =jobIds.look4jobId[pid];
		jobIds.look4pid.erase(localJobId);
		jobIds.look4jobId.erase(pid);
		cleanSomething=true;

		if(WIFEXITED(status)){ // "normal exit"
			int exitCode=WEXITSTATUS(status);
			if(exitCode!=0)
			{	//store bad Pid
				jobIds.errorsByPid.push_back ( std::pair<pid_t, int >(pid, exitCode) );
				jobIds.errorsByJobId.push_back ( std::pair<jobIdType, int >(localJobId, exitCode) );
				jobIds.errorsByJobId.pop_front();
				jobIds.errorsByPid.pop_front();
			}
		}else{
			if(WIFSIGNALED(status)){ // "prematurated exit"
				int exitCode=256+WTERMSIG(status);
				jobIds.errorsByPid.push_back ( std::pair<pid_t, int >(pid, exitCode) );
				jobIds.errorsByJobId.push_back ( std::pair<jobIdType, int >(localJobId, exitCode) );
				jobIds.errorsByJobId.pop_front();
				jobIds.errorsByPid.pop_front();
			}
		}
		pid = waitpid(-1, &status, WNOHANG);
	}
	return cleanSomething;
}

int statusJobs(jobArray &jobIds, jobIdType jobId){
	cleanJobs(jobIds);
	if(jobIds.look4pid.find(jobId)!=jobIds.look4pid.end())
		return -1;
	else{
		for (auto it = jobIds.errorsByJobId.cbegin(); it != jobIds.errorsByJobId.cend(); it++)
		{
			if(it->first ==jobId)
				return it->second;
		}
	}
	return 0;
}


 