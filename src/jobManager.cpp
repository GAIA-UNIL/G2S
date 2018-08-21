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

void cleanJobs(jobArray &jobIds){
	
	pid_t pid;
	pid = waitpid(-1, NULL, WNOHANG);
	while ( pid > 0 ){
		jobIds.look4pid.erase(jobIds.look4jobId[pid]);
		jobIds.look4jobId.erase(pid);
		pid = waitpid(-1, NULL, WNOHANG);
	}
}


 