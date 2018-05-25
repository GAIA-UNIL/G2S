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


 