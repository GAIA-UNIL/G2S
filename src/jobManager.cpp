#include "jobManager.hpp"


void recieveKill(jobArray &jobIds, jobIdType jobId ){
	pid_t pid=jobIds.pids[jobId];
	kill(pid, SIGTERM);

	bool died = false;
	for (int loop=0; !died && loop < 6 ; ++loop)
	{
	    int status;
	    usleep(300);
	    if (waitpid(pid, &status, WNOHANG) == pid) died = true;
	}

	if (!died) kill(pid, SIGKILL);
}