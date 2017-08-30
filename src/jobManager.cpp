#include "jobManager.hpp"


void recieveKill(jobArray &jobIds, jobIdType jobId ){
	pid_t pid=jobIds.pids[jobId];
	kill(pid, SIGTERM);

	bool died = false;
	for (int loop; !died && loop < 6 ; ++loop)
	{
	    int status;
	    pid_t id;
	    usleep(300);
	    if (waitpid(pid, &status, WNOHANG) == pid) died = true;
	}

	if (!died) kill(pid, SIGKILL);
}