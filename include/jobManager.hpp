#ifndef JOB_MANAGER_HPP
#define JOB_MANAGER_HPP

#include <map>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>

typedef unsigned jobIdType;

struct jobArray{
	std::map<jobIdType, pid_t> look4pid;
	std::map<pid_t,jobIdType> look4jobId;
};

void recieveKill(jobArray &jobIds, jobIdType jobId );
void cleanJobs(jobArray &jobIds);
void cleanId(jobArray &jobIds, jobIdType id);

#endif // JOB_MANAGER_HPP