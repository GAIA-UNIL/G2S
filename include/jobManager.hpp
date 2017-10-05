#ifndef JOB_MANAGER_HPP
#define JOB_MANAGER_HPP

#include <map>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>

typedef unsigned jobIdType;

struct jobArray{
	std::map<jobIdType, pid_t> pids;
};

void recieveKill(jobArray &jobIds, jobIdType jobId );

#endif // JOB_MANAGER_HPP