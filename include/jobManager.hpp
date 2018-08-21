/*
 * G2S (c) by Mathieu Gravey (gravey.mathieu@gmail.com)
 * 
 * G2S is licensed under a
 * Creative Commons Attribution-NonCommercial 4.0 International License.
 * 
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
 */

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

#endif // JOB_MANAGER_HPP