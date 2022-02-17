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

#ifndef JOB_MANAGER_HPP
#define JOB_MANAGER_HPP

#include <map>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <stack>
#include <deque>
#include <json/json.h>
#include <utility>

typedef unsigned jobIdType;

struct jobArray{
	std::map<jobIdType, pid_t> look4pid;
	std::map<pid_t,jobIdType> look4jobId;
	std::deque<std::pair<jobIdType, int> > errorsByJobId;
	std::deque<std::pair<pid_t, int> > errorsByPid;
};

typedef std::tuple<jobIdType, Json::Value, std::vector<jobIdType> > jobTask;
typedef std::deque<jobTask > jobQueue;

void recieveKill(jobArray &jobIds, jobQueue &queue, jobIdType jobId );
int statusJobs(jobArray &jobIds, jobQueue &queue, jobIdType jobId);
bool cleanJobs(jobArray &jobIds);

#endif // JOB_MANAGER_HPP