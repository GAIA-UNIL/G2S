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

#ifndef JOB_TASKING_HPP
#define JOB_TASKING_HPP


#include <unistd.h> /* for fork */
#include <sys/types.h> /* for pid_t */
#include <sys/wait.h> /* for wait */

#include <iostream>
#include <deque>
#include <json/json.h>
#include "jobManager.hpp"

typedef std::tuple<jobIdType, Json::Value, std::vector<jobIdType> > jobTask;
typedef std::deque<jobTask > jobQueue;

jobIdType recieveJob(jobQueue &queue,void* data, size_t sizeBuffer);
bool runJobInQueue(jobQueue &queue, jobArray &jobIds, bool singleTask, bool functionMode);

#endif // JOB_TASKING_HPP