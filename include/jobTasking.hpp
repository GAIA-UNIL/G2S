#ifndef JOB_TASKING_HPP
#define JOB_TASKING_HPP


#include <unistd.h> /* for fork */
#include <sys/types.h> /* for pid_t */
#include <sys/wait.h> /* for wait */

#include <iostream>
#include <json/json.h>
#include "jobManager.hpp"

jobIdType recieveJob(jobArray &jobIds, void* data, size_t sizeBuffer, bool singleTask=true, bool functionMode=true);

#endif // JOB_TASKING_HPP