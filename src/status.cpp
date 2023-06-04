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

#include "status.hpp"
#include <regex>
#include <iostream>
#include "jobManager.hpp"

int lookForStatus(void* data, size_t dataSize){
	if(dataSize==sizeof(jobIdType)){
		jobIdType id=*((jobIdType*)data);
		char filename[4096];
		sprintf(filename,"/tmp/G2S/logs/%u.log",id);

		FILE *fd;

		if ((fd = fopen(filename, "r")) != NULL) // open file
		{
			char lineHeader[2048];
			float truePourcentage=0.f;

			std::regex base_regex("(([0-9]|\\.)+%)");
			std::regex base_regex2("(([0-9]|\\.)+\\s%)");
			std::cmatch base_search;
			while(fgets(lineHeader,2048,fd)!= NULL){

				std::regex_search(lineHeader, base_search, base_regex);
				if (base_search.size() >= 1) {
					sscanf(base_search[0].str().c_str(),"%f%%",&truePourcentage);
				}else{
					std::regex_search(lineHeader, base_search, base_regex2);
					if (base_search.size() >= 1) {
						sscanf(base_search[0].str().c_str(),"%f %%",&truePourcentage);
					}
				}
			}
			fclose(fd);
			return int(truePourcentage*1000.);
		}else{
			fprintf(stderr, "can not open file : %s\n",filename );
		}
	}
	return -1;
}


int lookForDuration(void* data, size_t dataSize){
	if(dataSize==sizeof(jobIdType)){
		jobIdType id=*((jobIdType*)data);
		char filename[4096];
		sprintf(filename,"/tmp/G2S/logs/%u.log",id);

		FILE *fd;

		if ((fd = fopen(filename, "r")) != NULL) // open file
		{
			char lineHeader[2048];
			float duration;

			std::regex base_regex1("(([0-9]|\\.)+ms)");
			std::regex base_regex2("(([0-9]|\\.)+s)");
			std::regex base_regex3("(([0-9]|\\.)+\\sms)");
			std::regex base_regex4("(([0-9]|\\.)+\\ss)");
			std::cmatch base_search1;
			std::cmatch base_search2;
			std::cmatch base_search3;
			std::cmatch base_search4;
			while(fgets(lineHeader,2048,fd)!= NULL){
				std::regex_search(lineHeader, base_search1, base_regex1);
				std::regex_search(lineHeader, base_search2, base_regex2);
				std::regex_search(lineHeader, base_search3, base_regex3);
				std::regex_search(lineHeader, base_search4, base_regex4);
				bool done=false;
				if (!done && (base_search1.size() >= 1)){
					done=true;
					sscanf(base_search1[0].str().c_str(),"%fms",&duration);
				}
				if (!done && (base_search3.size() >= 1)){
					done=true;
					sscanf(base_search3[0].str().c_str(),"%f ms",&duration);
				}
				if (!done && (base_search2.size() >= 1)){
					done=true;
					sscanf(base_search2[0].str().c_str(),"%fs",&duration);
					duration*=1000; //---> convert in milisecond
				}
				if (!done && (base_search4.size() >= 1)){
					done=true;
					sscanf(base_search4[0].str().c_str(),"%f s",&duration);
					duration*=1000; //---> convert in milisecond
				}
			}
			fclose(fd);
			return int(duration);
		}else{
			fprintf(stderr, "can not open file : %s\n",filename );
		}
	}
	return -1;
}