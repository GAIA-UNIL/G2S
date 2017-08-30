#include "status.hpp"
#include <regex>
#include <iostream>

int lookForStatus(void* data, size_t dataSize){
	if(dataSize==sizeof(long long)){
		long long id=*((long long*)data);
		char filename[4096];
		sprintf(filename,"logs/%ld.log",id);

		FILE *fd;

		if ((fd = fopen(filename, "r")) != NULL) // open file
		{
			char lineHeader[2048];
			char endOfline[2048];
			float pourcentage;
			float truePourcentage;

			std::regex base_regex("(([0-9]|\\.)+%)");
			std::regex base_regex2("(([0-9]|\\.)+\\s%)");
			std::cmatch base_search;
			while(fgets(lineHeader,2048,fd)!= NULL){

				std::regex_search(lineHeader, base_search, base_regex);
				//fprintf(stderr, "matchs: %d\n", base_search.size());
				if (base_search.size() >= 1) {
					sscanf(base_search[0].str().c_str(),"%f%",&truePourcentage);
				}else{
					std::regex_search(lineHeader, base_search, base_regex2);
					if (base_search.size() >= 1) {
						sscanf(base_search[0].str().c_str(),"%f %",&truePourcentage);
					}
				}
			}
			fclose(fd);
			return int(truePourcentage*1000.);
		}
	}
	return -1;
}