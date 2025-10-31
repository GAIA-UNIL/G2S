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

#include "DataImage.hpp"
#include "picosha2.h"
#if defined(_WIN32)
  #include <io.h>
  #include <process.h>
#else
  #include <unistd.h>
#endif
#ifndef _WIN32
	#include "zlib.h"
#endif

void createLink(char* outputFullFilename, char* fullFilename){
	#ifndef _WIN32
	(void)symlink(outputFullFilename, fullFilename);
	#endif
}


char* loadRawData(const char * hash){
	char* data=nullptr;
	char* ptr=nullptr;
	char filename[4096];

	//fprintf(stderr, "look For File %s \n",hash);

	#ifndef _WIN32
	snprintf(filename,4096,"/tmp/G2S/data/%s.bgrid.gz",hash);
	if(!data &&   g2s::file_exist(filename)){
		gzFile dataFile=gzopen(filename,"rb");
		if(dataFile) {
			size_t fullSize;
			gzread (dataFile, &fullSize, sizeof(fullSize));
			gzrewind (dataFile);
			ptr =(char*) malloc (sizeof(char)*fullSize);
			data=ptr;
			gzread (dataFile, data, fullSize);
			gzclose(dataFile);
		}
	}
	#endif
	snprintf(filename,4096,"/tmp/G2S/data/%s.bgrid",hash);
	//fprintf(stderr, "%s\n",filename );
	if(!data &&  g2s::file_exist(filename)){
		FILE* dataFile=fopen(filename,"rb");
		if(dataFile) {
			size_t fullSize;
			(void)fread (&fullSize, 1, sizeof(fullSize), dataFile);
			rewind (dataFile);
			ptr = (char*)malloc (sizeof(char)*fullSize);
			data=ptr;
			(void)fread (data,1,fullSize,dataFile);
			fclose(dataFile);
		}
	}

	if(!data){
		fprintf(stderr, "file not found\n");
	}

	return data;
}

char* writeRawData(char* data, bool compresed){
	size_t fullSize=*((size_t*)data);
	std::vector<unsigned char> hash(32);
	picosha2::hash256((unsigned char*)data, ((unsigned char*)data)+fullSize-1, hash.begin(), hash.end());

	char* hashInHexa=(char*)malloc(65);
	memset(hashInHexa,0,65);
	for (int i = 0; i < 32; ++i)
	{
		snprintf(hashInHexa+2*i,65-2*i,"%02x",hash.data()[i]);
	}
	//fprintf(stderr, "%s\n", hashInHexa);

	char filename[4096];

	#ifndef _WIN32
	if(compresed) {
		snprintf(filename,4096,"/tmp/G2S/data/%s.bgrid.gz",hashInHexa);

		gzFile dataFile=gzopen(filename,"wb");
		if(dataFile) {
			gzwrite (dataFile, data, fullSize);
			gzclose(dataFile);
		}
	}
	else 
	#endif
	{
			snprintf(filename,4096,"/tmp/G2S/data/%s.bgrid",hashInHexa);

		FILE* dataFile=fopen(filename,"wb");
		if(dataFile) {
			fwrite (data, 1, fullSize, dataFile);
			fclose(dataFile);
		}
	}
	return hashInHexa;
}