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
#include <vector>
#if defined(_WIN32)
  #include <io.h>
  #include <process.h>
#else
  #include <unistd.h>
#endif
#ifndef _WIN32
	#include "zlib.h"
#endif

namespace {
const size_t MaxRawPayloadSize = g2s::DataImage::MaxSerializedBytes;

bool readCompressedFile(const char* filename, std::vector<char>& output){
	gzFile dataFile=gzopen(filename,"rb");
	if(!dataFile) return false;

	output.clear();
	char buffer[16384];
	bool ok=true;
	while(true){
		int bytesRead=gzread(dataFile, buffer, sizeof(buffer));
		if(bytesRead > 0){
			if(output.size()+size_t(bytesRead) > MaxRawPayloadSize){
				ok=false;
				break;
			}
			output.insert(output.end(), buffer, buffer+bytesRead);
			continue;
		}
		if(bytesRead == 0) break;
		ok=false;
		break;
	}
	gzclose(dataFile);
	return ok;
}

bool readPlainFile(const char* filename, std::vector<char>& output){
	FILE* dataFile=fopen(filename,"rb");
	if(!dataFile) return false;
	if(fseek(dataFile, 0L, SEEK_END) != 0){
		fclose(dataFile);
		return false;
	}
	long fileSize=ftell(dataFile);
	if(fileSize < 0 || size_t(fileSize) > MaxRawPayloadSize){
		fclose(dataFile);
		return false;
	}
	rewind(dataFile);
	output.resize(size_t(fileSize));
	if(!output.empty() && fread(output.data(), 1, output.size(), dataFile) != output.size()){
		fclose(dataFile);
		output.clear();
		return false;
	}
	fclose(dataFile);
	return true;
}

char* copyValidatedRaw(const std::vector<char>& data, size_t* outSize){
	if(!g2s::DataImage::validateSerializedData(data.data(), data.size())) return nullptr;
	char* ptr=(char*)malloc(data.size());
	if(!ptr) return nullptr;
	memcpy(ptr, data.data(), data.size());
	if(outSize) *outSize=data.size();
	return ptr;
}
}

void createLink(char* outputFullFilename, char* fullFilename){
	#ifndef _WIN32
	(void)symlink(outputFullFilename, fullFilename);
	#endif
}


char* loadRawData(const char * hash, size_t* outSize){
	char filename[4096];
	if(outSize) *outSize=0;
	std::vector<char> data;

	//fprintf(stderr, "look For File %s \n",hash);

	#ifndef _WIN32
	snprintf(filename,4096,"/tmp/G2S/data/%s.bgrid.gz",hash);
	if(g2s::file_exist(filename) && readCompressedFile(filename, data)){
		char* raw=copyValidatedRaw(data, outSize);
		if(raw) return raw;
	}
	#endif
	snprintf(filename,4096,"/tmp/G2S/data/%s.bgrid",hash);
	//fprintf(stderr, "%s\n",filename );
	if(g2s::file_exist(filename) && readPlainFile(filename, data)){
		char* raw=copyValidatedRaw(data, outSize);
		if(raw) return raw;
	}

	fprintf(stderr, "file not found or invalid\n");
	return nullptr;
}

char* writeRawData(char* data, bool compresed){
	if(!data) return nullptr;
	size_t fullSize=*((size_t*)data);
	if(!g2s::DataImage::validateSerializedData(data, fullSize)) return nullptr;
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
