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
  #include <sys/stat.h>
#endif
#include <atomic>
#ifndef _WIN32
	#include "zlib.h"
#endif

namespace {
const size_t MaxRawPayloadSize = g2s::DataImage::MaxSerializedBytes;

#ifndef _WIN32
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
#endif

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

#ifndef _WIN32
namespace {
// A name unique to this process and call, next to `path`, for atomic replacement.
void temporaryName(const char* path, char* out, size_t size){
	static std::atomic<unsigned> counter(0);
	snprintf(out, size, "%s.tmp.%ld.%u", path, (long)getpid(), counter++);
}

bool isSymbolicLink(const char* path){
	struct stat info;
	return lstat(path, &info) == 0 && S_ISLNK(info.st_mode);
}
}
#endif

// Links the name of an image (fullFilename, e.g. /tmp/G2S/data/im_1_<id>.bgrid.gz)
// to its content-addressed file (outputFullFilename, named by the SHA-256 of
// the content). Writing a name again replaces its link atomically: the new
// link is made under a temporary name and renamed over the old one, so a
// reader sees the old image or the new one, never a missing or stale mix, and
// concurrent writers of the same name leave one of their links. Only links
// are replaced: a regular file under that name (a content-addressed file) is
// left as it is. The link of the other compression (.bgrid vs .bgrid.gz),
// which loadRawData would otherwise prefer or fall back to, is removed.
void createLink(char* outputFullFilename, char* fullFilename){
	#ifndef _WIN32
	if(strcmp(outputFullFilename, fullFilename)==0) return; // the name is the content hash itself
	struct stat info;
	if(lstat(fullFilename, &info)==0 && !S_ISLNK(info.st_mode)) return;
	char temporary[4096];
	temporaryName(fullFilename, temporary, sizeof(temporary));
	if(symlink(outputFullFilename, temporary)!=0) return;
	if(rename(temporary, fullFilename)!=0){
		unlink(temporary);
		return;
	}
	char other[4096];
	const size_t length=strlen(fullFilename);
	if(length>3 && strcmp(fullFilename+length-3, ".gz")==0) snprintf(other, sizeof(other), "%.*s", int(length-3), fullFilename);
	else snprintf(other, sizeof(other), "%s.gz", fullFilename);
	if(isSymbolicLink(other)) unlink(other);
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
	picosha2::hash256((unsigned char*)data, ((unsigned char*)data)+fullSize, hash.begin(), hash.end());

	char* hashInHexa=(char*)malloc(65);
	memset(hashInHexa,0,65);
	for (int i = 0; i < 32; ++i)
	{
		snprintf(hashInHexa+2*i,65-2*i,"%02x",hash.data()[i]);
	}
	//fprintf(stderr, "%s\n", hashInHexa);

	char filename[4096];

	// The content file is written under a temporary name and renamed, so that
	// a reader (or a link to it) never sees a partly written file, also when
	// another job is writing the same content at the same time.
	#ifndef _WIN32
	char temporary[4096];
	if(compresed) {
		snprintf(filename,4096,"/tmp/G2S/data/%s.bgrid.gz",hashInHexa);
		temporaryName(filename, temporary, sizeof(temporary));
		gzFile dataFile=gzopen(temporary,"wb");
		if(dataFile) {
			const bool written=gzwrite(dataFile, data, unsigned(fullSize))==int(fullSize);
			if(gzclose(dataFile)==Z_OK && written) {
				if(rename(temporary, filename)!=0) unlink(temporary);
			} else {
				unlink(temporary);
			}
		}
	}
	else {
		snprintf(filename,4096,"/tmp/G2S/data/%s.bgrid",hashInHexa);
		temporaryName(filename, temporary, sizeof(temporary));
		FILE* dataFile=fopen(temporary,"wb");
		if(dataFile) {
			const bool written=fwrite(data, 1, fullSize, dataFile)==fullSize;
			if(fclose(dataFile)==0 && written) {
				if(rename(temporary, filename)!=0) unlink(temporary);
			} else {
				unlink(temporary);
			}
		}
	}
	#else
	{
		snprintf(filename,4096,"/tmp/G2S/data/%s.bgrid",hashInHexa);

		FILE* dataFile=fopen(filename,"wb");
		if(dataFile) {
			fwrite (data, 1, fullSize, dataFile);
			fclose(dataFile);
		}
	}
	#endif
	return hashInHexa;
}
