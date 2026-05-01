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

#include "dataManagement.hpp"
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <unistd.h>
#include <vector>

inline bool fileExist (char* name);

namespace {
const size_t HashSize = 64;
const size_t MaxPayloadSize = g2s::DataImage::MaxSerializedBytes;

bool isValidHash(const char* data, size_t size){
	if(!data || size < HashSize) return false;
	for(size_t i = 0; i < HashSize; ++i){
		if(!std::isxdigit(static_cast<unsigned char>(data[i]))) return false;
	}
	return true;
}

bool copySafeDataName(char* output, const char* data, size_t size){
	if(!output || !data || size < HashSize) return false;

	size_t nameSize=0;
	while(nameSize < HashSize && data[nameSize] != '\0') nameSize++;
	if(nameSize == 0) return false;

	for(size_t i = 0; i < nameSize; ++i){
		unsigned char c=static_cast<unsigned char>(data[i]);
		if(!std::isalnum(c) && c != '_' && c != '-' && c != '.') return false;
	}
	for(size_t i = nameSize; i < HashSize; ++i){
		if(data[i] != '\0') return false;
	}
	memcpy(output, data, nameSize);
	output[nameSize]='\0';
	return true;
}

void hashToHex(const std::vector<unsigned char>& hash, char* output){
	for (int i = 0; i < 32; ++i)
	{
		snprintf(output+2*i,65-2*i,"%02x",hash.data()[i]);
	}
	output[64]='\0';
}

bool computePayloadHash(const char* data, size_t size, char* output){
	if(!data || !output || size == 0) return false;
	std::vector<unsigned char> hash(32);
	picosha2::hash256((const unsigned char*)data, ((const unsigned char*)data)+size, hash.begin(), hash.end());
	hashToHex(hash, output);
	return true;
}

bool computeSerializedDataHash(const char* data, size_t size, char* output){
	if(!data || !output || size == 0) return false;

	std::vector<unsigned char> hash(32);
	picosha2::hash256((const unsigned char*)data, ((const unsigned char*)data)+size, hash.begin(), hash.end());
	hashToHex(hash, output);
	return true;
}

int publishTempFile(const char* tmpFilename, const char* filename, bool force){
	if(force){
		if(rename(tmpFilename, filename) == 0) return 0;
		remove(tmpFilename);
		return -1;
	}
	if(link(tmpFilename, filename) == 0){
		remove(tmpFilename);
		return 0;
	}
	const int publishError=errno;
	remove(tmpFilename);
	return publishError == EEXIST ? 1 : -1;
}

int writePayloadFile(const char* filename, const char* data, size_t size, bool force, bool compressed){
	static unsigned tempCounter=0;
	char tmpFilename[4096];
	snprintf(tmpFilename,4096,"%s.tmp.%ld.%u",filename,(long)getpid(),tempCounter++);

	if(compressed)
	{
		gzFile dataFile=gzopen(tmpFilename,"wb");
		if(!dataFile) return -1;
		if(gzwrite(dataFile, data, size) != int(size)){
			gzclose(dataFile);
			remove(tmpFilename);
			return -1;
		}
		if(gzclose(dataFile) != Z_OK){
			remove(tmpFilename);
			return -1;
		}
	}else{
		FILE* dataFile=fopen(tmpFilename,"wb");
		if(!dataFile) return -1;
		if(fwrite (data , sizeof(char), size, dataFile) != size){
			fclose(dataFile);
			remove(tmpFilename);
			return -1;
		}
		if(fclose(dataFile) != 0){
			remove(tmpFilename);
			return -1;
		}
	}

	return publishTempFile(tmpFilename, filename, force);
}

bool readCompressedFile(const char* filename, std::vector<char>& output, size_t maxSize){
	gzFile dataFile=gzopen(filename,"rb");
	if(!dataFile) return false;

	output.clear();
	char buffer[16384];
	bool ok=true;
	while(true){
		int bytesRead=gzread(dataFile, buffer, sizeof(buffer));
		if(bytesRead > 0){
			if(output.size()+size_t(bytesRead) > maxSize){
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

bool readPlainFile(const char* filename, std::vector<char>& output, size_t maxSize){
	FILE* dataFile=fopen(filename,"rb");
	if(!dataFile) return false;
	if(fseek(dataFile, 0L, SEEK_END) != 0){
		fclose(dataFile);
		return false;
	}
	long fileSize=ftell(dataFile);
	if(fileSize < 0 || size_t(fileSize) > maxSize){
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

zmq::message_t makeReply(const std::vector<char>& data){
	zmq::message_t reply(data.size());
	if(!data.empty()) memcpy(reply.data(), data.data(), data.size());
	return reply;
}

bool readNamedPayload(const char* hash, const char* extension, bool compressed, std::vector<char>& data, size_t maxSize){
	char filename[4096];
	snprintf(filename,4096,"/tmp/G2S/data/%s.%s%s",hash,extension,compressed ? ".gz" : "");
	if(!fileExist(filename)) return false;
	return compressed ? readCompressedFile(filename, data, maxSize) : readPlainFile(filename, data, maxSize);
}
}

inline bool fileExist (char* name) {
    if (FILE *file = fopen(name, "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }   
}


int storeData(char* data, size_t sizeBuffer,bool force, bool compressed=true){
	size_t inSize=sizeBuffer;
	char hash[65]={0};
	if(isValidHash(data, inSize) && inSize>=HashSize+sizeof(size_t)){
		memcpy(hash,data,HashSize);
		//fprintf(stderr, "add file %s\n", hash);
		data+=HashSize;
		inSize-=HashSize;
		if(!g2s::DataImage::validateSerializedData(data, inSize)) return -1;
		char computedHash[65]={0};
		if(!computeSerializedDataHash(data, inSize, computedHash)) return -1;
		if(memcmp(hash, computedHash, HashSize) != 0) return -1;
		//int dim=((int*)data)[0];
		//int nbVariable=((int*)data)[1];
		//fprintf(stderr, "dim:%d nV:%d \n",dim, nbVariable  );
		//if((dim>0) && (nbVariable>0))
		{

			char filename[4096];
			if(compressed)
			{
				snprintf(filename,4096,"/tmp/G2S/data/%s.bgrid.gz",hash);
			}else{
				snprintf(filename,4096,"/tmp/G2S/data/%s.bgrid",hash);
			}
			return writePayloadFile(filename, data, inSize, force, compressed);
		}
		return 0;
	}
	return -1;
}

zmq::message_t sendData( char* dataName){
	
	char hash[65]={0};
	if(!copySafeDataName(hash, dataName, HashSize)) return zmq::message_t(0);
	std::vector<char> buffer;

	//fprintf(stderr, "look For File %s \n",hash);

	if(readNamedPayload(hash, "bgrid", true, buffer, MaxPayloadSize) &&
		g2s::DataImage::validateSerializedData(buffer.data(), buffer.size())){
		return makeReply(buffer);
	}
	if(readNamedPayload(hash, "bgrid", false, buffer, MaxPayloadSize) &&
		g2s::DataImage::validateSerializedData(buffer.data(), buffer.size())){
		return makeReply(buffer);
	}

	fprintf(stderr, "file not found or invalid\n");
	return zmq::message_t(0);
}

int storeJson(char* data, size_t sizeBuffer,bool force, bool compressed=true){
	size_t inSize=sizeBuffer;
	char hash[65]={0};
	if(isValidHash(data, inSize) && inSize>HashSize && inSize-HashSize<=MaxPayloadSize){
		memcpy(hash,data,HashSize);
		//fprintf(stderr, "add file %s\n", hash);
		data+=HashSize;
		inSize-=HashSize;
		char computedHash[65]={0};
		if(!computePayloadHash(data, inSize, computedHash)) return -1;
		if(memcmp(hash, computedHash, HashSize) != 0) return -1;
		//int dim=((int*)data)[0];
		//int nbVariable=((int*)data)[1];
		//fprintf(stderr, "dim:%d nV:%d \n",dim, nbVariable  );
		//if((dim>0) && (nbVariable>0))
		{

			char filename[4096];
			if(compressed)
			{
				snprintf(filename,4096,"/tmp/G2S/data/%s.json.gz",hash);
			}else{
				snprintf(filename,4096,"/tmp/G2S/data/%s.json",hash);
			}
			return writePayloadFile(filename, data, inSize, force, compressed);
		}
		return 0;
	}
	return -1;
}

zmq::message_t sendJson( char* dataName){
	
	char hash[65]={0};
	if(!copySafeDataName(hash, dataName, HashSize)) return zmq::message_t(0);
	std::vector<char> buffer;

	//fprintf(stderr, "look For File %s \n",hash);

	if(readNamedPayload(hash, "json", true, buffer, MaxPayloadSize)) return makeReply(buffer);
	if(readNamedPayload(hash, "json", false, buffer, MaxPayloadSize)) return makeReply(buffer);
	fprintf(stderr, "file not found or invalid\n");
	return zmq::message_t(0);
}


zmq::message_t sendText( char* dataName){
	
	char hash[65]={0};
	if(!copySafeDataName(hash, dataName, HashSize)) return zmq::message_t(0);
	std::vector<char> buffer;

	//fprintf(stderr, "look For File %s \n",hash);

	if(readNamedPayload(hash, "txt", true, buffer, MaxPayloadSize)) return makeReply(buffer);
	if(readNamedPayload(hash, "txt", false, buffer, MaxPayloadSize)) return makeReply(buffer);
	fprintf(stderr, "file not found or invalid\n");
	return zmq::message_t(0);
}

int dataIsPresent(char* dataName){
	char hash[65]={0};
	if(!copySafeDataName(hash, dataName, HashSize)) return 0;
	int isPresent=0;
	char filename[4096];
	snprintf(filename,4096,"/tmp/G2S/data/%s.bgrid.gz",hash);
	if(fileExist(filename)) isPresent=1;
	snprintf(filename,4096,"/tmp/G2S/data/%s.bgrid",hash);
	if(fileExist(filename)) isPresent=1;
	snprintf(filename,4096,"/tmp/G2S/data/%s.json.gz",hash);
	if(fileExist(filename)) isPresent=2;
	snprintf(filename,4096,"/tmp/G2S/data/%s.json",hash);
	if(fileExist(filename)) isPresent=2;
	snprintf(filename,4096,"/tmp/G2S/data/%s.txt.gz",hash);
	if(fileExist(filename)) isPresent=3;
	snprintf(filename,4096,"/tmp/G2S/data/%s.txt",hash);
	if(fileExist(filename)) isPresent=3;
	return isPresent;
}
