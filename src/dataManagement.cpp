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
	if(inSize>=64+sizeof(int)*2){
		memcpy(hash,data,64);
		//fprintf(stderr, "add file %s\n", hash);
		data+=64;
		inSize-=64;
		//int dim=((int*)data)[0];
		//int nbVariable=((int*)data)[1];
		//fprintf(stderr, "dim:%d nV:%d \n",dim, nbVariable  );
		//if((dim>0) && (nbVariable>0))
		{

			char filename[4096];
			if(compressed)
			{
				snprintf(filename,4096,"./data/%s.bgrid.gz",hash);
				if(!force && fileExist(filename))return 1;
				gzFile dataFile=gzopen(filename,"wb");
				if(dataFile) {
					gzwrite(dataFile, data, inSize);
					gzclose(dataFile);
				}
			}else{
				snprintf(filename,4096,"./data/%s.bgrid",hash);
				if(!force && fileExist(filename))return 1;
				FILE* dataFile=fopen(filename,"wb");
				if(dataFile) {
					fwrite (data , sizeof(char), inSize, dataFile);
					fclose(dataFile);
				}
			}
		}
	}
	return 0;
}

zmq::message_t sendData( char* dataName){
	
	char hash[65]={0};
	memcpy(hash,dataName,64);

	char filename[4096];
	void* buffer=nullptr;
	size_t fullSize;

	//fprintf(stderr, "look For File %s \n",hash);

	snprintf(filename,4096,"./data/%s.bgrid.gz",hash);
	if(!buffer &&  fileExist(filename)){
		gzFile dataFile=gzopen(filename,"rb");
		if(dataFile) {
			gzread (dataFile, &fullSize, sizeof(fullSize));
			gzrewind (dataFile);
			buffer =(char*) malloc (sizeof(char)*fullSize);
			gzread (dataFile, buffer, fullSize);
			gzclose(dataFile);
		}
	}
	snprintf(filename,4096,"./data/%s.bgrid",hash);
	//fprintf(stderr, "requested data %s\n", filename);
	if(!buffer && fileExist(filename)){
		FILE* dataFile=fopen(filename,"rb");
		if(dataFile) {
			(void)fread (&fullSize, 1, sizeof(fullSize), dataFile);
			rewind (dataFile);
			buffer = (char*)malloc (sizeof(char)*fullSize);
			(void)fread (buffer,1,fullSize,dataFile);
			fclose(dataFile);
		}
	}

	if(buffer){
		//fprintf(stderr, "send file\n");
		zmq::message_t reply (fullSize);
		memcpy (reply.data (), buffer, fullSize);
		free(buffer);
		return reply;
		//
		//fprintf(stderr, "%d\n", size);
	}
	fprintf(stderr, "file not fund\n");
	return zmq::message_t(0);
}

int storeJson(char* data, size_t sizeBuffer,bool force, bool compressed=true){
	size_t inSize=sizeBuffer;
	char hash[65]={0};
	if(inSize>=64+sizeof(int)*2){
		memcpy(hash,data,64);
		//fprintf(stderr, "add file %s\n", hash);
		data+=64;
		inSize-=64;
		//int dim=((int*)data)[0];
		//int nbVariable=((int*)data)[1];
		//fprintf(stderr, "dim:%d nV:%d \n",dim, nbVariable  );
		//if((dim>0) && (nbVariable>0))
		{

			char filename[4096];
			if(compressed)
			{
				snprintf(filename,4096,"./data/%s.json.gz",hash);
				if(!force && fileExist(filename))return 1;
				gzFile dataFile=gzopen(filename,"wb");
				if(dataFile) {
					gzwrite(dataFile, data, inSize);
					gzclose(dataFile);
				}
			}else{
				snprintf(filename,4096,"./data/%s.json",hash);
				if(!force && fileExist(filename))return 1;
				FILE* dataFile=fopen(filename,"wb");
				if(dataFile) {
					fwrite (data , sizeof(char), inSize, dataFile);
					fclose(dataFile);
				}
			}
		}
	}
	return 0;
}

zmq::message_t sendJson( char* dataName){
	
	char hash[65]={0};
	memcpy(hash,dataName,64);

	char filename[4096];
	void* buffer=nullptr;
	size_t fullSize;

	//fprintf(stderr, "look For File %s \n",hash);

	snprintf(filename,4096,"./data/%s.json.gz",hash);
	if(!buffer &&  fileExist(filename)){
		gzFile dataFile=gzopen(filename,"rb");
		if(dataFile) {
			gzread (dataFile, &fullSize, sizeof(fullSize));
			gzrewind (dataFile);
			buffer =(char*) malloc (sizeof(char)*fullSize);
			gzread (dataFile, buffer, fullSize);
			gzclose(dataFile);
		}
	}
	snprintf(filename,4096,"./data/%s.json",hash);
	//fprintf(stderr, "requested data %s\n", filename);
	if(!buffer && fileExist(filename)){
		FILE* dataFile=fopen(filename,"rb");
		if(dataFile) {
			(void)fread (&fullSize, 1, sizeof(fullSize), dataFile);
			rewind (dataFile);
			buffer = (char*)malloc (sizeof(char)*fullSize);
			(void)fread (buffer,1,fullSize,dataFile);
			fclose(dataFile);
		}
	}

	if(buffer){
		//fprintf(stderr, "send file\n");
		zmq::message_t reply (fullSize);
		memcpy (reply.data (), buffer, fullSize);
		free(buffer);
		return reply;
		//
		//fprintf(stderr, "%d\n", size);
	}
	fprintf(stderr, "file not fund\n");
	return zmq::message_t(0);
}


zmq::message_t sendText( char* dataName){
	
	char hash[65]={0};
	memcpy(hash,dataName,64);

	char filename[4096];
	void* buffer=nullptr;
	size_t fullSize;

	//fprintf(stderr, "look For File %s \n",hash);

	snprintf(filename,4096,"./data/%s.txt.gz",hash);
	if(!buffer &&  fileExist(filename)){
		gzFile dataFile=gzopen(filename,"rb");
		if(dataFile) {
			gzread (dataFile, &fullSize, sizeof(fullSize));
			gzrewind (dataFile);
			buffer =(char*) malloc (sizeof(char)*fullSize);
			gzread (dataFile, buffer, fullSize);
			gzclose(dataFile);
		}
	}
	snprintf(filename,4096,"./data/%s.txt",hash);
	//fprintf(stderr, "requested data %s\n", filename);
	if(!buffer && fileExist(filename)){
		FILE* dataFile=fopen(filename,"r");
		if(dataFile) {
			fseek(dataFile, 0L, SEEK_END);
			fullSize = ftell(dataFile);
			rewind (dataFile);
			buffer = (char*)malloc (sizeof(char)*fullSize);
			(void)fread (buffer,1,fullSize,dataFile);
			fclose(dataFile);
		}
	}

	if(buffer){
		//fprintf(stderr, "send file\n");
		zmq::message_t reply (fullSize);
		memcpy (reply.data (), buffer, fullSize);
		free(buffer);
		return reply;
		//
		//fprintf(stderr, "%d\n", size);
	}
	fprintf(stderr, "file not fund\n");
	return zmq::message_t(0);
}

int dataIsPresent(char* dataName){
	char hash[65]={0};
	memcpy(hash,dataName,64);
	int isPresent=0;
	char filename[4096];
	snprintf(filename,4096,"./data/%s.bgrid.gz",hash);
	if(fileExist(filename)) isPresent=1;
	snprintf(filename,4096,"./data/%s.bgrid",hash);
	if(fileExist(filename)) isPresent=1;
	snprintf(filename,4096,"./data/%s.json.gz",hash);
	if(fileExist(filename)) isPresent=2;
	snprintf(filename,4096,"./data/%s.json",hash);
	if(fileExist(filename)) isPresent=2;
	snprintf(filename,4096,"./data/%s.txt.gz",hash);
	if(fileExist(filename)) isPresent=3;
	snprintf(filename,4096,"./data/%s.txt",hash);
	if(fileExist(filename)) isPresent=3;
	return isPresent;
}




