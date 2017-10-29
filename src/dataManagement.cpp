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
		int dim=((int*)data)[0];
		int nbVariable=((int*)data)[1];
		//fprintf(stderr, "dim:%d nV:%d \n",dim, nbVariable  );
		if((dim>0) && (nbVariable>0)){

			char filename[4096];
			if(compressed)
			{
				sprintf(filename,"./data/%s.bgrid.gz",hash);
				if(!force && fileExist(filename))return 1;
				gzFile dataFile=gzopen(filename,"wb");
				if(dataFile) {
					gzwrite(dataFile, data, inSize);
					gzclose(dataFile);
				}
			}else{
				sprintf(filename,"./data/%s.bgrid",hash);
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
	unsigned size=0;

	//fprintf(stderr, "look For File %s \n",hash);

	sprintf(filename,"./data/%s.bgrid.gz",hash);
	if(!buffer &&  fileExist(filename)){
		gzFile dataFile=gzopen(filename,"rb");
		if(dataFile) {
			gzseek (dataFile , 0 , SEEK_END);
			size = gztell (dataFile);
			gzrewind (dataFile);
			buffer = malloc (sizeof(char)*size);
			gzread (dataFile, buffer, size);
			gzclose(dataFile);
		}
	}
	sprintf(filename,"./data/%s.bgrid",hash);
	//fprintf(stderr, "requested data %s\n", filename);
	if(!buffer && fileExist(filename)){
		FILE* dataFile=fopen(filename,"rb");
		if(dataFile) {
			fseek (dataFile , 0 , SEEK_END);
			size = ftell (dataFile);
			rewind (dataFile);
			buffer = malloc (sizeof(char)*size);
			fread (buffer,1,size,dataFile);
			fclose(dataFile);
		}
	}

	if(buffer){
		//fprintf(stderr, "send file\n");
		zmq::message_t reply (size);
		memcpy (reply.data (), buffer, size);
		return reply;
		//free(buffer);
		//fprintf(stderr, "%d\n", size);
	}
	return zmq::message_t(0);
}

int dataIsPresent(char* dataName){
	char hash[65]={0};
	memcpy(hash,dataName,64);
	bool isPresent=false;
	char filename[4096];
	sprintf(filename,"./data/%s.bgrid.gz",hash);
	isPresent|=fileExist(filename);
	sprintf(filename,"./data/%s.bgrid",hash);
	isPresent|=fileExist(filename);
	return isPresent;
}




