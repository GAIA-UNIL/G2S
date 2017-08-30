/*
 * Mathieu Gravey
 * Copyright (C) 2017 Mathieu Gravey
 * 
 * This program is protected software: you can not redistribute, use, and/or modify it
 * without the explicit accord from the author : Mathieu Gravey, gravey.mathieu@gmail.com
 *
 */
namespace ffmps {
	template<typename T,typename writeOneCellFunction>
	inline void writeSGEMS(FILE* rapportFile,const char * fileName,int sizeX, int sizeY, int sizeZ, int dim, T* data, int nbVariable,char** varName, writeOneCellFunction writeFunction){

		T* result=NULL;
		FILE *file;
		file = fopen(fileName, "w");
		if (file) {
			fprintf(file,"%d %d %d\n",sizeX, sizeY, sizeZ);
			fprintf(file,"%d\n",nbVariable);
			for (int i = 0; i < nbVariable; ++i)
			{
				char variableName[1024];
				fprintf(file,"%s ",varName[i]);
			}
			fprintf(file,"\n");
			for (int i = 0; i < sizeX * sizeY * sizeZ; ++i)
			{
				writeFunction(file,data[i]);
			}
			fclose(file);
		}else{
			fprintf(rapportFile, "cannot open file %s \n", fileName);
		}
	}
} // ds