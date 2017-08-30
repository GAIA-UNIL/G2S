/*
 * Mathieu Gravey
 * Copyright (C) 2017 Mathieu Gravey
 * 
 * This program is protected software: you can not redistribute, use, and/or modify it
 * without the explicit accord from the author : Mathieu Gravey, gravey.mathieu@gmail.com
 *
 */
namespace ffmps {
	template<typename T,typename readOneCellFunction>
	inline T* readSGEMS(FILE* rapportFile,const char * fileName,int &sizeX, int &sizeY, int &sizeZ, int &dim, int &nbVariable, readOneCellFunction readFunction){

		T* result=NULL;
		FILE *file;
		file = fopen(fileName, "r");
		if (file) {
			fscanf(file,"%d %d %d",&sizeX, &sizeY, &sizeZ);
			dim=(sizeX>1)+(sizeY>1)+(sizeZ>1);
			fprintf(rapportFile, "dim = %d\n",dim );
			fscanf(file,"%d",&nbVariable);
			for (int i = 0; i < nbVariable; ++i)
			{
				char variableName[1024];
				fscanf(file,"%s",variableName);
				fprintf(rapportFile,"variable %d : %s\n", i,variableName);
			}
			result=(T*)malloc(sizeof(T) * sizeX * sizeY * sizeZ * nbVariable);
			T readedValue;
			for (int i = 0; i < sizeX * sizeY * sizeZ*  nbVariable; ++i)
			{
				result[i]=readFunction(file );
			}
			fclose(file);
		}else{
			fprintf(rapportFile, "cannot open file %s \n", fileName);
		}
		return result;
	}
} // ds