#include <iostream>
#include "utils.hpp"
#include "DataImage.hpp"
#include "jobManager.hpp"

int main(int argc, char const *argv[]) {

	std::multimap<std::string, std::string> arg=g2s::argumentReader(argc,argv);

	char logFileName[2048]={0};
	char sourceFileName[2048];
	char outputFilename[2048];
	jobIdType uniqueID=-1;
	bool run=true;

	FILE *reportFile=NULL;
	if (arg.count("-r") > 1)
	{
		fprintf(reportFile,"only one rapport file is possible\n");
		run=false;
	}else{
		if(arg.count("-r") ==1){
			if(!strcmp((arg.find("-r")->second).c_str(),"stderr")){
				reportFile=stderr;
			}
			if(!strcmp((arg.find("-r")->second).c_str(),"stdout")){
				reportFile=stdout;
			}
			if (reportFile==NULL) {
				strcpy(logFileName,(arg.find("-r")->second).c_str());
				reportFile=fopen((arg.find("-r")->second).c_str(),"a");
				setvbuf ( reportFile , nullptr , _IOLBF , 0 ); // maybe  _IONBF


				jobIdType logId;
				if(sscanf(logFileName,"logs/%u.log",&logId)==1){
					sprintf(outputFilename,"%u",logId);
					//symlink(outputName, fullFilename);
					uniqueID=logId;
				}
			}
			if (reportFile==NULL){
				fprintf(stderr,"Impossible to open the rapport file\n");
				run=false;
			}
		}
	}
	arg.erase("-r");
	for (int i = 0; i < argc; ++i)
	{
		fprintf(reportFile,"%s ",argv[i]);
	}
	fprintf(reportFile,"\n");

	if (arg.count("-ti") != 1)
	{
		fprintf(reportFile,"error source\n");
		run=false;
	}else{
		strcpy(sourceFileName,(arg.find("-ti")->second).c_str());
	}
	arg.erase("-ti");

	int outputCount=arg.count("-o");
	if (outputCount >0)
	{
		strcpy(outputFilename,(arg.find("-o")->second).c_str());
	}
	arg.erase("-o");

	for (std::multimap<std::string, std::string>::iterator it=arg.begin(); it!=arg.end(); ++it){
		fprintf(reportFile, "%s %s <== ignored !\n", it->first.c_str(), it->second.c_str());
	}

	if(!run) return 0;

	g2s::DataImage srcIput=g2s::DataImage::createFromFile(sourceFileName);

	srcIput.write(outputFilename);


	return 0;
}