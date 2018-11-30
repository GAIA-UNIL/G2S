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

#include <iostream>
#include <cmath>
#include <chrono>
#include "utils.hpp"
#include "DataImage.hpp"
#include "jobManager.hpp"

#include "simulation.hpp"

void printHelp(){
	printf ("that is the help");
}

int main(int argc, char const *argv[]) {

	std::multimap<std::string, std::string> arg=g2s::argumentReader(argc,argv);

	char logFileName[2048]={0};
	std::vector<std::string> sourceFileNameVector;
	std::string targetFileName;
	std::string kernelFileName;
	std::string simuationPathFileName;

	std::string outputFilename;
	std::string outputIndexFilename;


	jobIdType uniqueID=-1;
	bool run=true;


	// manage report file
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
				if(sscanf(logFileName,"logs/%d.log",&logId)==1){
					std::to_string(logId);
					//sprintf(outputFilename,"%d",logId);
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

	// LOOK FOR STANDARD PARAMETER

	unsigned nbThreads=1;
	bool verbose=false;

	if (arg.count("-j") == 1)
	{
		nbThreads=atoi((arg.find("-j")->second).c_str());
	}
	arg.erase("-j");

	if (arg.count("--jobs") == 1)
	{
		nbThreads=atoi((arg.find("--jobs")->second).c_str());
	}
	arg.erase("--jobs");	

	if ((arg.count("-h") == 1)|| (arg.count("--help") == 1))
	{
		printHelp();
		return 0;
	}
	arg.erase("-h");

	if ((arg.count("-v") == 1) || (arg.count("--verbose") == 1))
	{
		verbose=true;
	}
	arg.erase("--verbose");






	// LOOK FOR DATA FILES
	//look for training images
	if (arg.count("-ti") > 0)
	{
		std::multimap<std::string, std::string>::iterator it;
	    for (it=arg.equal_range("-ti").first; it!=arg.equal_range("-ti").second; ++it)
	    {
	    	sourceFileNameVector.push_back(it->second);
	    }
	}else{	
		fprintf(reportFile,"error source\n");
		run=false;
	}
	arg.erase("-ti");

	//look for destination images (hard data)
	if (arg.count("-di") ==1)
	{
		targetFileName=arg.find("-di")->second;
	}else{	
		fprintf(reportFile,"error target\n");
		run=false;
	}
	arg.erase("-di");

	//look for -ki			: kernel image 
	if (arg.count("-ki") ==1)
	{
		kernelFileName=arg.find("-ki")->second;
	}else{	
		fprintf(reportFile,"error target\n");
	}
	arg.erase("-ki");

	//look for -sp			: simulation path 
	if (arg.count("-sp") ==1)
	{
		simuationPathFileName=arg.find("-sp")->second;
	}else{	
		fprintf(reportFile,"error target\n");
	}
	arg.erase("-sp");





	// LOOK FOR OUTPUT
	if (arg.count("-o") ==1)
	{
		outputFilename=arg.find("-o")->second;
		run=false;
	}else{
		outputFilename=std::to_string(uniqueID);
		outputIndexFilename=std::string("id_")+std::to_string(uniqueID);
	}
	arg.erase("-o");

	if (arg.count("-oi") ==1)
	{
		outputIndexFilename=arg.find("-oi")->second;
	}
	arg.erase("-oi");





	// LOOK FOR SETINGS

	std::vector<float> thresholds;		// threshold for DS ...
	int nbNeighbors=-1;						// number of nighbors QS, DS ...
	float mer=std::nanf("0");				// maximum exploration ratio, called f in ds
	float nbCandidate=std::nanf("0");		// 1/f for QS
	float narrowness=std::nanf("0");		// narrowness for NDS
	unsigned seed=std::chrono::high_resolution_clock::now().time_since_epoch().count();
	g2s::DistanceType searchDistance=g2s::EUCLIDIEN;
	bool requestFullSimulation=false;

	if (arg.count("-fs") == 1)
	{
		requestFullSimulation=true;
	}
	arg.erase("-fs");

	std::multimap<std::string, std::string>::iterator it;
	for (it=arg.equal_range("-th").first; it!=arg.equal_range("-th").second; ++it)
	{
		thresholds.push_back(atof(it->second.c_str()));
	}
	arg.erase("-th");

	if (arg.count("-f") == 1)
	{
		mer=atof((arg.find("-f")->second).c_str());
	}
	arg.erase("-f");

	if (arg.count("-mer") == 1)
	{
		mer=atof((arg.find("-mer")->second).c_str());
	}
	arg.erase("-mer");

	if (arg.count("-k") == 1)
	{
		nbCandidate=atof((arg.find("-k")->second).c_str());
	}
	arg.erase("-k");

	if (arg.count("-nw") == 1)
	{
		narrowness=atof((arg.find("-nw")->second).c_str());
	}
	arg.erase("-nw");

	if (arg.count("-n") == 1)
	{
		nbNeighbors=atoi((arg.find("-n")->second).c_str());
	}
	arg.erase("-n");

	if (arg.count("-s") == 1)
	{
		seed=atoi((arg.find("-s")->second).c_str());
	}
	arg.erase("-s");

	if (arg.count("-wd") == 1)
	{
		searchDistance=g2s::KERNEL;
	}
	arg.erase("-wd");

	if (arg.count("-ed") == 1)
	{
		searchDistance=g2s::EUCLIDIEN;
	}
	arg.erase("-ed");

	if (arg.count("-md") == 1)
	{
		searchDistance=g2s::MANAHTTAN;
	}
	arg.erase("-md");



	// special DS
	float alpha=0;
	int distanceToCenterForNeighbour=-1;

	if (arg.count("-alpha") == 1)
	{
		sscanf((arg.find("-alpha")->second).c_str(),"%f",&alpha);
	}
	arg.erase("-alpha");

	if (arg.count("-Ndc") == 1)
	{
		distanceToCenterForNeighbour=atoi((arg.find("-Ndc")->second).c_str());
	}
	arg.erase("-Ndc");


	//add extra paremetre here


	// precheck | check what is mandatory

	if(nbNeighbors<0){
		run=false;
		fprintf(reportFile, "%s\n", "number of neighbor not valide" );
	}
	if(thresholds.size()<1){
		run=false;
		fprintf(reportFile, "%s\n", "threshold need to be seted" );
	}
	if(std::isnan(mer) && std::isnan(nbCandidate)){
		run=false;
		fprintf(reportFile, "%s\n", "maximum exploration ratio or numer of candidate need to be seted" );
	}

	if(std::isnan(mer)){
		mer=1./nbCandidate;
	}

	if(!run) return 0;

	// print all ignored parameters
	for (std::multimap<std::string, std::string>::iterator it=arg.begin(); it!=arg.end(); ++it){
		fprintf(reportFile, "%s %s <== ignored !\n", it->first.c_str(), it->second.c_str());
	}

	if(!run) return 0;

	std::vector<g2s::DataImage> sourceImages;
	g2s::DataImage destinationImage=g2s::DataImage::createFromFile(targetFileName);

	for (int i = 0; i < sourceFileNameVector.size(); ++i)
	{
		sourceImages.push_back(g2s::DataImage::createFromFile(sourceFileNameVector[i]));
	}

	//put nan to Magic value
	for (int j = 0; j < sourceImages.size(); ++j)
	{
		float* dataPtr= sourceImages[j]._data;
		for (int i = 0; i < sourceImages[j].dataSize(); ++i)
		{
			if(std::isnan(dataPtr[i]))dataPtr[i]=-9999999;
		}
	}

	{
		float* dataPtr= destinationImage._data;
		for (int i = 0; i < destinationImage.dataSize(); ++i)
		{
			if(std::isnan(dataPtr[i]))dataPtr[i]=-9999999;
		}
	}



	//writing filechar

	for (int j = 0; j < sourceImages.size(); ++j)
	{
		char dsTIFileName[128];
		sprintf(dsTIFileName, "data/ti_%lld_%d.gslib",uniqueID,j);
		sourceImages[0].writeSGEMS(dsTIFileName);
	}

	{
		char dsDIFileName[128];
		sprintf(dsDIFileName, "data/di_%lld.gslib",uniqueID);
		destinationImage.writeSGEMS(dsDIFileName);
	}


	char dsOutFileName[128];
	char dsReportFileName[128];
	
	sprintf(dsOutFileName, "data/output_%lld_real00000.gslib",uniqueID);
	sprintf(dsReportFileName, "logs/report_%lld.txt\n",uniqueID);

	//start config file

	char dsConfigFilename[2048];
	sprintf(dsConfigFilename,"data/config_%lld.in", uniqueID);

	FILE* dsConfigFile=fopen(dsConfigFilename, "w");

	if(dsConfigFile){
		// write config file
		fprintf(dsConfigFile, "%d %d %d\n", destinationImage._dims[0], destinationImage._dims.size()>1 ? destinationImage._dims[1] : 1, destinationImage._dims.size()>2 ? destinationImage._dims[2] : 1); // write dimension of simulation
		fprintf(dsConfigFile, "%d %d %d\n", 1, 1, 1); // suppose uniforme spacing 
		fprintf(dsConfigFile, "%d %d %d\n", 0, 0, 0); // offset of 0
		fprintf(dsConfigFile, "\n");

		fprintf(dsConfigFile, "%d\n", destinationImage._types.size()); // number of variable

		for (int i = 0; i < destinationImage._types.size(); ++i)
		{
			std::string startName;
			switch(destinationImage._types[i]){
				case g2s::DataImage::Continuous:
					startName="Continuous";
					break;
				case g2s::DataImage::Categorical:
					startName="Categorical";
					break;
				default :
					startName="unkown";
			}
			fprintf(dsConfigFile, "%s_%d %d DEFAULT_FORMAT\n", startName.c_str(),i, 1);
		}
		fprintf(dsConfigFile, "\n");

		fprintf(dsConfigFile, "OUTPUT_SIM_ONE_FILE_PER_REALIZATION\n");
		fprintf(dsConfigFile, "data/output_%lld\n",uniqueID);
		fprintf(dsConfigFile, "\n");

		fprintf(dsConfigFile, "1\n"); // output report
		fprintf(dsConfigFile, "logs/report_%lld.txt\n",uniqueID);
		fprintf(dsConfigFile, "\n");

		//ti
		fprintf(dsConfigFile, "%d\n",sourceImages.size());
		for (int j = 0; j < sourceImages.size(); ++j)
		{
			char dsTIFileName[128];
			fprintf(dsConfigFile, "data/ti_%lld_%d.gslib\n",uniqueID,j);
		}

		//di
		fprintf(dsConfigFile, "1\n"); // with di
		fprintf(dsConfigFile, "data/di_%lld.gslib\n",uniqueID);
		fprintf(dsConfigFile, "\n");

		fprintf(dsConfigFile, "0\n"); // no data point set
		fprintf(dsConfigFile, "\n");

		fprintf(dsConfigFile, "0\n"); // no data mask image
		fprintf(dsConfigFile, "\n");

		fprintf(dsConfigFile, "0\n"); // no data HOMOTHETY
		fprintf(dsConfigFile, "\n");

		fprintf(dsConfigFile, "0\n"); // no data ROTATION
		fprintf(dsConfigFile, "\n");

		fprintf(dsConfigFile, "0.05\n"); // no data mask image
		fprintf(dsConfigFile, "\n");

		fprintf(dsConfigFile, "NORMALIZING_LINEAR\n"); // no data mask image
		fprintf(dsConfigFile, "\n");

		for (int i = 0; i < destinationImage._types.size(); ++i)
		{
			fprintf(dsConfigFile, "%d %d %d\n", distanceToCenterForNeighbour, distanceToCenterForNeighbour*(destinationImage._dims.size()>1), distanceToCenterForNeighbour*(destinationImage._dims.size()>2)); // write dimension of simulation
			fprintf(dsConfigFile, "%d %d %d\n", 1, 1, 1); // suppose non anisotropy
			fprintf(dsConfigFile, "%d %d %d\n", 0, 0, 0); // suppose non rotation
			fprintf(dsConfigFile, "%f\n",alpha);
		}
		fprintf(dsConfigFile, "\n");

		for (int i = 0; i < destinationImage._types.size(); ++i)
		{
			fprintf(dsConfigFile, "%d ",nbNeighbors); // NEIGHBORING
		}
		fprintf(dsConfigFile, "\n");
		fprintf(dsConfigFile, "\n");

		for (int i = 0; i < destinationImage._types.size(); ++i)
		{
			fprintf(dsConfigFile, "%f ",1.f); //DENSITY OF NEIGHBORING
		}
		fprintf(dsConfigFile, "\n");
		fprintf(dsConfigFile, "\n");

		for (int i = 0; i < destinationImage._types.size(); ++i)
		{
			fprintf(dsConfigFile, "%d ",0); //RELATIVE DISTANCE FLAG
		}
		fprintf(dsConfigFile, "\n");
		fprintf(dsConfigFile, "\n");

		for (int i = 0; i < destinationImage._types.size(); ++i)
		{
			fprintf(dsConfigFile, "%d ",2*(destinationImage._types[i]==g2s::DataImage::Continuous)); // L2 norm
		}
		fprintf(dsConfigFile, "\n");
		fprintf(dsConfigFile, "\n");

		for (int i = 0; i < destinationImage._types.size(); ++i)
		{
			fprintf(dsConfigFile, "%f ",1.0);// WEIGHT FACTOR FOR CONDITIONING DATA
		}
		fprintf(dsConfigFile, "\n");
		fprintf(dsConfigFile, "\n");

		if(requestFullSimulation){
			fprintf(dsConfigFile, "SIM_ONE_BY_ONE\n"); // no data mask image
		}
		else{
			fprintf(dsConfigFile, "SIM_VARIABLE_VECTOR\n"); // no data mask image
		}
		fprintf(dsConfigFile, "\n");

		fprintf(dsConfigFile, "PATH_RANDOM\n"); // no data mask image
		fprintf(dsConfigFile, "\n");

		for (int i = 0; i < destinationImage._types.size(); ++i)
		{
			fprintf(dsConfigFile, "%f ",std::min(std::max(0.000000000001f,thresholds[i%(thresholds.size())]),0.999999f));// THRESHOLD
		}
		fprintf(dsConfigFile, "\n");
		fprintf(dsConfigFile, "\n");

		for (int i = 0; i < destinationImage._types.size(); ++i)
		{
			fprintf(dsConfigFile, "%d ",0);//PROBABILITY CONSTRAINTS
		}
		fprintf(dsConfigFile, "\n");
		fprintf(dsConfigFile, "\n");

		for (int i = 0; i < destinationImage._types.size(); ++i)
		{
			fprintf(dsConfigFile, "%d ",0);//BLOCK DATA
		}
		fprintf(dsConfigFile, "\n");
		fprintf(dsConfigFile, "\n");

		fprintf(dsConfigFile, "%f\n",mer); //SCAN FRACTION
		fprintf(dsConfigFile, "\n");

		fprintf(dsConfigFile, "0.0\n"); //TOLERANCE
		fprintf(dsConfigFile, "\n");

		fprintf(dsConfigFile, "0\n"); // no post processing
		fprintf(dsConfigFile, "\n");

		fprintf(dsConfigFile, "0\n"); // pyramid general parameters
		fprintf(dsConfigFile, "\n");

		fprintf(dsConfigFile, "%d\n",seed);
		fprintf(dsConfigFile, "1\n"); 
		fprintf(dsConfigFile, "\n");

		fprintf(dsConfigFile, "1\n"); // single realisation
		fprintf(dsConfigFile, "\n");
		fprintf(dsConfigFile, "END");

	}

	fclose(dsConfigFile);
	dsConfigFile=nullptr;

	char pathExe[128]="../3party_bin/deesseOMP";
	char nbThreadStr[128];
	sprintf(nbThreadStr,"%d",nbThreads);
	char configFile[128];
	strcpy(configFile,dsConfigFilename);

	char* argvChild[4];
	argvChild[0]=pathExe;
	argvChild[1]=nbThreadStr;
	argvChild[2]=configFile;
	argvChild[3]=nullptr;

	auto begin = std::chrono::high_resolution_clock::now();
	pid_t pid=fork();
	if (pid==0) { // child process //
		
		char* env[3]={nullptr};
		env[0]="_STDBUF_O=0";
		env[1]="LD_PRELOAD=/usr/lib/x86_64-linux-gnu/coreutils/libstdbuf.so"; /// need to be improve with autosearch
		dup2(fileno(reportFile),STDOUT_FILENO); /*copy the file descriptor fd into standard output*/
		dup2(fileno(reportFile),STDERR_FILENO);
		execve(argvChild[0], (char**) argvChild,env);
		exit(127); // only if execv fails //
	}
	else { // pid!=0; parent process //
		waitpid(pid,0,0); // wait for child to exit //
	}
	auto end = std::chrono::high_resolution_clock::now();
	double time = 1.0e-6 * std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	fprintf(reportFile,"compuattion time: %7.2f s\n", time/1000);
	fprintf(reportFile,"compuattion time: %.0f ms\n", time);


	g2s::DataImage DI=g2s::DataImage::readSGEMS(dsOutFileName);
	//g2s::DataImage id=DI.emptyCopy(!fullSimulation);
	//id.setEncoding(g2s::DataImage::UInteger);
	//id.write(outputIndexFilename);
	DI.write(outputFilename);

	// free stuff
	for (int j = 0; j < sourceImages.size(); ++j)
	{
		char dsTIFileName[128];
		sprintf(dsTIFileName, "ti_%lld_%d.gslib",uniqueID,j);
		remove(dsTIFileName);
	}

	{
		char dsDIFileName[128];
		sprintf(dsDIFileName, "di_%lld.gslib",uniqueID);
		remove(dsDIFileName);
	}
	remove(dsOutFileName);
	remove(dsConfigFilename);
	remove(dsReportFileName);

	return 0;
}