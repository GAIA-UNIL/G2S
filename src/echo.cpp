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
#include "utils.hpp"
#include "DataImage.hpp"
#include "jobManager.hpp"
#include "jobReporting.hpp"

int main(int argc, char const *argv[]) {

	std::multimap<std::string, std::string> arg=g2s::argumentReader(argc,argv);

	char logFileName[2048]={0};
	char sourceFileName[2048];
	char outputFilename[2048]={0};
	jobIdType uniqueID=-1;
	bool run=true;

	FILE *reportFile=NULL;
	if (arg.count("-r") > 1)
	{
		fprintf(reportFile,"only one rapport file is possible\n");
		run=false;
	}else{
		if(arg.count("-r") ==1){
			strcpy(logFileName,(arg.find("-r")->second).c_str());
			reportFile=g2s::reporting::openReportFile((arg.find("-r")->second).c_str(), uniqueID);
			if (reportFile==NULL){
				fprintf(stderr,"Impossible to open the rapport file\n");
				run=false;
			}
		}
	}
	arg.erase("-r");
	if(reportFile==nullptr){
		reportFile=stderr;
	}
	if(outputFilename[0]=='\0' && uniqueID!=jobIdType(-1)){
		snprintf(outputFilename,2048,"%u",uniqueID);
	}
	g2s::reporting::markStarted(reportFile, "echo");
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

	g2s::DataImage srcInput=g2s::DataImage::createFromFile(sourceFileName);

	for (int i = 0; i < srcInput._types.size(); ++i)
	{
		fprintf(reportFile, "Variable %d is: %s\n", i, (srcInput._types[i]==g2s::DataImage::VariableType::Categorical ? "Categorical" : "Continuous") );
	}

	// to remove later
	srcInput.write(outputFilename);
	//end to remove

	// new filename 
	srcInput.write(std::string("im_1_")+std::to_string(uniqueID));
	g2s::reporting::recordOutputDescriptor(reportFile, 1, "simulation", std::string("im_1_")+std::to_string(uniqueID));
	g2s::reporting::logOutput(reportFile, "simulation_image_runtime", std::string("im_1_")+std::to_string(uniqueID), srcInput);
	g2s::reporting::markFinished(reportFile, 0.0);


	return 0;
}
