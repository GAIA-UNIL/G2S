/*
 * G2S
 * Copyright (C) 2018, Mathieu Gravey (gravey.mathieu@gmail.com) and UNIL
 * (University of Lausanne)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include "jobManager.hpp"
#include "jobReporting.hpp"
#include "utils.hpp"

namespace {

const char* defaultWarningMessage="report probe emitted a warning";
const char* defaultErrorMessage="report probe emitted a fatal error";

void printHelp(){
	printf("G2S report probe\n");
	printf("Options:\n");
	printf("  -mode <log|warning|error>\n");
	printf("  -steps <n>\n");
	printf("  -sleepMs <n>\n");
	printf("  -warningMessage <text>\n");
	printf("  -errorMessage <text>\n");
}

}

int main(int argc, char const *argv[]) {
	std::multimap<std::string, std::string> arg=g2s::argumentReader(argc,argv);

	if(arg.count("-h")>0 || arg.count("--help")>0){
		printHelp();
		return 0;
	}

	jobIdType uniqueID=jobIdType(-1);
	FILE *reportFile=nullptr;
	if(arg.count("-r")>1){
		fprintf(stderr,"only one report file is possible\n");
		return 1;
	}
	if(arg.count("-r")==1){
		reportFile=g2s::reporting::openReportFile((arg.find("-r")->second).c_str(), uniqueID);
		if(reportFile==nullptr){
			fprintf(stderr,"Impossible to open the report file\n");
			return 1;
		}
	}
	arg.erase("-r");
	if(reportFile==nullptr){
		reportFile=stderr;
	}

	if(arg.count("-id")==1){
		const long parsedId=atol((arg.find("-id")->second).c_str());
		if(parsedId>=0){
			uniqueID=static_cast<jobIdType>(parsedId);
		}
	}
	arg.erase("-id");

	g2s::reporting::markStarted(reportFile, "report_probe");

	for (int i = 0; i < argc; ++i)
	{
		fprintf(reportFile,"%s ",argv[i]);
	}
	fprintf(reportFile,"\n");

	std::string mode="warning";
	if(arg.count("-mode")>0){
		mode=arg.find("-mode")->second;
	}
	arg.erase("-mode");

	unsigned steps=4;
	if(arg.count("-steps")>0){
		const unsigned parsedSteps=std::max(1, atoi(arg.find("-steps")->second.c_str()));
		steps=parsedSteps;
	}
	arg.erase("-steps");

	unsigned sleepMs=150;
	if(arg.count("-sleepMs")>0){
		sleepMs=std::max(0, atoi(arg.find("-sleepMs")->second.c_str()));
	}
	arg.erase("-sleepMs");

	std::string warningMessage=defaultWarningMessage;
	if(arg.count("-warningMessage")>0){
		warningMessage=arg.find("-warningMessage")->second;
	}
	arg.erase("-warningMessage");

	std::string errorMessage=defaultErrorMessage;
	if(arg.count("-errorMessage")>0){
		errorMessage=arg.find("-errorMessage")->second;
	}
	arg.erase("-errorMessage");

	g2s::reporting::logParameter(reportFile, "probe_mode", mode);
	g2s::reporting::logParameter(reportFile, "steps", std::to_string(steps));
	g2s::reporting::logParameter(reportFile, "sleep_ms", std::to_string(sleepMs));
	g2s::reporting::recordMetric(reportFile, "probe_mode", mode);
	g2s::reporting::recordMetric(reportFile, "configured_steps", std::to_string(steps));

	const auto start=std::chrono::steady_clock::now();
	for (unsigned step = 1; step <= steps; ++step)
	{
		const double progressPercent=(100.0*step)/steps;
		const std::string detail=std::string("step ")+std::to_string(step)+" of "+std::to_string(steps);
		fprintf(reportFile, "[LOG] %s\n", detail.c_str());
		g2s::reporting::setProgress(reportFile, progressPercent, "report_probe", detail, step, steps);

		if(mode=="warning" && step==steps/2+1){
			g2s::reporting::recordWarning(reportFile, warningMessage);
		}

		if(mode=="error" && step==steps/2+1){
			g2s::reporting::recordWarning(reportFile, "report probe is about to fail");
		}

		if(sleepMs>0 && step<steps){
			std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
		}
	}

	const auto stop=std::chrono::steady_clock::now();
	const double durationMs=std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
	g2s::reporting::recordMetric(reportFile, "completed_steps", std::to_string(steps));
	g2s::reporting::recordMetric(reportFile, "duration_ms", std::to_string((long long)durationMs));

	if(mode=="error"){
		g2s::reporting::recordError(reportFile, errorMessage, 42);
		fprintf(reportFile, "[LOG] failure path complete\n");
		return 42;
	}

	if(mode!="log" && mode!="warning"){
		const std::string invalidModeMessage=std::string("unsupported report probe mode: ")+mode;
		g2s::reporting::recordError(reportFile, invalidModeMessage, 64);
		return 64;
	}

	fprintf(reportFile, "[LOG] success path complete\n");
	g2s::reporting::markFinished(reportFile, durationMs);
	return 0;
}
