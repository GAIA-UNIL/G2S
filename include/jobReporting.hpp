#ifndef JOB_REPORTING_HPP
#define JOB_REPORTING_HPP

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cerrno>
#include <chrono>

#include "jobManager.hpp"
#include "DataImage.hpp"

namespace g2s {
namespace reporting {

struct JobReportContext{
	jobIdType jobId=jobIdType(-1);
	std::string logPath;
	std::string warningPath;
	std::string errorPath;
	std::string progressPath;
	std::string metaPath;
	std::map<std::string,std::string> progressValues;
	unsigned warningCount=0;
};

inline std::mutex& registryMutex(){
	static std::mutex mutex;
	return mutex;
}

inline std::map<FILE*, JobReportContext>& registry(){
	static std::map<FILE*, JobReportContext> data;
	return data;
}

inline std::string runtimeRoot(){
	return "/tmp/G2S";
}

inline std::string logDirectory(){
	return runtimeRoot()+"/logs";
}

inline std::string warningDirectory(){
	return runtimeRoot()+"/warnings";
}

inline std::string errorDirectory(){
	return runtimeRoot()+"/errors";
}

inline std::string progressDirectory(){
	return runtimeRoot()+"/progress";
}

inline std::string metaDirectory(){
	return runtimeRoot()+"/meta";
}

inline std::string logPath(jobIdType id){
	return logDirectory()+"/"+std::to_string(id)+".log";
}

inline std::string warningPath(jobIdType id){
	return warningDirectory()+"/"+std::to_string(id)+".txt";
}

inline std::string errorPath(jobIdType id){
	return errorDirectory()+"/"+std::to_string(id)+".txt";
}

inline std::string progressPath(jobIdType id){
	return progressDirectory()+"/"+std::to_string(id)+".kv";
}

inline std::string metaPath(jobIdType id){
	return metaDirectory()+"/"+std::to_string(id)+".kv";
}

inline std::string artifactName(const char* prefix, jobIdType id){
	return std::string(prefix)+std::to_string(id);
}

inline std::string progressArtifactName(jobIdType id){
	return artifactName("progress_", id);
}

inline std::string warningArtifactName(jobIdType id){
	return artifactName("warning_", id);
}

inline std::string errorArtifactName(jobIdType id){
	return artifactName("error_", id);
}

inline std::string metaArtifactName(jobIdType id){
	return artifactName("meta_", id);
}

inline std::string logArtifactName(jobIdType id){
	return artifactName("log_", id);
}

inline bool ensureDirectory(const std::string& path){
	if(path.empty()) return false;
	if(mkdir(path.c_str(), 0770) != 0 && errno != EEXIST){
		return false;
	}
	struct stat info;
	if(lstat(path.c_str(), &info) != 0) return false;
	return S_ISDIR(info.st_mode) && !S_ISLNK(info.st_mode);
}

inline bool writeTextFile(const std::string& path, const std::string& content){
	FILE* fp=fopen(path.c_str(), "wb");
	if(!fp) return false;
	const size_t size=content.size();
	const bool ok=(size==0 || fwrite(content.data(), 1, size, fp)==size);
	fclose(fp);
	return ok;
}

inline bool appendLineToFile(const std::string& path, const std::string& line){
	FILE* fp=fopen(path.c_str(), "ab");
	if(!fp) return false;
	const bool writeTextOk=(line.empty() || fwrite(line.data(), 1, line.size(), fp)==line.size());
	const bool writeNewlineOk=(fwrite("\n", 1, 1, fp)==1);
	fclose(fp);
	return writeTextOk && writeNewlineOk;
}

inline bool clearFile(const std::string& path){
	return writeTextFile(path, "");
}

inline std::string nowUnixMsString(){
	const long long nowMs=std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::system_clock::now().time_since_epoch()).count();
	return std::to_string(nowMs);
}

inline bool parseJobIdFromLogPath(const char* reportName, jobIdType &jobId){
	if(!reportName) return false;
	unsigned parsed=0;
	if(sscanf(reportName, "/tmp/G2S/logs/%u.log", &parsed)==1){
		jobId=parsed;
		return true;
	}
	return false;
}

inline void rewriteProgressFile(const JobReportContext& context){
	std::string content;
	for (auto it=context.progressValues.begin(); it!=context.progressValues.end(); ++it)
	{
		content.append(it->first);
		content.push_back('=');
		content.append(it->second);
		content.push_back('\n');
	}
	writeTextFile(context.progressPath, content);
}

inline void appendMetaLine(const JobReportContext& context, const std::string& key, const std::string& value){
	appendLineToFile(context.metaPath, key+"="+value);
}

inline void setMeta(FILE* reportFile, const std::string& key, const std::string& value){
	std::lock_guard<std::mutex> lock(registryMutex());
	auto it=registry().find(reportFile);
	if(it==registry().end()) return;
	appendMetaLine(it->second, key, value);
}

inline void setProgressValue(JobReportContext& context, const std::string& key, const std::string& value){
	context.progressValues[key]=value;
}

inline void setProgress(FILE* reportFile, double progressPercent, const std::string& stage, const std::string& detail, long long currentStep=-1, long long totalSteps=-1){
	std::lock_guard<std::mutex> lock(registryMutex());
	auto it=registry().find(reportFile);
	if(it==registry().end()) return;
	JobReportContext& context=it->second;
	setProgressValue(context, "status", "running");
	setProgressValue(context, "progress_percent", std::to_string(progressPercent));
	setProgressValue(context, "stage", stage);
	setProgressValue(context, "stage_detail", detail);
	setProgressValue(context, "last_update_unix_ms", nowUnixMsString());
	if(currentStep>=0) setProgressValue(context, "current_step", std::to_string(currentStep));
	if(totalSteps>=0) setProgressValue(context, "total_steps", std::to_string(totalSteps));
	rewriteProgressFile(context);
}

inline void markStarted(FILE* reportFile, const std::string& algorithmName=std::string()){
	std::lock_guard<std::mutex> lock(registryMutex());
	auto it=registry().find(reportFile);
	if(it==registry().end()) return;
	JobReportContext& context=it->second;
	appendMetaLine(context, "job_id", std::to_string(context.jobId));
	if(!algorithmName.empty()) appendMetaLine(context, "algorithm", algorithmName);
	appendMetaLine(context, "status", "running");
	appendMetaLine(context, "start_time_unix_ms", nowUnixMsString());
	setProgressValue(context, "job_id", std::to_string(context.jobId));
	if(!algorithmName.empty()) setProgressValue(context, "algorithm", algorithmName);
	setProgressValue(context, "status", "running");
	setProgressValue(context, "progress_percent", "0.000000");
	setProgressValue(context, "stage", "starting");
	setProgressValue(context, "stage_detail", "job accepted");
	setProgressValue(context, "last_update_unix_ms", nowUnixMsString());
	rewriteProgressFile(context);
}

inline void markFinished(FILE* reportFile, double durationMs=-1.0){
	std::lock_guard<std::mutex> lock(registryMutex());
	auto it=registry().find(reportFile);
	if(it==registry().end()) return;
	JobReportContext& context=it->second;
	appendMetaLine(context, "status", "success");
	appendMetaLine(context, "end_time_unix_ms", nowUnixMsString());
	if(durationMs>=0.0){
		appendMetaLine(context, "duration_ms", std::to_string((long long)durationMs));
	}
	setProgressValue(context, "status", "success");
	setProgressValue(context, "progress_percent", "100.000000");
	setProgressValue(context, "stage", "completed");
	setProgressValue(context, "stage_detail", "job finished");
	setProgressValue(context, "last_update_unix_ms", nowUnixMsString());
	if(durationMs>=0.0){
		setProgressValue(context, "duration_ms", std::to_string((long long)durationMs));
	}
	rewriteProgressFile(context);
}

inline void recordMetric(FILE* reportFile, const std::string& key, const std::string& value){
	setMeta(reportFile, key, value);
	std::lock_guard<std::mutex> lock(registryMutex());
	auto it=registry().find(reportFile);
	if(it==registry().end()) return;
	it->second.progressValues[key]=value;
}

inline void recordOutputDescriptor(FILE* reportFile, unsigned outputIndex, const std::string& semanticName, const std::string& artifactName){
	if(outputIndex<1) return;
	recordMetric(reportFile, "result_output_"+std::to_string(outputIndex)+"_name", semanticName);
	recordMetric(reportFile, "result_output_"+std::to_string(outputIndex)+"_artifact", artifactName);
}

inline std::string joinUnsignedVector(const std::vector<unsigned>& values, const char* separator="x"){
	std::ostringstream stream;
	for (size_t i = 0; i < values.size(); ++i)
	{
		if(i>0) stream << separator;
		stream << values[i];
	}
	return stream.str();
}

inline const char* encodingTypeName(g2s::DataImage::EncodingType encodingType){
	switch(encodingType){
		case g2s::DataImage::Float: return "float";
		case g2s::DataImage::Integer: return "int";
		case g2s::DataImage::UInteger: return "uint";
		default: return "unknown";
	}
}

inline std::string variableTypesSummary(const g2s::DataImage& image){
	if(image._types.empty()) return "none";
	bool allContinuous=true;
	bool allCategorical=true;
	for (size_t i = 0; i < image._types.size(); ++i)
	{
		allContinuous &= (image._types[i]==g2s::DataImage::Continuous);
		allCategorical &= (image._types[i]==g2s::DataImage::Categorical);
	}
	if(allContinuous) return "continuous";
	if(allCategorical) return "categorical";
	return "mixed";
}

inline void logInput(FILE* reportFile, const std::string& parameterName, const std::string& sourceName, const g2s::DataImage& image, const std::string& status="loaded"){
	if(!reportFile) return;
	fprintf(reportFile,
		"[INPUT] %s source=%s dims=%s vars=%u encoding=%s types=%s status=%s\n",
		parameterName.c_str(),
		sourceName.c_str(),
		joinUnsignedVector(image._dims).c_str(),
		image._nbVariable,
		encodingTypeName(image._encodingType),
		variableTypesSummary(image).c_str(),
		status.c_str());
}

inline void logParameter(FILE* reportFile, const std::string& key, const std::string& value){
	if(!reportFile) return;
	fprintf(reportFile, "[PARAM] %s=%s\n", key.c_str(), value.c_str());
}

inline std::string boolString(bool value){
	return value ? "true" : "false";
}

inline void logOutput(FILE* reportFile, const std::string& label, const std::string& targetName, const g2s::DataImage& image){
	if(!reportFile) return;
	fprintf(reportFile,
		"[OUTPUT] %s target=%s dims=%s vars=%u encoding=%s types=%s\n",
		label.c_str(),
		targetName.c_str(),
		joinUnsignedVector(image._dims).c_str(),
		image._nbVariable,
		encodingTypeName(image._encodingType),
		variableTypesSummary(image).c_str());
}

inline void recordWarning(FILE* reportFile, const std::string& message){
	std::lock_guard<std::mutex> lock(registryMutex());
	auto it=registry().find(reportFile);
	if(it==registry().end()) return;
	JobReportContext& context=it->second;
	appendLineToFile(context.warningPath, "time_unix_ms="+nowUnixMsString()+" level=warning message="+message);
	appendLineToFile(context.logPath, "[warning] "+message);
	context.warningCount+=1;
	appendMetaLine(context, "warning_count", std::to_string(context.warningCount));
}

inline void recordError(FILE* reportFile, const std::string& message, int errorCode=1){
	std::lock_guard<std::mutex> lock(registryMutex());
	auto it=registry().find(reportFile);
	if(it==registry().end()) return;
	JobReportContext& context=it->second;
	writeTextFile(context.errorPath, "error_code="+std::to_string(errorCode)+"\nerror_message="+message+"\n");
	appendLineToFile(context.logPath, "[error] "+message);
	appendMetaLine(context, "status", "failed");
	appendMetaLine(context, "error_code", std::to_string(errorCode));
	appendMetaLine(context, "error_message", message);
	setProgressValue(context, "status", "failed");
	setProgressValue(context, "stage", "failed");
	setProgressValue(context, "stage_detail", message);
	setProgressValue(context, "last_update_unix_ms", nowUnixMsString());
	rewriteProgressFile(context);
}

inline void registerReportFile(FILE* reportFile, jobIdType jobId, const std::string& reportPath){
	if(!reportFile || jobId==jobIdType(-1)) return;
	ensureDirectory(runtimeRoot());
	ensureDirectory(logDirectory());
	ensureDirectory(warningDirectory());
	ensureDirectory(errorDirectory());
	ensureDirectory(progressDirectory());
	ensureDirectory(metaDirectory());

	JobReportContext context;
	context.jobId=jobId;
	context.logPath=reportPath;
	context.warningPath=warningPath(jobId);
	context.errorPath=errorPath(jobId);
	context.progressPath=progressPath(jobId);
	context.metaPath=metaPath(jobId);
	clearFile(context.warningPath);
	clearFile(context.errorPath);
	clearFile(context.progressPath);
	clearFile(context.metaPath);

	std::lock_guard<std::mutex> lock(registryMutex());
	registry()[reportFile]=context;
}

inline FILE* openReportFile(const char* reportName, jobIdType &jobId, bool lineBuffered=true){
	FILE* reportFile=nullptr;
	if(reportName && !strcmp(reportName, "stderr")) return stderr;
	if(reportName && !strcmp(reportName, "stdout")) return stdout;
	if(reportName){
		reportFile=fopen(reportName, "a");
		if(reportFile && lineBuffered){
			setvbuf(reportFile, nullptr, _IOLBF, 0);
		}
		if(reportFile){
			jobIdType parsedId=jobId;
			if(parseJobIdFromLogPath(reportName, parsedId)){
				jobId=parsedId;
				registerReportFile(reportFile, jobId, reportName);
			}
		}
	}
	return reportFile;
}

inline void recordExitCode(jobIdType jobId, int exitCode){
	if(jobId==jobIdType(-1)) return;
	ensureDirectory(runtimeRoot());
	ensureDirectory(errorDirectory());
	ensureDirectory(metaDirectory());
	const std::string errorFile=errorPath(jobId);
	if(access(errorFile.c_str(), F_OK)!=0){
		writeTextFile(errorFile, "error_code="+std::to_string(exitCode)+"\nerror_message=job exited with code "+std::to_string(exitCode)+"\n");
	}
	appendLineToFile(metaPath(jobId), "status=failed");
	appendLineToFile(metaPath(jobId), "error_code="+std::to_string(exitCode));
	appendLineToFile(metaPath(jobId), "end_time_unix_ms="+nowUnixMsString());
}

}
}

#endif
