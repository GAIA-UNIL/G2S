#ifndef INERFACE_TEMPLATE_HPP
#define INERFACE_TEMPLATE_HPP

#ifndef VERSION
	#define VERSION "unknown"
#endif



#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <iostream>
#include <zmq.hpp>
#include <vector>
#include "protocol.hpp"
#include "picosha2.h"
#include <json/json.h>
#include <numeric>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <map>
#include <any>
#include <set>
#include <thread>
#include <cctype>
#include <sstream>
#include <iomanip>
#include "DataImage.hpp"
#ifdef G2S_ENABLE_BROWSER_TRANSPORT
#include "browserTransport.hpp"

#ifndef G2S_BROWSER_ORIGIN
	#define G2S_BROWSER_ORIGIN "*"
#endif
#endif

typedef unsigned jobIdType;

//template <class T>
class InterfaceTemplate
{
public:

	static std::set<std::string> parametersUploadedWithDataType(){
		return {"-ti","-di","-nl"};
	}

	static std::set<std::string> parametersUploadedWithoutDataType(){
		return {"-ki","-sp","-ii","-mi","-ni","-kii","-kvi","-rmi","-rti","-smi","-sti"};
	}

	static std::set<std::string> jsonParametersUploaded(){
		return {"-jg","-job_grid_json","-endpoint_grid_json","-di_grid_json","-eg"};
	}

	virtual unsigned anyNativeToUnsigned(std::any val)=0;
	virtual float anyNativeToFloat(std::any val)=0;
	virtual double anyNativeToDouble(std::any val)=0;
	virtual long unsigned anyNativeToLongUnsigned(std::any val)=0;

	virtual void unlockThread(){}
	virtual void lockThread(){}
	// Bindings may detach their interpreter thread around blocking socket I/O.
	virtual bool sendMessage(zmq::socket_t& socket, zmq::message_t& message){
		return bool(socket.send(message, zmq::send_flags::none));
	}
	virtual bool receiveMessage(zmq::socket_t& socket, zmq::message_t& message){
		return bool(socket.recv(message));
	}
	virtual void updateDisplay(){};
	virtual bool userRequestInteruption(){return false;}

	virtual bool isDataMatrix(std::any val)=0;
	virtual bool shouldUploadWithoutDataType(const std::string& key, std::any val){
		(void)key;
		return isDataMatrix(val);
	}

	virtual std::string nativeToStandardString(std::any val)=0;
	virtual double nativeToScalar(std::any val)=0;
	virtual unsigned nativeToUint32(std::any val)=0;

	virtual std::any ScalarToNative(double val)=0;
	virtual std::any Uint32ToNative(unsigned val)=0;
	virtual std::any StringToNative(const std::string& val)=0;
	virtual void sendError(std::string val)=0;
	virtual void sendWarning(std::string val)=0;
	virtual void eraseAndPrint(std::string val)=0;
	virtual void printMessage(std::string val){ printf("%s\n", val.c_str()); }
	virtual std::any keyValueMapToNative(const std::map<std::string,std::string>& values)=0;
	virtual std::any anyMapToNative(const std::map<std::string,std::any>& values)=0;
	virtual bool encodeJobGridMatrixToJsonString(std::any matrix, std::string &jsonValue){return false;}

	std::string toString(std::any val){
		if(val.type()==typeid(nullptr))
			return "";
		if(val.type()==typeid(std::string))
			return std::any_cast<std::string>(val);
		if(isDataMatrix(val)){
			sendError("matrix argument was not uploaded before job serialization; check upload plumbing for this parameter");
		}
		return nativeToStandardString(val);
	}

	std::string toStringForParameter(const std::string &key, std::any val){
		if(val.type()==typeid(nullptr))
			return "";
		if(val.type()==typeid(std::string))
			return std::any_cast<std::string>(val);
		if(isDataMatrix(val)){
			sendError("matrix argument for "+key+" was not uploaded before job serialization; check upload plumbing for this parameter");
		}
		return nativeToStandardString(val);
	}

//


	virtual std::any convert2NativeMatrix(g2s::DataImage &im)=0;
	virtual g2s::DataImage convertNativeMatrix2DataImage(std::any matrix, std::any dataTypeVariable=nullptr)=0;

	enum class ReturnFormat {
		Legacy,
		Schema,
	};

	static bool equalsIgnoreCase(const std::string& lhs, const std::string& rhs){
		if(lhs.size()!=rhs.size()) return false;
		for (size_t i = 0; i < lhs.size(); ++i)
		{
			if(std::tolower(static_cast<unsigned char>(lhs[i]))!=std::tolower(static_cast<unsigned char>(rhs[i]))){
				return false;
			}
		}
		return true;
	}

	static ReturnFormat parseReturnFormat(const std::multimap<std::string, std::any>& input){
		if(input.count("-legacy_output")>0) return ReturnFormat::Legacy;
		const auto it=input.find("-returnFormat");
		if(it==input.end()) return ReturnFormat::Schema;
		const std::string value=trimWhitespaceCopyStatic(valueToString(it->second));
		if(equalsIgnoreCase(value, "schema")) return ReturnFormat::Schema;
		return ReturnFormat::Legacy;
	}

	static std::string valueToString(const std::any& value){
		if(value.type()==typeid(std::string)) return std::any_cast<std::string>(value);
		if(value.type()==typeid(const char*)) return std::string(std::any_cast<const char*>(value));
		if(value.type()==typeid(char*)) return std::string(std::any_cast<char*>(value));
		return "";
	}

	static std::string trimWhitespaceCopyStatic(const std::string &value){
		size_t begin=0;
		size_t end=value.size();
		while(begin<end && std::isspace(static_cast<unsigned char>(value[begin]))){
			++begin;
		}
		while(end>begin && std::isspace(static_cast<unsigned char>(value[end-1]))){
			--end;
		}
		return value.substr(begin,end-begin);
	}

	static std::string schemaSafeKey(const std::string& rawValue, const std::string& fallbackBase, unsigned fallbackIndex=0){
		std::string key;
		key.reserve(rawValue.size());
		for (size_t i = 0; i < rawValue.size(); ++i)
		{
			const unsigned char ch=static_cast<unsigned char>(rawValue[i]);
			if(std::isalnum(ch) || ch=='_'){
				key.push_back(static_cast<char>(std::tolower(ch)));
			}else if(ch=='-' || std::isspace(ch) || ch=='.'){
				key.push_back('_');
			}
		}
		while(!key.empty() && key.front()=='_'){
			key.erase(key.begin());
		}
		while(!key.empty() && key.back()=='_'){
			key.pop_back();
		}
		if(key.empty()){
			key=fallbackBase;
			if(fallbackIndex>0) key+="_"+std::to_string(fallbackIndex);
		}
		if(std::isdigit(static_cast<unsigned char>(key.front()))){
			key=fallbackBase+"_"+key;
		}
		return key;
	}

#ifdef G2S_ENABLE_BROWSER_TRANSPORT
	void runBrowserCommunication(std::multimap<std::string, std::any> input,
		std::multimap<std::string, std::any> &outputs,
		int maxOutput,
		bool returnMeta,
		bool silentMode,
		int configuredTimeout,
		bool timeoutWasSpecified,
		ReturnFormat returnFormat){
		std::string algorithm;
		if(input.count("-a")>0) algorithm=nativeToStandardString(input.find("-a")->second);
		std::transform(algorithm.begin(),algorithm.end(),algorithm.begin(),::tolower);
		if(algorithm!="qs" && algorithm!="quicksampling"){
			sendError("browser mode currently supports only the QS algorithm");
		}

		const std::set<std::string> unsupported={
			"-W_GPU","-W_CUDA","-jg","-job_grid_json","-eg","-endpoint_grid_json","-di_grid_json",
			"-adsim","-as","-submitOnly","-statusOnly","-waitAndDownload","-kill","-after"
		};
		for(const std::string& key:unsupported){
			if(input.count(key)>0) sendError("browser mode does not support "+key);
		}
		if(input.count("-ti")==0 || input.count("-di")!=1){
			sendError("browser QS requires at least one -ti array and exactly one -di array");
		}

		g2s::browser::Configuration configuration;
		configuration.allowedOrigin=G2S_BROWSER_ORIGIN;
		if(input.count("-browserOrigin")>0){
			configuration.allowedOrigin=nativeToStandardString(input.find("-browserOrigin")->second);
		}
		if(input.count("-browserPort")>0){
			const double requestedPort=nativeToScalar(input.find("-browserPort")->second);
			if(requestedPort<1 || requestedPort>65535) sendError("-browserPort must be between 1 and 65535");
			configuration.port=static_cast<unsigned short>(requestedPort);
		}
		configuration.timeout=std::chrono::milliseconds(timeoutWasSpecified ? configuredTimeout : 30000);
		if(configuration.timeout.count()<1) sendError("browser communication timeout must be positive");

		g2s::browser::Job job;
		Json::Value manifest(Json::objectValue);
		manifest["protocolVersion"]=1;
		manifest["algorithm"]="qs";
		manifest["parameters"]=Json::Value(Json::objectValue);
		manifest["arrays"]=Json::Value(Json::arrayValue);
		Json::Value& parameters=manifest["parameters"];
		Json::Value& arrayManifest=manifest["arrays"];
		const auto dataTypeIterator=input.find("-dt");
		const std::set<std::string> typedParameters=parametersUploadedWithDataType();
		const std::set<std::string> untypedParameters=parametersUploadedWithoutDataType();
		const std::set<std::string> controlParameters={
			"-a","-sa","-TO","-noTO","-browserOrigin","-browserPort","-returnMeta","-showLogs","-silent","-dt"
		};

		unsigned arrayIndex=0;
		for(auto iterator=input.begin();iterator!=input.end();++iterator){
			const std::string& key=iterator->first;
			if(controlParameters.count(key)>0) continue;
			if(isDataMatrix(iterator->second)){
				std::any dataTypes=nullptr;
				if(typedParameters.count(key)>0){
					if(dataTypeIterator==input.end()) sendError("-dt is required for browser array "+key);
					dataTypes=dataTypeIterator->second;
				}
				g2s::DataImage image=convertNativeMatrix2DataImage(iterator->second,dataTypes);
				g2s::browser::ArrayPayload payload;
				payload.id="array_"+std::to_string(arrayIndex++);
				payload.parameter=key;
				payload.dimensions=image._dims;
				payload.variableTypes.reserve(image._types.size());
				for(g2s::DataImage::VariableType type:image._types) payload.variableTypes.push_back(static_cast<unsigned>(type));
				payload.values.assign(image._data,image._data+image.dataSize());

				Json::Value description(Json::objectValue);
				description["id"]=payload.id;
				description["parameter"]=payload.parameter;
				description["encoding"]="float32";
				description["elementCount"]=Json::UInt64(payload.values.size());
				description["dimensions"]=Json::Value(Json::arrayValue);
				for(unsigned dimension:payload.dimensions) description["dimensions"].append(dimension);
				description["variableTypes"]=Json::Value(Json::arrayValue);
				for(unsigned type:payload.variableTypes) description["variableTypes"].append(type);
				arrayManifest.append(description);
				if(!parameters.isMember(key)) parameters[key]=Json::Value(Json::arrayValue);
				Json::Value reference(Json::objectValue);
				reference["arrayId"]=payload.id;
				parameters[key].append(reference);
				job.arrays.push_back(std::move(payload));
			}else{
				if((typedParameters.count(key)>0 || untypedParameters.count(key)>0) && key!="-nl"){
					sendError("browser mode requires an in-memory array for "+key);
				}
				if(!parameters.isMember(key)) parameters[key]=Json::Value(Json::arrayValue);
				parameters[key].append(toStringForParameter(key,iterator->second));
			}
		}

		Json::StreamWriterBuilder builder;
		builder["indentation"]="";
		job.manifestJson=Json::writeString(builder,manifest);

		g2s::browser::Transport transport;
		g2s::browser::Callbacks callbacks;
		callbacks.interrupted=[this](){ return userRequestInteruption(); };
		callbacks.updateDisplay=[this](){ updateDisplay(); };
		callbacks.progress=[this,silentMode](float percent,const std::string& message){
			if(silentMode) return;
			std::ostringstream stream;
			stream << "browser QS: " << std::fixed << std::setprecision(1) << percent << "%";
			if(!message.empty()) stream << " " << message;
			eraseAndPrint(stream.str());
		};

		g2s::browser::Result browserResult;
		std::string error;
		unlockThread();
		const bool ok=transport.run(job,configuration,callbacks,browserResult,error);
		lockThread();
		if(!ok) sendError(error);
		auto threadWarning=browserResult.metadata.find("thread_warning");
		if(threadWarning!=browserResult.metadata.end() && !silentMode) sendWarning(threadWarning->second);
		if(browserResult.arrays.count("simulation")==0 || browserResult.arrays.count("index")==0){
			sendError("browser QS response is missing simulation or index output");
		}

		auto addOutput=[&](const std::string& name,const std::string& outputKey){
			auto found=browserResult.arrays.find(name);
			if(found==browserResult.arrays.end()) return;
			const g2s::browser::ArrayPayload& payload=found->second;
			g2s::DataImage image(payload.dimensions.size(),const_cast<unsigned*>(payload.dimensions.data()),payload.variableTypes.size());
			for(size_t index=0;index<payload.variableTypes.size();++index){
				image._types[index]=payload.variableTypes[index]==1 ? g2s::DataImage::Categorical : g2s::DataImage::Continuous;
			}
			if(payload.encoding=="uint32") image.setEncoding(g2s::DataImage::UInteger);
			if(payload.encoding=="int32") image.setEncoding(g2s::DataImage::Integer);
			std::memcpy(image._data,payload.values.data(),payload.values.size()*sizeof(float));
			outputs.insert({outputKey,convert2NativeMatrix(image)});
		};
		if(maxOutput>0) addOutput("simulation","1");
		if(maxOutput>1) addOutput("index","2");
		outputs.insert({"t",ScalarToNative(browserResult.durationMs/1000.0)});
		if(returnMeta) outputs.insert({"meta",keyValueMapToNative(browserResult.metadata)});

		if(returnFormat==ReturnFormat::Schema){
			// Browser jobs have no server-side artifacts, so build the schema directly
			// from the in-memory result instead of going through buildSchemaResult.
			std::map<std::string,std::any> schema;
			auto simulation=outputs.find("1");
			if(simulation!=outputs.end()) schema["simulation"]=simulation->second;
			auto index=outputs.find("2");
			if(index!=outputs.end()) schema["indexmap"]=index->second;
			schema["time"]=ScalarToNative(browserResult.durationMs/1000.0);
			auto status=browserResult.metadata.find("status");
			schema["status"]=StringToNative(status!=browserResult.metadata.end() ? status->second : std::string("success"));
			schema["progress"]=ScalarToNative(100.0);
			for (auto it=browserResult.metadata.begin(); it!=browserResult.metadata.end(); ++it)
			{
				if(isReservedSchemaKey(it->first) || isInternalResultMetadataKey(it->first)) continue;
				if(schema.count(it->first)==0) schema[it->first]=StringToNative(it->second);
			}
			outputs.insert({"schema",anyMapToNative(schema)});
		}
	}
#endif

	void normalizeJsonGridParameter(std::multimap<std::string, std::any> &input,
		const std::vector<std::string> &aliases,
		const std::string &canonicalKey){
		std::vector<std::pair<std::string,std::any> > values;
		auto collectValues=[&](const std::string &key){
			auto range=input.equal_range(key);
			for (auto it=range.first; it!=range.second; ++it)
			{
				values.push_back(std::make_pair(key,it->second));
			}
			input.erase(key);
		};

		for (size_t i = 0; i < aliases.size(); ++i)
		{
			collectValues(aliases[i]);
		}

		for (size_t i = 0; i < values.size(); ++i)
		{
			if(isDataMatrix(values[i].second)){
				std::string jsonValue;
				if(!encodeJobGridMatrixToJsonString(values[i].second,jsonValue)){
					sendError("failed to encode "+values[i].first+" matrix into JSON");
				}
				input.insert(std::pair<std::string,std::any>(canonicalKey,jsonValue));
			}else{
				input.insert(std::pair<std::string,std::any>(canonicalKey,values[i].second));
			}
		}
	}

	void normalizeJsonGridParameters(std::multimap<std::string, std::any> &input){
		normalizeJsonGridParameter(input, {"-jg","-job_grid_json","-job_grid"}, "-jg");
		normalizeJsonGridParameter(input, {"-eg","-endpoint_grid_json"}, "-eg");
		normalizeJsonGridParameter(input, {"-di_grid_json"}, "-di_grid_json");
	}

		//to improuve
	std::string uploadData(zmq::socket_t &socket, std::any matrix, std::any dataTypeVariable=nullptr){
		bool withTimeout=false;
		char sourceName[65]={0};

		char* rawData=convertNativeMatrix2DataImage(matrix, dataTypeVariable).serialize();
		size_t fullsize=*((size_t*)rawData);
		std::vector<unsigned char> hash(32);
		picosha2::hash256((unsigned char*)rawData, ((unsigned char*)rawData)+fullsize, hash.begin(), hash.end());

		//check existance
		infoContainer task;
		task.version=1;
		task.task=EXIST;

		zmq::message_t request( sizeof(infoContainer) + 64 * sizeof(unsigned char));
		char * positionInTheStream=(char*)request.data();

		char hashInHexa[65]={0};
		for (int i = 0; i < 32; ++i)
		{
			snprintf(hashInHexa+2*i,65-2*i,"%02x",hash.data()[i]);
		}
		memcpy(sourceName,hashInHexa,65*sizeof(char));

		memcpy (positionInTheStream, &task, sizeof(infoContainer));
		positionInTheStream+=sizeof(infoContainer);
		memcpy (positionInTheStream, hashInHexa, 64 * sizeof(unsigned char));
		positionInTheStream+=64 * sizeof(unsigned char);

		if(!sendMessage(socket, request) && withTimeout){
			sendError("timeout sending data");
		}

		zmq::message_t reply;
		if(!receiveMessage(socket, reply) && withTimeout){
			sendError("timeout receive data");
		}
		if(reply.size()!=sizeof(int)) sendError("wrong answer if data exist!");
		int isPresent=*((int*)reply.data());

		//upload if needed
		if(!isPresent){
			infoContainer task;
			task.version=1;
			task.task=UPLOAD;

			zmq::message_t request( sizeof(infoContainer) + 64 * sizeof(unsigned char) + fullsize);
			char * positionInTheStream=(char*)request.data();

			memcpy (positionInTheStream, &task, sizeof(infoContainer));
			positionInTheStream+=sizeof(infoContainer);
			memcpy (positionInTheStream, hashInHexa, 64 * sizeof(unsigned char));
			positionInTheStream+=64 * sizeof(unsigned char);
			memcpy (positionInTheStream, rawData, fullsize);
			positionInTheStream+=fullsize;
			

			if(!sendMessage(socket, request) && withTimeout){
				sendError("timeout sending data");
			}

			zmq::message_t reply;
			if(!receiveMessage(socket, reply) && withTimeout){
				sendError("timeout receive data");
			}
			if(reply.size()!=sizeof(int)) printf( "%s\n", "wrong answer in upload!");
			if(*((int*)reply.data())!=0){
				if(*((int*)reply.data())==1) {
					printf( "%s\n", "data present before upload!");
				}
				else {
					printf( "%s code : %d\n", "error in upload data!",*((int*)reply.data()));
				}
			}
		}
		//clean
		free(rawData);

		return std::string(sourceName);
	}

	static std::string trimWhitespaceCopy(const std::string &value){
		return trimWhitespaceCopyStatic(value);
	}

	static bool isInlineJsonPayload(const std::string &value){
		const std::string trimmed=trimWhitespaceCopy(value);
		if(trimmed.empty()){
			return false;
		}
		const char first=trimmed.front();
		return (first=='[' || first=='{');
	}

	static bool isReservedSchemaKey(const std::string& key){
		return key=="time" || key=="job_id" || key=="status" ||
			key=="progress" || key=="artifacts" || key=="error" || key=="warnings";
	}

	static bool isInternalResultMetadataKey(const std::string& key){
		return key.rfind("result_output_",0)==0;
	}

	static std::map<unsigned,std::string> parseOutputSemanticNames(const std::map<std::string,std::string>& values){
		std::map<unsigned,std::string> names;
		for (auto it=values.begin(); it!=values.end(); ++it)
		{
			unsigned index=0;
			if(sscanf(it->first.c_str(),"result_output_%u_name",&index)==1 && index>0){
				names[index]=it->second;
			}
		}
		return names;
	}

	std::any buildSchemaResult(
		const std::multimap<std::string,std::any>& outputs,
		const std::map<std::string,std::string>& finalMeta,
		const std::map<std::string,std::string>& progressSnapshot,
		jobIdType id,
		bool waitAndDownload,
		bool statusOnly,
		float lastProgression,
		const std::string& logArtifactName,
		const std::string& warningArtifactName,
		const std::string& errorArtifactName,
		const std::string& progressArtifactName,
		const std::string& metaArtifactName) {

		std::map<std::string,std::any> schema;
		std::set<std::string> usedKeys;
		std::map<std::string,std::any> artifacts;
		artifacts["log"]=StringToNative(logArtifactName);
		artifacts["warning"]=StringToNative(warningArtifactName);
		artifacts["error"]=StringToNative(errorArtifactName);
		artifacts["progress"]=StringToNative(progressArtifactName);
		artifacts["meta"]=StringToNative(metaArtifactName);

		const std::map<unsigned,std::string> semanticNames=parseOutputSemanticNames(finalMeta);
		unsigned outputCount=0;
		for (auto it=outputs.begin(); it!=outputs.end(); ++it)
		{
			unsigned outputIndex=0;
			if(sscanf(it->first.c_str(),"%u",&outputIndex)==1 && outputIndex>0){
				++outputCount;
				std::string preferredName;
				const auto nameIter=semanticNames.find(outputIndex);
				if(nameIter!=semanticNames.end()){
					preferredName=nameIter->second;
				}else if(outputIndex==1){
					preferredName="simulation";
				}else if(outputIndex==2){
					preferredName="indexmap";
				}
				std::string key=schemaSafeKey(preferredName, "output", outputIndex);
				while(usedKeys.count(key)>0 || isReservedSchemaKey(key)){
					key=schemaSafeKey("output_"+std::to_string(outputIndex), "output", outputIndex);
					if(usedKeys.count(key)==0 && !isReservedSchemaKey(key)) break;
					key+="_"+std::to_string(outputIndex);
				}
				if(outputIndex==1 && preferredName.empty() && !usedKeys.count("simulation")){
					key="simulation";
				}
				usedKeys.insert(key);
				schema[key]=it->second;
				artifacts[key]=StringToNative("im_"+std::to_string(outputIndex)+"_"+std::to_string(id));
			}
		}

		if(finalMeta.count("duration_ms")>0){
			schema["time"]=ScalarToNative(atof(finalMeta.find("duration_ms")->second.c_str())/1000.0);
		}
		schema["job_id"]=Uint32ToNative(id);

		std::string statusValue;
		if(finalMeta.count("status")>0) statusValue=finalMeta.find("status")->second;
		else if(progressSnapshot.count("status")>0) statusValue=progressSnapshot.find("status")->second;
		else if(statusOnly) statusValue="running";
		else if(!waitAndDownload) statusValue="submitted";
		if(!statusValue.empty()){
			schema["status"]=StringToNative(statusValue);
		}

		if(progressSnapshot.count("progress_percent")>0){
			schema["progress"]=ScalarToNative(atof(progressSnapshot.find("progress_percent")->second.c_str()));
		}else if(lastProgression>=0.f){
			schema["progress"]=ScalarToNative(lastProgression);
		}

		schema["artifacts"]=anyMapToNative(artifacts);

		for (auto it=progressSnapshot.begin(); it!=progressSnapshot.end(); ++it)
		{
			if(isReservedSchemaKey(it->first) || isInternalResultMetadataKey(it->first)) continue;
			if(schema.count(it->first)==0){
				schema[it->first]=StringToNative(it->second);
			}
		}
		for (auto it=finalMeta.begin(); it!=finalMeta.end(); ++it)
		{
			if(isReservedSchemaKey(it->first) || isInternalResultMetadataKey(it->first)) continue;
			if(schema.count(it->first)==0){
				schema[it->first]=StringToNative(it->second);
			}
		}

		return anyMapToNative(schema);
	}

	std::string uploadJson(zmq::socket_t &socket, const std::string &jsonPayload){
		bool withTimeout=false;
		char sourceName[65]={0};
		const size_t payloadSize=jsonPayload.size();
		const unsigned char *payloadBytes=(const unsigned char*)jsonPayload.data();
		std::vector<unsigned char> hash(32);
		picosha2::hash256(payloadBytes,payloadBytes+payloadSize,hash.begin(),hash.end());

		infoContainer task;
		task.version=1;
		task.task=EXIST;

		zmq::message_t request(sizeof(infoContainer)+64*sizeof(unsigned char));
		char * positionInTheStream=(char*)request.data();

		char hashInHexa[65]={0};
		for (int i = 0; i < 32; ++i)
		{
			snprintf(hashInHexa+2*i,65-2*i,"%02x",hash.data()[i]);
		}
		memcpy(sourceName,hashInHexa,65*sizeof(char));

		memcpy(positionInTheStream,&task,sizeof(infoContainer));
		positionInTheStream+=sizeof(infoContainer);
		memcpy(positionInTheStream,hashInHexa,64*sizeof(unsigned char));
		positionInTheStream+=64*sizeof(unsigned char);

		if(!sendMessage(socket, request) && withTimeout){
			sendError("timeout sending json existence request");
		}

		zmq::message_t reply;
		if(!receiveMessage(socket, reply) && withTimeout){
			sendError("timeout receiving json existence response");
		}
		if(reply.size()!=sizeof(int)) sendError("wrong answer if json data exist!");
		const int isPresent=*((int*)reply.data());

		if(isPresent!=2){
			infoContainer uploadTask;
			uploadTask.version=1;
			uploadTask.task=UPLOAD_JSON;

			zmq::message_t uploadRequest(sizeof(infoContainer)+64*sizeof(unsigned char)+payloadSize);
			char * uploadPos=(char*)uploadRequest.data();
			memcpy(uploadPos,&uploadTask,sizeof(infoContainer));
			uploadPos+=sizeof(infoContainer);
			memcpy(uploadPos,hashInHexa,64*sizeof(unsigned char));
			uploadPos+=64*sizeof(unsigned char);
			if(payloadSize>0){
				memcpy(uploadPos,jsonPayload.data(),payloadSize);
				uploadPos+=payloadSize;
			}

			if(!sendMessage(socket, uploadRequest) && withTimeout){
				sendError("timeout sending json upload");
			}

			zmq::message_t uploadReply;
			if(!receiveMessage(socket, uploadReply) && withTimeout){
				sendError("timeout receive json upload");
			}
			if(uploadReply.size()!=sizeof(int)) sendError("wrong answer in json upload");
			if(*((int*)uploadReply.data())!=0 && *((int*)uploadReply.data())!=1){
				sendError("error in json upload");
			}
		}
		return std::string(sourceName);
	}

	void lookForUpload(zmq::socket_t &socket, std::multimap<std::string, std::any> &input){

		auto dataTypeVariable=input.find("-dt");
		const std::set<std::string> listOfParameterToUploadIfNeededWithdataTypeVariable=parametersUploadedWithDataType();
		const std::set<std::string> listOfParameterToUploadIfNeededWithoutdataTypeVariable=parametersUploadedWithoutDataType();
		const std::set<std::string> listOfJsonParameterToUploadIfNeeded=jsonParametersUploaded();
		for (auto it=input.begin(); it!=input.end(); ++it){
			if(listOfParameterToUploadIfNeededWithdataTypeVariable.find(it->first) != listOfParameterToUploadIfNeededWithdataTypeVariable.end())
				if(isDataMatrix(it->second)){
					if(dataTypeVariable!=input.end())
						it->second=std::any(uploadData(socket, it->second,dataTypeVariable->second));
					else
						sendError("-dt is missing, impossible to uplaod a matrix without data type");
			}
			if(listOfParameterToUploadIfNeededWithoutdataTypeVariable.find(it->first) != listOfParameterToUploadIfNeededWithoutdataTypeVariable.end()){
				if(shouldUploadWithoutDataType(it->first,it->second)){
					const std::string uploadedName=uploadData(socket, it->second);
					if(uploadedName.empty() && (it->first=="-rmi" || it->first=="-smi")){
						sendError("failed to upload transform map for "+it->first);
					}
					it->second=std::any(uploadedName);
				}
			}
			if(listOfJsonParameterToUploadIfNeeded.find(it->first) != listOfJsonParameterToUploadIfNeeded.end()){
				const std::string payload=toStringForParameter(it->first,it->second);
				if(isInlineJsonPayload(payload)){
					const std::string jsonHash=uploadJson(socket,payload);
					it->second=std::any(std::string("/tmp/G2S/data/")+jsonHash+std::string(".json"));
				}
			}
		}
		input.erase("-dt");
	}

	void uploadRemainingMatrices(zmq::socket_t &socket, std::multimap<std::string, std::any> &input){
		for (auto it=input.begin(); it!=input.end(); ++it)
		{
			if(isDataMatrix(it->second)){
				const std::string uploadedName=uploadData(socket,it->second);
				if(uploadedName.empty()){
					sendError("failed to upload matrix argument for "+it->first);
				}
				it->second=std::any(uploadedName);
			}
		}
	}

	inline void sendKill(zmq::socket_t &socket, jobIdType id, bool silent=false){
		infoContainer task;
		task.version=1;
		task.task=KILL;
			
		zmq::message_t request (sizeof(infoContainer)+sizeof(jobIdType));
		memcpy(request.data (), &task, sizeof(infoContainer));
		memcpy((char*)request.data()+sizeof(infoContainer),&id,sizeof(jobIdType));
		sendMessage(socket, request);
		zmq::message_t reply;
		receiveMessage(socket, reply);
		
		//if(!silent)sendError("Ctrl C, user interrupted");
	}

	std::string downloadTextArtifact(zmq::socket_t &socket, const std::string& artifactName, bool withTimeout){
		infoContainer task;
		task.version=1;
		task.task=DOWNLOAD_TEXT;
		zmq::message_t request(sizeof(infoContainer)+64);
		memset((char*)request.data()+sizeof(infoContainer), 0, 64);
		memcpy(request.data(), &task, sizeof(infoContainer));
		memcpy((char*)request.data()+sizeof(infoContainer), artifactName.c_str(), std::min<size_t>(artifactName.size(), 63));
		if(!sendMessage(socket, request) && withTimeout){
			sendError("timeout asking for text artifact");
		}
		zmq::message_t reply;
		if(!receiveMessage(socket, reply) && withTimeout){
			sendError("timeout receiving text artifact");
		}
		if(reply.size()==0) return "";
		return std::string((char*)reply.data(), reply.size());
	}

	bool textArtifactExists(zmq::socket_t &socket, const std::string& artifactName, bool withTimeout){
		infoContainer task;
		task.version=1;
		task.task=EXIST;
		zmq::message_t request(sizeof(infoContainer)+64);
		memset((char*)request.data()+sizeof(infoContainer), 0, 64);
		memcpy(request.data(), &task, sizeof(infoContainer));
		memcpy((char*)request.data()+sizeof(infoContainer), artifactName.c_str(), std::min<size_t>(artifactName.size(), 63));
		if(!sendMessage(socket, request) && withTimeout){
			sendError("timeout checking text artifact");
		}
		zmq::message_t reply;
		if(!receiveMessage(socket, reply) && withTimeout){
			sendError("timeout receiving text artifact status");
		}
		return reply.size()==sizeof(int) && *((int*)reply.data())==3;
	}

	static std::map<std::string,std::string> parseKeyValueText(const std::string& text){
		std::map<std::string,std::string> values;
		std::istringstream stream(text);
		std::string line;
		while(std::getline(stream, line)){
			if(line.empty()) continue;
			const size_t pos=line.find('=');
			if(pos==std::string::npos) continue;
			values[line.substr(0,pos)]=line.substr(pos+1);
		}
		return values;
	}

	void printNewTextChunk(const std::string& content, size_t& offset, bool warningChannel){
		if(content.size()<offset){
			offset=0;
		}
		if(content.size()==offset) return;
		std::string delta=content.substr(offset);
		offset=content.size();
		std::istringstream stream(delta);
		std::string line;
		while(std::getline(stream, line)){
			if(line.empty()) continue;
			if(warningChannel){
				const size_t messagePos=line.find("message=");
				if(messagePos!=std::string::npos){
					line=line.substr(messagePos+8);
				}
				sendWarning(line);
			}else{
				printMessage(line);
			}
		}
	}

	void runStandardCommunication(std::multimap<std::string, std::any> input, std::multimap<std::string, std::any> &outputs, int maxOutput=INT_MAX){

		std::atomic<bool> done(false);
		normalizeJsonGridParameters(input); // canonicalize JSON grid keys before upload/serialization

		jobIdType id=0;
		bool stop=false;
		bool withTimeout=true;
		int timeout=60000;
		bool noOutput=false;
		bool spectifiedTimeout=false;

		bool submit=true;
		bool statusOnly=false;
		bool waitAndDownload=true;
		bool kill=false;
		bool serverShutdown=false;
		bool silentMode=false;
		bool requestServerStatus=false; // to check programmatically if the server is running
		bool requestServerVersion=false;
		bool showLogs=false;
		bool returnMeta=false;
		ReturnFormat returnFormat=ReturnFormat::Schema;

		if(input.count("-noTO")>0)
		{
			withTimeout=false;
		}
		if(input.count("-TO")>0)
		{
			timeout=anyNativeToUnsigned(input.find("-TO")->second);
			spectifiedTimeout=true;
		}
		if(input.count("-submitOnly")>0)
		{
			waitAndDownload=false;
		}
		if(input.count("-shutdown")>0)
		{
			serverShutdown=true;
			waitAndDownload=false;
			noOutput=true;
		}
		if(input.count("-silent")>0)
		{
			silentMode=true;
		}
		if(input.count("-showLogs")>0)
		{
			showLogs=true;
		}
		if(input.count("-returnMeta")>0)
		{
			returnMeta=true;
		}
		if(input.count("-returnFormat")>0)
		{
			returnFormat=equalsIgnoreCase(trimWhitespaceCopy(toString(input.find("-returnFormat")->second)), "schema") ?
				ReturnFormat::Schema : ReturnFormat::Legacy;
		}
		if(input.count("-legacy_output")>0)
		{
			returnFormat=ReturnFormat::Legacy;
		}
		const int outputDownloadLimit=(returnFormat==ReturnFormat::Schema ? INT_MAX : maxOutput);
		input.erase("-showLogs");
		input.erase("-returnMeta");
		input.erase("-returnFormat");
		input.erase("-legacy_output");

		if(input.count("-statusOnly")>0){
			statusOnly=true;
			waitAndDownload=false;
			submit=false;
			input.insert(std::pair<std::string, std::any>("-id",input.find("-statusOnly")->second));					
		}
		if(input.count("-waitAndDownload")>0){
			submit=false;
			input.insert(std::pair<std::string, std::any>("-id",input.find("-waitAndDownload")->second));
		}
		if(input.count("-kill")>0){
			kill=true;
			submit=false;
			input.insert(std::pair<std::string, std::any>("-id",input.find("-kill")->second));
		}

		if(input.count("-serverStatus")>0){
			requestServerStatus=true;
		}
		if(input.count("-serverVersion")>0){
			requestServerVersion=true;
		}

		std::string serverAddress="localhost";
		if(input.count("-sa")>0)serverAddress=nativeToStandardString(input.find("-sa")->second);
		input.erase("-sa");
		std::transform(serverAddress.begin(), serverAddress.end(), serverAddress.begin(),::tolower);
		if(serverAddress=="browser" || serverAddress=="web"){
		#ifdef G2S_ENABLE_BROWSER_TRANSPORT
			if(!submit || !waitAndDownload || kill || serverShutdown || requestServerStatus){
				sendError("browser mode supports synchronous submissions only");
			}
			runBrowserCommunication(input,outputs,outputDownloadLimit,returnMeta,silentMode,timeout,spectifiedTimeout,returnFormat);
			done=true;
			return;
		#else
			sendError("this interface build does not include browser transport support");
			return;
		#endif
		}

		zmq::context_t context (1);
		zmq::socket_t socket (context, ZMQ_REQ);
		
		socket.setsockopt(ZMQ_LINGER, 500);
		if(withTimeout){
			socket.setsockopt(ZMQ_RCVTIMEO, 500);
			socket.setsockopt(ZMQ_SNDTIMEO, 500);
		}

		if(spectifiedTimeout){
			socket.setsockopt(ZMQ_LINGER, timeout);
			socket.setsockopt(ZMQ_RCVTIMEO, timeout);
			socket.setsockopt(ZMQ_SNDTIMEO, timeout);
		}

		short port=8128;
		if(input.count("-p")>0)port=nativeToScalar(input.find("-p")->second);

		char address[4096];
		snprintf(address,4096, "tcp://%s:%d",serverAddress.c_str(),port);
		socket.connect (address);

		if (requestServerStatus){

			int serverStatus=0;

			infoContainer task;
			task.version=1;
			task.task=SERVER_STATUS;

			zmq::message_t request (sizeof(infoContainer));
			memcpy(request.data (), &task, sizeof(infoContainer));
			if(!sendMessage(socket, request) ){
				serverStatus=-1;
			}
			zmq::message_t reply;
			if(serverStatus==0 && !receiveMessage(socket, reply) ){
				serverStatus=-2;
			}
			if(serverStatus==0 && reply.size()==sizeof(int))
				serverStatus=*((int*)reply.data());
			outputs.insert({"1",ScalarToNative(double(serverStatus))});
			done=true;
			return;
		}

		if (requestServerVersion){

			std::string serverVersion;

			infoContainer task;
			task.version=1;
			task.task=SERVER_VERSION;

			zmq::message_t request (sizeof(infoContainer));
			memcpy(request.data (), &task, sizeof(infoContainer));
			if(!sendMessage(socket, request) && withTimeout ){
				done=true;
				sendError("The server is probably off-line, please execute first ./server. If you try to connect to a remote server maybe the network connection  ");
			}
			zmq::message_t reply;
			if(!receiveMessage(socket, reply) && withTimeout){
				done=true;
				sendError("The server is probably off-line, please execute first ./server. If you try to connect to a remote server maybe the network connection  ");
			}
			if(reply.size()==sizeof(int)){
				int serverReply=*((int*)reply.data());
				if(serverReply<0){
					done=true;
					sendError("The connected server does not support -serverVersion. Rebuild or update the server first.");
				}
			}
			serverVersion=std::string((char*)reply.data(), reply.size());
			outputs.insert({"1",StringToNative(serverVersion)});
			done=true;
			return;
		}


		{// check if server is available
			infoContainer task;
			task.version=1;
			task.task=SERVER_STATUS;

			zmq::message_t request (sizeof(infoContainer));
			memcpy(request.data (), &task, sizeof(infoContainer));
			if(!sendMessage(socket, request) && withTimeout ){
				done=true;
				sendError("The server is probably off-line, please execute first ./server. If you try to connect to a remote server maybe the network connection  ");
			}
			zmq::message_t reply;
			if(!receiveMessage(socket, reply) && withTimeout){
				done=true;
				sendError("The server is probably off-line, please execute first ./server. If you try to connect to a remote server maybe the network connection  ");
			}
			if(reply.size()!=sizeof(int)) sendError("wrong answer!");
			int serverStatus=*((int*)reply.data());
			if(serverStatus<1){
				sendError("something wrong on the server !");
			}
		}

		socket.setsockopt(ZMQ_LINGER, timeout);
		socket.setsockopt(ZMQ_RCVTIMEO, timeout);
		socket.setsockopt(ZMQ_SNDTIMEO, timeout);

		if(serverShutdown){
			infoContainer task;
			task.version=1;
			task.task=SHUTDOWN;

			zmq::message_t request (sizeof(infoContainer));
			memcpy(request.data (), &task, sizeof(infoContainer));
			if(!sendMessage(socket, request) && withTimeout ){
				{
					done=true;
					sendError("fail to shutdown the server");
				}
			}
			done=true;
			return;
		}


		if (submit) {
			lookForUpload(socket, input);
			uploadRemainingMatrices(socket, input);
		}

		// start the process
		
		//std::this_thread::sleep_for(std::chrono::seconds(10));

		if(!done && submit){
			infoContainer task;
			task.version=1;

			Json::Value object(Json::objectValue);
			std::string test;
			if(input.count("-a")>0) object["Algorithm"]=nativeToStandardString(input.find("-a")->second);
			else if(!silentMode) sendWarning("Upload only! You maybe want to use '-a' to specify an algorithm.");
			input.erase("-a");
			{
				object["Priority"]="1";
				{
					Json::Value jsonArray(Json::arrayValue);
					for (auto it=input.equal_range("-after").first; it!=input.equal_range("-after").second; ++it)
					{
						jsonArray.append(anyNativeToUnsigned(it->second));
					}
					input.erase("-after");
					object["Dependency"]=jsonArray;
				}
				
				Json::Value parameter(Json::objectValue);

				std::set<std::string> listOfAllKeys;
				for (auto it = input.begin(); it != input.end(); ++it)
				{
					listOfAllKeys.insert(it->first);					
				}

				for (auto itKey = listOfAllKeys.begin(); itKey != listOfAllKeys.end(); ++itKey)
				{	
					Json::Value jsonArray(Json::arrayValue);	
					for (auto it=input.equal_range(*itKey).first; it!=input.equal_range(*itKey).second; ++it)
					{
						jsonArray.append(toStringForParameter(*itKey,it->second));
					}
					parameter[*itKey]=jsonArray;
				}

				object["Parameter"]=parameter;
			}
			/*
			Json::FastWriter fast;
		    std::string jsonJob = fast.write(object);
			*/
			Json::StreamWriterBuilder builder;
			builder["commentStyle"] = "None";
			builder["indentation"] = "";  // or whatever you like
			std::string jsonJob = Json::writeString(builder, object);
			const char* jsonJob_c=jsonJob.c_str();

		    //printf( "%s\n", jsonJob_c);

			task.task=JOB;

			zmq::message_t request (sizeof(infoContainer)+strlen(jsonJob_c));
			memcpy(request.data (), &task, sizeof(infoContainer));
			memcpy((char*)request.data()+sizeof(infoContainer),jsonJob_c,strlen(jsonJob_c));
			if(!sendMessage(socket, request) && withTimeout ){
				sendError("timeout sending job");
			}

			zmq::message_t reply;
			if(!receiveMessage(socket, reply) && withTimeout){
				sendError("timeout starting job, maybe job run on server !!");
			}
			if(reply.size()!=sizeof(jobIdType) && !silentMode) printf( "%s\n", "wrong answer !");
			id=*((jobIdType*)reply.data());
			if(id<0 && !silentMode) printf( "%s\n", "error in job distribution!");
			if(!silentMode)printf("job Id is: %u\n",id );
			updateDisplay();

			if(done) {
				sendKill(socket, id);
				stop=true;
			}
		
		}else if (!submit) {
			id=anyNativeToUnsigned(input.find("-id")->second);
		}

		outputs.insert(std::pair<std::string, std::any>("id",Uint32ToNative(id)));
		unlockThread();
		if(submit && !waitAndDownload){
			noOutput=true;
			done=true;
		}

		float lastProgression=-1.f;

		char nameFile[65]={0};
		snprintf(nameFile,65,"%u",id);
		const std::string progressArtifactName=std::string("progress_")+std::to_string(id);
		const std::string warningArtifactName=std::string("warning_")+std::to_string(id);
		const std::string errorArtifactName=std::string("error_")+std::to_string(id);
		const std::string metaArtifactName=std::string("meta_")+std::to_string(id);
		const std::string logArtifactName=std::string("log_")+std::to_string(id);
		size_t logOffset=0;
		size_t warningOffset=0;
		std::map<std::string,std::string> finalMeta;
		std::map<std::string,std::string> progressSnapshot;

		if(kill){
			sendKill(socket, id, true);
			done=true;
		}

		if(waitAndDownload || statusOnly) outputs.insert({"progression",0.f});

		while(!done && (waitAndDownload || statusOnly)) {

			// status
			if(!silentMode || statusOnly){
				infoContainer task;
				task.version=1;
				task.task=PROGESSION;
				
				zmq::message_t request (sizeof(infoContainer)+sizeof( jobIdType ));
				memcpy(request.data (), &task, sizeof(infoContainer));
				memcpy((char*)request.data()+sizeof(infoContainer),&id,sizeof( jobIdType ));
				if(!sendMessage(socket, request)){
					continue;
				}
				zmq::message_t reply;
				if(!receiveMessage(socket, reply) && withTimeout){
					continue;
				}

				if(reply.size()!=sizeof(int)) sendWarning( "wrong answer !");
				else{
					int progress=*((int*)reply.data());
					if(textArtifactExists(socket, progressArtifactName, withTimeout)){
						progressSnapshot=parseKeyValueText(downloadTextArtifact(socket, progressArtifactName, withTimeout));
						auto progressIter=progressSnapshot.find("progress_percent");
						if(progressIter!=progressSnapshot.end()){
							progress=int(1000.0*atof(progressIter->second.c_str()));
						}
					}
					if((progress>=0) && fabs(lastProgression-progress/1000.)>0.0001){
						char buff[100];
						snprintf(buff, sizeof(buff), "progress %.3f%%",progress/1000.);
						std::string buffAsStdStr = buff;
						if(!silentMode)eraseAndPrint(buffAsStdStr);
						lastProgression=progress/1000.;
						if(!silentMode)updateDisplay();
					}
					outputs.find("progression")->second=lastProgression;
				}
				if(showLogs && !statusOnly){
					printNewTextChunk(downloadTextArtifact(socket, logArtifactName, withTimeout), logOffset, false);
					if(textArtifactExists(socket, warningArtifactName, withTimeout)){
						printNewTextChunk(downloadTextArtifact(socket, warningArtifactName, withTimeout), warningOffset, true);
					}
				}
				if(statusOnly) break;
			}

			if(userRequestInteruption()){
				kill=true;
				break;
			}

			{
				infoContainer task;
				task.version=1;
				task.task=JOB_STATUS;

				zmq::message_t request(sizeof(infoContainer)+sizeof( jobIdType ));
				memcpy(request.data(), &task, sizeof(infoContainer));
				memcpy((char*)request.data()+sizeof(infoContainer),&id,sizeof( jobIdType ));
				if(!sendMessage(socket, request)){
					continue;
				}
				zmq::message_t reply;
				if(!receiveMessage(socket, reply) && withTimeout){
					continue;
				}

				if(reply.size()!=sizeof(int)){
					sendWarning( "handel error !");
				}else{

					int statusCode=*((int*)reply.data());
					if(statusCode==0){
						char buff[100];
						snprintf(buff, sizeof(buff), "progress %.3f%%\n",100.);
						std::string buffAsStdStr = buff;
						if(!silentMode)eraseAndPrint(buffAsStdStr);
						lastProgression=100.;
						outputs.find("progression")->second=lastProgression;
						break;
					}
					if(statusCode>0){
						// error occurred
						// check for error
						char nameFile_local[65]={0};
						snprintf(nameFile_local,65,"error_%u",id);
					
						{
							infoContainer task;
							task.version=1;
							task.task=EXIST;

							zmq::message_t request( sizeof(infoContainer) + 64 * sizeof(unsigned char));
							char * positionInTheStream=(char*)request.data();

							memcpy (positionInTheStream, &task, sizeof(infoContainer));
							positionInTheStream+=sizeof(infoContainer);
							memcpy (positionInTheStream, nameFile_local, 64 * sizeof(unsigned char));
							positionInTheStream+=64 * sizeof(unsigned char);

							if(!sendMessage(socket, request) && withTimeout){
								sendError("timeout sending data");
							}

							zmq::message_t reply;
							if(!receiveMessage(socket, reply) && withTimeout){
								sendError("timeout receive data");
							}
							if(reply.size()!=sizeof(int)) sendError("wrong answer if data exist!");
							int isPresent=*((int*)reply.data());
							if (!isPresent) break;
						}

						{
							infoContainer task;
							task.version=1;
							task.task=DOWNLOAD_TEXT;

							//printf("%s\n", nameFile);
							zmq::message_t request (sizeof(infoContainer)+64);
							memcpy (request.data(), &task, sizeof(infoContainer));
							memcpy ((char*)request.data()+sizeof(infoContainer), nameFile_local, 64);
							//std::cout << "ask for data" << std::endl;
							if(!sendMessage(socket, request) && withTimeout ){
								sendError("timeout asking for data");
							}
							zmq::message_t reply;
							if(!receiveMessage(socket, reply) && withTimeout){
								sendError("timeout : get data dont answer");
							}
							
							if(reply.size()!=0){
								std::string errorText=std::string((char*)reply.data());
								std::map<std::string,std::string> errorValues=parseKeyValueText(errorText);
								if(errorValues.count("error_message")>0){
									errorText=errorValues["error_message"];
								}
								sendError(errorText);
							}
						}
						break;
					}
				}

			}

			if(userRequestInteruption()){
				kill=true;
				break;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(600));
		}

		if(kill && !done) {
			sendKill(socket, id);
			done=true;
			sendError("canceled job");
		}


		lockThread();
		//id.write(std::string("im_2_")+std::to_string(uniqueID));

		//look for imageData result
		int dataIndex=1;
		int nbElement=0;
		while(waitAndDownload){
			char nameFile_local[65]={0};
			snprintf(nameFile_local,65,"im_%d_%u",dataIndex,id);

			{
				infoContainer task;
				task.version=1;
				task.task=EXIST;

				zmq::message_t request( sizeof(infoContainer) + 64 * sizeof(unsigned char));
				char * positionInTheStream=(char*)request.data();

				memcpy (positionInTheStream, &task, sizeof(infoContainer));
				positionInTheStream+=sizeof(infoContainer);
				memcpy (positionInTheStream, nameFile_local, 64 * sizeof(unsigned char));
				positionInTheStream+=64 * sizeof(unsigned char);

				if(!sendMessage(socket, request) && withTimeout){
					sendError("timeout sending data");
				}

				zmq::message_t reply;
				if(!receiveMessage(socket, reply) && withTimeout){
					sendError("timeout receive data");
				}
				if(reply.size()!=sizeof(int)) sendError("wrong answer if data exist!");
				int isPresent=*((int*)reply.data());
				if (!isPresent) break;
			}

			{
				infoContainer task;
				task.version=1;
				task.task=DOWNLOAD;

				//printf("%s\n", nameFile);
				zmq::message_t request (sizeof(infoContainer)+64);
				memcpy (request.data(), &task, sizeof(infoContainer));
				memcpy ((char*)request.data()+sizeof(infoContainer), nameFile_local, 64);
				//std::cout << "ask for data" << std::endl;
				if(!sendMessage(socket, request) && withTimeout ){
					sendError("timeout asking for data");
				}
				zmq::message_t reply;
				if(!receiveMessage(socket, reply) && withTimeout){
					sendError("timeout : get data dont answer");
				}
				
				if(reply.size()!=0){
					g2s::DataImage image((char*)reply.data(), reply.size());
					outputs.insert({std::to_string(dataIndex),convert2NativeMatrix(image)});
				}
				dataIndex++;
				nbElement++;
			}
			if(outputDownloadLimit-1<nbElement) break;
		}

		
		if(waitAndDownload)
		{
			if(textArtifactExists(socket, metaArtifactName, withTimeout)){
				finalMeta=parseKeyValueText(downloadTextArtifact(socket, metaArtifactName, withTimeout));
			}
			auto durationIter=finalMeta.find("duration_ms");
			if(durationIter!=finalMeta.end()){
				outputs.insert({"t",ScalarToNative(atof(durationIter->second.c_str())/1000.0)});
			}else{
				infoContainer task;
				task.version=1;
				task.task=DURATION;
				
				zmq::message_t request (sizeof(infoContainer)+sizeof( jobIdType ));
				memcpy(request.data (), &task, sizeof(infoContainer));
				memcpy((char*)request.data()+sizeof(infoContainer),&id,sizeof( jobIdType ));
				if(!sendMessage(socket, request)){
					sendError("timeout asking for data");
				}
				zmq::message_t reply;
				if(!receiveMessage(socket, reply) && withTimeout){
					sendError("timeout : get data dont answer");
				}

				if(reply.size()!=sizeof(int))
					printf( "%s\n", "wrong answer !");
				else{
					outputs.insert({"t",ScalarToNative(double(*((int*)reply.data()))/1000.)});				
				}
			}
		}
		if(returnMeta){
			if(finalMeta.empty() && textArtifactExists(socket, metaArtifactName, withTimeout)){
				finalMeta=parseKeyValueText(downloadTextArtifact(socket, metaArtifactName, withTimeout));
			}
			outputs.insert({"meta", keyValueMapToNative(finalMeta)});
		}
		if(returnFormat==ReturnFormat::Schema){
			if(progressSnapshot.empty() && textArtifactExists(socket, progressArtifactName, withTimeout)){
				progressSnapshot=parseKeyValueText(downloadTextArtifact(socket, progressArtifactName, withTimeout));
			}
			if(finalMeta.empty() && textArtifactExists(socket, metaArtifactName, withTimeout)){
				finalMeta=parseKeyValueText(downloadTextArtifact(socket, metaArtifactName, withTimeout));
			}
			outputs.insert({"schema", buildSchemaResult(outputs, finalMeta, progressSnapshot, id, waitAndDownload, statusOnly, lastProgression, logArtifactName, warningArtifactName, errorArtifactName, progressArtifactName, metaArtifactName)});
		}
		done=true;
		
	};
};

#endif // INERFACE_TEMPLATE_HPP
