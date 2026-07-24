/*
 * G2S browser transport
 * SPDX-License-Identifier: LGPL-3.0-only
 */

#include "browserTransport.hpp"

#include "third_party/cpp-httplib/httplib.h"
#include <json/json.h>

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <iomanip>
#include <limits>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

namespace g2s {
namespace browser {

namespace {

std::string randomHex(size_t bytes){
	std::random_device randomDevice;
	std::mt19937_64 generator(randomDevice());
	std::uniform_int_distribution<unsigned> distribution(0,255);
	std::ostringstream stream;
	stream << std::hex << std::setfill('0');
	for(size_t i=0;i<bytes;++i){
		stream << std::setw(2) << distribution(generator);
	}
	return stream.str();
}

std::vector<unsigned> parseUnsignedList(const std::string& text){
	std::vector<unsigned> values;
	std::istringstream stream(text);
	std::string item;
	while(std::getline(stream,item,',')){
		if(item.empty()) return {};
		char* end=nullptr;
		const unsigned long parsed=std::strtoul(item.c_str(),&end,10);
		if(end==item.c_str() || *end!='\0' || parsed>std::numeric_limits<unsigned>::max()) return {};
		values.push_back(static_cast<unsigned>(parsed));
	}
	return values;
}

std::string jsonString(const Json::Value& value){
	Json::StreamWriterBuilder builder;
	builder["indentation"]="";
	return Json::writeString(builder,value);
}

bool parseJson(const std::string& text, Json::Value& value){
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream input(text);
	return Json::parseFromStream(builder,input,&value,&errors);
}

} // namespace

struct Transport::Implementation {
	std::mutex mutex;
	std::condition_variable condition;
	bool connected=false;
	bool complete=false;
	bool cancelRequested=false;
	bool progressPending=false;
	float progressPercent=0.f;
	std::string progressMessage;
	std::string browserError;
	std::chrono::steady_clock::time_point lastHeartbeat;
	Result result;
};

Transport::Transport():implementation_(new Implementation()){}
Transport::~Transport(){ delete implementation_; }

bool Transport::run(const Job& job,
	const Configuration& configuration,
	const Callbacks& callbacks,
	Result& result,
	std::string& error){
	if(configuration.allowedOrigin.empty()){
		error="browser origin is empty; use '*' for permissive browser access or configure an exact origin";
		return false;
	}
	if(configuration.port==0){
		error="browser bridge port must be between 1 and 65535";
		return false;
	}

	Implementation& state=*implementation_;
	{
		std::lock_guard<std::mutex> lock(state.mutex);
		state.connected=false;
		state.complete=false;
		state.cancelRequested=false;
		state.progressPending=false;
		state.browserError.clear();
		state.result=Result();
		state.lastHeartbeat=std::chrono::steady_clock::now();
	}

	const std::string nonce=randomHex(24);
	const std::string sessionId=randomHex(16);
	httplib::Server server;
	server.set_payload_max_length(size_t(1) << 30);
	server.new_task_queue=[](){ return new httplib::ThreadPool(2); };
	server.set_socket_options([](socket_t socket){
#ifdef SO_REUSEPORT
		// cpp-httplib enables SO_REUSEPORT by default. That can allow a second
		// listener to bind the bridge port instead of reporting the conflict.
		(void)httplib::set_socket_opt(socket,SOL_SOCKET,SO_REUSEPORT,0);
#endif
		(void)httplib::set_socket_opt(socket,SOL_SOCKET,SO_REUSEADDR,1);
	});

	auto originAllowed=[&](const httplib::Request& request){
		if(!request.has_header("Origin")) return false;
		return configuration.allowedOrigin=="*" ||
			request.get_header_value("Origin")==configuration.allowedOrigin;
	};
	auto applyCors=[&](const httplib::Request& request,httplib::Response& response){
		if(originAllowed(request)){
			// Echoing the requesting origin is more compatible with browser
			// loopback/private-network checks than a literal wildcard.
			response.set_header("Access-Control-Allow-Origin",request.get_header_value("Origin"));
		}
		response.set_header("Vary","Origin");
		response.set_header("Access-Control-Allow-Methods","GET, POST, OPTIONS");
		response.set_header("Access-Control-Allow-Headers","Content-Type, X-G2S-Protocol-Version, X-G2S-Session-Id, X-G2S-Nonce, X-G2S-Dimensions, X-G2S-Variable-Types, X-G2S-Encoding");
		response.set_header("Access-Control-Allow-Private-Network","true");
		response.set_header("Cache-Control","no-store");
	};

	auto authorize=[&](const httplib::Request& request, httplib::Response& response){
		applyCors(request,response);
		if(!originAllowed(request)){
			response.status=403;
			response.set_content("origin not allowed","text/plain");
			return false;
		}
		const std::string supplied=request.has_header("X-G2S-Nonce") ? request.get_header_value("X-G2S-Nonce") : std::string();
		if(supplied!=nonce){
			response.status=401;
			response.set_content("invalid session nonce","text/plain");
			return false;
		}
		if(!request.has_header("X-G2S-Protocol-Version") || request.get_header_value("X-G2S-Protocol-Version")!="1" ||
			!request.has_header("X-G2S-Session-Id") || request.get_header_value("X-G2S-Session-Id")!=sessionId){
			response.status=409;
			response.set_content("protocol version or session ID mismatch","text/plain");
			return false;
		}
		return true;
	};
	auto heartbeat=[&](){
		std::lock_guard<std::mutex> lock(state.mutex);
		state.connected=true;
		state.lastHeartbeat=std::chrono::steady_clock::now();
		state.condition.notify_all();
	};

	server.Options(R"(.*)",[&](const httplib::Request& request, httplib::Response& response){
		applyCors(request,response);
		response.status=originAllowed(request) ? 204 : 403;
	});
	server.Get("/v1/session",[&](const httplib::Request& request, httplib::Response& response){
		applyCors(request,response);
		if(!originAllowed(request)){
			response.status=403;
			response.set_content("origin not allowed","text/plain");
			return;
		}
		heartbeat();
		Json::Value body(Json::objectValue);
		body["protocolVersion"]=1;
		body["sessionId"]=sessionId;
		body["nonce"]=nonce;
		body["algorithm"]="qs";
		response.set_content(jsonString(body),"application/json");
	});
	server.Get("/v1/job",[&](const httplib::Request& request, httplib::Response& response){
		if(!authorize(request,response)) return;
		heartbeat();
		response.set_content(job.manifestJson,"application/json");
	});
	server.Get(R"(/v1/arrays/([A-Za-z0-9_.-]+))",[&](const httplib::Request& request, httplib::Response& response){
		if(!authorize(request,response)) return;
		heartbeat();
		const std::string id=request.matches[1];
		auto iterator=std::find_if(job.arrays.begin(),job.arrays.end(),[&](const ArrayPayload& payload){ return payload.id==id; });
		if(iterator==job.arrays.end()){
			response.status=404;
			return;
		}
		response.set_content(reinterpret_cast<const char*>(iterator->values.data()),iterator->values.size()*sizeof(float),"application/octet-stream");
	});
	server.Get("/v1/control",[&](const httplib::Request& request, httplib::Response& response){
		if(!authorize(request,response)) return;
		heartbeat();
		Json::Value body(Json::objectValue);
		{
			std::lock_guard<std::mutex> lock(state.mutex);
			body["cancel"]=state.cancelRequested;
		}
		response.set_content(jsonString(body),"application/json");
	});
	server.Post("/v1/progress",[&](const httplib::Request& request, httplib::Response& response){
		if(!authorize(request,response)) return;
		heartbeat();
		Json::Value body;
		if(!parseJson(request.body,body) || !body.isMember("percent")){
			response.status=400;
			return;
		}
		{
			std::lock_guard<std::mutex> lock(state.mutex);
			state.progressPercent=body["percent"].asFloat();
			state.progressMessage=body.get("message","").asString();
			state.progressPending=true;
			state.condition.notify_all();
		}
		response.status=204;
	});
	server.Post(R"(/v1/results/([A-Za-z0-9_.-]+))",[&](const httplib::Request& request, httplib::Response& response){
		if(!authorize(request,response)) return;
		heartbeat();
		if(request.body.size()%sizeof(float)!=0 || !request.has_header("X-G2S-Dimensions") || !request.has_header("X-G2S-Variable-Types")){
			response.status=400;
			return;
		}
		ArrayPayload payload;
		payload.id=request.matches[1];
		payload.parameter=payload.id;
		payload.encoding=request.has_header("X-G2S-Encoding") ? request.get_header_value("X-G2S-Encoding") : "float32";
		if(payload.encoding!="float32" && payload.encoding!="uint32" && payload.encoding!="int32"){
			response.status=400;
			return;
		}
		payload.dimensions=parseUnsignedList(request.get_header_value("X-G2S-Dimensions"));
		payload.variableTypes=parseUnsignedList(request.get_header_value("X-G2S-Variable-Types"));
		if(payload.dimensions.empty() || payload.variableTypes.empty()){
			response.status=400;
			return;
		}
		size_t expected=payload.variableTypes.size();
		for(unsigned dimension:payload.dimensions){
			if(dimension==0 || expected>std::numeric_limits<size_t>::max()/dimension){ response.status=400; return; }
			expected*=dimension;
		}
		if(expected!=request.body.size()/sizeof(float)){
			response.status=400;
			return;
		}
		payload.values.resize(expected);
		std::memcpy(payload.values.data(),request.body.data(),request.body.size());
		{
			std::lock_guard<std::mutex> lock(state.mutex);
			state.result.arrays[payload.id]=std::move(payload);
		}
		response.status=204;
	});
	server.Post("/v1/complete",[&](const httplib::Request& request, httplib::Response& response){
		if(!authorize(request,response)) return;
		heartbeat();
		Json::Value body;
		if(!parseJson(request.body,body)){
			response.status=400;
			return;
		}
		{
			std::lock_guard<std::mutex> lock(state.mutex);
			state.result.durationMs=body.get("durationMs",0.0).asDouble();
			if(body.isMember("metadata") && body["metadata"].isObject()){
				for(const std::string& key:body["metadata"].getMemberNames()) state.result.metadata[key]=body["metadata"][key].asString();
			}
			state.complete=true;
			state.condition.notify_all();
		}
		response.status=204;
	});
	server.Post("/v1/error",[&](const httplib::Request& request, httplib::Response& response){
		if(!authorize(request,response)) return;
		heartbeat();
		Json::Value body;
		parseJson(request.body,body);
		{
			std::lock_guard<std::mutex> lock(state.mutex);
			state.browserError=body.get("message",request.body).asString();
			state.complete=true;
			state.condition.notify_all();
		}
		response.status=204;
	});

	if(!server.bind_to_port("127.0.0.1",configuration.port)){
		error="cannot bind browser bridge to 127.0.0.1:"+std::to_string(configuration.port);
		return false;
	}
	std::thread serverThread([&server](){ server.listen_after_bind(); });

	const auto connectionDeadline=std::chrono::steady_clock::now()+configuration.timeout;
	bool success=false;
	while(true){
		std::unique_lock<std::mutex> lock(state.mutex);
		if(state.complete){
			success=state.browserError.empty();
			break;
		}
		const auto now=std::chrono::steady_clock::now();
		if(!state.connected && now>=connectionDeadline){
			error="browser page did not connect within "+std::to_string(configuration.timeout.count())+" ms";
			break;
		}
		if(state.connected && now-state.lastHeartbeat>=configuration.timeout){
			error="browser heartbeat was lost for "+std::to_string(configuration.timeout.count())+" ms";
			break;
		}
		lock.unlock();
		bool reportProgress=false;
		float progressPercent=0.f;
		std::string progressMessage;
		{
			std::lock_guard<std::mutex> progressLock(state.mutex);
			if(state.progressPending){
				reportProgress=true;
				progressPercent=state.progressPercent;
				progressMessage=state.progressMessage;
				state.progressPending=false;
			}
		}
		if(reportProgress && callbacks.progress) callbacks.progress(progressPercent,progressMessage);
		if(callbacks.interrupted && callbacks.interrupted()){
			std::lock_guard<std::mutex> cancelLock(state.mutex);
			state.cancelRequested=true;
		}
		if(callbacks.updateDisplay) callbacks.updateDisplay();
		lock.lock();
		state.condition.wait_for(lock,std::chrono::milliseconds(100));
	}

	server.stop();
	if(serverThread.joinable()) serverThread.join();
	if(!success){
		if(error.empty()) error=state.browserError.empty() ? "browser simulation failed" : state.browserError;
		return false;
	}
	result=std::move(state.result);
	return true;
}

} // namespace browser
} // namespace g2s
