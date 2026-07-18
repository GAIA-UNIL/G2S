/* G2S browser transport tests - SPDX-License-Identifier: GPL-3.0-or-later */

#include "browserTransport.hpp"
#include "third_party/cpp-httplib/httplib.h"

#include <json/json.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

constexpr const char* Origin="http://localhost:8000";

Json::Value parseJson(const std::string& input){
	Json::Value value;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(input);
	if(!Json::parseFromStream(builder,stream,&value,&errors)) throw std::runtime_error(errors);
	return value;
}

void authorize(httplib::Headers& headers,const Json::Value& session){
	headers.emplace("Origin",Origin);
	headers.emplace("X-G2S-Nonce",session["nonce"].asString());
	headers.emplace("X-G2S-Protocol-Version","1");
	headers.emplace("X-G2S-Session-Id",session["sessionId"].asString());
}

void require(bool condition,const std::string& message){
	if(!condition) throw std::runtime_error(message);
}

void testTimeout(){
	g2s::browser::Transport transport;
	g2s::browser::Job job{"{\"protocolVersion\":1,\"arrays\":[]}",{}};
	g2s::browser::Configuration configuration{Origin,18129,std::chrono::milliseconds(250)};
	g2s::browser::Result result;
	std::string error;
	const auto started=std::chrono::steady_clock::now();
	require(!transport.run(job,configuration,{},result,error),"missing browser unexpectedly succeeded");
	const auto elapsed=std::chrono::steady_clock::now()-started;
	require(error.find("did not connect")!=std::string::npos,"missing browser returned the wrong error: "+error);
	require(elapsed<std::chrono::seconds(2),"missing browser timeout was not bounded");
}

void testExchange(){
	g2s::browser::Transport transport;
	g2s::browser::Job job{"{\"protocolVersion\":1,\"arrays\":[]}",{}};
	g2s::browser::Configuration configuration{Origin,18130,std::chrono::milliseconds(2000)};
	g2s::browser::Result result;
	std::string error;
	auto run=std::async(std::launch::async,[&](){ return transport.run(job,configuration,{},result,error); });

	httplib::Client client("127.0.0.1",18130);
	httplib::Result sessionResponse;
	for(unsigned attempt=0;attempt<100 && !sessionResponse;++attempt){
		httplib::Headers headers{{"Origin",Origin}};
		sessionResponse=client.Get("/v1/session",headers);
		if(!sessionResponse) std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	require(sessionResponse && sessionResponse->status==200,"browser session was unavailable");
	const Json::Value session=parseJson(sessionResponse->body);

	httplib::Headers wrongHeaders{{"Origin","http://attacker.invalid"}};
	auto rejected=client.Get("/v1/session",wrongHeaders);
	require(rejected && rejected->status==403,"wrong origin was not rejected");

	httplib::Headers headers;
	authorize(headers,session);
	auto manifest=client.Get("/v1/job",headers);
	require(manifest && manifest->status==200,"authorized manifest request failed");
	auto progress=client.Post("/v1/progress",headers,"{\"percent\":50,\"message\":\"half\"}","application/json");
	require(progress && progress->status==204,"progress request failed");

	const float simulationValues[]={1.f,0.f,1.f,0.f};
	httplib::Headers simulationHeaders=headers;
	simulationHeaders.emplace("X-G2S-Dimensions","2,2");
	simulationHeaders.emplace("X-G2S-Variable-Types","1");
	simulationHeaders.emplace("X-G2S-Encoding","float32");
	std::string simulation(reinterpret_cast<const char*>(simulationValues),sizeof(simulationValues));
	auto simulationResponse=client.Post("/v1/results/simulation",simulationHeaders,simulation,"application/octet-stream");
	require(simulationResponse && simulationResponse->status==204,"simulation upload failed");

	const uint32_t indexValues[]={0,1,2,3};
	httplib::Headers indexHeaders=headers;
	indexHeaders.emplace("X-G2S-Dimensions","2,2");
	indexHeaders.emplace("X-G2S-Variable-Types","0");
	indexHeaders.emplace("X-G2S-Encoding","uint32");
	std::string index(reinterpret_cast<const char*>(indexValues),sizeof(indexValues));
	auto indexResponse=client.Post("/v1/results/index",indexHeaders,index,"application/octet-stream");
	require(indexResponse && indexResponse->status==204,"index upload failed");

	auto complete=client.Post("/v1/complete",headers,"{\"durationMs\":12.5,\"metadata\":{\"status\":\"success\"}}","application/json");
	require(complete && complete->status==204,"completion request failed");
	require(run.get(),"transport exchange failed: "+error);
	require(result.arrays.size()==2 && result.arrays.at("simulation").values.size()==4,"result arrays were not preserved");
	require(result.durationMs==12.5,"duration was not preserved");
}

void testPortConflict(){
	httplib::Server blocker;
	require(blocker.bind_to_port("127.0.0.1",18131),"test could not reserve conflict port");
	std::thread blockerThread([&](){ blocker.listen_after_bind(); });
	std::this_thread::sleep_for(std::chrono::milliseconds(20));
	g2s::browser::Transport transport;
	g2s::browser::Job job{"{\"protocolVersion\":1,\"arrays\":[]}",{}};
	g2s::browser::Configuration configuration{Origin,18131,std::chrono::milliseconds(250)};
	g2s::browser::Result result;
	std::string error;
	const bool transportResult=transport.run(job,configuration,{},result,error);
	blocker.stop();
	if(blockerThread.joinable()) blockerThread.join();
	require(!transportResult,"port conflict unexpectedly succeeded");
	require(error.find("cannot bind")!=std::string::npos,"port conflict returned the wrong error: "+error);
}

} // namespace

int main(){
	try {
		testTimeout();
		testExchange();
		testPortConflict();
		std::cout << "browser transport tests passed\n";
		return 0;
	} catch(const std::exception& error){
		std::cerr << error.what() << '\n';
		return 1;
	}
}
