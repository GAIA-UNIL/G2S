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

#include "cvtZMQ2WS.hpp"
#include <iostream>
#include <zmq.hpp>

#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

typedef websocketpp::server<websocketpp::config::asio> server;
void on_message(server* s, websocketpp::connection_hdl hdl, server::message_ptr msg, zmq::socket_t *proxySocketZMQ, std::atomic<bool> &waitInstructionFromSource) {
	
	zmq::message_t reply (msg->get_payload().size());
	memcpy (reply.data (), msg->get_payload().data(), msg->get_payload().size());
	proxySocketZMQ->send (reply);
	waitInstructionFromSource=true;
}

void on_open(server* s, std::vector<websocketpp::connection_hdl> *connections, websocketpp::connection_hdl newConnection, zmq::socket_t *proxySocketZMQ, std::atomic<bool> &waitInstructionFromSource){
	
	fprintf(stderr, "open\n");
	connections->push_back(newConnection);

	waitInstructionFromSource=true;
}


void on_close(std::vector<websocketpp::connection_hdl> *connections, websocketpp::connection_hdl, std::atomic<bool> &waitInstructionFromSource){
	connections->clear();
	fprintf(stderr,"%s\n", "one coection closed");
	waitInstructionFromSource=false;
}

bool on_validate(std::vector<websocketpp::connection_hdl> *connections, websocketpp::connection_hdl){
	return (connections->size()==0);
}

void zmqServer(std::string from, std::atomic<bool> &done, std::atomic<bool> &waitInstructionFromSource, std::vector<websocketpp::connection_hdl> &connections, server &proxyServerWS, zmq::socket_t &proxySocketZMQ ){
	// init ZMQ

	proxySocketZMQ.bind(from.c_str());

	while(!done)
	{
		while(!done && !waitInstructionFromSource){
			std::this_thread::sleep_for(std::chrono::milliseconds(300));
		}

		zmq::message_t request;
		while(!done && !proxySocketZMQ.recv (&request));
		proxyServerWS.send(connections[0], request.data(), request.size(), websocketpp::frame::opcode::binary);
		waitInstructionFromSource=false;
	}
}

void wsServer(int portTo, std::atomic<bool> &done, std::atomic<bool> &waitInstructionFromSource, std::vector<websocketpp::connection_hdl> &connections, server &proxyServerWS, zmq::socket_t &proxySocketZMQ ){
	// init ZMQ

	
	try {
		//proxyServerWS.clear_access_channels(websocketpp::log::alevel::frame_header | websocketpp::log::alevel::frame_payload);
		proxyServerWS.clear_access_channels(websocketpp::log::alevel::all);
		proxyServerWS.clear_error_channels(websocketpp::log::alevel::all);
		// Set logging settings
		// proxyServerWS.set_access_channels(websocketpp::log::alevel::all);
		// proxyServerWS.clear_access_channels(websocketpp::log::alevel::frame_payload);

		// Initialize Asio
		proxyServerWS.init_asio();

		// Register our message handler
		proxyServerWS.set_open_handler(bind(&on_open, &proxyServerWS, &connections, websocketpp::lib::placeholders::_1, &proxySocketZMQ, std::ref(waitInstructionFromSource)));
		proxyServerWS.set_message_handler(bind(&on_message, &proxyServerWS, websocketpp::lib::placeholders::_1, websocketpp::lib::placeholders::_2, &proxySocketZMQ, std::ref(waitInstructionFromSource)));
		proxyServerWS.set_close_handler(bind(&on_close, &connections, websocketpp::lib::placeholders::_1,std::ref(waitInstructionFromSource)));
		proxyServerWS.set_validate_handler(bind(&on_validate, &connections, websocketpp::lib::placeholders::_1));


		// Listen on port 
		fprintf(stderr,"port %d\n", portTo);
		proxyServerWS.listen(portTo);

		// Start the server accept loop
		proxyServerWS.start_accept();
		
		// Start the ASIO io_service run loop
		std::thread network_thread(&server::run,&proxyServerWS);

		while(!done){
			std::this_thread::sleep_for(std::chrono::milliseconds(300));
		}

		proxyServerWS.stop_listening();

		/*for (int i = 0; i < connections.size(); ++i)
		{
			try{
				fprintf(stderr, "%s%d\n", "request to close connection: ",i);
				//proxyServerWS.pause_reading(connections[i]);
				proxyServerWS.close(connections[i], websocketpp::close::status::normal, "");
			}catch(websocketpp::lib::error_code ec){
				std::cout<<"lib::error_code "<<ec<<std::endl;
			}
		}*/
		proxyServerWS.stop();

		if(network_thread.joinable())network_thread.join();

	} catch (websocketpp::exception const & e) {
		std::cout << e.what() << std::endl;
	} catch (...) {
		std::cout << "other exception" << std::endl;
	}
}


void cvtServer(char* from, char* to, std::atomic<bool> &serverRun, std::atomic<bool> &done){
	//serverRunGlobal=&serverRun;
	fprintf(stderr,"%s ==> %s\n", from, to);

	int port=0;
	char conncetionType[128];
	char address[128];
	sscanf(to,"%[^:]://%[^:]:%d",conncetionType,address,&port);
	fprintf(stderr, "%s\n", conncetionType);
	fprintf(stderr, "%s\n", address);
	fprintf(stderr, "%d\n", port);

	//init WS

	server proxyServerWS;
	proxyServerWS.set_reuse_addr(true);
	std::vector<websocketpp::connection_hdl> connections;

	// init ZMQ
	zmq::context_t context (1);
	zmq::socket_t proxySocketZMQ (context, ZMQ_REP);
	int timeout=1000;
	proxySocketZMQ.setsockopt(ZMQ_LINGER, timeout);
	proxySocketZMQ.setsockopt(ZMQ_RCVTIMEO, timeout);
	proxySocketZMQ.setsockopt(ZMQ_SNDTIMEO, timeout);
	
	//std::atomic<bool> waitInstructionFromSource;

	//start

	std::thread zmqServerThread(zmqServer,std::string(from),std::ref(done),std::ref(serverRun),std::ref(connections),std::ref(proxyServerWS),std::ref(proxySocketZMQ));
	wsServer(port,std::ref(done),std::ref(serverRun),std::ref(connections),std::ref(proxyServerWS),std::ref(proxySocketZMQ));

	if(zmqServerThread.joinable())zmqServerThread.join();

}