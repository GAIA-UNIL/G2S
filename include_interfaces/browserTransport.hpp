/*
 * G2S browser transport
 * SPDX-License-Identifier: LGPL-3.0-only
 */

#ifndef G2S_BROWSER_TRANSPORT_HPP
#define G2S_BROWSER_TRANSPORT_HPP

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace g2s {
namespace browser {

struct ArrayPayload {
	std::string id;
	std::string parameter;
	std::string encoding="float32";
	std::vector<unsigned> dimensions;
	std::vector<unsigned> variableTypes;
	std::vector<float> values;
};

struct Job {
	std::string manifestJson;
	std::vector<ArrayPayload> arrays;
};

struct Result {
	std::map<std::string, ArrayPayload> arrays;
	double durationMs=0.0;
	std::map<std::string,std::string> metadata;
};

struct Configuration {
	std::string allowedOrigin;
	unsigned short port=8129;
	std::chrono::milliseconds timeout{30000};
};

struct Callbacks {
	std::function<void(float,const std::string&)> progress;
	std::function<bool()> interrupted;
	std::function<void()> updateDisplay;
};

class Transport {
public:
	Transport();
	~Transport();
	Transport(const Transport&)=delete;
	Transport& operator=(const Transport&)=delete;

	bool run(const Job& job,
		const Configuration& configuration,
		const Callbacks& callbacks,
		Result& result,
		std::string& error);

private:
	struct Implementation;
	Implementation* implementation_;
};

} // namespace browser
} // namespace g2s

#endif
