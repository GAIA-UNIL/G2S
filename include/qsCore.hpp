/*
 * Reusable in-memory QS entry point.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef G2S_QS_CORE_HPP
#define G2S_QS_CORE_HPP

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace g2s {
namespace qs {

struct Array {
	std::string name;
	std::vector<unsigned> dimensions;
	std::vector<unsigned> variableTypes;
	std::string encoding="float32";
	std::vector<float> values;
};

struct Request {
	std::vector<Array> arrays;
	std::vector<std::string> arguments;
};

struct Result {
	std::vector<Array> arrays;
	double durationMs=0.0;
	std::map<std::string,std::string> metadata;
};

struct Observer {
	std::function<void(double,const std::string&)> progress;
};

bool runInMemory(const Request& request, Result& result, std::string& error, const Observer* observer=nullptr);

inline const Observer*& activeObserverSlot(){
	static thread_local const Observer* observer=nullptr;
	return observer;
}

inline void setActiveObserver(const Observer* observer){ activeObserverSlot()=observer; }

inline void notifyProgress(double percent, const std::string& detail){
	const Observer* observer=activeObserverSlot();
	if(observer && observer->progress) observer->progress(percent,detail);
}

} // namespace qs
} // namespace g2s

// Shared implementation used by the native executable and the in-memory adapter.
int g2sQsProgramMain(int argc, const char* argv[]);

#endif
