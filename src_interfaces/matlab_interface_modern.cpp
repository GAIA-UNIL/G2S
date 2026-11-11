#include "matlab_interface_modern.hpp"

#include <algorithm>
#include <cstdio>
#include <string>

using matlab::mex::ArgumentList;

namespace {

// A flag is a char row vector or scalar string that starts with '-' and has at least two characters.
inline bool isFlag(const matlab::data::Array& a, std::string& out) {
	return g2sMatlab::arrayToString(a, out) && out.size() > 1 && out[0] == '-';
}

// Let MATLAB refresh figures; a pending Ctrl+C is left for MATLAB to handle after the MEX call returns.
inline void drawNow(const std::shared_ptr<matlab::engine::MATLABEngine>& engine) {
	try {
		engine->eval(u"drawnow");
	} catch (...) {
	}
}

} // namespace

class MexFunction : public matlab::mex::Function {
public:
	void operator()(ArgumentList outputs, ArgumentList inputs) {
		matlab::data::ArrayFactory factory;
		std::shared_ptr<matlab::engine::MATLABEngine> engine = getEngine();

		std::string firstArgument;
		if (!inputs.empty() && g2sMatlab::arrayToString(inputs[0], firstArgument) && firstArgument == "--version") {
			if (outputs.size() > 0 && inputs.size() == 1) {
				if (outputs.size() >= 1) outputs[0] = factory.createCharArray(VERSION);
				if (outputs.size() >= 2) outputs[1] = factory.createCharArray(__DATE__);
				if (outputs.size() >= 3) outputs[2] = factory.createCharArray(__TIME__);
				for (size_t i = 3; i < outputs.size(); ++i) outputs[i] = factory.createArray<double>({0, 0});
			} else {
				char buff[1000];
				snprintf(buff, sizeof(buff), "G2S version %s, compiled the %s %s with R%x (C++ MEX API)", VERSION, __DATE__, __TIME__, MATLAB_VERSION);
				std::vector<matlab::data::Array> args{factory.createCharArray(buff)};
				engine->feval(u"disp", 0, args);
			}
			return;
		}

		InterfaceTemplateMatlabModern iface(engine);

		// Group the inputs by flag: '-key', value, value, ... Cell values are expanded.
		std::multimap<std::string, std::any> in;
		std::vector<size_t> flagIndex;
		std::vector<std::string> flagName;
		for (size_t i = 0; i < inputs.size(); ++i) {
			std::string name;
			if (isFlag(inputs[i], name)) {
				flagIndex.push_back(i);
				flagName.push_back(name);
			}
		}
		flagIndex.push_back(inputs.size());

		for (size_t j = 0; j + 1 < flagIndex.size(); ++j) {
			const size_t start = flagIndex[j];
			const size_t stop = flagIndex[j + 1];
			const std::string& key = flagName[j];
			if (start + 1 == stop) {
				in.insert({key, nullptr});
				continue;
			}
			for (size_t k = start + 1; k < stop; ++k) {
				const matlab::data::Array value = inputs[k];
				if (value.getType() == matlab::data::ArrayType::CELL) {
					const matlab::data::CellArray cells(value);
					// Dereferencing a CellArray iterator yields a Reference<Array>; store a real Array.
					for (auto it = cells.begin(); it != cells.end(); ++it) in.insert({key, matlab::data::Array(*it)});
				} else {
					in.insert({key, value});
				}
			}
		}

		std::multimap<std::string, std::any> out;
		const size_t nlhs = outputs.size();
		iface.runStandardCommunication(in, out, static_cast<int>(nlhs));

		auto asArray = [](const std::any& value) { return std::any_cast<matlab::data::Array>(value); };

		auto schemaIter = out.find("schema");
		if (schemaIter != out.end()) {
			if (nlhs > 0) outputs[0] = asArray(schemaIter->second);
			for (size_t i = 1; i < nlhs; ++i) outputs[i] = factory.createArray<double>({0, 0});
			drawNow(engine);
			return;
		}

		// Positional outputs: numbered results first, then t, meta, progression and id.
		size_t position = 0;
		for (size_t i = 0; i < nlhs; ++i) {
			auto iter = out.find(std::to_string(i + 1));
			if (iter != out.end() && position < std::max<size_t>(nlhs - 1, 1)) {
				outputs[position++] = asArray(iter->second);
			}
		}
		if (position < nlhs) {
			auto iter = out.find("t");
			if (iter != out.end()) outputs[position++] = asArray(iter->second);
		}
		if (position < nlhs) {
			auto iter = out.find("meta");
			if (iter != out.end()) outputs[position++] = asArray(iter->second);
		}
		if (position < nlhs) {
			auto iter = out.find("progression");
			if (iter != out.end()) {
				double progression = 0.0;
				if (iter->second.type() == typeid(float)) progression = std::any_cast<float>(iter->second);
				else if (iter->second.type() == typeid(double)) progression = std::any_cast<double>(iter->second);
				outputs[position++] = factory.createScalar(progression);
			}
		}
		if (position < nlhs) {
			auto iter = out.find("id");
			if (iter != out.end()) outputs[position++] = asArray(iter->second);
		}

		drawNow(engine);
	}
};
