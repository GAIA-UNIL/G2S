#include "matlab_interface_modern.hpp"

#include <algorithm>
#include <string>
#include <cstdio>

using matlab::mex::ArgumentList;

namespace {

inline bool isFlagString(const matlab::data::Array& a, std::string& out) {
    using T = matlab::data::ArrayType;
    if (a.getType() == T::MATLAB_STRING) {
        auto sa = matlab::data::StringArray(a);
        if (sa.getNumberOfElements() == 0) return false;
        matlab::data::optional<std::u16string> opt = sa[0];
        if (!opt.has_value()) return false;
        const std::u16string &u16 = *opt;
        std::string s; s.reserve(u16.size());
        for (char16_t ch : u16) s.push_back(static_cast<char>(ch));
        out = std::move(s);
        return true;
    }
    if (a.getType() == T::CHAR) {
        auto ca = matlab::data::CharArray(a);
        out = ca.toAscii();
        return true;
    }
    return false;
}

} // namespace

class MexFunction : public matlab::mex::Function {
public:
    void operator()(ArgumentList outputs, ArgumentList inputs) {
        matlab::data::ArrayFactory factory;
        auto eng = getEngine();
        InterfaceTemplateMatlabModern iface(eng);

        // Special: --version
        if (!inputs.empty()) {
            std::string sflag;
            if (isFlagString(inputs[0], sflag) && sflag == "--version") {
                if (outputs.size() > 0 && inputs.size() == 1) {
                    if (outputs.size() >= 1) outputs[0] = factory.createCharArray(VERSION);
                    if (outputs.size() >= 2) outputs[1] = factory.createCharArray(__DATE__);
                    if (outputs.size() >= 3) outputs[2] = factory.createCharArray(__TIME__);
                } else {
                    char buff[1000];
                    snprintf(buff, sizeof(buff), "G2S version %s, compiled the %s %s", VERSION, __DATE__, __TIME__);
                    iface.eraseAndPrint(std::string(buff));
                }
                return;
            }
        }

        // Build input multimap from ArgumentList
        std::multimap<std::string, std::any> in;
        std::vector<int> flagIdx;
        for (size_t i = 0; i < inputs.size(); ++i) {
            std::string s;
            if (isFlagString(inputs[i], s)) {
                if (!s.empty() && s[0] == '-') flagIdx.push_back(static_cast<int>(i));
            }
        }
        flagIdx.push_back(static_cast<int>(inputs.size()));

        for (size_t j = 0; j + 1 < flagIdx.size(); ++j) {
            const int start = flagIdx[j];
            const int stop = flagIdx[j+1];
            std::string key;
            isFlagString(inputs[start], key);
            if (start + 1 == stop) {
                in.insert({key, nullptr});
            } else {
                for (int k = start + 1; k < stop; ++k) {
                    auto a = inputs[k];
                    if (a.getType() == matlab::data::ArrayType::CELL) {
                        auto ca = matlab::data::CellArray(a);
                        for (auto it = ca.begin(); it != ca.end(); ++it) in.insert({key, *it});
                    } else {
                        in.insert({key, a});
                    }
                }
            }
        }

        // Run core logic
        std::multimap<std::string, std::any> out;
        iface.runStandardCommunication(in, out, static_cast<int>(outputs.size()));

        // Marshal outputs: numbered first, then t, progression, id
        size_t pos = 0;
        const size_t nlhs = outputs.size();
        for (size_t i = 0; i < nlhs; ++i) {
            auto it = out.find(std::to_string(i+1));
            if (it != out.end() && pos < std::max(static_cast<int>(nlhs) - 1, 1)) {
                outputs[pos++] = std::any_cast<matlab::data::Array>(it->second);
            }
        }

        if (pos < nlhs) {
            auto it = out.find("t");
            if (it != out.end()) outputs[pos++] = std::any_cast<matlab::data::Array>(it->second);
        }

        if (pos < nlhs) {
            auto it = out.find("progression");
            if (it != out.end()) {
                double v = 0.0;
                if (it->second.type() == typeid(float)) v = static_cast<double>(std::any_cast<float>(it->second));
                else if (it->second.type() == typeid(double)) v = std::any_cast<double>(it->second);
                outputs[pos++] = factory.createScalar(v);
            }
        }

        if (pos < nlhs) {
            auto it = out.find("id");
            if (it != out.end()) outputs[pos++] = std::any_cast<matlab::data::Array>(it->second);
        }

        // Keep UI responsive
        eng->eval(u"drawnow");
    }
};
