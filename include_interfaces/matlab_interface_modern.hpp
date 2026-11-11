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

/*
 * MATLAB interface built on the C++ MEX / MATLAB Data API (R2018a+).
 * Functionally equivalent to matlab_interface.hpp (C API), which remains as a fallback.
 */

#ifndef MATLAB_INTERFACE_MODERN_HPP
#define MATLAB_INTERFACE_MODERN_HPP

#define MATLAB

#include "interfaceTemplate.hpp"

#include "mex.hpp"
#include "mexAdapter.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef MATLAB_VERSION
#define MATLAB_VERSION 0
#endif

// Ctrl+C detection through libut (same mechanism as include/mexInterrupt.hpp, which
// depends on the C MEX API and therefore cannot be used from a C++ MEX file).
class std::shared_ptr<struct InterruptStateData> const & utGetInterruptStateData(void);

class utInterruptState
{
public:
	static bool IsInterruptPending(std::shared_ptr<struct InterruptStateData> const& state);
};

namespace g2sMatlab {

inline std::string toUtf8(const std::u16string& value){
	return matlab::engine::convertUTF16StringToUTF8String(value);
}

inline std::u16string toUtf16(const std::string& value){
	return matlab::engine::convertUTF8StringToUTF16String(value);
}

// Returns true and fills `out` when `a` is a char row vector or a scalar MATLAB string.
inline bool arrayToString(const matlab::data::Array& a, std::string& out){
	using T = matlab::data::ArrayType;
	if(a.getType()==T::CHAR){
		matlab::data::CharArray chars(a);
		out=toUtf8(chars.toUTF16());
		return true;
	}
	if(a.getType()==T::MATLAB_STRING && a.getNumberOfElements()==1){
		matlab::data::StringArray strings(a);
		matlab::data::MATLABString value=strings[0];
		if(!value.has_value()) return false;
		out=toUtf8(*value);
		return true;
	}
	return false;
}

inline bool isNumericOrLogical(const matlab::data::Array& a){
	using T = matlab::data::ArrayType;
	switch(a.getType()){
		case T::DOUBLE: case T::SINGLE:
		case T::INT8: case T::UINT8: case T::INT16: case T::UINT16:
		case T::INT32: case T::UINT32: case T::INT64: case T::UINT64:
		case T::LOGICAL:
			return true;
		default:
			return false;
	}
}

// Calls function(TypedArray<T>) with the array's actual element type.
template <typename Function>
inline auto dispatchNumeric(const matlab::data::Array& a, Function&& function) -> decltype(function(matlab::data::TypedArray<double>(a))) {
	using T = matlab::data::ArrayType;
	switch(a.getType()){
		case T::DOUBLE: return function(matlab::data::TypedArray<double>(a));
		case T::SINGLE: return function(matlab::data::TypedArray<float>(a));
		case T::INT8:   return function(matlab::data::TypedArray<int8_t>(a));
		case T::UINT8:  return function(matlab::data::TypedArray<uint8_t>(a));
		case T::INT16:  return function(matlab::data::TypedArray<int16_t>(a));
		case T::UINT16: return function(matlab::data::TypedArray<uint16_t>(a));
		case T::INT32:  return function(matlab::data::TypedArray<int32_t>(a));
		case T::UINT32: return function(matlab::data::TypedArray<uint32_t>(a));
		case T::INT64:  return function(matlab::data::TypedArray<int64_t>(a));
		case T::UINT64: return function(matlab::data::TypedArray<uint64_t>(a));
		case T::LOGICAL:return function(matlab::data::TypedArray<bool>(a));
		default: throw std::invalid_argument("non numeric MATLAB array");
	}
}

} // namespace g2sMatlab

class InterfaceTemplateMatlabModern : public InterfaceTemplate
{
	std::shared_ptr<matlab::engine::MATLABEngine> _engine;
	matlab::data::ArrayFactory _factory;
	std::atomic<bool> _done{false};
	std::atomic<bool> _stopInterruptCheck{false};
	std::future<void> _interruptCheck;

	static matlab::data::Array asArray(const std::any& val){
		if(val.type()==typeid(matlab::data::Array)) return std::any_cast<matlab::data::Array>(val);
		throw std::invalid_argument("unexpected value type in MATLAB interface");
	}

	// Display helpers must never throw past the template: a Ctrl+C pending inside MATLAB makes
	// feval/eval fail, and escaping here would skip the template's job-kill path. Record it as
	// an interruption instead; the template then kills the job and calls sendError.
	void callMatlab(const char16_t* function, std::vector<matlab::data::Array> args){
		if(_done) return;
		try {
			_engine->feval(function, 0, args);
		} catch (...) {
			_done=true;
		}
	}

	void evalMatlab(const char16_t* statement){
		if(_done) return;
		try {
			_engine->eval(statement);
		} catch (...) {
			_done=true;
		}
	}

public:
	explicit InterfaceTemplateMatlabModern(std::shared_ptr<matlab::engine::MATLABEngine> engine)
	: _engine(std::move(engine)) {
		// The interrupt state is captured on the MATLAB thread; the polling thread never
		// touches the MATLAB engine.
		std::shared_ptr<InterruptStateData> interruptState=utGetInterruptStateData();
		_interruptCheck=std::async(std::launch::async, [this, interruptState](){
			while(!_stopInterruptCheck && !_done){
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				if(utInterruptState::IsInterruptPending(interruptState)) _done=true;
			}
		});
	}

	~InterfaceTemplateMatlabModern(){
		_stopInterruptCheck=true;
		if(_interruptCheck.valid()) _interruptCheck.wait();
	}

	void updateDisplay() override {
		evalMatlab(u"drawnow");
	}

	bool userRequestInteruption() override {
		return _done;
	}

	bool isDataMatrix(std::any val) override {
		if(val.type()!=typeid(matlab::data::Array)) return false;
		const matlab::data::Array array=std::any_cast<matlab::data::Array>(val);
		return g2sMatlab::isNumericOrLogical(array) && array.getNumberOfElements()!=1;
	}

	std::string nativeToStandardString(std::any val) override {
		if(val.type()==typeid(std::string)) return std::any_cast<std::string>(val);
		if(val.type()!=typeid(matlab::data::Array)) return "";
		const matlab::data::Array array=std::any_cast<matlab::data::Array>(val);
		std::string text;
		if(g2sMatlab::arrayToString(array,text)) return text;
		if(g2sMatlab::isNumericOrLogical(array) && array.getNumberOfElements()==1)
			return std::to_string(float(anyNativeTo<double>(val))); // same formatting as the C API interface
		return "";
	}

	double nativeToScalar(std::any val) override { return anyNativeTo<double>(val); }
	unsigned nativeToUint32(std::any val) override { return anyNativeTo<unsigned>(val); }
	unsigned anyNativeToUnsigned(std::any val) override { return anyNativeTo<unsigned>(val); }
	float anyNativeToFloat(std::any val) override { return anyNativeTo<float>(val); }
	double anyNativeToDouble(std::any val) override { return anyNativeTo<double>(val); }
	long unsigned anyNativeToLongUnsigned(std::any val) override { return anyNativeTo<long unsigned>(val); }

	template <typename type>
	type anyNativeTo(std::any val){
		if(val.type()==typeid(type)) return std::any_cast<type>(val);
		if(val.type()!=typeid(matlab::data::Array)) return type();
		const matlab::data::Array array=std::any_cast<matlab::data::Array>(val);
		if(array.getNumberOfElements()<1 || !g2sMatlab::isNumericOrLogical(array)) return type();
		return g2sMatlab::dispatchNumeric(array, [](auto typed)->type{ return static_cast<type>(*typed.begin()); });
	}

	std::any ScalarToNative(double val) override {
		return std::any(matlab::data::Array(_factory.createScalar(val)));
	}

	std::any Uint32ToNative(unsigned val) override {
		return std::any(matlab::data::Array(_factory.createScalar<uint32_t>(val)));
	}

	std::any StringToNative(const std::string& val) override {
		return std::any(matlab::data::Array(_factory.createCharArray(g2sMatlab::toUtf16(val))));
	}

	bool elementToUnsignedString(const matlab::data::Array& matrix, size_t linearIndex, std::string &stringValue){
		using T = matlab::data::ArrayType;
		if(matrix.getType()==T::LOGICAL){
			stringValue=(matlab::data::TypedArray<bool>(matrix).begin()[linearIndex] ? "1" : "0");
			return true;
		}
		if(matrix.getType()==T::DOUBLE || matrix.getType()==T::SINGLE){
			const double value=(matrix.getType()==T::DOUBLE)
				? double(matlab::data::TypedArray<double>(matrix).begin()[linearIndex])
				: double(matlab::data::TypedArray<float>(matrix).begin()[linearIndex]);
			if(!std::isfinite(value) || value<0. || std::floor(value)!=value || value>double(std::numeric_limits<unsigned long long>::max())) return false;
			stringValue=std::to_string((uint64_t)value);
			return true;
		}
		if(!g2sMatlab::isNumericOrLogical(matrix)) return false;
		return g2sMatlab::dispatchNumeric(matrix, [&](auto typed)->bool{
			const auto value=typed.begin()[linearIndex];
			if(value<0) return false;
			stringValue=std::to_string((uint64_t)value);
			return true;
		});
	}

	bool encodeJobGridMatrixToJsonString(std::any matrix, std::string &jsonValue) override {
		if(matrix.type()!=typeid(matlab::data::Array)) return false;
		const matlab::data::Array prh=std::any_cast<matlab::data::Array>(matrix);
		if(!g2sMatlab::isNumericOrLogical(prh)) return false;

		const matlab::data::ArrayDimensions dims=prh.getDimensions();
		const size_t nbDim=dims.size();
		std::vector<size_t> strides(nbDim,1);
		for (size_t i = 1; i < nbDim; ++i) strides[i]=strides[i-1]*dims[i-1];
		std::vector<size_t> coordinate(nbDim,0);

		bool isValid=true;
		auto currentLinearIndex=[&]()->size_t{
			size_t linearIndex=0;
			for (size_t i = 0; i < nbDim; ++i) linearIndex+=coordinate[i]*strides[i];
			return linearIndex;
		};

		std::function<Json::Value(size_t)> buildJson=[&](size_t dim)->Json::Value{
			Json::Value values(Json::arrayValue);
			const size_t inputDim=nbDim-1-dim;
			if(dim==nbDim-1){
				for (size_t i = 0; i < dims[inputDim]; ++i)
				{
					coordinate[inputDim]=i;
					std::string stringValue;
					if(!elementToUnsignedString(prh,currentLinearIndex(),stringValue)){
						isValid=false;
						return Json::Value();
					}
					values.append(stringValue);
				}
				return values;
			}
			for (size_t i = 0; i < dims[inputDim]; ++i)
			{
				coordinate[inputDim]=i;
				values.append(buildJson(dim+1));
				if(!isValid) return Json::Value();
			}
			return values;
		};

		Json::Value root=buildJson(0);
		if(!isValid) return false;

		Json::StreamWriterBuilder builder;
		builder["commentStyle"] = "None";
		builder["indentation"] = "";
		jsonValue=Json::writeString(builder, root);
		return true;
	}

	// Thrown directly (no call into MATLAB) so it also works while a Ctrl+C is pending; the MEX
	// adapter turns it into a MATLAB error with this identifier and message.
	void sendError(std::string val) override {
		throw matlab::engine::MATLABException(_done ? "g2s:interrupted" : "g2s:error", g2sMatlab::toUtf16(val));
	}

	// warning() treats its message as a format string, so pass it through '%s'.
	void sendWarning(std::string val) override {
		callMatlab(u"warning", {
			_factory.createCharArray("g2s:warning"),
			_factory.createCharArray("%s"),
			_factory.createCharArray(g2sMatlab::toUtf16(val))
		});
	}

	void eraseAndPrint(std::string val) override {
		callMatlab(u"disp", { _factory.createCharArray(g2sMatlab::toUtf16(val)) });
	}

	void printMessage(std::string val) override {
		callMatlab(u"disp", { _factory.createCharArray(g2sMatlab::toUtf16(val)) });
	}

	std::any keyValueMapToNative(const std::map<std::string,std::string>& values) override {
		std::vector<std::string> fieldNames;
		fieldNames.reserve(values.size());
		for (auto it=values.begin(); it!=values.end(); ++it) fieldNames.push_back(it->first);
		matlab::data::StructArray output=_factory.createStructArray({1,1}, fieldNames);
		for (auto it=values.begin(); it!=values.end(); ++it)
			output[0][it->first]=_factory.createCharArray(g2sMatlab::toUtf16(it->second));
		return std::any(matlab::data::Array(output));
	}

	std::any anyMapToNative(const std::map<std::string,std::any>& values) override {
		std::vector<std::string> fieldNames;
		fieldNames.reserve(values.size());
		for (auto it=values.begin(); it!=values.end(); ++it) fieldNames.push_back(it->first);
		matlab::data::StructArray output=_factory.createStructArray({1,1}, fieldNames);
		for (auto it=values.begin(); it!=values.end(); ++it)
		{
			if(it->second.type()==typeid(matlab::data::Array)){
				output[0][it->first]=std::any_cast<matlab::data::Array>(it->second);
			}else if(it->second.type()==typeid(std::map<std::string,std::any>)){
				output[0][it->first]=asArray(anyMapToNative(std::any_cast<std::map<std::string,std::any>>(it->second)));
			}else if(it->second.type()==typeid(std::string)){
				output[0][it->first]=_factory.createCharArray(g2sMatlab::toUtf16(std::any_cast<std::string>(it->second)));
			}
		}
		return std::any(matlab::data::Array(output));
	}

	// G2S stores Integer/UInteger images as raw 32-bit integers inside the float buffer:
	// copy the bits, never convert the float value.
	template <typename OutType>
	matlab::data::Array imageToTypedArray(g2s::DataImage &image, const std::vector<size_t>& dims){
		const size_t total=image.dataSize();
		matlab::data::buffer_ptr_t<OutType> buffer=_factory.createBuffer<OutType>(total);
		OutType* destination=buffer.get();
		const OutType* source=reinterpret_cast<const OutType*>(image._data);
		for (size_t i = 0; i < total; ++i) destination[image.flippedCoordinates(i)]=source[i];
		return _factory.createArrayFromBuffer<OutType>(dims, std::move(buffer));
	}

	std::any convert2NativeMatrix(g2s::DataImage &image) override {
		std::vector<size_t> dims(image._dims.rbegin(), image._dims.rend());
		dims.push_back(image._nbVariable);
		switch(image._encodingType){
			case g2s::DataImage::Integer:  return std::any(imageToTypedArray<int32_t>(image, dims));
			case g2s::DataImage::UInteger: return std::any(imageToTypedArray<uint32_t>(image, dims));
			case g2s::DataImage::Float:
			default:                       return std::any(imageToTypedArray<float>(image, dims));
		}
	}

	g2s::DataImage convertNativeMatrix2DataImage(std::any matrix, std::any dataTypeVariable=nullptr) override {
		const matlab::data::Array prh=asArray(matrix);
		matlab::data::Array variableTypeArray;
		const bool hasVariableType=(dataTypeVariable.type()==typeid(matlab::data::Array));
		if(hasVariableType) variableTypeArray=std::any_cast<matlab::data::Array>(dataTypeVariable);

		if(!g2sMatlab::isNumericOrLogical(prh)) sendError("data matrices must be numeric or logical arrays");

		const size_t dataSize=prh.getNumberOfElements();
		int nbOfVariable=1;
		if(hasVariableType) nbOfVariable=int(variableTypeArray.getNumberOfElements());
		const matlab::data::ArrayDimensions dim_array=prh.getDimensions();
		const int dimData=int(dim_array.size())-(nbOfVariable>1);
		if(nbOfVariable>1 && int(dim_array[dimData])!=nbOfVariable)
			sendError("Last dimension of the inputed matrix do not fit -dt parameter size");

		std::vector<unsigned> dimArray(dimData);
		for (int i = 0; i < dimData; ++i) dimArray[i]=unsigned(dim_array[i]);
		std::reverse(dimArray.begin(),dimArray.end());

		g2s::DataImage image(dimData,dimArray.data(),nbOfVariable);
		float *data=image._data;

		if(hasVariableType && g2sMatlab::isNumericOrLogical(variableTypeArray)){
			g2sMatlab::dispatchNumeric(variableTypeArray, [&](auto typed){
				for (int i = 0; i < nbOfVariable; ++i)
				{
					const double value=double(typed.begin()[i]);
					if(value==0.) image._types[i]=g2s::DataImage::VariableType::Continuous;
					if(value==1.) image._types[i]=g2s::DataImage::VariableType::Categorical;
				}
				return 0;
			});
		}

		std::fill(data, data+dataSize, 0.f);
		g2sMatlab::dispatchNumeric(prh, [&](auto typed){
			auto source=typed.begin();
			for (size_t i = 0; i < dataSize; ++i) data[i]=float(source[image.flippedCoordinates(i)]);
			return 0;
		});

		return image;
	}
};

#endif // MATLAB_INTERFACE_MODERN_HPP
