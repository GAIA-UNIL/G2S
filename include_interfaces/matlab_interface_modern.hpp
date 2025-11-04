/*
 * Modern MATLAB interface (C++ Data API)
 */
#ifndef MATLAB_INTERFACE_MODERN_HPP
#define MATLAB_INTERFACE_MODERN_HPP

#include "interfaceTemplate.hpp"

#include "mex.hpp"
#include "mexAdapter.hpp"

#include <memory>
#include <string>
#include <vector>

class InterfaceTemplateMatlabModern : public InterfaceTemplate {
    std::shared_ptr<matlab::engine::MATLABEngine> engine_;
    matlab::data::ArrayFactory factory_;
    std::atomic<bool> done_{false};

public:
    explicit InterfaceTemplateMatlabModern(std::shared_ptr<matlab::engine::MATLABEngine> eng)
    : engine_(std::move(eng)) {}

    ~InterfaceTemplateMatlabModern() { done_ = true; }

    void updateDisplay() override {
        if (engine_)
            engine_->eval(u"drawnow");
    }

    bool userRequestInteruption() override {
        return done_.load();
    }

    bool isDataMatrix(std::any val) override {
        if (val.type() != typeid(matlab::data::Array)) return false;
        auto arr = std::any_cast<matlab::data::Array>(val);
        using T = matlab::data::ArrayType;
        auto at = arr.getType();
        return at == T::DOUBLE || at == T::SINGLE ||
               at == T::INT8 || at == T::UINT8 ||
               at == T::INT16 || at == T::UINT16 ||
               at == T::INT32 || at == T::UINT32 ||
               at == T::INT64 || at == T::UINT64 ||
               at == T::LOGICAL;
    }

    std::string nativeToStandardString(std::any val) override {
        if (val.type() == typeid(std::string)) return std::any_cast<std::string>(val);
        if (val.type() != typeid(matlab::data::Array)) return std::string();
        auto arr = std::any_cast<matlab::data::Array>(val);
        using T = matlab::data::ArrayType;
        switch (arr.getType()) {
            case T::MATLAB_STRING: {
                auto sa = matlab::data::StringArray(arr);
                if (sa.getNumberOfElements() == 0) return std::string();
                matlab::data::optional<std::u16string> opt = sa[0];
                if (!opt.has_value()) return std::string();
                const std::u16string &u16 = *opt;
                std::string s; s.reserve(u16.size());
                for (char16_t ch : u16) s.push_back(static_cast<char>(ch));
                return s;
            }
            case T::CHAR: {
                auto ca = matlab::data::CharArray(arr);
                return ca.toAscii();
            }
            default:
                // Numeric scalar case
                if (arr.getNumberOfElements() == 1) {
                    try { return std::to_string(anyNativeTo<double>(arr)); }
                    catch (...) { return std::string(); }
                }
                return std::string();
        }
    }

    double nativeToScalar(std::any val) override { return anyNativeTo<double>(val); }

    unsigned nativeToUint32(std::any val) override { return anyNativeTo<unsigned>(val); }

    unsigned anyNativeToUnsigned(std::any val) override { return anyNativeTo<unsigned>(val); }
    float anyNativeToFloat(std::any val) override { return anyNativeTo<float>(val); }
    double anyNativeToDouble(std::any val) override { return anyNativeTo<double>(val); }
    long unsigned anyNativeToLongUnsigned(std::any val) override { return anyNativeTo<long unsigned>(val); }

    template <typename type>
    type anyNativeTo(std::any val){
        if (val.type() == typeid(type)) return std::any_cast<type>(val);
        if (val.type() != typeid(matlab::data::Array)) return type();
        auto a = std::any_cast<matlab::data::Array>(val);
        if (a.getNumberOfElements() < 1) return type();
        using T = matlab::data::ArrayType;
        switch (a.getType()){
            case T::DOUBLE: return static_cast<type>(matlab::data::TypedArray<double>(a)[0]);
            case T::SINGLE: return static_cast<type>(matlab::data::TypedArray<float>(a)[0]);
            case T::INT8: return static_cast<type>(matlab::data::TypedArray<int8_t>(a)[0]);
            case T::UINT8: return static_cast<type>(matlab::data::TypedArray<uint8_t>(a)[0]);
            case T::INT16: return static_cast<type>(matlab::data::TypedArray<int16_t>(a)[0]);
            case T::UINT16:return static_cast<type>(matlab::data::TypedArray<uint16_t>(a)[0]);
            case T::INT32: return static_cast<type>(matlab::data::TypedArray<int32_t>(a)[0]);
            case T::UINT32:return static_cast<type>(matlab::data::TypedArray<uint32_t>(a)[0]);
            case T::INT64: return static_cast<type>(matlab::data::TypedArray<int64_t>(a)[0]);
            case T::UINT64:return static_cast<type>(matlab::data::TypedArray<uint64_t>(a)[0]);
            case T::LOGICAL: return static_cast<type>(matlab::data::TypedArray<bool>(a)[0]);
            default: return type();
        }
    }

    std::any ScalarToNative(double val) override {
        auto scalar = factory_.createScalar(val);
        matlab::data::Array a = scalar;
        return std::any(a);
    }

    std::any Uint32ToNative(unsigned val) override {
        auto scalar = factory_.createScalar<uint32_t>(val);
        matlab::data::Array a = scalar;
        return std::any(a);
    }

    void sendError(std::string val) override {
        if (engine_) {
            std::vector<matlab::data::Array> args{
                factory_.createCharArray("g2s:error"),
                factory_.createCharArray(val)
            };
            engine_->feval(u"error", 0, std::move(args));
        }
    }

    void sendWarning(std::string val) override {
        if (engine_) {
            std::vector<matlab::data::Array> args{
                factory_.createCharArray("g2s:warning"),
                factory_.createCharArray(val)
            };
            engine_->feval(u"warning", 0, std::move(args));
        }
    }

    void eraseAndPrint(std::string val) override {
        if (engine_) {
            std::vector<matlab::data::Array> args{ factory_.createCharArray(val) };
            engine_->feval(u"disp", 0, std::move(args));
        }
    }

    std::any convert2NativeMatrix(g2s::DataImage &image) override {
        // Build dims: reverse(image._dims) then append nbVariable
        std::vector<size_t> dims(image._dims.rbegin(), image._dims.rend());
        dims.push_back(image._nbVariable);

        auto total = static_cast<size_t>(image.dataSize());

        switch (image._encodingType) {
            case g2s::DataImage::Float: {
                auto out = factory_.createArray<float>(dims);
                auto it = out.begin();
                #pragma omp parallel for
                for (int i = 0; i < static_cast<int>(total); ++i) {
                    it[image.flippedCoordinates(i)] = static_cast<float>(image._data[i]);
                }
                matlab::data::Array a = out;
                return std::any(a);
            }
            case g2s::DataImage::Integer: {
                auto out = factory_.createArray<int32_t>(dims);
                auto it = out.begin();
                #pragma omp parallel for
                for (int i = 0; i < static_cast<int>(total); ++i) {
                    it[image.flippedCoordinates(i)] = static_cast<int32_t>(image._data[i]);
                }
                matlab::data::Array a = out;
                return std::any(a);
            }
            case g2s::DataImage::UInteger: {
                auto out = factory_.createArray<uint32_t>(dims);
                auto it = out.begin();
                #pragma omp parallel for
                for (int i = 0; i < static_cast<int>(total); ++i) {
                    it[image.flippedCoordinates(i)] = static_cast<uint32_t>(image._data[i]);
                }
                matlab::data::Array a = out;
                return std::any(a);
            }
        }
        // Fallback
        auto out = factory_.createArray<float>(dims);
        matlab::data::Array a = out;
        return std::any(a);
    }

    g2s::DataImage convertNativeMatrix2DataImage(std::any matrix, std::any dataTypeVariable=nullptr) override {
        matlab::data::Array arr = std::any_cast<matlab::data::Array>(matrix);
        matlab::data::Array varType;
        if (dataTypeVariable.type() == typeid(matlab::data::Array))
            varType = std::any_cast<matlab::data::Array>(dataTypeVariable);

        int nbOfVariable = varType.isEmpty() ? 1 : static_cast<int>(varType.getNumberOfElements());
        const auto& dims = arr.getDimensions();
        int dimData = static_cast<int>(dims.size()) - (nbOfVariable > 1 ? 1 : 0);
        if (nbOfVariable > 1 && static_cast<int>(dims[dimData]) != nbOfVariable)
            sendError("Last dimension of the inputed matrix do not fit -dt parameter size");

        std::vector<unsigned> dimArray(dimData);
        for (int i = 0; i < dimData; ++i) dimArray[i] = static_cast<unsigned>(dims[i]);
        std::reverse(dimArray.begin(), dimArray.end());

        g2s::DataImage image(dimData, dimArray.data(), nbOfVariable);
        float *data = image._data;

        // variable types
        if (!varType.isEmpty()) {
            if (varType.getType() == matlab::data::ArrayType::SINGLE) {
                auto vt = matlab::data::TypedArray<float>(varType);
                for (int i = 0; i < nbOfVariable; ++i)
                    image._types[i] = (vt[i] == 0.f) ? g2s::DataImage::VaraibleType::Continuous : g2s::DataImage::VaraibleType::Categorical;
            } else if (varType.getType() == matlab::data::ArrayType::DOUBLE) {
                auto vt = matlab::data::TypedArray<double>(varType);
                for (int i = 0; i < nbOfVariable; ++i)
                    image._types[i] = (vt[i] == 0.) ? g2s::DataImage::VaraibleType::Continuous : g2s::DataImage::VaraibleType::Categorical;
            }
        }

        const int dataSize = static_cast<int>(arr.getNumberOfElements());
        std::fill(data, data + dataSize, 0.f);

        using T = matlab::data::ArrayType;
        auto type = arr.getType();
        if (type == T::DOUBLE) {
            auto src = matlab::data::TypedArray<double>(arr);
            auto it = src.begin();
            #pragma omp parallel for
            for (int i = 0; i < dataSize; ++i) data[i] = static_cast<float>(it[image.flippedCoordinates(i)]);
        } else if (type == T::SINGLE) {
            auto src = matlab::data::TypedArray<float>(arr);
            auto it = src.begin();
            #pragma omp parallel for
            for (int i = 0; i < dataSize; ++i) data[i] = it[image.flippedCoordinates(i)];
        } else if (type == T::UINT8) {
            auto src = matlab::data::TypedArray<uint8_t>(arr);
            auto it = src.begin();
            #pragma omp parallel for
            for (int i = 0; i < dataSize; ++i) data[i] = static_cast<float>(it[image.flippedCoordinates(i)]);
        } else if (type == T::UINT16) {
            auto src = matlab::data::TypedArray<uint16_t>(arr);
            auto it = src.begin();
            #pragma omp parallel for
            for (int i = 0; i < dataSize; ++i) data[i] = static_cast<float>(it[image.flippedCoordinates(i)]);
        } else if (type == T::UINT32) {
            auto src = matlab::data::TypedArray<uint32_t>(arr);
            auto it = src.begin();
            #pragma omp parallel for
            for (int i = 0; i < dataSize; ++i) data[i] = static_cast<float>(it[image.flippedCoordinates(i)]);
        } else if (type == T::UINT64) {
            auto src = matlab::data::TypedArray<uint64_t>(arr);
            auto it = src.begin();
            #pragma omp parallel for
            for (int i = 0; i < dataSize; ++i) data[i] = static_cast<float>(it[image.flippedCoordinates(i)]);
        } else if (type == T::INT8) {
            auto src = matlab::data::TypedArray<int8_t>(arr);
            auto it = src.begin();
            #pragma omp parallel for
            for (int i = 0; i < dataSize; ++i) data[i] = static_cast<float>(it[image.flippedCoordinates(i)]);
        } else if (type == T::INT16) {
            auto src = matlab::data::TypedArray<int16_t>(arr);
            auto it = src.begin();
            #pragma omp parallel for
            for (int i = 0; i < dataSize; ++i) data[i] = static_cast<float>(it[image.flippedCoordinates(i)]);
        } else if (type == T::INT32) {
            auto src = matlab::data::TypedArray<int32_t>(arr);
            auto it = src.begin();
            #pragma omp parallel for
            for (int i = 0; i < dataSize; ++i) data[i] = static_cast<float>(it[image.flippedCoordinates(i)]);
        } else if (type == T::INT64) {
            auto src = matlab::data::TypedArray<int64_t>(arr);
            auto it = src.begin();
            #pragma omp parallel for
            for (int i = 0; i < dataSize; ++i) data[i] = static_cast<float>(it[image.flippedCoordinates(i)]);
        } else if (type == T::LOGICAL) {
            auto src = matlab::data::TypedArray<bool>(arr);
            auto it = src.begin();
            #pragma omp parallel for
            for (int i = 0; i < dataSize; ++i) data[i] = it[image.flippedCoordinates(i)] ? 1.f : 0.f;
        }

        return image;
    }
};

#endif // MATLAB_INTERFACE_MODERN_HPP
