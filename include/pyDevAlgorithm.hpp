/*
 * G2S - pyDev: prototype sequential-simulation algorithms in Python.
 * Copyright (C) 2018-2026, Mathieu Gravey (gravey.mathieu@gmail.com) and UNIL (University of Lausanne)
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

// Embedded CPython runtime used by the pyDev algorithm.
//
// Threading model (Python >= 3.14, with or without the GIL):
//   * the main thread initialises the interpreter, loads the user class,
//     calls build(), then detaches (PyEval_SaveThread) before workers start;
//   * every worker owns one PyThreadState for its whole life (AttachedWorker)
//     and is attached only while it builds call arguments, calls sample() and
//     converts the result; it is detached during the C++ neighbour search and
//     while it waits for neighbours simulated by other threads, so no thread
//     ever blocks while holding the interpreter;
//   * on a free-threaded build (python3.14t) sample() calls run truly in
//     parallel; on a default build they are serialised by the GIL but still
//     correct.
//
// Every PyObject* produced here is owned by a PyRef and released while the
// thread is attached.

#ifndef PY_DEV_ALGORITHM_HPP
#define PY_DEV_ALGORITHM_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#if PY_VERSION_HEX < 0x030E0000
#error "pyDev requires Python >= 3.14 (build with PYDEV_PYTHON pointing to a 3.14+ interpreter)"
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "DataImage.hpp"

namespace g2s {
namespace pydev {

// Owning reference to a Python object. Must be destroyed while attached.
class PyRef {
public:
	PyRef() = default;
	explicit PyRef(PyObject* object) : _object(object) {}
	PyRef(const PyRef&) = delete;
	PyRef& operator=(const PyRef&) = delete;
	PyRef(PyRef&& other) noexcept : _object(other._object) { other._object = nullptr; }
	PyRef& operator=(PyRef&& other) noexcept {
		if (this != &other) {
			Py_XDECREF(_object);
			_object = other._object;
			other._object = nullptr;
		}
		return *this;
	}
	~PyRef() { Py_XDECREF(_object); }

	PyObject* get() const { return _object; }
	PyObject* release() { PyObject* object = _object; _object = nullptr; return object; }
	void reset(PyObject* object = nullptr) { Py_XDECREF(_object); _object = object; }
	explicit operator bool() const { return _object != nullptr; }

private:
	PyObject* _object = nullptr;
};

// Formats (and clears) the pending Python exception, traceback included.
inline std::string takePythonError(const std::string& context) {
	PyObject* raised = PyErr_GetRaisedException();
	if (!raised) return context + ": unknown Python error";
	PyRef exception(raised);
	std::string message;
	PyRef tracebackModule(PyImport_ImportModule("traceback"));
	if (tracebackModule) {
		PyRef lines(PyObject_CallMethod(tracebackModule.get(), "format_exception", "O", exception.get()));
		if (lines) {
			PyRef separator(PyUnicode_FromString(""));
			PyRef joined(separator ? PyUnicode_Join(separator.get(), lines.get()) : nullptr);
			if (joined) {
				const char* text = PyUnicode_AsUTF8(joined.get());
				if (text) message = text;
			}
		}
	}
	if (message.empty()) {
		PyErr_Clear();
		PyRef text(PyObject_Str(exception.get()));
		const char* utf8 = text ? PyUnicode_AsUTF8(text.get()) : nullptr;
		message = utf8 ? utf8 : "unprintable Python exception";
	}
	PyErr_Clear();
	while (!message.empty() && message.back() == '\n') message.pop_back();
	return context + ":\n" + message;
}

// DataImage -> NumPy (copy). Axis order is reversed so that the array has the
// same layout as on the client side, with variables on a trailing axis when
// there is more than one (same convention as the Python interface).
inline PyObject* dataImageToNumpy(g2s::DataImage& image) {
	const size_t ndim = image._dims.size();
	std::vector<npy_intp> dims(ndim + 1);
	for (size_t i = 0; i < ndim; ++i) dims[i] = image._dims[i];
	std::reverse(dims.begin(), dims.begin() + ndim);
	dims[ndim] = image._nbVariable;
	int type = NPY_FLOAT32;
	if (image._encodingType == g2s::DataImage::Integer) type = NPY_INT32;
	if (image._encodingType == g2s::DataImage::UInteger) type = NPY_UINT32;
	PyObject* array = PyArray_SimpleNew(int(ndim + (image._nbVariable > 1)), dims.data(), type);
	if (!array) return nullptr;
	std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)), image._data, size_t(image.dataSize()) * sizeof(float));
	return array;
}

// Process-wide interpreter. Initialise once on the main thread.
class Runtime {
public:
	Runtime() = default;
	Runtime(const Runtime&) = delete;
	Runtime& operator=(const Runtime&) = delete;
	~Runtime() { finalize(); }

	// pythonExecutable: interpreter (or venv python) whose prefix and
	// site-packages should be used; empty keeps CPython's default discovery.
	bool initialize(const std::string& pythonExecutable, std::string& error) {
		PyConfig config;
		PyConfig_InitPythonConfig(&config);
		config.install_signal_handlers = 0; // let the server's SIGTERM/SIGKILL behave natively
		config.parse_argv = 0;
		PyStatus status = PyStatus_Ok();
		if (!pythonExecutable.empty()) {
			status = PyConfig_SetBytesString(&config, &config.program_name, pythonExecutable.c_str());
			if (!PyStatus_Exception(status))
				status = PyConfig_SetBytesString(&config, &config.executable, pythonExecutable.c_str());
		}
		if (!PyStatus_Exception(status)) status = Py_InitializeFromConfig(&config);
		PyConfig_Clear(&config);
		if (PyStatus_Exception(status)) {
			error = std::string("Python initialisation failed: ") + (status.err_msg ? status.err_msg : "unknown error");
			return false;
		}
		_initialized = true;
		if (_import_array() < 0) {
			error = takePythonError("unable to import numpy (is numpy installed for this interpreter?)");
			return false;
		}
		return true;
	}

	bool gilEnabled() const {
		PyRef sys(PyImport_ImportModule("sys"));
		if (!sys) { PyErr_Clear(); return true; }
		PyRef enabled(PyObject_CallMethod(sys.get(), "_is_gil_enabled", nullptr));
		if (!enabled) { PyErr_Clear(); return true; }
		return PyObject_IsTrue(enabled.get()) == 1;
	}

	std::string version() const { return Py_GetVersion(); }

	// Main thread: release the interpreter before starting workers.
	void detachMainThread() { if (!_mainState) _mainState = PyEval_SaveThread(); }
	void attachMainThread() { if (_mainState) { PyEval_RestoreThread(_mainState); _mainState = nullptr; } }

	void finalize() {
		if (!_initialized) return;
		attachMainThread();
		Py_FinalizeEx();
		_initialized = false;
	}

private:
	bool _initialized = false;
	PyThreadState* _mainState = nullptr;
};

// One per worker thread: creates the thread state once, keeps it detached
// except inside an Attach scope. Works on the main thread too (re-uses the
// main thread state through PyGILState).
class AttachedWorker {
public:
	AttachedWorker() {
		_gilState = PyGILState_Ensure();
		_threadState = PyEval_SaveThread();
	}
	~AttachedWorker() {
		PyEval_RestoreThread(_threadState);
		PyGILState_Release(_gilState);
	}
	AttachedWorker(const AttachedWorker&) = delete;
	AttachedWorker& operator=(const AttachedWorker&) = delete;

	class Attach {
	public:
		explicit Attach(AttachedWorker& worker) : _worker(worker) { PyEval_RestoreThread(_worker._threadState); }
		~Attach() { _worker._threadState = PyEval_SaveThread(); }
		Attach(const Attach&) = delete;
		Attach& operator=(const Attach&) = delete;
	private:
		AttachedWorker& _worker;
	};

private:
	PyGILState_STATE _gilState;
	PyThreadState* _threadState = nullptr;
};

// Result of one sample() call.
struct SampleResult {
	std::vector<double> values;
	unsigned index = 0;
};

// The user's Python class. All methods must be called while attached.
class Algorithm {
public:
	// Loads `className` from the Python file `filePath` and instantiates it with
	// (TIs, kernel, di, dt, path).
	bool load(const std::string& filePath, const std::string& className,
	          std::vector<g2s::DataImage>& TIs, g2s::DataImage& kernel, g2s::DataImage& di,
	          g2s::DataImage* simulationPath, std::string& error) {
		_nbVariable = di._nbVariable;
		_nbDims = unsigned(di._dims.size());

		static const char* loaderSource =
			"def _g2s_pydev_load(path, class_name):\n"
			"    import importlib.util, os, sys\n"
			"    path = os.path.abspath(path)\n"
			"    folder = os.path.dirname(path)\n"
			"    if folder not in sys.path:\n"
			"        sys.path.insert(0, folder)\n"
			"    name = os.path.splitext(os.path.basename(path))[0]\n"
			"    spec = importlib.util.spec_from_file_location(name, path)\n"
			"    if spec is None or spec.loader is None:\n"
			"        raise ImportError(f'cannot load pyDev algorithm from {path}')\n"
			"    module = importlib.util.module_from_spec(spec)\n"
			"    sys.modules[name] = module\n"
			"    spec.loader.exec_module(module)\n"
			"    if not hasattr(module, class_name):\n"
			"        raise AttributeError(f'{path} does not define a class named {class_name!r}')\n"
			"    return getattr(module, class_name)\n";

		PyRef globals(PyDict_New());
		PyRef builtins(PyImport_ImportModule("builtins"));
		if (!globals || !builtins || PyDict_SetItemString(globals.get(), "__builtins__", builtins.get()) < 0) {
			error = takePythonError("pyDev loader setup failed");
			return false;
		}
		PyRef executed(PyRun_String(loaderSource, Py_file_input, globals.get(), globals.get()));
		if (!executed) { error = takePythonError("pyDev loader setup failed"); return false; }
		PyObject* loader = PyDict_GetItemString(globals.get(), "_g2s_pydev_load"); // borrowed, kept alive by globals
		PyRef algorithmClass(PyObject_CallFunction(loader, "ss", filePath.c_str(), className.c_str()));
		if (!algorithmClass) { error = takePythonError("unable to load " + filePath + ":" + className); return false; }

		PyRef tis(PyList_New(Py_ssize_t(TIs.size())));
		if (!tis) { error = takePythonError("unable to build TI list"); return false; }
		for (size_t i = 0; i < TIs.size(); ++i) {
			PyObject* ti = dataImageToNumpy(TIs[i]);
			if (!ti) { error = takePythonError("unable to convert TI"); return false; }
			PyList_SET_ITEM(tis.get(), Py_ssize_t(i), ti); // steals ti
		}
		PyRef kernelArray(dataImageToNumpy(kernel));
		PyRef diArray(dataImageToNumpy(di));
		npy_intp typeCount = npy_intp(di._types.size());
		PyRef types(PyArray_SimpleNew(1, &typeCount, NPY_INT32));
		if (!kernelArray || !diArray || !types) { error = takePythonError("unable to convert inputs"); return false; }
		int32_t* typePtr = static_cast<int32_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(types.get())));
		for (size_t i = 0; i < di._types.size(); ++i)
			typePtr[i] = (di._types[i] == g2s::DataImage::Categorical) ? 1 : 0;
		PyRef path;
		if (simulationPath && !simulationPath->isEmpty()) {
			path.reset(dataImageToNumpy(*simulationPath));
			if (!path) { error = takePythonError("unable to convert simulation path"); return false; }
		} else {
			path.reset(Py_NewRef(Py_None));
		}

		_instance.reset(PyObject_CallFunctionObjArgs(algorithmClass.get(), tis.get(), kernelArray.get(), diArray.get(), types.get(), path.get(), nullptr));
		if (!_instance) { error = takePythonError(className + ".__init__ failed"); return false; }

		_sampleName.reset(PyUnicode_InternFromString("sample"));
		if (!_sampleName) { error = takePythonError("internal error"); return false; }
		int hasSample = PyObject_HasAttrWithError(_instance.get(), _sampleName.get());
		if (hasSample != 1) {
			if (hasSample < 0) error = takePythonError(className + " lookup of sample() failed");
			else error = className + " must define a sample(self, coordinates, values, randUniform, indexInThePath, threadId, fullStationary, variableOfInterest, idTI4Sampling) method";
			return false;
		}
		return true;
	}

	// Optional build(n) hook.
	bool build(const std::vector<unsigned>& nbNeighbors, std::string& error) {
		return callOptional("build", &nbNeighbors, error);
	}

	// Optional finalize() hook, called once after the simulation succeeded.
	bool finalize(std::string& error) { return callOptional("finalize", nullptr, error); }

	// Drops the Python instance (runs __del__). Call while attached.
	void release() { _instance.reset(); _sampleName.reset(); }

	// Calls sample(). offsets are in DataImage axis order; they are passed to
	// Python in NumPy axis order (reversed) so they index the arrays given to
	// __init__ directly. values holds nbVariable entries per neighbour.
	bool sample(const std::vector<int>& offsets, const std::vector<double>& values, size_t neighborCount,
	            float randUniform, uint64_t indexInThePath, unsigned threadId, int idTI4Sampling,
	            SampleResult& result, std::string& error) {
		npy_intp coordinateDims[2] = { npy_intp(neighborCount), npy_intp(_nbDims) };
		npy_intp valueDims[2] = { npy_intp(neighborCount), npy_intp(_nbVariable) };
		PyRef coordinates(PyArray_SimpleNew(2, coordinateDims, NPY_INT32));
		PyRef valueArray(PyArray_SimpleNew(2, valueDims, NPY_FLOAT64));
		if (!coordinates || !valueArray) { error = takePythonError("unable to allocate sample() arguments"); return false; }
		int32_t* coordinatePtr = static_cast<int32_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(coordinates.get())));
		for (size_t n = 0; n < neighborCount; ++n)
			for (unsigned d = 0; d < _nbDims; ++d)
				coordinatePtr[n * _nbDims + d] = offsets[n * _nbDims + (_nbDims - 1 - d)];
		if (!values.empty())
			std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject*>(valueArray.get())), values.data(), neighborCount * _nbVariable * sizeof(double));

		PyRef seed(PyFloat_FromDouble(randUniform));
		PyRef pathIndex(PyLong_FromUnsignedLongLong(indexInThePath));
		PyRef thread(PyLong_FromUnsignedLong(threadId));
		PyRef variable(PyLong_FromLong(0));
		PyRef tiIndex(PyLong_FromLong(idTI4Sampling));
		if (!seed || !pathIndex || !thread || !variable || !tiIndex) { error = takePythonError("unable to allocate sample() arguments"); return false; }

		PyObject* args[] = { _instance.get(), coordinates.get(), valueArray.get(), seed.get(), pathIndex.get(),
		                     thread.get(), Py_False, variable.get(), tiIndex.get() };
		PyRef output(PyObject_VectorcallMethod(_sampleName.get(), args, sizeof(args) / sizeof(args[0]), nullptr));
		if (!output) { error = takePythonError("sample() raised at path index " + std::to_string(indexInThePath)); return false; }
		return parseSampleOutput(output.get(), indexInThePath, result, error);
	}

private:
	bool callOptional(const char* name, const std::vector<unsigned>* nbNeighbors, std::string& error) {
		int has = PyObject_HasAttrStringWithError(_instance.get(), name);
		if (has < 0) { error = takePythonError(std::string("lookup of ") + name + "() failed"); return false; }
		if (has == 0) return true;
		PyRef output;
		if (nbNeighbors) {
			PyRef list(PyList_New(Py_ssize_t(nbNeighbors->size())));
			if (!list) { error = takePythonError("internal error"); return false; }
			for (size_t i = 0; i < nbNeighbors->size(); ++i) {
				PyObject* value = PyLong_FromUnsignedLong((*nbNeighbors)[i]);
				if (!value) { error = takePythonError("internal error"); return false; }
				PyList_SET_ITEM(list.get(), Py_ssize_t(i), value);
			}
			output.reset(PyObject_CallMethod(_instance.get(), name, "O", list.get()));
		} else {
			output.reset(PyObject_CallMethod(_instance.get(), name, nullptr));
		}
		if (!output) { error = takePythonError(std::string(name) + "() raised"); return false; }
		return true;
	}

	bool parseSampleOutput(PyObject* output, uint64_t indexInThePath, SampleResult& result, std::string& error) {
		const std::string where = "sample() at path index " + std::to_string(indexInThePath);
		PyObject* valuesObject = output; // borrowed
		result.index = 0;
		if (PyTuple_Check(output)) {
			if (PyTuple_GET_SIZE(output) != 2) {
				error = where + " must return values or a (values, index) tuple, got a tuple of size " + std::to_string(PyTuple_GET_SIZE(output));
				return false;
			}
			valuesObject = PyTuple_GET_ITEM(output, 0);
			PyObject* indexObject = PyTuple_GET_ITEM(output, 1);
			PyRef asIndex(PyNumber_Index(indexObject));
			if (!asIndex) { error = takePythonError(where + ": the second tuple item must be an integer index"); return false; }
			long long index = PyLong_AsLongLong(asIndex.get());
			if (index == -1 && PyErr_Occurred()) { error = takePythonError(where + ": index out of range"); return false; }
			if (index < 0 || index > (long long)UINT32_MAX) { error = where + ": index must be in [0, 2^32-1], got " + std::to_string(index); return false; }
			result.index = unsigned(index);
		}
		PyRef array(PyArray_FromAny(valuesObject, PyArray_DescrFromType(NPY_FLOAT64), 0, 1,
		                            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST, nullptr));
		if (!array) { error = takePythonError(where + ": returned values are not convertible to a float vector"); return false; }
		PyArrayObject* arrayObject = reinterpret_cast<PyArrayObject*>(array.get());
		const npy_intp size = PyArray_SIZE(arrayObject);
		if (size != npy_intp(_nbVariable)) {
			error = where + " returned " + std::to_string((long long)size) + " value(s), expected " + std::to_string(_nbVariable) + " (one per variable)";
			return false;
		}
		const double* data = static_cast<const double*>(PyArray_DATA(arrayObject));
		result.values.assign(data, data + size);
		return true;
	}

	PyRef _instance;
	PyRef _sampleName;
	unsigned _nbVariable = 1;
	unsigned _nbDims = 0;
};

} // namespace pydev
} // namespace g2s

#endif // PY_DEV_ALGORITHM_HPP
