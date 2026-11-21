/*
 * G2S - geostat: shared foundation of the Gaussian geostatistics algorithms.
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

// Command-line front end of the covariance models, shared by every Gaussian
// executable so that -cov, -cvar and -ii are read identically:
//
//   -cov  "model"             one or more models (repeat -cov, or several
//                             values after one -cov); model i is index i
//   -cvar name value          binds a variable; repeat for each variable.
//                             value is a number (constant) or an image name
//                             (one variable): a spatially varying parameter,
//                             on the -di grid, or the covariance map of
//                             map(name), of any size with one axis per lag
//                             component and lag 0 at its centre (dims/2).
//                             An image whose values are all equal is bound as
//                             a constant (stationary fast path).
//                             "-cvar name=value" (one token) is accepted too.
//   -ii   image               model index per node (0-based, integers), same
//                             grid as -di; model 0 everywhere when absent.
//
// Negative constants: argumentReader() treats "-30" as a flag, so
// readCovarianceArguments re-reads argv after each -cvar and accepts tokens
// that are numbers even when they start with '-' (and removes the bogus "-30"
// keys from the argument map). Through the Python, MATLAB and R interfaces
// the server rebuilds argv grouped by flag, in the order the values were
// given for that flag, so g2s('-cvar','theta',-30,'-cvar','L',40) reaches
// the executable as "-cvar theta -30 L 40" and works. A value given as text
// ('-30') works too: the interfaces treat a text that starts with '-' and a
// digit (or '.' and a digit) as a value, not a flag (isG2sFlag in
// include_interfaces/interfaceTemplate.hpp); clients built before that
// change send '-30' as a separate flag, which the server moves away from
// -cvar ("'theta' has no value"). "theta=-30" (one token) works everywhere,
// as does putting the value in the model text.
//
// Usage in an executable (pyDev style):
//   g2s::geostat::CovarianceArguments covarianceArguments;
//   if (!g2s::geostat::readCovarianceArguments(arg, argc, argv, covarianceArguments, error)) ...   // before the "ignored" loop
//   ... load -di ...
//   g2s::geostat::CovarianceModelSet models;
//   std::vector<unsigned> grid = DI._dims;
//   if (!g2s::geostat::loadCovarianceModels(reportFile, covarianceArguments, grid, unsigned(grid.size()), models, error)) ...
//   node ids are then cell indices of the -di grid.

#ifndef G2S_GEOSTAT_COVARIANCE_ARGUMENTS_HPP
#define G2S_GEOSTAT_COVARIANCE_ARGUMENTS_HPP

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "DataImage.hpp"
#include "jobReporting.hpp"
#include "geostat/covarianceModel.hpp"

namespace g2s {
namespace geostat {

struct CovarianceArguments {
	std::vector<std::string> models;                               // -cov, in order
	std::vector<std::pair<std::string, std::string> > variables;   // -cvar name value, in order
	std::string indexFileName;                                     // -ii, empty when absent
};

// True when the whole text is a finite number (strtod syntax, sign allowed).
inline bool parseNumberText(const std::string& text, double* value = nullptr) {
	if (text.empty()) return false;
	char* end = nullptr;
	const double parsed = std::strtod(text.c_str(), &end);
	if (end != text.c_str() + text.size() || !std::isfinite(parsed)) return false;
	for (char c : text) if (std::isalpha(static_cast<unsigned char>(c)) && c != 'e' && c != 'E') return false; // no inf/nan/hex
	if (value) *value = parsed;
	return true;
}

// Takes -cov, -cvar and -ii out of `arg`. argv may be null (then the -cvar
// values come from `arg` only and cannot start with '-').
inline bool readCovarianceArguments(std::multimap<std::string, std::string>& arg, int argc, char const* argv[],
                                    CovarianceArguments& out, std::string& error) {
	out = CovarianceArguments();
	bool ok = true;
	for (auto it = arg.lower_bound("-cov"); it != arg.upper_bound("-cov"); ++it) {
		if (it->second.empty()) { error = "-cov needs a covariance model, e.g. -cov \"1*exponential(range=10)\""; ok = false; }
		else out.models.push_back(it->second);
	}
	arg.erase("-cov");

	std::vector<std::string> values;
	if (argv) {
		for (int i = 0; i < argc; ++i) {
			if (std::string(argv[i]) != "-cvar") continue;
			for (int j = i + 1; j < argc; ++j) {
				const std::string token = argv[j];
				const bool numeric = parseNumberText(token);
				if (token.size() > 1 && token[0] == '-' && !numeric) break;
				values.push_back(token);
				if (token.size() > 1 && token[0] == '-') arg.erase(token); // argumentReader took it for a flag
			}
		}
	} else {
		for (auto it = arg.lower_bound("-cvar"); it != arg.upper_bound("-cvar"); ++it) if (!it->second.empty()) values.push_back(it->second);
	}
	arg.erase("-cvar");
	for (size_t i = 0; i < values.size();) {
		const size_t equal = values[i].find('=');
		if (equal != std::string::npos && equal > 0) {
			out.variables.push_back({ values[i].substr(0, equal), values[i].substr(equal + 1) });
			i += 1;
		} else if (i + 1 < values.size()) {
			out.variables.push_back({ values[i], values[i + 1] });
			i += 2;
		} else {
			error = "-cvar expects name value pairs; '" + values[i] + "' has no value (a negative number given as text by an older client"
			        " is taken for a flag: write -cvar " + values[i] + "=value instead)";
			ok = false;
			break;
		}
	}
	for (const std::pair<std::string, std::string>& variable : out.variables) {
		if (variable.second.empty()) { error = "-cvar " + variable.first + ": missing value"; ok = false; }
		if (!isValidVariableName(variable.first)) {
			error = isReservedWord(variable.first) ? "-cvar: '" + variable.first + "' is a reserved word (structure, keyword or function name) and cannot be a variable"
			                                       : "-cvar: '" + variable.first + "' is not a valid variable name (letters, digits and '_', not starting with a digit)";
			ok = false;
		}
	}

	if (arg.count("-ii") > 1) { error = "only one -ii (model index image) is possible"; ok = false; }
	else if (arg.count("-ii") == 1) out.indexFileName = arg.find("-ii")->second;
	arg.erase("-ii");
	return ok;
}

// Loads the -cvar images and the -ii index, checks the per-node images
// (fields, -ii) against the grid and builds the model set. Map tables are not
// checked against the grid. gridDims: the -di grid (DataImage axis order);
// when empty it is taken from the first per-node image (and returned), which
// lets covEval run without -di. ndim: number of lag components; 0 means gridDims.size().
// Logs the inputs and the canonical models to reportFile (may be null).
inline bool loadCovarianceModels(FILE* reportFile, const CovarianceArguments& args, std::vector<unsigned>& gridDims, unsigned ndim,
                                 CovarianceModelSet& models, std::string& error) {
	try {
		if (args.models.empty()) throw Error("-cov is mandatory (covariance model)");
		std::vector<std::string> mapNames;
		for (const std::string& text : args.models) {
			const ModelSpec spec = ModelSpec::parse(text);
			for (const std::string& name : spec.mapNames()) mapNames.push_back(name);
		}
		auto isMap = [&](const std::string& name) { return std::find(mapNames.begin(), mapNames.end(), name) != mapNames.end(); };
		auto gridText = [](const std::vector<unsigned>& dims) { return g2s::reporting::joinUnsignedVector(dims); };
		// onGrid: one value per node (fields, -ii); a map table has its own size.
		auto loadImage = [&](const std::string& label, const std::string& file, g2s::DataImage& image, bool onGrid) {
			image = g2s::DataImage::createFromFile(file);
			g2s::reporting::logInput(reportFile, label, file, image);
			if (image.isEmpty()) throw Error("failed to load " + label + " " + file);
			if (image._nbVariable != 1) throw Error(label + " must have a single variable, " + file + " has " + std::to_string(image._nbVariable));
			if (!onGrid) return;
			if (gridDims.empty()) gridDims = image._dims;
			else if (image._dims != gridDims)
				throw Error(label + " (" + file + ") has size " + gridText(image._dims) + " but the grid is " + gridText(gridDims));
		};

		Bindings bindings;
		for (const std::pair<std::string, std::string>& variable : args.variables) {
			const std::string& name = variable.first;
			const std::string label = "-cvar " + name;
			double number = 0.0;
			if (parseNumberText(variable.second, &number)) {
				if (isMap(name)) throw Error(label + ": map(" + name + ") needs a covariance map image, not a number");
				bindings.setConstant(name, number);
				g2s::reporting::logParameter(reportFile, "cvar_" + name, variable.second);
				continue;
			}
			g2s::DataImage image;
			loadImage(label, variable.second, image, !isMap(name));
			const size_t count = image.dataSize();
			std::vector<float> data(image._data, image._data + count);
			if (isMap(name)) {
				bindings.setTable(name, std::move(data), image._dims);
				continue;
			}
			bool constant = true;
			for (size_t i = 1; i < count && constant; ++i) constant = data[i] == data[0];
			if (constant && std::isfinite(data[0])) {
				bindings.setConstant(name, double(data[0]));
				g2s::reporting::logParameter(reportFile, "cvar_" + name, "constant image " + formatNumber(double(data[0])));
			} else {
				bindings.setField(name, std::move(data));
			}
		}

		std::vector<unsigned> modelIndex;
		if (!args.indexFileName.empty()) {
			g2s::DataImage index;
			loadImage("-ii", args.indexFileName, index, true);
			modelIndex.resize(index.dataSize());
			for (size_t i = 0; i < modelIndex.size(); ++i) {
				const float v = index._data[i];
				if (!(v >= 0) || v != std::floor(v) || v > 1e9f)
					throw Error("-ii must hold model indices (0, 1, ...); found " + formatNumber(double(v)) + " at node " + std::to_string(i));
				modelIndex[i] = unsigned(v);
			}
		}

		if (ndim == 0) ndim = unsigned(gridDims.size());
		if (!gridDims.empty() && ndim != gridDims.size())
			throw Error("the lags have " + std::to_string(ndim) + " components but the grid has " + std::to_string(gridDims.size()) + " dimensions");
		models = buildCovarianceModelSet(args.models, bindings, ndim, modelIndex);
		const std::vector<std::string> canonical = models.canonical();
		for (size_t i = 0; i < canonical.size(); ++i) g2s::reporting::logParameter(reportFile, "cov_" + std::to_string(i), canonical[i]);
		g2s::reporting::logParameter(reportFile, "cov_stationary", models.isStationary() ? "true" : "false");
		return true;
	} catch (const Error& e) {
		error = e.what();
		return false;
	}
}

} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_COVARIANCE_ARGUMENTS_HPP
