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

// Covariance models of the Gaussian geostatistics algorithms (sgs, kriging,
// FFT-MA, spectral, turning bands, covEval). This header knows nothing about
// grids: a "node" is an index into the per-node arrays given as bindings (a
// grid cell index for the grid front end, a point index for a future point
// front end). Grid-specific glue (DataImage loading, -di checks) lives in
// covarianceArguments.hpp.
//
// ---------------------------------------------------------------- grammar
// A model is a sum of nested structures; the factor in front of a structure
// is its partial sill:
//
//   0.1*nugget + 0.6*spherical(range=[40,20], angle=30) + 0.3*exponential(range=100)
//   nug*nugget + (1-nug)*spherical(range=[L, L/2], angle=theta)
//
//   model     := term ('+' term)*
//   term      := [sill '*'] structure
//   sill      := a product/quotient of factors (see expression.hpp); a sum or
//                difference must be parenthesised: (1-nug)*spherical(...).
//                Without a sill the partial sill is 1. Structures cannot be
//                subtracted.
//   structure := nugget | nugget() | map(NAME) | TYPE(arguments)
//   TYPE      := spherical | exponential | gaussian | cubic | matern | stable
//   arguments := KEY=value (',' KEY=value)*, any order, each key at most once
//   value     := expression | '[' expression (',' expression)* ']'
//
// Structure names, keywords and functions are case-insensitive and reserved
// (also "h", the lag, which no expression can use). Variables are bound with
// -cvar (covarianceArguments.hpp) or Bindings below.
//
// ---------------------------------------------------------------- parameters
//   range=r | range=[r0, r1, ...]  practical range (see below); one value
//                                  (isotropic) or one per array axis
//                                  (geometric anisotropy, before rotation)
//   scale=s | scale=[s0, ...]      the raw scale parameter instead of range
//   angle=a | angle=[a, b, c]      rotation in degrees (2D: one angle; 3D: one
//                                  or three)
//   nu=v                           Matérn smoothness, 0 < nu <= 100
//   alpha=v                        stable exponent, 0 < alpha <= 2
// Required: range or scale (not both) for every structure except nugget and
// map; nu for matern; alpha for stable. nugget and map take nothing else.
//
// ------------------------------------------------- correlation functions
// Let t = |A h| be the lag measured in practical ranges (A below), and
// rho(t) the correlation, so that C(h) = sill * rho(t):
//   spherical    rho = 1 - 1.5 t + 0.5 t^3 for t < 1, 0 beyond;   range = scale
//   cubic        rho = 1 - 7 t^2 + 35/4 t^3 - 7/2 t^5 + 3/4 t^7 for t < 1,
//                0 beyond (GSLIB cubic);                          range = scale
//   exponential  rho = exp(-3 t)          = exp(-h/scale),        range = 3 scale
//   gaussian     rho = exp(-3 t^2)        = exp(-(h/scale)^2),    range = sqrt(3) scale
//   stable       rho = exp(-3 t^alpha)    = exp(-(h/scale)^alpha), range = 3^(1/alpha) scale
//   matern       rho = 2^(1-nu)/Gamma(nu) (h/scale)^nu K_nu(h/scale),
//                range = u95(nu) scale, where u95(nu) solves rho = exp(-3) for
//                scale 1 (maternPracticalRange; u95(0.5) = 3, u95(1.5) = 4.7490,
//                u95(2.5) = 5.9245). This is the plain h/scale form, not the
//                sqrt(2 nu) h/scale variant.
//   nugget       sill at lag exactly 0, 0 elsewhere.
//   map(name)    sill_coefficient * M[centre + round(h)], 0 outside M; the
//                centre of M is dims/2 (integer division) on each axis.
// So "range" is the GSLIB practical range: spherical and cubic reach 0 there,
// the others have dropped to exp(-3) = 4.98% of their sill, and matern with
// nu = 0.5 is exactly exponential with the same range, stable with alpha = 1
// (2) is exactly exponential (gaussian) with the same range.
// For map(name) the partial sill (ResolvedStructure::sill) is the coefficient
// times the value at the centre of M.
//
// ----------------------------------------------------- axes, anisotropy, angles
// Lags are in grid-cell units, one component per array axis, in DataImage
// axis order: component 0 is the fastest-varying axis, which is the LAST index
// of the numpy / MATLAB array (columns of a 2D array), component 1 is the one
// before it (rows of a 2D array), component 2 the first index of a 3D array.
// range=[r0, r1, r2] and angle use the same order.
//
// R_ij(theta) is the rotation in the plane of axes i and j that turns axis i
// towards axis j by theta degrees: R e_i = cos(theta) e_i + sin(theta) e_j,
// R e_j = -sin(theta) e_i + cos(theta) e_j. Then
//   2D: R = R_01(a)                         (angle=a)
//   3D: R = R_01(a) * R_02(b) * R_12(c)     (angle=[a, b, c]; angle=a means [a, 0, 0])
// The k-th column of R is the direction of range[k]. With axis 0 to the right
// and axis 1 upwards, a > 0 is counter-clockwise (on an image displayed with
// row 0 at the top, e.g. imshow, it appears clockwise). In 3D, the direction
// of range[0] is (cos a cos b, sin a cos b, sin b): a is its azimuth in the
// (0,1) plane measured from axis 0 towards axis 1, b its elevation towards
// axis 2, and c turns the two other principal axes about it (axis 1 towards
// axis 2).
// The anisotropy transform is A = diag(1/range) * R^T, i.e. the lag expressed
// in the principal axes and divided by the practical ranges; t = |A h| and
// t = 1 on the practical-range ellipsoid.
//
// --------------------------------------------------- stationarity, threads
// Parameters are resolved at a node (local stationarity): every variable takes
// its value at that node, then the structure is a plain stationary covariance.
// When no variable is spatially varying, the model is resolved once
// (isStationary()). Every evaluation method is const and allocation-free;
// a ResolvedModel used as scratch must belong to one thread.

#ifndef G2S_GEOSTAT_COVARIANCE_MODEL_HPP
#define G2S_GEOSTAT_COVARIANCE_MODEL_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "geostat/expression.hpp"
#include "geostat/specialFunctions.hpp"

namespace g2s {
namespace geostat {

// Largest number of axes a model supports (rotations only in 2D and 3D).
constexpr unsigned kMaxDims = 4;

enum class StructureType { Nugget, Spherical, Exponential, Gaussian, Cubic, Matern, Stable, Map };

inline const char* structureName(StructureType type) {
	switch (type) {
		case StructureType::Nugget: return "nugget";
		case StructureType::Spherical: return "spherical";
		case StructureType::Exponential: return "exponential";
		case StructureType::Gaussian: return "gaussian";
		case StructureType::Cubic: return "cubic";
		case StructureType::Matern: return "matern";
		case StructureType::Stable: return "stable";
		case StructureType::Map: return "map";
	}
	return "?";
}

inline bool structureFromName(const std::string& name, StructureType& type) {
	const std::string lower = toLower(name);
	static const StructureType all[] = { StructureType::Nugget, StructureType::Spherical, StructureType::Exponential, StructureType::Gaussian,
	                                     StructureType::Cubic, StructureType::Matern, StructureType::Stable, StructureType::Map };
	for (StructureType candidate : all) {
		if (lower == structureName(candidate)) { type = candidate; return true; }
	}
	return false;
}

// Structure names, parameter keywords, function names and "h" (compared case-insensitively).
inline bool isReservedWord(const std::string& name) {
	StructureType type;
	if (structureFromName(name, type)) return true;
	const std::string lower = toLower(name);
	static const char* const keywords[] = { "range", "scale", "angle", "nu", "alpha", "h" };
	for (const char* keyword : keywords) if (lower == keyword) return true;
	for (const std::string& function : expressionFunctionNames()) if (lower == function) return true;
	return false;
}

// A valid variable name: an identifier that is not a reserved word.
inline bool isValidVariableName(const std::string& name) {
	if (name.empty() || !(std::isalpha(static_cast<unsigned char>(name[0])) || name[0] == '_')) return false;
	for (char c : name) if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) return false;
	return !isReservedWord(name);
}

// ============================================================ parsed model

// One structure as written, with its parameters still symbolic.
struct StructureSpec {
	StructureType type = StructureType::Nugget;
	Expression sill;                // partial sill (coefficient for map)
	bool useScale = false;          // scale= was given instead of range=
	std::vector<Expression> range;  // empty (nugget, map), 1 value (isotropic) or one per axis
	std::vector<Expression> angle;  // empty (no rotation), 1 or 3 values
	Expression shape;               // nu (matern) or alpha (stable), empty otherwise
	unsigned mapName = 0;           // map(name): index in ModelSpec::names
};

// A parsed model. names lists every identifier the model refers to, in order
// of first appearance; isMapName[i] tells whether names[i] is a map(name)
// table (otherwise it is a scalar variable used in expressions).
class ModelSpec {
public:
	std::vector<StructureSpec> structures;
	std::vector<std::string> names;
	std::vector<bool> isMapName;

	// Parses a model; throws Error with a message naming the problem and its column.
	static ModelSpec parse(const std::string& text);

	// The canonical text: parse(canonical()) gives the same model and the same
	// canonical text. Rules: terms joined by " + "; each term is
	// "<sill>*<structure>" with the sill always written (1 when absent) and
	// parenthesised when it is a sum or difference; "nugget" without
	// parentheses; "map(name)"; other structures as
	// "type(range=..., angle=..., nu=...)" (scale= in place of range= when
	// given; keyword order range|scale, angle, nu|alpha; angle only when given);
	// a one-element vector is written as a scalar, longer ones as "[a, b]";
	// expressions as described in expression.hpp (constants folded).
	std::string canonical() const {
		std::string out;
		for (size_t s = 0; s < structures.size(); ++s) {
			const StructureSpec& st = structures[s];
			if (s > 0) out += " + ";
			const Expression::Op top = st.sill.nodes[size_t(st.sill.root)].op;
			const std::string sill = st.sill.toString(names);
			out += (top == Expression::Op::Add || top == Expression::Op::Subtract) ? "(" + sill + ")" : sill;
			out += "*";
			out += structureName(st.type);
			if (st.type == StructureType::Nugget) continue;
			if (st.type == StructureType::Map) { out += "(" + names[st.mapName] + ")"; continue; }
			out += st.useScale ? "(scale=" : "(range=";
			out += vectorText(st.range);
			if (!st.angle.empty()) out += ", angle=" + vectorText(st.angle);
			if (st.type == StructureType::Matern) out += ", nu=" + st.shape.toString(names);
			if (st.type == StructureType::Stable) out += ", alpha=" + st.shape.toString(names);
			out += ")";
		}
		return out;
	}

	// Scalar variables (names used in expressions) and map tables, in order of first appearance.
	std::vector<std::string> variableNames() const {
		std::vector<std::string> result;
		for (size_t i = 0; i < names.size(); ++i) if (!isMapName[i]) result.push_back(names[i]);
		return result;
	}
	std::vector<std::string> mapNames() const {
		std::vector<std::string> result;
		for (size_t i = 0; i < names.size(); ++i) if (isMapName[i]) result.push_back(names[i]);
		return result;
	}

	// Checks the parameter values of one structure; returns an empty string
	// when they are valid, else the problem ("range must be > 0, got -1").
	// Used at parse time for constants and at binding time for every node.
	static std::string checkValues(StructureType type, double sill, const double* range, size_t rangeCount,
	                               const double* angle, size_t angleCount, double shape) {
		auto number = [](double v) { return formatNumber(v); };
		if (!std::isfinite(sill)) return "the partial sill is not finite (" + number(sill) + ")";
		if (sill < 0) return "the partial sill must be >= 0, got " + number(sill);
		for (size_t k = 0; k < rangeCount; ++k) {
			if (!std::isfinite(range[k]) || !(range[k] > 0)) return "range/scale must be > 0, got " + number(range[k]);
		}
		for (size_t k = 0; k < angleCount; ++k) {
			if (!std::isfinite(angle[k])) return "angle is not finite (" + number(angle[k]) + ")";
		}
		if (type == StructureType::Matern && !(shape > 0 && shape <= 100)) return "nu must be in (0, 100], got " + number(shape);
		if (type == StructureType::Stable && !(shape > 0 && shape <= 2)) return "alpha must be in (0, 2], got " + number(shape);
		return std::string();
	}

private:
	std::string vectorText(const std::vector<Expression>& values) const {
		if (values.size() == 1) return values[0].toString(names);
		std::string out = "[";
		for (size_t i = 0; i < values.size(); ++i) {
			if (i > 0) out += ", ";
			out += values[i].toString(names);
		}
		return out + "]";
	}
};

inline ModelSpec ModelSpec::parse(const std::string& text) {
	const std::string context = "covariance model \"" + text + "\": ";
	std::vector<Token> tokens;
	try {
		tokens = tokenize(text);
	} catch (const Error& error) {
		throw Error(context + error.what());
	}
	const size_t end = tokens.size() - 1; // index of the End token
	if (end == 0) throw Error(context + "empty model");
	auto fail = [&](size_t tokenIndex, const std::string& message) -> Error {
		return Error(context + message + " (column " + std::to_string(tokens[tokenIndex].position + 1) + ")");
	};
	auto isSymbol = [&](size_t i, const char* symbol) { return i < end && tokens[i].kind == Token::Symbol && tokens[i].text == symbol; };

	// Matching bracket of each opening '(' or '['.
	std::vector<size_t> match(tokens.size(), 0);
	{
		std::vector<size_t> stack;
		for (size_t i = 0; i < end; ++i) {
			if (isSymbol(i, "(") || isSymbol(i, "[")) stack.push_back(i);
			else if (isSymbol(i, ")") || isSymbol(i, "]")) {
				if (stack.empty()) throw fail(i, "unbalanced '" + tokens[i].text + "'");
				const size_t open = stack.back();
				if ((tokens[open].text == "(") != (tokens[i].text == ")")) throw fail(i, "'" + tokens[open].text + "' closed by '" + tokens[i].text + "'");
				match[open] = i;
				stack.pop_back();
			}
		}
		if (!stack.empty()) throw fail(stack.back(), "unclosed '" + tokens[stack.back()].text + "'");
	}

	ModelSpec spec;
	std::vector<bool> usedInExpression;
	auto parseExpression = [&](size_t from, size_t to, const std::string& what) {
		if (from >= to) throw fail(std::min(from, end), "missing value for " + what);
		ExpressionParser parser(tokens, from, to, spec.names, &isReservedWord);
		Expression expression;
		try {
			expression = parser.parse();
		} catch (const Error& error) {
			throw Error(context + error.what());
		}
		if (parser.position() != to) throw fail(parser.position(), "unexpected '" + tokens[parser.position()].text + "' in " + what);
		return expression;
	};
	// Splits [from, to) at the top-level commas.
	auto splitCommas = [&](size_t from, size_t to) {
		std::vector<std::pair<size_t, size_t> > parts;
		size_t start = from;
		for (size_t i = from; i < to; ++i) {
			if (isSymbol(i, "(") || isSymbol(i, "[")) { i = match[i]; continue; }
			if (isSymbol(i, ",")) { parts.push_back({ start, i }); start = i + 1; }
		}
		parts.push_back({ start, to });
		return parts;
	};

	// ---- terms
	std::vector<std::pair<size_t, size_t> > terms;
	{
		size_t start = 0;
		for (size_t i = 0; i < end; ++i) {
			if (isSymbol(i, "(") || isSymbol(i, "[")) { i = match[i]; continue; }
			if (isSymbol(i, "+")) { terms.push_back({ start, i }); start = i + 1; }
		}
		terms.push_back({ start, end });
	}

	for (size_t termIndex = 0; termIndex < terms.size(); ++termIndex) {
		const size_t from = terms[termIndex].first, to = terms[termIndex].second;
		if (from >= to) throw fail(std::min(from, end), "empty term (a '+' without a structure)");
		// The structure: a structure name at the top level of the term.
		size_t structureToken = to;
		for (size_t i = from; i < to; ++i) {
			if (isSymbol(i, "(") || isSymbol(i, "[")) {
				for (size_t j = i + 1; j < match[i]; ++j) {
					StructureType inner;
					if (tokens[j].kind == Token::Name && structureFromName(tokens[j].text, inner) && !(j + 1 < end && isSymbol(j + 1, "=")))
						throw fail(j, "structure '" + tokens[j].text + "' inside brackets; write sill*" + toLower(tokens[j].text) + "(...)");
				}
				i = match[i];
				continue;
			}
			StructureType type;
			if (tokens[i].kind == Token::Name && structureFromName(tokens[i].text, type)) {
				if (structureToken != to) throw fail(i, "two structures in one term; separate them with '+'");
				structureToken = i;
			}
		}
		if (structureToken == to) {
			for (size_t i = from; i + 1 < to; ++i) {
				if (isSymbol(i, "(") || isSymbol(i, "[")) { i = match[i]; continue; }
				if (tokens[i].kind != Token::Name || !isSymbol(i + 1, "(")) continue;
				const std::vector<std::string>& functions = expressionFunctionNames();
				if (std::find(functions.begin(), functions.end(), toLower(tokens[i].text)) == functions.end())
					throw fail(i, "unknown structure '" + tokens[i].text + "' (nugget, spherical, exponential, gaussian, cubic, matern, stable or map)");
			}
			std::string termText;
			for (size_t i = from; i < to; ++i) termText += tokens[i].text;
			throw fail(from, "term '" + termText + "' has no structure (nugget, spherical, exponential, gaussian, cubic, matern, stable or map)");
		}
		StructureSpec structure;
		structureFromName(tokens[structureToken].text, structure.type);
		const std::string name = structureName(structure.type);

		// ---- partial sill
		if (structureToken == from) {
			structure.sill = Expression::constant(1.0);
		} else {
			if (!isSymbol(structureToken - 1, "*"))
				throw fail(structureToken, "expected '*' between the partial sill and '" + tokens[structureToken].text + "'");
			const size_t sillEnd = structureToken - 1;
			// A top-level binary '+'/'-' would silently change the meaning ("1-0.5*nugget").
			for (size_t i = from; i < sillEnd; ++i) {
				if (isSymbol(i, "(") || isSymbol(i, "[")) { i = match[i]; continue; }
				if (isSymbol(i, "-") && i > from) {
					const Token& previous = tokens[i - 1];
					const bool binary = previous.kind == Token::Number || previous.kind == Token::Name ||
					                    (previous.kind == Token::Symbol && (previous.text == ")" || previous.text == "]"));
					if (binary) throw fail(i, "a structure cannot be subtracted; parenthesise a difference in a partial sill: (a - b)*" + name);
				}
			}
			structure.sill = parseExpression(from, sillEnd, "the partial sill of " + name);
		}

		// ---- arguments
		size_t argumentsFrom = structureToken + 1, argumentsTo = structureToken + 1;
		if (structureToken + 1 < to) {
			const bool parenthesis = isSymbol(structureToken + 1, "(");
			if (!parenthesis || match[structureToken + 1] + 1 != to) {
				const size_t after = parenthesis ? match[structureToken + 1] + 1 : structureToken + 1;
				if (isSymbol(after, "*") || isSymbol(after, "/"))
					throw fail(after, "the partial sill must come before the structure: sill*" + name + "(...)");
				throw fail(after, "unexpected '" + tokens[after].text + "' after " + name);
			}
			argumentsFrom = structureToken + 2;
			argumentsTo = to - 1;
		}
		const bool hasArguments = argumentsFrom < argumentsTo;

		if (structure.type == StructureType::Nugget) {
			if (hasArguments) throw fail(argumentsFrom, "nugget takes no parameter");
		} else if (structure.type == StructureType::Map) {
			if (!hasArguments || argumentsTo != argumentsFrom + 1 || tokens[argumentsFrom].kind != Token::Name)
				throw fail(hasArguments ? argumentsFrom : structureToken, "map takes one table name: map(name)");
			const std::string& table = tokens[argumentsFrom].text;
			if (!isValidVariableName(table)) throw fail(argumentsFrom, "'" + table + "' is a reserved word and cannot name a map");
			size_t index = 0;
			while (index < spec.names.size() && spec.names[index] != table) ++index;
			if (index == spec.names.size()) spec.names.push_back(table);
			structure.mapName = unsigned(index);
		} else {
			if (!hasArguments) throw fail(structureToken, name + " needs range= or scale=");
			bool seenRange = false, seenAngle = false, seenShape = false;
			for (const std::pair<size_t, size_t>& argument : splitCommas(argumentsFrom, argumentsTo)) {
				const size_t a = argument.first, b = argument.second;
				if (a >= b) throw fail(std::min(a, end), "empty argument in " + name + "(...)");
				if (tokens[a].kind != Token::Name || !isSymbol(a + 1, "="))
					throw fail(a, "expected keyword=value in " + name + "(...)");
				const std::string keyword = toLower(tokens[a].text);
				const bool isRange = keyword == "range" || keyword == "scale";
				const bool isShape = (keyword == "nu" && structure.type == StructureType::Matern) ||
				                     (keyword == "alpha" && structure.type == StructureType::Stable);
				if (!isRange && !isShape && keyword != "angle") {
					std::string allowed = "range, scale, angle";
					if (structure.type == StructureType::Matern) allowed += ", nu";
					if (structure.type == StructureType::Stable) allowed += ", alpha";
					throw fail(a, "unknown keyword '" + tokens[a].text + "' for " + name + " (allowed: " + allowed + ")");
				}
				bool& seen = isRange ? seenRange : isShape ? seenShape : seenAngle;
				if (seen) throw fail(a, isRange ? "range/scale given twice in " + name : "'" + keyword + "' given twice in " + name);
				seen = true;
				std::vector<Expression> values;
				const size_t valueFrom = a + 2;
				if (isSymbol(valueFrom, "[") && match[valueFrom] + 1 == b) {
					if (isShape) throw fail(valueFrom, keyword + " takes a single value");
					if (match[valueFrom] == valueFrom + 1) throw fail(valueFrom, "empty vector for " + keyword);
					for (const std::pair<size_t, size_t>& element : splitCommas(valueFrom + 1, match[valueFrom]))
						values.push_back(parseExpression(element.first, element.second, keyword + " of " + name));
				} else {
					values.push_back(parseExpression(valueFrom, b, keyword + " of " + name));
				}
				if (isRange) {
					if (values.size() > kMaxDims) throw fail(valueFrom, keyword + " has more than " + std::to_string(kMaxDims) + " values");
					structure.useScale = keyword == "scale";
					structure.range = values;
				} else if (isShape) {
					structure.shape = values[0];
				} else {
					if (values.size() != 1 && values.size() != 3) throw fail(valueFrom, "angle takes one value (2D) or three values (3D)");
					structure.angle = values;
				}
			}
			if (!seenRange) throw fail(structureToken, name + " needs range= or scale=");
			if (structure.type == StructureType::Matern && !seenShape) throw fail(structureToken, "matern needs nu=");
			if (structure.type == StructureType::Stable && !seenShape) throw fail(structureToken, "stable needs alpha=");
		}

		// ---- constant checks (variable parameters are checked when bound)
		const double sill = structure.sill.isConstant() ? structure.sill.constantValue() : 0.0;
		std::vector<double> range, angle;
		for (const Expression& e : structure.range) if (e.isConstant()) range.push_back(e.constantValue());
		for (const Expression& e : structure.angle) if (e.isConstant()) angle.push_back(e.constantValue());
		double shape = 1.0;
		if (!structure.shape.empty() && structure.shape.isConstant()) shape = structure.shape.constantValue();
		const std::string problem = checkValues(structure.type, sill, range.data(), range.size(), angle.data(), angle.size(), shape);
		if (!problem.empty()) throw fail(structureToken, name + ": " + problem);

		spec.structures.push_back(structure);
	}

	// ---- names: a name is either a map table or a scalar variable
	usedInExpression.assign(spec.names.size(), false);
	for (const StructureSpec& structure : spec.structures) {
		structure.sill.collectVariables(usedInExpression);
		for (const Expression& e : structure.range) e.collectVariables(usedInExpression);
		for (const Expression& e : structure.angle) e.collectVariables(usedInExpression);
		structure.shape.collectVariables(usedInExpression);
	}
	spec.isMapName.assign(spec.names.size(), false);
	for (const StructureSpec& structure : spec.structures) {
		if (structure.type != StructureType::Map) continue;
		if (usedInExpression[structure.mapName])
			throw Error(context + "'" + spec.names[structure.mapName] + "' is used both as a map table and as a parameter variable");
		spec.isMapName[structure.mapName] = true;
	}
	return spec;
}

// ============================================================ bindings

// Values of the model variables. A name is bound to
//   * a constant;
//   * a field: one value per node (spatially varying parameter);
//   * a table: a lag table for map(name), dims in DataImage axis order (axis 0
//     fastest), lag 0 at dims/2.
// Fields and tables are shared (not copied) by the models built from them.
class Bindings {
public:
	enum class Kind { None, Constant, Field, Table };

	void setConstant(const std::string& name, double value) {
		if (!std::isfinite(value)) throw Error("variable '" + name + "': the value is not finite");
		add(name).kind = Kind::Constant;
		_entries.back().constant = value;
	}
	void setField(const std::string& name, std::vector<float> values) {
		Entry& entry = add(name);
		entry.kind = Kind::Field;
		entry.values = std::make_shared<const std::vector<float> >(std::move(values));
	}
	void setTable(const std::string& name, std::vector<float> values, std::vector<unsigned> dims) {
		size_t count = 1;
		for (unsigned d : dims) count *= d;
		if (dims.empty() || count != values.size()) throw Error("map table '" + name + "': the values do not match the dimensions");
		Entry& entry = add(name);
		entry.kind = Kind::Table;
		entry.values = std::make_shared<const std::vector<float> >(std::move(values));
		entry.dims = std::move(dims);
	}

	Kind kind(const std::string& name) const { const Entry* e = find(name); return e ? e->kind : Kind::None; }
	double constant(const std::string& name) const { return find(name)->constant; }
	std::shared_ptr<const std::vector<float> > values(const std::string& name) const { return find(name)->values; }
	const std::vector<unsigned>& dims(const std::string& name) const { return find(name)->dims; }
	std::vector<std::string> names() const {
		std::vector<std::string> result;
		for (const Entry& e : _entries) result.push_back(e.name);
		return result;
	}

private:
	struct Entry {
		std::string name;
		Kind kind = Kind::None;
		double constant = 0.0;
		std::shared_ptr<const std::vector<float> > values;
		std::vector<unsigned> dims;
	};
	std::vector<Entry> _entries;

	Entry& add(const std::string& name) {
		if (!isValidVariableName(name)) {
			if (isReservedWord(name)) throw Error("'" + name + "' is a reserved word (structure, keyword or function name) and cannot be a variable");
			throw Error("'" + name + "' is not a valid variable name (letters, digits and '_', not starting with a digit)");
		}
		if (find(name)) throw Error("variable '" + name + "' is bound twice");
		_entries.push_back(Entry());
		_entries.back().name = name;
		return _entries.back();
	}
	const Entry* find(const std::string& name) const {
		for (const Entry& e : _entries) if (e.name == name) return &e;
		return nullptr;
	}
};

// ============================================================ resolved model

// R_ij products for ndim = 2 or 3 (see the header comment); R is row-major
// ndim x ndim, column k = principal direction k. angleCount: 1 or 3 (3D only).
inline void rotationMatrix(unsigned ndim, const double* anglesDegrees, unsigned angleCount, double* R) {
	for (unsigned i = 0; i < ndim; ++i)
		for (unsigned j = 0; j < ndim; ++j) R[i * ndim + j] = i == j ? 1.0 : 0.0;
	if (ndim < 2 || angleCount == 0) return;
	const double toRadians = 3.14159265358979323846 / 180.0;
	auto applyRight = [&](unsigned i, unsigned j, double degrees) { // R = R * R_ij(theta)
		if (degrees == 0.0) return;
		const double c = std::cos(degrees * toRadians), s = std::sin(degrees * toRadians);
		for (unsigned row = 0; row < ndim; ++row) {
			const double ri = R[row * ndim + i], rj = R[row * ndim + j];
			R[row * ndim + i] = c * ri + s * rj;
			R[row * ndim + j] = -s * ri + c * rj;
		}
	};
	applyRight(0, 1, anglesDegrees[0]);
	if (ndim >= 3 && angleCount >= 3) {
		applyRight(0, 2, anglesDegrees[1]);
		applyRight(1, 2, anglesDegrees[2]);
	}
}

// One structure with all parameters resolved to numbers.
struct ResolvedStructure {
	StructureType type = StructureType::Nugget;
	unsigned ndim = 0;
	double coefficient = 0.0;       // the partial-sill expression as written
	double sill = 0.0;              // covariance at lag 0 (coefficient, or coefficient * M[centre] for map)
	double shape = 0.0;             // nu (matern), alpha (stable), 0 otherwise
	double range[kMaxDims] = {};    // practical range along principal direction k (0 for nugget and map)
	double angle[3] = {};           // degrees, as given (3D with one angle: [a, 0, 0])
	double rotation[kMaxDims * kMaxDims] = {};  // R, row-major ndim x ndim; column k = direction of range[k]
	double transform[kMaxDims * kMaxDims] = {}; // A = diag(1/range) R^T, row-major: lag -> unit-range lag
	double scaleToRange = 1.0;      // practical range / scale; h/scale = |A h| * scaleToRange
	bool isotropic = true;          // all ranges equal: |A h| = |h| / range[0]
	double maternLogNorm = 0.0;     // (1 - nu) ln 2 - lgamma(nu)
	// map(name) only: the table (owned by the CovarianceModel), its dims, strides and centre.
	const float* table = nullptr;
	unsigned tableDims[kMaxDims] = {};
	size_t tableStride[kMaxDims] = {};
	int tableCentre[kMaxDims] = {};

	// |A h|: the lag in practical ranges (not meaningful for nugget and map).
	double unitDistance(const double* lag) const {
		if (isotropic) {
			double sum = 0.0;
			for (unsigned d = 0; d < ndim; ++d) sum += lag[d] * lag[d];
			return std::sqrt(sum) / range[0];
		}
		double sum = 0.0;
		for (unsigned k = 0; k < ndim; ++k) {
			double u = 0.0;
			for (unsigned d = 0; d < ndim; ++d) u += transform[k * ndim + d] * lag[d];
			sum += u * u;
		}
		return std::sqrt(sum);
	}

	// Correlation at unit distance t (spherical ... matern); 1 at t = 0.
	double correlation(double t) const {
		switch (type) {
			case StructureType::Spherical: return t < 1.0 ? 1.0 - t * (1.5 - 0.5 * t * t) : 0.0;
			case StructureType::Cubic: {
				if (t >= 1.0) return 0.0;
				const double t2 = t * t;
				return 1.0 - t2 * (7.0 - t * (8.75 - t2 * (3.5 - 0.75 * t2)));
			}
			case StructureType::Exponential: return std::exp(-3.0 * t);
			case StructureType::Gaussian: return std::exp(-3.0 * t * t);
			case StructureType::Stable: return t > 0 ? std::exp(-3.0 * std::pow(t, shape)) : 1.0;
			case StructureType::Matern: return maternCorrelation(shape, t * scaleToRange, maternLogNorm);
			case StructureType::Nugget: return t == 0.0 ? 1.0 : 0.0;
			case StructureType::Map: return 0.0;
		}
		return 0.0;
	}

	double covariance(const double* lag) const {
		switch (type) {
			case StructureType::Nugget:
				for (unsigned d = 0; d < ndim; ++d) if (lag[d] != 0.0) return 0.0;
				return sill;
			case StructureType::Map: {
				size_t index = 0;
				for (unsigned d = 0; d < ndim; ++d) {
					const long position = long(tableCentre[d]) + std::lround(lag[d]);
					if (position < 0 || position >= long(tableDims[d])) return 0.0;
					index += size_t(position) * tableStride[d];
				}
				return coefficient * double(table[index]);
			}
			default:
				return sill * correlation(unitDistance(lag));
		}
	}

	// Half-extent along each array axis beyond which the structure is
	// negligible: the practical-range ellipsoid's extent (sqrt(sum_k (R_dk
	// range_k)^2)), the table half-size for map, 0 for nugget.
	void practicalRange(double* perAxis) const {
		for (unsigned d = 0; d < ndim; ++d) {
			if (type == StructureType::Nugget) { perAxis[d] = 0.0; continue; }
			if (type == StructureType::Map) { perAxis[d] = double(tableCentre[d]); continue; }
			double sum = 0.0;
			for (unsigned k = 0; k < ndim; ++k) {
				const double v = rotation[d * ndim + k] * range[k];
				sum += v * v;
			}
			perAxis[d] = std::sqrt(sum);
		}
	}

	bool sameParameters(const ResolvedStructure& o) const {
		if (type != o.type || ndim != o.ndim || coefficient != o.coefficient || shape != o.shape || table != o.table) return false;
		for (unsigned k = 0; k < ndim; ++k) if (range[k] != o.range[k]) return false;
		for (unsigned k = 0; k < 3; ++k) if (angle[k] != o.angle[k]) return false;
		return true;
	}
};

// A model resolved at one node: a plain stationary covariance.
class ResolvedModel {
public:
	unsigned ndim = 0;
	unsigned modelIndex = 0; // which model of the CovarianceModelSet it comes from
	std::vector<ResolvedStructure> structures;

	// C(0), the total sill (nugget included).
	double sill() const { return _sill; }

	// C(h) for one lag of ndim components.
	double covariance(const double* lag) const {
		double c = 0.0;
		for (const ResolvedStructure& s : structures) c += s.covariance(lag);
		return c;
	}
	// gamma(h) = C(0) - C(h).
	double variogram(const double* lag) const { return _sill - covariance(lag); }

	// out[i] = C(lags + i*ndim) for i < count.
	void covariances(const double* lags, size_t count, double* out) const {
		for (size_t i = 0; i < count; ++i) out[i] = 0.0;
		for (const ResolvedStructure& s : structures)
			for (size_t i = 0; i < count; ++i) out[i] += s.covariance(lags + i * ndim);
	}

	// Kriging system of `count` neighbours with lags x_i - x_0 (row-major count x ndim):
	// matrix[i*ld + j] = C(lag_i - lag_j) (full symmetric matrix written) and,
	// when rhs is not null, rhs[i] = C(lag_i).
	void covarianceMatrix(const double* lags, size_t count, double* matrix, size_t ld, double* rhs) const {
		double difference[kMaxDims];
		for (size_t i = 0; i < count; ++i) {
			const double* li = lags + i * ndim;
			if (rhs) rhs[i] = covariance(li);
			for (size_t j = 0; j <= i; ++j) {
				const double* lj = lags + j * ndim;
				for (unsigned d = 0; d < ndim; ++d) difference[d] = li[d] - lj[d];
				const double c = covariance(difference);
				matrix[i * ld + j] = c;
				matrix[j * ld + i] = c;
			}
		}
	}

	// Largest practical range of the structures along each array axis.
	void practicalRange(double* perAxis) const {
		double local[kMaxDims];
		for (unsigned d = 0; d < ndim; ++d) perAxis[d] = 0.0;
		for (const ResolvedStructure& s : structures) {
			s.practicalRange(local);
			for (unsigned d = 0; d < ndim; ++d) perAxis[d] = std::max(perAxis[d], local[d]);
		}
	}
	std::vector<double> practicalRange() const {
		std::vector<double> result(ndim);
		practicalRange(result.data());
		return result;
	}

	// 64-bit hash of the resolved parameters (types, coefficients, ranges,
	// angles, shapes, map tables). Equal parameters give equal keys; confirm a
	// key match with sameParameters() before sharing a solved system.
	uint64_t key() const { return _key; }

	bool sameParameters(const ResolvedModel& o) const {
		if (ndim != o.ndim || structures.size() != o.structures.size()) return false;
		for (size_t s = 0; s < structures.size(); ++s) if (!structures[s].sameParameters(o.structures[s])) return false;
		return true;
	}

	// Recomputes sill() and key() after `structures` was edited by hand.
	void update() {
		_sill = 0.0;
		uint64_t hash = 1469598103934665603ull;
		auto mix = [&](const void* data, size_t size) {
			const unsigned char* bytes = static_cast<const unsigned char*>(data);
			for (size_t i = 0; i < size; ++i) { hash ^= bytes[i]; hash *= 1099511628211ull; }
		};
		auto mixDouble = [&](double v) { if (v == 0.0) v = 0.0; mix(&v, sizeof(v)); };
		mix(&ndim, sizeof(ndim));
		for (const ResolvedStructure& s : structures) {
			_sill += s.sill;
			const int type = int(s.type);
			mix(&type, sizeof(type));
			mixDouble(s.coefficient);
			mixDouble(s.shape);
			for (unsigned k = 0; k < ndim; ++k) mixDouble(s.range[k]);
			for (unsigned k = 0; k < 3; ++k) mixDouble(s.angle[k]);
			const float* table = s.table;
			mix(&table, sizeof(table));
		}
		_key = hash;
	}

private:
	friend class CovarianceModel;
	double _sill = 0.0;
	uint64_t _key = 0;
	std::vector<double> _variables; // scratch: variable values at the node
};

// ============================================================ bound model

// A parsed model bound to its variables for ndim-dimensional lags. All
// parameters are validated at construction (at every node for fields), so
// resolve() never fails afterwards.
class CovarianceModel {
public:
	CovarianceModel() = default;

	// Throws Error when a variable is unbound or of the wrong kind, a vector
	// length does not match ndim, a map table has the wrong dimensionality,
	// fields have different node counts, or a value is invalid at some node.
	// Unused bindings are not an error here (see buildCovarianceModelSet).
	CovarianceModel(const ModelSpec& spec, const Bindings& bindings, unsigned ndim) : _spec(spec), _ndim(ndim) {
		const std::string context = "covariance model \"" + spec.canonical() + "\": ";
		if (ndim < 1 || ndim > kMaxDims) throw Error(context + "lags must have 1 to " + std::to_string(kMaxDims) + " components, got " + std::to_string(ndim));
		const size_t nameCount = spec.names.size();
		_constants.assign(nameCount, 0.0);
		_fields.assign(nameCount, nullptr);
		_tables.assign(nameCount, Table());
		for (size_t i = 0; i < nameCount; ++i) {
			const std::string& name = spec.names[i];
			const Bindings::Kind kind = bindings.kind(name);
			if (kind == Bindings::Kind::None) throw Error(context + "variable '" + name + "' is not bound (use -cvar " + name + " value)");
			if (spec.isMapName[i]) {
				if (kind != Bindings::Kind::Table) throw Error(context + "map(" + name + ") needs a covariance map image, not a number");
				Table& table = _tables[i];
				table.values = bindings.values(name);
				table.dims = bindings.dims(name);
				if (table.dims.size() != ndim)
					throw Error(context + "map '" + name + "' has " + std::to_string(table.dims.size()) + " dimensions, the lags have " + std::to_string(ndim));
				size_t centre = 0, stride = 1;
				for (unsigned d = 0; d < ndim; ++d) { centre += size_t(table.dims[d] / 2) * stride; stride *= table.dims[d]; }
				for (float v : *table.values) if (!std::isfinite(v)) throw Error(context + "map '" + name + "' contains non-finite values");
				if (!((*table.values)[centre] > 0)) throw Error(context + "map '" + name + "' must be positive at its centre (lag 0)");
			} else if (kind == Bindings::Kind::Constant) {
				_constants[i] = bindings.constant(name);
			} else if (kind == Bindings::Kind::Field) {
				_fields[i] = bindings.values(name);
				if (_fieldIndices.empty()) _nodeCount = _fields[i]->size();
				else if (_fields[i]->size() != _nodeCount) throw Error(context + "variable '" + name + "' has " + std::to_string(_fields[i]->size()) + " values, other fields have " + std::to_string(_nodeCount));
				_fieldIndices.push_back(unsigned(i));
			} else {
				throw Error(context + "'" + name + "' is bound to a map table but used as a parameter variable");
			}
		}
		for (size_t s = 0; s < spec.structures.size(); ++s) {
			const StructureSpec& st = spec.structures[s];
			const std::string where = context + "structure " + std::to_string(s + 1) + " (" + structureName(st.type) + "): ";
			if (st.range.size() > 1 && st.range.size() != ndim)
				throw Error(where + "range/scale has " + std::to_string(st.range.size()) + " values for " + std::to_string(ndim) + "D lags");
			if (!st.angle.empty()) {
				if (ndim < 2) throw Error(where + "angle needs 2D or 3D lags");
				if (ndim == 2 && st.angle.size() != 1) throw Error(where + "2D lags take one angle");
				if (ndim > 3) throw Error(where + "angle is only supported for 2D and 3D lags");
			}
		}
		// Validate every node (or once when nothing varies).
		std::vector<double> values = _constants;
		_stationary = _fieldIndices.empty();
		const size_t nodes = _stationary ? 1 : _nodeCount;
		Raw raw;
		for (size_t node = 0; node < nodes; ++node) {
			for (unsigned i : _fieldIndices) values[i] = double((*_fields[i])[node]);
			for (size_t s = 0; s < spec.structures.size(); ++s) {
				evaluateRaw(spec.structures[s], values.data(), raw);
				const std::string problem = ModelSpec::checkValues(spec.structures[s].type, raw.coefficient, raw.range, raw.rangeCount, raw.angle, raw.angleCount, raw.shape);
				if (!problem.empty()) {
					std::string message = context + "structure " + std::to_string(s + 1) + " (" + structureName(spec.structures[s].type) + "): " + problem;
					if (!_stationary) message += " at node " + std::to_string(node);
					throw Error(message);
				}
			}
		}
		// Matérn practical range is costly: compute it once when nu does not vary.
		_maternRange.assign(spec.structures.size(), 0.0);
		for (size_t s = 0; s < spec.structures.size(); ++s) {
			const StructureSpec& st = spec.structures[s];
			if (st.type == StructureType::Matern && !varies(st.shape)) _maternRange[s] = maternPracticalRange(st.shape.evaluate(_constants.data()));
		}
		if (_stationary) resolve(0, _stationaryModel);
	}

	// Sets ResolvedModel::modelIndex of the stationary model (CovarianceModelSet does it).
	void setModelIndex(unsigned index) { _stationaryModel.modelIndex = index; }

	unsigned ndim() const { return _ndim; }
	const ModelSpec& spec() const { return _spec; }
	std::string canonical() const { return _spec.canonical(); }

	// True when no parameter varies with the node: stationaryModel() is then valid.
	bool isStationary() const { return _stationary; }
	const ResolvedModel& stationaryModel() const { return _stationaryModel; }

	// Number of nodes the fields cover (0 when stationary).
	size_t nodeCount() const { return _nodeCount; }

	// Resolves the parameters at `node` into `out` (no allocation once `out`
	// has been used for this model). For a stationary model the node is ignored.
	void resolve(size_t node, ResolvedModel& out) const {
		out._variables.assign(_constants.begin(), _constants.end());
		if (!_stationary) for (unsigned i : _fieldIndices) out._variables[i] = double((*_fields[i])[node]);
		out.ndim = _ndim;
		out.structures.resize(_spec.structures.size());
		Raw raw;
		for (size_t s = 0; s < _spec.structures.size(); ++s) {
			evaluateRaw(_spec.structures[s], out._variables.data(), raw);
			finish(s, raw, out.structures[s]);
		}
		out.update();
	}

	// The stationary model, or `scratch` resolved at `node`.
	const ResolvedModel& at(size_t node, ResolvedModel& scratch) const {
		if (_stationary) return _stationaryModel;
		resolve(node, scratch);
		return scratch;
	}

private:
	struct Table {
		std::shared_ptr<const std::vector<float> > values;
		std::vector<unsigned> dims;
	};
	struct Raw {
		double coefficient = 0.0, shape = 0.0;
		double range[kMaxDims] = {};
		double angle[3] = {};
		unsigned rangeCount = 0, angleCount = 0;
	};

	ModelSpec _spec;
	unsigned _ndim = 0;
	bool _stationary = true;
	size_t _nodeCount = 0;
	std::vector<double> _constants;
	std::vector<std::shared_ptr<const std::vector<float> > > _fields;
	std::vector<Table> _tables;
	std::vector<unsigned> _fieldIndices;
	std::vector<double> _maternRange;
	ResolvedModel _stationaryModel;

	bool varies(const Expression& e) const {
		std::vector<bool> used(_spec.names.size(), false);
		e.collectVariables(used);
		for (unsigned i : _fieldIndices) if (used[i]) return true;
		return false;
	}

	void evaluateRaw(const StructureSpec& st, const double* values, Raw& raw) const {
		raw.coefficient = st.sill.evaluate(values);
		raw.rangeCount = unsigned(st.range.size());
		for (unsigned k = 0; k < raw.rangeCount; ++k) raw.range[k] = st.range[k].evaluate(values);
		raw.angleCount = unsigned(st.angle.size());
		for (unsigned k = 0; k < raw.angleCount; ++k) raw.angle[k] = st.angle[k].evaluate(values);
		raw.shape = st.shape.empty() ? 0.0 : st.shape.evaluate(values);
	}

	void finish(size_t index, const Raw& raw, ResolvedStructure& rs) const {
		const StructureSpec& st = _spec.structures[index];
		rs = ResolvedStructure();
		rs.type = st.type;
		rs.ndim = _ndim;
		rs.coefficient = raw.coefficient;
		rs.sill = raw.coefficient;
		if (st.type == StructureType::Nugget) return;
		if (st.type == StructureType::Map) {
			const Table& table = _tables[st.mapName];
			rs.table = table.values->data();
			size_t stride = 1, centre = 0;
			for (unsigned d = 0; d < _ndim; ++d) {
				rs.tableDims[d] = table.dims[d];
				rs.tableStride[d] = stride;
				rs.tableCentre[d] = int(table.dims[d] / 2);
				centre += size_t(rs.tableCentre[d]) * stride;
				stride *= table.dims[d];
			}
			rs.sill = raw.coefficient * double(rs.table[centre]);
			return;
		}
		rs.shape = raw.shape;
		switch (st.type) {
			case StructureType::Exponential: rs.scaleToRange = 3.0; break;
			case StructureType::Gaussian: rs.scaleToRange = std::sqrt(3.0); break;
			case StructureType::Stable: rs.scaleToRange = std::pow(3.0, 1.0 / raw.shape); break;
			case StructureType::Matern:
				rs.scaleToRange = _maternRange[index] > 0 ? _maternRange[index] : maternPracticalRange(raw.shape);
				rs.maternLogNorm = maternLogNorm(raw.shape);
				break;
			default: rs.scaleToRange = 1.0; break;
		}
		for (unsigned k = 0; k < _ndim; ++k) {
			const double value = raw.range[raw.rangeCount == 1 ? 0 : k];
			rs.range[k] = st.useScale ? value * rs.scaleToRange : value;
		}
		unsigned angleCount = 0;
		if (raw.angleCount > 0 && _ndim >= 2) {
			rs.angle[0] = raw.angle[0];
			if (raw.angleCount == 3) { rs.angle[1] = raw.angle[1]; rs.angle[2] = raw.angle[2]; }
			angleCount = _ndim >= 3 ? 3 : 1;
		}
		rotationMatrix(_ndim, rs.angle, angleCount, rs.rotation);
		rs.isotropic = true;
		for (unsigned k = 1; k < _ndim; ++k) if (rs.range[k] != rs.range[0]) rs.isotropic = false;
		for (unsigned k = 0; k < _ndim; ++k)
			for (unsigned d = 0; d < _ndim; ++d) rs.transform[k * _ndim + d] = rs.rotation[d * _ndim + k] / rs.range[k];
	}
};

// ============================================================ model set

// The models of one run (one per -cov) and the optional per-node model index
// (-ii). Model 0 is used everywhere when there is no index.
class CovarianceModelSet {
public:
	std::vector<CovarianceModel> models;
	std::vector<unsigned> modelIndex; // per node, or empty

	size_t size() const { return models.size(); }
	unsigned ndim() const { return models.empty() ? 0 : models[0].ndim(); }
	unsigned modelAt(size_t node) const { return modelIndex.empty() ? 0u : modelIndex[node]; }

	// Number of nodes covered by the index or the fields (0 when nothing varies).
	size_t nodeCount() const {
		if (!modelIndex.empty()) return modelIndex.size();
		for (const CovarianceModel& m : models) if (m.nodeCount() > 0) return m.nodeCount();
		return 0;
	}

	// True when every node uses the same stationary model.
	bool isStationary() const { return _stationaryModel >= 0; }
	const ResolvedModel& stationaryModel() const { return models[size_t(_stationaryModel)].stationaryModel(); }

	// The model at `node`: a stationary model by reference, or `scratch` filled in.
	const ResolvedModel& at(size_t node, ResolvedModel& scratch) const {
		if (_stationaryModel >= 0) return models[size_t(_stationaryModel)].stationaryModel();
		const unsigned m = modelAt(node);
		const ResolvedModel& result = models[m].at(node, scratch);
		if (&result == &scratch) scratch.modelIndex = m;
		return result;
	}
	void resolve(size_t node, ResolvedModel& out) const {
		const unsigned m = modelAt(node);
		models[m].resolve(node, out);
		out.modelIndex = m;
	}

	// Largest total sill and largest practical range per axis over all nodes
	// (for buffer sizing; visits every node when a model is not stationary).
	double maxSill() const {
		double result = 0.0;
		visit([&](const ResolvedModel& m) { result = std::max(result, m.sill()); });
		return result;
	}
	std::vector<double> maxPracticalRange() const {
		std::vector<double> result(ndim(), 0.0), local(ndim());
		visit([&](const ResolvedModel& m) {
			m.practicalRange(local.data());
			for (size_t d = 0; d < result.size(); ++d) result[d] = std::max(result[d], local[d]);
		});
		return result;
	}

	std::vector<std::string> canonical() const {
		std::vector<std::string> result;
		for (const CovarianceModel& m : models) result.push_back(m.canonical());
		return result;
	}

	// Called by buildCovarianceModelSet; call it again after editing members.
	void finalize() {
		std::vector<bool> used(models.size(), false);
		if (modelIndex.empty()) { if (!used.empty()) used[0] = true; }
		else for (unsigned m : modelIndex) used[m] = true;
		_used = used;
		for (size_t m = 0; m < models.size(); ++m) models[m].setModelIndex(unsigned(m));
		int single = -1, count = 0;
		for (size_t m = 0; m < used.size(); ++m) if (used[m]) { single = int(m); ++count; }
		_stationaryModel = (count == 1 && models[size_t(single)].isStationary()) ? single : -1;
	}

private:
	int _stationaryModel = -1;
	std::vector<bool> _used;

	template <class F>
	void visit(F f) const {
		ResolvedModel scratch;
		for (size_t m = 0; m < models.size(); ++m) {
			if (m < _used.size() && !_used[m]) continue;
			if (models[m].isStationary()) { f(models[m].stationaryModel()); continue; }
			for (size_t node = 0; node < models[m].nodeCount(); ++node) {
				if (modelAt(node) != m) continue;
				models[m].resolve(node, scratch);
				f(scratch);
			}
		}
	}
};

// Parses and binds every model text, checks that every binding is used by at
// least one model and that the index (when given) refers to existing models
// and covers as many nodes as the fields. Throws Error.
inline CovarianceModelSet buildCovarianceModelSet(const std::vector<std::string>& texts, const Bindings& bindings, unsigned ndim,
                                                  std::vector<unsigned> modelIndex = std::vector<unsigned>()) {
	if (texts.empty()) throw Error("no covariance model given (-cov)");
	CovarianceModelSet set;
	std::vector<std::string> used;
	for (const std::string& text : texts) {
		const ModelSpec spec = ModelSpec::parse(text);
		for (const std::string& name : spec.names) used.push_back(name);
		set.models.push_back(CovarianceModel(spec, bindings, ndim));
	}
	for (const std::string& name : bindings.names()) {
		if (std::find(used.begin(), used.end(), name) == used.end())
			throw Error("variable '" + name + "' is bound (-cvar) but no covariance model uses it");
	}
	for (size_t i = 0; i < modelIndex.size(); ++i) {
		if (modelIndex[i] >= set.models.size())
			throw Error("model index " + std::to_string(modelIndex[i]) + " at node " + std::to_string(i) + " but only " +
			            std::to_string(set.models.size()) + " model(s) given (-ii is 0-based)");
	}
	for (const CovarianceModel& m : set.models) {
		if (!modelIndex.empty() && m.nodeCount() > 0 && m.nodeCount() != modelIndex.size())
			throw Error("the model index covers " + std::to_string(modelIndex.size()) + " nodes but the variables cover " + std::to_string(m.nodeCount()));
		if (m.nodeCount() > 0 && m.nodeCount() != set.nodeCount())
			throw Error("the spatially varying variables of different models cover different numbers of nodes");
	}
	set.modelIndex = std::move(modelIndex);
	set.finalize();
	return set;
}

} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_COVARIANCE_MODEL_HPP
