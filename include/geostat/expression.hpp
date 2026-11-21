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

// Parameter expressions of the covariance-model grammar (covarianceModel.hpp).
//
// An expression is built from
//   * numbers: 12, 0.5, .5, 1e-3, 2.5E+4 (no sign: "-2" is unary minus applied to 2);
//   * variable names: [A-Za-z_][A-Za-z0-9_]*, case-sensitive, bound at run time
//     (-cvar name value); no sigil, so the shell never expands them;
//   * + - * / with the usual precedence and left associativity, parentheses,
//     unary minus;
//   * the functions min(a, b), max(a, b), sqrt(x), exp(x), log(x) (natural), abs(x).
// Reserved words (structure names, parameter keywords, function names and the
// lag "h") cannot be variable names; reserved words are matched
// case-insensitively. The lag is never available to an expression, so every
// parameter is a constant (or a node-wise constant) and every model stays
// positive-definite.
//
// Constant folding: every sub-expression without a variable is evaluated when
// it is parsed, so "(1-0.1)*nugget" is the same model as "0.9*nugget". A
// constant sub-expression that is not finite (1/0, log(0), sqrt(-1)) is a
// parse error.
//
// Canonical printing (Expression::toString):
//   * numbers: "0" for zero; integers with |v| < 1e15 without decimals
//     ("100000", "-3"); otherwise the shortest C printf "%.{p}g" (p = 1..17)
//     that reads back to the same double ("0.1", "1e-05", "1.5e+16");
//     Python reproduces it with: next(s for p in range(1, 18)
//     if float(s := '%.*g' % (p, v)) == v);
//   * "a + b", "a - b" with one space around the operator; "a*b", "a/b"
//     without spaces; "-a" for unary minus; "min(a, b)", "sqrt(a)";
//   * parentheses only where needed to read back the same tree: a left
//     operand is parenthesised when it binds less tightly than the operator,
//     a right operand when it binds less tightly or equally ("a - (b + c)",
//     "a/(b*c)", "(a + b)*c"); a negative number or a negation inside a
//     product is not parenthesised ("-2*a", "a*-b").

#ifndef G2S_GEOSTAT_EXPRESSION_HPP
#define G2S_GEOSTAT_EXPRESSION_HPP

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace g2s {
namespace geostat {

// Every parse, binding or validation problem of the geostat foundation is
// reported with this exception; what() is a complete, user-facing message.
class Error : public std::runtime_error {
public:
	explicit Error(const std::string& message) : std::runtime_error(message) {}
};

inline std::string toLower(std::string text) {
	for (char& c : text) c = char(std::tolower(static_cast<unsigned char>(c)));
	return text;
}

// Shortest round-trip text of a double (see the canonical printing rules above).
inline std::string formatNumber(double value) {
	if (value == 0.0) return "0";
	if (std::isnan(value)) return "nan";
	if (std::isinf(value)) return value > 0 ? "inf" : "-inf";
	char buffer[64];
	if (std::fabs(value) < 1e15 && value == std::floor(value)) {
		snprintf(buffer, sizeof(buffer), "%.0f", value);
		return buffer;
	}
	for (int precision = 1; precision <= 17; ++precision) {
		snprintf(buffer, sizeof(buffer), "%.*g", precision, value);
		if (std::strtod(buffer, nullptr) == value) break;
	}
	return buffer;
}

struct Token {
	enum Kind { Number, Name, Symbol, End };
	Kind kind = End;
	std::string text;
	double value = 0.0;
	size_t position = 0; // 0-based character offset in the source text
};

// Splits a model or expression text into tokens; throws Error on an invalid character or number.
inline std::vector<Token> tokenize(const std::string& text) {
	std::vector<Token> tokens;
	size_t i = 0;
	while (i < text.size()) {
		const char c = text[i];
		if (std::isspace(static_cast<unsigned char>(c))) { ++i; continue; }
		Token token;
		token.position = i;
		if (std::isdigit(static_cast<unsigned char>(c)) || (c == '.' && i + 1 < text.size() && std::isdigit(static_cast<unsigned char>(text[i + 1])))) {
			size_t j = i;
			while (j < text.size() && std::isdigit(static_cast<unsigned char>(text[j]))) ++j;
			if (j < text.size() && text[j] == '.') { ++j; while (j < text.size() && std::isdigit(static_cast<unsigned char>(text[j]))) ++j; }
			if (j < text.size() && (text[j] == 'e' || text[j] == 'E')) {
				size_t k = j + 1;
				if (k < text.size() && (text[k] == '+' || text[k] == '-')) ++k;
				if (k >= text.size() || !std::isdigit(static_cast<unsigned char>(text[k])))
					throw Error("invalid number '" + text.substr(i, k - i) + "' at column " + std::to_string(i + 1));
				while (k < text.size() && std::isdigit(static_cast<unsigned char>(text[k]))) ++k;
				j = k;
			}
			if (j < text.size() && (std::isalpha(static_cast<unsigned char>(text[j])) || text[j] == '_' || text[j] == '.'))
				throw Error("invalid number '" + text.substr(i, j - i + 1) + "' at column " + std::to_string(i + 1));
			token.kind = Token::Number;
			token.text = text.substr(i, j - i);
			token.value = std::strtod(token.text.c_str(), nullptr);
			if (!std::isfinite(token.value)) throw Error("number '" + token.text + "' is out of range");
			i = j;
		} else if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
			size_t j = i;
			while (j < text.size() && (std::isalnum(static_cast<unsigned char>(text[j])) || text[j] == '_')) ++j;
			token.kind = Token::Name;
			token.text = text.substr(i, j - i);
			i = j;
		} else if (std::string("+-*/()[],=").find(c) != std::string::npos) {
			token.kind = Token::Symbol;
			token.text = std::string(1, c);
			++i;
		} else {
			throw Error(std::string("unexpected character '") + c + "' at column " + std::to_string(i + 1));
		}
		tokens.push_back(token);
	}
	Token end;
	end.kind = Token::End;
	end.position = text.size();
	tokens.push_back(end);
	return tokens;
}

inline const std::vector<std::string>& expressionFunctionNames() {
	static const std::vector<std::string> names = { "min", "max", "sqrt", "exp", "log", "abs" };
	return names;
}

class Expression {
public:
	enum class Op : unsigned char { Constant, Variable, Negate, Add, Subtract, Multiply, Divide, Min, Max, Sqrt, Exp, Log, Abs };
	struct Node {
		Op op = Op::Constant;
		double value = 0.0;   // Constant
		unsigned variable = 0; // Variable: index in the owner's name table
		int left = -1, right = -1;
	};

	std::vector<Node> nodes;
	int root = -1;

	static Expression constant(double value) {
		Expression e;
		Node node;
		node.value = value;
		e.root = e.push(node);
		return e;
	}

	bool empty() const { return root < 0; }
	bool isConstant() const { return root >= 0 && nodes[size_t(root)].op == Op::Constant; }
	double constantValue() const { return nodes[size_t(root)].value; }

	// variables[i] is the value of the name with index i.
	double evaluate(const double* variables) const { return evaluateNode(root, variables); }

	// Marks the names this expression refers to.
	void collectVariables(std::vector<bool>& used) const {
		if (root >= 0) collect(root, used);
	}

	std::string toString(const std::vector<std::string>& names) const {
		if (root < 0) return std::string();
		return print(root, names);
	}

	// Adds a node, folding it into a constant when its operands are constant.
	int push(const Node& node) {
		Node n = node;
		const bool unary = n.op == Op::Negate || n.op == Op::Sqrt || n.op == Op::Exp || n.op == Op::Log || n.op == Op::Abs;
		const bool binary = n.op == Op::Add || n.op == Op::Subtract || n.op == Op::Multiply || n.op == Op::Divide || n.op == Op::Min || n.op == Op::Max;
		if ((unary && nodes[size_t(n.left)].op == Op::Constant) ||
		    (binary && nodes[size_t(n.left)].op == Op::Constant && nodes[size_t(n.right)].op == Op::Constant)) {
			const double a = nodes[size_t(n.left)].value;
			const double b = binary ? nodes[size_t(n.right)].value : 0.0;
			const double value = apply(n.op, a, b);
			if (!std::isfinite(value)) throw Error("constant sub-expression " + printOperation(n) + " is not finite");
			// The operands were the last nodes pushed: drop them.
			const int lowest = binary ? std::min(n.left, n.right) : n.left;
			if (size_t(lowest) + (binary ? 2 : 1) == nodes.size()) nodes.resize(size_t(lowest));
			Node folded;
			folded.value = value;
			n = folded;
		}
		nodes.push_back(n);
		return int(nodes.size() - 1);
	}

	static double apply(Op op, double a, double b) {
		switch (op) {
			case Op::Negate: return -a;
			case Op::Add: return a + b;
			case Op::Subtract: return a - b;
			case Op::Multiply: return a * b;
			case Op::Divide: return a / b;
			case Op::Min: return std::min(a, b);
			case Op::Max: return std::max(a, b);
			case Op::Sqrt: return std::sqrt(a);
			case Op::Exp: return std::exp(a);
			case Op::Log: return std::log(a);
			case Op::Abs: return std::fabs(a);
			default: return a;
		}
	}

private:
	double evaluateNode(int index, const double* variables) const {
		const Node& n = nodes[size_t(index)];
		switch (n.op) {
			case Op::Constant: return n.value;
			case Op::Variable: return variables[n.variable];
			case Op::Negate: case Op::Sqrt: case Op::Exp: case Op::Log: case Op::Abs:
				return apply(n.op, evaluateNode(n.left, variables), 0.0);
			default:
				return apply(n.op, evaluateNode(n.left, variables), evaluateNode(n.right, variables));
		}
	}

	void collect(int index, std::vector<bool>& used) const {
		const Node& n = nodes[size_t(index)];
		if (n.op == Op::Variable) { if (n.variable < used.size()) used[n.variable] = true; return; }
		if (n.left >= 0) collect(n.left, used);
		if (n.right >= 0) collect(n.right, used);
	}

	// 1: + -, 2: * /, 3: unary minus and negative numbers, 4: atoms and calls.
	int precedence(int index) const {
		const Node& n = nodes[size_t(index)];
		switch (n.op) {
			case Op::Add: case Op::Subtract: return 1;
			case Op::Multiply: case Op::Divide: return 2;
			case Op::Negate: return 3;
			case Op::Constant: return n.value < 0 ? 3 : 4;
			default: return 4;
		}
	}

	std::string printOperation(const Node& n) const {
		std::vector<std::string> noNames;
		auto operand = [&](int i) { return nodes[size_t(i)].op == Op::Constant ? formatNumber(nodes[size_t(i)].value) : std::string("..."); };
		switch (n.op) {
			case Op::Negate: return "-" + operand(n.left);
			case Op::Add: return operand(n.left) + " + " + operand(n.right);
			case Op::Subtract: return operand(n.left) + " - " + operand(n.right);
			case Op::Multiply: return operand(n.left) + "*" + operand(n.right);
			case Op::Divide: return operand(n.left) + "/" + operand(n.right);
			case Op::Min: return "min(" + operand(n.left) + ", " + operand(n.right) + ")";
			case Op::Max: return "max(" + operand(n.left) + ", " + operand(n.right) + ")";
			case Op::Sqrt: return "sqrt(" + operand(n.left) + ")";
			case Op::Exp: return "exp(" + operand(n.left) + ")";
			case Op::Log: return "log(" + operand(n.left) + ")";
			case Op::Abs: return "abs(" + operand(n.left) + ")";
			default: return "?";
		}
	}

	std::string print(int index, const std::vector<std::string>& names) const {
		const Node& n = nodes[size_t(index)];
		auto wrap = [&](int child, bool parenthesise) {
			const std::string text = print(child, names);
			return parenthesise ? "(" + text + ")" : text;
		};
		switch (n.op) {
			case Op::Constant: return formatNumber(n.value);
			case Op::Variable: return n.variable < names.size() ? names[n.variable] : std::string("?");
			case Op::Negate: return "-" + wrap(n.left, precedence(n.left) < 3);
			case Op::Add: case Op::Subtract: case Op::Multiply: case Op::Divide: {
				const int own = precedence(index);
				const char* symbol = n.op == Op::Add ? " + " : n.op == Op::Subtract ? " - " : n.op == Op::Multiply ? "*" : "/";
				return wrap(n.left, precedence(n.left) < own) + symbol + wrap(n.right, precedence(n.right) <= own);
			}
			case Op::Min: return "min(" + print(n.left, names) + ", " + print(n.right, names) + ")";
			case Op::Max: return "max(" + print(n.left, names) + ", " + print(n.right, names) + ")";
			case Op::Sqrt: return "sqrt(" + print(n.left, names) + ")";
			case Op::Exp: return "exp(" + print(n.left, names) + ")";
			case Op::Log: return "log(" + print(n.left, names) + ")";
			case Op::Abs: return "abs(" + print(n.left, names) + ")";
		}
		return std::string();
	}
};

// Recursive-descent parser over tokens[position, end). Variable names are
// added to (or found in) `names`; `isReserved` rejects reserved words.
class ExpressionParser {
public:
	ExpressionParser(const std::vector<Token>& tokens, size_t position, size_t end, std::vector<std::string>& names,
	                 bool (*isReserved)(const std::string&))
		: _tokens(tokens), _position(position), _end(end), _names(names), _isReserved(isReserved) {}

	// Parses one full expression and returns it; stops before ',' ']' ')' or the end.
	Expression parse() {
		_expression = Expression();
		_expression.root = parseSum();
		return _expression;
	}

	size_t position() const { return _position; }

private:
	const std::vector<Token>& _tokens;
	size_t _position, _end;
	std::vector<std::string>& _names;
	bool (*_isReserved)(const std::string&);
	Expression _expression;

	const Token& peek() const {
		static const Token end;
		return _position < _end ? _tokens[_position] : (_end < _tokens.size() ? _tokens[_end] : end);
	}
	bool isSymbol(const char* symbol) const { return _position < _end && peek().kind == Token::Symbol && peek().text == symbol; }

	[[noreturn]] void fail(const std::string& message) const {
		const size_t column = peek().position + 1;
		throw Error(message + " (column " + std::to_string(column) + ")");
	}

	std::string describe() const {
		if (_position >= _end || peek().kind == Token::End) return "end of the expression";
		return "'" + peek().text + "'";
	}

	void expect(const char* symbol) {
		if (!isSymbol(symbol)) fail(std::string("expected '") + symbol + "' but found " + describe());
		++_position;
	}

	int node(Expression::Op op, int left, int right = -1) {
		Expression::Node n;
		n.op = op;
		n.left = left;
		n.right = right;
		return _expression.push(n);
	}

	int parseSum() {
		int left = parseProduct();
		while (isSymbol("+") || isSymbol("-")) {
			const Expression::Op op = peek().text == "+" ? Expression::Op::Add : Expression::Op::Subtract;
			++_position;
			left = node(op, left, parseProduct());
		}
		return left;
	}

	int parseProduct() {
		int left = parseUnary();
		while (isSymbol("*") || isSymbol("/")) {
			const Expression::Op op = peek().text == "*" ? Expression::Op::Multiply : Expression::Op::Divide;
			++_position;
			left = node(op, left, parseUnary());
		}
		return left;
	}

	int parseUnary() {
		if (isSymbol("-")) { ++_position; return node(Expression::Op::Negate, parseUnary()); }
		return parsePrimary();
	}

	int parsePrimary() {
		if (_position >= _end) fail("expected a value but found " + describe());
		const Token& token = peek();
		if (token.kind == Token::Number) {
			++_position;
			Expression::Node n;
			n.value = token.value;
			return _expression.push(n);
		}
		if (isSymbol("(")) {
			++_position;
			const int inner = parseSum();
			expect(")");
			return inner;
		}
		if (token.kind == Token::Name) {
			const std::string lower = toLower(token.text);
			static const Expression::Op functionOps[] = { Expression::Op::Min, Expression::Op::Max, Expression::Op::Sqrt,
			                                              Expression::Op::Exp, Expression::Op::Log, Expression::Op::Abs };
			const std::vector<std::string>& functions = expressionFunctionNames();
			for (size_t f = 0; f < functions.size(); ++f) {
				if (lower != functions[f]) continue;
				++_position;
				expect("(");
				const int first = parseSum();
				int result;
				if (f < 2) {
					expect(",");
					const int second = parseSum();
					result = node(functionOps[f], first, second);
				} else {
					result = node(functionOps[f], first);
				}
				expect(")");
				return result;
			}
			if (lower == "h") fail("the lag 'h' cannot be used in a parameter: parameters are constant or vary with the node, never with the lag");
			if (_isReserved && _isReserved(token.text)) fail("'" + token.text + "' is a reserved word and cannot be used as a variable");
			++_position;
			size_t index = 0;
			while (index < _names.size() && _names[index] != token.text) ++index;
			if (index == _names.size()) _names.push_back(token.text);
			Expression::Node n;
			n.op = Expression::Op::Variable;
			n.variable = unsigned(index);
			return _expression.push(n);
		}
		fail("expected a value but found " + describe());
	}
};

} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_EXPRESSION_HPP
