// Tests of the geostat foundation (include/geostat): covariance grammar and
// models, kriging solver, neighbour search, random numbers, the -cov/-cvar/-ii
// helper and (when built) the covEval executable.
//
//   make -C test/geostat            # build and run
//   COVEVAL=/path/to/covEval make -C test/geostat   # other covEval binary

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>

#include "geostat/covarianceArguments.hpp"
#include "geostat/covarianceModel.hpp"
#include "geostat/kriging.hpp"
#include "geostat/neighborSearch.hpp"
#include "geostat/random.hpp"
#include "geostat/specialFunctions.hpp"
#include "../geostat_common/testData.hpp"

using namespace g2s::geostat;
using geostat_test::testId;
using geostat_test::testIdText;

namespace {

// Image names unique to this run (removed at exit with the outputs of testId()).
std::string N(const std::string& stem) { return geostat_test::testName(stem); }

int failures = 0;
int checks = 0;
std::string currentSection;

void report(bool ok, const std::string& what, const char* file, int line) {
	++checks;
	if (ok) return;
	++failures;
	fprintf(stderr, "FAIL [%s] %s:%d: %s\n", currentSection.c_str(), file, line, what.c_str());
}

#define CHECK(condition) report(bool(condition), #condition, __FILE__, __LINE__)
#define CHECK_MSG(condition, message) report(bool(condition), std::string(#condition) + " -- " + (message), __FILE__, __LINE__)
#define CHECK_NEAR(a, b, tolerance) do { const double va_ = (a), vb_ = (b); \
	report(std::fabs(va_ - vb_) <= (tolerance), std::string(#a " ~ " #b ": ") + formatNumber(va_) + " vs " + formatNumber(vb_), __FILE__, __LINE__); } while (0)

// Runs f and checks it throws Error with `fragment` in the message.
template <class F>
void checkThrows(F f, const std::string& fragment, const char* file, int line) {
	try {
		f();
		report(false, "expected an error containing '" + fragment + "', got none", file, line);
	} catch (const Error& e) {
		const std::string message = e.what();
		report(message.find(fragment) != std::string::npos, "error '" + message + "' should contain '" + fragment + "'", file, line);
	}
}
#define CHECK_THROWS(expression, fragment) checkThrows([&]() { expression; }, fragment, __FILE__, __LINE__)

void section(const std::string& name) { currentSection = name; printf("-- %s\n", name.c_str()); }

std::string trim(const std::string& s) {
	size_t a = 0, b = s.size();
	while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
	while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
	return s.substr(a, b - a);
}

ResolvedModel stationary(const std::string& text, unsigned ndim, const Bindings& bindings = Bindings()) {
	return CovarianceModel(ModelSpec::parse(text), bindings, ndim).stationaryModel();
}

double cov1(const ResolvedModel& m, double h) { return m.covariance(&h); }
double cov2(const ResolvedModel& m, double x, double y) { const double lag[2] = { x, y }; return m.covariance(lag); }

// ============================================================ grammar

void testGrammar(const std::string& casesFile) {
	section("grammar: cov_cases.txt");
	std::ifstream input(casesFile);
	CHECK_MSG(input.good(), "cannot read " + casesFile);
	std::string line;
	int cases = 0;
	while (std::getline(input, line)) {
		if (trim(line).empty() || trim(line)[0] == '#') continue;
		const size_t separator = line.find(" ==> ");
		if (separator == std::string::npos) { CHECK_MSG(false, "malformed line: " + line); continue; }
		const std::string text = trim(line.substr(0, separator));
		const std::string expected = trim(line.substr(separator + 5));
		++cases;
		if (expected.compare(0, 7, "!error ") == 0) {
			const std::string fragment = expected.substr(7);
			try {
				const ModelSpec spec = ModelSpec::parse(text);
				CHECK_MSG(false, "'" + text + "' should fail with '" + fragment + "' but gave " + spec.canonical());
			} catch (const Error& e) {
				const std::string message = e.what();
				CHECK_MSG(message.find(fragment) != std::string::npos, "'" + text + "': message '" + message + "' lacks '" + fragment + "'");
			}
			continue;
		}
		try {
			const std::string canonical = ModelSpec::parse(text).canonical();
			CHECK_MSG(canonical == expected, "'" + text + "' gave '" + canonical + "', expected '" + expected + "'");
			const std::string again = ModelSpec::parse(canonical).canonical();
			CHECK_MSG(again == canonical, "canonical '" + canonical + "' reparsed as '" + again + "'");
		} catch (const Error& e) {
			CHECK_MSG(false, "'" + text + "' failed: " + e.what());
		}
	}
	CHECK(cases > 50);

	section("grammar: details");
	{
		const ModelSpec spec = ModelSpec::parse("nug*nugget + (1-nug)*spherical(range=[L, L/2], angle=theta) + 0.2*map(C)");
		CHECK(spec.structures.size() == 3);
		CHECK(spec.variableNames() == std::vector<std::string>({ "nug", "L", "theta" }));
		CHECK(spec.mapNames() == std::vector<std::string>({ "C" }));
		CHECK(spec.structures[1].range.size() == 2 && spec.structures[1].angle.size() == 1);
	}
	CHECK(formatNumber(0.1) == "0.1");
	CHECK(formatNumber(-3) == "-3");
	CHECK(formatNumber(1e-5) == "1e-05");
	CHECK(formatNumber(123456.7) == "123456.7");
	CHECK(formatNumber(1e15) == "1e+15");
	CHECK(formatNumber(-0.0) == "0");
	CHECK(isReservedWord("Matern") && isReservedWord("h") && isReservedWord("sqrt") && !isReservedWord("L"));
	// Values survive the canonical round trip exactly.
	{
		std::mt19937_64 random(7);
		std::uniform_real_distribution<double> u(1e-3, 1e3);
		for (int i = 0; i < 200; ++i) {
			const double v = u(random);
			const std::string text = formatNumber(v) + "*nugget";
			const ResolvedModel m = stationary(ModelSpec::parse(text).canonical(), 1);
			CHECK(m.sill() == v);
		}
	}
}

// ============================================================ models

double maternExact(double nu, double u) {
	// Half-integer closed forms through K_{n+1/2} recurrences.
	const double k05 = std::sqrt(3.14159265358979323846 / (2 * u)) * std::exp(-u);
	double km = k05, k = k05 * (1 + 1 / u), v = 1.5;
	if (nu == 0.5) k = k05;
	while (v < nu - 1e-12) { const double next = km + 2 * v / u * k; km = k; k = next; v += 1; }
	return std::pow(2.0, 1 - nu) / std::tgamma(nu) * std::pow(u, nu) * k;
}

void testModels() {
	section("models: closed forms and practical range");
	const double e3 = std::exp(-3.0);
	{
		const ResolvedModel m = stationary("2*exponential(range=30)", 1);
		CHECK_NEAR(m.sill(), 2.0, 0);
		CHECK_NEAR(cov1(m, 0), 2.0, 1e-15);
		CHECK_NEAR(cov1(m, 30) / 2.0, e3, 1e-15);
		CHECK(cov1(m, 30) / 2.0 > 0.049 && cov1(m, 30) / 2.0 < 0.051);
		CHECK_NEAR(cov1(m, 7), 2.0 * std::exp(-3 * 7 / 30.0), 1e-15);
		CHECK_NEAR(m.variogram(std::vector<double>{ 7 }.data()), 2.0 - 2.0 * std::exp(-0.7), 1e-14);
	}
	{
		const ResolvedModel m = stationary("exponential(scale=10)", 1);
		CHECK_NEAR(cov1(m, 10), std::exp(-1.0), 1e-15);
		CHECK_NEAR(m.structures[0].range[0], 30.0, 1e-12);
		CHECK_NEAR(m.structures[0].scaleToRange, 3.0, 0);
	}
	{
		const ResolvedModel m = stationary("gaussian(range=20)", 1);
		CHECK_NEAR(cov1(m, 20), e3, 1e-15);
		CHECK_NEAR(cov1(m, 5), std::exp(-3.0 * 25 / 400), 1e-15);
		const ResolvedModel s = stationary("gaussian(scale=4)", 1);
		CHECK_NEAR(cov1(s, 6), std::exp(-36.0 / 16), 1e-15);
	}
	{
		const ResolvedModel m = stationary("3*spherical(range=10)", 1);
		for (double h : { 0.0, 1.0, 4.5, 9.99 }) {
			const double t = h / 10;
			CHECK_NEAR(cov1(m, h), 3 * (1 - 1.5 * t + 0.5 * t * t * t), 1e-14);
		}
		CHECK(cov1(m, 10) == 0 && cov1(m, 12) == 0);
	}
	{
		const ResolvedModel m = stationary("cubic(range=10)", 1);
		for (double h : { 0.0, 2.0, 5.0, 9.0 }) {
			const double t = h / 10;
			CHECK_NEAR(cov1(m, h), 1 - 7 * t * t + 35.0 / 4 * std::pow(t, 3) - 3.5 * std::pow(t, 5) + 0.75 * std::pow(t, 7), 1e-14);
		}
		CHECK(cov1(m, 10) == 0);
		CHECK(std::fabs(cov1(m, 9.999)) < 1e-9);
	}
	{
		const ResolvedModel s1 = stationary("stable(range=12, alpha=1)", 1), e = stationary("exponential(range=12)", 1);
		const ResolvedModel s2 = stationary("stable(range=12, alpha=2)", 1), g = stationary("gaussian(range=12)", 1);
		const ResolvedModel s15 = stationary("stable(range=12, alpha=1.5)", 1);
		for (double h : { 0.5, 3.0, 12.0, 20.0 }) {
			CHECK_NEAR(cov1(s1, h), cov1(e, h), 1e-15);
			CHECK_NEAR(cov1(s2, h), cov1(g, h), 1e-15);
		}
		CHECK_NEAR(cov1(s15, 12), e3, 1e-15);
		const ResolvedModel scale = stationary("stable(scale=2, alpha=0.5)", 1);
		CHECK_NEAR(cov1(scale, 8), std::exp(-std::sqrt(4.0)), 1e-14);
	}
	{
		const ResolvedModel n = stationary("0.3*nugget + 0.7*spherical(range=5)", 2);
		CHECK_NEAR(n.sill(), 1.0, 1e-15);
		CHECK_NEAR(cov2(n, 0, 0), 1.0, 1e-15);
		CHECK_NEAR(cov2(n, 1, 0), 0.7 * (1 - 0.3 + 0.5 * 0.008), 1e-14);
		CHECK_NEAR(cov2(n, 0, 1e-9), 0.7 * (1 - 1.5 * 2e-10), 1e-12);
	}

	section("models: Bessel K and Matern");
	for (double nu : { 0.5, 1.5, 2.5, 3.5, 7.5, 10.5 }) {
		for (double x : { 0.01, 0.1, 1.0, 5.0, 30.0, 200.0 }) {
			const double k05 = std::sqrt(3.14159265358979323846 / (2 * x)) * std::exp(-x);
			double km = k05, k = k05 * (1 + 1 / x), v = 1.5;
			if (nu == 0.5) k = k05;
			while (v < nu - 1e-12) { const double next = km + 2 * v / x * k; km = k; k = next; v += 1; }
			if (!(k > 0) || !std::isfinite(k)) continue; // underflow at large x
			CHECK_NEAR(logBesselK(nu, x), std::log(k), 1e-12 * std::max(1.0, std::fabs(std::log(k))));
		}
	}
	for (double nu : { 0.3, 1.7, 7.3, 40.2 }) {
		for (double x : { 0.05, 0.8, 3.0, 25.0 }) {
			// K_{nu+1} = K_{nu-1} + 2 nu / x K_nu
			const double kp = std::exp(logBesselK(nu + 1, x)), km = std::exp(logBesselK(nu - 1, x)), k = std::exp(logBesselK(nu, x));
			if (!std::isfinite(kp) || !(km > 0)) continue;
			CHECK_NEAR((km + 2 * nu / x * k) / kp, 1.0, 1e-12);
		}
	}
	CHECK_NEAR(maternPracticalRange(0.5), 3.0, 0);
	for (double nu : { 0.05, 0.2, 0.7, 1.5, 2.5, 3.2, 10.0, 100.0 }) {
		const double u95 = maternPracticalRange(nu);
		CHECK_NEAR(maternCorrelation(nu, u95), e3, 1e-13);
		const ResolvedModel m = stationary("matern(range=25, nu=" + formatNumber(nu) + ")", 1);
		CHECK_NEAR(cov1(m, 25), e3, 1e-13);
		CHECK_NEAR(cov1(m, 0), 1.0, 0);
		CHECK(cov1(m, 1e-6) <= 1.0 && (nu < 0.5 || cov1(m, 1e-6) > 0.99));
	}
	for (double nu : { 1.5, 2.5, 4.5 }) {
		for (double u : { 0.01, 0.3, 1.0, 4.0, 15.0 }) CHECK_NEAR(maternCorrelation(nu, u), maternExact(nu, u), 1e-13);
		// The generic path (nu slightly off the closed forms) is continuous with them.
		CHECK_NEAR(maternCorrelation(nu + 1e-9, 2.0), maternCorrelation(nu, 2.0), 1e-8);
	}
	CHECK_NEAR(maternPracticalRange(1.5), 4.7490313860127005, 1e-12); // (1 + u) exp(-u) = exp(-3)
	CHECK_NEAR(maternPracticalRange(2.5), 5.924462618951795, 1e-12);
	{
		const ResolvedModel m = stationary("matern(range=30, nu=0.5)", 1), e = stationary("exponential(range=30)", 1);
		for (double h : { 1.0, 10.0, 30.0, 70.0 }) CHECK_NEAR(cov1(m, h), cov1(e, h), 1e-15);
		const ResolvedModel s = stationary("matern(scale=4, nu=1.5)", 1);
		CHECK_NEAR(cov1(s, 6), (1 + 1.5) * std::exp(-1.5), 1e-14);
		const ResolvedModel big = stationary("matern(range=30, nu=100)", 1), g = stationary("gaussian(range=30)", 1);
		for (double h : { 5.0, 15.0, 30.0 }) CHECK_NEAR(cov1(big, h), cov1(g, h), 0.01);
	}

	section("models: anisotropy and rotation");
	{
		const ResolvedModel m = stationary("spherical(range=[40,20], angle=30)", 2);
		const double c = std::cos(30 * M_PI / 180), s = std::sin(30 * M_PI / 180);
		CHECK_NEAR(cov2(m, 40 * c, 40 * s), 0.0, 1e-12);
		CHECK_NEAR(cov2(m, 39.9 * c, 39.9 * s), 1 - 1.5 * 39.9 / 40 + 0.5 * std::pow(39.9 / 40, 3), 1e-12);
		CHECK_NEAR(cov2(m, -20 * s, 20 * c), 0.0, 1e-12);
		CHECK_NEAR(cov2(m, 20 * c, 20 * s), cov2(m, -10 * s, 10 * c), 1e-14);
		CHECK_NEAR(cov2(m, 20 * c, 20 * s), 1 - 0.75 + 0.0625, 1e-12);
		CHECK_NEAR(cov2(m, 10, 3), cov2(m, -10, -3), 1e-15);
		// Counter-clockwise from axis 0 towards axis 1.
		CHECK(cov2(m, 10, 10) > cov2(m, 10, -10));
		const std::vector<double> extent = m.practicalRange();
		CHECK_NEAR(extent[0], std::sqrt(40 * 40 * c * c + 20 * 20 * s * s), 1e-12);
		CHECK_NEAR(extent[1], std::sqrt(40 * 40 * s * s + 20 * 20 * c * c), 1e-12);
		const ResolvedModel r90 = stationary("spherical(range=[40,20], angle=90)", 2);
		CHECK_NEAR(r90.practicalRange()[0], 20, 1e-12);
		CHECK_NEAR(r90.practicalRange()[1], 40, 1e-12);
		// Transform: unit-range lag.
		const ResolvedStructure& st = m.structures[0];
		const double lag[2] = { 40 * c, 40 * s };
		CHECK_NEAR(st.transform[0] * lag[0] + st.transform[1] * lag[1], 1.0, 1e-14);
		CHECK_NEAR(st.transform[2] * lag[0] + st.transform[3] * lag[1], 0.0, 1e-14);
		CHECK(!st.isotropic);
	}
	{
		const double a = 30, b = 20, cAngle = 10;
		const ResolvedModel m = stationary("exponential(range=[50, 20, 10], angle=[30, 20, 10])", 3);
		const ResolvedStructure& st = m.structures[0];
		const double ra = a * M_PI / 180, rb = b * M_PI / 180;
		const double direction[3] = { std::cos(ra) * std::cos(rb), std::sin(ra) * std::cos(rb), std::sin(rb) };
		for (int d = 0; d < 3; ++d) CHECK_NEAR(st.rotation[d * 3 + 0], direction[d], 1e-15);
		const double lag[3] = { 50 * direction[0], 50 * direction[1], 50 * direction[2] };
		CHECK_NEAR(m.covariance(lag), e3, 1e-14);
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j) {
				double dot = 0;
				for (int k = 0; k < 3; ++k) dot += st.rotation[k * 3 + i] * st.rotation[k * 3 + j];
				CHECK_NEAR(dot, i == j ? 1.0 : 0.0, 1e-15);
			}
		// c turns the minor axes about the major one: axis 1 towards axis 2.
		const ResolvedModel roll = stationary("exponential(range=[50, 20, 10], angle=[0, 0, 90])", 3);
		const double y20[3] = { 0, 0, 20 }, z10[3] = { 0, -10, 0 };
		CHECK_NEAR(roll.covariance(y20), e3, 1e-14);
		CHECK_NEAR(roll.covariance(z10), e3, 1e-14);
		(void)cAngle;
		const ResolvedModel one = stationary("exponential(range=[50, 20, 10], angle=40)", 3);
		const ResolvedModel three = stationary("exponential(range=[50, 20, 10], angle=[40, 0, 0])", 3);
		const double probe[3] = { 7, -3, 2 };
		CHECK_NEAR(one.covariance(probe), three.covariance(probe), 0);
	}
	{
		const ResolvedModel iso = stationary("gaussian(range=10, angle=33)", 2);
		CHECK(iso.structures[0].isotropic);
		CHECK_NEAR(cov2(iso, 3, 4), std::exp(-3 * 0.25), 1e-15);
		CHECK_THROWS(stationary("spherical(range=10, angle=10)", 1), "angle needs 2D or 3D");
		CHECK_THROWS(stationary("spherical(range=10, angle=[1,2,3])", 2), "2D lags take one angle");
		CHECK_THROWS(stationary("spherical(range=[1,2,3])", 2), "range/scale has 3 values for 2D lags");
		CHECK_THROWS(stationary("spherical(range=[1,2])", 3), "range/scale has 2 values for 3D lags");
		CHECK_THROWS(stationary("spherical(range=[1,2,3,4,5])", 2), "more than 4 values");
		CHECK_THROWS(stationary("spherical(range=10)", 5), "1 to 4 components");
		const ResolvedModel four = stationary("spherical(range=[1, 2, 3, 4])", 4);
		const double lag4[4] = { 0, 0, 0, 2 };
		CHECK_NEAR(four.covariance(lag4), 1 - 0.75 + 0.0625, 1e-14);
	}

	section("models: evaluation helpers");
	{
		const ResolvedModel m = stationary("0.2*nugget + 0.5*exponential(range=[10, 5], angle=20) + 0.3*matern(range=8, nu=1.2)", 2);
		const double lags[8] = { 0, 0, 1, 2, -3, 4, 10, -1 };
		double values[4];
		m.covariances(lags, 4, values);
		for (int i = 0; i < 4; ++i) CHECK_NEAR(values[i], m.covariance(lags + 2 * i), 1e-15);
		double matrix[16], rhs[4];
		m.covarianceMatrix(lags, 4, matrix, 4, rhs);
		for (int i = 0; i < 4; ++i) {
			CHECK_NEAR(rhs[i], values[i], 1e-15);
			CHECK_NEAR(matrix[i * 4 + i], m.sill(), 1e-15);
			for (int j = 0; j < 4; ++j) {
				const double d[2] = { lags[2 * i] - lags[2 * j], lags[2 * i + 1] - lags[2 * j + 1] };
				CHECK_NEAR(matrix[i * 4 + j], m.covariance(d), 1e-15);
			}
		}
		const ResolvedModel same = stationary("0.2*nugget + 0.5*exponential(range=[10, 5], angle=20) + 0.3*matern(range=8, nu=1.2)", 2);
		const ResolvedModel other = stationary("0.2*nugget + 0.5*exponential(range=[10, 5], angle=21) + 0.3*matern(range=8, nu=1.2)", 2);
		CHECK(m.key() == same.key() && m.sameParameters(same));
		CHECK(m.key() != other.key() && !m.sameParameters(other));
	}
}

// ============================================================ variables

void testVariables() {
	section("variables: constants and fields");
	{
		Bindings b;
		b.setConstant("nug", 0.1);
		b.setConstant("L", 40);
		b.setConstant("theta", 30);
		const ResolvedModel v = stationary("nug*nugget + (1-nug)*spherical(range=[L, L/2], angle=theta)", 2, b);
		const ResolvedModel literal = stationary("0.1*nugget + 0.9*spherical(range=[40, 20], angle=30)", 2);
		for (double x : { 0.0, 3.0, 17.0 })
			for (double y : { 0.0, -5.0, 9.0 }) CHECK_NEAR(cov2(v, x, y), cov2(literal, x, y), 1e-15);
		CHECK(v.sameParameters(literal));
	}
	{
		Bindings b;
		b.setField("L", { 10, 20, 30, 20 });
		b.setConstant("s", 2);
		const CovarianceModelSet set = buildCovarianceModelSet({ "s*exponential(range=L)" }, b, 1);
		CHECK(!set.isStationary());
		CHECK(set.nodeCount() == 4);
		ResolvedModel scratch;
		const ResolvedModel& at2 = set.at(2, scratch);
		CHECK_NEAR(at2.structures[0].range[0], 30, 0);
		CHECK_NEAR(cov1(at2, 30), 2 * std::exp(-3.0), 1e-15);
		ResolvedModel a, c, d;
		set.resolve(1, a);
		set.resolve(3, c);
		set.resolve(0, d);
		CHECK(a.key() == c.key() && a.sameParameters(c));
		CHECK(a.key() != d.key() && !a.sameParameters(d));
		CHECK_NEAR(set.maxPracticalRange()[0], 30, 0);
		CHECK_NEAR(set.maxSill(), 2, 0);
	}
	{
		Bindings b;
		b.setField("L", { 10, -1, 30 });
		CHECK_THROWS(buildCovarianceModelSet({ "exponential(range=L)" }, b, 1), "range/scale must be > 0, got -1 at node 1");
		Bindings n;
		n.setField("L", { 10, std::nanf(""), 30 });
		CHECK_THROWS(buildCovarianceModelSet({ "exponential(range=L)" }, n, 1), "at node 1");
		Bindings s;
		s.setField("w", { 0.5f, 1.5f });
		CHECK_THROWS(buildCovarianceModelSet({ "(1-w)*nugget + w*spherical(range=3)" }, s, 1), "partial sill must be >= 0, got -0.5 at node 1");
		Bindings nu;
		nu.setField("v", { 0.5f, 1.5f, 120.f });
		CHECK_THROWS(buildCovarianceModelSet({ "matern(range=3, nu=v)" }, nu, 1), "nu must be in (0, 100]");
	}
	{
		Bindings b;
		CHECK_THROWS(buildCovarianceModelSet({ "exponential(range=L)" }, b, 1), "variable 'L' is not bound");
		b.setConstant("L", 3);
		b.setConstant("M", 3);
		CHECK_THROWS(buildCovarianceModelSet({ "exponential(range=L)" }, b, 1), "'M' is bound (-cvar) but no covariance model uses it");
		CHECK_THROWS(b.setConstant("L", 4), "bound twice");
		CHECK_THROWS(b.setConstant("range", 4), "reserved word");
		CHECK_THROWS(b.setConstant("2x", 4), "not a valid variable name");
		Bindings c;
		c.setConstant("C", 1);
		CHECK_THROWS(buildCovarianceModelSet({ "map(C)" }, c, 2), "needs a covariance map image");
		Bindings t;
		t.setTable("L", std::vector<float>(9, 1.f), { 3, 3 });
		CHECK_THROWS(buildCovarianceModelSet({ "exponential(range=L)" }, t, 2), "bound to a map table");
		Bindings f;
		f.setField("a", { 1, 2 });
		f.setField("b", { 1, 2, 3 });
		CHECK_THROWS(buildCovarianceModelSet({ "a*exponential(range=b)" }, f, 1), "has 3 values, other fields have 2");
	}

	section("variables: map tables");
	{
		// 5 x 3 table (axis 0 has 5 cells): centre (2, 1).
		std::vector<float> table(15);
		for (int y = 0; y < 3; ++y)
			for (int x = 0; x < 5; ++x) table[size_t(y * 5 + x)] = float(10 - std::abs(x - 2) - 2 * std::abs(y - 1));
		Bindings b;
		b.setTable("C", table, { 5, 3 });
		const CovarianceModelSet set = buildCovarianceModelSet({ "0.5*map(C)" }, b, 2);
		CHECK(set.isStationary());
		const ResolvedModel& m = set.stationaryModel();
		CHECK_NEAR(m.sill(), 5.0, 0);
		CHECK_NEAR(cov2(m, 1, 0), 0.5 * 9, 0);
		CHECK_NEAR(cov2(m, -2, 1), 0.5 * 6, 0);
		CHECK_NEAR(cov2(m, 0.4, -0.6), 0.5 * 8, 0); // rounded to the nearest cell
		CHECK(cov2(m, 3, 0) == 0 && cov2(m, 0, 2) == 0);
		CHECK_NEAR(m.practicalRange()[0], 2, 0);
		CHECK_NEAR(m.practicalRange()[1], 1, 0);
		Bindings bad;
		bad.setTable("C", std::vector<float>(15, 0.f), { 5, 3 });
		CHECK_THROWS(buildCovarianceModelSet({ "map(C)" }, bad, 2), "positive at its centre");
		CHECK_THROWS(buildCovarianceModelSet({ "map(C)" }, b, 3), "has 2 dimensions, the lags have 3");
		// Keys and sameParameters() compare tables by content, not by address.
		Bindings two;
		two.setTable("C", table, { 5, 3 });
		two.setTable("D", table, { 5, 3 });
		std::vector<float> changed = table;
		changed[0] += 1.f;
		two.setTable("E", changed, { 5, 3 });
		two.setTable("F", table, { 3, 5 });
		const CovarianceModelSet maps = buildCovarianceModelSet({ "0.5*map(C)", "0.5*map(D)", "0.5*map(E)", "0.5*map(F)" }, two, 2, { 0, 1, 2, 3 });
		ResolvedModel mc, md, me, mf;
		maps.resolve(0, mc); maps.resolve(1, md); maps.resolve(2, me); maps.resolve(3, mf);
		CHECK(mc.structures[0].table != md.structures[0].table);
		CHECK(mc.key() == md.key() && mc.sameParameters(md));
		CHECK(mc.key() != me.key() && !mc.sameParameters(me));
		CHECK(mc.key() != mf.key() && !mc.sameParameters(mf));
		CHECK(m.key() == mc.key()); // same content in another model set
		CHECK(covarianceModelSetHash(set, 0) == covarianceModelSetHash(buildCovarianceModelSet({ "0.5*map(C)" }, b, 2), 0));
	}

	section("variables: model set and index");
	{
		Bindings b;
		const CovarianceModelSet set = buildCovarianceModelSet({ "exponential(range=10)", "2*spherical(range=5)" }, b, 1, { 0, 1, 1, 0 });
		CHECK(!set.isStationary());
		ResolvedModel scratch;
		CHECK_NEAR(set.at(1, scratch).sill(), 2, 0);
		CHECK(set.at(1, scratch).modelIndex == 1);
		CHECK_NEAR(set.at(3, scratch).sill(), 1, 0);
		CHECK(set.at(3, scratch).modelIndex == 0);
		CHECK_NEAR(set.maxSill(), 2, 0);
		CHECK_NEAR(set.maxPracticalRange()[0], 10, 0);
		const CovarianceModelSet single = buildCovarianceModelSet({ "exponential(range=10)", "2*spherical(range=5)" }, b, 1, { 1, 1, 1 });
		CHECK(single.isStationary());
		CHECK_NEAR(single.stationaryModel().sill(), 2, 0);
		CHECK(single.stationaryModel().modelIndex == 1);
		CHECK_THROWS(buildCovarianceModelSet({ "exponential(range=10)" }, b, 1, { 0, 2 }), "model index 2 at node 1");
		CHECK_THROWS(buildCovarianceModelSet({}, b, 1), "no covariance model");
	}
}

// ============================================================ kriging

// Independent dense solve with partial pivoting (reference).
std::vector<double> gaussSolve(std::vector<double> a, std::vector<double> b) {
	const size_t n = b.size();
	for (size_t c = 0; c < n; ++c) {
		size_t p = c;
		for (size_t r = c + 1; r < n; ++r) if (std::fabs(a[r * n + c]) > std::fabs(a[p * n + c])) p = r;
		for (size_t k = 0; k < n; ++k) std::swap(a[c * n + k], a[p * n + k]);
		std::swap(b[c], b[p]);
		for (size_t r = c + 1; r < n; ++r) {
			const double f = a[r * n + c] / a[c * n + c];
			for (size_t k = c; k < n; ++k) a[r * n + k] -= f * a[c * n + k];
			b[r] -= f * b[c];
		}
	}
	std::vector<double> x(n);
	for (size_t i = n; i-- > 0;) {
		double s = b[i];
		for (size_t k = i + 1; k < n; ++k) s -= a[i * n + k] * x[k];
		x[i] = s / a[i * n + i];
	}
	return x;
}

void testKriging() {
	section("kriging: hand-computed cases");
	KrigingWorkspace ws;
	KrigingSolution sol;
	const ResolvedModel m = stationary("exponential(range=3)", 1); // C(h) = exp(-|h|)
	{
		const double lags[1] = { 1 };
		krige(KrigingType::Simple, m, lags, 1, ws, sol);
		const double c = std::exp(-1.0);
		CHECK(sol.status == KrigingOk);
		CHECK_NEAR(sol.weights[0], c, 1e-15);
		CHECK_NEAR(sol.variance, 1 - c * c, 1e-15);
		CHECK_NEAR(sol.meanWeight, 1 - c, 1e-15);
		krige(KrigingType::Ordinary, m, lags, 1, ws, sol);
		CHECK_NEAR(sol.weights[0], 1, 1e-15);
		CHECK_NEAR(sol.variance, 2 * (1 - c), 1e-15);
		CHECK_NEAR(sol.meanWeight, 0, 0);
	}
	{
		// SK with two neighbours at -1 and +2: K = [1 e^-3; e^-3 1], k0 = [e^-1, e^-2].
		const double lags[2] = { -1, 2 };
		krige(KrigingType::Simple, m, lags, 2, ws, sol);
		const double k12 = std::exp(-3.0), a = std::exp(-1.0), b = std::exp(-2.0);
		const double det = 1 - k12 * k12;
		const double w1 = (a - k12 * b) / det, w2 = (b - k12 * a) / det;
		CHECK_NEAR(sol.weights[0], w1, 1e-15);
		CHECK_NEAR(sol.weights[1], w2, 1e-15);
		CHECK_NEAR(sol.variance, 1 - w1 * a - w2 * b, 1e-15);
		// Symmetric OK: equal weights.
		const double sym[2] = { -2, 2 };
		krige(KrigingType::Ordinary, m, sym, 2, ws, sol);
		CHECK_NEAR(sol.weights[0], 0.5, 1e-15);
		CHECK_NEAR(sol.weights[1], 0.5, 1e-15);
		const double kk = std::exp(-4.0), k0 = std::exp(-2.0);
		CHECK_NEAR(sol.variance, 1 + 0.5 * (1 + kk) - 2 * k0, 1e-14);
	}
	section("kriging: general systems against a direct solve");
	{
		std::mt19937_64 random(3);
		std::uniform_real_distribution<double> u(-15, 15);
		const ResolvedModel g = stationary("0.1*nugget + 0.6*spherical(range=[20, 8], angle=25) + 0.3*gaussian(range=12)", 2);
		for (int trial = 0; trial < 20; ++trial) {
			const size_t k = 3 + size_t(trial);
			std::vector<double> lags(2 * k);
			for (double& v : lags) v = std::round(u(random));
			std::vector<double> big((k + 1) * (k + 1), 1.0), rhs(k + 1, 1.0);
			for (size_t i = 0; i < k; ++i) {
				rhs[i] = g.covariance(&lags[2 * i]);
				for (size_t j = 0; j < k; ++j) {
					const double d[2] = { lags[2 * i] - lags[2 * j], lags[2 * i + 1] - lags[2 * j + 1] };
					big[i * (k + 1) + j] = g.covariance(d);
				}
			}
			big[k * (k + 1) + k] = 0.0;
			bool duplicate = false;
			for (size_t i = 0; i < k; ++i)
				for (size_t j = 0; j < i; ++j) duplicate |= lags[2 * i] == lags[2 * j] && lags[2 * i + 1] == lags[2 * j + 1];
			const unsigned status = krige(KrigingType::Ordinary, g, lags.data(), k, ws, sol);
			double sum = 0;
			for (double w : sol.weights) sum += w;
			CHECK_NEAR(sum, 1.0, 1e-10);
			if (duplicate) { CHECK(status & KrigingJittered); continue; }
			CHECK(status == KrigingOk);
			const std::vector<double> reference = gaussSolve(big, rhs);
			for (size_t i = 0; i < k; ++i) CHECK_NEAR(sol.weights[i], reference[i], 1e-9);
			CHECK_NEAR(sol.lagrange, reference[k], 1e-9);
			double variance = g.sill() - reference[k];
			for (size_t i = 0; i < k; ++i) variance -= reference[i] * rhs[i];
			CHECK_NEAR(sol.variance, std::max(variance, 0.0), 1e-9);
			// SK against the direct solve of K w = k0.
			std::vector<double> small(k * k), k0(k);
			for (size_t i = 0; i < k; ++i) { k0[i] = rhs[i]; for (size_t j = 0; j < k; ++j) small[i * k + j] = big[i * (k + 1) + j]; }
			krige(KrigingType::Simple, g, lags.data(), k, ws, sol);
			const std::vector<double> sk = gaussSolve(small, k0);
			for (size_t i = 0; i < k; ++i) CHECK_NEAR(sol.weights[i], sk[i], 1e-9);
		}
	}
	section("kriging: exactness, singular systems, no neighbour");
	{
		const ResolvedModel g = stationary("0.2*nugget + 0.8*spherical(range=10)", 2);
		const double lags[6] = { 3, 1, 0, 0, -2, 4 }; // neighbour 1 sits on the target
		for (KrigingType type : { KrigingType::Simple, KrigingType::Ordinary }) {
			krige(type, g, lags, 3, ws, sol);
			CHECK_NEAR(sol.weights[1], 1.0, 1e-12);
			CHECK_NEAR(sol.weights[0], 0.0, 1e-12);
			CHECK_NEAR(sol.weights[2], 0.0, 1e-12);
			CHECK_NEAR(sol.variance, 0.0, 1e-12);
		}
		const double duplicate[6] = { 1, 1, 1, 1, -3, 0 };
		const unsigned status = krige(KrigingType::Simple, stationary("spherical(range=10)", 2), duplicate, 3, ws, sol);
		CHECK(status == KrigingJittered);
		CHECK(sol.jitter > 0 && sol.jitter < 1e-6);
		CHECK_NEAR(sol.weights[0], sol.weights[1], 1e-6);
		CHECK(std::isfinite(sol.variance) && sol.variance >= 0);
		krige(KrigingType::Ordinary, stationary("spherical(range=10)", 2), duplicate, 3, ws, sol);
		CHECK(std::isfinite(sol.weights[0]) && std::fabs(sol.weights[0] + sol.weights[1] + sol.weights[2] - 1) < 1e-9);
		// Gaussian model with close neighbours: badly conditioned but must not blow up.
		const double close[8] = { 1, 0, 1.001, 0, 2, 0, 2.0005, 0.0001 };
		krige(KrigingType::Simple, stationary("gaussian(range=50)", 2), close, 4, ws, sol);
		for (double w : sol.weights) CHECK(std::isfinite(w));
		// Unfactorisable: all-zero covariance.
		const double zero[4] = { 0, 0, 0, 0 }, k0[2] = { 0, 0 };
		CHECK(solveKrigingSystem(KrigingType::Simple, zero, 2, k0, 1.0, 2, ws, sol) == KrigingFailed);
		CHECK(sol.weights.size() == 2 && sol.weights[0] == 0 && sol.variance == 1 && sol.meanWeight == 1);
		for (KrigingType type : { KrigingType::Simple, KrigingType::Ordinary }) {
			CHECK(krige(type, g, nullptr, 0, ws, sol) == KrigingNoNeighbor);
			CHECK(sol.weights.empty() && sol.variance == g.sill() && sol.meanWeight == 1);
		}
		const double values[2] = { 3, 5 };
		KrigingSolution manual;
		manual.weights = { 0.25, 0.5 };
		manual.meanWeight = 0.25;
		CHECK_NEAR(krigingEstimate(manual, values, 2.0), 0.75 + 2.5 + 0.5, 1e-15);
		// The callable overload gives the same answer.
		const double lags2[4] = { 1, 2, -3, 1 };
		KrigingSolution a, b;
		krige(KrigingType::Ordinary, g, lags2, 2, ws, a);
		krige(KrigingType::Ordinary, [&](const double* h) { return g.covariance(h); }, g.sill(), 2, lags2, 2, ws, b);
		CHECK_NEAR(a.weights[0], b.weights[0], 0);
		CHECK_NEAR(a.variance, b.variance, 0);
	}
	section("kriging: dense Cholesky");
	{
		const size_t n = 400, nrhs = 3;
		std::mt19937_64 random(11);
		std::normal_distribution<double> normal;
		std::vector<double> b(n * n), a(n * n, 0.0);
		for (double& v : b) v = normal(random);
		for (size_t i = 0; i < n; ++i)
			for (size_t j = 0; j <= i; ++j) {
				const double s = dotProduct(&b[i * n], &b[j * n], n);
				a[i * n + j] = a[j * n + i] = s + (i == j ? double(n) : 0.0);
			}
		std::vector<double> l = a;
		CHECK(choleskyFactor(l.data(), n, n) == 0);
		std::vector<double> x(n), rhs(n * nrhs);
		for (double& v : rhs) v = normal(random);
		for (size_t i = 0; i < n; ++i) x[i] = rhs[i * nrhs];
		choleskySolve(l.data(), n, n, x.data());
		std::vector<double> many = rhs;
		choleskySolve(l.data(), n, n, many.data(), nrhs, nrhs);
		double residual = 0, scale = 0;
		for (size_t i = 0; i < n; ++i) {
			double s = 0;
			for (size_t j = 0; j < n; ++j) s += a[i * n + j] * x[j];
			residual = std::max(residual, std::fabs(s - rhs[i * nrhs]));
			scale = std::max(scale, std::fabs(rhs[i * nrhs]));
			CHECK_NEAR(many[i * nrhs], x[i], 1e-12);
		}
		CHECK(residual < 1e-10 * scale * double(n));
		std::vector<double> singular = { 1, 1, 1, 1 };
		CHECK(choleskyFactor(singular.data(), 2, 2, 1e-12) == 2);
		std::vector<double> factor;
		CHECK(choleskyFactorWithJitter(std::vector<double>({ 1, 1, 1, 1 }).data(), 2, 2, factor) > 0);
	}
}

// ============================================================ neighbours

void testNeighbors() {
	section("neighbours: offsets");
	{
		const std::vector<std::vector<int> > offsets = kernelOffsets({ 5, 5 });
		CHECK(offsets.size() == 24);
		const std::vector<std::vector<int> > first = { { -1, 0 }, { 0, -1 }, { 0, 1 }, { 1, 0 } };
		CHECK(std::vector<std::vector<int> >(offsets.begin(), offsets.begin() + 4) == first);
		CHECK((offsets[4] == std::vector<int>{ -1, -1 }));
		for (size_t i = 1; i < offsets.size(); ++i)
			CHECK(offsets[i - 1][0] * offsets[i - 1][0] + offsets[i - 1][1] * offsets[i - 1][1] <= offsets[i][0] * offsets[i][0] + offsets[i][1] * offsets[i][1]);
		std::vector<float> weights(9, 1.f);
		weights[0] = 0.f;           // offset (-1, -1) dropped
		weights[5] = 3.f;           // offset (+1, 0) first by weight
		const std::vector<std::vector<int> > weighted = kernelOffsets({ 3, 3 }, weights.data(), 1, true);
		CHECK(weighted.size() == 7);
		CHECK((weighted[0] == std::vector<int>{ 1, 0 }));
		std::vector<std::vector<int> > aniso = kernelOffsets({ 7, 7 });
		const double metric[4] = { 1.0, 0.0, 0.0, 0.25 }; // axis 1 counts a quarter
		sortOffsetsByMetric(aniso, metric, 2);
		CHECK(std::abs(aniso[0][1]) == 1 && aniso[0][0] == 0);
		CHECK(aniso[1][0] == 0 && std::abs(aniso[1][1]) == 1 && aniso[2][0] == 0 && std::abs(aniso[2][1]) == 2);
	}
	section("neighbours: grid search");
	{
		const std::vector<unsigned> dims = { 13, 9 };
		const size_t cells = 13 * 9;
		std::vector<unsigned> path(cells);
		for (size_t i = 0; i < cells; ++i) path[i] = unsigned(i);
		std::mt19937 random(5);
		std::shuffle(path.begin(), path.end(), random);
		std::vector<bool> known(cells, false);
		for (size_t i = 0; i < cells; i += 17) known[i] = true;
		const std::vector<int64_t> ranks = pathRanks(cells, path.data(), path.size(), [&](size_t c) { return bool(known[c]); });
		for (size_t c = 0; c < cells; ++c) {
			if (known[c]) CHECK(ranks[c] == kRankKnown);
			else CHECK(path[size_t(ranks[c])] == c);
		}
		const std::vector<std::vector<int> > offsets = kernelOffsets({ 9, 9 });
		GridNeighborSearch search(dims, offsets, ranks.data());
		CHECK(search.ndim() == 2 && search.nodeCount() == cells);
		NeighborSet set;
		// Brute force: first n informed cells in offset order.
		for (size_t node : { size_t(0), size_t(40), size_t(cells - 1), size_t(77) }) {
			const int64_t limit = ranks[node] < 0 ? 0 : ranks[node];
			search.find(node, limit, 6, set);
			std::vector<size_t> expected;
			const long x = long(node % 13), y = long(node / 13);
			for (const std::vector<int>& o : offsets) {
				const long nx = x + o[0], ny = y + o[1];
				if (nx < 0 || ny < 0 || nx >= 13 || ny >= 9) continue;
				const size_t id = size_t(ny * 13 + nx);
				if (ranks[id] < limit) expected.push_back(id);
				if (expected.size() == 6) break;
			}
			CHECK(set.ids == expected);
			CHECK(set.refs.size() == set.size());
			for (size_t i = 0; i < set.size(); ++i) {
				const long nx = long(set.ids[i] % 13), ny = long(set.ids[i] / 13);
				CHECK(set.lag(i)[0] == double(nx - x) && set.lag(i)[1] == double(ny - y));
				CHECK(set.ref(i) < offsets.size() && offsets[set.ref(i)][0] == nx - x && offsets[set.ref(i)][1] == ny - y);
			}
		}
		// Determinism across threads: every node, 4 threads vs 1.
		std::vector<std::vector<size_t> > serial(cells), parallel(cells);
		for (size_t node = 0; node < cells; ++node) {
			search.find(node, std::max<int64_t>(ranks[node], 0), 8, set);
			serial[node] = set.ids;
		}
		std::atomic<size_t> next(0);
		std::vector<std::thread> threads;
		for (int t = 0; t < 4; ++t) threads.emplace_back([&]() {
			NeighborSet local;
			for (size_t node; (node = next.fetch_add(1)) < cells;) {
				search.find(node, std::max<int64_t>(ranks[node], 0), 8, local);
				parallel[node] = local.ids;
			}
		});
		for (std::thread& t : threads) t.join();
		CHECK(serial == parallel);
	}
	section("neighbours: circular and sectors");
	{
		const std::vector<unsigned> dims = { 6, 5 };
		std::vector<int64_t> ranks(30, kRankKnown);
		const std::vector<std::vector<int> > offsets = kernelOffsets({ 3, 3 });
		NeighborSet set;
		GridNeighborSearch plain(dims, offsets, ranks.data());
		plain.find(0, 0, 100, set);
		CHECK(set.size() == 3);
		GridSearchOptions circularOptions;
		circularOptions.circular = true;
		GridNeighborSearch circular(dims, offsets, ranks.data(), circularOptions);
		circular.find(0, 0, 100, set);
		CHECK(set.size() == 8);
		CHECK(set.ids[0] == 5 && set.lag(0)[0] == -1 && set.lag(0)[1] == 0);  // (-1, 0) wraps to x = 5
		CHECK(set.ids[1] == 24 && set.lag(1)[0] == 0 && set.lag(1)[1] == -1); // (0, -1) wraps to y = 4
		// Quadrants: the four axis neighbours fall into four sectors.
		std::vector<int64_t> all(100, kRankKnown);
		GridSearchOptions sectorOptions;
		sectorOptions.maxPerSector = 1;
		GridNeighborSearch quadrant({ 10, 10 }, kernelOffsets({ 7, 7 }), all.data(), sectorOptions);
		quadrant.find(55, 0, 100, set);
		CHECK(set.size() == 4);
		std::vector<unsigned> sectors;
		for (size_t i = 0; i < set.size(); ++i) {
			sectors.push_back(sectorOf(set.lag(i), 2));
			CHECK(std::fabs(set.lag(i)[0]) + std::fabs(set.lag(i)[1]) == 1);
		}
		std::sort(sectors.begin(), sectors.end());
		CHECK((sectors == std::vector<unsigned>{ 0, 1, 2, 3 }));
		sectorOptions.maxPerSector = 2;
		GridNeighborSearch two({ 10, 10 }, kernelOffsets({ 7, 7 }), all.data(), sectorOptions);
		two.find(55, 0, 100, set);
		CHECK(set.size() == 8);
		two.find(55, 0, 5, set);
		CHECK(set.size() == 5);
		// Octants in 3D.
		std::vector<int64_t> cube(7 * 7 * 7, kRankKnown);
		sectorOptions.maxPerSector = 1;
		GridNeighborSearch octant({ 7, 7, 7 }, kernelOffsets({ 5, 5, 5 }), cube.data(), sectorOptions);
		octant.find(3 + 7 * 3 + 49 * 3, 0, 100, set);
		CHECK(set.size() == 8);
		std::vector<unsigned> octants;
		for (size_t i = 0; i < set.size(); ++i) octants.push_back(sectorOf(set.lag(i), 3));
		std::sort(octants.begin(), octants.end());
		CHECK((octants == std::vector<unsigned>{ 0, 1, 2, 3, 4, 5, 6, 7 }));
		const double axes[6][3] = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
		std::vector<unsigned> axisSectors;
		for (const auto& a : axes) axisSectors.push_back(sectorOf(a, 3));
		std::sort(axisSectors.begin(), axisSectors.end());
		CHECK(std::unique(axisSectors.begin(), axisSectors.end()) == axisSectors.end());
		// Rank limit: only strictly smaller ranks.
		std::vector<int64_t> line = { 0, 1, 2, 3, 4 };
		GridNeighborSearch row({ 5 }, kernelOffsets({ 9 }), line.data());
		row.find(2, 2, 10, set);
		CHECK((set.ids == std::vector<size_t>{ 1, 0 }));
	}
}

// ============================================================ random numbers

void testRandom() {
	section("random: Philox known answers");
	{
		const std::array<uint32_t, 4> zero = philox4x32({ 0, 0, 0, 0 }, { 0, 0 });
		CHECK((zero == std::array<uint32_t, 4>{ 0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u }));
		const std::array<uint32_t, 4> ones = philox4x32({ 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu }, { 0xffffffffu, 0xffffffffu });
		CHECK((ones == std::array<uint32_t, 4>{ 0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu }));
		const std::array<uint32_t, 4> pi = philox4x32({ 0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u }, { 0xa4093822u, 0x299f31d0u });
		CHECK((pi == std::array<uint32_t, 4>{ 0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u }));
	}
	section("random: reproducibility and moments");
	{
		CHECK(standardNormal(42, 3, 1000) == standardNormal(42, 3, 1000));
		CHECK(standardNormal(42, 3, 1000) != standardNormal(42, 4, 1000));
		CHECK(standardNormal(42, 3, 1000) != standardNormal(42, 3, 1001));
		CHECK(standardNormal(42, 3, 1000) != standardNormal(42, 3, 1000, 1));
		CHECK(standardNormal(42, 3, 1000) != standardNormal(43, 3, 1000));
		CHECK(uniform(1, 0, uint64_t(1) << 40) != uniform(1, 0, 0));
		const size_t n = 2000000;
		double s1 = 0, s2 = 0, s3 = 0, s4 = 0, u1 = 0, uMin = 1, uMax = 0;
		for (size_t i = 0; i < n; ++i) {
			const double z = standardNormal(2026, 7, i);
			s1 += z; s2 += z * z; s3 += z * z * z; s4 += z * z * z * z;
			const double u = uniform(2026, 7, i, 9);
			u1 += u; uMin = std::min(uMin, u); uMax = std::max(uMax, u);
		}
		const double mean = s1 / n, variance = s2 / n - mean * mean;
		CHECK_NEAR(mean, 0, 0.004);
		CHECK_NEAR(variance, 1, 0.006);
		CHECK_NEAR(s3 / n, 0, 0.012);
		CHECK_NEAR(s4 / n, 3, 0.03);
		CHECK_NEAR(u1 / n, 0.5, 0.001);
		CHECK(uMin > 0 && uMax < 1);
		// Thread independence.
		std::vector<double> serial(10000), parallel(10000);
		for (size_t i = 0; i < serial.size(); ++i) serial[i] = standardNormal(9, 1, i);
		std::vector<std::thread> threads;
		for (size_t t = 0; t < 4; ++t) threads.emplace_back([&, t]() { for (size_t i = t; i < parallel.size(); i += 4) parallel[i] = standardNormal(9, 1, i); });
		for (std::thread& t : threads) t.join();
		CHECK(serial == parallel);
		double a, b;
		standardNormalPair(5, 2, 77, 3, a, b);
		CHECK(a == standardNormal(5, 2, 77, 3) && b == inverseNormalCdf(uniform2(5, 2, 77, 3)));
	}
	section("random: fast normals");
	{
		// Fills equal the scalar functions whatever the start and the count.
		bool fillsMatch = true;
		std::vector<float> values(300);
		for (uint64_t first : { uint64_t(0), uint64_t(1), uint64_t(2), uint64_t(3), uint64_t(6), (uint64_t(1) << 33) + 1 })
			for (size_t count : { size_t(0), size_t(1), size_t(2), size_t(3), size_t(4), size_t(5), size_t(127), size_t(128), size_t(129), size_t(300) }) {
				fillFastNormalsByIndex(7, 3, first, count, 2, values.data());
				for (size_t k = 0; k < count; ++k) fillsMatch &= values[k] == fastNormalByIndex(7, 3, first + k, 2);
				fillFastNormalsByRealisation(7, uint32_t(first), count, (uint64_t(5) << 32) + 9, 2, values.data());
				for (size_t k = 0; k < count; ++k) fillsMatch &= values[k] == fastNormalByRealisation(7, uint32_t(first + k), (uint64_t(5) << 32) + 9, 2);
			}
		CHECK(fillsMatch);
		CHECK(fastNormalByIndex(42, 3, 1000) != fastNormalByIndex(42, 4, 1000));
		CHECK(fastNormalByIndex(42, 3, 1000) != fastNormalByIndex(42, 3, 1001));
		CHECK(fastNormalByIndex(42, 3, 1000) != fastNormalByIndex(42, 3, 1000, 1));
		CHECK(fastNormalByIndex(42, 3, 1000) != fastNormalByIndex(43, 3, 1000));
		CHECK(fastNormalByIndex(42, 3, uint64_t(1) << 40) != fastNormalByIndex(42, 3, 0));
		CHECK(fastNormalByRealisation(42, 3, 1000) != fastNormalByIndex(42, 3, 1000));
		CHECK(fastNormalByRealisation(42, 3, 1000) != fastNormalByRealisation(42, 3, 1000 + (uint64_t(1) << 32)));
		// Moments and tails of both families.
		const size_t n = 4000000;
		std::vector<float> byIndex(n), byRealisation(n);
		fillFastNormalsByIndex(2026, 7, 0, n, 0, byIndex.data());
		for (size_t i = 0; i < n; i += 64) fillFastNormalsByRealisation(2026, 0, 64, i / 64, 0, byRealisation.data() + i);
		for (const std::vector<float>* z : { &byIndex, &byRealisation }) {
			double s1 = 0, s2 = 0, s3 = 0, s4 = 0, largest = 0;
			size_t beyond2 = 0, beyond3 = 0;
			for (float v : *z) {
				const double x = v;
				s1 += x; s2 += x * x; s3 += x * x * x; s4 += x * x * x * x;
				largest = std::max(largest, std::fabs(x));
				beyond2 += std::fabs(x) > 2; beyond3 += std::fabs(x) > 3;
			}
			const double mean = s1 / n, variance = s2 / n - mean * mean;
			CHECK_NEAR(mean, 0, 0.003);
			CHECK_NEAR(variance, 1, 0.004);
			CHECK_NEAR(s3 / n, 0, 0.008);
			CHECK_NEAR(s4 / n, 3, 0.02);
			CHECK_NEAR(double(beyond2) / n, std::erfc(2 / std::sqrt(2.0)), 0.0005);
			CHECK_NEAR(double(beyond3) / n, std::erfc(3 / std::sqrt(2.0)), 0.0001);
			CHECK(largest < 6.77 && largest > 4.5);
		}
		// Neighbouring values of a block are uncorrelated.
		double lag1 = 0;
		for (size_t i = 0; i + 1 < n; ++i) lag1 += double(byIndex[i]) * double(byIndex[i + 1]);
		CHECK_NEAR(lag1 / n, 0, 0.003);
		// The building blocks.
		double logError = 0;
		for (uint64_t k = 1; k < (uint64_t(1) << 32); k += 65537) {
			const float u = float(k) * 0x1p-32f + 0x1p-33f;
			logError = std::max(logError, std::fabs(double(detail::fastLog(u)) - std::log(double(u))) / std::max(1e-6, std::fabs(std::log(double(u)))));
		}
		CHECK(logError < 3e-7);
		double angleError = 0;
		const double radius = std::sqrt(-2.0 * std::log(0.5 + std::ldexp(1.0, -33)));
		for (uint64_t k = 0; k < (uint64_t(1) << 32); k += 65537) {
			float c, s;
			detail::boxMuller(0x80000000u, uint32_t(k), c, s);
			const double angle = 6.283185307179586 * double(k) / 4294967296.0;
			angleError = std::max(angleError, std::max(std::fabs(c - radius * std::cos(angle)), std::fabs(s - radius * std::sin(angle))));
		}
		CHECK(angleError < 5e-7);
	}
	section("random: inverse normal CDF");
	{
		CHECK_NEAR(inverseNormalCdf(0.5), 0.0, 1e-16);
		CHECK_NEAR(inverseNormalCdf(0.975), 1.959963984540054, 1e-14);
		CHECK_NEAR(inverseNormalCdf(0.025), -1.959963984540054, 1e-14);
		CHECK_NEAR(inverseNormalCdf(0.8413447460685429), 1.0, 1e-13);
		CHECK_NEAR(inverseNormalCdf(1e-10), -6.361340902404056, 1e-12);
		CHECK(std::isinf(inverseNormalCdf(0.0)) && inverseNormalCdf(0.0) < 0 && std::isinf(inverseNormalCdf(1.0)));
		double worst = 0;
		for (double p = 1e-300; p < 0.5; p *= 1.7) {
			const double x = inverseNormalCdf(p);
			worst = std::max(worst, std::fabs(normalCdf(x) - p) / p / (1 + x * x)); // conditioning of the tail
		}
		CHECK_MSG(worst < 1e-15, "lower tail relative error " + formatNumber(worst));
		worst = 0;
		for (int i = 1; i < 10000; ++i) {
			const double p = i / 10000.0;
			worst = std::max(worst, std::fabs(normalCdf(inverseNormalCdf(p)) - p));
			CHECK(std::fabs(inverseNormalCdf(p) + inverseNormalCdf(1 - p)) < 1e-12);
		}
		CHECK_MSG(worst < 1e-15, "absolute error " + formatNumber(worst));
	}
}

// ============================================================ command line

bool writeImage(const std::string& name, const std::vector<unsigned>& dims, const std::vector<float>& values) {
	std::vector<unsigned> d = dims;
	g2s::DataImage image(unsigned(d.size()), d.data(), 1);
	for (size_t i = 0; i < values.size(); ++i) image._data[i] = values[i];
	image.write(name);
	return !g2s::DataImage::createFromFile(name).isEmpty();
}

void testArguments() {
	section("arguments: -cov / -cvar / -ii");
	{
		const char* argv[] = { "x", "-cov", "nug*nugget + (1-nug)*spherical(range=L, angle=theta)", "-cvar", "theta", "-30",
		                       "-cvar", "nug", "0.1", "L=40", "-n", "5", "-cvar", "k", "-1e-3", "-ii", "idx" };
		const int argc = int(sizeof(argv) / sizeof(argv[0]));
		std::multimap<std::string, std::string> arg = g2s::argumentReader(argc, argv);
		CovarianceArguments ca;
		std::string error;
		CHECK_MSG(readCovarianceArguments(arg, argc, argv, ca, error), error);
		CHECK(ca.models.size() == 1);
		const std::vector<std::pair<std::string, std::string> > expected = { { "theta", "-30" }, { "nug", "0.1" }, { "L", "40" }, { "k", "-1e-3" } };
		CHECK(ca.variables == expected);
		CHECK(ca.indexFileName == "idx");
		CHECK(arg.size() == 1 && arg.count("-n") == 1); // -30, -1e-3, -cov, -cvar, -ii removed
		const char* odd[] = { "x", "-cvar", "a", "1", "b" };
		std::multimap<std::string, std::string> oddArg = g2s::argumentReader(5, odd);
		CHECK(!readCovarianceArguments(oddArg, 5, odd, ca, error));
		CHECK(error.find("no value") != std::string::npos);
		const char* reserved[] = { "x", "-cvar", "range", "1" };
		std::multimap<std::string, std::string> reservedArg = g2s::argumentReader(4, reserved);
		CHECK(!readCovarianceArguments(reservedArg, 4, reserved, ca, error));
		CHECK(error.find("reserved word") != std::string::npos);
	}
	mkdir("/tmp/G2S", 0770);
	mkdir("/tmp/G2S/data", 0770);
	{
		// 4 x 3 grid.
		const std::vector<unsigned> grid = { 4, 3 };
		std::vector<float> ranges(12), constant(12, 7.f), table(12, 0.f), index(12, 0.f);
		for (size_t i = 0; i < 12; ++i) { ranges[i] = float(5 + i); index[i] = float(i % 2); }
		table[1 * 4 + 2] = 2.f; // centre (2, 1)
		table[1 * 4 + 3] = 1.f; // lag (+1, 0)
		CHECK(writeImage(N("geostat_test_range"), grid, ranges));
		CHECK(writeImage(N("geostat_test_const"), grid, constant));
		CHECK(writeImage(N("geostat_test_table"), grid, table));
		CHECK(writeImage(N("geostat_test_index"), grid, index));
		CHECK(writeImage(N("geostat_test_small"), { 2, 2 }, { 1, 2, 3, 4 }));

		CovarianceArguments ca;
		ca.models = { "exponential(range=L)", "s*spherical(range=K) + 0.5*map(C)" };
		ca.variables = { { "L", N("geostat_test_range") }, { "K", N("geostat_test_const") }, { "s", "2" }, { "C", N("geostat_test_table") } };
		ca.indexFileName = N("geostat_test_index");
		std::vector<unsigned> dims;
		CovarianceModelSet models;
		std::string error;
		CHECK_MSG(loadCovarianceModels(nullptr, ca, dims, 0, models, error), error);
		CHECK(dims == grid);
		CHECK(models.size() == 2 && !models.isStationary());
		CHECK(!models.models[0].isStationary() && models.models[1].isStationary()); // K constant image -> constant
		ResolvedModel scratch;
		const ResolvedModel& m6 = models.at(6, scratch); // index 0, L = 11
		CHECK(m6.modelIndex == 0);
		CHECK_NEAR(m6.structures[0].range[0], 11, 0);
		const ResolvedModel& m7 = models.at(7, scratch); // index 1
		CHECK(m7.modelIndex == 1);
		const double lag[2] = { 1, 0 };
		CHECK_NEAR(m7.covariance(lag), 2 * (1 - 1.5 / 7 + 0.5 / 343) + 0.5 * 1, 1e-6);
		CHECK_NEAR(m7.sill(), 2 + 0.5 * 2, 1e-12);

		std::vector<unsigned> wrong = { 4, 4 };
		CHECK(!loadCovarianceModels(nullptr, ca, wrong, 0, models, error));
		CHECK(error.find("has size 4x3 but the grid is 4x4") != std::string::npos);
		CovarianceArguments small = ca;
		small.variables[0].second = N("geostat_test_small");
		dims.clear();
		CHECK(!loadCovarianceModels(nullptr, small, dims, 0, models, error));
		CovarianceArguments mapNumber;
		mapNumber.models = { "map(C)" };
		mapNumber.variables = { { "C", "3" } };
		dims.clear();
		CHECK(!loadCovarianceModels(nullptr, mapNumber, dims, 2, models, error));
		CHECK(error.find("needs a covariance map image") != std::string::npos);
		CovarianceArguments missing;
		missing.models = { "exponential(range=L)" };
		missing.variables = { { "L", N("geostat_test_does_not_exist") } };
		dims.clear();
		CHECK(!loadCovarianceModels(nullptr, missing, dims, 1, models, error));
		CHECK(error.find("failed to load -cvar L") != std::string::npos);
		CovarianceArguments stationaryArgs;
		stationaryArgs.models = { "0.1*nugget + 0.9*gaussian(range=[30,10], angle=-15)" };
		dims.clear();
		CHECK(loadCovarianceModels(nullptr, stationaryArgs, dims, 2, models, error));
		CHECK(models.isStationary() && dims.empty());

		// Map tables have their own size (any size, lag 0 at the centre), and
		// several maps of different sizes can be combined; per-node images
		// still have to match the grid.
		std::vector<float> wide(7 * 5, 0.f), tall(3 * 9, 0.f);
		wide[2 * 7 + 3] = 4.f; // centre (3, 2)
		wide[2 * 7 + 4] = 2.f; // lag (+1, 0)
		tall[4 * 3 + 1] = 3.f; // centre (1, 4)
		tall[5 * 3 + 1] = 1.f; // lag (0, +1)
		CHECK(writeImage(N("geostat_test_wide"), { 7, 5 }, wide));
		CHECK(writeImage(N("geostat_test_tall"), { 3, 9 }, tall));
		CovarianceArguments maps;
		maps.models = { "0.5*map(A) + map(B) + 0.1*exponential(range=L)" };
		maps.variables = { { "A", N("geostat_test_wide") }, { "B", N("geostat_test_tall") }, { "L", N("geostat_test_range") } };
		dims = grid;
		CHECK_MSG(loadCovarianceModels(nullptr, maps, dims, 0, models, error), error);
		CHECK(dims == grid && !models.isStationary());
		{
			const ResolvedModel& m = models.at(5, scratch); // L = 10
			const double zero[2] = { 0, 0 }, right[2] = { 1, 0 }, up[2] = { 0, 1 }, far[2] = { 4, 0 };
			CHECK_NEAR(m.covariance(zero), 0.5 * 4 + 3 + 0.1, 1e-12);
			CHECK_NEAR(m.covariance(right), 0.5 * 2 + 0.1 * std::exp(-0.3), 1e-7);
			CHECK_NEAR(m.covariance(up), 1 + 0.1 * std::exp(-0.3), 1e-7);
			CHECK_NEAR(m.covariance(far), 0.1 * std::exp(-1.2), 1e-7); // outside both tables
		}
		// Without -di, maps do not define the grid; a field does.
		CovarianceArguments mapsOnly;
		mapsOnly.models = { "0.5*map(A) + map(B)" };
		mapsOnly.variables = { { "A", N("geostat_test_wide") }, { "B", N("geostat_test_tall") } };
		dims.clear();
		CHECK_MSG(loadCovarianceModels(nullptr, mapsOnly, dims, 2, models, error), error);
		CHECK(dims.empty() && models.isStationary());
		CHECK_NEAR(models.stationaryModel().sill(), 5, 1e-12);
		dims = { 30, 30 };
		CHECK_MSG(loadCovarianceModels(nullptr, mapsOnly, dims, 0, models, error), error); // maps larger or smaller than the grid
		dims.clear();
		CHECK_MSG(loadCovarianceModels(nullptr, maps, dims, 0, models, error), error);
		CHECK(dims == grid); // from the field
		CovarianceArguments wrongField = maps;
		wrongField.variables[2].second = N("geostat_test_small");
		dims = grid;
		CHECK(!loadCovarianceModels(nullptr, wrongField, dims, 0, models, error));
		CHECK(error.find("has size 2x2 but the grid is 4x3") != std::string::npos);
		CovarianceArguments wrongDims = mapsOnly;
		dims.clear();
		CHECK(!loadCovarianceModels(nullptr, wrongDims, dims, 3, models, error));
		CHECK(error.find("has 2 dimensions, the lags have 3") != std::string::npos);
	}
}

void testCovEval(const std::string& binary) {
	section("covEval executable");
	FILE* probe = fopen(binary.c_str(), "r");
	if (!probe) { printf("   skipped: %s not built\n", binary.c_str()); return; }
	fclose(probe);
	std::vector<float> lags = { 0, 0, 10, 0, 0, 10, 20, 5, -7, 3 }; // 5 x 2
	CHECK(writeImage(N("geostat_test_lags"), { 2, 5 }, lags));
	std::vector<float> ranges(12);
	for (size_t i = 0; i < 12; ++i) ranges[i] = float(5 + i);
	CHECK(writeImage(N("geostat_test_range"), { 4, 3 }, ranges));
	CHECK(writeImage(N("geostat_test_pos"), { 2, 1 }, { 2, 1 })); // cell (2, 1) -> L = 11

	const std::string model = "0.1*nugget + 0.9*spherical(range=[40,20], angle=30)";
	std::string command = binary + " -cov '" + model + "' -lag " + N("geostat_test_lags") + " -vario -id " + testIdText(650) + " -o " + N("geostat_test_out") + " 2>/dev/null";
	CHECK(std::system(command.c_str()) == 0);
	g2s::DataImage c = g2s::DataImage::createFromFile("im_1_" + testIdText(650));
	g2s::DataImage v = g2s::DataImage::createFromFile("im_2_" + testIdText(650));
	CHECK(!c.isEmpty() && !v.isEmpty() && c._dims == std::vector<unsigned>{ 5 });
	if (!c.isEmpty() && !v.isEmpty()) {
		const ResolvedModel m = stationary(model, 2);
		for (size_t i = 0; i < 5; ++i) {
			CHECK_NEAR(c._data[i], m.covariance(std::vector<double>{ lags[2 * i], lags[2 * i + 1] }.data()), 1e-6);
			CHECK_NEAR(v._data[i], m.sill() - c._data[i], 1e-6);
		}
	}
	command = binary + " -cov 'exponential(range=L)' -cvar L " + N("geostat_test_range") + " -lag " + N("geostat_test_lags") + " -pos " + N("geostat_test_pos") + " -id " + testIdText(651) + " -o " + N("geostat_test_out") + " 2>/dev/null";
	CHECK(std::system(command.c_str()) == 0);
	g2s::DataImage n = g2s::DataImage::createFromFile("im_1_" + testIdText(651));
	CHECK(!n.isEmpty());
	if (!n.isEmpty()) CHECK_NEAR(n._data[1], std::exp(-3.0 * 10 / 11), 1e-6);
	// Non-stationary without -pos and invalid model are errors.
	command = binary + " -cov 'exponential(range=L)' -cvar L " + N("geostat_test_range") + " -lag " + N("geostat_test_lags") + " -id " + testIdText(652) + " -o " + N("geostat_test_out") + " 2>/dev/null";
	CHECK(std::system(command.c_str()) != 0);
	command = binary + " -cov 'exponential(range=-1)' -lag " + N("geostat_test_lags") + " -id " + testIdText(653) + " -o " + N("geostat_test_out") + " 2>/dev/null";
	CHECK(std::system(command.c_str()) != 0);
	command = binary + " -cov 'exponential(range=theta*-1)' -cvar theta -5 -lag " + N("geostat_test_lags") + " -id " + testIdText(654) + " -o " + N("geostat_test_out") + " 2>/dev/null";
	CHECK(std::system(command.c_str()) == 0);
}

} // namespace

int main(int argc, char** argv) {
	const std::string casesFile = argc > 1 ? argv[1] : "cov_cases.txt";
	const std::string covEval = argc > 2 ? argv[2] : "../../build/c++-build/covEval";
	testGrammar(casesFile);
	testModels();
	testVariables();
	testKriging();
	testNeighbors();
	testRandom();
	testArguments();
	testCovEval(covEval);
	printf("%d checks, %d failures\n", checks, failures);
	return failures == 0 ? 0 : 1;
}
