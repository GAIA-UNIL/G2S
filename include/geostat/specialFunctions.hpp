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

// Special functions used by the geostat foundation, without external
// libraries (libc++ has no std::cyl_bessel_k):
//   * logBesselK(nu, x): log of the modified Bessel function of the second kind;
//   * maternCorrelation(nu, u): 2^(1-nu)/Gamma(nu) u^nu K_nu(u), u = h/scale;
//   * maternPracticalRange(nu): the u where maternCorrelation(nu, u) = exp(-3);
//   * normalCdf(x) and inverseNormalCdf(p) of the standard normal.
// All functions are pure and thread-safe.

#ifndef G2S_GEOSTAT_SPECIAL_FUNCTIONS_HPP
#define G2S_GEOSTAT_SPECIAL_FUNCTIONS_HPP

#include <algorithm>
#include <cmath>
#include <limits>

namespace g2s {
namespace geostat {

// log K_nu(x) for x > 0, any real nu (K_-nu = K_nu). Uses the integral
// K_nu(x) = int_0^inf exp(-x cosh t) cosh(nu t) dt with the trapezoidal rule,
// which converges geometrically for this analytic, rapidly decaying integrand.
// Terms are scaled by the value of -x cosh t + nu t at its maximum, so large
// nu or tiny x do not overflow; one exp per term. Relative accuracy is close
// to double precision (checked in test/geostat).
inline double logBesselK(double nu, double x) {
	nu = std::fabs(nu);
	if (!(x > 0)) return std::numeric_limits<double>::infinity();
	// The integrand peaks near sinh t = nu/x with a width of about
	// (x^2 + nu^2)^(-1/4); half a width per step keeps the aliasing error
	// below exp(-70), and 0.2 bounds the step where the peak is wide.
	const double step = std::min(0.2, 0.5 / std::pow(x * x + nu * nu, 0.25));
	const double peak = std::asinh(nu / x);
	// cosh(nu t) = exp(nu t) (1 + exp(-2 nu t)) / 2; reference = max of -x cosh t + nu t.
	const double reference = -x * std::cosh(peak) + nu * peak;
	const double growth = std::exp(step), decay = std::exp(-2.0 * nu * step);
	double e = 1.0, d = 1.0; // exp(t), exp(-2 nu t)
	double sum = 0.5 * std::exp(-x - reference); // t = 0, half weight (cosh(0) = 1)
	for (int k = 1; k < 1000000; ++k) {
		e *= growth;
		d *= decay;
		const double t = k * step;
		const double exponent = -x * 0.5 * (e + 1.0 / e) + nu * t - reference;
		sum += std::exp(exponent) * 0.5 * (1.0 + d);
		if (t > peak && exponent < -40.0) break;
	}
	return std::log(step) + reference + std::log(sum);
}

// Matérn correlation in its standard scale form,
//   rho(u) = 2^(1-nu) / Gamma(nu) * u^nu * K_nu(u),   u = h / scale,
// so rho(0) = 1; nu = 0.5 gives exp(-u), nu = 1.5 gives (1 + u) exp(-u),
// nu = 2.5 gives (1 + u + u^2/3) exp(-u) (these three are evaluated in closed form).
// logNorm may be passed as (1 - nu) ln 2 - lgamma(nu) to save its evaluation.
inline double maternCorrelation(double nu, double u, double logNorm) {
	if (!(u > 0)) return 1.0;
	if (nu == 0.5) return std::exp(-u);
	if (nu == 1.5) return (1.0 + u) * std::exp(-u);
	if (nu == 2.5) return (1.0 + u + u * u / 3.0) * std::exp(-u);
	const double logRho = logNorm + nu * std::log(u) + logBesselK(nu, u);
	return logRho >= 0.0 ? 1.0 : std::exp(logRho);
}

inline double maternLogNorm(double nu) { return (1.0 - nu) * 0.69314718055994530942 - std::lgamma(nu); }

inline double maternCorrelation(double nu, double u) { return maternCorrelation(nu, u, maternLogNorm(nu)); }

// The u > 0 where maternCorrelation(nu, u) = exp(-3) (about 4.98%): the
// practical range of a Matérn structure of scale 1. Equal to 3 for nu = 0.5
// (exponential); grows like sqrt(nu) for large nu.
// Solved with the Illinois method on log u; about ten correlation evaluations.
inline double maternPracticalRange(double nu) {
	if (nu == 0.5) return 3.0;
	const double logNorm = maternLogNorm(nu);
	auto f = [&](double logU) {
		const double rho = maternCorrelation(nu, std::exp(logU), logNorm);
		return (rho > 0 ? std::log(rho) : -745.0) + 3.0;
	};
	double a = std::log(3.0), fa = f(a);
	double b = a, fb = fa;
	// Bracket the root: f is decreasing in log u.
	if (fa > 0) {
		do { b += 1.0; fb = f(b); } while (fb > 0 && b < 700);
	} else {
		do { a -= 1.0; fa = f(a); } while (fa <= 0 && a > -700);
	}
	if (fa == 0) return std::exp(a);
	if (fb == 0) return std::exp(b);
	int side = 0;
	for (int iteration = 0; iteration < 200; ++iteration) {
		const double c = (a * fb - b * fa) / (fb - fa);
		const double fc = f(c);
		if (fc == 0 || std::fabs(b - a) < 1e-15 * std::max(1.0, std::fabs(c))) return std::exp(c);
		if ((fc > 0) == (fa > 0)) {
			a = c; fa = fc;
			if (side == -1) fb *= 0.5;
			side = -1;
		} else {
			b = c; fb = fc;
			if (side == 1) fa *= 0.5;
			side = 1;
		}
	}
	return std::exp(0.5 * (a + b));
}

// Standard normal cumulative distribution function.
inline double normalCdf(double x) { return 0.5 * std::erfc(-x * 0.70710678118654752440); }

// Standard normal quantile for p in (0, 1); -inf at p <= 0, +inf at p >= 1,
// NaN for NaN. Acklam's rational approximation (relative error < 1.2e-9)
// refined by one Halley step on erfc, which brings it to about 1e-15 over
// the whole range, tails included.
inline double inverseNormalCdf(double p) {
	if (std::isnan(p)) return p;
	if (p <= 0.0) return -std::numeric_limits<double>::infinity();
	if (p >= 1.0) return std::numeric_limits<double>::infinity();
	static const double a[] = { -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
	                            1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00 };
	static const double b[] = { -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
	                            6.680131188771972e+01, -1.328068155288572e+01 };
	static const double c[] = { -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
	                            -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00 };
	static const double d[] = { 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
	                            3.754408661907416e+00 };
	const double low = 0.02425;
	double x;
	if (p < low) {
		const double q = std::sqrt(-2.0 * std::log(p));
		x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
		    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
	} else if (p <= 1.0 - low) {
		const double q = p - 0.5, r = q * q;
		x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
		    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
	} else {
		const double q = std::sqrt(-2.0 * std::log1p(-p));
		x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
		    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
	}
	// Halley refinement. In the upper tail, work with the complement so that
	// the residual keeps its relative accuracy.
	if (p > 0.5) {
		const double e = 0.5 * std::erfc(x * 0.70710678118654752440) - (1.0 - p);
		const double u = -e * 2.50662827463100050242 * std::exp(0.5 * x * x);
		x = x - u / (1.0 + 0.5 * x * u);
	} else {
		const double e = 0.5 * std::erfc(-x * 0.70710678118654752440) - p;
		const double u = e * 2.50662827463100050242 * std::exp(0.5 * x * x);
		x = x - u / (1.0 + 0.5 * x * u);
	}
	return x;
}

} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_SPECIAL_FUNCTIONS_HPP
