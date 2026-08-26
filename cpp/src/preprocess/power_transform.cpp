#include "preprocess/power_transform.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "preprocess/stats.h"

namespace tabicl {

namespace {

constexpr double kEps = 2.220446049250313e-16;      // np.finfo(float64).eps
constexpr double kSpacing1 = 2.220446049250313e-16;  // np.spacing(1.0)
constexpr double kTiny = 2.2250738585072014e-308;    // np.finfo(float64).tiny
constexpr double kMax = 1.7976931348623157e308;      // np.finfo(float64).max

// yeojohnson_llf's negative: -llf, with llf = -n/2*log(var(trans)) +
// (lmb-1)*sum(sign(x)*log1p|x|). Variance/sums are 1D pairwise (numpy).
double neg_llf(double lmb, const std::vector<double>& x, double sign_log1p_sum) {
  const int64_t n = static_cast<int64_t>(x.size());
  std::vector<double> trans(x.size());
  for (size_t i = 0; i < x.size(); ++i)
    trans[i] = yeojohnson_transform_value(x[i], lmb);
  const double mean = pairwise_sum(trans.data(), n) / static_cast<double>(n);
  for (double& t : trans) {
    const double dev = t - mean;
    t = dev * dev;
  }
  const double var = pairwise_sum(trans.data(), n) / static_cast<double>(n);
  if (std::isnan(var)) return std::numeric_limits<double>::quiet_NaN();  // scipy propagates
  if (var < kTiny || std::isinf(var)) {
    // scipy: tiny/inf variance -> llf = +/-inf -> llf[isinf] = -inf -> -llf = +inf.
    return std::numeric_limits<double>::infinity();
  }
  const double loglike = -static_cast<double>(n) / 2.0 * std::log(var) +
                         (lmb - 1.0) * sign_log1p_sum;
  return -loglike;
}

}  // namespace

double yeojohnson_transform_value(double x, double lmbda) {
  if (x >= 0.0) {
    if (std::abs(lmbda) < kSpacing1) return std::log1p(x);
    return (std::pow(x + 1.0, lmbda) - 1.0) / lmbda;
  }
  if (std::abs(lmbda - 2.0) > kSpacing1)
    return -(std::pow(-x + 1.0, 2.0 - lmbda) - 1.0) / (2.0 - lmbda);
  return -std::log1p(-x);
}

double yeojohnson_normmax(const std::vector<double>& x) {
  const int64_t n = static_cast<int64_t>(x.size());
  if (n == 0) throw std::runtime_error("yeojohnson: empty column");
  bool all_zero = true, all_neg = true, any_neg = false;
  double max_abs = 0.0;
  for (double v : x) {
    if (!std::isfinite(v)) throw std::runtime_error("Yeo-Johnson input must be finite.");
    all_zero &= (v == 0.0);
    all_neg &= (v < 0.0);
    any_neg |= (v < 0.0);
    max_abs = std::max(max_abs, std::abs(v));
  }
  if (all_zero) return 1.0;

  // Data-derived bounds (scipy yeojohnson_normmax).
  const double log1p_max_x = std::log1p(20.0 * max_abs);
  const double log_eps = std::log(kEps);
  const double log_tiny_float = (std::log(kTiny) - log_eps) / 2.0;
  const double log_max_float = (std::log(kMax) + log_eps) / 2.0;
  const double lb0 = log_tiny_float / log1p_max_x;
  const double ub0 = log_max_float / log1p_max_x;
  double lb = lb0, ub = ub0;
  // scipy computes the tuple RHS from the ORIGINAL lb/ub before assignment.
  if (all_neg) {
    lb = 2.0 - ub0;
    ub = 2.0 - lb0;
  } else if (any_neg) {
    lb = std::max(2.0 - ub0, lb0);
    ub = std::min(2.0 - lb0, ub0);
  }

  // Precompute (lmb-1) coefficient term: sum(sign(x)*log1p(|x|)) (1D pairwise).
  std::vector<double> slg(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    const double s = x[i] > 0.0 ? 1.0 : (x[i] < 0.0 ? -1.0 : 0.0);
    slg[i] = s * std::log1p(std::abs(x[i]));
  }
  const double sign_log1p_sum = pairwise_sum(slg.data(), n);

  // scipy.optimize.fminbound (_minimize_scalar_bounded), xatol=1.48e-8,
  // maxfun=500 — iterate-for-iterate port.
  const double xatol = 1.48e-08;
  const int maxfun = 500;
  const double sqrt_eps = std::sqrt(2.2e-16);
  const double golden_mean = 0.5 * (3.0 - std::sqrt(5.0));
  double a = lb, b = ub;
  double fulc = a + golden_mean * (b - a);
  double nfc = fulc, xf = fulc;
  double rat = 0.0, e = 0.0;
  double fx = neg_llf(xf, x, sign_log1p_sum);
  int num = 1;
  double fu = std::numeric_limits<double>::infinity();
  double ffulc = fx, fnfc = fx;
  double xm = 0.5 * (a + b);
  double tol1 = sqrt_eps * std::abs(xf) + xatol / 3.0;
  double tol2 = 2.0 * tol1;

  while (std::abs(xf - xm) > (tol2 - 0.5 * (b - a))) {
    bool golden = true;
    if (std::abs(e) > tol1) {
      golden = false;
      double r = (xf - nfc) * (fx - ffulc);
      double q = (xf - fulc) * (fx - fnfc);
      double p = (xf - fulc) * q - (xf - nfc) * r;
      q = 2.0 * (q - r);
      if (q > 0.0) p = -p;
      q = std::abs(q);
      r = e;
      e = rat;
      if ((std::abs(p) < std::abs(0.5 * q * r)) && (p > q * (a - xf)) &&
          (p < q * (b - xf))) {
        rat = p / q;
        double xtrial = xf + rat;
        if (((xtrial - a) < tol2) || ((b - xtrial) < tol2)) {
          double si = (xm - xf > 0.0 ? 1.0 : (xm - xf < 0.0 ? -1.0 : 0.0)) +
                      ((xm - xf) == 0.0 ? 1.0 : 0.0);
          rat = tol1 * si;
        }
      } else {
        golden = true;
      }
    }
    if (golden) {
      if (xf >= xm) e = a - xf;
      else e = b - xf;
      rat = golden_mean * e;
    }
    const double si = (rat > 0.0 ? 1.0 : (rat < 0.0 ? -1.0 : 0.0)) +
                      (rat == 0.0 ? 1.0 : 0.0);
    const double xtry = xf + si * std::max(std::abs(rat), tol1);
    fu = neg_llf(xtry, x, sign_log1p_sum);
    num += 1;

    if (fu <= fx) {
      if (xtry >= xf) a = xf;
      else b = xf;
      fulc = nfc; ffulc = fnfc;
      nfc = xf; fnfc = fx;
      xf = xtry; fx = fu;
    } else {
      if (xtry < xf) a = xtry;
      else b = xtry;
      if ((fu <= fnfc) || (nfc == xf)) {
        fulc = nfc; ffulc = fnfc;
        nfc = xtry; fnfc = fu;
      } else if ((fu <= ffulc) || (fulc == xf) || (fulc == nfc)) {
        fulc = xtry; ffulc = fu;
      }
    }
    xm = 0.5 * (a + b);
    tol1 = sqrt_eps * std::abs(xf) + xatol / 3.0;
    tol2 = 2.0 * tol1;
    if (num >= maxfun) break;
  }
  return xf;
}

Matrix PowerTransformerYJ::fit_transform(const Matrix& X) {
  const int64_t n = X.n, d = X.d;
  lambdas_.assign(static_cast<size_t>(d), 0.0);
  Matrix out = X;
  constexpr double eps = kEps;

  // mean/var per column (F-order in Python -> per-column pairwise).
  std::vector<double> col(static_cast<size_t>(n));
  for (int64_t j = 0; j < d; ++j) {
    for (int64_t i = 0; i < n; ++i) col[static_cast<size_t>(i)] = X.at(i, j);
    const double mean = pairwise_sum(col.data(), n) / static_cast<double>(n);
    std::vector<double> dev(col);
    for (double& v : dev) {
      const double t = v - mean;
      v = t * t;
    }
    const double var = pairwise_sum(dev.data(), n) / static_cast<double>(n);
    const double nd = static_cast<double>(n);
    const double upper_bound = nd * eps * var + (nd * mean * eps) * (nd * mean * eps);
    if (var <= upper_bound) {
      lambdas_[static_cast<size_t>(j)] = 1.0;  // constant feature: identity
      continue;
    }
    lambdas_[static_cast<size_t>(j)] = yeojohnson_normmax(col);
    for (int64_t i = 0; i < n; ++i)
      out.at(i, j) = yeojohnson_transform_value(X.at(i, j), lambdas_[static_cast<size_t>(j)]);
  }
  scaler_.fit(out);
  return scaler_.transform(out);
}

Matrix PowerTransformerYJ::transform(const Matrix& X) const {
  // sklearn's transform() applies YJ to EVERY column, including lambda=1 ones
  // (((x+1)^1 - 1)/1 is not bitwise x for tiny x) — unlike fit_transform,
  // which skips constant columns entirely. Both behaviors are replicated.
  Matrix out = X;
  for (int64_t i = 0; i < X.n; ++i)
    for (int64_t j = 0; j < X.d; ++j)
      out.at(i, j) = yeojohnson_transform_value(X.at(i, j), lambdas_[static_cast<size_t>(j)]);
  return scaler_.transform(out);
}

}  // namespace tabicl
