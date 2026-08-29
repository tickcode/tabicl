#include "quantile_dist.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tabicl {

namespace {

// QuantileDistributionConfig (quantile_dist.py).
constexpr float kTol = 1e-6f;
constexpr float kMinBeta = 0.01f;
constexpr float kMaxBeta = 100.0f;
constexpr float kMinEta = -0.49f;
constexpr float kMaxEta = 0.49f;
constexpr float kEtaTolerance = 0.01f;
constexpr float kMaxLogRatio = 15.0f;
constexpr float kMaxExponent = 15.0f;
constexpr int64_t kTailQuantiles = 20;  // TAIL_QUANTILES_FOR_ESTIMATION

// Mean of n floats: double accumulate, round once (the dtype ladder the rest
// of the regressor path uses for its quantile reductions).
float mean_f(const float* v, int64_t n) {
  double s = 0.0;
  for (int64_t i = 0; i < n; ++i) s += static_cast<double>(v[i]);
  return static_cast<float>(s / static_cast<double>(n));
}

// Slope of Q regressed on log(u) over k knots, |.| and range-clamped.
// u is alpha on the left tail and 1-alpha on the right; the right-hand model
// is Q = -beta*log(1-a) + c, so its slope is negated before taking |.|.
float regress_beta(const float* q, const float* u, int64_t k, bool negate) {
  std::vector<float> ln(static_cast<size_t>(k)), tmp(static_cast<size_t>(k));
  for (int64_t i = 0; i < k; ++i)
    ln[static_cast<size_t>(i)] = std::log(std::max(u[i], kTol));
  const float ln_mean = mean_f(ln.data(), k);
  for (int64_t i = 0; i < k; ++i) ln[static_cast<size_t>(i)] -= ln_mean;

  const float q_mean = mean_f(q, k);
  for (int64_t i = 0; i < k; ++i)
    tmp[static_cast<size_t>(i)] = (q[i] - q_mean) * ln[static_cast<size_t>(i)];
  const float cov = mean_f(tmp.data(), k);

  for (int64_t i = 0; i < k; ++i)
    tmp[static_cast<size_t>(i)] = ln[static_cast<size_t>(i)] * ln[static_cast<size_t>(i)];
  const float var = mean_f(tmp.data(), k);

  const float beta = (negate ? -cov : cov) / std::max(var, kTol);
  return std::clamp(std::abs(beta), kMinBeta, kMaxBeta);
}

struct Betas {
  float left, right;
};

// estimate_exp_tail_params.
Betas estimate_exp_betas(const float* q, const std::vector<float>& alpha, int64_t k) {
  const int64_t n = static_cast<int64_t>(alpha.size());
  std::vector<float> u(static_cast<size_t>(k));
  for (int64_t i = 0; i < k; ++i) u[static_cast<size_t>(i)] = alpha[static_cast<size_t>(i)];
  const float left = regress_beta(q, u.data(), k, /*negate=*/false);
  for (int64_t i = 0; i < k; ++i)
    u[static_cast<size_t>(i)] = 1.0f - alpha[static_cast<size_t>(n - k + i)];
  const float right = regress_beta(q + (n - k), u.data(), k, /*negate=*/true);
  return {left, right};
}

// One side of the Pickands-like shape estimator: how far the spacing ratio of
// three knots departs from the exponential expectation. `u` values are alpha
// (left tail) or 1-alpha (right tail), ordered most-extreme first, and dq are
// the corresponding quantile gaps.
float pickands_eta(float dq12, float dq23, float u1, float u2, float u3) {
  const float ln12 = std::log(std::max(u2 / u1, kTol));
  const float ln23 = std::log(std::max(u3 / u2, kTol));
  const float expected = ln12 / std::max(ln23, kTol);
  const float actual = dq12 / std::max(dq23, kTol);
  const float deviation = actual / std::max(expected, kTol);
  const float eta = deviation > kTol
                        ? std::log(std::max(deviation, kTol)) /
                              std::max(std::abs(ln12), kTol)
                        : 0.0f;
  return std::clamp(eta, kMinEta, kMaxEta);
}

struct GpdParams {
  float eta_l, mu_l, eta_r, mu_r;
};

// estimate_gpd_tail_params: exponential betas supply the scale mu; the shape
// eta comes from the three-knot spacing ratio on each side.
GpdParams estimate_gpd(const float* q, const std::vector<float>& alpha, int64_t k) {
  const int64_t n = static_cast<int64_t>(alpha.size());
  const Betas beta = estimate_exp_betas(q, alpha, k);

  const int64_t i1 = 0, i2 = k / 3, i3 = 2 * k / 3;
  const float eta_l =
      pickands_eta(q[i2] - q[i1], q[i3] - q[i2], alpha[static_cast<size_t>(i1)],
                   alpha[static_cast<size_t>(i2)], alpha[static_cast<size_t>(i3)]);

  const int64_t j1 = n - 1, j2 = n - 1 - k / 3, j3 = n - 1 - 2 * k / 3;
  const float eta_r = pickands_eta(
      q[j1] - q[j2], q[j2] - q[j3],
      std::max(1.0f - alpha[static_cast<size_t>(j1)], kTol),
      std::max(1.0f - alpha[static_cast<size_t>(j2)], kTol),
      std::max(1.0f - alpha[static_cast<size_t>(j3)], kTol));

  return {eta_l, beta.left, eta_r, beta.right};
}

// _icdf_spline: piecewise-linear between knots, pinned to q_r at the top.
float icdf_spline(const float* q, const std::vector<float>& alpha, float a) {
  const int64_t nq = static_cast<int64_t>(alpha.size());
  if (a >= alpha[static_cast<size_t>(nq - 1)]) return q[nq - 1];
  // seg = searchsorted(alpha[:-1], a, right=True) - 1, clamped to [0, nq-2]
  const auto it = std::upper_bound(alpha.begin(), alpha.end() - 1, a);
  int64_t seg = static_cast<int64_t>(it - alpha.begin()) - 1;
  seg = std::clamp<int64_t>(seg, 0, nq - 2);
  const float denom = std::max(alpha[static_cast<size_t>(seg + 1)] -
                                   alpha[static_cast<size_t>(seg)],
                               kTol);
  float t = (a - alpha[static_cast<size_t>(seg)]) / denom;
  t = std::clamp(t, 0.0f, 1.0f);
  return q[seg] + t * (q[seg + 1] - q[seg]);
}

float icdf_exp_left(const QuantileTails& t, float a) {
  return t.a_l * std::log(std::max(a, kTol)) + t.b_l;
}

float icdf_exp_right(const QuantileTails& t, float a) {
  return t.a_r * std::log(std::max(1.0f - a, kTol)) + t.b_r;
}

// Q(a) = q_L - (mu/eta)*((alpha_L/a)^eta - 1), degenerating to the
// exponential form q_L - mu*log(alpha_L/a) as eta -> 0.
float icdf_gpd_left(const QuantileTails& t, float a) {
  const float ratio = t.alpha_l / std::max(a, kTol);
  const float log_ratio = std::min(std::log(ratio), kMaxLogRatio);
  if (std::abs(t.eta_l) < kEtaTolerance) return t.q_l - t.mu_l * log_ratio;
  const float ratio_pow = std::exp(std::min(t.eta_l * log_ratio, kMaxExponent));
  return t.q_l - t.mu_l / t.eta_l * (ratio_pow - 1.0f);
}

float icdf_gpd_right(const QuantileTails& t, float a) {
  const float ratio = (1.0f - t.alpha_r) / std::max(1.0f - a, kTol);
  const float log_ratio = std::min(std::log(ratio), kMaxLogRatio);
  if (std::abs(t.eta_r) < kEtaTolerance) return t.q_r + t.mu_r * log_ratio;
  const float ratio_pow = std::exp(std::min(t.eta_r * log_ratio, kMaxExponent));
  return t.q_r + t.mu_r / t.eta_r * (ratio_pow - 1.0f);
}

}  // namespace

std::vector<float> quantile_alpha_levels(int64_t num_quantiles) {
  // torch.linspace(0, 1, steps)[1:-1], reproduced bitwise. ATen derives `step`
  // in the tensor dtype (fp32) but evaluates start + step*i -- and, for the
  // upper half, end - step*(steps-1-i) -- with a single final rounding.
  //
  // The mirrored upper half is not cosmetic: it fixes alpha_r, and the tails
  // divide by 1 - alpha_r (~1e-3), which amplifies a last-ULP grid error into
  // a ~6e-5 relative error in the extrapolated quantiles.
  const int64_t steps = num_quantiles + 2;
  const int64_t halfway = steps / 2;
  const double step = static_cast<double>(1.0f / static_cast<float>(steps - 1));
  std::vector<float> a(static_cast<size_t>(num_quantiles));
  for (int64_t i = 1; i <= num_quantiles; ++i) {
    const double v = i < halfway
                         ? step * static_cast<double>(i)
                         : 1.0 - step * static_cast<double>(steps - 1 - i);
    a[static_cast<size_t>(i - 1)] = static_cast<float>(v);
  }
  return a;
}

QuantileTails fit_quantile_tails(const float* q_sorted,
                                 const std::vector<float>& alpha, TailType type) {
  const int64_t n = static_cast<int64_t>(alpha.size());
  // k = min(TAIL_QUANTILES_FOR_ESTIMATION, n // 4); the GPD estimator indexes
  // up to 2k/3, so a grid shorter than this cannot fit a tail at all.
  const int64_t k = std::min<int64_t>(kTailQuantiles, n / 4);
  if (k < 2) throw std::runtime_error("quantile tails: quantile grid too short");

  QuantileTails t;
  t.type = type;
  t.alpha_l = alpha.front();
  t.alpha_r = alpha.back();
  t.q_l = q_sorted[0];
  t.q_r = q_sorted[n - 1];

  if (type == TailType::Exp) {
    const Betas beta = estimate_exp_betas(q_sorted, alpha, k);
    // Pin each log-line through its boundary knot.
    t.a_l = beta.left;
    t.b_l = t.q_l - t.a_l * std::log(std::max(t.alpha_l, kTol));
    t.a_r = -beta.right;
    t.b_r = t.q_r - t.a_r * std::log(1.0f - std::min(t.alpha_r, 1.0f - kTol));
  } else {
    const GpdParams g = estimate_gpd(q_sorted, alpha, k);
    t.eta_l = g.eta_l;
    t.mu_l = g.mu_l;
    t.eta_r = g.eta_r;
    t.mu_r = g.mu_r;
  }
  return t;
}

float quantile_icdf(const float* q_sorted, const std::vector<float>& alpha,
                    const QuantileTails& tails, float a) {
  // Strict comparisons: a == alpha_r takes the spline branch, which pins to
  // q_r (matching icdf's nested torch.where).
  if (a < tails.alpha_l)
    return tails.type == TailType::Exp ? icdf_exp_left(tails, a)
                                       : icdf_gpd_left(tails, a);
  if (a > tails.alpha_r)
    return tails.type == TailType::Exp ? icdf_exp_right(tails, a)
                                       : icdf_gpd_right(tails, a);
  return icdf_spline(q_sorted, alpha, a);
}

}  // namespace tabicl
