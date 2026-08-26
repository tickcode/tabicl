#include "tabicl/regressor.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "estimator_core.h"
#include "preprocess/sklearn_scaler.h"

namespace tabicl {

namespace {

// torch.linspace(0, 1, 1001)[1:-1] in fp32: the quantile alpha grid.
std::vector<float> alpha_levels(int64_t num_quantiles) {
  std::vector<float> a(static_cast<size_t>(num_quantiles));
  const float step = 1.0f / static_cast<float>(num_quantiles + 1);
  for (int64_t i = 0; i < num_quantiles; ++i)
    a[static_cast<size_t>(i)] = static_cast<float>(i + 1) * step;
  return a;
}

// QuantileDistribution icdf, linear spline between sorted quantile knots
// (crossing_method="sort", fp32 like torch). alpha must lie in
// [alpha_levels.front(), 1); values >= alpha_levels.back() clamp to the last
// knot (matching _icdf_spline's right-boundary handling).
float icdf_sorted(const float* q_sorted, const std::vector<float>& alpha, float a) {
  const int64_t nq = static_cast<int64_t>(alpha.size());
  if (a < alpha.front())
    throw std::runtime_error("predict_quantiles: alpha below grid (exp tail not ported)");
  if (a >= alpha.back()) return q_sorted[nq - 1];
  // seg = searchsorted(alpha[:-1], a, right=True) - 1, clamped to [0, nq-2]
  const auto it = std::upper_bound(alpha.begin(), alpha.end() - 1, a);
  int64_t seg = static_cast<int64_t>(it - alpha.begin()) - 1;
  seg = std::clamp<int64_t>(seg, 0, nq - 2);
  const float denom = std::max(alpha[static_cast<size_t>(seg + 1)] -
                                   alpha[static_cast<size_t>(seg)],
                               1e-6f);
  float t = (a - alpha[static_cast<size_t>(seg)]) / denom;
  t = std::clamp(t, 0.0f, 1.0f);
  return q_sorted[seg] + t * (q_sorted[seg + 1] - q_sorted[seg]);
}

}  // namespace

TabICLRegressor::TabICLRegressor(std::shared_ptr<Model> model)
    : TabICLRegressor(std::move(model), EstimatorOptions{}) {}

TabICLRegressor::TabICLRegressor(std::shared_ptr<Model> model,
                                 const EstimatorOptions& opts)
    : model_(std::move(model)), core_(std::make_unique<EstimatorCore>()) {
  if (model_->config().task != Task::Regression)
    throw std::runtime_error("TabICLRegressor: model is not a regressor");
  opts_ = std::make_unique<EstimatorOptions>(opts);
}

TabICLRegressor::~TabICLRegressor() = default;

void TabICLRegressor::fit(const double* X, const double* y, int64_t n, int64_t d) {
  // sklearn StandardScaler on y (fp64 stats over the fp32-cast targets,
  // then per-op fp32 rounding in transform — mirrors Python's dtype ladder).
  Matrix ym = Matrix::zeros(n, 1);
  for (int64_t i = 0; i < n; ++i)
    ym.at(i, 0) = static_cast<double>(static_cast<float>(y[i]));
  SkStandardScaler scaler;
  scaler.fit(ym);
  y_mean_ = scaler.mean()[0];
  y_scale_ = scaler.scale()[0];

  std::vector<float> y_scaled(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    const float step1 = static_cast<float>(
        static_cast<double>(static_cast<float>(y[i])) - y_mean_);
    y_scaled[static_cast<size_t>(i)] =
        static_cast<float>(static_cast<double>(step1) / y_scale_);
  }
  core_->fit(X, n, d, std::move(y_scaled), /*n_classes=*/0, *opts_, model_.get());
}

void TabICLRegressor::predict(const double* X, int64_t n_test, double* out) const {
  auto res = core_->predict_outputs(*model_, X, n_test);
  const int64_t nq = res.out_dim;
  const int64_t n_members = static_cast<int64_t>(res.member_outputs.size());

  // Per member: sort quantiles, mean over quantiles (fp32 head, double
  // accumulate), inverse-transform (fp64), then mean over members (fp64).
  std::vector<double> acc(static_cast<size_t>(n_test), 0.0);
  std::vector<float> qs(static_cast<size_t>(nq));
  for (int64_t m = 0; m < n_members; ++m) {
    auto& o = res.member_outputs[static_cast<size_t>(m)];
    for (int64_t r = 0; r < n_test; ++r) {
      std::copy(o.begin() + r * nq, o.begin() + (r + 1) * nq, qs.begin());
      std::sort(qs.begin(), qs.end());
      double s = 0.0;
      for (float v : qs) s += static_cast<double>(v);
      const float mean_f = static_cast<float>(s / static_cast<double>(nq));
      acc[static_cast<size_t>(r)] +=
          static_cast<double>(mean_f) * y_scale_ + y_mean_;
    }
  }
  for (int64_t r = 0; r < n_test; ++r)
    out[r] = acc[static_cast<size_t>(r)] / static_cast<double>(n_members);
}

void TabICLRegressor::predict_quantiles(const double* X, int64_t n_test,
                                        const double* alphas, int64_t n_alphas,
                                        double* out) const {
  auto res = core_->predict_outputs(*model_, X, n_test);
  const int64_t nq = res.out_dim;
  const int64_t n_members = static_cast<int64_t>(res.member_outputs.size());
  const auto alpha = alpha_levels(nq);

  std::vector<double> acc(static_cast<size_t>(n_test * n_alphas), 0.0);
  std::vector<float> qs(static_cast<size_t>(nq));
  for (int64_t m = 0; m < n_members; ++m) {
    auto& o = res.member_outputs[static_cast<size_t>(m)];
    for (int64_t r = 0; r < n_test; ++r) {
      std::copy(o.begin() + r * nq, o.begin() + (r + 1) * nq, qs.begin());
      std::sort(qs.begin(), qs.end());
      for (int64_t a = 0; a < n_alphas; ++a) {
        const float v = icdf_sorted(qs.data(), alpha, static_cast<float>(alphas[a]));
        acc[static_cast<size_t>(r * n_alphas + a)] +=
            static_cast<double>(v) * y_scale_ + y_mean_;
      }
    }
  }
  for (size_t i = 0; i < acc.size(); ++i)
    out[i] = acc[i] / static_cast<double>(n_members);
}

}  // namespace tabicl
