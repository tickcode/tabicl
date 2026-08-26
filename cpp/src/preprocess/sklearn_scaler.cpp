#include "preprocess/sklearn_scaler.h"

#include <cmath>
#include <limits>

#include "preprocess/stats.h"

namespace tabicl {

void SkStandardScaler::fit(const Matrix& X) {
  const int64_t n = X.n, d = X.d;
  mean_.assign(static_cast<size_t>(d), 0.0);
  var_.assign(static_cast<size_t>(d), 0.0);
  scale_.assign(static_cast<size_t>(d), 0.0);
  constexpr double eps = 2.220446049250313e-16;  // np.finfo(float64).eps

  std::vector<double> col(static_cast<size_t>(n));
  for (int64_t j = 0; j < d; ++j) {
    for (int64_t i = 0; i < n; ++i) col[static_cast<size_t>(i)] = X.at(i, j);
    // _incremental_mean_and_var with last_count = 0 (single batch, no NaNs):
    const double new_sum = pairwise_sum(col.data(), n);
    const double mean = new_sum / static_cast<double>(n);
    const double T = new_sum / static_cast<double>(n);
    std::vector<double> temp(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
      temp[static_cast<size_t>(i)] = col[static_cast<size_t>(i)] - T;
    const double correction = pairwise_sum(temp.data(), n);
    for (double& t : temp) t = t * t;
    double unnorm_var = pairwise_sum(temp.data(), n);
    unnorm_var -= correction * correction / static_cast<double>(n);
    const double var = unnorm_var / static_cast<double>(n);
    mean_[static_cast<size_t>(j)] = mean;
    var_[static_cast<size_t>(j)] = var;
    // _is_constant_feature on the raw variance:
    const double nd = static_cast<double>(n);
    const double upper_bound = nd * eps * var + (nd * mean * eps) * (nd * mean * eps);
    const bool constant = var <= upper_bound;
    scale_[static_cast<size_t>(j)] = constant ? 1.0 : std::sqrt(var);
  }
}

Matrix SkStandardScaler::transform(const Matrix& X) const {
  Matrix out = Matrix::zeros(X.n, X.d);
  for (int64_t i = 0; i < X.n; ++i)
    for (int64_t j = 0; j < X.d; ++j)
      out.at(i, j) = (X.at(i, j) - mean_[static_cast<size_t>(j)]) /
                     scale_[static_cast<size_t>(j)];
  return out;
}

Matrix SkStandardScaler::inverse_transform(const Matrix& X) const {
  Matrix out = Matrix::zeros(X.n, X.d);
  for (int64_t i = 0; i < X.n; ++i)
    for (int64_t j = 0; j < X.d; ++j)
      out.at(i, j) = X.at(i, j) * scale_[static_cast<size_t>(j)] +
                     mean_[static_cast<size_t>(j)];
  return out;
}

}  // namespace tabicl
