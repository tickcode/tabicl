#include "preprocess/transforms.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "preprocess/stats.h"

namespace tabicl {

// ---------------------------------------------------------------------------
// MeanImputer
// ---------------------------------------------------------------------------

void MeanImputer::fit(const Matrix& X) {
  d_in_ = X.d;
  std::vector<double> means(static_cast<size_t>(X.d));
  std::vector<int64_t> counts(static_cast<size_t>(X.d));
  nanmean_axis0(X.data.data(), X.n, X.d, means.data(), counts.data());
  // sklearn semantics: statistics_ spans ALL input columns, NaN marking
  // all-NaN columns, which transform() then drops (keep_empty_features=False).
  statistics_ = means;
  kept_.clear();
  for (int64_t j = 0; j < X.d; ++j)
    if (counts[static_cast<size_t>(j)] > 0) kept_.push_back(j);
}

Matrix MeanImputer::transform(const Matrix& X) const {
  if (X.d != d_in_) throw std::runtime_error("MeanImputer: feature count mismatch");
  Matrix out = Matrix::zeros(X.n, static_cast<int64_t>(kept_.size()));
  for (int64_t i = 0; i < X.n; ++i) {
    for (size_t jj = 0; jj < kept_.size(); ++jj) {
      const double v = X.at(i, kept_[jj]);
      out.at(i, static_cast<int64_t>(jj)) =
          std::isnan(v) ? statistics_[static_cast<size_t>(kept_[jj])] : v;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// UniqueFeatureFilter
// ---------------------------------------------------------------------------

void UniqueFeatureFilter::fit(const Matrix& X, int threshold) {
  d_in_ = X.d;
  keep_.assign(static_cast<size_t>(X.d), true);
  if (X.n <= threshold) return;
  std::vector<double> col(static_cast<size_t>(X.n));
  for (int64_t j = 0; j < X.d; ++j) {
    for (int64_t i = 0; i < X.n; ++i) col[static_cast<size_t>(i)] = X.at(i, j);
    std::sort(col.begin(), col.end());
    // len(np.unique(col)): count of distinct values (exact equality; no NaNs
    // reach this point — inputs are imputed and validated finite).
    int64_t uniq = X.n > 0 ? 1 : 0;
    for (int64_t i = 1; i < X.n; ++i)
      if (col[static_cast<size_t>(i)] != col[static_cast<size_t>(i - 1)]) uniq++;
    keep_[static_cast<size_t>(j)] = uniq > threshold;
  }
}

int64_t UniqueFeatureFilter::n_features_out() const {
  int64_t n = 0;
  for (bool b : keep_) n += b ? 1 : 0;
  return n;
}

Matrix UniqueFeatureFilter::transform(const Matrix& X) const {
  if (X.d != d_in_) throw std::runtime_error("UniqueFeatureFilter: feature count mismatch");
  Matrix out = Matrix::zeros(X.n, n_features_out());
  for (int64_t i = 0; i < X.n; ++i) {
    int64_t jj = 0;
    for (int64_t j = 0; j < X.d; ++j)
      if (keep_[static_cast<size_t>(j)]) out.at(i, jj++) = X.at(i, j);
  }
  return out;
}

// ---------------------------------------------------------------------------
// CustomStandardScaler
// ---------------------------------------------------------------------------

void CustomStandardScaler::fit(const Matrix& X) {
  mean_.resize(static_cast<size_t>(X.d));
  scale_.resize(static_cast<size_t>(X.d));
  // Post-filter matrices are F-order in Python -> per-column pairwise sums.
  mean_axis0_colpw(X.data.data(), X.n, X.d, mean_.data());
  std_axis0_colpw(X.data.data(), X.n, X.d, /*ddof=*/0, scale_.data());
  for (double& s : scale_) s += 1e-6;
}

Matrix CustomStandardScaler::transform(const Matrix& X) const {
  Matrix out = Matrix::zeros(X.n, X.d);
  for (int64_t i = 0; i < X.n; ++i)
    for (int64_t j = 0; j < X.d; ++j) {
      const double z = (X.at(i, j) - mean_[static_cast<size_t>(j)]) /
                       scale_[static_cast<size_t>(j)];
      out.at(i, j) = std::clamp(z, -100.0, 100.0);
    }
  return out;
}

// ---------------------------------------------------------------------------
// OutlierRemover
// ---------------------------------------------------------------------------

void OutlierRemover::fit(const Matrix& X, double threshold) {
  const int ddof = X.n > 1 ? 1 : 0;
  std::vector<double> means(static_cast<size_t>(X.d)), stds(static_cast<size_t>(X.d));
  nanmean_axis0_colpw(X.data.data(), X.n, X.d, means.data());
  nanstd_axis0_colpw(X.data.data(), X.n, X.d, ddof, stds.data());
  for (double& s : stds) s = std::max(s, 1e-6);  // NaN stays NaN (numpy maximum)

  // Stage 1: mark |z| > threshold as NaN.
  Matrix clean = X;
  for (int64_t j = 0; j < X.d; ++j) {
    const double lo = means[static_cast<size_t>(j)] - threshold * stds[static_cast<size_t>(j)];
    const double hi = means[static_cast<size_t>(j)] + threshold * stds[static_cast<size_t>(j)];
    for (int64_t i = 0; i < X.n; ++i) {
      const double v = X.at(i, j);
      // NaN < lo and NaN > hi are false, matching numpy comparison semantics.
      if (v < lo || v > hi)
        clean.at(i, j) = std::numeric_limits<double>::quiet_NaN();
    }
  }

  // Stage 2: recompute on the cleaned data.
  nanmean_axis0_colpw(clean.data.data(), clean.n, clean.d, means.data());
  nanstd_axis0_colpw(clean.data.data(), clean.n, clean.d, ddof, stds.data());
  for (double& s : stds) s = std::max(s, 1e-6);

  lower_.resize(static_cast<size_t>(X.d));
  upper_.resize(static_cast<size_t>(X.d));
  for (int64_t j = 0; j < X.d; ++j) {
    lower_[static_cast<size_t>(j)] =
        means[static_cast<size_t>(j)] - threshold * stds[static_cast<size_t>(j)];
    upper_[static_cast<size_t>(j)] =
        means[static_cast<size_t>(j)] + threshold * stds[static_cast<size_t>(j)];
  }
}

Matrix OutlierRemover::transform(const Matrix& X) const {
  Matrix out = Matrix::zeros(X.n, X.d);
  for (int64_t i = 0; i < X.n; ++i)
    for (int64_t j = 0; j < X.d; ++j) {
      // X = max(-log1p(|X|) + lower, X); then X = min(log1p(|X|) + upper, X)
      // -- the second line sees the output of the first (Python reassigns X).
      double v = X.at(i, j);
      v = std::max(-std::log1p(std::abs(v)) + lower_[static_cast<size_t>(j)], v);
      v = std::min(std::log1p(std::abs(v)) + upper_[static_cast<size_t>(j)], v);
      out.at(i, j) = v;
    }
  return out;
}

}  // namespace tabicl
