// Ports of tabicl's basic preprocessing transforms (fp64, numpy semantics):
// MeanImputer (sklearn SimpleImputer strategy="mean"), UniqueFeatureFilter,
// CustomStandardScaler, OutlierRemover.
//
// Matrices are row-major (n, d) contiguous doubles.
#pragma once

#include <cstdint>
#include <vector>

namespace tabicl {

struct Matrix {
  std::vector<double> data;
  int64_t n = 0, d = 0;
  double& at(int64_t i, int64_t j) { return data[static_cast<size_t>(i * d + j)]; }
  double at(int64_t i, int64_t j) const { return data[static_cast<size_t>(i * d + j)]; }
  static Matrix zeros(int64_t n, int64_t d) {
    return Matrix{std::vector<double>(static_cast<size_t>(n * d), 0.0), n, d};
  }
};

// sklearn SimpleImputer(strategy="mean"): NaN -> column mean of observed
// values; columns with no observed values are DROPPED from the output.
class MeanImputer {
 public:
  void fit(const Matrix& X);
  Matrix transform(const Matrix& X) const;

  // Full input-column length; NaN marks dropped (all-NaN) columns.
  const std::vector<double>& statistics() const { return statistics_; }
  const std::vector<int64_t>& kept_columns() const { return kept_; }
  int64_t d_in() const { return d_in_; }
  void restore(std::vector<double> statistics, std::vector<int64_t> kept,
               int64_t d_in) {
    statistics_ = std::move(statistics);
    kept_ = std::move(kept);
    d_in_ = d_in;
  }

 private:
  std::vector<double> statistics_;
  std::vector<int64_t> kept_;
  int64_t d_in_ = 0;
};

// tabicl UniqueFeatureFilter(threshold=1): drop constant columns
// (when n_samples > threshold; otherwise keep everything).
class UniqueFeatureFilter {
 public:
  void fit(const Matrix& X, int threshold = 1);
  Matrix transform(const Matrix& X) const;

  const std::vector<bool>& features_to_keep() const { return keep_; }
  int64_t n_features_out() const;
  int64_t d_in() const { return d_in_; }
  void restore(std::vector<bool> keep, int64_t d_in) {
    keep_ = std::move(keep);
    d_in_ = d_in;
  }

 private:
  std::vector<bool> keep_;
  int64_t d_in_ = 0;
};

// tabicl CustomStandardScaler: (x - mean) / (std_ddof0 + 1e-6), clip [-100, 100].
class CustomStandardScaler {
 public:
  void fit(const Matrix& X);
  Matrix transform(const Matrix& X) const;

  const std::vector<double>& mean() const { return mean_; }
  const std::vector<double>& scale() const { return scale_; }
  void restore(std::vector<double> mean, std::vector<double> scale) {
    mean_ = std::move(mean);
    scale_ = std::move(scale);
  }

 private:
  std::vector<double> mean_, scale_;
};

// tabicl OutlierRemover(threshold=4.0): two-stage z-score fit, soft log1p clip.
class OutlierRemover {
 public:
  void fit(const Matrix& X, double threshold = 4.0);
  Matrix transform(const Matrix& X) const;

  const std::vector<double>& lower_bounds() const { return lower_; }
  const std::vector<double>& upper_bounds() const { return upper_; }
  void restore(std::vector<double> lower, std::vector<double> upper) {
    lower_ = std::move(lower);
    upper_ = std::move(upper);
  }

 private:
  std::vector<double> lower_, upper_;
};

}  // namespace tabicl
