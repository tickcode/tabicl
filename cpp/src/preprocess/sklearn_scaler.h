// sklearn.preprocessing.StandardScaler (dense, unweighted, no-NaN fit path),
// bit-faithful to _incremental_mean_and_var + _is_constant_feature +
// _handle_zeros_in_scale on F-order inputs (per-column pairwise sums).
#pragma once

#include <cstdint>
#include <vector>

#include "preprocess/transforms.h"

namespace tabicl {

class SkStandardScaler {
 public:
  void fit(const Matrix& X);
  Matrix transform(const Matrix& X) const;
  Matrix inverse_transform(const Matrix& X) const;

  const std::vector<double>& mean() const { return mean_; }
  const std::vector<double>& var() const { return var_; }
  const std::vector<double>& scale() const { return scale_; }

 private:
  std::vector<double> mean_, var_, scale_;
};

}  // namespace tabicl
