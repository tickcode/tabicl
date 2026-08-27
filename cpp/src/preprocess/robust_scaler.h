// sklearn.preprocessing.RobustScaler(unit_variance=True) — dense path,
// quantile_range (25, 75).
#pragma once

#include <vector>

#include "preprocess/transforms.h"

namespace tabicl {

class RobustScalerUV {
 public:
  void fit(const Matrix& X);
  Matrix transform(const Matrix& X) const;

  const std::vector<double>& center() const { return center_; }
  const std::vector<double>& scale() const { return scale_; }
  void restore(std::vector<double> center, std::vector<double> scale) {
    center_ = std::move(center);
    scale_ = std::move(scale);
  }

 private:
  std::vector<double> center_, scale_;
};

}  // namespace tabicl
