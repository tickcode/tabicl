#include "preprocess/robust_scaler.h"

#include <cmath>

#include "preprocess/cephes_ndtri.h"
#include "preprocess/quantile_utils.h"

namespace tabicl {

void RobustScalerUV::fit(const Matrix& X) {
  const int64_t d = X.d;
  center_.resize(static_cast<size_t>(d));
  scale_.resize(static_cast<size_t>(d));
  constexpr double eps10 = 10.0 * 2.220446049250313e-16;
  const double adjust = ndtri(75.0 / 100.0) - ndtri(25.0 / 100.0);
  for (int64_t j = 0; j < d; ++j) {
    const auto col = sorted_finite_column(X.data.data(), X.n, X.d, j);
    const int64_t m = static_cast<int64_t>(col.size());
    center_[static_cast<size_t>(j)] = np_median_sorted(col.data(), m);
    const double q25 = np_percentile_sorted(col.data(), m, 25.0);
    const double q75 = np_percentile_sorted(col.data(), m, 75.0);
    double s = q75 - q25;
    if (s < eps10) s = 1.0;  // _handle_zeros_in_scale (no constant mask)
    scale_[static_cast<size_t>(j)] = s / adjust;
  }
}

Matrix RobustScalerUV::transform(const Matrix& X) const {
  Matrix out = Matrix::zeros(X.n, X.d);
  for (int64_t i = 0; i < X.n; ++i)
    for (int64_t j = 0; j < X.d; ++j)
      out.at(i, j) = (X.at(i, j) - center_[static_cast<size_t>(j)]) /
                     scale_[static_cast<size_t>(j)];
  return out;
}

}  // namespace tabicl
