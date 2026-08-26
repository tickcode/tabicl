// sklearn.preprocessing.PowerTransformer(method="yeo-johnson",
// standardize=True), with lambda estimated per column exactly like
// scipy.stats.yeojohnson (yeojohnson_normmax + fminbound, xatol 1.48e-8,
// maxiter 500). Iterate-for-iterate port of scipy's bounded Brent optimizer.
#pragma once

#include <vector>

#include "preprocess/sklearn_scaler.h"
#include "preprocess/transforms.h"

namespace tabicl {

class PowerTransformerYJ {
 public:
  // fit_transform on training data (sklearn's fused path); plain fit is
  // fit_transform with the result discarded — identical fitted state.
  Matrix fit_transform(const Matrix& X);
  Matrix transform(const Matrix& X) const;

  const std::vector<double>& lambdas() const { return lambdas_; }
  const SkStandardScaler& scaler() const { return scaler_; }

 private:
  std::vector<double> lambdas_;
  SkStandardScaler scaler_;
};

// scipy.stats.yeojohnson_normmax(x) for finite 1D data (exposed for tests).
double yeojohnson_normmax(const std::vector<double>& x);

// Elementwise Yeo-Johnson transform with parameter lambda.
double yeojohnson_transform_value(double x, double lmbda);

}  // namespace tabicl
