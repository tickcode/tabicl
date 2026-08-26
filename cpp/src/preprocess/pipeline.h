// tabicl PreprocessingPipeline: CustomStandardScaler -> normalizer (per
// method) -> OutlierRemover, with the transformed training matrix memoized.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "preprocess/power_transform.h"
#include "preprocess/quantile_transform.h"
#include "preprocess/robust_scaler.h"
#include "preprocess/transforms.h"

namespace tabicl {

class PreprocessingPipeline {
 public:
  // method: "none" | "power" | "quantile" | "quantile_rtdl" | "robust"
  PreprocessingPipeline(std::string method, double outlier_threshold,
                        uint64_t random_state);

  // Fit on training data; memoizes the transformed training matrix.
  void fit(const Matrix& X);
  Matrix transform(const Matrix& X) const;
  const Matrix& train_transformed() const { return train_transformed_; }

 private:
  std::string method_;
  double outlier_threshold_;
  uint64_t random_state_;
  CustomStandardScaler scaler_;
  std::unique_ptr<PowerTransformerYJ> power_;
  std::unique_ptr<QuantileTransformerNormal> quantile_;
  std::unique_ptr<RTDLQuantileTransformer> rtdl_;
  std::unique_ptr<RobustScalerUV> robust_;
  OutlierRemover outlier_;
  Matrix train_transformed_;
};

}  // namespace tabicl
