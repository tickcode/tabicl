#include "preprocess/pipeline.h"

#include <stdexcept>

namespace tabicl {

PreprocessingPipeline::PreprocessingPipeline(std::string method,
                                             double outlier_threshold,
                                             uint64_t random_state)
    : method_(std::move(method)),
      outlier_threshold_(outlier_threshold),
      random_state_(random_state) {
  if (method_ != "none" && method_ != "power" && method_ != "quantile" &&
      method_ != "quantile_rtdl" && method_ != "robust")
    throw std::runtime_error("unknown normalization method: " + method_);
}

void PreprocessingPipeline::fit(const Matrix& X) {
  scaler_.fit(X);
  Matrix scaled = scaler_.transform(X);

  Matrix normalized;
  if (method_ == "power") {
    power_ = std::make_unique<PowerTransformerYJ>();
    normalized = power_->fit_transform(scaled);
  } else if (method_ == "quantile") {
    quantile_ = std::make_unique<QuantileTransformerNormal>();
    quantile_->fit(scaled, /*n_quantiles=*/1000, /*subsample=*/10000,
                   static_cast<uint32_t>(random_state_ & 0xFFFFFFFFull));
    normalized = quantile_->transform(scaled);
  } else if (method_ == "quantile_rtdl") {
    rtdl_ = std::make_unique<RTDLQuantileTransformer>();
    rtdl_->fit(scaled, random_state_);
    normalized = rtdl_->transform(scaled);
  } else if (method_ == "robust") {
    robust_ = std::make_unique<RobustScalerUV>();
    robust_->fit(scaled);
    normalized = robust_->transform(scaled);
  } else {  // none
    normalized = scaled;
  }

  outlier_.fit(normalized, outlier_threshold_);
  train_transformed_ = outlier_.transform(normalized);
}

Matrix PreprocessingPipeline::transform(const Matrix& X) const {
  Matrix out = scaler_.transform(X);
  // Python wraps the normalizer in a try/except ValueError with a clip-to-
  // train-range retry; the scaler's +/-100 clip makes inputs finite, so the
  // fallback is unreachable in the numeric-input path and is not ported.
  if (method_ == "power") out = power_->transform(out);
  else if (method_ == "quantile") out = quantile_->transform(out);
  else if (method_ == "quantile_rtdl") out = rtdl_->transform(out);
  else if (method_ == "robust") out = robust_->transform(out);
  return outlier_.transform(out);
}

}  // namespace tabicl
