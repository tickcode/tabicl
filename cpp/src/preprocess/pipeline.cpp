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

void PreprocessingPipeline::save(FittedWriter& w, const std::string& prefix) const {
  w.put_str(prefix + "method", method_);
  w.put_f64_tensor(prefix + "scaler.mean", scaler_.mean());
  w.put_f64_tensor(prefix + "scaler.scale", scaler_.scale());
  if (method_ == "power") {
    w.put_f64_tensor(prefix + "yj.lambdas", power_->lambdas());
    w.put_f64_tensor(prefix + "yj.std.mean", power_->scaler().mean());
    w.put_f64_tensor(prefix + "yj.std.var", power_->scaler().var());
    w.put_f64_tensor(prefix + "yj.std.scale", power_->scaler().scale());
  } else if (method_ == "quantile") {
    w.put_u64(prefix + "qt.nq", static_cast<uint64_t>(quantile_->n_quantiles_fitted()));
    w.put_u64(prefix + "qt.d", static_cast<uint64_t>(quantile_->d()));
    w.put_f64_tensor(prefix + "qt.references", quantile_->references());
    w.put_f64_tensor(prefix + "qt.quantiles", quantile_->quantiles());
  } else if (method_ == "quantile_rtdl") {
    w.put_u64(prefix + "rtdl.qt.nq", static_cast<uint64_t>(rtdl_->qt().n_quantiles_fitted()));
    w.put_u64(prefix + "rtdl.qt.d", static_cast<uint64_t>(rtdl_->qt().d()));
    w.put_f64_tensor(prefix + "rtdl.qt.references", rtdl_->qt().references());
    w.put_f64_tensor(prefix + "rtdl.qt.quantiles", rtdl_->qt().quantiles());
    w.put_f64_tensor(prefix + "rtdl.std.mean", rtdl_->scaler().mean());
    w.put_f64_tensor(prefix + "rtdl.std.var", rtdl_->scaler().var());
    w.put_f64_tensor(prefix + "rtdl.std.scale", rtdl_->scaler().scale());
  } else if (method_ == "robust") {
    w.put_f64_tensor(prefix + "robust.center", robust_->center());
    w.put_f64_tensor(prefix + "robust.scale", robust_->scale());
  }
  w.put_f64_tensor(prefix + "outlier.lower", outlier_.lower_bounds());
  w.put_f64_tensor(prefix + "outlier.upper", outlier_.upper_bounds());
  w.put_u64(prefix + "train.n", static_cast<uint64_t>(train_transformed_.n));
  w.put_u64(prefix + "train.d", static_cast<uint64_t>(train_transformed_.d));
  w.put_f64_tensor(prefix + "train.data", train_transformed_.data);
}

void PreprocessingPipeline::load(const FittedReader& r, const std::string& prefix) {
  const std::string method = r.get_str(prefix + "method");
  if (method != method_)
    throw std::runtime_error("fitted: pipeline method mismatch at " + prefix);
  scaler_.restore(r.get_f64_tensor(prefix + "scaler.mean"),
                  r.get_f64_tensor(prefix + "scaler.scale"));
  if (method_ == "power") {
    SkStandardScaler sk;
    sk.restore(r.get_f64_tensor(prefix + "yj.std.mean"),
               r.get_f64_tensor(prefix + "yj.std.var"),
               r.get_f64_tensor(prefix + "yj.std.scale"));
    power_ = std::make_unique<PowerTransformerYJ>();
    power_->restore(r.get_f64_tensor(prefix + "yj.lambdas"), std::move(sk));
  } else if (method_ == "quantile") {
    quantile_ = std::make_unique<QuantileTransformerNormal>();
    quantile_->restore(static_cast<int64_t>(r.get_u64(prefix + "qt.nq")),
                       static_cast<int64_t>(r.get_u64(prefix + "qt.d")),
                       r.get_f64_tensor(prefix + "qt.references"),
                       r.get_f64_tensor(prefix + "qt.quantiles"));
  } else if (method_ == "quantile_rtdl") {
    QuantileTransformerNormal qt;
    qt.restore(static_cast<int64_t>(r.get_u64(prefix + "rtdl.qt.nq")),
               static_cast<int64_t>(r.get_u64(prefix + "rtdl.qt.d")),
               r.get_f64_tensor(prefix + "rtdl.qt.references"),
               r.get_f64_tensor(prefix + "rtdl.qt.quantiles"));
    SkStandardScaler sk;
    sk.restore(r.get_f64_tensor(prefix + "rtdl.std.mean"),
               r.get_f64_tensor(prefix + "rtdl.std.var"),
               r.get_f64_tensor(prefix + "rtdl.std.scale"));
    rtdl_ = std::make_unique<RTDLQuantileTransformer>();
    rtdl_->restore(std::move(qt), std::move(sk));
  } else if (method_ == "robust") {
    robust_ = std::make_unique<RobustScalerUV>();
    robust_->restore(r.get_f64_tensor(prefix + "robust.center"),
                     r.get_f64_tensor(prefix + "robust.scale"));
  }
  outlier_.restore(r.get_f64_tensor(prefix + "outlier.lower"),
                   r.get_f64_tensor(prefix + "outlier.upper"));
  train_transformed_.n = static_cast<int64_t>(r.get_u64(prefix + "train.n"));
  train_transformed_.d = static_cast<int64_t>(r.get_u64(prefix + "train.d"));
  train_transformed_.data = r.get_f64_tensor(prefix + "train.data");
}

}  // namespace tabicl
