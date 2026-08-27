// TabICLRegressor: numeric-input C++ port of tabicl.TabICLRegressor
// (fit/predict mean + quantiles via the 999-quantile head).
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "tabicl/model.h"

namespace tabicl {

struct EstimatorOptions;
class EstimatorCore;

class TabICLRegressor {
 public:
  explicit TabICLRegressor(std::shared_ptr<Model> model);
  TabICLRegressor(std::shared_ptr<Model> model, const EstimatorOptions& opts);
  TabICLRegressor(TabICLRegressor&&) noexcept;
  TabICLRegressor& operator=(TabICLRegressor&&) noexcept;
  ~TabICLRegressor();

  // X row-major (n, d) double, NaN = missing; y (n) targets.
  void fit(const double* X, const double* y, int64_t n, int64_t d);

  // Mean prediction (ensemble average), out (n_test).
  void predict(const double* X, int64_t n_test, double* out) const;

  // Quantile predictions for alphas in [0.001, 0.999); out (n_test, n_alphas).
  void predict_quantiles(const double* X, int64_t n_test, const double* alphas,
                         int64_t n_alphas, double* out) const;

  // Fitted-state persistence (see TabICLClassifier::save/load).
  void save(const std::string& path) const;
  static TabICLRegressor load(const std::string& path,
                              std::shared_ptr<Model> model,
                              int n_threads_override = -1);

 private:
  std::shared_ptr<Model> model_;
  std::unique_ptr<EstimatorCore> core_;
  std::unique_ptr<EstimatorOptions> opts_;
  double y_mean_ = 0.0, y_scale_ = 1.0;
};

}  // namespace tabicl
