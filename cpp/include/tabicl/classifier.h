// TabICLClassifier: numeric-input C++ port of tabicl.TabICLClassifier
// (fit/predict/predict_proba, standard <=max_classes path).
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "tabicl/model.h"
#include "tabicl/options.h"

namespace tabicl {

class EstimatorCore;

class TabICLClassifier {
 public:
  explicit TabICLClassifier(std::shared_ptr<Model> model);
  TabICLClassifier(std::shared_ptr<Model> model, const EstimatorOptions& opts);
  TabICLClassifier(TabICLClassifier&&) noexcept;
  TabICLClassifier& operator=(TabICLClassifier&&) noexcept;
  ~TabICLClassifier();

  // X row-major (n, d) double, NaN = missing; y (n) class labels (any doubles).
  void fit(const double* X, const double* y, int64_t n, int64_t d);

  // out: row-major (n_test, n_classes) probabilities.
  void predict_proba(const double* X, int64_t n_test, float* out) const;
  // out_labels: (n_test) original label values.
  void predict(const double* X, int64_t n_test, double* out_labels) const;

  const std::vector<double>& classes() const { return classes_; }
  int64_t n_classes() const { return static_cast<int64_t>(classes_.size()); }

  // Persist the fitted state (preprocessing, ensemble configs, KV cache) to a
  // versioned GGUF file. load() verifies the file was fitted with the SAME
  // checkpoint (fingerprint) and restores a ready-to-predict estimator;
  // n_threads_override >= 0 replaces the stored thread count.
  void save(const std::string& path) const;
  static TabICLClassifier load(const std::string& path,
                               std::shared_ptr<Model> model,
                               int n_threads_override = -1);

 private:
  std::shared_ptr<Model> model_;
  std::unique_ptr<EstimatorCore> core_;
  std::unique_ptr<EstimatorOptions> opts_;
  std::vector<double> classes_;
};

}  // namespace tabicl
