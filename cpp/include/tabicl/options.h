// Public configuration for TabICLClassifier / TabICLRegressor.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tabicl {

enum class CacheMode {
  None,  // no cache: predict re-runs training rows through the model
  KV,    // cache per-layer K/V of the training rows (fastest predict)
  Repr,  // cache row representations only (~24x smaller, slower predict)
};

struct EstimatorOptions {
  CacheMode cache = CacheMode::None;
  int n_estimators = 8;
  std::vector<std::string> norm_methods = {"none", "power"};
  int batch_size = 8;
  uint64_t random_state = 42;
  int n_threads = 0;  // 0 = hardware concurrency
  float softmax_temperature = 0.9f;
  bool average_logits = true;  // classifier only
  // Peak per-graph scratch budget. Attention/activations are chunked along
  // embarrassingly-parallel axes to stay under it; results are bitwise
  // independent of the budget (tight budgets run slower, never fail).
  int64_t max_scratch_bytes = int64_t(1) << 30;  // 1 GiB
};

}  // namespace tabicl
