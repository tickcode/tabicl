// Hierarchical >max_classes classification (learning.py _fit_hierarchical /
// _predict_hierarchical): balanced class tree over row representations,
// chain-rule probabilities, pseudo-logits tau*log(p + 1e-6).
#pragma once

#include <cstdint>
#include <vector>

#include "ggml_utils.h"
#include "tabicl/model.h"

namespace tabicl {

// R: row-major (T, D) representations of ONE table (train rows first);
// y: (train) encoded class indices. Returns (n_test, n_classes) pseudo-logits.
std::vector<float> hierarchical_member_logits(const Model& model,
                                              GraphRunner& runner, const float* R,
                                              const float* y, int64_t T,
                                              int64_t train, int64_t n_classes);

}  // namespace tabicl
