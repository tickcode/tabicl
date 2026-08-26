// TabICL model forward (inference only) on preprocessed fp32 inputs.
// Mirrors TabICL._inference_forward: ColEmbedding -> RowInteraction ->
// ICLearning -> head. Test rows never enter any K/V set.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "ggml_utils.h"
#include "tabicl/model.h"

namespace tabicl {

struct ForwardOptions {
  int64_t train_size = 0;
  // Classifier: number of distinct classes in y_train (head slice width).
  int64_t num_classes = 0;
  bool return_logits = true;  // false: apply softmax(logits / temperature)
};

// X row-major (B, T, H); y_train row-major (B, train_size) as floats
// (class indices for the classifier). Returns row-major
// (B, T - train_size, out_dim) where out_dim = num_classes (clf) or
// num_quantiles (reg).
std::vector<float> tabicl_forward(const Model& model, GraphRunner& runner,
                                  const float* X, const float* y_train,
                                  int64_t B, int64_t T, int64_t H,
                                  const ForwardOptions& opts);

// Stage entry points (exposed for stage-level parity tests).
// col_out: row-major (B, T, H, E) — only the H real feature slots (the 4
// reserved CLS slots are never computed; RowInteraction overwrites them).
std::vector<float> col_embedding_forward(const Model& model, GraphRunner& runner,
                                         const float* X, const float* y_train,
                                         int64_t B, int64_t T, int64_t H,
                                         int64_t train_size, int64_t num_classes);

// In: (B, T, H, E) col embeddings; out row-major (B, T, D=E*num_cls).
std::vector<float> row_interaction_forward(const Model& model, GraphRunner& runner,
                                           const float* col_out, int64_t B,
                                           int64_t T, int64_t H);

// In: (B, T, D) representations; out row-major (B, T, out_dim) raw decoder
// output over ALL rows (caller slices test rows / classes).
std::vector<float> icl_forward(const Model& model, GraphRunner& runner,
                               const float* reprs, const float* y_train,
                               int64_t B, int64_t T, int64_t train_size,
                               int64_t num_classes);

// ---------------------------------------------------------------------------
// KV cache (mirrors tabicl kv_cache.py, "kv" and "repr" modes)
// ---------------------------------------------------------------------------

struct TabICLCache {
  enum class Mode { KV, Repr };
  Mode mode = Mode::KV;
  int64_t B = 0;           // cached ensemble members
  int64_t train_size = 0;
  int64_t H = 0;           // kept feature count at fit time
  // KV mode — col: per ISAB block, attn2 K/V of `hidden`, ggml layout
  // (hd, M, nh, H*B); icl: per layer, train-prefix K/V, (hd, train, nh, B).
  std::vector<std::vector<float>> col_k, col_v;
  std::vector<std::vector<float>> icl_k, icl_v;
  // Repr mode — row representations WITH y injected: row-major (B, train, D).
  std::vector<float> row_repr;

  // Concatenate another cache along the member axis (same geometry).
  void append(const TabICLCache& other);
};

// Build a cache from training data only (Python forward_with_cache store pass).
TabICLCache tabicl_build_cache(const Model& model, GraphRunner& runner,
                               const float* X_train, const float* y_train,
                               int64_t B, int64_t train_size, int64_t H,
                               TabICLCache::Mode mode);

// Cached forward on test rows only (Python forward_with_cache use pass).
// Returns row-major (B, n_test, out_dim) logits/raw quantiles.
std::vector<float> tabicl_forward_cached(const Model& model, GraphRunner& runner,
                                         const float* X_test, int64_t n_test,
                                         const TabICLCache& cache,
                                         const ForwardOptions& opts);

}  // namespace tabicl
