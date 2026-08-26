// TabICL model: immutable weights + config loaded from a GGUF file
// produced by scripts/export_gguf.py.
#pragma once

#include <memory>
#include <string>
#include <unordered_map>

struct ggml_context;
struct ggml_tensor;

namespace tabicl {

enum class Task { Classification, Regression };

struct ModelConfig {
  Task task = Task::Classification;
  int max_classes = 10;
  int num_quantiles = 999;
  int embed_dim = 128;
  int col_num_blocks = 3;
  int col_num_heads = 8;
  int col_num_inds = 128;
  int col_feature_group_size = 3;
  int row_num_blocks = 3;
  int row_num_heads = 8;
  int row_num_cls = 4;
  float row_rope_base = 100000.0f;
  int icl_num_blocks = 12;
  int icl_num_heads = 8;
  int ff_factor = 2;
  bool bias_free_ln = false;
  float norm_eps = 1e-5f;
  float softmax_temperature = 0.9f;

  int icl_dim() const { return embed_dim * row_num_cls; }
};

class Model {
 public:
  // Throws std::runtime_error on IO/validation failure.
  static std::shared_ptr<Model> load(const std::string& gguf_path);
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  const ModelConfig& config() const { return config_; }

  // Weight tensor by shortened name (see scripts/export_gguf.py); throws if absent.
  ggml_tensor* tensor(const std::string& name) const;
  // nullptr if absent (used for bias-free LayerNorm variants).
  ggml_tensor* tensor_or_null(const std::string& name) const;
  int64_t n_tensors() const { return static_cast<int64_t>(tensors_.size()); }

 private:
  Model() = default;
  ModelConfig config_;
  ggml_context* weights_ctx_ = nullptr;  // owns all weight data
  std::unordered_map<std::string, ggml_tensor*> tensors_;
};

}  // namespace tabicl
